import numpy as np
import pandas as pd
from itertools import product
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple, List, Dict

from scipy import optimize, special

from ecf_einsum import ecf_grid_einsum  # uses einsum path pinning


def mvt_cf_on_grid(t_stack: np.ndarray,
                   mu: np.ndarray,
                   Sigma_scale: np.ndarray,
                   nu: float,
                   dtype=np.complex128) -> np.ndarray:
    """
    Evaluate the multivariate Student-t characteristic function on a grid.

    Parameters
    ----------
    t_stack : (..., d) array of frequency vectors on the grid
    mu      : (d,) mean vector
    Sigma_scale : (d, d) *scale* matrix (NOT covariance)
    nu      : degrees of freedom (> 2 if you rely on covariance)
    dtype   : complex dtype

    Returns
    -------
    phi : complex array with shape t_stack.shape[:-1]
    """
    # quadratic form q = t^T Σ t
    q = np.einsum('...i,ij,...j->...', t_stack, Sigma_scale, t_stack, optimize=True)

    # argument and order for Bessel K
    order = nu / 2.0
    z = np.sqrt(np.maximum(0.0, nu) * np.maximum(0.0, q))  # real nonnegative

    # core scalar factor g(z; nu) with safe handling at z=0 (limit -> 1)
    # g = 2^{1-ν/2}/Γ(ν/2) * (z)^{ν/2} K_{ν/2}(z)
    g = np.empty_like(z, dtype=float)
    g.fill(1.0)
    mask = z > 0
    if np.any(mask):
        K = special.kv(order, z[mask])
        # numerical safety: kv can under/overflow at extreme z, clip via log
        # direct expression:
        pref = 2.0 ** (1.0 - order) / special.gamma(order)
        g[mask] = pref * (z[mask] ** order) * K

        # If any NaNs due to extreme parameters, fall back to log-form evaluation
        bad = ~np.isfinite(g[mask])
        if np.any(bad):
            zb = z[mask][bad]
            Kb = special.kv(order, zb)
            gb = pref * np.exp(order * np.log(zb)) * Kb
            g[mask][bad] = gb

    # multiply by exp(i t^T mu)
    lin = np.tensordot(t_stack, mu, axes=([-1], [0]))  # shape grid
    phi = (np.exp(1j * lin) * g).astype(dtype)
    return phi


# ---------- Utilities to construct COS grids and sign matrices ----------
def all_sign_patterns(d: int) -> np.ndarray:
    """
    Generate all 2^(d-1) sign patterns with first sign fixed to +1.
    Returns array with shape (2^(d-1), d)
    """
    rest = list(product([-1, 1], repeat=d-1))
    S = np.array([[1] + list(r) for r in rest], dtype=int)
    return S


def build_t_stack_for_s(a: np.ndarray, b: np.ndarray, s: np.ndarray, K: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Build the multidimensional COS frequency grid t = pi * s * k / (b - a)
    for a single sign vector s.

    Returns
    -------
    t_stack : (..., d) array of t vectors on the grid
    k_grids : list of k-grid arrays (for masking k=0 cell etc.)
    """
    d = len(K)
    alphas = np.pi * s / (b - a)  # shape (d,)

    # mesh of all k
    k_grids = np.meshgrid(*[np.arange(K[j]) for j in range(d)], indexing='ij')
    t_axes = [alphas[j] * k_grids[j] for j in range(d)]
    t_stack = np.stack(t_axes, axis=-1)  # (..., d)
    return t_stack, k_grids


# ---------- Main estimator ----------
@dataclass
class MVTCF:
    df: pd.DataFrame
    feature_cols: Iterable[str]
    K: Optional[Iterable[int]] = None          # e.g. [64, 64, ...]
    a: Optional[Iterable[float]] = None
    b: Optional[Iterable[float]] = None
    L_box: float = 7.0                          # if a/b not given: mu ± L*std
    decay: float = 0.0                          # exp(-decay * ||t||) weight
    cov_is_covariance: bool = True              # treat sample cov as covariance
    bounds_nu: Tuple[float, float] = (2.05, 200.0)
    dtype: np.dtype = np.complex128

    def __post_init__(self):
        X = self.df.loc[:, list(self.feature_cols)].to_numpy(dtype=float)
        if np.isnan(X).any():
            raise ValueError("Input features contain NaNs. Please impute/clean first.")
        self.X = X
        self.N, self.d = X.shape

        # sample stats
        self.mu_hat = self._mu(self.X)
        self.Sigma_hat = self._Sigma(self.X)

        # default K, a, b
        if self.K is None:
            self.K = np.array([64] * self.d, dtype=int)
        else:
            self.K = np.asarray(self.K, dtype=int)
            assert len(self.K) == self.d

        if self.a is None or self.b is None:
            std = np.sqrt(np.clip(np.diag(self.Sigma_hat), 1e-12, np.inf))
            self.a = self.mu_hat - self.L_box * std
            self.b = self.mu_hat + self.L_box * std
        else:
            self.a = np.asarray(self.a, dtype=float)
            self.b = np.asarray(self.b, dtype=float)
            assert self.a.shape == (self.d,) and self.b.shape == (self.d,)

        # pre-build sign patterns and corresponding t-grids
        self.S = all_sign_patterns(self.d)          # shape (#S, d)
        self.t_stacks: List[np.ndarray] = []
        self.k_grids_per_s: List[List[np.ndarray]] = []
        self.t_norms: List[np.ndarray] = []

        for s in self.S:
            t_stack, k_grids = build_t_stack_for_s(self.a, self.b, s, self.K)
            self.t_stacks.append(t_stack)
            self.k_grids_per_s.append(k_grids)
            self.t_norms.append(np.linalg.norm(t_stack, axis=-1))

        # empirical CF on each sign-grid
        self._ecf_per_s: Optional[List[np.ndarray]] = None

    @staticmethod
    def _mu(X: np.ndarray) -> np.ndarray:
        return X.mean(axis=0)

    @staticmethod
    def _Sigma(X: np.ndarray) -> np.ndarray:
        Xc = X - X.mean(axis=0, keepdims=True)
        return np.einsum('ni,nj->ij', Xc, Xc) / (X.shape[0] - 1)

    def _ecf_on_grids(self) -> List[np.ndarray]:
        if self._ecf_per_s is not None:
            return self._ecf_per_s

        ecfs = []
        for s in self.S:
            ecf = ecf_grid_einsum(self.X, self.a, self.b, s.astype(int), self.K, dtype=self.dtype)
            ecfs.append(ecf)
        self._ecf_per_s = ecfs
        return ecfs


    def _Sigma_scale(self, nu: float) -> np.ndarray:
        if self.cov_is_covariance:
            return ((nu - 2.0) / nu) * self.Sigma_hat
        return self.Sigma_hat


    def _model_cf_on_grids(self, nu: float) -> List[np.ndarray]:
        Sig = self._Sigma_scale(nu)
        phis = []
        for t_stack in self.t_stacks:
            phi = mvt_cf_on_grid(t_stack, self.mu_hat, Sig, nu, dtype=self.dtype)
            phis.append(phi)
        return phis


    def _loss(self, nu: float) -> float:
        model = self._model_cf_on_grids(nu)
        ecfs = self._ecf_on_grids()

        loss = 0.0
        for (phi, ecf, t_stack, k_grids, norms) in zip(
                model, ecfs, self.t_stacks, self.k_grids_per_s, self.t_norms):
            w = np.exp(-self.decay * norms)

            # do not over-weight the DC cell (k==0 for all dimensions)
            k_sum = np.zeros_like(w, dtype=int)
            for kg in k_grids:
                k_sum += kg
            w = np.where(k_sum == 0, 0.0, w)

            diff2 = np.abs(phi - ecf) ** 2
            loss += np.sum(w * diff2).real  # real scalar


        return loss / len(self.S)

    # ---- gradient of loss wrt nu (robust numerical) ----
    def _grad(self, nu: float, eps: float = 1e-4) -> np.ndarray:
        # central difference; robust and accurate for 1D parameter
        f_plus = self._loss(nu + eps)
        f_minus = self._loss(nu - eps)
        g = (f_plus - f_minus) / (2.0 * eps)
        return np.array([g], dtype=float)

    # optional: derivative of CF wrt nu (used if you want it explicitly)
    def _dcfdnu_on_grids(self, nu: float, eps: float = 1e-6) -> List[np.ndarray]:
        # central difference on the *CF* itself, per grid
        ph_plus = self._model_cf_on_grids(nu + eps)
        ph_minus = self._model_cf_on_grids(nu - eps)
        return [(p - m) / (2.0 * eps) for p, m in zip(ph_plus, ph_minus)]

    # ---- fit nu by L-BFGS-B ----
    def fit(self, nu0: float = 8.0, return_dict: bool = True):
        """
        Estimate nu by minimizing weighted squared error between
        the ECF and the MVT CF on the COS grids.

        Parameters
        ----------
        nu0 : float
            Starting value for degrees of freedom.
        return_dict : bool
            If True returns a dict with fitted params; else returns (mu, Sigma, nu).

        Returns
        -------
        dict or tuple
        """
        def fun(nu_arr):
            (nu,) = nu_arr
            return self._loss(nu)

        def jac(nu_arr):
            (nu,) = nu_arr
            return self._grad(nu)

        res = optimize.minimize(
            fun=fun,
            x0=np.array([nu0], dtype=float),
            jac=jac,
            method="L-BFGS-B",
            bounds=[self.bounds_nu]
        )
        self.nu_hat = float(res.x[0])
        self.opt_result_ = res

        if return_dict:
            return {
                "mu": self.mu_hat,
                "Sigma_cov": self.Sigma_hat,
                "nu": self.nu_hat,
                "Sigma_scale": self._Sigma_scale(self.nu_hat),
                "success": res.success,
                "message": res.message,
                "fun": res.fun,
                "nit": res.nit
            }
        else:
            return self.mu_hat, self.Sigma_hat, self.nu_hat

    # ---- COS coefficients A_k (for inversion/pricing) ----
    def cos_A(self, nu: Optional[float] = None) -> np.ndarray:
        """
        Compute A_{k1..kd} coefficients for the COS expansion:
            A_k = 2 * prod_i 1/(b_i - a_i) * sum_s Re[ exp(-i t·a) φ(t) ],
        where t = π s ⊙ k / (b - a).

        Returns
        -------
        A : complex array with shape K (same as grids)
        """
        if nu is None:
            if not hasattr(self, "nu_hat"):
                raise ValueError("Call fit() or pass nu explicitly.")
            nu = self.nu_hat

        Sig = self._Sigma_scale(nu)
        inv_box = np.prod(1.0 / (self.b - self.a))
        A = np.zeros(tuple(self.K.tolist()), dtype=np.complex128)

        for s, t_stack in zip(self.S, self.t_stacks):
            phi = mvt_cf_on_grid(t_stack, self.mu_hat, Sig, nu, dtype=self.dtype)
            # factor exp(-i t·a)
            phase = np.exp(-1j * np.tensordot(t_stack, self.a, axes=([-1], [0])))
            A += np.real(phase * phi)

        A *= 2.0 * inv_box
        return A
