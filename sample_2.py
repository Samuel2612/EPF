import numpy as np
import pandas as pd
from itertools import product
from dataclasses import dataclass
from typing import Iterable, List, Tuple, Dict, Optional

from scipy.optimize import minimize
from scipy.special import kv as besselk, gamma, digamma
from numpy.linalg import cholesky
from ecf_einsum import ecf_grid_einsum 


def all_sign_vectors(d: int) -> np.ndarray:
    """
    Generate all 2^(d-1) sign vectors with first entry fixed to +1.
    """
    if d == 1:
        return np.array([[1.0]])
    tails = np.array(list(product([-1.0, 1.0], repeat=d-1)), dtype=float)
    heads = np.ones((tails.shape[0], 1), dtype=float)
    return np.hstack([heads, tails])


def build_cos_grid(a: np.ndarray, b: np.ndarray, s: np.ndarray, K: np.ndarray):
    """
    Return stacked grid of frequencies t for a given sign vector s:

        t = π * s ∘ k / (b - a), with k_j ∈ {0,...,K_j-1}

    Shape: (*K, d)
    """
    d = len(K)
    alpha = np.pi * s / (b - a)
    meshes = np.meshgrid(
        *[alpha[j] * np.arange(K[j], dtype=float) for j in range(d)],
        indexing="ij"
    )
    t = np.stack(meshes, axis=-1)  # (..., d)
    return t


def quad_form(t_stack: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
    """
    Compute q(t)=t^T Σ t on the stacked grid. Shape-preserving.
    """
    return np.einsum("...i,ij,...j->...", t_stack, Sigma, t_stack, optimize=True)


def mvt_cf_on_grid(
    a: np.ndarray, b: np.ndarray, s: np.ndarray, K: np.ndarray,
    mu: np.ndarray, Sigma: np.ndarray, nu: float, dtype=np.complex128
) -> np.ndarray:
    """
    Multivariate Student-t CF on COS grid for one sign vector s.
    Uses closed form with modified Bessel K.

    Returns array of shape (*K,) complex.
    """
    t = build_cos_grid(a, b, s, K)               
    q = quad_form(t, Sigma)                       
    z = np.sqrt(np.maximum(0.0, nu * q))          
    phase = np.exp(1j * np.tensordot(t, mu, axes=([-1], [0]))).astype(dtype)


    out = np.empty(q.shape, dtype=dtype)
    mask0 = (q == 0.0)
    if np.any(~mask0):
        z_nz = z[~mask0]
        q_nz = q[~mask0]
        num = (nu ** (nu/2.0)) * (q_nz ** (nu/4.0)) * besselk(nu/2.0, z_nz)
        den = (2.0 ** (nu/2.0 - 1.0)) * gamma(nu/2.0)
        out[~mask0] = phase[~mask0] * (num / den).astype(dtype)
    if np.any(mask0):
        out[mask0] = 1.0 + 0.0j  

    return out.astype(dtype)


def exp_decay_weights_from_q(q: np.ndarray, decay: float) -> np.ndarray:
    """
    Frequency weights w = exp(-decay * sqrt(q(t))). Set decay=0 for uniform.
    """
    return np.exp(-decay * np.sqrt(np.maximum(0.0, q)))



@dataclass
class MVTConfig:
    # COS grid parameters (fixed across fit/eval)
    a: np.ndarray
    b: np.ndarray
    K: np.ndarray                 # integers per dimension
    s_list: Optional[np.ndarray] = None  # (m, d). If None, use all_sign_vectors(d)
    decay: float = 0.05           # exponential decay in frequency domain
    nu_bounds: Tuple[float, float] = (2.05, 200.0)  # ν>2 for finite covariance
    nu_init: float = 8.0
    scale_data: bool = True       # z-score the features before moments
    complex_dtype: any = np.complex128


class MVT:
    """
    Fit a multivariate Student-t characteristic function on a COS grid by
    matching to the empirical characteristic function (ECF) computed with a
    fast einsum builder (your ecf_einsum.py).

    Parameters are μ (sample mean), Σ (sample covariance), ν (df by CF-matching).
    """
    def __init__(self, config: MVTConfig):
        self.cfg = config
        self.mu_: Optional[np.ndarray] = None
        self.Sigma_: Optional[np.ndarray] = None
        self.nu_: Optional[float] = None
        self._fit_stats: Dict = {}

    # ----- feature preprocessing / moments -----
    def _prepare_matrix(self, df: pd.DataFrame, cols: List[str]) -> np.ndarray:
        X = np.asarray(df[cols].values, dtype=float)
        if self.cfg.scale_data:
            X = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, ddof=1, keepdims=True) + 1e-12)
        return X

    @staticmethod
    def _mu(X: np.ndarray) -> np.ndarray:
        return X.mean(axis=0)

    @staticmethod
    def _Sigma(X: np.ndarray) -> np.ndarray:
        # unbiased sample covariance (row-wise observations)
        return np.cov(X, rowvar=False, ddof=1)

    # ----- grids / ECF -----
    def _s_list(self, d: int) -> np.ndarray:
        return self.cfg.s_list if self.cfg.s_list is not None else all_sign_vectors(d)

    def _ecf_on_grid(self, X: np.ndarray, s: np.ndarray) -> np.ndarray:
        # Uses your efficient einsum-based ECF builder on the COS grid
        return ecf_grid_einsum(
            X, self.cfg.a, self.cfg.b, s, self.cfg.K, dtype=self.cfg.complex_dtype
        )  # shape (*K,)

    # ----- loss on a single s -----
    def _grid_and_weights(self, s: np.ndarray, Sigma: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        t = build_cos_grid(self.cfg.a, self.cfg.b, s, self.cfg.K)
        q = quad_form(t, Sigma)
        w = exp_decay_weights_from_q(q, self.cfg.decay)
        return q, w

    def _loss_for_nu(
        self, nu: float, mu: np.ndarray, Sigma: np.ndarray,
        ecf_by_s: Dict[Tuple[float, ...], np.ndarray]
    ) -> float:
        loss = 0.0
        for s_vec, ecf in ecf_by_s.items():
            s = np.array(s_vec, dtype=float)
            model = mvt_cf_on_grid(self.cfg.a, self.cfg.b, s, self.cfg.K, mu, Sigma, nu, self.cfg.complex_dtype)
            q, w = self._grid_and_weights(s, Sigma)
            diff = model - ecf
            loss += np.sum(w * (diff.real**2 + diff.imag**2)) / w.size
        return float(loss / len(ecf_by_s))

    def _dloss_dnu_num(
        self, nu: float, mu: np.ndarray, Sigma: np.ndarray,
        ecf_by_s: Dict[Tuple[float, ...], np.ndarray]
    ) -> float:
        # Robust central difference; step proportional to ν
        h = max(1e-3, 1e-2 * nu)
        f1 = self._loss_for_nu(nu + h, mu, Sigma, ecf_by_s)
        f0 = self._loss_for_nu(nu - h, mu, Sigma, ecf_by_s)
        return (f1 - f0) / (2.0 * h)

    # ----- public API -----
    def fit(self, df: pd.DataFrame, feature_cols: List[str]) -> Dict[str, np.ndarray]:
        """
        1) Extract/scale features
        2) μ, Σ from sample moments
        3) ν by CF-matching on the COS grid using L-BFGS-B
        """
        X = self._prepare_matrix(df, feature_cols)
        d = X.shape[1]
        mu = self._mu(X)
        Sigma = self._Sigma(X)

        # Precompute ECF for each sign vector
        s_mat = self._s_list(d)
        ecf_by_s = {tuple(s): self._ecf_on_grid(X, s) for s in s_mat}

        # Objective wrapper
        def fun(nu_scalar):
            nu = float(nu_scalar[0])
            return self._loss_for_nu(nu, mu, Sigma, ecf_by_s)

        def jac(nu_scalar):
            nu = float(nu_scalar[0])
            return np.array([self._dloss_dnu_num(nu, mu, Sigma, ecf_by_s)], dtype=float)

        # Optimize ν
        res = minimize(
            fun, x0=np.array([self.cfg.nu_init], dtype=float),
            method="L-BFGS-B",
            jac=jac,
            bounds=[self.cfg.nu_bounds],
            options=dict(maxiter=200, ftol=1e-10)
        )

        self.mu_, self.Sigma_, self.nu_ = mu, Sigma, float(res.x[0])
        self._fit_stats = dict(success=res.success, message=res.message, nfev=res.nfev, loss=res.fun)

        return dict(mu=self.mu_, Sigma=self.Sigma_, nu=self.nu_, stats=self._fit_stats)

    # Evaluate the fitted CF on the COS grid for a given sign vector (or all of them)
    def cf_on_grid(self, s: Optional[np.ndarray] = None) -> Dict[Tuple[float, ...], np.ndarray]:
        assert self.mu_ is not None and self.Sigma_ is not None and self.nu_ is not None, "Call fit() first."
        d = self.mu_.shape[0]
        s_mat = np.atleast_2d(s) if s is not None else self._s_list(d)
        out = {}
        for svec in s_mat:
            s_key = tuple(np.asarray(svec, float))
            out[s_key] = mvt_cf_on_grid(self.cfg.a, self.cfg.b, np.asarray(svec, float),
                                        self.cfg.K, self.mu_, self.Sigma_, self.nu_, self.cfg.complex_dtype)
        return out

