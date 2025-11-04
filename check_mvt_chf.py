import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple, List
from itertools import product

from scipy import optimize, special

# Try using your fast ECF kernel; fall back to a simple (but slower) version
try:
    from ecf_einsum import ecf_grid_einsum
    HAS_ECF_KERNEL = True
except Exception:
    HAS_ECF_KERNEL = False


# ---------------------------- Utilities ----------------------------

 def simulate_mvt(n: int, mu: np.ndarray, Sigma_cov: np.ndarray, nu: float, rng=None) -> np.ndarray:
    """
    Simulate from a d-dim multivariate Student-t with mean mu, covariance Sigma_cov, dof nu.
    Uses standard Normal + scaled-chi2 mixture; requires nu > 2.
    """
    if rng is None:
        rng = np.random.default_rng(7)
    d = mu.size
    # Convert covariance to *scale* for t: Sigma_scale = ((nu-2)/nu) * Sigma_cov
    Sigma_scale = ((nu - 2.0) / nu) * Sigma_cov
    L = np.linalg.cholesky(Sigma_scale)

    z = rng.standard_normal((n, d))
    u = rng.chisquare(df=nu, size=n)  # chi^2_nu
    w = np.sqrt(u / nu).reshape(-1, 1)  # scale per sample
    x = mu + (z @ L.T) / w
    return x


def all_sign_patterns(d: int, first_is_pos=True) -> np.ndarray:
    """2^(d-1) sign patterns with the first sign fixed to +1 (if requested)."""
    if first_is_pos:
        rest = list(product([-1, 1], repeat=d-1))
        S = np.array([[1] + list(r) for r in rest], dtype=int)
    else:
        S = np.array(list(product([-1, 1], repeat=d)), dtype=int)
    return S


def build_axis_frequencies(a: np.ndarray, b: np.ndarray, s: np.ndarray, K: np.ndarray) -> List[np.ndarray]:
    """
    Returns a list [t1, t2, ..., td] where tj has shape (Kj,), tj = pi * s_j * (0..Kj-1)/(b_j-a_j).
    """
    alpha = np.pi * s / (b - a)
    ts = [alpha[j] * np.arange(K[j], dtype=float) for j in range(len(K))]
    return ts


def mvt_cf_slice_broadcast(k1: int,
                           t_axes: List[np.ndarray],     # [t1,t2,t3,t4,t5], each 1D arrays
                           mu: np.ndarray,               # (d,)
                           Sigma_scale: np.ndarray,      # (d,d) *scale* matrix
                           nu: float) -> np.ndarray:
    """
    Compute the multivariate Student-t CF on the 4D slice with fixed k1, i.e.
      phi[k2,k3,k4,k5] for t = [t1[k1], t2[k2], t3[k3], t4[k4], t5[k5]].

    Uses: φ(t) = exp(i t·mu) * 2^{1-ν/2}/Γ(ν/2) * (√(ν t^T Σ t))^{ν/2} K_{ν/2}(√(ν t^T Σ t))
    with the convention φ(0)=1 (limit).
    """
    # Unpack (assume d=5 for this test)
    t1, t2, t3, t4, t5 = t_axes
    d = 5
    assert len(mu) == d == Sigma_scale.shape[0] == Sigma_scale.shape[1]

    # Build broadcastable axes
    T1 = float(t1[k1])                            # scalar
    T2 = t2[:, None, None, None]                  # (K2,1,1,1)
    T3 = t3[None, :, None, None]                  # (1,K3,1,1)
    T4 = t4[None, None, :, None]                  # (1,1,K4,1)
    T5 = t5[None, None, None, :]                  # (1,1,1,K5)

    # Quadratic form q = t^T Σ t (broadcast to (K2,K3,K4,K5))
    S = Sigma_scale
    # Diagonal terms
    q = (S[0, 0] * T1**2
         + S[1, 1] * T2**2
         + S[2, 2] * T3**2
         + S[3, 3] * T4**2
         + S[4, 4] * T5**2)
    # Cross terms (2 * Σ_ij t_i t_j)
    q += 2.0 * (
        S[0, 1] * T1 * T2 + S[0, 2] * T1 * T3 + S[0, 3] * T1 * T4 + S[0, 4] * T1 * T5
        + S[1, 2] * T2 * T3 + S[1, 3] * T2 * T4 + S[1, 4] * T2 * T5
        + S[2, 3] * T3 * T4 + S[2, 4] * T3 * T5
        + S[3, 4] * T4 * T5
    )

    # Linear term t·mu (same shape as q)
    lin = (mu[0] * T1
           + mu[1] * T2
           + mu[2] * T3
           + mu[3] * T4
           + mu[4] * T5)

    order = nu / 2.0
    z = np.sqrt(np.maximum(0.0, nu) * np.maximum(0.0, q))
    # Numerically stable g(z)
    g = np.ones_like(z, dtype=float)
    mask = z > 0
    if np.any(mask):
        pref = 2.0 ** (1.0 - order) / special.gamma(order)
        K = special.kv(order, z[mask])
        g[mask] = pref * (z[mask] ** order) * K
        # Fallback if numerical issues:
        bad = ~np.isfinite(g[mask])
        if np.any(bad):
            zb = z[mask][bad]
            Kb = special.kv(order, zb)
            g[mask][bad] = pref * np.exp(order * np.log(zb)) * Kb

    phi = np.exp(1j * lin) * g
    return phi  # complex array (K2,K3,K4,K5)


def compute_weights_slice(k1: int, t_axes: List[np.ndarray], decay: float) -> np.ndarray:
    """
    Exponential decay weights w = exp(-decay * ||t||) on the 4D slice.
    """
    t1, t2, t3, t4, t5 = t_axes
    T1 = float(t1[k1])
    T2 = t2[:, None, None, None]
    T3 = t3[None, :, None, None]
    T4 = t4[None, None, :, None]
    T5 = t5[None, None, None, :]
    norms = np.sqrt(T1**2 + T2**2 + T3**2 + T4**2 + T5**2)
    return np.exp(-decay * norms)


def ecf_on_grid_single_sign(X: np.ndarray,
                            a: np.ndarray, b: np.ndarray,
                            s: np.ndarray, K: np.ndarray,
                            dtype=np.complex64) -> np.ndarray:
    """
    ECF over a full K1x...xKd grid for a single sign vector s.
    Uses the provided fast kernel if available.
    """
    if HAS_ECF_KERNEL:
        return ecf_grid_einsum(X, a, b, s, K, dtype=dtype)

    # Fallback (simple-and-slow): directly evaluate exp(i X t) for each grid point.
    # This fallback *materializes* the full grid and may be very slow at 32^5.
    # Kept here only so the file runs even without your optimized kernel.
    print("[WARN] ecf_einsum not found; using slow fallback.")
    d = X.shape[1]
    ts = build_axis_frequencies(a, b, s, K)
    grids = np.meshgrid(*ts, indexing='ij')  # list of d arrays each with shape K1..Kd
    t_stack = np.stack(grids, axis=-1).reshape(-1, d)  # (K_total, d)

    exps = np.exp(1j * (X @ t_stack.T))  # (N, K_total)
    phi = exps.mean(axis=0).reshape(*K.tolist()).astype(dtype, copy=False)
    return phi



@dataclass
class MVTCFEstimator:
    X: np.ndarray                    # (N,d)
    K: Iterable[int]
    a: Optional[np.ndarray] = None
    b: Optional[np.ndarray] = None
    decay: float = 0.03              # frequency decay in the objective
    use_all_signs: bool = False      # if True uses the full 2^(d-1) sign set
    dtype_ecf: np.dtype = np.complex64
    L_box: float = 7.0               # default box = mu ± L * std

    def __post_init__(self):
        self.N, self.d = self.X.shape
        self.K = np.asarray(self.K, dtype=int)
        assert len(self.K) == self.d

        # sample mean/cov
        self.mu_hat = self.X.mean(axis=0)
        Xc = self.X - self.mu_hat
        self.Sigma_hat = (Xc.T @ Xc) / (self.N - 1)

        if self.a is None or self.b is None:
            std = np.sqrt(np.clip(np.diag(self.Sigma_hat), 1e-12, np.inf))
            self.a = self.mu_hat - self.L_box * std
            self.b = self.mu_hat + self.L_box * std
        else:
            self.a = np.asarray(self.a, float)
            self.b = np.asarray(self.b, float)

        # Sign patterns
        self.S = all_sign_patterns(self.d, first_is_pos=True)
        if not self.use_all_signs:
            self.S = self.S[:1]  # only [+1,...,+1]

        # Build axis frequencies per sign up front (cheap)
        self.t_axes_per_s = [build_axis_frequencies(self.a, self.b, s, self.K) for s in self.S]

        # Precompute ECF(s) once (these do NOT depend on nu)
        self._ecf_per_s = [ecf_on_grid_single_sign(self.X, self.a, self.b, s, self.K, dtype=self.dtype_ecf)
                           for s in self.S]

    def _Sigma_scale(self, nu: float) -> np.ndarray:
        # Treat Sigma_hat as covariance; convert to t-scale:
        return ((nu - 2.0) / nu) * self.Sigma_hat

    def _loss(self, nu: float) -> float:
        Sig = self._Sigma_scale(nu)
        loss = 0.0
        for ecf, t_axes in zip(self._ecf_per_s, self.t_axes_per_s):
            # process k1 slices to keep memory modest
            for k1 in range(self.K[0]):
                phi_model = mvt_cf_slice_broadcast(k1, t_axes, self.mu_hat, Sig, nu)
                w = compute_weights_slice(k1, t_axes, self.decay)

                # de-weight DC cell (k==0 vector)
                if k1 == 0:
                    w = w.copy()
                    w[0, 0, 0, 0] = 0.0

                slice_ecf = ecf[k1]  # (K2,K3,K4,K5)
                diff2 = np.abs(phi_model - slice_ecf) ** 2
                loss += np.sum(w * diff2).real
        return loss / len(self.S)

    def _grad(self, nu: float, eps: float = 1e-4) -> np.ndarray:
        f1 = self._loss(nu + eps)
        f0 = self._loss(nu - eps)
        return np.array([(f1 - f0) / (2.0 * eps)])

    def fit(self, nu0: float = 8.0, bounds: Tuple[float, float] = (2.05, 200.0)):
        def fun(v):
            return self._loss(v[0])
        def jac(v):
            return self._grad(v[0])

        res = optimize.minimize(fun, x0=np.array([nu0]),
                                jac=jac, method="L-BFGS-B", bounds=[bounds])
        self.nu_hat = float(res.x[0])
        self.opt_result_ = res
        return {"mu": self.mu_hat,
                "Sigma_cov": self.Sigma_hat,
                "nu": self.nu_hat,
                "success": res.success,
                "message": res.message,
                "fun": res.fun,
                "nit": res.nit}



if __name__ == "__main__":
    rng = np.random.default_rng(123)

    d = 5
    mu_true = np.array([0.02, -0.01, 0.015, 0.0, -0.005])
    sds = np.array([0.08, 0.06, 0.07, 0.09, 0.05])
    corr = 0.3
    R = (1 - corr) * np.eye(d) + corr * np.ones((d, d))
    Sigma_cov_true = (sds[:, None] @ sds[None, :]) * R
    nus = [3.5, 5.0, 7.25, 10.0, 15.0]
    for nu_true in nus:

        
        N = 12 * 365
        X = simulate_mvt(N, mu_true, Sigma_cov_true, nu_true, rng=rng)
    
        
        cols = [f"x{i+1}" for i in range(d)]
        df = pd.DataFrame(X, columns=cols)
    
        
        K = np.array([16] * d, dtype=int)
    
        est = MVTCFEstimator(
            X=df[cols].to_numpy(),
            K=K,
            decay=0.03,          # tweak if high-freqs dominate
            use_all_signs=False, # set True if you want all 2^(d-1) sign patterns
            dtype_ecf=np.complex64,
            L_box=7.0
        )
    
        out = est.fit(nu0=8.0, bounds=(2.05, 200.0))
    
        # --- Report ---
        print("\n=== True vs Estimated parameters ===")
        print(f"True ν:        {nu_true:.4f}")
        print(f"Estimated ν:   {out['nu']:.4f}")
        print(f"Optimize ok?:  {out['success']}, message: {out['message']}, iters: {out['nit']}")
        print(f"Objective @ ν*: {out['fun']:.6e}")
    
        # Optional quick sanity check at the truth (no optimization):
        # est._ecf_per_s is cached; we only recompute the model CF term.
        f_true = est._loss(nu_true)
        print(f"Objective @ ν(true): {f_true:.6e}\n")
