import numpy as np
from numpy.linalg import eigvalsh
from scipy.special import kv as besselk, gamma, loggamma
from scipy.optimize import minimize
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from ecf_einsum import ecf_grid_einsum  # fast empirical CF on grid  



def _alphas_from_box(a: np.ndarray, b: np.ndarray, s: np.ndarray) -> np.ndarray:
    return np.pi * s / (b - a)

def _grid_t_stack(alphas: np.ndarray, K: np.ndarray) -> np.ndarray:
    d = len(K)
    axes = [alphas[j] * np.arange(K[j]) for j in range(d)]
    grids = np.meshgrid(*axes, indexing='ij')  # list of d arrays, each shape K1..Kd
    return np.stack(grids, axis=-1)  # (..., d)

def _tquad_and_lin(t_stack: np.ndarray, Sigma: np.ndarray, mu: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # q = t^T Σ t, lin = t^T μ  on the whole grid; result shapes are K1..Kd
    q = np.einsum('...i,ij,...j->...', t_stack, Sigma, t_stack, optimize=True)
    lin = np.tensordot(t_stack, mu, axes=([-1], [0]))
    return q, lin

def _student_t_cf_from_q(q: np.ndarray, lin: np.ndarray, nu: float) -> np.ndarray:
    """
    CF for multivariate Student t with df=nu, location via `lin`, and scale through q = t^T Σ t.
    Uses the closed-form with modified Bessel K. Handles the origin robustly.
    """
    # z = sqrt(nu * q)
    z = np.sqrt(np.maximum(nu * q, 0.0))
    v = 0.5 * nu

    # Avoid z=0 singularity; at z=0 the CF magnitude term tends to 1.
    small = (z < 1e-12)
    out = np.empty_like(z, dtype=np.complex128)

    # Main formula: ((z)**v * K_v(z)) / (2^{v-1} Γ(v))
    # Compute in log-domain for stability for moderate/large z.
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        # direct branch
        mag = (z**v) * besselk(v, z) / (2.0**(v - 1.0) * gamma(v))
        mag = np.where(small, 1.0, mag)  # set the removable singularity exactly
    out = np.exp(1j * lin) * mag.astype(np.complex128)
    return out

def _student_t_cf_on_grid(a: np.ndarray, b: np.ndarray, s: np.ndarray, K: np.ndarray,
                          mu: np.ndarray, Sigma_scale: np.ndarray, nu: float) -> np.ndarray:
    """Model CF evaluated on the same grid definition as the ECF."""
    alphas = _alphas_from_box(a, b, s)
    t_stack = _grid_t_stack(alphas, K)     # (..., d)
    q, lin = _tquad_and_lin(t_stack, Sigma_scale, mu)
    return _student_t_cf_from_q(q, lin, nu)

def _finite_diff_dphi_dnu(a, b, s, K, mu, Sigma_scale_fn, nu, eps=1e-3):
    """
    Central difference derivative of the CF wrt nu on the grid:
      dphi/dnu ≈ [phi(nu+eps) - phi(nu-eps)] / (2 eps)
    Sigma_scale_fn handles the fact that Sigma_scale depends on nu.
    """
    nu_p = nu + eps
    nu_m = max(2.001, nu - eps)  # ensure > 2 due to covariance usage

    phi_p = _student_t_cf_on_grid(a, b, s, K, mu, Sigma_scale_fn(nu_p), nu_p)
    phi_m = _student_t_cf_on_grid(a, b, s, K, mu, Sigma_scale_fn(nu_m), nu_m)
    return (phi_p - phi_m) / (nu_p - nu_m)

# ---------- Fitting ν by CF–ECF matching ----------

@dataclass
class CFMatchConfig:
    K: Sequence[int]                   # grid sizes per dimension (e.g., [24,24,24,24,24,24])
    s: Optional[Sequence[float]] = None
    box_k: float = 7.0                 # a,b = mu ± box_k * std per dim
    nu_bounds: Tuple[float, float] = (2.05, 100.0)  # keep >2 because we use sample covariance
    eps_nu: float = 1e-3               # step for derivative wrt nu
    ridge: float = 1e-8                # jitter for near-singular covariance
    weight_power: float = 0.0          # optional weight ~ (1 + q)^(-weight_power/2); 0 = unweighted

def _safe_cov(X: np.ndarray, ridge: float) -> np.ndarray:
    C = np.cov(X, rowvar=False, bias=False)
    # tiny ridge if needed
    lam_min = eigvalsh(C).min()
    if lam_min <= 0:
        C = C + (ridge - min(0.0, lam_min) + 1e-12) * np.eye(C.shape[0])
    return C

def _make_box(mu: np.ndarray, Sigma_like: np.ndarray, box_k: float, s: Optional[np.ndarray], K: np.ndarray):
    std = np.sqrt(np.maximum(np.diag(Sigma_like), 1e-12))
    a = mu - box_k * std
    b = mu + box_k * std
    if s is None:
        s = np.ones_like(mu)
    return a, b, np.asarray(s, dtype=float), np.asarray(K, dtype=int)

def _weights_from_q(a, b, s, K, Sigma_scale, power: float):
    if power <= 0:
        return None
    alphas = _alphas_from_box(a, b, s)
    t_stack = _grid_t_stack(alphas, K)
    q = np.einsum('...i,ij,...j->...', t_stack, Sigma_scale, t_stack, optimize=True)
    return (1.0 + q)**(-0.5 * power)

def fit_nu_by_cf_match(X_raw: np.ndarray,
                       dims_keep: Optional[Sequence[int]] = None,
                       cfg: CFMatchConfig = CFMatchConfig(K=[16,16,16,16,16,16])):
    """
    X_raw: (N, D_full) feature matrix; may contain NaNs in some columns (e.g., missing neighbor hour).
    dims_keep: subset of columns to use (e.g., choose only columns available for this hour).
    Returns: dict with nu_hat, mu_hat, Cov_hat, Sigma_scale(nu_hat), diagnostics.
    """
    # --- select dimensions & drop rows with NaNs ---
    if dims_keep is not None:
        X = X_raw[:, dims_keep]
    else:
        X = X_raw
    mask = ~np.isnan(X).any(axis=1)
    X = X[mask]
    if X.shape[0] < 50:  # arbitrary minimal sample size guard
        raise ValueError("Not enough valid rows after masking NaNs.")

    # --- sample stats (your spec) ---
    mu_hat = X.mean(axis=0)
    Cov_hat = _safe_cov(X, ridge=cfg.ridge)

    # --- grid box ---
    a, b, s, K = _make_box(mu_hat, Cov_hat, cfg.box_k, cfg.s, np.asarray(cfg.K, dtype=int))

    # --- empirical CF on the same grid (your fast routine) ---
    # Note: ecf_grid_einsum expects raw X; it does NOT center by mu.
    ecf = ecf_grid_einsum(X, a, b, s, K, dtype=np.complex128)  # :contentReference[oaicite:3]{index=3}

    # --- mapping from nu -> Sigma_scale(nu) so Cov(model) equals Cov_hat for each nu (>2) ---
    def Sigma_scale_of(nu: float) -> np.ndarray:
        # Cov(model) == (nu/(nu-2))*Sigma_scale  =>  Sigma_scale = ((nu-2)/nu)*Cov_hat
        factor = max(0.0, (nu - 2.0) / nu)  # safe for nu>2
        return factor * Cov_hat

    # --- optional frequency weighting to de-emphasize very high |t| where ECF is noisy ---
    W = _weights_from_q(a, b, s, K, Sigma_scale_of(10.0), power=cfg.weight_power)  # any nu>2 ok for weights
    if W is None:
        W = 1.0

    # --- objective & gradient in nu (scalar) ---
    def loss_and_grad(nu_arr):
        nu = float(nu_arr[0])
        if not (cfg.nu_bounds[0] < nu < cfg.nu_bounds[1]):
            # soft barrier to keep optimizer inside bounds
            return 1e12, np.array([0.0])

        phi = _student_t_cf_on_grid(a, b, s, K, mu_hat, Sigma_scale_of(nu), nu)
        diff = ecf - phi
        L = np.mean(W * (diff.conj() * diff)).real

        # dL/dnu = 2 * Re <diff, d(diff)/dnu> with negative sign on model CF term
        dphi = _finite_diff_dphi_dnu(a, b, s, K, mu_hat, Sigma_scale_of, nu, eps=cfg.eps_nu)
        dL = 2.0 * np.mean((W * diff.conj() * (-dphi))).real
        return L, np.array([dL])

    # --- optimize ν ---
    x0 = np.array([(cfg.nu_bounds[0] + cfg.nu_bounds[1]) * 0.5])
    res = minimize(lambda x: loss_and_grad(x)[0],
                   x0,
                   jac=lambda x: loss_and_grad(x)[1],
                   method="L-BFGS-B",
                   bounds=[cfg.nu_bounds])

    nu_hat = float(res.x[0])
    Sigma_scale_hat = Sigma_scale_of(nu_hat)
    phi_hat = _student_t_cf_on_grid(a, b, s, K, mu_hat, Sigma_scale_hat, nu_hat)
    loss_final = np.mean((W * (ecf - phi_hat).conj() * (ecf - phi_hat))).real

    return {
        "nu_hat": nu_hat,
        "mu_hat": mu_hat,
        "Cov_hat": Cov_hat,
        "Sigma_scale_hat": Sigma_scale_hat,
        "grid": {"a": a, "b": b, "s": s, "K": K},
        "ecf_shape": ecf.shape,
        "opt_success": bool(res.success),
        "opt_message": res.message,
        "loss_final": loss_final,
        "res": res,
    }
