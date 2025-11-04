#!/usr/bin/env python3
"""
COS–TT error study: Gaussian vs Student-t (same plot, per-distribution boxes)

- Computes average |TT − exact| for conditional raw moments E[Y^p|X], p=1..4
- K per dimension in {4, 8, ..., MAX_K}
- Single log–log figure: Gaussian lines (winter), Student-t lines (autumn)
- NEW: different truncation boxes per distribution (e.g., 8σ vs 5σ)
"""

import itertools, math
from typing import List, Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import kve, gammaln, gamma
from numpy.linalg import cholesky
from scipy.stats import norm, t as t_dist
# --------------------------
# Config (tweak for runtime)
# --------------------------
MAX_K         = 512        # bump to 1024/2048 later if needed
TT_EPS        = 1e-15     # relax if you need more speed (e.g., 1e-10)
N_XS          = 150          # X samples kept inside box for averaging
NU_T          = 5       # Student-t df
BOX_STD_GAUSS = 10.0          # per-axis half-width in stds for Gaussian
BOX_STD_T     = 100.0         # per-axis half-width in stds for Student-t
CUMU_L_GAUSS   = 8.0      # L for Gaussian cumulant box
ALPHA_GAUSS = 1e-15   # two-sided tail for Gaussian box (e.g., 1e-8 ⇒ ~5.61σ)
ALPHA_T     = 1e-6   # two-sided tail for Student-t box (heavier tails ⇒ larger alpha)
CUMU_L_T       = 10.0      # L for Student-t cumulant box (tails are heavier)
RNG_SEED      = 42



# ==========================
# Utilities
# ==========================
def quantile_box_gaussian(mu: np.ndarray, Sigma: np.ndarray, alpha: float):
    """
    Per-axis quantile box for a multivariate Gaussian N(mu, Sigma).
    For each axis j: a_j = Q_{alpha/2}, b_j = Q_{1-alpha/2} of N(mu_j, sqrt(Sigma_jj)).
    """
    mu = np.asarray(mu, float)
    s  = np.sqrt(np.maximum(0.0, np.diag(np.asarray(Sigma, float))))
    ql = norm.ppf(alpha/2.0, loc=mu, scale=s)
    qh = norm.ppf(1.0 - alpha/2.0, loc=mu, scale=s)
    return ql, qh


def quantile_box_student_t(mu: np.ndarray, Sigma: np.ndarray, nu: float, alpha: float):
    """
    Per-axis quantile box for a multivariate Student-t with df=nu and scatter Sigma.
    Each marginal is t_ν(μ_j, scale = sqrt(Sigma_jj)).
    """
    mu = np.asarray(mu, float)
    s  = np.sqrt(np.maximum(0.0, np.diag(np.asarray(Sigma, float))))
    ql = t_dist.ppf(alpha/2.0, df=nu, loc=mu, scale=s)
    qh = t_dist.ppf(1.0 - alpha/2.0, df=nu, loc=mu, scale=s)
    return ql, qh

def index_grid(Ks):
    grid = np.indices(Ks, dtype=np.float64)
    return np.moveaxis(grid, 0, -1)

def chf_gaussian(t, mu, Sigma):
    tf   = t.reshape(-1, t.shape[-1])                    # (N,d)
    phase = 1j * tf @ mu                                 # (N,)
    quad  = np.einsum("ij,jk,ik->i", tf, Sigma, tf)      # (N,)
    return np.exp(phase - 0.5 * quad).reshape(t.shape[:-1])

def chf_student_t_besselk(t, mu, Sigma, nu: float, small_z: float = 1e-10):
    tf = t.reshape(-1, t.shape[-1])
    r2 = np.einsum("ij,jk,ik->i", tf, Sigma, tf)
    r  = np.sqrt(np.maximum(r2, 0.0))
    z  = np.sqrt(nu) * r
    v  = 0.5 * nu
    # kve returns e^z * K_v(z); so K_v(z) = e^{-z} * kve(v,z)
    logC = (1.0 - v) * np.log(2.0) - gammaln(v) + v * np.log(np.maximum(z, 1.0))
    core = np.exp(logC - z) * kve(v, z)
    core = np.where(z < small_z, 1.0, core)
    phase = np.exp(1j * (tf @ mu))
    return (phase * core).reshape(t.shape[:-1])

def chf_student_t_besselk2(t, mu, Sigma, nu, small_z=1e-20):
    """
    Multivariate Student-t characteristic function, log-stable.
    Uses log-Gamma (gammaln) and kve for numerical robustness.

    Parameters
    ----------
    t : array_like (..., d)
    mu : array_like (d,)
    Sigma : array_like (d, d) positive-definite scatter
    nu : float > 0
    small_z : threshold for small-argument handling

    Returns
    -------
    complex ndarray with shape t.shape[:-1] (or complex scalar)
    """
    t  = np.asarray(t,  dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    Sigma = 0.5*(np.asarray(Sigma, dtype=np.float64) + np.asarray(Sigma, dtype=np.float64).T)

    d = mu.shape[-1]
    if t.shape[-1] != d or Sigma.shape != (d, d):
        raise ValueError("Shapes must be: t[..., d], mu[d], Sigma[d,d].")
    if nu <= 0:
        raise ValueError("nu must be > 0.")
    # PD check
    cholesky(Sigma)

    # quadratic form and r
    quad = np.einsum('...i,ij,...j->...', t, Sigma, t, optimize=True)
    r = np.sqrt(np.maximum(quad, 0.0))
    z = np.sqrt(nu) * r
    v = 0.5 * nu

    # phase exp(i t^T mu)
    phase = np.exp(1j * np.einsum('...i,i->...', t, mu, optimize=True))

    base = np.empty_like(r, dtype=np.float64)

    # small-z: exact limiting value is 1; add quadratic term if variance exists
    small = (z < small_z)
    if np.any(small):
        base[small] = 1.0
        if nu > 2:
            # φ(t) ≈ 1 - 0.5 * t^T Cov t, with Cov = (nu/(nu-2)) Σ
            base[small] -= 0.5 * (nu/(nu-2.0)) * quad[small]

    # big-z: build prefactor in log-domain and multiply by kve with exp(-z)
    big = ~small
    if np.any(big):
        zb = z[big]
        # log( 2^{1-v} z^{v} / Γ(v) ) = (1-v)ln2 + v ln z - ln Γ(v)
        log_pref = (1.0 - v) * np.log(2.0) + v * np.log(zb) - gammaln(v)
        # since kve(v,z) = e^{z} K_v(z), we want e^{-z} * kve(v,z)
        base[big] = np.exp(log_pref - zb) * kve(v, zb)

    out = phase * base
    return out if out.shape else out.item()


def chf_student_t_besselk3(t, mu, Sigma, nu, small_z=1e-15):
    """
    Multivariate Student-t characteristic function (double precision).

    Parameters
    ----------
    t : array_like, shape (d,) or (..., d)
        Evaluation direction(s).
    mu : array_like, shape (d,)
        Location vector.
    Sigma : array_like, shape (d, d)
        Positive-definite scatter matrix.
    nu : float
        Degrees of freedom (>0).
    small_z : float
        Threshold for small-argument safeguard.

    Returns
    -------
    phi : complex ndarray with shape t.shape[:-1] (or complex scalar)
    """
    t  = np.asarray(t,  dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    Sigma = np.asarray(Sigma, dtype=np.float64)
    d = mu.shape[-1]
    if t.shape[-1] != d or Sigma.shape != (d, d):
        raise ValueError("Shapes must be: t[..., d], mu[d], Sigma[d,d].")
    if nu <= 0:
        raise ValueError("nu must be > 0.")

    # Symmetrize Sigma lightly and (optionally) check PD
    Sigma = 0.5*(Sigma + Sigma.T)
    # A cheap PD check (will raise if not PD)
    try:
        _ = cholesky(Sigma)
    except np.linalg.LinAlgError:
        raise ValueError("Sigma must be symmetric positive-definite.")

    # Quadratic form r = sqrt(t^T Sigma t), supports batched t
    # r has shape t.shape[:-1]
    quad = np.einsum('...i,ij,...j->...', t, Sigma, t, optimize=True)
    r = np.sqrt(np.maximum(quad, 0.0))

    v = 0.5 * nu
    # base = 2^{1-v} / Gamma(v) * (sqrt(nu)*r)^v * K_v(sqrt(nu)*r)
    z = np.sqrt(nu) * r
    phi_shape = r.shape
    base = np.empty_like(r, dtype=np.float64)

    # Small-argument safeguard: for very small r, use second-order expansion
    # φ(t) ≈ exp(i t^T mu) * (1 - 0.5 t^T Cov t), where Cov = nu/(nu-2) Σ (if nu>2)
    small = (z < small_z)
    if np.any(small):
        base[small] = 1.0  # we'll multiply by exp(i t^T mu) outside
        if nu > 2:
            coef = 0.5 * (nu / (nu - 2.0))  # scales Σ inside the quadratic form
            base[small] -= coef * quad[small]

    # Regular branch for z >= small_z using kve (overflow-safe)
    big = ~small
    if np.any(big):
        zb = z[big]
        # kve(v, z) = e^{z} K_v(z)  =>  K_v(z) = e^{-z} kve(v, z)
        pref = (2.0**(1.0 - v)) / gamma(v)
        base[big] = pref * (zb**v) * np.exp(-zb) * kve(v, zb)

    # phase = exp(i t^T mu)
    phase = np.exp(1j * np.einsum('...i,i->...', t, mu, optimize=True))
    out = phase * base
    return out if out.shape else out.item()


def marginal_cumulants_gaussian(mu: np.ndarray, Sigma: np.ndarray):
    """
    Per-axis cumulants for a multivariate Gaussian N(mu, Sigma).
    κ1 = mu_j, κ2 = Sigma_jj, κ3 = 0, κ4 = 0 per axis.
    """
    mu = np.asarray(mu, float)
    Sigma = np.asarray(Sigma, float)
    c1 = mu.copy()
    c2 = np.diag(Sigma).astype(float)
    c3 = np.zeros_like(c2)
    c4 = np.zeros_like(c2)
    return c1, c2, c3, c4


def marginal_cumulants_student_t(mu: np.ndarray, Sigma: np.ndarray, nu: float):
    """
    Per-axis cumulants for a multivariate Student-t with df=nu and scatter Sigma.
    For each axis j:
      κ1 = mu_j
      κ2 = Var = (nu/(nu-2)) * Sigma_jj,  (exists if nu>2)
      κ3 = 0                             (exists if nu>3)
      κ4 = (excess kurtosis) * κ2^2 = [6/(nu-4)] * κ2^2  (exists if nu>4)

    Returns NaN for cumulants that don't exist at the given nu.
    """
    mu = np.asarray(mu, float)
    s2 = np.diag(np.asarray(Sigma, float)).astype(float)

    c1 = mu.copy()

    if nu > 2:
        var = (nu / (nu - 2.0)) * s2
    else:
        var = np.full_like(s2, np.nan)
    c2 = var

    c3 = np.zeros_like(s2) if nu > 3 else np.full_like(s2, np.nan)

    if nu > 4:
        excess = 6.0 / (nu - 4.0)
        c4 = excess * (var ** 2)
    else:
        c4 = np.full_like(s2, np.nan)

    return c1, c2, c3, c4


def box_from_cumulants(c1: np.ndarray, c2: np.ndarray, c4: np.ndarray, L: float):
    """
    COS truncation per axis: [a_j, b_j] = c1_j ± L * sqrt( c2_j + sqrt( c4_j ) ).
    If c4_j is NaN (e.g., nu <= 4), we treat sqrt(c4_j) as 0.
    """
    c1 = np.asarray(c1, float)
    c2 = np.asarray(c2, float)
    c4 = np.asarray(c4, float)

    # Safely handle NaNs/infs in κ4 by replacing with 0 contribution
    safe_sqrt_c4 = np.sqrt(np.nan_to_num(c4, nan=0.0, posinf=0.0, neginf=0.0))
    scale = np.sqrt(np.maximum(0.0, c2) + safe_sqrt_c4)

    a = c1 - L * scale
    b = c1 + L * scale
    return a, b


def build_cos_tensor_cf(Ks, a, b, cf_func, **cf_kwargs):
    n      = len(Ks)
    L      = b - a
    factor = 2.0 / L.prod()
    rest   = np.array(list(itertools.product([1.0, -1.0], repeat=n-1)), float)
    Svec   = np.c_[np.ones(rest.shape[0]), rest]        # (2^{n-1},n)

    grid   = index_grid(Ks)                             # (*Ks,n)
    A      = np.zeros(Ks, dtype=np.float64)
    for s in Svec:
        t     = math.pi * grid * s / L
        phi   = cf_func(t, **cf_kwargs)
        phase = np.exp(-1j * (t * a).sum(-1))
        A    += np.real(phase * phi)
    return factor * A

def tt_svd(tensor, eps):
    """Plain TT-SVD."""
    T = np.array(tensor, copy=True)
    dims = T.shape
    d = len(dims)
    if d == 0:
        raise ValueError("Input must be at least 1-D.")

    norm2 = np.linalg.norm(T) ** 2
    denom = max(d - 1, 1)
    thr = (eps / math.sqrt(denom)) ** 2 * norm2

    cores: List[np.ndarray] = []
    ranks: List[int] = [1]

    unfold = T
    for k in range(d - 1):
        m = ranks[-1] * dims[k]
        unfold = unfold.reshape(m, -1)

        U, S, Vh = np.linalg.svd(unfold, full_matrices=False)
        tail = np.cumsum(S[::-1] ** 2)[::-1]
        mask = tail <= thr
        r = int(np.argmax(mask)) + 1 if mask.any() else len(S)
        r = max(1, r)

        cores.append(U[:, :r].reshape(ranks[-1], dims[k], r))
        ranks.append(r)
        unfold = (S[:r, None] * Vh[:r])

    cores.append(unfold.reshape(ranks[-1], dims[-1], 1))
    ranks.append(1)
    return cores, ranks

def tt_svd2(X, eps):
    """
    NumPy-only TT-SVD (Oseledets, 2011) with global relative Frobenius tolerance.

    Parameters
    ----------
    X : np.ndarray
        Full tensor of shape (n1, n2, ..., nd).
    eps : float
        Global relative tolerance. The algorithm uses the standard local rule
        delta = eps / sqrt(d-1) to truncate each SVD so that the discarded tail
        energy at that split is <= (delta^2) * ||unfolding||_F^2.

    Returns
    -------
    cores : list[np.ndarray]
        TT cores [G1, ..., Gd], where Gk has shape (r_{k-1}, n_k, r_k),
        with r_0 = r_d = 1.
    ranks : list[int]
        TT ranks [r1, ..., r_{d-1}].
    """
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy.ndarray")
    shape = X.shape
    d = len(shape)
    if d < 2:
        # Trivial 1D "tensor": single core (1, n1, 1)
        return [X.reshape(1, shape[0], 1)], []

    delta = float(eps) / np.sqrt(d - 1)

    # Maintain the remaining tensor as (r_prev, n_k, n_{k+1}, ..., n_d)
    R = X.reshape((1,) + shape)  # prepend r_prev = 1
    cores = []
    ranks = []

    for k in range(d - 1):
        r_prev = R.shape[0]
        n_k = R.shape[1]

        # Unfold: (r_prev * n_k) x (rest)
        unfolding = R.reshape(r_prev * n_k, -1)

        # Compact SVD
        U, S, VT = np.linalg.svd(unfolding, full_matrices=False)

        # Choose minimal rank r_k by tail energy criterion
        s2 = S**2
        total = s2.sum()
        tau2 = (delta**2) * total
        cum = np.cumsum(s2)
        cutoff = total - tau2
        r_k = int(np.searchsorted(cum, cutoff, side="left") + 1)
        r_k = max(1, min(r_k, S.size))

        # Build core G_k: reshape U[:, :r_k] -> (r_prev, n_k, r_k)
        U_r = U[:, :r_k]
        Gk = U_r.reshape(r_prev, n_k, r_k)
        cores.append(Gk)
        ranks.append(r_k)

        # Form remainder for the next step: diag(S[:r_k]) @ VT[:r_k, :]
        remainder = (S[:r_k, None] * VT[:r_k, :])

        if k < d - 2:
            # Next remaining tensor: (r_k, n_{k+1}, ..., n_d)
            R = remainder.reshape((r_k,) + R.shape[2:])
        else:
            # Last core: reshape to (r_{d-1}, n_d, 1)
            G_last = remainder.reshape(r_k, R.shape[2], 1)
            cores.append(G_last)

    return cores, ranks

def cos_basis(x, a, b, N):
    k   = np.arange(N)
    vec = np.cos(k * math.pi * (x - a) / (b - a))
    vec[0] *= 0.5
    return vec

def closed_form_moment_power(a, b, N, p):
    """J_k = ∫_a^b x^p cos(kπ(x-a)/(b-a)) dx for k>=0."""
    L = b - a
    J = np.zeros(N, dtype=np.float64)
    J[0] = 0.5 * (b**(p + 1) - a**(p + 1)) / (p + 1)
    for k in range(1, N):
        ilambda  = 1j * k * math.pi
        pow_series = [(ilambda)**m / math.factorial(m) for m in range(p + 1)]
        acc = 0.0
        for j in range(p + 1):
            S_j  = sum(pow_series[:j + 1])
            I_j  = math.factorial(j) / ilambda**(j + 1) * (1 - np.exp(ilambda) * S_j)
            acc += math.comb(p, j) * a**(p - j) * L**(j + 1) * I_j.real
        J[k] = acc
    return J

def closed_form_moment_power1(a, b, N, p):
    """
    Robust, real-valued weights J_k^{(p)} for the COS expansion of y^p (p>=0).
    Uses stable recurrences for C_j(α)=∫_0^1 z^j cos(α z) dz, α=kπ,
    with exact parity (no complex exponentials).
    Compatible with prime-sum: we HALVE J[0].
    """
    L = b - a
    J = np.zeros(N, dtype=np.float64)

    # k = 0  (prime-sum: half weight)
    J[0] = 0.5 * (b**(p + 1) - a**(p + 1)) / (p + 1)

    # k >= 1: use C_j/S_j recurrences with α = kπ
    for k in range(1, N):
        alpha = k * math.pi
        inv_alpha = 1.0 / alpha
        # exact parity: sin(α)=0, cos(α)=(-1)^k when α=kπ
        cos_a = -1.0 if (k & 1) else 1.0

        # Base (j=0)
        C_prev = 0.0                     # C_0 = sin α / α = 0
        S_prev = (1.0 - cos_a) * inv_alpha  # S_0 = (1 - cos α)/α

        # Accumulate sum over j
        acc = 0.0
        # j=0 term contributes L^{1} * C_0 = 0, so start from j=1
        for j in range(1, p + 1):
            # C_j = sin α / α - (j/α) S_{j-1} = -(j/α) S_{j-1}
            C_j = - (j * inv_alpha) * S_prev
            # S_j = -cos α / α + (j/α) C_{j-1}
            S_j = - cos_a * inv_alpha + (j * inv_alpha) * C_prev

            acc += math.comb(p, j) * (a ** (p - j)) * (L ** (j + 1)) * C_j

            C_prev, S_prev = C_j, S_j

        # Note: j=0 term is zero since C_0 = 0
        J[k] = acc

    return J

def build_row_core(G0_mat, weights):
    return (weights[:, None] * G0_mat).sum(0, keepdims=True)

def feature_contraction(x_feat, cores, a, b):
    vec = None
    for i, G in enumerate(cores[1:], start=1):
        N     = G.shape[1]
        basis = cos_basis(x_feat[i-1], a[i], b[i], N)
        M     = np.tensordot(G, basis, axes=([1], [0]))
        vec   = M if vec is None else vec @ M
    return vec  # shape (r1,1)

# --------------------------
# Conditional moments (raw)
# --------------------------
def gaussian_conditional_moments(mu, Sigma, x_feat):
    Sigma_yx     = Sigma[0,1:]
    Sigma_xx     = Sigma[1:,1:]
    Sigma_xx_inv = np.linalg.inv(Sigma_xx)
    mu_y, mu_x   = mu[0], mu[1:]
    mu_c   = float(mu_y + Sigma_yx @ Sigma_xx_inv @ (x_feat - mu_x))
    sigma2_base = float(Sigma[0,0] - Sigma_yx @ Sigma_xx_inv @ Sigma_yx)  # Schur complement
    m1 = mu_c
    m2 = sigma2_base + mu_c**2
    m3 = mu_c**3 + 3*mu_c*sigma2_base
    m4 = mu_c**4 + 6*mu_c**2*sigma2_base + 3*sigma2_base**2
    return np.array([m1, m2, m3, m4], dtype=float)

def student_t_conditional_moments(mu, Sigma, nu, x_feat):
    q = len(mu) - 1
    nu_p = nu + q
    Sigma_yx     = Sigma[0,1:]
    Sigma_xx     = Sigma[1:,1:]
    Sigma_xx_inv = np.linalg.inv(Sigma_xx)
    mu_y, mu_x   = mu[0], mu[1:]
    dx = x_feat - mu_x
    mu_c = float(mu_y + Sigma_yx @ Sigma_xx_inv @ dx)
    sigma2_base = float(Sigma[0,0] - Sigma_yx @ Sigma_xx_inv @ Sigma_yx)
    delta_x = float(dx @ Sigma_xx_inv @ dx)
    s2 = ((nu + delta_x) / (nu + q)) * sigma2_base
    def exists(k): return nu_p > k
    m1 = mu_c if exists(1) else np.nan
    if exists(2):
        ET2 = nu_p / (nu_p - 2.0); m2 = mu_c**2 + s2 * ET2
    else:
        m2 = np.nan
    if exists(3):
        ET2 = nu_p / (nu_p - 2.0); m3 = mu_c**3 + 3*mu_c*s2*ET2
    else:
        m3 = np.nan
    if exists(4):
        ET2 = nu_p / (nu_p - 2.0); ET4 = (3.0 * nu_p**2) / ((nu_p - 2.0) * (nu_p - 4.0))
        m4 = mu_c**4 + 6*mu_c**2*s2*ET2 + (s2**2) * ET4
    else:
        m4 = np.nan
    return np.array([m1, m2, m3, m4], dtype=float)

# --------------------------
# Helpers for boxes & samples
# --------------------------
def make_box(mu: np.ndarray, Sigma: np.ndarray, std_mult: float) -> Tuple[np.ndarray, np.ndarray]:
    std = np.sqrt(np.diag(Sigma))
    a = mu - std_mult * std
    b = mu + std_mult * std
    return a, b

def sampler_gaussian(mu, Sigma, a, b, rng, n=N_XS):
    mu_x = mu[1:]
    Sig_xx = Sigma[1:,1:]
    Lx = np.linalg.cholesky(Sig_xx)
    z = rng.normal(size=(n, len(mu_x)))
    xs = mu_x + z @ Lx.T
    mask = (xs >= a[1:]).all(axis=1) & (xs <= b[1:]).all(axis=1)
    return xs[mask]

def sampler_student(mu, Sigma, a, b, rng, nu=NU_T, n=N_XS):
    mu_x = mu[1:]
    Sig_xx = Sigma[1:,1:]
    Lx = np.linalg.cholesky(Sig_xx)
    z = rng.normal(size=(n, len(mu_x)))
    chi2 = rng.chisquare(df=nu, size=n)
    scales = np.sqrt(nu / chi2)
    xs = mu_x + (z @ Lx.T) * scales[:, None]
    mask = (xs >= a[1:]).all(axis=1) & (xs <= b[1:]).all(axis=1)
    return xs[mask]

# --------------------------
# Experiment driver
# --------------------------
def run_and_plot():
    # 2D setup (Y, X)
    mu    = np.array([0.0, 0.0])
    Sigma = np.array([[0.5, 0.2],
                     [0.2, 0.5]], dtype=float)

    # Per-distribution boxes
    # a_g, b_g = make_box(mu, Sigma, BOX_STD_GAUSS)
    # a_t, b_t = make_box(mu, Sigma, BOX_STD_T)
    
    # --- Cumulant-based boxes per distribution ---
    # Gaussian
    c1g, c2g, _, c4g = marginal_cumulants_gaussian(mu, Sigma)
    a_g, b_g = box_from_cumulants(c1g, c2g, c4g, L=CUMU_L_GAUSS)
     
    # Student-t
    c1t, c2t, _, c4t = marginal_cumulants_student_t(mu, Sigma, NU_T)
    a_t, b_t = box_from_cumulants(c1t, c2t, c4t, L=CUMU_L_T)

    # a_g, b_g = quantile_box_gaussian(mu, Sigma, alpha=ALPHA_GAUSS)
    # a_t, b_t = quantile_box_student_t(mu, Sigma, nu=NU_T, alpha=ALPHA_T)
    # K list: 4..MAX_K (powers of two)
    K_list = [2**k for k in range(2, int(math.log2(MAX_K)) + 1)]

    rng = np.random.default_rng(RNG_SEED)

    def avg_errors_for(dist: str) -> Tuple[list, Dict[int, list]]:
        if dist == "gaussian":
            a, b = a_g, b_g
            xs = sampler_gaussian(mu, Sigma, a, b, rng)
            # xs = sampler_student(mu, Sigma, a, b, rng, nu=NU_T)
        else:
            a, b = a_t, b_t
            # xs = sampler_gaussian(mu, Sigma, a, b, rng)
            xs = sampler_student(mu, Sigma, a, b, rng, nu=NU_T)

        if xs.size == 0:
            raise RuntimeError(
                f"All X samples fell outside the COS box for {dist}; "
                f"enlarge its BOX_STD or increase N_XS."
            )
            
        def interior_mask(xs, a, b, frac=0.02):
            L = b[1:] - a[1:]
            return ((xs - a[1:]) > frac*L).all(axis=1) & ((b[1:] - xs) > frac*L).all(axis=1)
        xs = xs[interior_mask(xs, a, b)]

        avg_errs = {1: [], 2: [], 3: [], 4: []}
        for K in K_list:
            Ks = [K, K]

            # Build COS tensor with the proper box
            if dist == "gaussian":
                A = build_cos_tensor_cf(Ks, a, b, chf_gaussian, mu=mu, Sigma=Sigma)
            else:
                A = build_cos_tensor_cf(Ks, a, b, chf_student_t_besselk3, mu=mu, Sigma=Sigma, nu=NU_T)

            # TT-SVD
            cores, ranks = tt_svd2(A, eps=TT_EPS)

            # Precompute row cores for p = 0..4 (using the dist-specific box for Y)
            G0_mat = cores[0].reshape(Ks[0], -1)
            row_cores = {}
            for p in range(5):
                w = closed_form_moment_power1(a[0], b[0], Ks[0], p)
                row_cores[p] = build_row_core(G0_mat, w)
            denom_core = row_cores[0]

            # Evaluate average |TT - exact|
            errs = {1: [], 2: [], 3: [], 4: []}
            for x in xs:
                P    = feature_contraction(x, cores, a, b)
                norm = float((denom_core @ P).squeeze())
                m_tt = {}
                for p in (1,2,3,4):
                    num_p  = float((row_cores[p] @ P).squeeze())
                    m_tt[p] = num_p / norm

                if dist == "gaussian":
                    m_ex = gaussian_conditional_moments(mu, Sigma, x)
                else:
                    m_ex = student_t_conditional_moments(mu, Sigma, NU_T, x)

                for idx,p in enumerate((1,2,3,4)):
                    if np.isnan(m_ex[idx]):
                        continue
                    errs[p].append(abs(m_tt[p] - m_ex[idx]))

            for p in (1,2,3,4):
                avg_errs[p].append(np.nanmean(errs[p]) if len(errs[p]) else np.nan)

        return K_list, avg_errs

    # ---- Run both dists with their own boxes
    K_vals, avg_errs_gauss = avg_errors_for("gaussian")
    _,      avg_errs_t     = avg_errors_for("student_t")

    # ---- Single plot: Gaussian (winter) + Student-t (autumn)
    plt.figure(figsize=(8,5.5))

    cmap_g = plt.get_cmap("winter")
    cmap_t = plt.get_cmap("autumn")

    # Gaussian lines: p=1..4
    for i,p in enumerate((1,2,3,4)):
        plt.plot(K_vals, avg_errs_gauss[p], marker="o", linestyle="-",
                 label=f"Gaussian p={p} )", color=cmap_g((i+1)/5))

    # Student-t lines: p=1..4
    for i,p in enumerate((1,2,3,4)):
        
        plt.plot(K_vals, avg_errs_t[p], marker="s", linestyle="--",
                 label=f"Student-t (ν={NU_T}) p={p} )", color=cmap_t((i+1)/5))

    plt.xscale("log"); plt.yscale("log")
    plt.xlabel("K (per-dimension COS terms, log scale)")
    plt.ylabel("Avg |TT − exact| (conditional raw moment)")
    plt.title("COS–TT average conditional-moment error vs K: Gaussian vs Student-t\n(per-distribution truncation boxes)")
    plt.legend(ncol=2)
    plt.grid(True, which="both", ls=":", alpha=0.5)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_and_plot()
