#!/usr/bin/env python3
"""COS–TT demo: conditional moments E[Y^p|X] (p=1..4) for a 4-D Gaussian or Student-t.

* pure NumPy core + SciPy for quantiles/Bessel-K
* TT compression via a plain TT-SVD
* no external TT libraries or cross approximation required
"""
import itertools, math, time
from typing import List

import numpy as np
from scipy.stats import t as t_dist, norm
from scipy.special import kve, gammaln, gamma
from numpy.linalg import cholesky



def index_grid(Ks):
    grid = np.indices(Ks, dtype=np.float64)
    return np.moveaxis(grid, 0, -1)

def quantile_box_axis_aligned(mu: np.ndarray, Sigma: np.ndarray, *,
                              dist: str, nu: float | None = None, eps: float = 1e-8):
    """
    Axis-aligned truncation box via per-axis marginal quantiles (keeps axis semantics).
    Uses scatter scales s_j = sqrt(Sigma_jj). Total outside-box prob <= eps (union bound).
    """
    d = len(mu)
    alpha = eps / (2.0 * d)
    s = np.sqrt(np.diag(Sigma))  # scatter-based per-axis scales

    if dist.lower() == "student_t":
        if nu is None:
            raise ValueError("nu must be provided for Student-t.")
        q = float(t_dist.ppf(1.0 - alpha, df=nu))
    elif dist.lower() == "gaussian":
        q = float(norm.ppf(1.0 - alpha))
    else:
        raise ValueError("dist must be 'gaussian' or 'student_t'.")

    a = mu - q * s
    b = mu + q * s
    return a, b


def cumulant_box_axis_aligned(mu, Sigma, *, dist: str, nu: float | None = None, Lmult: float = 10.0):
    """
    Per-axis box using the 'first 4 cumulants' rule from the original COS paper:
        w_j = Lmult * sqrt(c2_j + sqrt(max(c4_j, 0))) ;  a_j = mu_j - w_j ; b_j = mu_j + w_j
    Only valid if the necessary cumulants exist (Student-t requires nu > 4).
    Sigma is a scatter/covariance-like scale; for Student-t it's the scatter.
    """
    mu = np.asarray(mu); d = len(mu)
    s  = np.sqrt(np.diag(Sigma))  # per-axis scale

    if dist.lower() == "gaussian":
        # c2 = σ^2, c4 = 0
        c2 = s**2
        c4 = np.zeros_like(c2)
    elif dist.lower() == "student_t":
        if nu is None or nu <= 4:
            raise ValueError("Cumulant box for Student-t needs nu > 4 (variance & κ4 finite).")
        # Univariate Student-t(ν, scale=s):
        # Var = (ν/(ν-2)) s^2,  fourth cumulant κ4 = μ4 − 3Var^2 = 6 ν^2 s^4 / ((ν-2)^2 (ν-4))
        c2 = (nu/(nu-2.0)) * s**2
        c4 = (6.0 * nu**2 / ((nu-2.0)**2 * (nu-4.0))) * s**4
    else:
        raise ValueError("dist must be 'gaussian' or 'student_t'.")

    w = Lmult * np.sqrt(c2 + np.sqrt(np.maximum(c4, 0.0)))
    a = mu - w
    b = mu + w
    return a, b


def cap_box_for_frequency(a, b, Sigma, nu_or_1, Ks, zstar=20.0):
    """
    Shrink half-width Lj if needed so the largest COS frequency π(Kj-1)/Lj
    reaches scaled argument z_max,j >= zstar in the CF:
        Gaussian: z = sqrt(Σ_jj) * t_j        (pass nu_or_1=1)
        Student-t: z = sqrt(nu) * sqrt(Σ_jj) * t_j  (pass nu_or_1=nu)
    """
    mu = (a + b) * 0.5
    L  = (b - a).copy()
    s  = np.sqrt(np.diag(Sigma))
    for j, Kj in enumerate(Ks):
        if Kj <= 1: 
            continue
        Lcap = (np.sqrt(nu_or_1) * s[j] * math.pi * (Kj - 1)) / zstar
        if L[j] > Lcap:
            L[j] = Lcap
    return mu - 0.5*L, mu + 0.5*L


# ===========================
# Characteristic functions
# ===========================
def chf_gaussian(t, mu, Sigma):
    tf   = t.reshape(-1, t.shape[-1])                    # (N,d)
    phase = 1j * tf @ mu                                 # (N,)
    quad  = np.einsum("ij,jk,ik->i", tf, Sigma, tf)      # (N,)
    return np.exp(phase - 0.5 * quad).reshape(t.shape[:-1])

def chf_student_t_besselk(t, mu, Sigma, nu: float, small_z: float = 1e-15):
    """
    Multivariate Student-t CF using the Bessel-K closed form (stable with kve).

    phi(t) = exp(i t^T mu) * 2^{1 - nu/2} / Gamma(nu/2)
             * (sqrt(nu)*r)^{nu/2} * K_{nu/2}(sqrt(nu)*r),
    r = sqrt(t^T Sigma t). For nu<=2, Sigma is a scatter matrix (variance may be infinite).
    """
    tf_dim = t.shape[-1]
    tf = t.reshape(-1, tf_dim)
    r2 = np.einsum("ij,jk,ik->i", tf, Sigma, tf)  # (N,)
    r = np.sqrt(np.maximum(r2, 0.0))
    z = np.sqrt(nu) * r
    v = 0.5 * nu

    # kve returns e^z * K_v(z); so K_v(z) = e^{-z} * kve(v,z)
    logC = (1.0 - v) * np.log(2.0) - gammaln(v) + v * np.log(np.maximum(z, 1.0))
    core = np.exp(logC - z) * kve(v, z)
    core = np.where(z < small_z, 1.0, core)  # limit as z->0 -> 1

    phase = np.exp(1j * (tf @ mu))
    out = (phase * core).reshape(t.shape[:-1])
    return out




def chf_student_t_besselk2(t, mu, Sigma, nu, small_z=1e-15):
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

# ===========================
# COS tensor builder (CF-agnostic)
# ===========================
def build_cos_tensor_cf(Ks, a, b, cf_func, **cf_kwargs):
    """
    CF-agnostic COS coefficient tensor:
    A[k] = (2/prod L) * sum_{s in {1,-1}^{d-1}} Re{ exp(-i t·a) * phi(t) }, with t_j = pi * s_j * k_j / L_j
    """
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


# ===========================
# TT-SVD (unchanged)
# ===========================
def tt_svd(tensor, eps) :
    """Compute a tensor-train (TT) decomposition via SVD."""
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
        if mask.any():
            r = int(np.argmax(mask)) + 1
        else:
            r = len(S)

        r = max(1, r)

        cores.append(U[:, :r].reshape(ranks[-1], dims[k], r))
        ranks.append(r)
        unfold = (S[:r, None] * Vh[:r])

    cores.append(unfold.reshape(ranks[-1], dims[-1], 1))
    ranks.append(1)

    return cores, ranks


def cos_basis(x, a, b, N):
    k   = np.arange(N)
    vec = np.cos(k * math.pi * (x - a) / (b - a))
    vec[0] *= 0.5
    return vec

def cos_moment_const(a, b, N):
    J = np.zeros(N)
    J[0] = 0.5 * (b - a)
    return J

def cos_moment_mean(a, b, N):
    J = np.zeros(N)
    J[0] = 0.25 * (a + b) * (b - a)
    if N > 1:
        k     = np.arange(1, N)
        J[1:] = (b - a)**2 * ((-1.0)**k - 1.0) / (k * math.pi)**2
    return J

def closed_form_moment_power(a, b, N, p):
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



def gaussian_conditional_moments(mu, Sigma, x_feat):
    """First 4 raw moments of Y|X=x for a Gaussian partition Y (1d) and X (q-d)."""
    Sigma_yx     = Sigma[0,1:]
    Sigma_xx     = Sigma[1:,1:]
    Sigma_xx_inv = np.linalg.inv(Sigma_xx)
    mu_y, mu_x   = mu[0], mu[1:]
    mu_c   = float(mu_y + Sigma_yx @ Sigma_xx_inv @ (x_feat - mu_x))
    sigma2_base = float(Sigma[0,0] - Sigma_yx @ Sigma_xx_inv @ Sigma_yx)  # Schur complement
    # For Gaussian, conditional variance doesn't depend on x
    m1 = mu_c
    m2 = sigma2_base + mu_c**2
    m3 = mu_c**3 + 3*mu_c*sigma2_base
    m4 = mu_c**4 + 6*mu_c**2*sigma2_base + 3*sigma2_base**2
    return np.array([m1, m2, m3, m4], dtype=float)

def student_t_conditional_moments(mu, Sigma, nu, x_feat):
    """
    First 4 raw moments of Y|X=x for a multivariate Student-t with df nu.
    Partition: Y is the first coordinate, X are the remaining q=d-1.
    Y|X=x ~ t_{nu'}(mu_c, s^2), with nu' = nu + q,
      mu_c = mu_y + Σ_yx Σ_xx^{-1}(x-mu_x),
      s^2  = ((nu + δ_x)/(nu + q)) * (Σ_yy - Σ_yx Σ_xx^{-1} Σ_xy),
      δ_x  = (x-mu_x)^T Σ_xx^{-1} (x-mu_x).
    Moments exist if nu' > k for order k.
    """
    q = len(mu) - 1
    nu_p = nu + q

    Sigma_yx     = Sigma[0,1:]
    Sigma_xx     = Sigma[1:,1:]
    Sigma_xx_inv = np.linalg.inv(Sigma_xx)
    mu_y, mu_x   = mu[0], mu[1:]
    dx = x_feat - mu_x
    mu_c = float(mu_y + Sigma_yx @ Sigma_xx_inv @ dx)
    sigma2_base = float(Sigma[0,0] - Sigma_yx @ Sigma_xx_inv @ Sigma_yx)  # Schur complement (scalar)

    delta_x = float(dx @ Sigma_xx_inv @ dx)
    s2 = ((nu + delta_x) / (nu + q)) * sigma2_base  # conditional scale^2
    s  = math.sqrt(max(s2, 0.0))

    # Standard t moments (mean 0): E[T]=0 (nu'>1), E[T^2]=nu'/(nu'-2), E[T^4]=3*nu'^2/((nu'-2)(nu'-4))
    def exists(k): return nu_p > k

    # m1
    m1 = mu_c if exists(1) else np.nan

    # m2
    if exists(2):
        ET2 = nu_p / (nu_p - 2.0)
        m2 = mu_c**2 + s2 * ET2
    else:
        m2 = np.nan

    # m3
    if exists(3):
        ET2 = nu_p / (nu_p - 2.0)
        m3 = mu_c**3 + 3*mu_c*s2*ET2  # E[T^3]=0
    else:
        m3 = np.nan

    # m4
    if exists(4):
        ET2 = nu_p / (nu_p - 2.0)
        ET4 = (3.0 * nu_p**2) / ((nu_p - 2.0) * (nu_p - 4.0))
        m4 = mu_c**4 + 6*mu_c**2*s2*ET2 + (s2**2) * ET4
    else:
        m4 = np.nan

    return np.array([m1, m2, m3, m4], dtype=float)


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




if __name__ == "__main__":

    # ---- choose distribution ----
    DIST = "student_t"        # "gaussian" or "student_t"
    nu   = 5             # df for Student-t (ignored if Gaussian)
    BOX_EPS = 1e-7        # total outside-box probability (union bound across axes)

    Ks    = [256]*2
    mu    = np.array([0.0, 0.0])
    Sigma = np.array([[0.5, 0.2],
                      [0.2, 0.5]], dtype=float)
    
    
    # Ks    = [100]*4
    # mu    = np.array([1.0, 0.5, -1.1, 1.3])
    # Sigma = np.array([[1.0, 0.4, 0.3, 0.1],
    #                   [0.4, 1.0, 0.6, 0.5],
    #                   [0.3, 0.6, 1.0, 0.5],
    #                   [0.1, 0.5, 0.5, 1.0]])
    std   = np.sqrt(np.diag(Sigma))
    

    
    std   = np.sqrt(np.diag(Sigma))
    a     = mu - 30.0*std
    b     = mu + 30.0*std

    print("Building COS tensor ...")
    if DIST == "gaussian":
        A = build_cos_tensor_cf(Ks, a, b, chf_gaussian, mu=mu, Sigma=Sigma)
    else:
        A = build_cos_tensor_cf(Ks, a, b, chf_student_t_besselk3, mu=mu, Sigma=Sigma, nu=nu)
    print("Tensor shape:", A.shape)

    print("Compressing with TT-SVD ...")
    t0 = time.time()
    cores, ranks = tt_svd(A,  eps=1e-10 )
    print(f"TT-SVD done in {time.time()-t0:.2f}s  - ranks {ranks}")


    G0_mat = cores[0].reshape(Ks[0], -1)
    row_cores = {}
    for p in range(5):
        w = closed_form_moment_power(a[0], b[0], Ks[0], p)
        row_cores[p] = build_row_core(G0_mat, w)
    tilde_G0 = row_cores[0]

    rng = np.random.default_rng(42)
    qdim = len(mu) - 1
    xs = None

    if DIST == "gaussian":
        # sample X from Gaussian marginal
        mu_x = mu[1:]
        Sig_xx = Sigma[1:,1:]
        Lx = np.linalg.cholesky(Sig_xx)
        z = rng.normal(size=(100, qdim))
        xs = mu_x + z @ Lx.T
        # when drawing xs for the check:
        mask = (xs >= a[1:]).all(axis=1) & (xs <= b[1:]).all(axis=1)
        xs = xs[mask]
        if xs.size == 0:
            print("Warning: all sampled X fell outside the COS box; increase BOX_EPS or enlarge Ks.")


        errs  = {p: [] for p in (1,2,3,4)}
        for x in xs:
            P    = feature_contraction(x, cores, a, b)
            norm = float((tilde_G0 @ P).squeeze())
            m_tt = {}
            for p in (1,2,3,4):
                num_p  = float((row_cores[p] @ P).squeeze())
                m_tt[p] = num_p / norm
            m_ex = gaussian_conditional_moments(mu, Sigma, x)
            for idx,p in enumerate((1,2,3,4)):
                errs[p].append(abs(m_tt[p] - m_ex[idx]))

        print("\nConditional raw-moment errors (Gaussian, TT vs. exact):")
        for p in (1,2,3,4):
            arr = np.array(errs[p])
            print(f" p={p}  max error = {arr.max():.3e}, avg error = {arr.mean():.3e}")

    else:
        # sample X from the Student-t marginal (q-dim) via chi-square mixture
        mu_x = mu[1:]
        Sig_xx = Sigma[1:,1:]
        Lx = np.linalg.cholesky(Sig_xx)
        z = rng.normal(size=(100, qdim))
        chi2 = rng.chisquare(df=nu, size=100)  # S ~ χ²_ν
        scales = np.sqrt(nu / chi2)            # 1/sqrt(S/ν)
        xs = mu_x + (z @ Lx.T) * scales[:, None]
        # when drawing xs for the check:
        mask = (xs >= a[1:]).all(axis=1) & (xs <= b[1:]).all(axis=1)
        xs = xs[mask]
        if xs.size == 0:
            print("Warning: all sampled X fell outside the COS box; increase BOX_EPS or enlarge Ks.")


        errs  = {p: [] for p in (1,2,3,4)}
        for x in xs:
            P    = feature_contraction(x, cores, a, b)
            norm = float((tilde_G0 @ P).squeeze())
            m_tt = {}
            for p in (1,2,3,4):
                num_p  = float((row_cores[p] @ P).squeeze())
                m_tt[p] = num_p / norm
            m_ex = student_t_conditional_moments(mu, Sigma, nu, x)
            for idx,p in enumerate((1,2,3,4)):
                # if the analytic moment doesn't exist, skip
                if np.isnan(m_ex[idx]):
                    continue
                errs[p].append(abs(m_tt[p] - m_ex[idx]))

        print("\nConditional raw-moment errors (Student-t, TT vs. exact):")
        for p in (1,2,3,4):
            arr = np.array(errs[p]) if len(errs[p]) else np.array([np.nan])
            print(f" p={p}  max error = {np.nanmax(arr):.3e}, avg error = {np.nanmean(arr):.3e}")
