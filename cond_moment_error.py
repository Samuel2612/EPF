#!/usr/bin/env python3
"""
COS–TT error study: Gaussian vs Student-t (same plot)

- Computes average |TT − exact| for conditional raw moments E[Y^p|X], p=1..4
- K per dimension in {4, 8, ..., MAX_K}
- Single log–log figure: Gaussian lines (winter), Student-t lines (autumn)
"""

import itertools, math
from typing import List, Dict

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t as t_dist, norm
from scipy.special import kve, gammaln

# --------------------------
# Config (tweak for runtime)
# --------------------------
MAX_K    = 512          # bump to 1024/2048 later if needed
TT_EPS   = 1e-11        # relax if you need more speed (e.g., 1e-10)
N_XS     = 150          # X samples kept inside box for averaging
NU_T     = 10           # Student-t df
BOX_STD  = 7.0          # box half-width = BOX_STD * per-axis std
RNG_SEED = 7

# ==========================
# Utilities
# ==========================
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
# Experiment driver
# --------------------------
def run_and_plot():
    # 2D setup (Y, X)
    mu    = np.array([1.0, 0.5])
    Sigma = np.array([[1.0, 0.4],
                      [0.4, 1.0]], dtype=float)

    std   = np.sqrt(np.diag(Sigma))
    a_box = mu - BOX_STD * std
    b_box = mu + BOX_STD * std

    # K list: 4..MAX_K (powers of two)
    K_list = [2**k for k in range(2, int(math.log2(MAX_K)) + 1)]

    rng = np.random.default_rng(RNG_SEED)

    def sample_x_gaussian(n=N_XS):
        mu_x = mu[1:]
        Sig_xx = Sigma[1:,1:]
        Lx = np.linalg.cholesky(Sig_xx)
        z = rng.normal(size=(n, len(mu_x)))
        xs = mu_x + z @ Lx.T
        mask = (xs >= a_box[1:]).all(axis=1) & (xs <= b_box[1:]).all(axis=1)
        return xs[mask]

    def sample_x_student(n=N_XS):
        mu_x = mu[1:]
        Sig_xx = Sigma[1:,1:]
        Lx = np.linalg.cholesky(Sig_xx)
        z = rng.normal(size=(n, len(mu_x)))
        chi2 = rng.chisquare(df=NU_T, size=n)
        scales = np.sqrt(NU_T / chi2)
        xs = mu_x + (z @ Lx.T) * scales[:, None]
        mask = (xs >= a_box[1:]).all(axis=1) & (xs <= b_box[1:]).all(axis=1)
        return xs[mask]

    def avg_errors_for(dist: str) -> Dict[int, list]:
        xs = sample_x_gaussian() if dist == "gaussian" else sample_x_student()
        if xs.size == 0:
            raise RuntimeError("All X samples fell outside the COS box; enlarge BOX_STD or increase N_XS.")

        avg_errs = {1: [], 2: [], 3: [], 4: []}
        for K in K_list:
            Ks = [K, K]

            # Build COS tensor
            if dist == "gaussian":
                A = build_cos_tensor_cf(Ks, a_box, b_box, chf_gaussian, mu=mu, Sigma=Sigma)
            else:
                A = build_cos_tensor_cf(Ks, a_box, b_box, chf_student_t_besselk, mu=mu, Sigma=Sigma, nu=NU_T)

            # TT-SVD
            cores, ranks = tt_svd(A, eps=TT_EPS)

            # Precompute row cores for p = 0..4
            G0_mat = cores[0].reshape(Ks[0], -1)
            row_cores = {}
            for p in range(5):
                w = closed_form_moment_power(a_box[0], b_box[0], Ks[0], p)
                row_cores[p] = build_row_core(G0_mat, w)
            denom_core = row_cores[0]

            # Evaluate average |TT - exact|
            errs = {1: [], 2: [], 3: [], 4: []}
            for x in xs:
                P    = feature_contraction(x, cores, a_box, b_box)
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

    # ---- Run both dists
    K_vals, avg_errs_gauss = avg_errors_for("gaussian")
    _,      avg_errs_t     = avg_errors_for("student_t")

    # ---- Single plot: Gaussian (winter) + Student-t (autumn)
    plt.figure(figsize=(8,5.5))

    cmap_g = plt.get_cmap("winter")
    cmap_t = plt.get_cmap("autumn")

    # Gaussian lines: p=1..4
    for i,p in enumerate((1,2,3,4)):
        plt.plot(K_vals, avg_errs_gauss[p], marker="o", linestyle="-",
                 label=f"Gaussian p={p}", color=cmap_g((i+1)/5))

    # Student-t lines: p=1..4
    for i,p in enumerate((1,2,3,4)):
        plt.plot(K_vals, avg_errs_t[p], marker="s", linestyle="--",
                 label=f"Student-t (ν={NU_T}) p={p}", color=cmap_t((i+1)/5))

    plt.xscale("log"); plt.yscale("log")
    plt.xlabel("K (per-dimension COS terms, log scale)")
    plt.ylabel("Avg |TT − exact| (conditional raw moment)")
    plt.title("COS–TT average conditional-moment error vs K: Gaussian vs Student-t")
    plt.legend(ncol=2)
    plt.grid(True, which="both", ls=":", alpha=0.5)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 2D setup (Y, X)
    mu    = np.array([1.0, 0.5])
    Sigma = np.array([[1.0, 0.4],
                      [0.4, 1.0]], dtype=float)

    std   = np.sqrt(np.diag(Sigma))
    a_box = mu - BOX_STD * std
    b_box = mu + BOX_STD * std

    # K list: 4..MAX_K (powers of two)
    K_list = [2**k for k in range(2, int(math.log2(MAX_K)) + 1)]

    rng = np.random.default_rng(RNG_SEED)

    def sample_x_gaussian(n=N_XS):
        mu_x = mu[1:]
        Sig_xx = Sigma[1:,1:]
        Lx = np.linalg.cholesky(Sig_xx)
        z = rng.normal(size=(n, len(mu_x)))
        xs = mu_x + z @ Lx.T
        mask = (xs >= a_box[1:]).all(axis=1) & (xs <= b_box[1:]).all(axis=1)
        return xs[mask]

    def sample_x_student(n=N_XS):
        mu_x = mu[1:]
        Sig_xx = Sigma[1:,1:]
        Lx = np.linalg.cholesky(Sig_xx)
        z = rng.normal(size=(n, len(mu_x)))
        chi2 = rng.chisquare(df=NU_T, size=n)
        scales = np.sqrt(NU_T / chi2)
        xs = mu_x + (z @ Lx.T) * scales[:, None]
        mask = (xs >= a_box[1:]).all(axis=1) & (xs <= b_box[1:]).all(axis=1)
        return xs[mask]

    def avg_errors_for(dist: str) -> Dict[int, list]:
        xs = sample_x_gaussian() if dist == "gaussian" else sample_x_student()
        if xs.size == 0:
            raise RuntimeError("All X samples fell outside the COS box; enlarge BOX_STD or increase N_XS.")

        avg_errs = {1: [], 2: [], 3: [], 4: []}
        for K in K_list:
            Ks = [K, K]

            # Build COS tensor
            if dist == "gaussian":
                A = build_cos_tensor_cf(Ks, a_box, b_box, chf_gaussian, mu=mu, Sigma=Sigma)
            else:
                A = build_cos_tensor_cf(Ks, a_box, b_box, chf_student_t_besselk, mu=mu, Sigma=Sigma, nu=NU_T)

            # TT-SVD
            cores, ranks = tt_svd(A, eps=TT_EPS)

            # Precompute row cores for p = 0..4
            G0_mat = cores[0].reshape(Ks[0], -1)
            row_cores = {}
            for p in range(5):
                w = closed_form_moment_power(a_box[0], b_box[0], Ks[0], p)
                row_cores[p] = build_row_core(G0_mat, w)
            denom_core = row_cores[0]

            # Evaluate average |TT - exact|
            errs = {1: [], 2: [], 3: [], 4: []}
            for x in xs:
                P    = feature_contraction(x, cores, a_box, b_box)
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

    # ---- Run both dists
    K_vals, avg_errs_gauss = avg_errors_for("gaussian")
    _,      avg_errs_t     = avg_errors_for("student_t")

    # ---- Single plot: Gaussian (winter) + Student-t (autumn)
    plt.figure(figsize=(8,5.5))

    cmap_g = plt.get_cmap("winter")
    cmap_t = plt.get_cmap("autumn")

    # Gaussian lines: p=1..4
    for i,p in enumerate((1,2,3,4)):
        plt.plot(K_vals, avg_errs_gauss[p], marker="o", linestyle="-",
                 label=f"Gaussian p={p}", color=cmap_g((i+1)/5))

    # Student-t lines: p=1..4
    for i,p in enumerate((1,2,3,4)):
        plt.plot(K_vals, avg_errs_t[p], marker="s", linestyle="--",
                 label=f"Student-t (ν={NU_T}) p={p}", color=cmap_t((i+1)/5))

    plt.xscale("log"); plt.yscale("log")
    plt.xlabel("K (per-dimension COS terms, log scale)")
    plt.ylabel("Avg |TT − exact| (conditional raw moment)")
    plt.title("COS–TT average conditional-moment error vs K: Gaussian vs Student-t")
    plt.legend(ncol=2)
    plt.grid(True, which="both", ls=":", alpha=0.5)
    plt.tight_layout()
    plt.show()

