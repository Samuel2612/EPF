#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import math
import numpy as np

# ---- special functions: prefer scipy, else mpmath fallback ----
try:
    from scipy.special import kv as besselk, gamma
    _USE_SCIPY = True
except Exception:
    import mpmath as mp
    def besselk(nu, x):
        return float(mp.besselk(nu, x))
    def gamma(x):
        return float(mp.gamma(x))
    _USE_SCIPY = False

import tt
from tt.cross.rect_cross import cross as rect_cross

# ==============================================================
# 1) Multivariate Student-t characteristic function (non-centered)
# ==============================================================

def student_t_chf(u, nu, mu, Sigma_chol):
    """
    φ(u) = E[exp(i u^T X)] for X ~ t_ν(μ, Σ), non-centered.
    Closed form (radial in r = ||Σ^{1/2} u||):
        φ(u) = exp(i μ^T u) * [ 2^{1-ν/2} (sqrt(ν) r)^{ν/2} K_{ν/2}(sqrt(ν) r) / Γ(ν/2) ]  ,
    where K_· is the modified Bessel function of the second kind and r = ||Σ^{1/2} u||.
    For r → 0, the bracket term → 1 (continuous limit).
    """
    # r = ||Σ^{1/2} u|| computed via precomputed Cholesky
    Su = Sigma_chol @ u
    r = float(np.linalg.norm(Su))
    phase = complex(np.cos(np.dot(mu, u)), np.sin(np.dot(mu, u)))  # exp(i μ^T u)

    if r < 1e-14:
        return phase  # limit is 1 inside brackets

    z = math.sqrt(nu) * r
    nu_over_2 = 0.5 * nu
    # bracket term
    K = besselk(nu_over_2, z)
    # numerical guard: K can underflow for large z; the combination below is stable in double for moderate z
    coeff = 2.0**(1.0 - nu_over_2) / gamma(nu_over_2)
    bracket = coeff * (z**(nu_over_2)) * K
    return phase * bracket

# ==============================================================
# 2) Tensor grid & wrappers
# ==============================================================

def tensor_grid_1d(a, b, n):
    return np.linspace(a, b, n, dtype=float)

def make_axes(d, n_per_dim, box):
    lo, hi = box
    return [tensor_grid_1d(lo, hi, n_per_dim) for _ in range(d)]

class GridCallableRect:
    """
    For rect_cross.cross: expects f(multi_index) -> scalar
    """
    def __init__(self, axes, f_phys):
        self.axes = axes
        self.f_phys = f_phys
        self.d = len(axes)

    def __call__(self, multi_idx):
        u = np.array([self.axes[k][multi_idx[k]] for k in range(self.d)], dtype=float)
        return self.f_phys(u)

def tt_element_at(tt_tensor, idx):
    """
    Evaluate a TT-tensor at a multi-index by contracting its cores:
    Works for tt.vector-like tensors with mode sizes given during construction.
    """
    cores = tt.vector.to_list(tt_tensor)  # list of (rk-1, nk, rk)
    v = cores[0][:, idx[0], :]            # shape (1, r1)
    for k in range(1, len(cores)):
        v = v @ cores[k][:, idx[k], :]    # (1, rk) @ (rk,1,rk+1)->(1,rk+1)
    return float(v[0, 0])

def spotcheck_error(axes, f_phys, tt_tensor, n_checks=500, rng=123):
    rs = np.random.default_rng(rng)
    d = len(axes)
    errs = []
    for _ in range(n_checks):
        idx = tuple(rs.integers(0, len(ax)) for ax in axes)
        u = np.array([axes[k][idx[k]] for k in range(d)], dtype=float)
        true = f_phys(u)
        approx = tt_element_at(tt_tensor, idx)
        denom = max(1e-12, abs(true))
        errs.append(abs(approx - true) / denom)
    errs = np.array(errs)
    return float(np.percentile(errs, 95)), float(errs.max())

# ==============================================================
# 3) Benchmark harness for 3 algorithms
# ==============================================================

def run_all(nu=5.0,
            d=5,
            n_per_dim=41,
            box=(-10.0, 10.0),
            eps=1e-6,
            nswp=15,
            kickrank=10,
            rmax=4000,
            verb=1):
    # fixed nonzero location (non-centered)
    mu = np.array([0.3, -0.5, 0.8, 0.2, -0.1], dtype=float)[:d]

    # SPD scatter matrix Σ with mild correlations; build & Cholesky
    A = np.eye(d)
    for i in range(d):
        for j in range(i+1, d):
            A[i, j] = A[j, i] = 0.2 * (0.8 ** abs(i - j))
    Sigma = A @ A.T  # ensure SPD & stronger diagonals
    Sigma_chol = np.linalg.cholesky(Sigma)

    # physical function on R^d
    def f_phys(u):
        return student_t_chf(u, nu=nu, mu=mu, Sigma_chol=Sigma_chol)

    axes = make_axes(d, n_per_dim, box)
    sizes = [len(ax) for ax in axes]

    # ------- A) rect_cross.cross (index-based callback) -------
    f_rect = GridCallableRect(axes, f_phys)
    t0 = time.time()
    tt_rect = rect_cross(
        f_rect,
        sizes,
        nswp=nswp,
        eps=eps,
        eps_abs=0.0,
        kickrank=2,
        rf=2.0,
        verbose=(verb >= 2),
        stop_fun=None,
        approx_fun=None
    )
    t_rect = time.time() - t0
    ranks_rect = tt.rectify(tt_rect).r

    e95_rect, emax_rect = spotcheck_error(axes, f_phys, tt_rect, n_checks=600)

    # ------- B) multifuncrs (coord-based callback) -------
    # For multifuncrs/2, X is the list of 1D coordinate arrays (axes);
    # fun signature must be f(x1, ..., xd) -> scalar
    def mf_wrapper(*coords):
        return f_phys(np.array(coords, dtype=float))

    t0 = time.time()
    tt_mf = tt.multifuncrs(
        axes,
        mf_wrapper,
        eps=eps,
        nswp=nswp,
        kickrank=kickrank,
        y0=None,
        rmax=rmax,
        kicktype='amr-two',
        pcatype='svd',
        trunctype='fro',
        d2=1,
        do_qr=False,
        verb=verb
    )
    t_mf = time.time() - t0
    ranks_mf = tt.rectify(tt_mf).r
    e95_mf, emax_mf = spotcheck_error(axes, f_phys, tt_mf, n_checks=600)

    # ------- C) multifuncrs2 (coord-based callback) -------
    t0 = time.time()
    tt_mf2 = tt.multifuncrs2(
        axes,
        mf_wrapper,
        eps=eps,
        nswp=nswp,
        rmax=rmax,
        verb=verb,
        kickrank=kickrank,
        kickrank2=kickrank // 2,
        d2=1,
        eps_exit=None,
        y0=None,
        do_qr=False,
        restart_it=1
    )
    t_mf2 = time.time() - t0
    ranks_mf2 = tt.rectify(tt_mf2).r
    e95_mf2, emax_mf2 = spotcheck_error(axes, f_phys, tt_mf2, n_checks=600)

    return {
        "grid": {"d": d, "n_per_dim": n_per_dim, "box": box},
        "t_params": {"nu": nu, "mu": mu, "Sigma": Sigma},
        "settings": {"eps": eps, "nswp": nswp, "kickrank": kickrank, "rmax": rmax},
        "rect_cross": {"time_s": t_rect, "ranks": ranks_rect.tolist(), "err95": e95_rect, "err_max": emax_rect, "tt": tt_rect},
        "multifuncrs": {"time_s": t_mf, "ranks": ranks_mf.tolist(), "err95": e95_mf, "err_max": emax_mf, "tt": tt_mf},
        "multifuncrs2": {"time_s": t_mf2, "ranks": ranks_mf2.tolist(), "err95": e95_mf2, "err_max": emax_mf2, "tt": tt_mf2},
    }

def _brief(stats):
    return f"time={stats['time_s']:.2f}s | maxrank={max(stats['ranks'])} | err95={stats['err95']:.2e} | errmax={stats['err_max']:.2e}"

# ==============================================================
# 4) Main: run benchmark
# ==============================================================

if __name__ == "__main__":
    print("Backend for special funcs:", "SciPy" if _USE_SCIPY else "mpmath(fallback)")
    out = run_all(
        nu=5.0,          # degrees of freedom
        d=5,             # 5D as requested
        n_per_dim=41,    # grid points per dimension
        box=(-10.0, 10.0),
        eps=1e-6,
        nswp=15,
        kickrank=10,
        rmax=4000,
        verb=1
    )

    print("\n=== Grid & Settings ===")
    print(out["grid"])
    print(out["settings"])
    print("\n=== Results ===")
    print("[rect_cross]   ", _brief(out["rect_cross"]))
    print("[multifuncrs]  ", _brief(out["multifuncrs"]))
    print("[multifuncrs2] ", _brief(out["multifuncrs2"]))