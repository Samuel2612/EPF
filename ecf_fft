# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 14:42:49 2025

@author: samue
"""

"""
ecf_nufft.py
============
Empirical characteristic function on a Cartesian frequency grid
using the adjoint Non-Uniform Fast Fourier Transform (NUFFT).

Requires
--------
pip install pynfft numpy

Usage
-----
python ecf_nufft.py  # runs a quick accuracy/benchmark test
"""
import numpy as np
from time import perf_counter
from pynfft import NFFT                     

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _scale_nodes_to_nufft_cube(X, L):
    """
    Convert real-space samples X to nodes x in (-1/2, 1/2] needed by NFFT.

    We use           x = X / (2 L)
    so that          2π k·x = (π/L) k·X       ( ≡ exponent used before ).
    """
    return X / (2.0 * L)                     # shape (N, d)


def ecf_grid_nufft(X, a, b, K, dtype=np.complex64):
    """
    Empirical CF on a rectangular grid via an adjoint NUFFT.

    Parameters
    ----------
    X   : (N,d) samples
    a,b : vectors that bound X  (only their difference L = b-a is used)
    K   : array_like of grid sizes  (same length as X.shape[1])
    """
    X   = np.asarray(X, dtype=float)
    N,d = X.shape
    K   = np.asarray(K, dtype=int)
    L   = (b - a)                          # side lengths

    
    x_nufft = _scale_nodes_to_nufft_cube(X, L)     # (N,d)


    plan = NFFT(K, N)                               # geometry   :contentReference[oaicite:1]{index=1}
    plan.x = -x_nufft                               # minus sign ⇒ +i2π… phase
    plan.precompute()

    # 3) Supply strengths (= 1/N for an empirical expectation) & run adjoint
    plan.f = np.ones(N, dtype=dtype) / N
    plan.adjoint()                                  # fills plan.f_hat

    # 4) Reshape flat Fourier coeffs to the desired K-tensor
    phi = plan.f_hat.reshape(*K).astype(dtype, copy=False)
    return phi


def true_gaussian_cf_grid(a, b, K, mu, Sigma, dtype=np.complex64):
    """Analytic CF of a d-variate normal on the same grid."""
    d   = len(K)
    L   = (b - a)
    alpha = np.pi / L
    grids = np.meshgrid(*[alpha[j]*np.arange(K[j]) for j in range(d)],
                        indexing='ij')
    t    = np.stack(grids, axis=-1)                 # (..., d)
    quad = np.einsum('...i,ij,...j->...', t, Sigma, t, optimize=True)
    lin  = t @ mu
    return np.exp(1j*lin - 0.5*quad).astype(dtype)


# ---------------------------------------------------------------------------
# quick test / benchmark
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # problem size -----------------------------------------------------------
    N     = 1_000          # samples
    d     = 4
    K     = np.array([64]*4)

    # target N(μ, Σ) ---------------------------------------------------------
    mu    = np.zeros(d)
    Sigma = np.array([[1.0, 0.4, 0.3, 0.1],
                      [0.4, 1.0, 0.6, 0.5],
                      [0.3, 0.6, 1.0, 0.5],
                      [0.1, 0.5, 0.5, 1.0]])
    std   = np.sqrt(np.diag(Sigma))
    a     = mu - 7*std
    b     = mu + 7*std

    # draw samples
    X = rng.multivariate_normal(mean=mu, cov=Sigma, size=N)

    # empirical CF via NUFFT --------------------------------------------------
    t0 = perf_counter()
    phi_hat = ecf_grid_nufft(X, a, b, K, dtype=np.complex64)
    t1 = perf_counter()

    # analytic CF -------------------------------------------------------------
    phi_true = true_gaussian_cf_grid(a, b, K, mu, Sigma, dtype=phi_hat.dtype)

    # errors + timing
    rmse   = np.sqrt(np.mean(np.abs(phi_hat - phi_true)**2))
    print(f"grid shape  : {phi_hat.shape}")
    print(f"NUFFT time  : {t1 - t0:6.3f}  s")
    print(f"RMSE        : {rmse:9.2e}")