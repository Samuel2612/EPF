# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 09:45:28 2025

@author: samue
"""


import numpy as np
from itertools import product
from time import perf_counter


def empirical_cf(T, X, batch=50000):
    """
    Empirical characteristic function  φ(T_j) = mean(exp(1j * T_j · X_i)), over i.
    Parameters
    ----------
    T : (m_t, n) array of t-vectors
    X : (m_x, n) sample matrix
    batch : process T in chunks to save RAM
    Returns
    -------
    φ : (m_t,) complex128
    """
    m_t = T.shape[0]
    out = np.empty(m_t, dtype=np.complex128)
    start = 0
    while start < m_t:
        stop = min(start + batch, m_t)
        # (stop-start, n) @ (n, m_x) -> (stop-start, m_x)
        phases = T[start:stop] @ X.T
        out[start:stop] = np.exp(1j * phases).mean(axis=1)
        start = stop
    return out


def build_tensor(K, a, b, X):
    """
    Compute full tensor A for given K=(K1,...,Kn).
    Returns ndarray with shape K.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = a.size

    # --- generate all k-combinations (cartesian product) ---
    grids = np.meshgrid(*[np.arange(1, Ki + 1) for Ki in K], indexing='ij')
    k_stack = np.stack([g.reshape(-1) for g in grids], axis=1)  # (num_k, n)
    num_k = k_stack.shape[0]

    # --- all sign vectors s ∈ {±1}^n ---
    rest   = np.array(list(itertools.product([1, -1], repeat=n-1)), float)
    s_mat = np.array(list(product(*([(-1, 1)] * n))), dtype=np.int8)  # (2^n, n)
    num_s = s_mat.shape[0]

    # --- broadcast to get γ = π * (s*k)/(b-a) ---
    denom = (b - a)
    gamma = (np.pi * s_mat[:, None, :] * k_stack[None, :, :]) / denom  # (num_s, num_k, n)

    # --- empirical CF evaluated at all γ vectors (flatten, then reshape) ---
    T = gamma.reshape(num_s * num_k, n)
    phi = empirical_cf(T, X).reshape(num_s, num_k)

    # --- phase shift exp(-i π γ·a) ---
    shift = np.exp(-1j * np.pi * (gamma * a).sum(axis=2))  # (num_s, num_k)

    contrib = np.real(shift * phi)      # (num_s, num_k)
    summed = contrib.sum(axis=0)        # (num_k,)

    pref = 2.0 * np.prod(1.0 / denom)
    A_flat = pref * summed

    return A_flat.reshape(K)


def demo():
    # Problem setup -----------------------------------------------------------
    n = 4
    K = (32, 32, 32, 32)      # tensor size; change as you like
    a = np.array([-2., -2., -2., -2.])
    b = np.array([ 2.,  2.,  2.,  2.])

    # Generate samples from a 4D Gaussian for the ECF ------------------------
    rng = np.random.default_rng(0)
    mean = np.zeros(n)
    cov = np.array([[1.0, 0.3, 0.2, 0.1],
                    [0.3, 1.2, 0.0, 0.2],
                    [0.2, 0.0, 0.8, 0.3],
                    [0.1, 0.2, 0.3, 1.5]])
    m_samples = 10_000
    X = rng.multivariate_normal(mean, cov, size=m_samples)

    # Compute ---------------------------------------------------------------
    t0 = perf_counter()
    A = build_tensor(K, a, b, X)
    t1 = perf_counter()

    print(f"A tensor shape: {A.shape}, dtype: {A.dtype}")
    print(f"Elapsed: {t1 - t0:.3f} s")
    # Example: inspect a single entry
    print("A[0,0,0,0] =", A[0, 0, 0, 0])


if __name__ == "__main__":
    demo()
