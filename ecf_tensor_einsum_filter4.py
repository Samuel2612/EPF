# -*- coding: utf-8 -*-
"""ecf_cos_fastkde.py – **complete COS‑method pipeline**

Add‑on in this revision
=======================
* **`build_cos_coefficients`** – constructs the full *tensor* of cosine
  coefficients  A_{k₁…k_d} *after* the self‑consistent fastKDE filter.
* Works for arbitrary dimension *d*.  Internally it evaluates the
  filtered characteristic function for **all 2^{d‑1} sign combinations**
  required by Fang–Oosterlee’s multi‑COS formula.
* The original public helpers remain unchanged – you can still fetch raw
  or filtered ECF grids separately if you want.

The self‑test at the bottom now compares RMSE of the COS coefficients
vs. ground‑truth analytic Gaussian coefficients *with* and *without*
filtering.
"""
from __future__ import annotations

from collections import deque
from itertools import product
from time import perf_counter
from typing import Iterable, Sequence, Tuple

import numpy as np
import opt_einsum as oe

################################################################################
#  1. ECF on grid (einsum)  — unchanged                                        #
################################################################################


def _pin_path(subs, K, dtype, warmup_N=1000):
    dummy = [np.empty((warmup_N, Kj), dtype=dtype) for Kj in K]
    _EINSUM_PATH, info = np.einsum_path(subs, *dummy, optimize='greedy')
    return _EINSUM_PATH, info

def ecf_grid_einsum(X, a, b, s, K, dtype=np.complex64):
    X = np.asarray(X)
    N, d = X.shape
    a, b, s, K = map(np.asarray, (a, b, s, K))

    Es = []
    for j in range(d):
        alpha = np.pi * s[j] / (b[j] - a[j])
        kj = np.arange(K[j])
        Es.append(np.exp(1j * alpha * X[:, j, None] * kj[None, :]).astype(dtype))

    letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    in_subs  = [f"n{letters[j]}" for j in range(len(K))]
    out_subs = "".join(letters[:len(K)])
    subs     = ",".join(in_subs) + "->" + out_subs

    path, info = _pin_path(subs, K, dtype)
    # print(info)
    return np.einsum(subs, *Es, optimize=path) / N



def _flood_fill(mask: np.ndarray, conn: int = 1) -> np.ndarray:
    ndim = mask.ndim
    if not (1 <= conn <= ndim):
        raise ValueError("conn must be between 1 and ndim inclusive")

    origin = tuple(0 for _ in range(ndim))
    if not mask[origin]:
        return np.zeros_like(mask, dtype=bool)

    visited = np.zeros_like(mask, dtype=bool)
    visited[origin] = True
    q: deque[Tuple[int, ...]] = deque([origin])

    # Generate neighbour offsets respecting connectivity -----------------------
    offsets: list[Tuple[int, ...]] = []
    for delta in product((-1, 0, 1), repeat=ndim):
        if delta == (0,) * ndim or sum(map(abs, delta)) > conn:
            continue
        offsets.append(delta)

    while q:
        idx = q.popleft()
        for off in offsets:
            nb = tuple(i + o for i, o in zip(idx, off))
            if any(n < 0 or n >= s for n, s in zip(nb, mask.shape)):
                continue
            if mask[nb] and not visited[nb]:
                visited[nb] = True
                q.append(nb)
    return visited




def self_consistent_filter(phi_hat: np.ndarray, *, N: int, conn: int = 1) -> np.ndarray:
    Cmin = 2 * np.sqrt(N - 1) / N
    abs_phi = np.abs(phi_hat)
    eligible = abs_phi >= Cmin
    accept = _flood_fill(eligible, conn=conn)

    gain = np.zeros_like(abs_phi)
    if np.any(accept):
        numer = N / (2 * (N - 1))
        tmp = 1 - (4 * (N - 1)) / (N ** 2 * abs_phi[accept] ** 2)
        gain[accept] = numer * (1 + np.sqrt(tmp))
    return gain * phi_hat




def filtered_ecf_grid_einsum(
    X: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    s: np.ndarray,
    K: Sequence[int],
    *,
    dtype: np.dtype = np.complex128,
    conn: int = 1,
    return_raw: bool = False,
):
    phi_raw = ecf_grid_einsum(X, a, b, s, K, dtype=dtype)
    phi_sc = self_consistent_filter(phi_raw, N=len(X), conn=conn)
    return (phi_sc, phi_raw) if return_raw else phi_sc



def _sign_matrix(d: int) -> np.ndarray:
    """Return (2^{d-1}, d) matrix of sign vectors with first column all +1."""
    combos = list(product([-1, 1], repeat=d - 1))
    mat = np.empty((len(combos), d), dtype=int)
    mat[:, 0] = 1
    mat[:, 1:] = combos
    return mat


def build_cos_coefficients(
    X: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    K: Sequence[int],
    *,
    conn: int = 1,
    dtype: np.dtype = np.complex128,
) -> np.ndarray:
    """Return tensor A_{k₁…k_d} using fastKDE‑filtered ECF.

    Parameters
    ----------
    X, a, b, K : see Fang–Oosterlee COS method.
    conn : flood‑fill connectivity for filtering.
    dtype : complex dtype for internal characteristic functions.
    """
    X = np.asarray(X)
    a, b, K = map(np.asarray, (a, b, K))
    d = len(K)
    signs = _sign_matrix(d)

    # Compute filtered φ for every sign pattern -------------------------------
    phi_grids = []
    for s_vec in signs:
        phi_sc = filtered_ecf_grid_einsum(X, a, b, s_vec, K, dtype=dtype, conn=conn)
        phi_grids.append(phi_sc)

    phi_grids = np.stack(phi_grids)  # shape (2^{d-1}, K1, …, Kd)

    # Pre‑compute phase factor  exp(-iπ (s*k)/(b-a) · a) -----------------------
    grids = np.meshgrid(*[np.arange(Kj) for Kj in K], indexing="ij")  # each shape K1×…×Kd
    k_tensor = np.stack(grids, axis=-1)  # (..., d)
    A = np.zeros(K, dtype=float)

    prod_prefactor = 2.0 * np.prod(1.0 / (b - a))

    # Vectorised accumulation over sign patterns ------------------------------
    #   For each s, phase = exp(-i π Σ_j s_j k_j a_j / (b_j-a_j))
    for idx_s, s_vec in enumerate(signs):
        phase = np.exp(
            -1j
            * np.pi
            * np.tensordot(k_tensor, s_vec * a / (b - a), axes=([ -1], [0]))
        )
        contrib = np.real(phase * phi_grids[idx_s])
        A += contrib

    A *= prod_prefactor
    return A

################################################################################
#  6. Example / quick comparison                                               #
################################################################################
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    mu = np.array([0.0, 0.0, 0.0, 0.0])
    Sigma = np.array(
        [
            [1.0, 0.4, 0.3, 0.1],
            [0.4, 1.0, 0.6, 0.5],
            [0.3, 0.6, 1.0, 0.5],
            [0.1, 0.5, 0.5, 1.0],
        ]
    )
    std = np.sqrt(np.diag(Sigma))
    a = mu - 7 * std
    b = mu + 7 * std
    K = np.full(4,64)  # modest grid

    N = 50_000
    X = rng.multivariate_normal(mu, Sigma, size=N)

    # Build coefficients with filter ------------------------------------------
    t0 = perf_counter()
    A_filt = build_cos_coefficients(X, a, b, K)
    t1 = perf_counter()

