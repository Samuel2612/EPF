# -*- coding: utf-8 -*-
"""
Created on Thu May  1 10:35:50 2025

@author: samue
"""

import numpy as np
from itertools import product


def index_grid(Ks):
    """
    Generate an N-dimensional array of all index combinations.

    Parameters:
    -----------
    Ks : list of int
        Inclusive upper bounds for each index k_i.
        The grid shape along axis i will be (Ks[i] + 1).

    Returns:
    --------
    grid : np.ndarray, shape (*dims, n)
        An array where dims = [K1+1, K2+1, ..., Kn+1]
        and n = len(Ks). Each entry at
        grid[k1, k2, ..., kn] == [k1, k2, ..., kn].
    """

    dims = [K + 1 for K in Ks]
    indices = np.indices(dims)
    grid = np.moveaxis(indices, 0, -1)
    return grid

def chf_gaussian(t, mu, Sigma) -> np.ndarray:
    """
    Characteristic function φ(t) = exp(i t·μ − ½ tᵀΣt)   (broadcast-ready).

    *t* may have any leading shape (..., n); only the last axis is dimension n.
    """
    lead_shape, n = t.shape[:-1], t.shape[-1]
    tf = t.reshape(-1, n)                               # (N, n)

    # i t·μ  term
    phase_lin = 1j * tf @ mu                            # (N,)

    # ½ tᵀ Σ t  term
    quad = np.einsum("ij,jk,ik->i", tf, Sigma, tf)      # (N,)

    return np.exp(phase_lin - 0.5 * quad).reshape(lead_shape)

def build_tensor(Ks, a, b, mu, Sigma):
    """Compute the tensor A according to the formula."""
    n = len(Ks)
    

    factor = 2.0 / np.prod(b - a)                # 2 · ∏ 1/(b_i − a_i)
    sign_rest = product([1, -1], repeat= n- 1)
    s_vectors = np.array([[1, *rest] for rest in sign_rest], dtype=float)

    grid = index_grid(Ks)                        # (*dims, n)
    k_scaled = np.pi * grid[..., None, :] / (b - a)  # (*dims,1,n)
    
    t = k_scaled * s_vectors 

    A = np.zeros(grid.shape[:-1], dtype=float)   # (*dims,)

     # phase and φ evaluated for every s in parallel
    phase = np.exp(-1j * np.sum(t * a, axis=-1))  # (*dims, S)
    phi   = chf_gaussian(t, mu, Sigma)                # (*dims, S)

    A = factor * np.real(phase * phi).sum(axis=-1)  # sum over S axis
    return factor * A

def tt_svd(tensor,
           eps = 1e-12,
           max_rank = None
           ) :
    """
    Standard TT-SVD (Oseledets, 2011).

    Parameters
    ----------
    tensor   : full tensor  (n₁,…,n_d)
    eps      : relative Frobenius error bound
    max_rank : optional cap on TT ranks

    Returns
    -------
    cores : list of TT cores  G¹,…,Gᵈ
            G¹  shape (n₁, r₁)
            Gᵏ  shape (r_{k-1}, nₖ, rₖ)  for k=2…d-1
            Gᵈ  shape (r_{d-1}, n_d)
    ranks : list of TT ranks  [1,r₁,…,r_{d-1},1]
    """
    dims = tensor.shape
    d = len(dims)
    mats = tensor.copy().reshape(dims[0], -1)   # (n₁, rest)
    norm2 = np.linalg.norm(tensor)**2
    cores, ranks = [], [1]

    for k in range(d - 1):
        m, n_rest = mats.shape
        # target SVD truncation threshold for this unfolding
        thresh = eps / np.sqrt(d - 1) * np.linalg.norm(mats)
        u, s, vt = np.linalg.svd(mats, full_matrices=False)
        # rank truncation
        rk = s.size
        if thresh > 0:
            rk = np.searchsorted(np.cumsum(s[::-1]**2)[::-1],
                                 thresh**2, side="right") + 1
        if max_rank:
            rk = min(rk, max_rank)
        # keep first rk singular values
        u, s, vt = u[:, :rk], s[:rk], vt[:rk, :]
        cores.append(u.reshape(ranks[-1], dims[k], rk)
                       .transpose(1, 0, 2))               # (r_{k-1}, n_k, r_k) → (n_k,r_{k-1},r_k)
        ranks.append(rk)
        mats = (np.diag(s) @ vt).reshape(rk, -1)          # next unfolding
        if k == d - 2:
            cores.append(mats.reshape(rk, dims[-1]).transpose(1, 0))  # Gᵈ
            ranks.append(1)
    # bring cores to conventional shapes
    G1 = cores[0].transpose(0, 2)         # (n₁, r₁)
    cores[0] = G1
    for k in range(1, d-1):
        cores[k] = cores[k].transpose(1, 0, 2)  # (r_{k-1}, n_k, r_k)
    return cores, ranks

def cos_tt_tensor(mu, Sigma, a, b, Ks, eps=1e-10):
    """
    returns   torchtt.TT   (cores are torch tensors on the same device)
    """
    A_full = build_tensor(Ks, a, b, mu, Sigma)
    A_tt   = tt_svd(A_full, eps=eps)          # TT-SVD & rounding
    return A_tt
if __name__ == "__main__":
    d = 4
    mu     = np.zeros(d)
    Sigma  = np.array([[1.0, 0.4, 0.3, 0.1],
                       [0.4, 1.0, 0.2, 0.5],
                       [0.3, 0.2, 1.0, 0.5],
                       [0.1, 0.5, 0.5, 1.0]])
    std    = np.sqrt(np.diag(Sigma))
    a, b   = (mu -8*std), (mu + 8*std)                # integration box
    Ks = [5, 5, 5, 5] 
    
    print("Building full tensor A …")
    A = build_tensor(Ks, a, b, mu, Sigma)
    print("Shape:", A.shape)

    print("Performing TT-SVD …")
    cores, ranks = tt_svd(A)
    print("TT ranks:", ranks)