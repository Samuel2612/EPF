# -*- coding: utf-8 -*-
"""
Created on Thu May  1 10:35:50 2025
@author: samue
"""
# from __future__ import annotations

import itertools
from functools import partial
import math
import time

import numpy as np
import torch
import tntorch as tn
from ttcross import tt_cross


def index_grid(Ks):
    """Return a (*dims, n) array with every multi-index 0 ≤ k_i < K_i."""
    grid = np.indices(Ks, dtype=np.float64)  # (n, *dims)
    return np.moveaxis(grid, 0, -1)          # (*dims, n)


def chf_gaussian(t, mu, Sigma):
    """Characteristic function φ(t) of 𝒩(μ,Σ), broadcast-compatible."""
    lead, n = t.shape[:-1], t.shape[-1]
    tf = t.reshape(-1, n)                     # (N, n)
    phase = 1j * tf @ mu                      # (N,)
    quad = np.einsum("ij,jk,ik->i", tf, Sigma, tf)
    return np.exp(phase - 0.5 * quad).reshape(lead)


def build_tensor_lowmem(Ks, a, b, mu, Sigma):
    n = len(Ks)
    factor = 2.0 / np.prod(b - a)

    # all 2^{n-1} sign patterns with leading +1
    rest = np.array(list(itertools.product([1, -1], repeat=n - 1)), float)
    svec = np.concatenate([np.ones((rest.shape[0], 1)), rest], axis=1)  # (S,n)

    grid = index_grid(Ks)                    # (*dims, n)
    A = np.zeros(Ks, dtype=np.float64)

    for s in svec:
        t = math.pi * grid * s / (b - a)     # (*dims, n)
        phi = chf_gaussian(t, mu, Sigma)     # (*dims,)
        phase = np.exp(-1j * np.sum(t * a, axis=-1))
        A += np.real(phase * phi)

    return factor * A

def build_tensor_lowmem_ix(Ks, a, b, mu, Sigma, dtype=np.float64):
    d = len(Ks)
    factor = 2.0 / np.prod(b - a)


    k = np.ix_(*[np.arange(K, dtype=dtype) for K in Ks])   

    A = np.zeros(Ks, dtype=dtype)
    rest = np.array(list(itertools.product([1, -1], repeat=d-1)), dtype=dtype)
    svec = np.concatenate([np.ones((rest.shape[0], 1), dtype=dtype), rest], axis=1)

    for s in svec:                     # 2^(d-1) sign patterns

        t = [math.pi * k_i * s_i / (b_i - a_i) for k_i, s_i, a_i, b_i in zip(k, s, a, b)]


        quad = np.zeros(Ks, dtype=dtype)
        for i in range(d):
            for j in range(d):
                quad += Sigma[i, j] * t[i] * t[j]

        phase = np.exp(-1j * sum(t_i * a_i for t_i, a_i in zip(t, a)))
        phi   = np.exp(1j * sum(t_i * mu_i for t_i, mu_i in zip(t, mu)) - 0.5 * quad)
        A    += np.real(phase * phi)

    return factor * A



def A_entries(k, a, b, mu, Sigma):
    """Entry generator for the cross algorithm (expects 2-D index array)."""
    # k.shape == (N, d)
    d = k.shape[1]
    rest_t = torch.tensor(list(itertools.product([1, -1], repeat=d-1)))
    svec_t = torch.cat((torch.ones(rest_t.size(0), 1), rest_t), dim=1)  
    svec_t = svec_t        
    res = torch.zeros(k.shape[0], dtype=torch.float64)
    for s in svec_t:                            # 2^(d-1) iterations
        t = math.pi * k * s / (b - a)       # (N,d)
        phase = torch.exp(-1j * (t * a_t).sum(dim=1))
        quad = torch.einsum('ni,ij,nj->n', t, Sigma, t)
        phi = torch.exp(1j * (t @ mu) - 0.5 * quad)
        res += torch.real(phase * phi)
    factor_t = 2.0 / torch.prod(b - a)
    return factor_t * res



def tt_svd(tensor, eps = 1e-12):
    """Simple TT-SVD with Frobenius accuracy ``eps`` · ‖A‖ₙᵥ."""
    T = tensor.copy()
    dims = T.shape
    d = len(dims)

    norm2 = np.linalg.norm(T) ** 2
    thr = (eps / math.sqrt(d - 1)) ** 2 * norm2

    cores = []
    ranks = [1]
    unfold = T

    for k in range(d - 1):
        m = ranks[-1] * dims[k]
        unfold = unfold.reshape(m, -1)

        U, S, Vh = np.linalg.svd(unfold, full_matrices=False)
        tail = np.cumsum(S[::-1] ** 2)[::-1]
        r = int(np.searchsorted(tail <= thr, True))

        cores.append(U[:, :r].reshape(ranks[-1], dims[k], r))
        ranks.append(r)
        unfold = (S[:r, None] * Vh[:r])

    cores.append(unfold.reshape(ranks[-1], dims[-1], 1))
    ranks.append(1)
    return cores, ranks



def _cos_basis(x_i, a_i, b_i, N,):
    """Cosine basis vector (half factor on k = 0)."""
    if isinstance(x_i, torch.Tensor):
        k = torch.arange(N, dtype=x_i.dtype, device=x_i.device)
        basis = torch.cos(k * math.pi * (x_i - a_i) / (b_i - a_i))
        basis[0] *= 0.5
        return basis
    # NumPy branch
    k = np.arange(N, dtype=float)
    basis = np.cos(k * math.pi * (x_i - a_i) / (b_i - a_i))
    basis[0] *= 0.5
    return basis



def pdf_tt(x,cores,a,b, method = "sum",  # "product" or "sum"
):
    """Evaluate the COS-TT pdf at *one* point x.

    * ``cores`` – list of NumPy arrays **or** torch-compatible TT.
    * ``method='product'`` – contract left-to-right (default).
    * ``method='sum'``     – basis first, then ordinary matrix chain.
    """

    if isinstance(x, torch.Tensor):
        tcores = cores.cores if hasattr(cores, "cores") else cores  # unwrap
        dtype = tcores[0].dtype
        vec = torch.ones(1, dtype=dtype, device=x.device)
        mats = []
        for i, G in enumerate(tcores):
            N = G.shape[1]
            basis = _cos_basis(x[i], a[i], b[i], N)
            if method == "product":
                tmp = torch.tensordot(vec, G, dims=([0], [0]))
                vec = torch.tensordot(basis, tmp, dims=([0], [0]))
            else:  
                mats.append(torch.tensordot(G, basis, dims=([1], [0])))
        if method == "product":
            return float(vec.squeeze())
        for M in mats[1:]:
            mats[0] = mats[0] @ M
        return float(mats[0].squeeze())

    vec = np.ones(1)
    mats_np = []
    for i, G in enumerate(cores):
        N = G.shape[1]
        basis = _cos_basis(x[i], a[i], b[i], N)
        if method == "product":
            tmp = np.tensordot(vec, G, axes=([0], [0]))
            vec = np.tensordot(basis, tmp, axes=([0], [0]))
        else:
            mats_np.append(np.tensordot(G, basis, axes=([1], [0])))
    if method == "product":
        return float(vec.squeeze())
    for M in mats_np[1:]:
        mats_np[0] = mats_np[0] @ M
    return float(mats_np[0].squeeze())


def _cos_moments_const(
    a_i, b_i, N, *, backend = "np", dtype=None):
    J = np.zeros(N, dtype=float) if backend == "np" else torch.zeros(N, dtype=dtype)
    J[0] = 0.5 * (b_i - a_i)
    return J


def _cos_moments_mean( a_i, b_i, N, *, backend = "np", dtype=None):
    J = np.zeros(N, dtype=float) if backend == "np" else torch.zeros(N, dtype=dtype)
    J[0] = 0.25 * (a_i + b_i) * (b_i - a_i)
    if N > 1:
        rng = np.arange(1, N) if backend == "np" else torch.arange(1, N, dtype=dtype)
        J[1:] = (b_i - a_i) ** 2 * (((-1.0) ** rng) - 1.0) / (rng * math.pi) ** 2
    return J


def mean_tt(cores,a,b):
    """Return the vector *μ̂* of means E[Z] encoded in a COS-TT density."""
    if isinstance(a, torch.Tensor):
        tcores = cores.cores if hasattr(cores, "cores") else cores
        dtype = tcores[0].dtype
        device = tcores[0].device
        d = len(tcores)
        mu_hat = torch.empty(d, dtype=dtype, device=device)
        for j in range(d):
            vec = torch.ones(1, dtype=dtype, device=device)
            for i, G in enumerate(tcores):
                N = G.shape[1]
                if i == j:
                    J = _cos_moments_mean(a[i], b[i], N, backend="torch", dtype=dtype).to(device)
                else:
                    J = _cos_moments_const(a[i], b[i], N, backend="torch", dtype=dtype).to(device)
                M_i = torch.tensordot(G, J, dims=([1], [0]))
                vec = torch.matmul(vec, M_i)
            mu_hat[j] = vec.squeeze()
        return mu_hat.cpu().numpy()

    d = len(cores)
    mu_hat = np.empty(d)
    for j in range(d):
        vec = np.ones(1)
        for i, G in enumerate(cores):
            N = G.shape[1]
            if i == j:
                J = _cos_moments_mean(a[i], b[i], N, backend="np")
            else:
                J = _cos_moments_const(a[i], b[i], N, backend="np")
            M_i = np.tensordot(G, J, axes=([1], [0]))
            vec = vec @ M_i
        mu_hat[j] = vec.squeeze()
    return mu_hat





if __name__ == "__main__":
    FULL_TENSOR = True   # build & compress the full tensor
    CROSS_TEST =  False  # tensor-train cross approximation only 

    d = 5
    N = 32
    Ks = [N] * d
    mu= np.array([1.0, 0.5, -1.1, 1.3, 0.3])
    Sigma =  np.array([[1.0, 0.4, 0.3, 0.1, 0.6],
                                [0.4, 1.0, 0.0, 0.2, 0.3],
                                [0.3, 0.0, 1.0, 0.0, 0.1],
                                [0.1, 0.2, 0.0, 1.0, 0.1],
                                [0.6, 0.3, 0.1, 0.1, 1.0]])
     
    x = np.array([0.2, 0.1, -0.6, 0.5, 0.3])
    std = np.sqrt(np.diag(Sigma))
    a = mu - 6 * std
    b = mu + 6 * std
    
            
    invS = torch.linalg.inv(torch.from_numpy(Sigma))
    detS = torch.linalg.det(torch.from_numpy(Sigma))
    norm_c = 1.0 / torch.sqrt((2*np.pi)**d * detS)
    diff = torch.from_numpy(x) - torch.from_numpy(mu)
    fx_ex = norm_c * torch.exp(-0.5 * diff @ invS @ diff)


    if FULL_TENSOR:
        print("\n=== Building full tensor  …")
        A = build_tensor_lowmem(Ks, a, b, mu, Sigma)
        print("Shape :", A.shape)

        print("\n=== Compressing with torchTT  …")
        t0 = time.time()
        A_tt = tn.Tensor(A)
        print(f"done in {time.time() - t0:.2f}s  –  ranks {A_tt.ranks_tt}")

        print("\n=== Compressing with TT-SVD (NumPy) …")
        t0 = time.time()
        cores_np, ranks = tt_svd(A, eps=1e-20)
        print(f"done in {time.time() - t0:.2f}s  –  ranks {ranks}")

        # ----------------------------------------------------------- pdf / mean
        
        x_t = torch.from_numpy(x)

        fx_torch = pdf_tt(torch.from_numpy(x), A_tt, torch.from_numpy(a), torch.from_numpy(b))
        fx_np = pdf_tt(x, cores_np, a, b)
        print(f"\npdf (torchTT) : {fx_torch:.8e}")
        print(f"pdf (NumPy TT): {fx_np:.8e}")
        print(f"Exact pdf  : {fx_ex:.8e}")

        mu_hat = mean_tt(A_tt, torch.from_numpy(a), torch.from_numpy(b))
        print("Estimated mean (Torch):", mu_hat)
        mu_hat = mean_tt(cores_np, a, b)
        print("Estimated mean (Numpy):", mu_hat)
        

   

    if CROSS_TEST:
       

        print("\n=== Cross approximation  …")
        mu_t = torch.from_numpy(mu)
        Sigma_t = torch.from_numpy(Sigma)
        a_t = torch.from_numpy(a)
        b_t = torch.from_numpy(b)
        factor_t = 2.0 / torch.prod(b_t - a_t)



        cross_start = time.time()
        A_cross = partial(A_entries, a=a_t, b=b_t, mu= mu_t, Sigma = Sigma_t)
        A_tt_cross = tn.cross(
            function=A_cross,
            domain=Ks,
            rmax=200,
            max_iter=100,
            eps=1e-6,
            verbose=False,
            function_arg='matrix',
        )
        cross_time = time.time() - cross_start
    
        fx_cross = pdf_tt(torch.from_numpy(x), A_tt_cross, torch.from_numpy(a), torch.from_numpy(b))
        print(f"pdf (TN-cross) : {fx_cross:.8e}")
        print(f"Exact pdf  : {fx_ex:.8e}")



    





