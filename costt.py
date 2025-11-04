# -*- coding: utf-8 -*-
"""
Created on Thu May  1 10:35:50 2025
@author: samue

Builds the tensor A_{k1…kn} (first sign fixed s₁=+1, general N(μ,Σ) characteristic fn)
and then computes its Tensor-Train (TT) decomposition using your own tt_svd.
"""

import numpy as np
import itertools
import math
import torch
import torchtt as tntt
from ttcross import tt_cross
import tntorch as tn
import time

torch.set_default_dtype(torch.float64)   

def index_grid(Ks):
    """(*dims, n) array of all k-vectors for 0≤k_i≤K_i."""
    dims = [K for K in Ks]
    grid = np.indices(dims, dtype=np.float32)  # (n,*dims)
    return np.moveaxis(grid, 0, -1)                      # (*dims,n)


def chf_gaussian(t, mu, Sigma):
    """
    φ(t) = exp(i t·μ − ½ tᵀΣt), broadcast-ready.
    t has shape (..., n).
    """
    lead, n = t.shape[:-1], t.shape[-1]
    tf = t.reshape(-1, n)                            # (N, n)
    phase = 1j * tf @ mu                             # (N,)
    quad = np.einsum("ij,jk,ik->i", tf, Sigma, tf)  # (N,)
    return np.exp(phase - 0.5 * quad).reshape(lead)


def build_tensor(Ks, a, b, mu, Sigma):
    """
    A_k = 2·∏ 1/(b_i−a_i) · Σ_{s∈S} Re[ e^{-iπ(s∘k)·a/(b−a)} · φ(π(s∘k)/(b−a)) ],
    with S = {(1,s₂,…,sₙ)}, |S|=2^{n−1}.
    """
    n = len(Ks)
    factor = 2.0 / np.prod(b - a)

    rest = np.array(list(itertools.product([1, -1], repeat=n-1)), dtype=float)
    svec = np.concatenate([np.ones((rest.shape[0],1)), rest], axis=1)  # (S, n)

    grid     = index_grid(Ks)                                  # (*dims, n)
    k_scaled = math.pi * grid[..., None, :] / (b - a)          # (*dims, 1, n)
    t        = k_scaled * svec                                 # (*dims, S, n)

    phase = np.exp(-1j * np.sum(t * a, axis=-1))               # (*dims, S)
    phi   = chf_gaussian(t, mu, Sigma)                         # (*dims, S)

    return factor * np.real(phase * phi).sum(axis=-1)          # (*dims,)

def build_tensor_lowmem(Ks, a, b, mu, Sigma):
    """
    Same A_k, but loops over each s∈S to keep memory small.
    """
    n = len(Ks)
    factor = 2.0 / np.prod(b - a)
    # create sign-vectors S = {(+1,s₂,…,sₙ)}, |S|=2^(n−1)
    rest = np.array(list(itertools.product([1, -1], repeat=n-1)), float)
    svec = np.concatenate(
        [np.ones((rest.shape[0], 1)), rest], axis=1)  # (|S|,n)

    grid = index_grid(Ks)            # (*dims, n)
    A = np.zeros(Ks, dtype=np.float32)

    # loop over each sign pattern, accumulate
    for s in svec:
        t = np.pi * grid * s / (b - a)       # (*dims, n)
        phi = chf_gaussian(t, mu, Sigma)         # (*dims,)
        phase = np.exp(-1j * np.sum(t * a, axis=-1))
        A += np.real(phase * phi)

    return factor * A


def tt_svd(tensor, eps=1e-12):
    """
    TT-SVD

    The truncation guarantees
        ‖A – A_TT‖_F  ≤  eps · ‖A‖_F .
    """
    T = tensor.copy()
    dims = T.shape
    d = len(dims)
    normA2 = np.linalg.norm(T) ** 2
    thr = (eps / np.sqrt(d - 1)) ** 2 * normA2

    cores, ranks = [], [1]           # r₀ = 1
    unfold = T

    for k in range(d - 1):
        # mode-k unfolding :  (r_{k-1}·n_k) × rest
        m = ranks[-1] * dims[k]
        unfold = unfold.reshape(m, -1)

        U, S, Vh = np.linalg.svd(unfold, full_matrices=False)
        tail = np.cumsum(S[::-1] ** 2)[::-1]
        r = np.searchsorted(tail <= thr, True)     # first tail ≤ thr

        U_r, S_r, Vh_r = U[:, :r], S[:r], Vh[:r, :]
        core = U_r.reshape(ranks[-1], dims[k], r)
        cores.append(core)
        ranks.append(r)
        unfold = (S_r[:, None] * Vh_r)

    # final core  (r_{d-1}, n_d, 1)
    Gd = unfold.reshape(ranks[-1], dims[-1], 1)
    cores.append(Gd)
    ranks.append(1)                                   # r_d = 1

    return cores, ranks


def cos_basis(x, a, b, N):
    k = torch.arange(N)
    basis = torch.cos(k * math.pi * (x - a) / (b - a))
    basis[0] *= 0.5                              # <-- apply the ½ here
    return basis


def pdf_tt(x, A_tt, a, b):
    """
    Evaluate COS-TT pdf at a single point x  (1-D torch tensor, len d)
    """
    dtype  = A_tt.cores[0].dtype 
    cores = A_tt.cores

    vec = torch.ones(1, dtype = dtype)  # r₀ = 1
    for i, G in enumerate(cores):                    # G: (r_{i-1}, n_i, r_i)
        basis = cos_basis(x[i], a[i], b[i], G.shape[1])
        tmp = torch.tensordot(vec, G, dims=([0], [0]))   # (n_i, r_i)
        basis = basis.to(dtype = dtype)
        tmp = tmp.to(dtype = dtype)
        vec = torch.tensordot(basis, tmp, dims=([0], [0]))  # (r_i,)
    return vec.item()

def pdf_tt_sum(x, A_tt, a, b):
    """
    Evaluate COS-TT pdf at a single point x  (1-D torch tensor, len d)
    """
    cores = A_tt.cores
    mats = []
    for i, G in enumerate(cores):
        N = G.shape[1]                       # K_i
        basis = cos_basis(x[i], a[i], b[i], N)  # (N,)
    
    
        M_i = torch.tensordot(G, basis, dims= ([1], [0]))
        mats.append(M_i)
    
    result = mats[0]
    for M in mats[1:]:
        result = result @ M                 # standard matrix multiplication
    
    return float(result.squeeze())          # scalar (because r_0=r_d=1)

def cos_basis_np(x_i, a_i, b_i, N):
    k = np.arange(N)
    c = np.cos(k * math.pi * (x_i - a_i) / (b_i - a_i))
    c[0] *= 0.5
    return c          # shape (N,)




def pdf_tt_sums(x, cores, a, b):
    """
    Evaluate f(x) according to the product-of-sums representation.

    """

    mats = []
    for i, G in enumerate(cores):
        N = G.shape[1]                       # K_i
        basis = cos_basis_np(x[i], a[i], b[i], N)  # (N,)


        M_i = np.tensordot(G, basis, axes=([1], [0]))
        mats.append(M_i)

    result = mats[0]
    for M in mats[1:]:
        result = result @ M                 # standard matrix multiplication

    return float(result.squeeze())          # scalar (because r_0=r_d=1)

def pdf_tt_np(x, cores, a, b):
    """
    Evaluate the COS-TT pdf at a point  x  (NumPy, no torch).

    """
    vec = np.ones(1)
    for i, G in enumerate(cores):
        N = G.shape[1]
        basis = cos_basis_np(x[i], a[i], b[i], N)
        tmp = np.tensordot(vec, G, axes=([0], [0]))
        vec = np.tensordot(basis, tmp, axes=([0], [0]))

    return float(vec.squeeze())   # scalar


def _cos_moments_const(a_i, b_i, N, *, np_like=True, dtype=None):
    """
    J_k = ∫_{a_i}^{b_i} 0.5·cos(kπu) · (b_i−a_i) du      (constant g=1)
    Only k = 0 survives.
    """
    J = (np.zeros if np_like else torch.zeros)(N, dtype=dtype)
    J[0] = 0.5 * (b_i - a_i)
    return J


def _cos_moments_mean(a_i, b_i, N, *, np_like=True, dtype=None):
    """
    J_k = ∫_{a_i}^{b_i} z · B_k(z) dz,   with  B_0 ≡ 0.5,  B_k = cos(kπu) (k>0)
         closed form (cf. proof):
           k = 0 :  (a_i + b_i)(b_i - a_i)/4
           k > 0 :  (b_i - a_i)^2 · ( (-1)^k - 1 ) / (kπ)^2
    """
    J = (np.zeros if np_like else torch.zeros)(N, dtype=dtype)
    # k = 0
    J[0] = 0.25 * (a_i + b_i) * (b_i - a_i)
    if N > 1:
        k = (np.arange if np_like else torch.arange)(1, N, dtype=dtype)
        J[1:] = (b_i - a_i) ** 2 * (((-1.0) ** k) - 1.0) / (k * math.pi) ** 2
    return J


def mean_tt(cores, a, b):
    """
    Vector of means  E[Z]  from a COS-TT density.

    Parameters
    ----------
    cores : list or object
        • list of NumPy arrays  (r_{j-1},  N_j,  r_j)   ––> NumPy branch
        • TT object with attribute `.cores` holding torch.Tensor cores
          of the same shape                                   ––> torch branch
    a, b : 1-D array-like
        Integration window endpoints used in the COS expansion.

    Returns
    -------
    μ_hat : 1-D NumPy array  (length = dimension d)
    """

    if isinstance(a, np.ndarray):        # NumPy
        d = len(cores)
        mu_hat = np.empty(d, dtype=float)

        for j in range(d):               # compute E[Z_j] one coordinate at a time
            vec = np.ones(1)             # r_0 = 1
            for i, G in enumerate(cores):
                N_i = G.shape[1]

                if i == j:     # moments for mean of coordinate j
                    J = _cos_moments_mean(a[i], b[i], N_i, np_like=True)
                else:          # constant moments
                    J = _cos_moments_const(a[i], b[i], N_i, np_like=True)

                M_i = np.tensordot(G, J, axes=([1], [0]))   # (r_{i-1}, r_i)
                vec = vec @ M_i                             # keep left contract

            mu_hat[j] = float(vec.squeeze())                # r_d = 1 → scalar

        return mu_hat

    else:                                  
        # Accept plain tensors, torchtt.TT, tntorch TT, ...
        torch_cores = cores.cores if hasattr(cores, 'cores') else cores
        dtype = torch_cores[0].dtype
        device = torch_cores[0].device

        a = torch.as_tensor(a, dtype=dtype, device=device)
        b = torch.as_tensor(b, dtype=dtype, device=device)

        d = len(torch_cores)
        mu_hat = torch.empty(d, dtype=dtype, device=device)

        for j in range(d):
            vec = torch.ones(1, dtype=dtype, device=device)
            for i, G in enumerate(torch_cores):
                N_i = G.shape[1]

                if i == j:
                    J = _cos_moments_mean(a[i], b[i], N_i,
                                          np_like=False, dtype=dtype).to(device)
                else:
                    J = _cos_moments_const(a[i], b[i], N_i,
                                           np_like=False, dtype=dtype).to(device)

                M_i = torch.tensordot(G, J, dims=([1], [0]))  # (r_{i-1}, r_i)
                vec = torch.matmul(vec, M_i)

            mu_hat[j] = vec.squeeze()

        return mu_hat.cpu().numpy()        # keep return type consistent
    
if __name__ == "__main__":

    Ks = [64]*4
    d = len(Ks)
    
    only_cross_test = False
    mu = np.array([1.0, 0.5, -1.1, 1.3])
    
    # Sigma = np.array([[1.0, 0.4, 0.3, 0.1, 0.1, 0.0],
    #                   [0.4, 1.0, 0.2, 0.5, 0.3, 0.2],
    #                   [0.3, 0.2, 1.0, 0.5, 0.0, 0.1],
    #                   [0.1, 0.5, 0.5, 1.0, 0.1, 0.5],
    #                   [0.1, 0.3, 0.0, 0.1, 1.0, 0.3],
    #                   [0.0, 0.2, 0.1, 0.5, 0.3, 1.0]])
    
    
    # Sigma = np.array([[1.0, 0.4, 0.3, 0.1, 0.6],
    #                   [0.4, 1.0, 0.0, 0.2, 0.3],
    #                   [0.3, 0.0, 1.0, 0.0, 0.1],
    #                   [0.1, 0.2, 0.0, 1.0, 0.1],
    #                   [0.6, 0.3, 0.1, 0.1, 1.0]])
    
    Sigma = np.array([[1.0, 0.4, 0.3, 0.1],
                        [0.4, 1.0, 0.6, 0.5],
                        [0.3, 0.6, 1.0, 0.5],
                        [0.1, 0.5, 0.5, 1.0]])
    
    # Sigma = np.array([[1.0, 0.4, 0.3],
    #                   [0.4, 1.0, 0.6],
    #                   [0.3, 0.6, 1.0]])
    
    
    def nearest_spd(A, tol=1e-10, maxiter=100):
        S = 0.5 * (A + A.T)
        Y = S.copy()
        for k in range(maxiter):
            eigval, eigvec = np.linalg.eigh(Y)
            eigval[eigval < 0] = 0
            Y_new = (eigvec * eigval) @ eigvec.T
            Y_new = S + (Y_new - S)      # enforce symmetry
            if np.linalg.norm(Y_new - Y, 'fro') < tol:
                break
            Y = Y_new
        # final jitter for positive-definite guarantee
        eps = 1e-12
        while True:
            try:
                np.linalg.cholesky(Y); break
            except np.linalg.LinAlgError:
                Y += eps * np.eye(Y.shape[0]); eps *= 10
        return Y
    
    # Sigma = nearest_spd(Sigma)
    
    std = np.sqrt(np.diag(Sigma))
    a = mu - 8*std
    b = mu + 8*std
    eps = 1e-20
    
    if not only_cross_test:
        print("Building full tensor A …")
        A = build_tensor_lowmem(Ks, a, b, mu, Sigma)
        print("A shape:", A.shape)
        A_torch = torch.from_numpy(A)
        print("\nComputing TT using torchTT library …")
        
        start = time.time()
        A_tt = tntt.TT(A_torch, eps=eps)
        print(f"Computation time (sec) = {time.time() - start}")
        print("dimensions (n_i):", A_tt.N)
        print("TT ranks        :", A_tt.R)     # TT-SVD & rounding
        
        
        print("\nComputing TT from (Oseledets, 2011)")
        
        start_ = time.time()
        cores, ranks = tt_svd(A, eps=eps)
        print(f"Computation time (sec) = {time.time() - start_}")
        print("\nTT ranks:", ranks)
        for idx, G in enumerate(cores, 1):
            print(f"Core {idx}: shape {G.shape}")
    
    
    
    
    x_point = np.array([0.2, 0.1, -0.6, 0.5])   # example evaluation point
    x_point_t = torch.from_numpy(x_point)
    a_t = torch.from_numpy(a)
    b_t = torch.from_numpy(b)
    mu_t = torch.from_numpy(mu)
    Sigma_t = torch.from_numpy(Sigma)
    
    if not only_cross_test:
        start_time = time.time()
        fx_tt_np = pdf_tt_np(x_point, cores, a, b)
        calc_time = time.time() - start_time
        start_time = time.time()
        fx_tt_sums = pdf_tt_sums(x_point, cores, a, b)
        calc_time_ = time.time() - start_time
        
        fx_tt = pdf_tt(x_point_t, A_tt, a_t, b_t)
    
    
    
    factor_t = 2.0 / torch.prod(b_t - a_t)
    # (+1, ±1, …) sign patterns – compute once, keep as torch tensor
    rest = torch.tensor(list(itertools.product([1, -1], repeat=d-1)))
    svec_t = torch.cat((torch.ones(rest.size(0), 1), rest), dim=1)  # (S,d)
    svec_t = svec_t.to(torch.float64)        # (16,5)
    
    
    def A_entries(k):       # idx shape (N,d)
        res = torch.zeros(k.shape[0])
    
        for s in svec_t:                                    # 16 iterations
            t = math.pi * k * s / (b_t - a_t)               # (N,d)
            phase = torch.exp(-1j * (t * a_t).sum(dim=1))   # (N,)
            quad = torch.einsum('ni,ij,nj->n', t, Sigma_t, t)
            phi = torch.exp(1j*(t @ mu_t) - 0.5*quad)     # (N,)
            res += torch.real(phase * phi)
    
        return factor_t * res
    
    
    cross_start = time.time()
    # A_tt_cross = tntt.interpolate.dmrg_cross(A_entries, Ks,  dtype = torch.float32, eps=1e-10, verbose=True)
    A_tt_cross = tn.cross(function = A_entries, domain = Ks, rmax = 750, max_iter = 300, eps=1e-10,  function_arg='matrix')
    # A_tt_cross = tt_cross(f = A_entries, shape = Ks, ranks= [1, 13,13,1 ])
    fx_tt_cross = pdf_tt_sum(x_point_t, A_tt_cross, a_t, b_t)
    cross_time = time.time() - cross_start 
    print("cross TT ranks:", A_tt_cross)
    
    
    
    invS = torch.linalg.inv(Sigma_t)
    detS = torch.linalg.det(Sigma_t)
    norm_c = 1.0 / torch.sqrt((2*np.pi)**d * detS)
    diff = x_point_t - mu_t
    fx_ex = norm_c * torch.exp(-0.5 * diff @ invS @ diff)
    

    
    if not only_cross_test:
        
        print(f"\nCOS-TT pdf product first (Own TT decomp) : {fx_tt_np:15.8e}")
        print(f"\nCalculation time : {calc_time:15.8e}")
        print(f"\nCOS-TT pdf sums first (Own TT decomp) : {fx_tt_sums:15.8e}")
        print(f"\nCalculation time : {calc_time_:15.8e}")
        print(f"COS-TT pdf (Existing TT decomp) : {fx_tt:15.8e}")
    
    
    
    print(f"\nCOS-TT pdf (Cross) : {fx_tt_cross:15.8e}")
    print(f"Calculation time : {cross_time:15.8e}")
    
    print(f"\nExact pdf   : {fx_ex.item():15.8e}")
    
    #     # ---- NumPy TT cores ------------------------------------------------------
    mu_hat_np = mean_tt(cores, a, b)
    print("Mean from NumPy TT  :", mu_hat_np)
    
    # ---- PyTorch / torchTT TT object ----------------------------------------
    mu_hat_torch = mean_tt(A_tt_cross, a_t, b_t)         # a_t, b_t are torch tensors
    print("Mean from torch TT :", mu_hat_torch)
        
    
    