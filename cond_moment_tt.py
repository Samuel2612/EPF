#!/usr/bin/env python3
"""COS–TT demo: conditional moments E[Y^p|X] (p=1..4) for a 4-D Gaussian.

* pure NumPy implementation
* TT compression via a plain TT-SVD
* no external TT libraries or cross approximation required
"""
import itertools, math, time
from typing import List, Tuple

import numpy as np


def index_grid(Ks):
    grid = np.indices(Ks, dtype=np.float64)
    return np.moveaxis(grid, 0, -1)

def chf_gaussian(t, mu, Sigma):
    tf   = t.reshape(-1, t.shape[-1])                    # (N,4)
    phase = 1j * tf @ mu                                 # (N,)
    quad  = np.einsum("ij,jk,ik->i", tf, Sigma, tf)     # (N,)
    return np.exp(phase - 0.5 * quad).reshape(t.shape[:-1])

def build_cos_tensor(Ks, a, b, mu, Sigma):
    n      = len(Ks)
    L      = b - a
    factor = 2.0 / L.prod()
    rest   = np.array(list(itertools.product([1, -1], repeat=n-1)), float)
    Svec   = np.c_[np.ones(rest.shape[0]), rest]        # (2^{n-1},n)
    grid   = index_grid(Ks)                             # (*Ks,n)
    A      = np.zeros(Ks, dtype=np.float64)
    for s in Svec:
        t     = math.pi * grid * s / L
        phi   = chf_gaussian(t, mu, Sigma)
        phase = np.exp(-1j * (t * a).sum(-1))
        A    += np.real(phase * phi)
    return factor * A

def tt_svd(tensor: np.ndarray, eps: float = 1e-12) -> Tuple[List[np.ndarray], List[int]]:
    T     = tensor.copy()
    dims  = T.shape
    d     = len(dims)
    norm2 = np.linalg.norm(T)**2
    thr   = (eps / math.sqrt(d-1))**2 * norm2

    cores = []
    ranks = [1]
    unfold = T
    for k in range(d-1):
        m = ranks[-1] * dims[k]
        unfold = unfold.reshape(m, -1)
        U, S, Vh = np.linalg.svd(unfold, full_matrices=False)
        tail     = np.cumsum(S[::-1]**2)[::-1]
        r        = np.argmax(tail <= thr)
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
    """
    Accurate weights J_k^{(p)} for the COS expansion of y^p (p≥2)
    compatible with cos_basis() where basis[0] is halved.
    """
    Δ = b - a
    J = np.zeros(N, dtype=np.float64)

    # k = 0  (half weight!)
    J[0] = 0.5 * (b**(p + 1) - a**(p + 1)) / (p + 1)

    # k ≥ 1
    for k in range(1, N):
        λ   = k * math.pi
        iλ  = 1j * λ
        # pre-compute powers of iλ / m!
        pow_series = [(iλ)**m / math.factorial(m) for m in range(p + 1)]

        acc = 0.0
        for j in range(p + 1):
            # I_j(k)
            S_j  = sum(pow_series[:j + 1])                   # Σ_{m=0}^j
            I_j  = math.factorial(j) / iλ**(j + 1) * (1 - np.exp(iλ) * S_j)
            acc += math.comb(p, j) * a**(p - j) * Δ**(j + 1) * I_j.real
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


    

if __name__ == "__main__":

    Ks    = [64]*4
    mu    = np.array([1.0, 0.5, -1.1, 1.3])
    Sigma = np.array([[1.0, 0.4, 0.3, 0.1],
                      [0.4, 1.0, 0.6, 0.5],
                      [0.3, 0.6, 1.0, 0.5],
                      [0.1, 0.5, 0.5, 1.0]])
    std   = np.sqrt(np.diag(Sigma))
    a     = mu - 7*std
    b     = mu + 7*std


    print("Building full COS tensor …")
    A = build_cos_tensor(Ks, a, b, mu, Sigma)
    print("Tensor shape:", A.shape)

    print("Compressing with TT-SVD …")
    t0 = time.time()
    cores, ranks = tt_svd(A, eps=1e-15)
    print(f"TT-SVD done in {time.time()-t0:.2f}s  –  ranks {ranks}")


    G0_mat = cores[0].reshape(Ks[0], -1)
    row_cores = {}
    for p in range(5):
        if p == 0:
            w = cos_moment_const(a[0], b[0], Ks[0])
        elif p == 1:
            w = cos_moment_mean(a[0], b[0], Ks[0])
        else:
            w = closed_form_moment_power(a[0], b[0], Ks[0], p)
        row_cores[p] = build_row_core(G0_mat, w)
    tilde_G0 = row_cores[0]

    # --- analytic conditional parameters ---
    Σ_yx     = Sigma[0,1:]
    Σ_xx_inv = np.linalg.inv(Sigma[1:,1:])
    μ_y, μ_x = mu[0], mu[1:]
    σ2_cond  = float(Sigma[0,0] - Σ_yx @ Σ_xx_inv @ Σ_yx)

    def gaussian_conditional_moments(x_feat):
        μ_c = μ_y + Σ_yx @ Σ_xx_inv @ (x_feat - μ_x)
        m1 = μ_c
        m2 = σ2_cond + μ_c**2
        m3 = μ_c**3 + 3*μ_c*σ2_cond
        m4 = μ_c**4 + 6*μ_c**2*σ2_cond + 3*σ2_cond**2
        return np.array([m1, m2, m3, m4])

    # --- test on random X samples ---
    rng   = np.random.default_rng(42)
    xs    = rng.normal(mu[1:], std[1:], size=(50,3))
    errs  = {p: [] for p in (1,2,3,4)}

    for x in xs:
        P    = feature_contraction(x, cores, a, b)
        norm = float((tilde_G0 @ P).squeeze())
        m_tt = {}
        for p in (1,2,3,4):
            num_p  = float((row_cores[p] @ P).squeeze())
            m_tt[p] = num_p / norm
        m_ex = gaussian_conditional_moments(x)
        for idx,p in enumerate((1,2,3,4)):
            errs[p].append(abs(m_tt[p] - m_ex[idx]))

    # --- report errors ---
    print("\nConditional raw-moment errors (TT vs. exact):")
    for p in (1,2,3,4):
        arr = np.array(errs[p])
        print(f" p={p}  max error = {arr.max():.3e}, avg error = {arr.mean():.3e}")
