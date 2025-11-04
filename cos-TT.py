import torch
import torchtt as tntt
import math
import numpy as np



def chf_gauss(u, mu, Sigma):
    """
    φ_X(u) for X ~ N(mu,Σ)       u: (... , d)  torch tensor (float64)
    """
    
    expo = 1j * (u @ mu) - 0.5 * torch.sum(u * (u @ Sigma), dim=-1)
    return torch.exp(expo)

def build_cos_coeffs(mu, Sigma, a, b, N):
    d      = len(mu)
    grids  = torch.meshgrid(*[torch.arange(N) for _ in range(d)], indexing='ij')
    k      = torch.stack(grids, dim=-1)

    u      = k * math.pi / (b - a)
    char   = chf_gauss(u.reshape(-1, d), mu, Sigma).reshape(k.shape[:-1])
    phase  = torch.exp(-1j * (u @ a))

    pref   = torch.prod(2.0 / (b - a))
    A      = pref * torch.real(char * phase)
    return A.to(dtype=torch.float64)


def cos_tt_tensor(mu, Sigma, a, b, N, eps=1e-10):
    """
    returns   torchtt.TT   (cores are torch tensors on the same device)
    """
    A_full = build_cos_coeffs(mu, Sigma, a, b, N)
    A_tt   = tntt.TT(A_full, eps=eps)          # TT-SVD & rounding
    return A_tt

def cos_basis(x, a, b, N):
    k = torch.arange(N)
    basis = torch.cos(k * math.pi * (x - a) / (b - a))
    basis[0] *= 0.5                              # <-- apply the ½ here
    return basis

def pdf_tt(x, A_tt, a, b):
    """
    Evaluate COS-TT pdf at a single point x  (1-D torch tensor, len d)
    """
    cores  = A_tt.cores                       # list of TT cores (torch tensors)


    vec = torch.ones(1)  # r₀ = 1
    for i, G in enumerate(cores):                    # G: (r_{i-1}, n_i, r_i)
        basis = cos_basis(x[i], a[i], b[i], G.shape[1])
        tmp   = torch.tensordot(vec, G, dims=([0], [0]))   # (n_i, r_i)
        vec   = torch.tensordot(basis, tmp, dims=([0], [0]))  # (r_i,)
    return vec.item()

# ----------------------------------------------------------
# 5) quick demo
# ----------------------------------------------------------
if __name__ == "__main__":
    

    # ----- parameters ----------------------------------------------------
    d = 4
    mu     = torch.zeros(d)
    Sigma  = torch.tensor([[1.0, 0.4, 0.3, 0.1],
                           [0.4, 1.0, 0.2, 0.5],
                           [0.3, 0.2, 1.0, 0.5],
                           [0.1, 0.5, 0.5, 1.0]])
    std    = torch.sqrt(torch.diag(Sigma))
    a, b   = (mu -8*std), (mu + 8*std)                # integration box
    N      = 64                                      # COS modes

    # ----- build TT ------------------------------------------------------
    print("Building TT …")
    A_tt   = cos_tt_tensor(mu, Sigma, a, b, N, eps=1e-10)
    print("dimensions (n_i):", A_tt.N)
    print("TT ranks        :", A_tt.R)

    # ----- test evaluation ----------------------------------------------
    x      = torch.tensor([0.2, 0.2, 0.3, 0.0])
    fx_tt  = pdf_tt(x, A_tt, a, b)

    # analytic MVN pdf for comparison
    invS   = torch.linalg.inv(Sigma)
    detS   = torch.linalg.det(Sigma)
    norm_c = 1.0 / torch.sqrt( (2*math.pi)**d * detS )
    diff   = x - mu
    fx_ex  = norm_c * torch.exp(-0.5 * diff @ invS @ diff)

    print(f"\nCOS-TT pdf  : {fx_tt:15.8e}")
    print(f"Exact pdf   : {fx_ex.item():15.8e}")
