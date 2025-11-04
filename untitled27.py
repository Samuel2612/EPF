import numpy as np, numba as nb, math
from functools import partial


def build_A(a, b, K, phi, *, dtype=np.float64):
    a = np.asarray(a, dtype=dtype)
    b = np.asarray(b, dtype=dtype)
    K = np.asarray(K, dtype=np.int64)

    shape  = tuple(K + 1)
    scale  = 2.0 * np.prod(1.0 / (b - a))
    tensor = np.empty(shape, dtype=dtype)

    _fill_A_parallel(a, b, K, scale, phi, tensor)   
    return tensor


@nb.njit(parallel=True)
def _fill_A_parallel(a, b, K, scale, phi, A):
    n   = a.size
    Ki  = K + 1
    tot = 1
    for m in Ki:
        tot *= m

    A1d = A.ravel()                 

    stride = np.empty(n, np.int64)
    stride[n-1] = 1
    for i in range(n - 2, -1, -1):
        stride[i] = stride[i + 1] * Ki[i + 1]

    for idx in nb.prange(tot):       
        k_vec   = _unravel(idx, stride)          
        A1d[idx] = _A_entry(k_vec, a, b, scale, phi)



@nb.njit(inline="always")
def _A_entry(k, a, b, scale, phi):
    n      = k.size
    acc    = 0.0
    two_nm1 = 1 << (n - 1)            # 2^(n-1) masks for s₁,…,s_{n-1}

    # --- i = 0 is hard-wired to  +1  -----------------------------
    s0      = 1
    denom0  = b[0] - a[0]
    omega0  = np.pi * s0 * k[0] / denom0
    phase0  = -np.pi * s0 * k[0] * a[0] / denom0
    # --------------------------------------------------------------

    for mask in range(two_nm1):
        omega = np.empty(n, a.dtype)
        phase = phase0                # start from the i=0 contribution
        omega[0] = omega0

        # iterate over the remaining axes (shifted index!)
        for i in range(1, n):
            s       = 1 if (mask >> (i - 1)) & 1 else -1
            denom   = b[i] - a[i]
            omega[i] = np.pi * s * k[i] / denom
            phase   += -np.pi * s * k[i] * a[i] / denom

        acc += (np.exp(1j * phase) * phi(omega)).real

    return scale * acc

@nb.njit                    
def _unravel(idx, stride):
    n     = stride.size
    k_vec = np.empty(n, np.int64)
    rem   = idx
    for d in range(n):
        q        = rem // stride[d]
        k_vec[d] = q
        rem     -= q * stride[d]
    return k_vec
# ----------------------------------------------------------------------
@nb.njit
def phi_std_normal(omega):
    """n-dimensional standard-normal characteristic function."""
    return np.exp(-0.5 * np.dot(omega, omega))

@nb.njit
def chf_gaussian(omega, mu, Sigma):
    """Characteristic function φ(t) of 𝒩(μ,Σ), broadcast-compatible."""
    lead, n = omega.shape[:-1], omega.shape[-1]
    omegaf = omega.reshape(-1, n)                     # (N, n)
    phase = 1j * omegaf @ mu                      # (N,)
    quad = np.einsum("ij,jk,ik->i", omegaf, Sigma, omegaf)
    return np.exp(phase - 0.5 * quad).reshape(lead)



if __name__ == "__main__":
    n = 5

    K = np.full(n, 50)               # tensor 33⁴ ≈ 1.18 M entries
    
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
    
                
    invS = np.linalg.inv(Sigma)
    detS = np.linalg.det(Sigma)
    norm_c = 1.0 / np.sqrt((2*np.pi)**n * detS)
    diff = x - mu
    fx_ex = norm_c * np.exp(-0.5 * diff @ invS @ diff)


    g = partial(chf_gaussian, mu=mu, Sigma=Sigma)

    A = build_A(a, b, K, phi_std_normal)
    print(A.shape, A.dtype, type(A))   # (33, 33, 33, 33) float64 <class 'numpy.ndarray'>
