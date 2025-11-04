import numpy as np
import matplotlib.pyplot as plt


def charfunc_gaussian(u, mu, Sigma):
    """Characteristic function ϕ(u) of N(mu, Sigma)."""
    u = np.atleast_2d(u)        # shape (m, d)
    expo = 1j * (u @ mu) - 0.5 * np.sum(u * (u @ Sigma), axis=1)
    return np.exp(expo)         # shape (m,)

def build_cos_coeffs(mu, Sigma, a, b, N):
    """
    Build full COS‑coefficient tensor A_{k1,…,kd} (size N^d) for
    a multivariate Gaussian on [a_i,b_i].
    """
    d = len(mu)
    grids = [np.arange(N) for _ in range(d)]
    A = np.zeros([N] * d, dtype=np.float64)

    # constants
    two_over_l = 2.0 / (b - a)          # vector length d
    scaling = np.prod(two_over_l)

    # precompute inverse and determinant for later pdf check
    for index in np.ndindex(*A.shape):
        k_vec = np.array(index)
        # gamma factor: 1 for k>0, 0.5 for k=0 (per dimension)
        gamma = np.prod(np.where(k_vec == 0, 0.5, 1.0))

        u_k = k_vec * np.pi / (b - a)
        phase = np.exp(-1j * (u_k @ a))
        coef = scaling * gamma * np.real(charfunc_gaussian(u_k, mu, Sigma) * phase)
        A[index] = coef
    return A

def tt_svd(tensor, eps=1e-12, max_rank=None):
    """
    Compress full tensor to Tensor‑Train using TT‑SVD.
    Returns list of cores G_i with shapes (r_{i-1}, n_i, r_i).
    """
    d = tensor.ndim
    dims = tensor.shape
    cores = []
    ranks = [1]

    unfold = tensor.copy()
    for k in range(d - 1):
        unfold = unfold.reshape(ranks[-1] * dims[k], -1)
        U, S, Vh = np.linalg.svd(unfold, full_matrices=False)

        # determine rank truncation
        if max_rank is None:
            # drop small singular values (simple absolute threshold)
            r = np.sum(S > eps)
        else:
            r = min(max_rank, np.sum(S > eps))
        U = U[:, :r]
        S = S[:r]
        Vh = Vh[:r, :]

        core = U.reshape(ranks[-1], dims[k], r)
        cores.append(core)
        unfold = (S[:, None] * Vh)
        ranks.append(r)

    cores.append(unfold.reshape(ranks[-1], dims[-1], 1))
    return cores

def tt_evaluate(cores, basis_vals):
    """
    Evaluate TT expansion ∑_k A_k ∏_i φ_{k_i}(x_i)
    at a single point given pre‑computed basis values per dimension.
    basis_vals[i] is a vector length n_i with cos‑basis values for dim i.
    """
    vec = np.array([1.0])
    for G, b in zip(cores, basis_vals):
        # contract over current dimension
        # vec shape      (r_{i-1},)
        # G shape        (r_{i-1}, n_i, r_i)
        # result tmp     (n_i, r_i)
        tmp = np.tensordot(vec, G, axes=[(0), (0)])
        # contract n_i with basis
        vec = np.tensordot(b, tmp, axes=[(0), (0)])  # shape (r_i,)
    return vec[0]

def multivariate_gaussian_pdf(x, mu, Sigma_inv, detSigma):
    d = len(mu)
    diff = x - mu
    exponent = -0.5 * diff @ Sigma_inv @ diff
    norm_const = 1.0 / np.sqrt((2 * np.pi) ** d * detSigma)
    return norm_const * np.exp(exponent)

# ----- Example parameters -----
d = 2
mu = np.array([0.0, 0.0])
Sigma = np.array([[1.0, 0.3],
                  [0.3, 2.0]])
Sigma_inv = np.linalg.inv(Sigma)
detSigma = np.linalg.det(Sigma)

# integration domain [a,b] (8 std on each axis)
L = 8
std = np.sqrt(np.diag(Sigma))
a = mu - L * std
b = mu + L * std

N = 32  # COS coefficients per dimension

# ----- Build coefficient tensor and compress -----
A_full = build_cos_coeffs(mu, Sigma, a, b, N)
cores = tt_svd(A_full, eps=1e-12)
ranks = [core.shape[0] for core in cores] + [1]

print("TT ranks (including last 1):", ranks)

# ----- Evaluate on a grid -----
nx = ny = 50
x1_vals = np.linspace(a[0], b[0], nx)
x2_vals = np.linspace(a[1], b[1], ny)

# Pre‑compute cosine bases for all grid points
basis_dim1 = np.cos(np.outer(x1_vals - a[0], np.arange(N)) * np.pi / (b[0]-a[0]))
basis_dim2 = np.cos(np.outer(x2_vals - a[1], np.arange(N)) * np.pi / (b[1]-a[1]))

approx_pdf = np.zeros((nx, ny))
exact_pdf  = np.zeros((nx, ny))

for i, x1 in enumerate(x1_vals):
    for j, x2 in enumerate(x2_vals):
        # basis for each dimension at (x1,x2)
        basis_vals = [basis_dim1[i, :], basis_dim2[j, :]]
        approx_pdf[i, j] = tt_evaluate(cores, basis_vals)
        exact_pdf[i, j]  = multivariate_gaussian_pdf(
            np.array([x1, x2]), mu, Sigma_inv, detSigma)

# ----- Error metrics -----
max_err  = np.max(np.abs(approx_pdf - exact_pdf))
l2_err   = np.sqrt(np.mean((approx_pdf - exact_pdf) ** 2))
print(f"Max abs error: {max_err:.3e}")
print(f"RMSE:         {l2_err:.3e}")

# ----- Plots -----
plt.figure()
plt.title("Exact PDF")
plt.imshow(exact_pdf, origin='lower', extent=[a[0], b[0], a[1], b[1]], aspect='auto')
plt.colorbar()

plt.figure()
plt.title("Approximate PDF (COS + TT)")
plt.imshow(approx_pdf, origin='lower', extent=[a[0], b[0], a[1], b[1]], aspect='auto')
plt.colorbar()

plt.figure()
plt.title("Absolute Error")
plt.imshow(np.abs(approx_pdf - exact_pdf), origin='lower', extent=[a[0], b[0], a[1], b[1]], aspect='auto')
plt.colorbar()

plt.show()
