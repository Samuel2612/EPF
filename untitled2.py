import time
import numpy as np
import tt

# ------- Problem -------
d = 10          # keep modest for sanity checks
n = 40
eps = 1e-8
kickrank = 2
nswp = 8
rf = 2
rng = np.random.default_rng(0)

# ------- Target function in "rectcross" (multi-index batch) API -------
def f_rect(I):
    # I is (m, d) with integer indices 0..n-1
    # return f(I) as (m,)
    return np.sin(I.sum(axis=1))

# ------- Same target function in "multifuncrs" (pointwise scalar) API -------
# Build d one-dimensional grids x_k(j) = j, j=0..n-1
# Each x_k is a TT *vector* representing the grid values along dim k.
# In ttpy, you provide them as length-n TT tensors per dimension via "tt.xfun"
# trick: start from the binary grid and linearly map to 0..n-1 if n is power of two.
# For general n, we make an explicit vector and lift it.
def one_dim_grid(n):
    # As a dense 1D numpy vector
    v = np.arange(n, dtype=float)
    # Turn into a TT tensor of shape (n,) with rank-1
    # (vector is represented as a 1D TT; in ttpy that's a 1-core tensor)
    return tt.tensor(v.reshape(n, 1))  # shape (n, 1) -> TT vector of size n

X_list = [one_dim_grid(n) for _ in range(d)]

def f_scalar(vals):
    """
    multifuncrs will call this with vals shaped (m, d),
    where vals[:,k] are samples from the k-th 1D grid.
    """
    return np.sin(vals.sum(axis=1))

# ------- Run rectcross on the same function -------
x0 = tt.rand(n, d, r=2)
t0 = time.time()
T_rect = tt.cross.rectcross.cross(f_rect, x0, nswp=nswp, kickrank=kickrank, rf=rf)
t_rect = time.time() - t0

# ------- Run multifuncrs on the wrapped version of the same function -------
# Note: y0 can help convergence a lot; start from ones.
t1 = time.time()
T_multi = tt.multifuncrs(X_list, f_scalar, eps, y0=tt.ones(n, d))
t_multi = time.time() - t1

# ------- Compare results on a small random subset of indices -------
def sample_and_eval(T, m=2000):
    I = np.column_stack([rng.integers(0, n, size=m) for _ in range(d)])
    # Evaluate TT at batch of indices using tt utility (fallback: one-by-one)
    # If your ttpy lacks a batch-eval helper, do a small m to keep it fast.
    vals = np.array([tt.get(T, I[k, :].tolist()) for k in range(m)])
    return I, vals

mcheck = 2000
I, v_rect = sample_and_eval(T_rect, mcheck)
_,  v_multi = sample_and_eval(T_multi, mcheck)
v_true = f_rect(I)

rmse_rect  = np.sqrt(np.mean((v_rect  - v_true)**2))
rmse_multi = np.sqrt(np.mean((v_multi - v_true)**2))

print(f"[rectcross]  time={t_rect:.2f}s  ranks={T_rect.rmax}  rmse(sample)={rmse_rect:.3e}")
print(f"[multifun]   time={t_multi:.2f}s  ranks={T_multi.rmax} rmse(sample)={rmse_multi:.3e}")

# Optional: compare TT norms and cross-agreement
diff_norm = (T_rect - T_multi).norm() / max(1.0, T_rect.norm())
print(f"Relative TT norm difference ||T_rect - T_multi|| / ||T_rect|| = {diff_norm:.3e}")