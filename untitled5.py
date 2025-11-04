import time
import numpy as np
import tt


# --------------------------
# Data generation (given)
# --------------------------
def simulate_mvt(n: int, mu: np.ndarray, Sigma_cov: np.ndarray, nu: float, rng=None) -> np.ndarray:
    """
    Simulate from a d-dim multivariate Student-t with mean mu, covariance Sigma_cov, dof nu.
    Uses standard Normal + scaled-chi2 mixture; requires nu > 2.
    """
    if rng is None:
        rng = np.random.default_rng(7)
    d = mu.size
    # Convert covariance to *scale* for t: Sigma_scale = ((nu-2)/nu) * Sigma_cov
    Sigma_scale = ((nu - 2.0) / nu) * Sigma_cov
    L = np.linalg.cholesky(Sigma_scale)

    z = rng.standard_normal((n, d))
    u = rng.chisquare(df=nu, size=n)  # chi^2_nu
    w = np.sqrt(u / nu).reshape(-1, 1)  # scale per sample
    x = mu + (z @ L.T) / w
    return x


# --------------------------
# Helpers
# --------------------------
def make_tt_vector_from_1d(v: np.ndarray):
    """
    Create a rank-1 TT 'vector' from a 1D numpy array v.
    Works across ttpy versions (tries tt.vector first, falls back to tensor trick).
    """
    v = np.asarray(v, dtype=float)
    try:
        return tt.vector(v)
    except Exception:
        # Fallback: interpret v as a 1D TT by giving a (n,1) shaped array
        return tt.tensor(v.reshape(v.size, 1))


def build_freq_grids(d: int, n_per_dim: int, W: float):
    """
    Per-dimension symmetric frequency grids: [-W, W], linspace n_per_dim each.
    Returns:
      grids_1d: list of 1D numpy arrays (length n_per_dim each)
      X_list:   list of TT 'vectors' (per-dim grid) for multifuncrs
    """
    grids_1d = [np.linspace(-W, W, n_per_dim) for _ in range(d)]
    X_list = [make_tt_vector_from_1d(g) for g in grids_1d]
    return grids_1d, X_list


def ecf_callable(X: np.ndarray, return_part: str = "real", max_chunk: int = 5000):
    """
    Build a callable f(vals) for multifuncrs/rectcross that evaluates ECF at frequency points.
    X: (N, d) samples
    return_part: "real" or "imag" (since TTs are real, we fit parts separately)
    max_chunk: split the sum over samples to control memory
    """
    N, d = X.shape
    X_chunks = np.array_split(X, max(1, int(np.ceil(N / max_chunk))), axis=0)

    def f(vals: np.ndarray) -> np.ndarray:
        # vals: (m, d) frequency points
        # ECF(ω) = (1/N) * sum_j exp(i * ω^T x_j)
        # Accumulate in complex then return requested part
        acc = np.zeros(vals.shape[0], dtype=np.complex128)
        for chunk in X_chunks:
            acc += np.exp(1j * (vals @ chunk.T)).sum(axis=1)
        acc /= N
        if return_part == "real":
            return np.real(acc)
        else:
            return np.imag(acc)

    return f


def index_to_coords(I: np.ndarray, grids_1d):
    """
    Map integer multi-indices I (m,d) -> frequency coordinates vals (m,d)
    using per-dimension 1D grids.
    """
    m, d = I.shape
    V = np.empty_like(I, dtype=float)
    for k in range(d):
        V[:, k] = grids_1d[k][I[:, k]]
    return V


def sample_rmse(T, grids_1d, f_true, m_samples=2000, rng=None):
    """
    Sample m_samples random grid points, compare TT value vs true function.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    d = len(grids_1d)
    n = [len(g) for g in grids_1d]
    I = np.column_stack([rng.integers(0, n[k], size=m_samples) for k in range(d)])
    vals = index_to_coords(I, grids_1d)
    y_true = f_true(vals)

    # Evaluate TT at these multi-indices
    y_tt = np.array([tt.get(T, I[j, :].tolist()) for j in range(m_samples)], dtype=float)

    rmse = np.sqrt(np.mean((y_tt - y_true) ** 2))
    return rmse


# --------------------------
# Benchmark parameters
# --------------------------
d = 5
N_samples = 1000        # ECF samples; raise/lower based on your machine
nu = 5.0                 # DoF (>2)
rng = np.random.default_rng(42)
mu = np.zeros(d)
# A mild-correlated covariance
A = rng.standard_normal((d, d))
Sigma = A @ A.T + 0.5 * np.eye(d)  # SPD

# Frequency grid
n_per_dim = 33            # mode size per dim (TT won't form full grid)
W = 6.0                   # frequency half-width per dim

# TT settings
eps = 1e-6                # target accuracy for multifuncrs
nswp = 8                  # sweeps for rectcross
kickrank = 2
rf = 2
r0 = 2                    # initial TT rank for rectcross
mcheck = 2000             # RMSE sampling points


# --------------------------
# Generate data
# --------------------------
X = simulate_mvt(N_samples, mu, Sigma, nu, rng=rng)  # (N, d)

# Build frequency grids
grids_1d, X_list = build_freq_grids(d, n_per_dim, W)

# ECF callables (real and imag)
f_real = ecf_callable(X, "real", max_chunk=4000)
f_imag = ecf_callable(X, "imag", max_chunk=4000)


# --------------------------
# multifuncrs (real)
# --------------------------
t0 = time.perf_counter()
T_re_multi = tt.multifuncrs(X_list, f_real, eps, y0=tt.ones(n_per_dim, d))
t1 = time.perf_counter()
time_re_multi = t1 - t0
rmse_re_multi = sample_rmse(T_re_multi, grids_1d, f_real, m_samples=mcheck, rng=rng)

# multifuncrs (imag)
t0 = time.perf_counter()
T_im_multi = tt.multifuncrs(X_list, f_imag, eps, y0=tt.zeros(n_per_dim, d))
t1 = time.perf_counter()
time_im_multi = t1 - t0
rmse_im_multi = sample_rmse(T_im_multi, grids_1d, f_imag, m_samples=mcheck, rng=rng)


# --------------------------
# rectcross (real)
# --------------------------
def f_rect_real(I):
    vals = index_to_coords(I, grids_1d)  # (m, d)
    return f_real(vals)

x0 = tt.rand(n_per_dim, d, r0)
t0 = time.perf_counter()
T_re_rect = tt.cross.rectcross.cross(f_rect_real, x0, nswp=nswp, kickrank=kickrank, rf=rf)
t1 = time.perf_counter()
time_re_rect = t1 - t0
rmse_re_rect = sample_rmse(T_re_rect, grids_1d, f_real, m_samples=mcheck, rng=rng)

# rectcross (imag)
def f_rect_imag(I):
    vals = index_to_coords(I, grids_1d)
    return f_imag(vals)

x0 = tt.rand(n_per_dim, d, r0)
t0 = time.perf_counter()
T_im_rect = tt.cross.rectcross.cross(f_rect_imag, x0, nswp=nswp, kickrank=kickrank, rf=rf)
t1 = time.perf_counter()
time_im_rect = t1 - t0
rmse_im_rect = sample_rmse(T_im_rect, grids_1d, f_imag, m_samples=mcheck, rng=rng)


# --------------------------
# Report
# --------------------------
def rmax(T):  # max TT-rank helper
    try:
        return T.rmax
    except Exception:
        # older ttpy: ranks() returns list
        return max(T.ranks())

print("\n=== TT ECF on 5D Student-t (grid {}^{} in [-{:.1f},{:.1f}]^d) ===".format(n_per_dim, d, W, W))
print("Samples N = {}, DoF nu = {:.1f}".format(N_samples, nu))
print("\n-- multifuncrs --")
print(" Re: time = {:.2f}s, rmax = {}, RMSE(sample) = {:.3e}".format(time_re_multi, rmax(T_re_multi), rmse_re_multi))
print(" Im: time = {:.2f}s, rmax = {}, RMSE(sample) = {:.3e}".format(time_im_multi, rmax(T_im_multi), rmse_im_multi))
print("\n-- rectcross --")
print(" Re: time = {:.2f}s, rmax = {}, RMSE(sample) = {:.3e}".format(time_re_rect, rmax(T_re_rect), rmse_re_rect))
print(" Im: time = {:.2f}s, rmax = {}, RMSE(sample) = {:.3e}".format(time_im_rect, rmax(T_im_rect), rmse_im_rect))

# Optional cross-agreement diagnostic
rel_diff_re = (T_re_multi - T_re_rect).norm() / max(1.0, T_re_multi.norm())
rel_diff_im = (T_im_multi - T_im_rect).norm() / max(1.0, T_im_multi.norm())
print("\nRelative TT norm difference (Re): {:.3e}".format(rel_diff_re))
print("Relative TT norm difference (Im): {:.3e}".format(rel_diff_im))