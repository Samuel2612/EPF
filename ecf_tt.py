
import numpy as np
from typing import List, Tuple, Optional

# -----------------------------


def _geom_vector(base: complex, K: int, scale: complex = 1.0, dtype=np.complex128) -> np.ndarray:
    """Return v[k] = scale * base**k for k=0..K-1 using a stable recurrence.
    Only one exponential needed (for base)."""
    v = np.empty(K, dtype=dtype)
    v[0] = scale
    if K > 1:
        v[1:] = base
        np.cumprod(v[1:], out=v[1:])  # v[1:] = base, base^2, ..., base^(K-1)
        v[1:] *= scale
    return v


def tt_rank1_from_vectors(vectors: List[np.ndarray]) -> List[np.ndarray]:
    """Build a rank-1 TT from a list of vectors (each length Kj).
    Returns TT cores as [G1, G2, ..., Gd] with ranks all = 1."""
    d = len(vectors)
    cores: List[np.ndarray] = []
    # First core: (K1, 1)
    cores.append(vectors[0][:, None])
    # Middle cores: (1, Kj, 1)
    for j in range(1, d-1):
        Gj = np.zeros((1, vectors[j].shape[0], 1), dtype=vectors[j].dtype)
        Gj[0, :, 0] = vectors[j]
        cores.append(Gj)
    if d > 1:
        # Last core: (1, Kd)
        cores.append(vectors[-1][None, :])
    return cores


def tt_add(A: List[np.ndarray], B: List[np.ndarray]) -> List[np.ndarray]:
    """Return TT for A + B via block-diagonal stacking on ranks (no rounding).
    Shapes:
      A[0] (K1, r1a),  B[0] (K1, r1b)       -> [K1, r1a+r1b]
      A[j] (r_{j-1}^A, Kj, r_j^A),
      B[j] (r_{j-1}^B, Kj, r_j^B)           -> block diag on both rank dims
      A[-1] (r_{d-1}^A, Kd), B[-1] (r_{d-1}^B, Kd) -> [r_{d-1}^A+r_{d-1}^B, Kd]
    """
    d = len(A)
    C: List[np.ndarray] = []

    # First core
    K1 = A[0].shape[0]
    r1a = A[0].shape[1]
    r1b = B[0].shape[1]
    C0 = np.zeros((K1, r1a + r1b), dtype=A[0].dtype)
    C0[:, :r1a] = A[0]
    C0[:, r1a:] = B[0]
    C.append(C0)

    # Middle cores
    for j in range(1, d-1):
        ra1, Kj, ra2 = A[j].shape
        rb1, KjB, rb2 = B[j].shape
        assert Kj == KjB
        Cj = np.zeros((ra1 + rb1, Kj, ra2 + rb2), dtype=A[j].dtype)
        Cj[:ra1, :, :ra2] = A[j]
        Cj[ra1:, :, ra2:] = B[j]
        C.append(Cj)

    # Last core
    rdma, Kd = A[-1].shape
    rdmb, KdB = B[-1].shape
    assert Kd == KdB
    Cd = np.zeros((rdma + rdmb, Kd), dtype=A[-1].dtype)
    Cd[:rdma, :] = A[-1]
    Cd[rdma:, :] = B[-1]
    C.append(Cd)
    return C


def _qr_reshape_left(core: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int]]:
    """Reshape a core to (left_rank*Kj, right_rank) for QR, then return Q, R and the shape tuple."""
    if core.ndim == 2:
        # first core: (K1, r1)
        K, r = core.shape
        M = core.reshape(K, r)
        Q, R = np.linalg.qr(M, mode='reduced')  # (K, rQ) (rQ, r)
        return Q, R, (1, K, Q.shape[1])  # marker for first core
    else:
        rl, K, rr = core.shape
        M = core.reshape(rl * K, rr)
        Q, R = np.linalg.qr(M, mode='reduced')  # (rl*K, rQ) (rQ, rr)
        return Q, R, (rl, K, Q.shape[1])


def _apply_R_to_next(next_core: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Multiply R into the next core along its left-rank dimension."""
    if next_core.ndim == 2:
        # last core (rd-1, Kd)
        rl, K = next_core.shape
        assert R.shape[1] == rl
        out = (R @ next_core)  # (new_rl, K)
        return out
    else:
        rl, K, rr = next_core.shape
        assert R.shape[1] == rl
        # tensordot over left-rank (axis 1 of R with axis 0 of next_core)
        out = np.tensordot(R, next_core, axes=(1, 0))  # (new_rl, K, rr)
        return out


def _svd_truncate(mat: np.ndarray, eps: float, max_rank: Optional[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """SVD with Frobenius-norm based truncation. Returns (U, S, Vh, r_new)."""
    U, S, Vh = np.linalg.svd(mat, full_matrices=False)
    if eps <= 0:
        r_new = S.size
    else:
        # keep smallest r such that tail energy <= eps^2
        sq = (S**2)[::-1]
        cumsum = np.cumsum(sq)
        tail = cumsum[::-1]
        r_new = S.size
        for k in range(S.size):
            if tail[k] <= eps**2:
                r_new = k + 1
                break
    if max_rank is not None:
        r_new = min(r_new, max_rank)
    U = U[:, :r_new]
    S = S[:r_new]
    Vh = Vh[:r_new, :]
    return U, S, Vh, r_new


def tt_round(tt: List[np.ndarray], eps: float = 1e-8, max_rank: Optional[int] = None) -> List[np.ndarray]:
    """TT-rounding (orthogonalize left, then right-to-left SVD truncation).
    The per-link tolerance is distributed as eps / sqrt(d-1)."""
    d = len(tt)
    if d == 1:
        return tt
    cores = [c.copy() for c in tt]

    # Left orthogonalization via QR
    for j in range(d - 1):
        Q, R, shp = _qr_reshape_left(cores[j])
        if cores[j].ndim == 2:
            K, _ = cores[j].shape
            cores[j] = Q  # (K, r_new)
        else:
            rl, K, rQ = shp
            cores[j] = Q.reshape(rl, K, rQ)
        cores[j + 1] = _apply_R_to_next(cores[j + 1], R)

    # Right-to-left truncation
    link_eps = eps / np.sqrt(max(d - 1, 1))
    for j in range(d - 1, 0, -1):
        core = cores[j]
        if core.ndim == 2:
            rl, K = core.shape
            M = core.reshape(rl, K)  # (rl, K)
            U, S, Vh, r_new = _svd_truncate(M, eps=link_eps, max_rank=max_rank)
            B = U @ np.diag(S)  # (rl, r_new)
            prev = cores[j - 1]
            if prev.ndim == 2:
                Kprev, rprev = prev.shape
                assert rprev == rl
                cores[j - 1] = (prev @ B)  # (Kprev, r_new)
            else:
                rlprev, Kprev, rprev = prev.shape
                assert rprev == rl
                cores[j - 1] = np.tensordot(prev, B, axes=(2, 0))  # (rlprev, Kprev, r_new)
            cores[j] = Vh  # (r_new, K)
        else:
            rl, K, rr = core.shape
            M = core.reshape(rl, K * rr)  # (rl, K*rr)
            U, S, Vh, r_new = _svd_truncate(M, eps=link_eps, max_rank=max_rank)
            B = U @ np.diag(S)  # (rl, r_new)
            prev = cores[j - 1]
            if prev.ndim == 2:
                Kprev, rprev = prev.shape
                assert rprev == rl
                cores[j - 1] = (prev @ B)  # (Kprev, r_new)
            else:
                rlprev, Kprev, rprev = prev.shape
                assert rprev == rl
                cores[j - 1] = np.tensordot(prev, B, axes=(2, 0))  # (rlprev, Kprev, r_new)
            cores[j] = Vh.reshape(r_new, K, rr)
    return cores


def tt_contract_element(tt: List[np.ndarray], multi_index: Tuple[int, ...]) -> complex:
    """Evaluate a single tensor entry given its multi-index."""
    d = len(tt)
    acc = tt[0][multi_index[0], :]
    for j in range(1, d - 1):
        acc = acc @ tt[j][:, multi_index[j], :]
    if d > 1:
        acc = acc @ tt[-1][:, multi_index[-1]]
    return acc


def tt_full(tt: List[np.ndarray]) -> np.ndarray:
    """Materialize the full tensor (only for small problems!)."""
    d = len(tt)
    Ks = [tt[0].shape[0]] + [tt[j].shape[1] for j in range(1, d - 1)] + [tt[-1].shape[1]]
    out = np.empty(Ks, dtype=tt[0].dtype)
    it = np.nditer(np.zeros(Ks, dtype=np.int8), flags=['multi_index'])
    while not it.finished:
        out[it.multi_index] = tt_contract_element(tt, it.multi_index)
        it.iternext()
    return out


# ---------- ECF -> TT (streaming sum of rank‑1 + rounding) ----------

def ecf_tt(
    X: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    s: np.ndarray,
    K: np.ndarray,
    *,
    batch: int = 128,
    eps: float = 1e-8,
    max_rank: Optional[int] = None,
    dtype=np.complex128,
) -> List[np.ndarray]:
    """Build a TT decomposition of the empirical characteristic function on a rectangular frequency grid.
    We *never* build the full tensor. We stream rank‑1 outer products and round aggressively.

    ECF on grid:
        phi[k1,...,kd] = (1/N) sum_{n=1}^N prod_{j=1}^d exp(i * alpha_j * x_{n,j} * k_j),
        alpha_j = pi * s_j / (b_j - a_j),   k_j = 0..K_j-1

    Args
    ----
    X : (N, d) samples
    a, b, s, K : length-d arrays as in COS method
    batch : accumulate this many samples before rounding
    eps : global rounding tolerance (Frobenius relative; per-link tolerance is eps/sqrt(d-1))
    max_rank : optional cap on TT ranks
    dtype : complex dtype

    Returns
    -------
    tt : list of TT cores
    """
    X = np.asarray(X)
    N, d = X.shape
    a, b, s, K = map(np.asarray, (a, b, s, K))
    alphas = np.pi * s / (b - a)
    invN = 1.0 / N
    tt_sum: Optional[List[np.ndarray]] = None

    def add_sample(n: int, tt_accum: Optional[List[np.ndarray]]) -> List[np.ndarray]:
        vectors = []
        for j in range(d):
            base = np.exp(1j * alphas[j] * X[n, j]).astype(dtype, copy=False)
            v = _geom_vector(base, int(K[j]), scale=invN, dtype=dtype)  # v[k] = invN * base**k
            vectors.append(v)
        tt_rank1 = tt_rank1_from_vectors(vectors)
        if tt_accum is None:
            return tt_rank1
        else:
            return tt_add(tt_accum, tt_rank1)

    n = 0
    while n < N:
        m = min(batch, N - n)
        if tt_sum is None:
            tt_sum = add_sample(n, None)
            start = n + 1
        else:
            start = n
        for idx in range(start, n + m):
            tt_sum = tt_add(tt_sum, add_sample(idx, None))
        tt_sum = tt_round(tt_sum, eps=eps, max_rank=max_rank)
        n += m

    assert tt_sum is not None
    return tt_sum


# ---------- Analytic Gaussian CF on a grid (for testing) ----------

def true_gaussian_cf_on_grid(a, b, s, K, mu, Sigma, dtype=np.complex128):
    d = len(K)
    alphas = np.pi * s / (b - a)
    grids = np.meshgrid(*[alphas[j] * np.arange(int(K[j])) for j in range(d)], indexing='ij')
    t_stack = np.stack(grids, axis=-1)  # (..., d)

    quad = np.einsum('...i,ij,...j->...', t_stack, Sigma, t_stack, optimize=True)   # t^T Σ t
    lin  = np.tensordot(t_stack, mu, axes=([-1], [0]))                              # t^T μ
    return np.exp(1j * lin - 0.5 * quad).astype(dtype)


def demo_small():
    rng = np.random.default_rng(0)
    d = 4
    mu    = np.zeros(d)
    Sigma = np.array([[1.0, 0.4, 0.3, 0.1],
                      [0.4, 1.0, 0.6, 0.5],
                      [0.3, 0.6, 1.0, 0.5],
                      [0.1, 0.5, 0.5, 1.0]])
    std   = np.sqrt(np.diag(Sigma))
    a     = mu - 7 * std
    b     = mu + 7 * std
    s     = np.ones_like(mu)
    K     = np.array([16]*d)

    N = 100
    X = rng.multivariate_normal(mean=mu, cov=Sigma, size=N)

    tt = ecf_tt(X, a, b, s, K, batch=256, eps=1e-8, max_rank=128, dtype=np.complex128)
    phi_true = true_gaussian_cf_on_grid(a, b, s, K, mu, Sigma, dtype=np.complex128)
    phi_tt   = tt_full(tt)

    abs_err = np.abs(phi_tt - phi_true)
    rmse = float(np.sqrt((abs_err**2).mean()))
    max_err = float(abs_err.max())

    ranks = [tt[0].shape[1]] + [c.shape[2] for c in tt[1:-1]] + [tt[-1].shape[0]]
    return {
        "shape": tuple(int(k) for k in K),
        "ranks": ranks,
        "rmse": rmse,
        "max_err": max_err,
    }


if __name__ == "__main__":
    stats = demo_small()
    print("ECF TT demo (small)")
    print("  grid shape:", stats["shape"])
    print("  TT ranks  :", stats["ranks"])
    print(f"  RMSE      : {stats['rmse']:.3e}")
    print(f"  Max error : {stats['max_err']:.3e}")
