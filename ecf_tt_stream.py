import math
import numpy as np
from typing import List, Tuple
from ecf_einsum import ecf_grid_einsum
from time import perf_counter

TT = Tuple[List[np.ndarray], List[int]]  # (cores, ranks)


# ----------------------------
# TT helpers (tuple semantics)
# ----------------------------
def as_tt(x) -> TT:
    """Normalize input to (cores, ranks)."""
    if isinstance(x, tuple) and len(x) == 2 and isinstance(x[0], list) and isinstance(x[1], list):
        return x
    # Assume x is a list of cores
    cores = x
    ranks = [cores[0].shape[0]] + [G.shape[2] for G in cores]
    return cores, ranks

def tt_copy(A: TT) -> TT:
    cores, ranks = as_tt(A)
    return [c.copy() for c in cores], ranks.copy()

def tt_cores(A: TT) -> List[np.ndarray]:
    return as_tt(A)[0]

def tt_ranks(A: TT) -> List[int]:
    return as_tt(A)[1]

def tt_validate(A: TT):
    cores, ranks = as_tt(A)
    d = len(cores)
    assert len(ranks) == d + 1 and ranks[0] == 1 and ranks[-1] == 1, "Invalid ranks."
    for k, G in enumerate(cores):
        rkm1, nk, rk = G.shape
        assert rkm1 == ranks[k] and rk == ranks[k+1], f"Rank mismatch at core {k}."


# ----------------------------
# TT-SVD from full tensor
# ----------------------------
def tt_svd_(tensor: np.ndarray, eps: float) -> TT:
    """Return (cores, ranks) s.t. ‖T − TT‖_F / ‖T‖_F ≤ eps."""
    T = tensor.copy()
    dims = T.shape
    d = len(dims)
    if d == 0:
        raise ValueError("Input must be at least 1-D.")
    norm2 = np.linalg.norm(T) ** 2
    denom = max(d - 1, 1)
    thr = (eps / math.sqrt(denom)) ** 2 * norm2

    cores: List[np.ndarray] = []
    ranks: List[int] = [1]

    unfold = T
    for k in range(d - 1):
        m = ranks[-1] * dims[k]
        unfold = unfold.reshape(m, -1)
        U, S, Vh = np.linalg.svd(unfold, full_matrices=False)
        tail = np.cumsum(S[::-1] ** 2)[::-1]
        mask = tail <= thr
        r = int(np.argmax(mask)) + 1 if mask.any() else len(S)
        r = max(1, r)
        cores.append(U[:, :r].reshape(ranks[-1], dims[k], r))
        ranks.append(r)
        unfold = (S[:r, None] * Vh[:r])

    cores.append(unfold.reshape(ranks[-1], dims[-1], 1))
    ranks.append(1)
    return cores, ranks


# ----------------------------
# Basic TT constructors & ops
# ----------------------------
def rank1_tt_from_vectors(v_list: List[np.ndarray]) -> TT:
    """Create rank-1 TT from mode vectors v_k (shape (n_k,))."""
    dtype = np.result_type(*[v.dtype for v in v_list] + [np.complex128])
    cores = [v.astype(dtype, copy=False).reshape(1, v.size, 1) for v in v_list]
    ranks = [1] * (len(v_list) + 1)
    return cores, ranks

def tt_scale(A: TT, alpha: complex) -> TT:
    cores, ranks = tt_copy(A)
    cores[0] = cores[0] * alpha
    return cores, ranks

def tt_add(A: TT, B: TT) -> TT:
    """Block-diagonal merge of cores; returns (cores, ranks)."""
    Acores, Aranks = as_tt(A)
    Bcores, Branks = as_tt(B)
    d = len(Acores)
    assert d == len(Bcores), "Dimension mismatch in tt_add."
    # first core concat along last rank
    A1, B1 = Acores[0], Bcores[0]
    assert A1.shape[0] == B1.shape[0] == 1 and A1.shape[1] == B1.shape[1], "Mode size mismatch."
    out = [np.concatenate([A1, B1], axis=2)]
    # interiors block-diag
    for k in range(1, d - 1):
        Ak, Bk = Acores[k], Bcores[k]
        assert Ak.shape[1] == Bk.shape[1], f"Mode {k} size mismatch."
        rA0, n, rA1 = Ak.shape
        rB0, _, rB1 = Bk.shape
        Ck = np.zeros((rA0 + rB0, n, rA1 + rB1), dtype=np.result_type(Ak, Bk))
        Ck[:rA0, :, :rA1] = Ak
        Ck[rA0:, :, rA1:] = Bk
        out.append(Ck)
    # last core stack along first rank
    Ad, Bd = Acores[-1], Bcores[-1]
    assert Ad.shape[1] == Bd.shape[1] and Ad.shape[2] == Bd.shape[2] == 1, "Last core mismatch."
    out.append(np.concatenate([Ad, Bd], axis=0))
    # ranks add (except exterior 1s)
    ranks = [1] + [Aranks[k] + Branks[k] for k in range(1, d)] + [1]
    return out, ranks

def tt_sub(A: TT, B: TT) -> TT:
    return tt_add(A, tt_scale(B, -1.0))

def tt_dot(A: TT, B: TT) -> complex:
    """<A,B> with conjugation on A; works for complex TTs."""
    Acores, _ = as_tt(A)
    Bcores, _ = as_tt(B)
    V = np.array([[1.0 + 0j]], dtype=np.result_type(Acores[0], Bcores[0]))
    for Ak, Bk in zip(Acores, Bcores):
        rA0, n, rA1 = Ak.shape
        rB0, _, rB1 = Bk.shape
        V = V.reshape(rB0, rA0)
        Vnext = np.zeros((rB1, rA1), dtype=np.result_type(Ak, Bk))
        for i in range(n):
            Bi = Bk[:, i, :]      # rB0 x rB1
            Ai = Ak[:, i, :]      # rA0 x rA1
            Vnext += Bi.T @ V @ Ai.conj()
        V = Vnext.reshape(1, -1)
    return V.squeeze()

def tt_norm2(A: TT) -> float:
    return tt_dot(A, A).real


# ----------------------------
# Rounding / truncation
# ----------------------------
def svd_truncate(mat: np.ndarray, delta: float, rmax: int | None = None):
    U, S, Vh = np.linalg.svd(mat, full_matrices=False)
    if delta and delta > 0:
        tail2 = np.cumsum(S[::-1]**2)[::-1]
        mask = tail2 <= (delta**2)
        r = int(np.argmax(mask)) + 1 if mask.any() else len(S)
    else:
        r = len(S)
    r = max(1, r)
    if rmax is not None:
        r = min(r, int(rmax))
    return U[:, :r], S[:r], Vh[:r, :]

def tt_truncate(A: TT, eps: float = 1e-10, rmax: int | None = None) -> TT:
    """Left-to-right rounding with global relative tolerance eps."""
    cores, ranks = as_tt(A)
    d = len(cores)
    nrm = math.sqrt(tt_norm2((cores, ranks)))
    if nrm == 0.0:
        return tt_copy((cores, ranks))
    delta = eps * nrm / math.sqrt(max(d - 1, 1))
    cores = [c.copy() for c in cores]
    new_ranks = [cores[0].shape[0]]
    for k in range(d - 1):
        rkm1, nk, rk = cores[k].shape
        M = cores[k].reshape(rkm1 * nk, rk)
        U, S, Vh = svd_truncate(M, delta, rmax=rmax)
        rnew = S.size
        cores[k] = U.reshape(rkm1, nk, rnew)
        SV = (S[:, None] * Vh)  # (rnew, rk)
        Gnext = cores[k + 1]
        rk_curr, nk1, rk1 = Gnext.shape
        assert rk_curr == rk
        Gnext_mat = Gnext.reshape(rk, nk1 * rk1)
        Gnext_mat = SV @ Gnext_mat
        cores[k + 1] = Gnext_mat.reshape(rnew, nk1, rk1)
        new_ranks.append(rnew)
    new_ranks.append(cores[-1].shape[2])  # should be 1
    return cores, new_ranks


# ----------------------------
# ECF in TT form (streaming)
# ----------------------------
def ecf_tt_streaming(X, a, b, s, K, eps=1e-8, round_every=2, rmax=None, dtype=np.complex128) -> TT:
    """
    Φ = (1/N) * sum_n  ⊗_{k=1}^d exp(i * α_k * x_{n,k} * j),  j=0..K_k-1.
    Streaming sum with periodic TT truncation; returns (cores, ranks).
    """
    X = np.asarray(X)
    N, d = X.shape
    a = np.asarray(a); b = np.asarray(b); s = np.asarray(s); K = np.asarray(K, dtype=int)

    alphas = np.pi * s / (b - a)
    grids = [np.arange(K[k]) for k in range(d)]

    TT_acc: TT | None = None
    since_round = 0

    for n in range(N):
        v_list = [np.exp(1j * alphas[k] * X[n, k] * grids[k]).astype(dtype, copy=False) for k in range(d)]
        rank1 = rank1_tt_from_vectors(v_list)  # tuple
        if TT_acc is None:
            TT_acc = rank1
        else:
            TT_acc = tt_add(TT_acc, rank1)
        since_round += 1
        if since_round >= round_every:
            TT_acc = tt_truncate(TT_acc, eps=eps, rmax=rmax)
            since_round = 0

    if TT_acc is None:
        # no samples -> zero TT
        zero_cores = [np.zeros((1, K[k], 1), dtype=dtype) for k in range(d)]
        return zero_cores, [1] * (d + 1)

    if since_round > 0:
        TT_acc = tt_truncate(TT_acc, eps=eps, rmax=rmax)

    TT_acc = tt_scale(TT_acc, 1.0 / float(N))
    return TT_acc


# ----------------------------
# Demo / test
# ----------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    d = 6
    K = np.array([4] * d)
    mu = np.zeros(d); std = np.ones(d)
    a = mu - 6 * std; b = mu + 6 * std; s = np.ones_like(mu)
    N = 1000
    cov = 0.5 * np.ones((d, d)) + 0.5 * np.eye(d)
    X = rng.multivariate_normal(mu, cov, size=N)

    # Streaming ECF TT
    t = perf_counter()
    TT_stream = ecf_tt_streaming(X, a, b, s, K, eps=1e-10, round_every=128, rmax=256)
    t1 = perf_counter()
    print(f"TT ranks (stream):  {tt_ranks(TT_stream)}, time = {t1 - t}")

    # Full tensor -> TT via TT-SVD
    t2 = perf_counter()
    A_full = ecf_grid_einsum(X, a, b, s, K, dtype=np.complex128)
    TT_svd = tt_svd_(A_full, 1e-10)
    t3 = perf_counter()
    print(f"TT ranks (svd):  {tt_ranks(TT_svd)}, time = {t3 - t2}")

    # Error ||Φ_stream − Φ_svd||_F^2
    diff = tt_sub(TT_stream, TT_svd)
    print("||Φ_stream − Φ_svd||_F^2:", tt_norm2(diff))
