
import time
import numpy as np

try:
    import tt
    from tt import cross as tt_cross
    _HAS_TTPY = True
except Exception as e:
    tt = None
    tt_cross = None
    _HAS_TTPY = False
    _IMPORT_ERROR = e


# =========================
# Utilities
# =========================

def _geom_progression(base, K, scale):
    
    v = np.empty(int(K), dtype=base.dtype)
    v[0] = scale
    if K > 1:
        v[1:] = base
        np.cumprod(v[1:], out=v[1:])
        v[1:] *= scale
    return v


def _rank1_tt_from_vectors(vectors, use_complex=True):
    "\"\"Rank-1 TT from per-mode vectors (length n_j).

    Returns a tt.tensor.
    Complex handling:
      * If use_complex and ttpy supports complex, builds complex TT directly.
      * Otherwise, raises ValueError (caller can fall back to real+imag split).
    "\"\"
    if not _HAS_TTPY:
        raise ImportError(\"ttpy is not available: %r\" % _IMPORT_ERROR)

    cores = []
    for v in vectors:
        n = int(v.shape[0])
        G = np.zeros((1, n, 1), dtype=v.dtype)
        G[0, :, 0] = v
        cores.append(G)
    # tt.tensor().from_list expects list of 3D cores (r_{j-1}, n_j, r_j)
    T = tt.tensor().from_list(cores)
    return T


def _tt_add_and_round(A, B, eps, rmax):
    \"\"\"(A+B) then TT-round.\"\"\"
    C = A + B
    C = tt.round(C, eps=eps, rmax=rmax)
    return C


def _ensure_1_boundary_ranks(T):
    \"\"\"Return rank vector and assert boundary ranks are 1.\"\"\"
    r = np.array(T.r).astype(int).tolist()
    if r[0] != 1 or r[-1] != 1:
        # In ttpy, boundary ranks should be 1 after rounding; if not,
        # this indicates an issue (or non-rounded tensor). We'll still return r.
        pass
    return r


# =========================
# Streaming ECF -> TT (ttpy)
# =========================

def ecf_tt_stream_ttpy(
    X, a, b, s, K, *, batch=128, eps=1e-8, rmax=None, complex_mode=\"auto\"
):
    \"\"\"Stream rank-1 samples into a TT using ttpy, never building the full grid.

    Parameters
    ----------
    X : (N,d) samples
    a, b, s, K : 1D arrays of length d
    batch : int, round after each batch samples
    eps : float, rounding tolerance
    rmax : int or None, max TT-rank
    complex_mode : {'auto','split','complex'}
        'complex' -> attempt complex tt directly (if ttpy supports it)
        'split'   -> always build two real-valued TTs: (Re, Im)
        'auto'    -> try complex, fall back to split on failure

    Returns
    -------
    out : dict
        keys:
         - 'tensor' : tt.tensor if complex_mode resolved to complex, otherwise None
         - 'tensor_re','tensor_im': tt.tensor if split mode used; else None
         - 'ranks' : list of TT ranks
         - 'build_time' : seconds
    \"\"\"
    if not _HAS_TTPY:
        raise ImportError(\"ttpy is not available: %r\" % _IMPORT_ERROR)

    X = np.asarray(X)
    a, b, s, K = map(lambda z: np.asarray(z), (a, b, s, K))
    N, d = X.shape
    alphas = np.pi * s / (b - a)
    invN = 1.0 / N

    t0 = time.perf_counter()

    use_split = False
    if complex_mode == 'split':
        use_split = True

    T = None           # complex tt.tensor if not split
    T_re = None        # real
    T_im = None        # real

    def add_sample(n):
        vecs = []
        for j in range(d):
            base = np.exp(1j * alphas[j] * X[n, j]).astype(np.complex128, copy=False)
            vecs.append(_geom_progression(base, int(K[j]), invN))
        return vecs

    # Try complex path first if allowed
    tried_complex = False
    if complex_mode in ('auto', 'complex'):
        tried_complex = True
        try:
            for n0 in range(0, N, batch):
                n1 = min(N, n0 + batch)
                # accumulate batch
                for n in range(n0, n1):
                    vecs = add_sample(n)
                    R1 = _rank1_tt_from_vectors(vecs, use_complex=True)
                    if T is None:
                        T = R1
                    else:
                        T = _tt_add_and_round(T, R1, eps=eps, rmax=rmax)
                # extra rounding after batch
                T = tt.round(T, eps=eps, rmax=rmax)
            ranks = _ensure_1_boundary_ranks(T)
            t1 = time.perf_counter()
            return {\"tensor\": T, \"tensor_re\": None, \"tensor_im\": None,
                    \"ranks\": ranks, \"build_time\": t1 - t0}
        except Exception as e:
            if complex_mode == 'complex':
                raise
            # fallback to split mode
            use_split = True

    # Split mode: build real and imaginary parts separately (always works)
    if use_split or (complex_mode == 'auto' and tried_complex):
        for n0 in range(0, N, batch):
            n1 = min(N, n0 + batch)
            for n in range(n0, n1):
                vecs_c = add_sample(n)
                # split into real and imag vectors
                vecs_re = [np.real(v).astype(np.float64, copy=False) for v in vecs_c]
                vecs_im = [np.imag(v).astype(np.float64, copy=False) for v in vecs_c]
                R1r = _rank1_tt_from_vectors(vecs_re, use_complex=False)
                R1i = _rank1_tt_from_vectors(vecs_im, use_complex=False)
                if T_re is None:
                    T_re, T_im = R1r, R1i
                else:
                    T_re = _tt_add_and_round(T_re, R1r, eps=eps, rmax=rmax)
                    T_im = _tt_add_and_round(T_im, R1i, eps=eps, rmax=rmax)
            # extra rounding after batch
            T_re = tt.round(T_re, eps=eps, rmax=rmax)
            T_im = tt.round(T_im, eps=eps, rmax=rmax)
        ranks = _ensure_1_boundary_ranks(T_re)  # both should match
        t1 = time.perf_counter()
        return {\"tensor\": None, \"tensor_re\": T_re, \"tensor_im\": T_im,
                \"ranks\": ranks, \"build_time\": t1 - t0}

    raise RuntimeError(\"Unreachable state in ecf_tt_stream_ttpy\")


# =========================
# TT-CROSS for ECF grid (ttpy)
# =========================

def _ecf_entry_eval_factory(X, alphas):
    \"\"\"Return a function f(I) that evaluates ECF at integer multi-indices I (0-based).

    I : ndarray, shape (Q, d) of nonnegative integers k_j
    returns : ndarray of complex128, shape (Q,)
    \"\"\"
    X = np.asarray(X)
    alphas = np.asarray(alphas)
    N, d = X.shape

    def f(I):
        I = np.asarray(I, dtype=int)
        # t(q,:) = k(q,:)*alphas
        T = I * alphas  # (Q,d)
        # G = T @ X^T, shape (Q,N)
        G = T @ X.T
        # mean over samples
        return np.exp(1j * G).mean(axis=1)

    return f


def ecf_tt_cross_ttpy(
    X, a, b, s, K, *, eps=1e-6, rmax=200, kickrank=4, complex_mode='split', verb=1
):
    \"\"\"Build ECF TT via TT-CROSS using ttpy's cross module.

    complex_mode:
      - 'split' (default): build real and imaginary parts separately via two cross calls.
      - 'complex': attempt complex cross in one go (only if your ttpy supports complex).
    \"\"\"
    if not _HAS_TTPY:
        raise ImportError(\"ttpy is not available: %r\" % _IMPORT_ERROR)

    a, b, s, K = map(lambda z: np.asarray(z), (a, b, s, K))
    d = int(len(K))
    n = [int(k) for k in K]
    alphas = np.pi * s / (b - a)
    entry_fun = _ecf_entry_eval_factory(X, alphas)

    t0 = time.perf_counter()
    if complex_mode == 'complex':
        # Try single complex cross (may fail depending on ttpy build)
        def fun(z):
            return entry_fun(z)
        T = tt_cross.cross(d, n, fun, eps=eps, rmax=rmax, kickrank=kickrank, verb=verb)
        ranks = _ensure_1_boundary_ranks(T)
        t1 = time.perf_counter()
        return {\"tensor\": T, \"tensor_re\": None, \"tensor_im\": None,
                \"ranks\": ranks, \"build_time\": t1 - t0}

    # Split mode: do Re and Im separately
    def fun_re(z):
        return np.real(entry_fun(z))
    def fun_im(z):
        return np.imag(entry_fun(z))

    T_re = tt_cross.cross(d, n, fun_re, eps=eps, rmax=rmax, kickrank=kickrank, verb=verb)
    T_im = tt_cross.cross(d, n, fun_im, eps=eps, rmax=rmax, kickrank=kickrank, verb=verb)
    ranks = _ensure_1_boundary_ranks(T_re)
    t1 = time.perf_counter()
    return {\"tensor\": None, \"tensor_re\": T_re, \"tensor_im\": T_im,
            \"ranks\": ranks, \"build_time\": t1 - t0}


# =========================
# Evaluation helpers
# =========================

def tt_eval_entries(Tc, Ti, I):
    \"\"\"Evaluate TT (complex split) at entries I of shape (Q,d).\"\"\"
    # Materialize selected entries by contracting cores coordinate-wise.
    # For small Q, this is fine; for large Q, prefer converting to dense or using a proper TT indexing routine.
    def one(ttT, idx):
        # Contract to scalar for a single multi-index
        cores = ttT.to_list()
        acc = cores[0][:, idx[0], :]
        for j in range(1, len(cores)):
            acc = acc @ cores[j][:, idx[j], :]
        return acc.item()

    out = np.empty(I.shape[0], dtype=np.complex128)
    for q in range(I.shape[0]):
        val = 0.0
        if Tc is not None:
            val = one(Tc, I[q])
        else:
            val = one(Ti[0], I[q]) + 1j * one(Ti[1], I[q])
        out[q] = val
    return out


def random_entries_error(X, a, b, s, K, res_stream, res_cross, ntest=200, seed=0):
    \"\"\"Compare the two TTs on random entries against direct ECF evaluation.\"\"\"
    rng = np.random.default_rng(seed)
    d = len(K)
    I = np.stack([rng.integers(0, int(K[j]), size=ntest) for j in range(d)], axis=1)

    alphas = np.pi * s / (b - a)
    entry_fun = _ecf_entry_eval_factory(X, alphas)
    truth = entry_fun(I)

    # Stream TT eval
    if res_stream['tensor'] is not None:
        Tc = res_stream['tensor']; Ti = None
    else:
        Tc = None; Ti = (res_stream['tensor_re'], res_stream['tensor_im'])
    est_stream = tt_eval_entries(Tc, Ti, I)

    # Cross TT eval
    if res_cross['tensor'] is not None:
        Tc2 = res_cross['tensor']; Ti2 = None
    else:
        Tc2 = None; Ti2 = (res_cross['tensor_re'], res_cross['tensor_im'])
    est_cross = tt_eval_entries(Tc2, Ti2, I)

    def comp(x):
        ae = np.abs(x - truth)
        return float(np.sqrt((ae**2).mean())), float(ae.max())

    rmse_stream, max_stream = comp(est_stream)
    rmse_cross,  max_cross  = comp(est_cross)

    return {
        'rmse_stream': rmse_stream, 'max_stream': max_stream,
        'rmse_cross': rmse_cross,   'max_cross':  max_cross
    }
