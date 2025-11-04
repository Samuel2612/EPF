from __future__ import annotations
import numpy as np
import itertools
from numpy.linalg import qr, norm, inv, pinv


def _maxvol(A, r):
    """
    Greedy maximum-volume row selection.
    """
    A = A.copy()
    m, _ = A.shape
    rows = []
    remaining_rows = list(range(m))

    for _ in range(r):
        # pick the remaining row with largest ℓ²-norm
        i_max = remaining_rows[int(np.argmax(np.linalg.norm(A[remaining_rows], axis=1)))]
        rows.append(i_max)
        remaining_rows.remove(i_max)

        # project the chosen row out of the rest
        v = A[i_max]
        if np.abs(v).max() > 1e-12:
            proj = (A[remaining_rows] @ v) / (v @ v)
            A[remaining_rows] -= np.outer(proj, v)

    sub = A[rows]                       # selected sub-matrix
    A_inv = inv(sub) if np.linalg.matrix_rank(sub) == r else pinv(sub)
    return np.asarray(rows, dtype=int), A_inv



def _left_right_step(
    eval_f,               
    k,
    shape, ranks, row_idx, col_idx
):
    """One left-to-right sweep: update row indices for core k+1."""
    n_k          = shape[k]
    r_k, r_k1    = ranks[k], ranks[k + 1]
    # Build unfolding of core–k using current skeleton indices
    rows, cols   = row_idx[k], col_idx[k]
    mat_rows     = []
    for i in rows:
        for j in range(n_k):
            for l in cols:
                idx = i + (j,) + l
                mat_rows.append(eval_f(idx))
    G = np.asarray(mat_rows, dtype=float).reshape(r_k * n_k, r_k1)

    # QR → maxvol → new row indices
    Q, _         = qr(G, mode="reduced")
    sel, _       = _maxvol(Q, r_k1)
    new_rows     = [rows[i // n_k] + (i % n_k,) for i in sel]
    return new_rows


def _right_left_step(
    eval_f,
    k,              # core index (1-based from the right)
    shape, ranks, row_idx, col_idx
):
    """One right-to-left sweep: update column indices for core k-1 and
    return the updated (k-1)-th TT core."""
    n_km1        = shape[k - 1]
    r_km1, r_k   = ranks[k - 1], ranks[k]
    rows         = row_idx[k - 1]
    cols         = col_idx[k - 1]

    mat_cols = []
    for i in rows:
        for j in cols:
            for l in range(n_km1):
                idx = i + (l,) + j
                mat_cols.append(eval_f(idx))
    G = np.asarray(mat_cols, dtype=float).reshape(r_km1, n_km1 * r_k).T  # (n_{k-1} r_k) × r_{k-1}

    # QR → maxvol → new column indices
    Q, _         = qr(G, mode="reduced")
    sel, Qinv    = _maxvol(Q, r_km1)
    new_cols     = [(sel_i // r_k,) + cols[sel_i % r_k] for sel_i in sel]

    # Build (k-1)-th core from Q @ Qinv
    C            = (Q @ Qinv).T.reshape(r_km1, n_km1, r_k)
    return new_cols, C


# --------------------------------------------------------------------------- #
# MAIN ALGORITHM                                                              #
# --------------------------------------------------------------------------- #
def tt_cross(
    f,                         # black-box evaluator: f(indices_tuple) → float
    shape,    # full tensor mode sizes
    ranks,    # TT ranks (either int or sequence of length d+1, with r₀ = r_d = 1)
    tol = 1e-6,  # stopping tolerance (on sampled error, not a guarantee)
    max_sweeps  = 20,    # outer iterations
    seed  = None):
    """
    TT-CROSS approximation of a (possibly huge) d-dimensional tensor accessible
    *only* through point-wise evaluations of `f`.

    Returns
    -------
    cores : list of length d
            Each core Gᵏ has shape  (r_{k-1}, n_k, r_k).
    """
    rng    = np.random.default_rng(seed)
    d      = len(shape)
    if isinstance(ranks, int):
        ranks = [1] + [ranks] * (d - 1) + [1]
    if len(ranks) != d + 1 or ranks[0] != 1 or ranks[-1] != 1:
        raise ValueError("`ranks` must have length d+1 with r0 = rd = 1")

    col_idx = []
    for k in range(d - 1):
        cols = set()
        while len(cols) < ranks[k + 1]:
            cols.add(tuple(rng.integers(shape[j]) for j in range(k + 1, d)))
        col_idx.append(list(cols))
    col_idx.append([()])                     # dummy for last core

   
    cores   = [rng.standard_normal((ranks[k], shape[k], ranks[k + 1]))
               for k in range(d)]


    row_idx = [[()]]  # left indices for core-0 (empty tuple)
    for sweep in range(max_sweeps):
       
        for k in range(d - 1):
            new_rows = _left_right_step(
                f, k, shape, ranks, row_idx, col_idx
            )
            row_idx.append(new_rows)


        col_idx[-1] = [()]                    # rightmost col-index set is empty
        for k in range(d, 1, -1):
            new_cols, Ckm1 = _right_left_step(
                f, k, shape, ranks, row_idx, col_idx
            )
            col_idx[k - 2] = new_cols
            cores[k - 1]   = Ckm1

        
        idx0 = (slice(None),) + tuple(zip(*col_idx[0]))
        # Explicit evaluation of a *small* slice to build G⁰
        slab      = np.array([f((i,)+col) for i in range(shape[0]) for col in zip(*col_idx[0])],
                              dtype=float).reshape(shape[0], 1, ranks[1])
        cores[0]  = slab.transpose(1, 0, 2)

        if sweep > 0:
            sample_err = 0.0
            for k in range(10):
                idx_rand = tuple(rng.integers(n) for n in shape)
                true_val = f(idx_rand)
                # quick TT interpolation of the same entry
                v = cores[0][0, idx_rand[0]]
                for core, j in zip(cores[1:], idx_rand[1:]):
                    v = v @ core[:, j, :]
                sample_err += (true_val - v[0])**2
            if np.sqrt(sample_err / 10) < tol:
                break

        # prepare for next sweep
        row_idx = [[()]]             # keep only the leftmost skeleton

    return cores



def tt_cross_from_dense(
    tensor,
    ranks,
    tol = 1e-6,
    max_sweeps = 20,
    seed = None,
):
    """When the *full* tensor is already in memory, wrap it so that the same
    implementation still works."""
    f = lambda ind: tensor[ind]      # noqa: E731
    return tt_cross(f, tensor.shape, ranks, tol, max_sweeps, seed)
