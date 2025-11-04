
import numpy as np
from time import perf_counter
from itertools import product
from collections import deque


def _pin_path(subs, K, dtype, warmup_N=1000):
    dummy = [np.empty((warmup_N, Kj), dtype=dtype) for Kj in K]
    _EINSUM_PATH, info = np.einsum_path(subs, *dummy, optimize='greedy')
    return _EINSUM_PATH, info


def _flood_fill_contiguous(mask):
    """
    Given a boolean ND-array `mask` (where True means the frequency already passes
    the power threshold), keep only the *lowest contiguous hyper‑volume* that is
    connected to the origin.  This is the rule proposed in the fastKDE / self‑
    consistent ECF filter paper.

    Parameters
    ----------
    mask : np.ndarray of bool
        Thresholded indicator array in k‑space, same shape as the empirical CF
        lattice.

    Returns
    -------
    connected : np.ndarray of bool
        Boolean array of the same shape with True only for the component that
        contains the 0‑frequency cell (index (0,…,0)).
    """
    if not mask[tuple([0] * mask.ndim)]:
        # If the origin itself is below threshold we return an all‑False mask.
        return np.zeros_like(mask, dtype=bool)

    connected = np.zeros_like(mask, dtype=bool)
    q = deque()
    q.append(tuple([0] * mask.ndim))
    connected[tuple([0] * mask.ndim)] = True

    # Neighbour offsets: Von‑Neumann neighbourhood (axis‑aligned)
    neighbours = [tuple(1 if i == axis else 0 for i in range(mask.ndim)) for axis in range(mask.ndim)]
    neighbours += [tuple(-1 if i == axis else 0 for i in range(mask.ndim)) for axis in range(mask.ndim)]

    shape = mask.shape
    while q:
        idx = q.popleft()
        for off in neighbours:
            nxt = tuple(i + o for i, o in zip(idx, off))
            # bounds check
            if any(j < 0 or j >= shape[k] for k, j in enumerate(nxt)):
                continue
            if mask[nxt] and not connected[nxt]:
                connected[nxt] = True
                q.append(nxt)
    return connected


def A_tensor_no_sign_blowup2(
    X,
    a,
    b,
    s,
    K,
    *,
    dtype=np.complex128,
    apply_frequency_filter=True,
    use_weighting=True,
):
    """
    Compute the multidimensional cosine coefficient tensor A_{k1…kd} using an
    empirical characteristic‑function (ECF) estimate, while optionally applying
    the self‑consistent *frequency filter* described in:

    
    """

    X = np.asarray(X)
    N, d = X.shape
    a, b, s, K = map(np.asarray, (a, b, s, K))
    alphas = np.pi * s / (b - a)

    # --------------------------------------------------
    # 1. Build G_j matrices (sign‑free trick)
    # --------------------------------------------------
    Gs = []
    for j in range(d):
        kj = np.arange(K[j])
        Ej_pos = (
            np.exp(1j * alphas[j] * X[:, j, None] * kj[None, :]).astype(dtype)
        )  # (N, K_j)

        # Phase factors for + and − signs
        P_plus = np.exp(-1j * a[j] * alphas[j] * kj).astype(dtype)  # (K_j,)
        if j == 0:  # first dim sign is fixed to +1
            Gj = Ej_pos * P_plus[None, :]
        else:
            P_minus = P_plus.conj()
            Gj = Ej_pos * P_plus[None, :] + Ej_pos.conj() * P_minus[None, :]

        Gs.append(Gj)

    # --------------------------------------------------
    # 2. Efficient einsum over n
    # --------------------------------------------------
    letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    in_subs = [f"n{letters[j]}" for j in range(d)]
    out_subs = "".join(letters[:d])
    subs = ",".join(in_subs) + "->" + out_subs

    path, info = _pin_path(subs, K, dtype)
    # print(info)  # Uncomment to see einsum optimisation details

    T = np.einsum(subs, *Gs, optimize=path) / X.shape[0]  # Empirical φ̂(t)

    # --------------------------------------------------
    # 3. Optional frequency filter
    # --------------------------------------------------
    if apply_frequency_filter:
        power = np.abs(T) ** 2
        Cmin2 = 4 * (N - 1) / N**2
        above_noise = power >= Cmin2

        # Keep only the component connected to the origin (d>=2)
        if d >= 2:
            mask = _flood_fill_contiguous(above_noise)
        else:
            mask = above_noise  # 1‑D: contiguous by construction

        if use_weighting:
            # κ̂(t) weighting
            kappa = np.zeros_like(T, dtype=float)
            sel = mask
            denom = N**2 * power[sel]
            inner = 1.0 - 4.0 * (N - 1) / denom
            inner = np.clip(inner, 0.0, 1.0)
            kappa_val = N / (2 * (N - 1)) * (1.0 + np.sqrt(inner))
            kappa[sel] = kappa_val
            T_filtered = T * kappa
        else:
            T_filtered = T * mask
    else:
        T_filtered = T

    # --------------------------------------------------
    # 4. Final cosine coefficients
    # --------------------------------------------------
    const = 2.0 * np.prod(1.0 / (b - a))
    A = const * np.real(T_filtered)
    return A


if __name__ == "__main__":
    # Quick self‑test on a 4‑D Gaussian
    mu = np.array([1.0, 0.5, -1.1, 1.3])
    Sigma = np.array(
        [
            [1.0, 0.4, 0.3, 0.1],
            [0.4, 1.0, 0.6, 0.5],
            [0.3, 0.6, 1.0, 0.5],
            [0.1, 0.5, 0.5, 1.0],
        ]
    )
    d = len(mu)
    std = np.sqrt(np.diag(Sigma))
    a = mu - 7 * std
    b = mu + 7 * std
    s = np.ones_like(mu)
    K = np.array([64] * d)

    N = 1_000
    rng = np.random.default_rng(0)
    X = rng.multivariate_normal(mean=mu, cov=Sigma, size=N)

    t0 = perf_counter()
    A = A_tensor_no_sign_blowup(
        X,
        a,
        b,
        s,
        K,
        apply_frequency_filter=True,
        use_weighting=True,
    )
    t1 = perf_counter()
    print("A shape:", A.shape, "time:", f"{t1 - t0:.3f}s")
