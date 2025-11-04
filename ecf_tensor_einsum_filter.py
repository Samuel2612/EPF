import numpy as np
from time import perf_counter
from itertools import product

def _pin_path(subs, K, dtype, warmup_N=1000):
    dummy = [np.empty((warmup_N, Kj), dtype=dtype) for Kj in K]
    _EINSUM_PATH, info = np.einsum_path(subs, *dummy, optimize='greedy')
    return _EINSUM_PATH, info

def A_tensor_no_sign_blowup(X, a, b, s, K, dtype=np.complex128):
    """
    Compute A_{k1...kd} without storing CFs for every sign combination.
    Uses one N×K_j matrix per dim and combines +/- via conjugation.
    """
    X = np.asarray(X)
    N, d = X.shape
    a, b, s, K = map(np.asarray, (a, b, s, K))
    alphas = np.pi * s / (b - a)

    # Build G_j matrices
    Gs = []
    for j in range(d):
        kj = np.arange(K[j])
        Ej_pos = np.exp(1j * alphas[j] * X[:, j, None] * kj[None, :]).astype(dtype)  # (N, K_j)

        # phase for + and - signs
        P_plus  = np.exp(-1j * a[j] * alphas[j] * kj).astype(dtype)    # shape (K_j,)
        if j == 0:   # first dim sign is fixed to +1
            Gj = Ej_pos * P_plus[None, :]
        else:
            P_minus = P_plus.conj()
            Gj = Ej_pos * P_plus[None, :] + Ej_pos.conj() * P_minus[None, :]

        Gs.append(Gj)

    # einsum over n, keep all k-axes
    letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    in_subs  = [f"n{letters[j]}" for j in range(d)]
    out_subs = "".join(letters[:d])
    subs = ",".join(in_subs) + "->" + out_subs

    # Pin a good path once (optional)
    path, info = _pin_path(subs, K, dtype)
    print(info)

    T = np.einsum(subs, *Gs, optimize=path) / N  # complex tensor, shape K1..Kd
    const = 2 * np.prod(1.0 / (b - a))
    A = const * np.real(T)
    return A


# -- Frequency‑filtered cosine coefficient estimator --
def A_tensor_frequency_filtered(X, a, b, s, K, dtype=np.complex128,
                                contiguous=True, num_hypervolumes=1):
    """
    Estimate the multidimensional cosine coefficients with a frequency‑domain
    filter based on the fastKDE method.  Small‑magnitude empirical
    characteristic function (ECF) values are suppressed to reduce noise, and
    the optimal transform kernel from Bernacchia & Pigolotti (2011) is used
    to weight the retained frequencies.  See the accompanying paper for
    details.

    Parameters
    ----------
    X : array_like, shape (N, d)
        Input sample points.
    a, b : array_like, shape (d,)
        Lower and upper integration bounds per dimension.
    s : array_like, shape (d,)
        Scaling factors for the cosine basis.
    K : array_like, shape (d,)
        Number of coefficients per dimension.
    dtype : complex type, optional
        Intermediate complex datatype.
    contiguous : bool, optional
        If True, keep only the lowest contiguous hypervolume of above‑threshold
        frequencies; otherwise use all above‑threshold frequencies.
    num_hypervolumes : int, optional
        Number of contiguous hypervolumes to retain (if contiguous=True).

    Returns
    -------
    A : ndarray
        Filtered real cosine coefficients of shape K.
    """
    X = np.asarray(X)
    N, d = X.shape
    a, b, s, K = map(np.asarray, (a, b, s, K))
    alphas = np.pi * s / (b - a)

    # Precompute exponentials and phase factors
    E_pos = []
    P_plus = []
    P_minus = []
    for j in range(d):
        kj = np.arange(K[j])
        # complex exponentials e^{i α_j x_j k_j}
        Ej_pos = np.exp(1j * alphas[j] * X[:, j, None] * kj[None, :]).astype(dtype)
        phase_plus  = np.exp(-1j * alphas[j] * a[j] * kj).astype(dtype)
        phase_minus = phase_plus.conj()
        E_pos.append(Ej_pos)
        P_plus.append(phase_plus)
        P_minus.append(phase_minus)

    # Construct G matrices for sign‑summed tensor
    G_sum = []
    for j in range(d):
        if j == 0:
            Gj = E_pos[j] * P_plus[j][None, :]
        else:
            Gj = E_pos[j] * P_plus[j][None, :] + E_pos[j].conj() * P_minus[j][None, :]
        G_sum.append(Gj)

    # Einstein summation specification
    letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    in_subs  = [f"n{letters[j]}" for j in range(d)]
    out_subs = "".join(letters[:d])
    subs = ",".join(in_subs) + "->" + out_subs

    # Pin contraction path
    path, _ = _pin_path(subs, K, dtype)

    # Compute raw summed ECF tensor
    T_sum = np.einsum(subs, *G_sum, optimize=path) / N

    # Compute positive‑frequency ECF tensor for |C(t)|^2
    G_pos = []
    for j in range(d):
        Gp = E_pos[j] * P_plus[j][None, :]
        G_pos.append(Gp)
    T_pos = np.einsum(subs, *G_pos, optimize=path) / N
    C_abs_sq = np.abs(T_pos)**2
    C_min = 4.0 * (N - 1) / (N * N)
    above = C_abs_sq >= C_min

    # Optionally restrict to contiguous hypervolumes
    if contiguous:
        shape = C_abs_sq.shape
        size = C_abs_sq.size
        above_flat = above.ravel()
        visited = np.zeros(size, dtype=bool)
        labels = np.zeros(size, dtype=int)
        # compute strides for flat indexing
        strides = [1]
        for sz in reversed(shape[1:]):
            strides.insert(0, strides[0] * sz)
        strides = np.array(strides, dtype=int)
        def flat_to_multi(idx):
            coords = []
            for s_j in strides:
                q, idx = divmod(idx, s_j)
                coords.append(q)
            return coords
        def multi_to_flat(coords):
            idx = 0
            for c, s_j in zip(coords, strides):
                idx += c * s_j
            return idx
        label_counter = 0
        for i in range(size):
            if not above_flat[i] or visited[i]:
                continue
            label_counter += 1
            stack = [i]
            visited[i] = True
            labels[i] = label_counter
            while stack:
                cur = stack.pop()
                coord = flat_to_multi(cur)
                for dim in range(d):
                    # neighbour in +1
                    if coord[dim] + 1 < shape[dim]:
                        nb = coord.copy(); nb[dim] += 1
                        nb_idx = multi_to_flat(nb)
                        if above_flat[nb_idx] and not visited[nb_idx]:
                            visited[nb_idx] = True
                            labels[nb_idx] = label_counter
                            stack.append(nb_idx)
                    # neighbour in -1
                    if coord[dim] - 1 >= 0:
                        nb = coord.copy(); nb[dim] -= 1
                        nb_idx = multi_to_flat(nb)
                        if above_flat[nb_idx] and not visited[nb_idx]:
                            visited[nb_idx] = True
                            labels[nb_idx] = label_counter
                            stack.append(nb_idx)
        # Determine clusters by distance to origin
        if num_hypervolumes >= 1:
            # compute centroid distances
            cluster_coords_sum = {}
            cluster_counts = {}
            it = np.nditer(labels.reshape(shape), flags=['multi_index'])
            for lbl in it:
                l = int(lbl)
                if l == 0:
                    continue
                if l not in cluster_counts:
                    cluster_counts[l] = 0
                    cluster_coords_sum[l] = np.zeros(d, dtype=float)
                cluster_counts[l] += 1
                cluster_coords_sum[l] += np.array(it.multi_index)
            centroids = {}
            distances = []
            for l in cluster_counts:
                centroids[l] = cluster_coords_sum[l] / cluster_counts[l]
                distances.append((np.linalg.norm(centroids[l]), l))
            distances.sort(key=lambda x: x[0])
            allowed = set([lbl for (_, lbl) in distances[:num_hypervolumes]])
            # rebuild above mask
            new_above_flat = np.zeros(size, dtype=bool)
            for idx in range(size):
                if labels[idx] in allowed:
                    new_above_flat[idx] = True
            above = new_above_flat.reshape(shape)

    # Construct weights
    w = np.zeros_like(C_abs_sq, dtype=float)
    if np.any(above):
        pref = N / (2.0 * (N - 1))
        arg = 1.0 - 4.0 * (N - 1) / (N * N) * C_abs_sq
        arg = np.clip(arg, 0.0, 1.0)
        sqrt_term = np.sqrt(arg)
        w[above] = pref * (1.0 + sqrt_term[above])
        w[~above] = 0.0
    # Apply weights and return real part
    T_filtered = T_sum * w
    const = 2.0 * np.prod(1.0 / (b - a))
    A = const * np.real(T_filtered)
    return A


if __name__ == "__main__":

    mu    = np.array([1.0, 0.5, -1.1, 1.3])
    Sigma = np.array([[1.0, 0.4, 0.3, 0.1],
                      [0.4, 1.0, 0.6, 0.5],
                      [0.3, 0.6, 1.0, 0.5],
                      [0.1, 0.5, 0.5, 1.0]])
    d = len(mu)
    std   = np.sqrt(np.diag(Sigma))
    a     = mu - 7 * std
    b     = mu + 7 * std
    s     = np.ones_like(mu)
    K     = np.array([64]*d)

    N = 1_000
    rng = np.random.default_rng(0)
    X = rng.multivariate_normal(mean=mu, cov=Sigma, size=N)

    t0 = perf_counter()
    A = A_tensor_no_sign_blowup(X, a, b, s, K)
    t1 = perf_counter()
    print("A shape:", A.shape, "time:", f"{t1-t0:.3f}s")
    
    # To use the frequency‑filtered estimator:
    A_filtered = A_tensor_frequency_filtered(X, a, b, s, K, contiguous=True)
