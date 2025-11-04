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
    K     = np.array([16]*d)

    N = 1_000
    rng = np.random.default_rng(0)
    X = rng.multivariate_normal(mean=mu, cov=Sigma, size=N)

    t0 = perf_counter()
    A = A_tensor_no_sign_blowup(X, a, b, s, K)
    t1 = perf_counter()
    print("A shape:", A.shape, "time:", f"{t1-t0:.3f}s")
    
    # Optional: sanity check vs brute-force loop
    def brute_force_A():
        d = len(K)
        alphas = np.pi / (b - a)
        grids = np.meshgrid(*[alphas[j] * np.arange(K[j]) for j in range(d)], indexing='ij')
        t_base = np.stack(grids, axis=-1)  # (..., d)

        A_loop = np.zeros(K, dtype=np.float64)
        # all sign patterns with s0 = +1
        for bits in product([1, -1], repeat=d-1):
            s_vec = np.array((1,)+bits, dtype=int)
            t = s_vec * t_base
            phase = np.exp(-1j * (t * a).sum(axis=-1))
            # empirical φ at t using samples
            exp_terms = np.exp(1j * (X @ t.reshape(-1, d).T)).mean(axis=0)
            exp_terms = exp_terms.reshape(K)
            A_loop += np.real(phase * exp_terms)
        A_loop *= 2 * np.prod(1/(b-a))
        return A_loop

    # Uncomment to verify (slow)
    A_ref = brute_force_A()
    print("RMSE(A):", np.sqrt(((A - A_ref)**2).mean()))
