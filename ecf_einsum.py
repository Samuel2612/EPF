import numpy as np
import opt_einsum as oe
from time import perf_counter



def _pin_path(subs, K, dtype, warmup_N=1000):
    dummy = [np.empty((warmup_N, Kj), dtype=dtype) for Kj in K]
    _EINSUM_PATH, info = np.einsum_path(subs, *dummy, optimize='greedy')
    return _EINSUM_PATH, info

def ecf_grid_einsum(X, a, b, s, K, dtype=np.complex64):
    X = np.asarray(X)
    N, d = X.shape
    a, b, s, K = map(np.asarray, (a, b, s, K))

    Es = []
    for j in range(d):
        alpha = np.pi * s[j] / (b[j] - a[j])
        kj = np.arange(K[j])
        Es.append(np.exp(1j * alpha * X[:, j, None] * kj[None, :]).astype(dtype))

    letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    in_subs  = [f"n{letters[j]}" for j in range(len(K))]
    out_subs = "".join(letters[:len(K)])
    subs     = ",".join(in_subs) + "->" + out_subs

    path, info = _pin_path(subs, K, dtype)
    # print(info)
    return np.einsum(subs, *Es, optimize=path) / N



def true_gaussian_cf_on_grid(a, b, s, K, mu, Sigma, dtype=np.complex64):
    d = len(K)
    alphas = np.pi * s / (b - a)
    grids = np.meshgrid(*[alphas[j] * np.arange(K[j]) for j in range(d)], indexing='ij')
    t_stack = np.stack(grids, axis=-1)  # (..., d)

    quad = np.einsum('...i,ij,...j->...', t_stack, Sigma, t_stack, optimize=True)   # t^T Σ t
    lin  = np.tensordot(t_stack, mu, axes=([-1], [0]))                              # t^T μ
    return np.exp(1j * lin - 0.5 * quad).astype(dtype)

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    mu    = np.array([0.0, 0.0, 0.0, 0.0])
    Sigma = np.array([[1.0, 0.4, 0.3, 0.1],
                      [0.4, 1.0, 0.6, 0.5],
                      [0.3, 0.6, 1.0, 0.5],
                      [0.1, 0.5, 0.5, 1.0]])
    std   = np.sqrt(np.diag(Sigma))
    a     = mu - 7 * std
    b     = mu + 7 * std
    s     = np.ones_like(mu)          # same as before
    K     = np.array([64]*4)


    N = 10000
    X = rng.multivariate_normal(mean=mu, cov=Sigma, size=N)

    # empirical CF
    t0 = perf_counter()
    phi_hat = ecf_grid_einsum(X, a, b, s, K, dtype=np.complex128)
    t1 = perf_counter()

    # analytic CF
    phi_true = true_gaussian_cf_on_grid(a, b, s, K, mu, Sigma, dtype=phi_hat.dtype)

    # errors
    abs_err = np.abs(phi_hat - phi_true)
    rmse = np.sqrt((abs_err**2).mean())
    max_err = abs_err.max()

    print(f"phi shape: {phi_hat.shape}, dtype: {phi_hat.dtype}")
    print(f"ECF build time: {t1 - t0:.3f}s")
    print(f"Global RMSE: {rmse:.3e}, Max err: {max_err:.3e}")
