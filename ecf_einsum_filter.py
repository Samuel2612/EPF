import numpy as np
import opt_einsum as oe
from time import perf_counter
from itertools import product



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
    print(info)
    return np.einsum(subs, *Es, optimize=path) / N

def true_gaussian_cf_on_grid(a, b, s, K, mu, Sigma, dtype=np.complex64):
    d = len(K)
    alphas = np.pi * s / (b - a)
    grids = np.meshgrid(*[alphas[j] * np.arange(K[j]) for j in range(d)], indexing='ij')
    t_stack = np.stack(grids, axis=-1)  # (..., d)

    quad = np.einsum('...i,ij,...j->...', t_stack, Sigma, t_stack, optimize=True)   # t^T Σ t
    lin  = np.tensordot(t_stack, mu, axes=([-1], [0]))                              # t^T μ
    return np.exp(1j * lin - 0.5 * quad).astype(dtype)



_expr_cache = {}

def ecf_grid_all_signs(X, a, b, s, K, dtype=np.complex64):
    """
    Return φ for every sign choice in {+,-}^d (sign axis length 2 per dim),
    shape: (2,2,...,2, K1,...,Kd). Axis order: [s0, s1, ..., sd-1, k0, ..., kd-1]
    """
    X = np.asarray(X)
    N, d = X.shape
    a, b, s, K = map(np.asarray, (a, b, s, K))
    letters_s = "uvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    letters_k = "abcdefghijklmno"
    assert d <= len(letters_s) and d <= len(letters_k)

    Es = []
    for j in range(d):
        alpha = np.pi * s[j] / (b[j] - a[j])
        kj = np.arange(K[j])
        Ej_pos = np.exp(1j * alpha * X[:, j, None] * kj[None, :]).astype(dtype)  # (N,Kj)
        # stack +/- along a sign axis: index 0 => '+', 1 => '-'
        Es.append(np.stack([Ej_pos, Ej_pos.conj()], axis=1))                     # (N,2,Kj)

    # n  s_j  k_j  -> all s_j, all k_j
    in_subs  = [f"n{letters_s[j]}{letters_k[j]}" for j in range(d)]
    out_subs = "".join(letters_s[:d]) + "".join(letters_k[:d])
    subs     = ",".join(in_subs) + "->" + out_subs

    shapes = [(N, 2, K[j]) for j in range(d)]
    key = (subs, tuple(shapes), dtype)
    if key not in _expr_cache:
        _expr_cache[key] = oe.contract_expression(subs, *shapes, optimize='greedy')
    expr = _expr_cache[key]

    phi_all = expr(*Es) / N
    return phi_all  



def compute_A(phi_all, a, b, K, dtype=np.complex64):
    """
    phi_all: output of ecf_grid_all_signs
    a, b:    vectors (length d)
    K:       grid sizes
    Returns A with shape (K1,...,Kd)
    """
    d = len(K)
    const = 2 * np.prod(1.0 / (b - a))

    # --- Build base t-grid WITHOUT signs: t_base[..., j] = π * k_j / (b_j - a_j)
    alphas = np.pi / (b - a)
    grids = np.meshgrid(*[alphas[j] * np.arange(K[j]) for j in range(d)], indexing='ij')
    t_base = np.stack(grids, axis=-1)  # shape (K1,...,Kd,d)

    # --- Build phase tensor for all sign choices (same sign axes as phi_all)
    # For each dim j, two versions: + and - (conjugates)
    phase_tensor = 1
    for j in range(d):
        ph_pos = np.exp(-1j * a[j] * t_base[..., j])            # (K1,...,Kd)
        ph_both = np.stack([ph_pos, ph_pos.conj()], axis=0)      # (2, K1,...,Kd)

        # reshape to align sign axis j in front
        shape = (1,)*j + (2,) + (1,)*(d-j-1) + tuple(K)
        phase_tensor = phase_tensor * ph_both.reshape(shape)

    # --- Fix first sign = +1 (index 0), sum over the remaining (d-1) sign axes
    # phi_all, phase_tensor have same sign-axis layout
    phi_sel   = phi_all[0]           # shape (2,2,...,2, K1..Kd) with (d-1) sign axes
    phase_sel = phase_tensor[0]

    sign_axes = tuple(range(d-1))    # axes over which to sum (the remaining sign axes)
    A = const * np.real(np.sum(phase_sel * phi_sel, axis=sign_axes))

    return A.astype(dtype)           # shape (K1,...,Kd)

# --------------------  Demo / test  --------------------

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    mu    = np.array([0.0, 0.0])
    Sigma = np.array([[1.0, 0.4],
                      [0.4, 1.0]])
    std   = np.sqrt(np.diag(Sigma))
    a     = mu - 7 * std
    b     = mu + 7 * std
    s     = np.ones_like(mu)
    K     = np.array([64]*2)

    N = 1000000
    X = rng.multivariate_normal(mean=mu, cov=Sigma, size=N)

    # (1) Empirical CF on positive grid (for comparison)
    t0 = perf_counter()
    phi_hat_pos = ecf_grid_einsum(X, a, b, s, K, dtype=np.complex64)
    t1 = perf_counter()
    print(f"phi_hat_pos shape: {phi_hat_pos.shape}, time: {t1-t0:.3f}s")

    # (2) All sign permutations in one go
    t2 = perf_counter()
    phi_all = ecf_grid_all_signs(X, a, b, s, K)
    t3 = perf_counter()
    print(f"phi_all shape: {phi_all.shape}, time: {t3-t2:.3f}s")

    # (3) Build A tensor
    A = compute_A(phi_all, a, b, K)
    print("A shape:", A.shape)

    # Optional: sanity check vs brute-force loop
    # (slow but small K can verify correctness)
    # def brute_force_A():
    #     d = len(K)
    #     alphas = np.pi / (b - a)
    #     grids = np.meshgrid(*[alphas[j] * np.arange(K[j]) for j in range(d)], indexing='ij')
    #     t_base = np.stack(grids, axis=-1)  # (..., d)

    #     A_loop = np.zeros(K, dtype=np.float64)
    #     # all sign patterns with s0 = +1
    #     for bits in product([1, -1], repeat=d-1):
    #         s_vec = np.array((1,)+bits, dtype=int)
    #         t = s_vec * t_base
    #         phase = np.exp(-1j * (t * a).sum(axis=-1))
    #         # empirical φ at t using samples
    #         exp_terms = np.exp(1j * (X @ t.reshape(-1, d).T)).mean(axis=0)
    #         exp_terms = exp_terms.reshape(K)
    #         A_loop += np.real(phase * exp_terms)
    #     A_loop *= 2 * np.prod(1/(b-a))
    #     return A_loop

    # # Uncomment to verify (slow)
    # A_ref = brute_force_A()
    # print("RMSE(A):", np.sqrt(((A - A_ref)**2).mean()))
