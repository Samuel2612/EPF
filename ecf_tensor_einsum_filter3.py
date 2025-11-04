import numpy as np
import itertools, math, time

# ================================================================
#  USER‑SUPPLIED TT‑SVD
# ================================================================
def tt_svd(tensor, eps=1e-12):
    """
    Tensor‑Train SVD with Frobenius relative tolerance `eps`.
    Returns a list of 3‑way cores and the TT‑ranks.
    """
    T = tensor.copy()
    dims = T.shape
    d = len(dims)
    thr = (eps / np.sqrt(d - 1)) ** 2 * np.linalg.norm(T) ** 2

    cores, ranks = [], [1]                       # r0 = 1
    unfold = T
    for k in range(d - 1):
        m = ranks[-1] * dims[k]
        unfold = unfold.reshape(m, -1)
        U, S, Vh = np.linalg.svd(unfold, full_matrices=False)

        tail = np.cumsum(S[::-1] ** 2)[::-1]
        r = np.searchsorted(tail <= thr, True)
        if r == 0:
            r = len(S)

        cores.append(U[:, :r].reshape(ranks[-1], dims[k], r))
        ranks.append(r)
        unfold = (S[:r, None] * Vh[:r])

    cores.append(unfold.reshape(ranks[-1], dims[-1], 1))
    ranks.append(1)
    return cores, ranks


# ================================================================
#  COS coefficients tensor  A_{k1…kn}   (section 6, eq. 3.5)
# ================================================================
def chf_gaussian(t, mu, Sigma):
    """
    Characteristic function  E[e^{i t·X}]  for  X ~ N(mu, Σ).
    `t` shape (..., n).
    """
    lead_shape = t.shape[:-1]
    t_flat = t.reshape(-1, t.shape[-1])                # (M,n)
    phase = 1j * t_flat @ mu
    quad = np.einsum('ij,jk,ik->i', t_flat, Sigma, t_flat)
    return np.exp(phase - 0.5 * quad).reshape(lead_shape)


def build_tensor_lowmem(Ks, a, b, mu, Sigma):
    """
    Build the COS coefficient tensor  A_{k1…kn}
    using theorem 2/3 (eq. 3.5) – low‑memory loop over sign vectors.
    """
    n = len(Ks)
    factor = 2.0 / np.prod(b - a)

    rest = np.array(list(itertools.product([1, -1], repeat=n - 1)), float)
    svecs = np.concatenate([np.ones((rest.shape[0], 1)), rest], axis=1)  # (2^{n-1}, n)

    grid = np.indices(Ks, dtype=float).T                                # (*dims, n)
    A = np.zeros(Ks, dtype=float)

    for s in svecs:
        t = math.pi * grid * s / (b - a)        # (*dims, n)
        phi = chf_gaussian(t, mu, Sigma)        # (*dims,)
        phase = np.exp(-1j * np.sum(t * a, axis=-1))
        A += np.real(phase * phi)

    return factor * A


# ================================================================
#  1‑D integral  I_{j,k}(ω)   (section 6, theorem 10)
# ================================================================
def I_jk(omega, w_j, S0_j, mu_j, sig_j, k, a_j, b_j, z_grid, cos_cache):
    """
    Numerical evaluation (trapezoid) of

        I_{j,k}(ω) = ∫_a^b e^{i ω w_j S0_j e^{μ_j + σ_j z}}
                          cos(k π (z - a)/(b - a)) dz
    """
    exp_arg = np.exp(mu_j + sig_j * z_grid)
    integrand = np.exp(1j * omega * w_j * S0_j * exp_arg) * cos_cache[k]
    return np.trapz(integrand, z_grid)


# ================================================================
#  Characteristic function  φ_H(ω)   via TT contraction
# ================================================================
def phi_H_tt(omega_vec, cores, params, prime, K, a, b, M=4001):
    """
    Evaluate φ_H(ω) for each ω in `omega_vec` using TT cores.

    params – list [(w_j, S0_j, mu_j, sig_j)]  length n.
    """
    n = len(params)
    # pre‑compute per‑dimension cosine cache
    cos_caches, z_grids = [], []
    for j in range(n):
        z = np.linspace(a[j], b[j], M)
        factor = math.pi / (b[j] - a[j])
        cos_cache = np.array([np.cos(k * factor * (z - a[j])) for k in range(K + 1)])
        cos_caches.append(cos_cache)
        z_grids.append(z)

    phi = np.empty(len(omega_vec), dtype=complex)

    for t, omega in enumerate(omega_vec):
        mat = np.array([[1.0 + 0j]])          # shape (1,1)
        for j, G in enumerate(cores):
            wj, S0j, muj, sigj = params[j]

            Ik = np.array([I_jk(omega, wj, S0j, muj, sigj,
                                 k, a[j], b[j], z_grids[j], cos_caches[j])
                           for k in range(K + 1)])
            vec = prime * Ik                  # (K+1,)

            # Contract core with vec over mode‑1
            Mj = np.tensordot(G, vec, axes=([1], [0]))   # (r_prev, r_next)
            mat = mat @ Mj
        phi[t] = mat[0, 0]
    return phi


# ================================================================
#  Pay‑off COS coefficients  V_k   (closed form for call / put)
# ================================================================
def V_k_call(k, K_strike, A, B):
    if k == 0:
        return (B - K_strike)
    u = k * math.pi / (B - A)
    return 2.0 / (B - A) * (
        (B - K_strike) * np.sin(u * (B - K_strike)) / u +
        (np.cos(u * (B - K_strike)) - 1.0) / (u ** 2)
    )


def V_k_put(k, K_strike, A, B):
    if k == 0:
        return (K_strike - A)
    u = k * math.pi / (B - A)
    return 2.0 / (B - A) * (
        (1.0 - np.cos(u * (K_strike - A))) / (u ** 2) +
        (K_strike - A) * np.sin(u * (K_strike - A)) / u
    )


# ================================================================
#  Main pricer  (section 6 algorithm fully)
# ================================================================
def basket_cos_tt(
    S0, w, sigma, mu_vec, Sigma,           # model
    r, T, K_strike, option_type="call",    # contract
    Ks=(64, 64), L_H=8.0, N_cos=256, eps_pdf=1e-12, eps_pay=1e-10,
    M_int=4001
):
    """
    Full COS‑TT basket option price as in paper section 6.

    * Uses user tt_svd for TT decompositions.
    * Works for arbitrary 2‑D correlation matrix (and easily n‑D).
    """
    S0 = np.asarray(S0, float)
    w = np.asarray(w, float)
    sigma = np.asarray(sigma, float)

    d = len(S0)
    Ks = list(Ks) if hasattr(Ks, "__len__") else [Ks] * d
    assert d == len(Ks) == len(w) == len(sigma)

    # truncation for each Z_j  :  [a_j , b_j] = μ_j ± 8 σ_j
    std = np.sqrt(np.diag(Sigma))
    a = mu_vec - 8 * std
    b = mu_vec + 8 * std

    # ----------------------------------------------------------------
    # 1) build coefficient tensor A and TT compress (pdf)
    # ----------------------------------------------------------------
    A_tensor = build_tensor_lowmem(Ks, a, b, mu_vec, Sigma)
    cores_pdf, ranks_pdf = tt_svd(A_tensor, eps=eps_pdf)

    # ----------------------------------------------------------------
    # 2) parameters for I_{j,k}(ω)
    # ----------------------------------------------------------------
    mu_S = (r - 0.5 * sigma ** 2) * T
    sig_S = sigma * np.sqrt(T)
    params = [(w[j], S0[j], mu_S[j], sig_S[j]) for j in range(d)]
    K_max = max(Ks) - 1
    prime = np.ones(K_max + 1)
    prime[0] = 0.5

    # ----------------------------------------------------------------
    # 3) choose [A,B] interval for H = Σ w_j S_j
    # ----------------------------------------------------------------
    # crude mean & variance of H  (Monte‑Carlo moment formula)
    E_S = S0 * np.exp(mu_S + 0.5 * sig_S ** 2)
    Var_S = S0**2 * np.exp(2 * mu_S + sig_S ** 2) * (np.exp(sig_S ** 2) - 1)
    mean_H = np.dot(w, E_S)
    std_H = math.sqrt(np.dot(w ** 2, Var_S))
    A_pay, B_pay = 0.0, mean_H + L_H * std_H

    # ----------------------------------------------------------------
    # 4) characteristic function φ_H at COS nodes
    # ----------------------------------------------------------------
    k_idx = np.arange(N_cos)
    u_k = k_idx * math.pi / (B_pay - A_pay)

    phi = phi_H_tt(u_k, cores_pdf, params, prime, K_max, a, b, M=M_int)

    # ----------------------------------------------------------------
    # 5) pay‑off coefficients V_k
    # ----------------------------------------------------------------
    if option_type == "call":
        V = np.array([V_k_call(k, K_strike, A_pay, B_pay) for k in k_idx])
    else:
        V = np.array([V_k_put(k, K_strike, A_pay, B_pay) for k in k_idx])

    weight = np.ones(N_cos)
    weight[0] = 0.5

    price = np.exp(-r * T) * np.sum(weight * np.real(phi * V))
    return price, ranks_pdf


# ================================================================
#  Demo run (2 assets, correlated)
# ================================================================
if __name__ == "__main__":
    # market parameters
    S0 = [100.0, 120.0]
    w = [0.6, 0.4]
    sigma = [0.20, 0.25]
    rho = 0.5
    Sigma = np.array([[1.0, rho],
                      [rho, 1.0]])
    mu_vec = np.zeros(2)             # Z ~ N(0,Σ)
    r, T = 0.05, 1.0
    K_strike = 110.0

    # COS‑TT price
    start = time.time()
    price_tt, ranks_pdf = basket_cos_tt(
        S0, w, sigma, mu_vec, Sigma,
        r, T, K_strike,
        option_type="call",
        Ks=(64, 64),
        L_H=8.0,
        N_cos=256,
        eps_pdf=1e-12,
        M_int=4001
    )
    elapsed = time.time() - start

    print(f"COS‑TT basket‑call price ≈ {price_tt: .6f}   (time {elapsed: .2f}s)")
    print(f"TT ranks pdf: {ranks_pdf}")

    # Monte‑Carlo verification
    Nmc = 2_000_000
    z_mc = np.random.multivariate_normal(mu_vec, Sigma, size=Nmc)
    mu_S = (r - 0.5 * np.array(sigma) ** 2) * T
    sig_S = np.array(sigma) * np.sqrt(T)
    S_mc = S0 * np.exp(mu_S + sig_S * z_mc)
    H_mc = S_mc @ w
    pay_mc = np.maximum(H_mc - K_strike, 0.0)
    price_mc = np.exp(-r * T) * pay_mc.mean()
    stderr_mc = np.exp(-r * T) * pay_mc.std(ddof=1) / np.sqrt(Nmc)
    print(f"Monte‑Carlo price        ≈ {price_mc: .6f} ± {stderr_mc: .6f}")
