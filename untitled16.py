import numpy as np


def _std_norm_pdf(z):
    """φ(z)  =  standard–normal pdf."""
    return np.exp(-0.5 * z**2) / np.sqrt(2.0 * np.pi)


def _trapz(y, x):
    """One-dimensional trapezoidal integral (vectorised)."""
    return np.trapz(y, x)



def _G_coeffs(K, a, b, M = 2001):
    """
    G_k  =  2/(b−a) ∫_a^b φ(z) cos(kπ(z−a)/(b−a)) dz,   k = 0..K
    computed with simple trapezoidal quadrature.
    """
    z = np.linspace(a, b, M)
    pdf = _std_norm_pdf(z)
    factor = np.pi / (b - a)

    G = np.empty(K + 1, dtype=float)
    for k in range(K + 1):
        cos_term = np.cos(k * factor * (z - a))
        G[k] = 2.0 / (b - a) * _trapz(pdf * cos_term, z)
    return G


def _I_jk(omega, wj, S0_j, mu_j, sig_j, k,a,b, z, cos_cache):
    """
    I_{j,k}(ω) = ∫_a^b  exp(iω w_j S0_j e^{μ_j+σ_j z}) · cos(kπ(z−a)/(b−a)) dz
    trapezoidal quadrature re-using cached cos terms.
    """
    exp_arg = np.exp(mu_j + sig_j * z)
    integrand = np.exp(1j * omega * wj * S0_j * exp_arg) * cos_cache[k]
    return _trapz(integrand, z)



def basket_cf(omega, S0, w, sigma, r, T, K_terms = 20, L_z = 8.0, M_int = 2001):

    S0 = np.asarray(S0)
    w = np.asarray(w)
    sigma = np.asarray(sigma)

    n = len(S0)
    assert S0.shape == w.shape == sigma.shape, "S0, w, sigma must have same length"

    a, b = -L_z, +L_z
    z_grid = np.linspace(a, b, M_int)
    factor = np.pi / (b - a)
    cos_cache = np.array([np.cos(k * factor * (z_grid - a)) for k in range(K_terms + 1)])

    G = _G_coeffs(K_terms, a, b, M_int)          
    mu_vec = (r - 0.5 * sigma**2) * T
    sig_vec = sigma * np.sqrt(T)
    prime = np.ones(K_terms + 1)
    prime[0] = 0.5
    phi = np.ones_like(omega, dtype=complex)

    for j in range(n):
        S0j, wj, muj, sigj = S0[j], w[j], mu_vec[j], sig_vec[j]

        I_jk = np.empty((K_terms + 1, omega.size), dtype=complex)
        for idx, om in enumerate(omega):
            for k in range(K_terms + 1):
                I_jk[k, idx] = _I_jk(
                    om, wj, S0j, muj, sigj,
                    k, a, b, z_grid, cos_cache
                )
        Sj = (prime[:, None] * G[:, None] * I_jk).sum(axis=0)
        phi *= Sj

    return phi



def _V_k_call(k, K_strike, A, B, N_pay = 4001):
    """V_k for a (H−K)^+ call, integrated numerically on [A,B]."""
    h = np.linspace(A, B, N_pay)
    payoff = np.maximum(h - K_strike, 0.0)
    cos_term = np.cos(k * np.pi * (h - A) / (B - A))
    integral = _trapz(payoff * cos_term, h)
    return 2.0 / (B - A) * integral


def _V_k_put(k, K_strike, A, B, N_pay = 4001):
    """V_k for a (K−H)^+ put, integrated numerically on [A,B]."""
    h = np.linspace(A, B, N_pay)
    payoff = np.maximum(K_strike - h, 0.0)
    cos_term = np.cos(k * np.pi * (h - A) / (B - A))
    integral = _trapz(payoff * cos_term, h)
    return 2.0 / (B - A) * integral


# ---------------------------------------------------------------------
#  COS-TT basket option pricer
# ---------------------------------------------------------------------
def basket_option_cos(S0,w,sigma,r,T, K_strike,  option_type = "call", N_cos = 64, K_terms = 20, L_z = 8.0, L_H = 8.0):
    """
    Price an n-asset basket option via the TT-COS method laid out in Section 6.

    
    """
    S0 = np.asarray(S0)
    w = np.asarray(w)
    sigma = np.asarray(sigma)
    n = len(S0)


    mu_vec = (r - 0.5 * sigma ** 2) * T
    sig_vec = sigma * np.sqrt(T)
    # upper bound of each asset within the truncated z-box:
    S_max = S0 * np.exp(mu_vec + sig_vec * L_z)
    H_max = np.dot(w, S_max)
    A, B = 0.0, (1.0 + L_H) * H_max        # A≥0, B safely above K


    k_idx = np.arange(N_cos)
    u_k = k_idx * np.pi / (B - A)

    phi = basket_cf(
        u_k,
        S0=S0,
        w=w,
        sigma=sigma,
        r=r,
        T=T,
        K_terms=K_terms,
        L_z=L_z,
    )


    V = np.empty(N_cos, dtype=float)
    for k in k_idx:
        if option_type == "call":
            V[k] = _V_k_call(k, K_strike, A, B)
        else:
            V[k] = _V_k_put(k, K_strike, A, B)


    weight = np.ones(N_cos)
    weight[0] = 0.5                     # prime on the outer sum
    price = np.exp(-r * T) * np.sum(weight * np.real(phi * V))
    return price



if __name__ == "__main__":
    np.random.seed(42)


    S0  = [100.0, 120.0]
    w   = [0.6, 0.4]           # weights add to 1
    sigma = [0.2, 0.25]
    r, T = 0.05, 1.0
    K_strike = 110.0

    # COS-TT price
    cos_px = basket_option_cos(
        S0, w, sigma, r, T, K_strike,
        option_type="call",
        N_cos=128,
        K_terms=50,
        L_z=8.0,
        L_H=4.0,
    )
    print(f"COS-TT basket-call ≈ {cos_px: .6f}")

    # very crude Monte-Carlo for a feel (1e5 paths)
    n_paths = 100_000
    Z = np.random.randn(n_paths, len(S0))
    S_T = S0 * np.exp((r - 0.5 * np.array(sigma) ** 2) * T + np.array(sigma) * np.sqrt(T) * Z)
    H_payoff = np.maximum(np.dot(S_T, w) - K_strike, 0.0)
    mc_px = np.exp(-r * T) * H_payoff.mean()
    print(f"Monte-Carlo price  ≈ {mc_px: .6f}")
