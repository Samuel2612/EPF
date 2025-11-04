
import numpy as np
from mpmath import mp



def mvt_chf(u, mu, Sigma, nu, prec=50):
    mp.dps = prec
    u = np.asarray(u, dtype=float)
    mu = np.asarray(mu, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)
    q = float(u @ Sigma @ u)
    s = mp.sqrt(mp.mpf(nu) * mp.mpf(q))
    alpha = mp.mpf(nu) / 2
    K = mp.besselk(alpha, s)
    pref = mp.power(2, 1 - nu/2) / mp.gamma(nu/2) * mp.power(nu * q, nu/4)
    g = pref * K
    phase = mp.e**(1j * mp.mpf(mu @ u))
    phi = phase * g
    return complex(phi)

def dphi_dnu(u, mu, Sigma, nu, prec=50):
    mp.dps = prec
    u = np.asarray(u, dtype=float)
    mu = np.asarray(mu, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)
    q = float(u @ Sigma @ u)
    s = mp.sqrt(mp.mpf(nu) * mp.mpf(q))
    alpha = mp.mpf(nu) / 2
    phi = mvt_chf(u, mu, Sigma, nu, prec=prec)

    term_basic = (-0.5 * mp.log(2)
                  - 0.5 * mp.digamma(alpha)
                  + 0.25 * mp.log(mp.mpf(nu) * mp.mpf(q))
                  + 0.25)

    def lnK(a):
        return mp.log(mp.besselk(a, s))
    dlnK_dalpha = mp.diff(lnK, alpha)

    K_alpha = mp.besselk(alpha, s)
    K_alpha_minus = mp.besselk(alpha - 1, s)
    K_alpha_plus  = mp.besselk(alpha + 1, s)
    Kprime_over_K = ( -0.5 * (K_alpha_minus + K_alpha_plus) ) / K_alpha
    ds_dnu = s / (2 * mp.mpf(nu))

    dlnK_dnu = 0.5 * dlnK_dalpha + Kprime_over_K * ds_dnu

    dlogphi_dnu = term_basic + dlnK_dnu
    dphi = phi * complex(dlogphi_dnu)
    return dphi

def fd_dphi(u, mu, Sigma, nu, h=1e-6, prec=120):
    mp.dps = prec
    fph = mvt_chf(u, mu, Sigma, nu + h, prec=prec)
    fmh = mvt_chf(u, mu, Sigma, nu - h, prec=prec)
    return (fph - fmh) / (2*h)

def random_spd_matrix(n, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, n))
    Sigma = A @ A.T
    Sigma += 0.5 * np.eye(n)
    return Sigma


    

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    n = 4
    mu = rng.normal(size=n)
    Sigma = random_spd_matrix(n, seed=123)
    U = [rng.normal(size=n) for _ in range(5)]
    U = [u / (1.0 + np.linalg.norm(u)) for u in U]
    nus = [3.5, 5.0, 7.25, 10.0, 15.0]
    h = 1e-6

    print("Testing ∂φ/∂ν (analytic) vs finite difference for 4D MVT chf")
    print(f"μ = {mu}")
    print(f"Σ (first row) = {Sigma[0]}")
    print(f"ν values = {nus}")
    print(f"h (finite-diff step) = {h} \n")

    max_abs_err = 0.0
    max_rel_err = 0.0
    worst_case = None

    for nu in nus:
        print(f"--- ν = {nu} ---")
        for i, u in enumerate(U):
            dphi_a = dphi_dnu(u, mu, Sigma, nu)
            dphi_fd = fd_dphi(u, mu, Sigma, nu, h=h)
            abs_err = abs(dphi_a - dphi_fd)
            denom = max(abs(dphi_fd), 1e-30)
            rel_err = abs_err / denom
            print(f"u[{i}] abs_err={abs_err:.3e}  rel_err={rel_err:.3e}")
            if abs_err > max_abs_err:
                max_abs_err = abs_err
                max_rel_err = rel_err
                worst_case = (nu, i)
        print()

    print("Summary:")
    print(f"Max abs error = {max_abs_err:.3e}, max rel error = {max_rel_err:.3e}, at (ν, u_idx) = {worst_case}")
