import numpy as np
from numpy.linalg import cholesky
from scipy.stats.qmc import Sobol
from typing import Literal, Sequence


def sobol_correlated_norm(dim: int, n_pts: int, corr: np.ndarray) -> np.ndarray:
    """
    Sobol points mapped to a zero-mean multivariate normal N(0, corr).
    """
    sampler = Sobol(d=dim, scramble=True, seed=42)
    u = sampler.random(n=n_pts)                 # (n_pts, dim) in (0,1)
    z = np.sqrt(2.0) * scipy.special.erfinv(2.0 * u - 1.0)  # iid N(0,1)
    L = cholesky(corr)
    return z @ L.T                              # correlated normals


def basket_price_correlated(
        S0: Sequence[float],
        w: Sequence[float],
        sigma: Sequence[float],
        corr: np.ndarray,
        r: float,
        T: float,
        K_strike: float,
        *,
        option_type: Literal["call", "put"] = "call",
        N_cos: int = 512,
        N_qmc: int = 100_000,
) -> float:
    """
    Basket option via 1-D COS *outer* sum, inner expectation by QMC
    (works with correlated assets).
    """
    S0, w, sigma = map(np.asarray, (S0, w, sigma))
    n = len(S0)
    assert corr.shape == (n, n)

    # ---- expectation E[e^{i u_k H}] on Sobol points -----------------
    z = sobol_correlated_norm(n, N_qmc, corr)          # shape (N_qmc, n)
    mu = (r - 0.5 * sigma ** 2) * T
    sig = sigma * np.sqrt(T)

    # pre-compute S_j(T) on all points
    S_t = S0 * np.exp(mu + sig * z)                    # (N_qmc, n)
    H_qmc = S_t @ w                                    # (N_qmc,)

    # choose [A,B]  from mean±8·stdev  of the QMC sample
    mean_H = H_qmc.mean()
    std_H  = H_qmc.std(ddof=1)
    A, B   = 0.0, mean_H + 8.0 * std_H

    k = np.arange(N_cos)
    u_k = k * np.pi / (B - A)

    # φ_H(u_k) via QMC
    exp_iuH = np.exp(1j * np.outer(u_k, H_qmc))        # (N_cos, N_qmc)
    phi = exp_iuH.mean(axis=1)                         # (N_cos,)

    # ---- payoff COS coefficients V_k  (analytic closed form) -------
    # call: V_k = 2/(B-A) * [ (cos - 1)/(u_k^2) + (K-A) * sin / u_k ]
    V = np.zeros_like(u_k, dtype=float)
    delta = K_strike - A
    for idx, uk in enumerate(u_k):
        if idx == 0:          # k = 0 limit
            if option_type == "call":
                V[idx] = (B - K_strike)               # integral of payoff
            else:
                V[idx] = (K_strike - A)
        else:
            cos_term = np.cos(uk * delta)
            sin_term = np.sin(uk * delta)
            if option_type == "call":
                V[idx] = 2.0 / (B - A) * (
                    (sin_term / uk) * (B - K_strike) +
                    (cos_term - 1.0) / (uk ** 2)
                )
            else:  # put
                V[idx] = 2.0 / (B - A) * (
                    (1.0 - cos_term) / (uk ** 2) +
                    (K_strike - A) * sin_term / uk
                )

    # prime weight
    weight = np.ones_like(u_k)
    weight[0] = 0.5

    price = np.exp(-r * T) * np.sum(weight * np.real(phi * V))
    return float(price)


# ---------------------- small demo -----------------------------------
if __name__ == "__main__":
    import scipy.special

    S0     = [100.0, 120.0, 90.0]
    w      = [0.4, 0.3, 0.3]
    sigma  = [0.25, 0.20, 0.30]
    corr   = np.array([[1.0, 0.6, 0.2],
                       [0.6, 1.0, 0.4],
                       [0.2, 0.4, 1.0]])           # positive-definite
    r, T   = 0.05, 1.0
    K_strike = 110.0

    price = basket_price_correlated(
        S0, w, sigma, corr, r, T, K_strike,
        option_type="call",
        N_cos=512,
        N_qmc=2**13    #  scrambles give ~O(N^{-1})
    )
    print(f"Correlated basket-call ≈ {price: .6f}")
