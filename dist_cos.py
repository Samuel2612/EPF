# intraday_student_t.py
"""Intraday multivariate Student‑t fitting with ECF‑GMM on a COS lattice
-----------------------------------------------------------------------
Core design
===========
* ``StudentTModel``   → all distribution‑specific machinery: characteristic
  function, analytic gradient, random sampling, initial‐value heuristics and
  Ledoit–Wolf covariance shrinkage.
* ``COSStudentTFitter`` → orchestrates optimisation of the ECF loss, handles
  lattice mini‑batching, diagnostic checks, and (optionally) COS inversion for
  pdf / moments.

Both classes are *stateless* w.r.t. historical data once ``fit`` is called, so
multiple delivery hours can be processed in parallel.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Tuple, Optional, Dict

import numpy as np
import pandas as pd
from numpy.random import Generator, PCG64
from scipy.optimize import minimize
from scipy.stats import chi2, kendalltau, t as sps_t
from sklearn.covariance import ledoit_wolf



def _ledoit_wolf_shrinkage(sample: np.ndarray) -> np.ndarray:
    """Return Ledoit–Wolf shrunk covariance given raw sample covariance.
.
    """
    lw_, shrinkage_ = ledoit_wolf(sample)
    return lw_


def _student_t_nu_mom(r: np.ndarray) -> float:
    """Method‑of‑moments ν from sample return array (1‑D)."""
    m2 = np.mean((r - r.mean()) ** 2)
    m4 = np.mean((r - r.mean()) ** 4)
    nu = 2.0 + (m2**2) / np.clip(m4 - m2**2, 1e-12, np.inf)
    return np.clip(nu, 2.1, 200.0)


@dataclass
class StudentTModel:
    """Multivariate Student‑t with Samuelson scaling on first component."""

    d: int                                   # dimension (5 in our case)
    k: int                                   # number of regression drivers
    mu: np.ndarray = field(init=False)       # (d,)
    B: np.ndarray = field(init=False)        # (d, k)
    Sigma: np.ndarray = field(init=False)    # (d, d)
    alpha: float = field(init=False)
    nu: float = field(init=False)


    def initialise(
        self,
        df: pd.DataFrame,
        x_cols: Tuple[str, ...],
        z_cols: Tuple[str, ...],
        tau_col: str,
        lags: int = 90,
        rng: Optional[Generator] = None,
    ) -> None:
        """Fill starting values using the last *lags* trading days.

        Parameters
        ----------
        df : DataFrame
            Must contain *at least* columns listed in ``x_cols + z_cols``.
        x_cols, z_cols : tuple
            Names of the X‑vector components and regression drivers.
        tau_col : str
            Column holding minutes‑to‑delivery.
        lags : int, default 90
            Look‑back horizon in *days* to form initial estimates.
        """
        hist = df.tail(lags * 24 * 12)  # crude: ~12×24 5‑min obs per day
        X = hist[list(x_cols)].to_numpy()

        trim_q = np.quantile(X, [0.05, 0.95], axis=0)
        mask = (X >= trim_q[0]) & (X <= trim_q[1])
        mu0 = np.where(mask, X, np.nan).mean(axis=0)
        self.mu = np.nan_to_num(mu0, nan=0.0)

       
        self.B = np.zeros((self.d, self.k))


        S = np.cov(X.T)
        self.Sigma = _ledoit_wolf_shrinkage(S)


        returns = X[:, 0]
        tau = hist[tau_col].to_numpy()
        qs = np.quantile(np.log(tau + 1e-9), np.linspace(0, 1, 6))
        var_bin = []
        centres = []
        for lo, hi in zip(qs[:-1], qs[1:]):
            sel = (np.log(tau) >= lo) & (np.log(tau) < hi)
            if sel.any():
                var_bin.append(np.var(returns[sel]))
                centres.append(np.exp((lo + hi) / 2))
        if len(var_bin) >= 2:
            slope, _ = np.polyfit(np.log(centres), np.log(var_bin), 1)
            self.alpha = np.clip(-slope, 0.1, 2.0)
        else:
            self.alpha = 0.6

        self.nu = _student_t_nu_mom(returns)


    def _samuelson_scaled_sigma(self, tau: float) -> np.ndarray:
        D = np.ones(self.d)
        D[0] = tau ** (-self.alpha / 2.0)
        return (D * self.Sigma) * D[:, None]  # fast diag scaling

    # vectorised CF for an array of u (J, d) & broadcast tau (J,)
    def phi(self, u: np.ndarray, tau: np.ndarray, Z: np.ndarray) -> np.ndarray:
        m = self.mu + (self.B @ Z.T).T        # (J, d)
        uq = np.einsum("jd,jd->j", u, (u @ self.Sigma))
        # scale tau only along price axis for q term
        uq_scaled = uq * (tau ** (-self.alpha))  # because only u_price gets scaled
        exponent = 1j * np.einsum("jd,jd->j", u, m)
        denom = (1.0 + uq_scaled / self.nu) ** (self.nu / 2.0)
        return np.exp(exponent) / denom

    # analytic gradient wrt parameters *on a single (u, tau, Z)*
    def grad_phi(
        self,
        u: np.ndarray,
        tau: float,
        Z: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Return ∂φ/∂params in a dictionary; used by outer optimiser."""
        # Pre‑compute reusable scalars
        Du = u.copy()
        Du[0] *= tau ** (-self.alpha / 2.0)
        q = Du @ self.Sigma @ Du / self.nu
        base = (1 + q) ** (-self.nu / 2.0)
        phi_val = np.exp(1j * u @ (self.mu + self.B @ Z)) * base


        grad_mu = 1j * u * phi_val               # (d,)
        grad_alpha = (
            phi_val
            * (-0.5 * np.log(tau) * (Du[0] ** 2) * (1 + q) ** (-1))
        )

        outer = np.outer(Du, Du) / (self.nu * (1 + q))
        grad_Sigma = -0.5 * phi_val * outer      # (d, d)
        grad_B = np.outer(grad_mu, Z)
        grad_nu = (
            phi_val
            * ( -0.5 * np.log(1 + q) + q / (2 * (1 + q)) )
        )
        return {
            "mu": grad_mu,
            "B": grad_B,
            "Sigma": grad_Sigma,
            "alpha": grad_alpha,
            "nu": grad_nu,
            "phi": phi_val,
        }


    def sample(self, n: int, tau: float, Z: Optional[np.ndarray] = None, rng=None) -> np.ndarray:
        rng = rng or Generator(PCG64())
        s = rng.chisquare(self.nu, size=n)
        eps = rng.standard_normal((n, self.d))
        L = np.linalg.cholesky(self.Sigma)
        D = np.ones(self.d)
        D[0] = tau ** (-self.alpha / 2.0)
        base = (np.sqrt(self.nu / s)[:, None]) * (eps @ L.T)
        X = self.mu + (self.B @ Z.T).T if Z is not None else self.mu
        return X + base * D



