# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 09:40:26 2025


@author: samue 
"""
import numpy as np
from distributions import JSUo, JSU
from sklearn.linear_model import LinearRegression  # could swap for LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_diabetes


class GAMLSS:
    """GAMLSS with active‑set CD, likelihood convergence and step dampening"""

    # ------------------------------------------------------------------
    # construction ------------------------------------------------------
    # ------------------------------------------------------------------
    def __init__(self, *,
                 distribution=JSUo(),
                 max_iter_outer: int = 25,
                 max_iter_inner: int = 25,
                 param_steps=None,
                 rel_tol: float = 1e-4):
        """Parameters
        ----------
        distribution : a class from ``distributions`` supporting the GAMLSS
            interface
        max_iter_outer, max_iter_inner : int
            iteration limits for the outer RS loop and the inner IRLS loop
        param_steps : None | float | sequence of float, optional
            Dampening factors (``0 < step ≤ 1``) for each distribution
            parameter.  If a single float is supplied it is recycled for all
            parameters.  ``None`` → use 1.0 (no dampening).
        rel_tol : float, optional
            Relative tolerance on the negative log‑likelihood for convergence
        """

        self.distribution = distribution
        self.max_iter_outer = max_iter_outer
        self.max_iter_inner = max_iter_inner
        self.rel_tol = rel_tol

        # ---- per‑parameter step sizes --------------------------------
        n_par = self.distribution.n_of_p
        if param_steps is None:
            self.param_steps = np.ones(n_par)
        else:
            param_steps = np.asarray(param_steps, dtype=float)
            if param_steps.size == 1:
                self.param_steps = np.repeat(param_steps, n_par)
            elif param_steps.size == n_par:
                self.param_steps = param_steps.copy()
            else:
                raise ValueError("Length of param_steps must be 1 or "
                                 f"{n_par}, got {param_steps.size}.")
        if np.any((self.param_steps <= 0) | (self.param_steps > 1)):
            raise ValueError("All step sizes must lie in (0, 1].")

        # model state ----------------------------------------------------
        self.betas = {p: None for p in range(self.distribution.n_of_p)}
        self.theta = None  # n_obs × n_of_p parameter matrix
        self.scaler = StandardScaler()
        self.fitted = False

    # ------------------------------------------------------------------
    # utilities ---------------------------------------------------------
    # ------------------------------------------------------------------
    def _neg_log_likelihood(self, y, theta):
        """Return *negative* log‑likelihood (so we minimise)."""
        log_pdf = np.log(np.clip(self.distribution.pdf(y, theta), 1e-20, None))
        return -2.0 * np.sum(log_pdf)

    # ------------------------------------------------------------------
    # public API --------------------------------------------------------
    # ------------------------------------------------------------------
    def fit(self, X, y):
        """Fit the GAMLSS model"""
        # add intercept -----------------------
        X_int = np.column_stack([np.ones(X.shape[0]), X])
        X_features_scaled = self.scaler.fit_transform(X_int[:, 1:])
        X_int[:, 1:] = X_features_scaled

        n_obs, n_features = X_int.shape
        self.n_obs = n_obs

        # initialise betas --------------------
        for p in range(self.distribution.n_of_p):
            self.betas[p] = np.zeros(n_features)

        # initialise theta from the distribution helper
        self.theta = self.distribution.init_val(y)

        # outer RS loop -----------------------
        nll_prev = self._neg_log_likelihood(y, self.theta)
        self.fit_hist_nll = [nll_prev]

        for iter_outer in range(1, self.max_iter_outer + 1):
            for p in range(self.distribution.n_of_p):
                nll_prev = self._inner_update(X_int, y, p, nll_prev)
                self.fit_hist_nll.append(nll_prev)

            # convergence check ---------------
            if len(self.fit_hist_nll) >= 2:
                rel_impr = (self.fit_hist_nll[-2] - self.fit_hist_nll[-1]) / abs(self.fit_hist_nll[-2])
                if rel_impr < self.rel_tol:
                    break

        self.fitted = True
        return self

    # ------------------------------------------------------------------
    # inner IRLS + step dampening --------------------------------------
    # ------------------------------------------------------------------
    def _inner_update(self, X, y, p, nll_prev_outer):
        """One RS *inner* cycle for parameter *p* with step dampening"""
        param_step = self.param_steps[p]
        nll_prev = nll_prev_outer

        for iter_inner in range(1, self.max_iter_inner + 1):
            # cache old state -----------------
            old_beta = self.betas[p].copy()
            old_theta_p = self.theta[:, p].copy()

            # working‑response + weights -------
            eta_old = self.distribution.link_function(old_theta_p, p)
            dr = 1.0 / self.distribution.link_inverse_derivative(eta_old, p)
            dl1dp1 = self.distribution.dll(y, self.theta, p)
            dl2dp2 = self.distribution.ddll(y, self.theta, p)
            wt = -(dl2dp2 / (dr ** 2))
            wt = np.clip(wt, 1e-10, 1e10)
            y_work = eta_old + dl1dp1 / (dr * wt)
            y_work = np.nan_to_num(y_work, nan=0.0, posinf=1e10, neginf=-1e10)
            wt = np.nan_to_num(wt, nan=1.0, posinf=1e10, neginf=1e-10)

            # IRLS / coordinate‑descent step --
            linreg = LinearRegression(fit_intercept=False)
            linreg.fit(X, y_work, sample_weight=wt)
            beta_new = linreg.coef_

            # convex mixing (step 2) ----------
            eta_prop = X @ beta_new
            eta_mixed = param_step * eta_prop + (1.0 - param_step) * eta_old
            theta_new_p = self.distribution.link_inverse(eta_mixed, p)

            # update global theta -------------
            self.betas[p] = beta_new
            self.theta[:, p] = theta_new_p

            # evaluate new NLL ---------------
            nll_new = self._neg_log_likelihood(y, self.theta)

            # if NLL increased, automatic step‑halving (step 3) ----------
            if nll_new > nll_prev:
                eta_trial = eta_mixed.copy()
                step_local = param_step
                success = False
                for _ in range(5):
                    step_local *= 0.5
                    eta_trial = step_local * eta_prop + (1.0 - step_local) * eta_old
                    theta_trial_p = self.distribution.link_inverse(eta_trial, p)
                    self.theta[:, p] = theta_trial_p
                    nll_trial = self._neg_log_likelihood(y, self.theta)
                    if nll_trial <= nll_prev:
                        # accept halved step
                        self.betas[p] = beta_new  # beta is still valid for any eta_trial because it yields eta_prop; dampening is on eta only
                        nll_new = nll_trial
                        success = True
                        break
                if not success:
                    # revert to old state --------------------------------
                    self.betas[p] = old_beta
                    self.theta[:, p] = old_theta_p
                    nll_new = nll_prev  # no improvement
                    break  # exit inner loop for p
            else:
                # improvement accepted -------
                pass

            # check inner convergence ----------
            rel_change = abs(nll_prev - nll_new) / (abs(nll_prev) + 1e-12)
            nll_prev = nll_new
            if rel_change < self.rel_tol:
                break

        return nll_prev

    # ------------------------------------------------------------------
    # prediction --------------------------------------------------------
    # ------------------------------------------------------------------
    def predict(self, X):
        if not self.fitted:
            raise RuntimeError("Model not fitted yet. Call fit() first.")

        X_int = np.column_stack([np.ones(X.shape[0]), X])
        X_int[:, 1:] = self.scaler.transform(X_int[:, 1:])

        theta_pred = np.zeros((X_int.shape[0], self.distribution.n_of_p))
        for p in range(self.distribution.n_of_p):
            eta = X_int @ self.betas[p]
            theta_pred[:, p] = self.distribution.link_inverse(eta, p)
        return theta_pred

    # convenience property ---------------------------------------------
    @property
    def beta(self):
        return {
            f"param_{p}": {
                "intercept": self.betas[p][0],
                "coefficients": self.betas[p][1:]
            }
            for p in range(self.distribution.n_of_p)
        }


# ----------------------------------------------------------------------
# quick smoke‑test ------------------------------------------------------
# ----------------------------------------------------------------------
if __name__ == "__main__":
    X, y = load_diabetes(return_X_y=True)
    model = GAMLSS(param_steps=0.8)  # dampen all parameters to 80 %
    model.fit(X, y)

    print("Final negative log‑likelihood:", model.fit_hist_nll[-1])
    print("Coefficients by parameter (intercept first):")
    for p, coefs in model.beta.items():
        print(p, coefs)
