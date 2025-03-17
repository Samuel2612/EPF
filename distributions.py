# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 11:56:33 2025

@author: samue
"""

import numpy as np
import scipy.stats as st

LOG_LOWER_BOUND = 1e-25
EXP_UPPER_BOUND = 25
SMALL_NUMBER = 1e-10

class link_log:
    """
    The log-link function.

    The log-link function is defined as \(g(x) = \log(x)\).
    """

    def __init__(self):
        pass

    def link(self, x: np.ndarray) -> np.ndarray:
        return np.log(np.clip(x, 1e-20, None))

    def inverse(self, x: np.ndarray) -> np.ndarray:
        return np.clip(
            np.exp(np.clip(x, None, 1e2)),
            1e-20, None
            )

    def inverse_derivative(self, x: np.ndarray) -> np.ndarray:
        return np.exp(np.clip(x, None, 1e2))

    def link_derivative(self, x: np.ndarray) -> np.ndarray:
        return 1 / x

    def link_second_derivative(self, x: np.ndarray) -> np.ndarray:
        return -1 / x**2


class link_id:
    """
    The identity link function.

    The identity link is defined as \(g(x) = x\).
    """

    def __init__(self):
        pass

    def link(self, x: np.ndarray) -> np.ndarray:
        return x

    def inverse(self, x: np.ndarray) -> np.ndarray:
        return x

    def inverse_derivative(self, x: np.ndarray) -> np.ndarray:
        return np.ones_like(x)

    def link_derivative(self, x: np.ndarray) -> np.ndarray:
        return np.ones_like(x)

    def link_second_derivative(self, x: np.ndarray) -> np.ndarray:
        return np.zeros_like(x)





class JSUo:
    """
    A Johnson SU (original) distribution class for GAMLSS-like usage.
    
    This implementation:
      - Defines score (first derivative) for each parameter
      - Defines approximate second derivatives as -(score^2)
      - Uses scipy.stats.johnsonsu for PDF and CDF
      - Allows link functions for each parameter
      - Provides initial values for iterative fitting
    """

    def __init__(
        self,
        loc_link=link_id(),
        scale_link=link_log(),
        skew_link=link_id(),
        kurt_link=link_log(),
    ):
        # We have 4 parameters:
        #   mu, sigma, nu, tau
        # mapped through link functions: loc_link, scale_link, skew_link, kurt_link.
        self.n_of_p = 4
        self.loc_link = loc_link
        self.scale_link = scale_link
        self.skew_link = skew_link
        self.kurt_link = kurt_link
        self.links = [
            self.loc_link,
            self.scale_link,
            self.skew_link,
            self.kurt_link,
        ]


    def theta_to_params(self, theta):
        """
        Extract mu, sigma, nu, tau from theta.

        Parameters
        ----------
        theta : np.ndarray
            shape (n, 4) array of linear-predictor values 
            for [mu, sigma, nu, tau], on the link scale.

        Returns
        -------
        mu, sigma, nu, tau : np.ndarrays, each shape (n,)
            Real-space parameters after link inverse.
        """
        # Each column of theta is the linear predictor for one parameter
        mu = theta[:, 0]
        sigma = theta[:, 1]
        nu = theta[:, 2]
        tau = theta[:, 3]
        return mu, sigma, nu, tau

    def link_function(self, x, p=0):
        """
        Apply link function to raw parameter x.

        Parameters
        ----------
        x : array_like
        param : int
            0 -> mu, 1 -> sigma, 2 -> nu, 3 -> tau
        """
        return self.links[p].link(x)

    def link_inverse(self, x, p=0):
        """
        Apply inverse link function to linear predictor x.

        Parameters
        ----------
        x : array_like
        param : int
            0 -> mu, 1 -> sigma, 2 -> nu, 3 -> tau
        """
        return self.links[p].inverse(x)

    def link_function_derivative(self, x, p=0):
        """
        Derivative of link w.r.t. x
        """
        return self.links[p].link_derivative(x)

    def link_inverse_derivative(self, x, p=0):
        """
        Derivative of inverse link w.r.t. x
        """
        return self.links[p].inverse_derivative(x)


    @staticmethod
    def _dmu(y, mu, sigma, nu, tau):
        """
        d/d(mu) of log-likelihood
        """
        z = (y - mu) / sigma
        r = nu + tau * np.arcsinh(z)
        # Score w.r.t. mu
        return (z / (sigma * (z**2 + 1))) + (r * tau) / (sigma * np.sqrt(z**2 + 1))

    @staticmethod
    def _dsigma(y, mu, sigma, nu, tau):
        """
        d/d(sigma) of log-likelihood
        """
        z = (y - mu) / sigma
        r = nu + tau * np.arcsinh(z)
        # Score w.r.t. sigma
        return (-1.0 / (sigma * (z**2 + 1))) + (r * tau * z) / (sigma * np.sqrt(z**2 + 1))

    @staticmethod
    def _dnu(y, mu, sigma, nu, tau):
        """
        d/d(nu) of log-likelihood
        """
        z = (y - mu) / sigma
        r = nu + tau * np.arcsinh(z)
        # Score w.r.t. nu
        return -r

    @staticmethod
    def _dtau(y, mu, sigma, nu, tau):
        """
        d/d(tau) of log-likelihood
        """
        z = (y - mu) / sigma
        r = nu + tau * np.arcsinh(z)
        # Score w.r.t. tau
        return (1.0 + r * nu - r * r) / tau

    def dll(self, y, theta, p=0):
        """
       first derivative of log-likelihood.

        """
        mu_, sigma_, nu_, tau_ = self.theta_to_params(theta)

        if p == 0:
            return self._dmu(y, mu_, sigma_, nu_, tau_)
        elif p == 1:
            return self._dsigma(y, mu_, sigma_, nu_, tau_)
        elif p == 2:
            return self._dnu(y, mu_, sigma_, nu_, tau_)
        elif p == 3:
            return self._dtau(y, mu_, sigma_, nu_, tau_)
        else:
            raise ValueError("p must be in {0,1,2,3}.")


    def _quasi_newton_score(self, y, theta, p=0):
        """
        Approximate second derivative as - (first_derivative^2).
        
        """
        
        dp = self.dll(y, theta, p=p)
        return -(dp * dp)

    def ddll(self, y, theta, p=0):
        """
        Approx. second derivative of log-likelihood wrt one parameter.

        """
        sc = self._quasi_newton_score(y, theta, p=p)
        
        return np.clip(sc, None, -1e-15)


    def initial_values(self, y, p=0):
        """
        Provide a simple initial guess for each parameter based on data.

        Parameters
        ----------
        y : np.ndarray
        param : int
            0 -> mu, 1 -> sigma, 2 -> nu, 3 -> tau
        axis : optional, used if you want row-wise or column-wise stats

        Returns
        -------
        init : np.ndarray
            shape (n,) array of initial guesses for param on real scale
        """
        if p == 0:
            # Mu initial = average of y
            return np.full_like(y, np.mean(y))
        elif p == 1:
            # Sigma initial = stdev of y
            return np.full_like(y, np.std(y))
        elif p == 2:
            # Nu initial = 0
            return np.zeros_like(y)
        elif p == 3:
            # Tau initial = 10 (arbitrary positive guess)
            return np.full_like(y, 1.0)



    def pdf(self, y, theta):
        """
        Johnson SU pdf at y, using SciPy's johnsonsu.

        Parameters
        ----------
        y : np.ndarray
        theta : np.ndarray, shape (n,4)

        Returns
        -------
        pdf_vals : np.ndarray, shape (n,)
        """
        mu_, sigma_, nu_, tau_ = self.theta_to_params(theta)
        return st.johnsonsu(a=nu_, b=tau_, loc=mu_, scale=sigma_).pdf(y)

    def cdf(self, y, theta):
        """
        Johnson SU cdf at y, using SciPy's johnsonsu.

        Parameters
        ----------
        y : np.ndarray
        theta : np.ndarray, shape (n,4)

        Returns
        -------
        cdf_vals : np.ndarray, shape (n,)
        """
        mu_, sigma_, nu_, tau_ = self.theta_to_params(theta)
        return st.johnsonsu(a=nu_, b=tau_, loc=mu_, scale=sigma_).cdf(y)

    def __repr__(self):
        return (
            f"JSUo("
            f"loc_link={self.loc_link}, "
            f"scale_link={self.scale_link}, "
            f"skew_link={self.skew_link}, "
            f"kurt_link={self.kurt_link}"
            f")"
        )
