# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 09:40:26 2025

@author: samue
"""

import numpy as np
import scipy.stats as st
from rolch.base import Distribution
from rolch.link import IdentityLink, LogLink

class MyGAMLSS:
    """GAMLSS with active set CD and likelihood-based convergence"""
    
    def __init__(self, distribution=JohnsonSU(), max_iter_outer=50, max_iter_inner=100,
                 tol=1e-4, lambda_n=100, lambda_eps=1e-4):
        self.distribution = distribution
        self.max_iter_outer = max_iter_outer
        self.max_iter_inner = max_iter_inner
        self.tol = tol
        self.lambda_n = lambda_n
        self.lambda_eps = lambda_eps
        self.coef_ = {}
        self.bic_values = {}
        self.loglik_history = []

    def fit(self, X, y, weights = None):
        """Fit model with active set updates and likelihood convergence"""
        self.n_observations = y.shape[0]
        
        if weights is not None:
            weights  = weights 
        else:
            weights  = np.ones(y.shape[0])
        
        # Initialize parameters
        self.theta = self.distribution.initial_values(y)
        self.ll, self.it_outer = self._outer_fit(X, y, weights)
        

        return print("Fit is done")

    def _enhanced_lasso_path(self, X, y_, weights, current_beta, param):
        """Advanced coordinate descent with active set and warm starts"""
        n_samples, n_features = X.shape

        
        # Lambda path setup
        lambda_max = np.max(np.abs(X.T @ y_)) / n_samples
        lambda_path = np.geomspace(lambda_max, lambda_max*self.lambda_eps, self.lambda_n)
        
        beta_path = np.zeros((self.lambda_n, n_features))
        active_set = np.zeros(n_features, dtype=bool)
        
        # Warm start with previous beta
        if np.any(current_beta != 0):
            beta = current_beta.copy()
            active_set = beta != 0
        else:
            beta = np.zeros(n_features)
        
        for i, lambd in enumerate(lambda_path):
            n_active = sum(active_set)
            
            for _ in range(self.max_iter_cd):
                beta_prev = beta.copy()
                
                # Full cycle first iteration, then active set
                features = np.arange(n_features) if _ == 0 else np.where(active_set)[0]
                
                for j in features:
                    x_j = X[:, j]
                    r = y_- X @ beta + x_j * beta[j]
                    wr = np.multiply(weights, r)
                    update = np.dot(x_j, wr) / n_samples
                    beta[j] = self._soft_threshold(update, lambd) / (np.dot(weights, np.square(x_j))/n_samples)
                    
                    # Update active set
                    active_set[j] = np.abs(beta[j]) > 1e-10
                
                # Check convergence
                if np.linalg.norm(beta - beta_prev) < self.tol_cd:
                    break
                
                # Early exit if no active features
                if n_active == 0 and i > 0:
                    beta_path[i:] = 0
                    return beta_path
                
            beta_path[i] = beta
            if i > 0 and np.allclose(beta_path[i], beta_path[i-1]):
                beta_path[i:] = beta_path[i]
                break
                
        return beta_path

    def _log_likelihood(self, y, theta, weights):
        """Compute total log-likelihood for convergence checking"""
        ll_ = -2 * np.log(self.distribution.pdf(y, theta))
        return np.sum(weights*ll_)

    def _calculate_bic(self, y, X, theta, beta_path, param, weights):
        """BIC calculation using log-likelihood"""
        n_samples = X.shape[0]
        bic = []
        
        for beta in beta_path:
            # Temporary parameter update
            eta = X @ beta
            theta_temp = theta.copy()
            theta_temp[:, param] = self.distribution.link_inverse(eta, param)
            
            # Log-likelihood based BIC
            ll = self._log_likelihood(y, theta_temp)
            k = np.sum(beta != 0)
            bic_val = -2 * ll + k * np.log(n_samples)
            bic.append(bic_val)
            
        return np.array(bic)

    @staticmethod
    def _soft_threshold(x, lambd):
        return np.sign(x) * np.maximum(np.abs(x) - lambd, 0)
    
    def _outer(self, X, y, weights): 
        ll = self._log_likelihood(y, self.theta, weights)
        ll_prev = ll
        
        for it_outer in range(1, self.max_it_outer+1):


            for p in range(self.distribution.n_p):
                ll = self._inner_fit(X=X, y=y, weights=weights, p=p,
                                             it_outer=it_outer, ll_inner=ll)
                
            if np.abs(ll_prev - ll) / np.abs(ll_prev) < self.rel_tol_outer:
                break
            if np.abs(ll_prev - ll) < self.abs_tol_outer:
                break
            
            
            
        return ll, it_outer
    
    def _inner(self, X, y, weights, p, it_outer, ll_inner):
        ll_inner_prev = self._log_likelihood(y, self.theta, weights)
        
        
        for it_inner in range(1, self.max_it_inner+1):
            eta = self.distribution.link_function(self.theta[:, p], p)
            dll1 = self.distribution.dll1(y, self.theta, p)
            dll2 = self.distribution.dll2(y, self.theta, p)
            dr = 1 / self.distribution.link_inverse_derivative(eta, p)
            
            # Calculate weights and pseudo-response
            w = np.clip(-dll2 / (dr**2), 1e-10, 1e10)
            y_ = eta + dll1 / (dr * w)
            
            # Fit LASSO path with active set updates
            beta_path = self._enhanced_lasso_path(
                X, y_, weights, 
                current_beta=self.coef_.get(p, np.zeros(X.shape[1])),
                param=p
            )
            
            # Select best model using BIC
            bic = self._calculate_bic(y_, X, self.theta, beta_path, p, weights)
            best_idx = np.argmin(bic)
            self.coef_[p] = beta_path[best_idx]
            self.bic_values[p] = bic[best_idx]
            
            # Update theta and check inner convergence
            eta_new = np.matmul(X, self.betas_[p])
            self.theta[:, p] = self.distribution.link_inverse(eta_new, p)
            
            ll_inner = self._log_likelihood(y, self.theta, weights = weights)
            
            if np.abs(ll_inner_prev - ll_inner) / np.abs(ll_inner_prev) < self.rel_tol_inner:
                break
            if np.abs(ll_inner_prev - ll_inner) < self.abs_tol_inner:
                break
            
            ll_inner_prev = ll_inner
           
        return ll_inner
    
    
    def predict(self, X):
        """
        Predict distribution parameters for new data
        Args:
            X: Feature matrix (n_samples, n_features)
            return_params: If True returns distribution parameters,
                        if False returns distribution object
        Returns:
            params (np.ndarray): (n_samples, n_params) array of distribution parameters
                        or
            dist: scipy distribution object
        """
        if not self.fitted:
            raise ValueError("Model not fitted yet. Call fit() first")
            
        X_aug = np.hstack([np.ones((X.shape[0], 1)), X])
        params = np.zeros((X.shape[0], self.distribution.n_params))
        
        for p in range(self.distribution.n_params):
            # Get linear predictor
            eta = X_aug @ self.coef_[p]
            # Apply inverse link function
            params[:, p] = self.distribution.link_inverse(eta, p)
            

        return params

    @property
    def betas(self):
        """Return coefficients for each distribution parameter"""
        return {
            f"param_{p}": {
                "intercept": self.coef_[p][0],
                "coefficients": self.coef_[p][1:]
            } 
            for p in range(self.distribution.n_params)
        }

LOG_LOWER_BOUND = 1e-25
EXP_UPPER_BOUND = 25
SMALL_NUMBER = 1e-10

class LogLink:
    """
    The log-link function.

    The log-link function is defined as \(g(x) = \log(x)\).
    """

    def __init__(self):
        pass

    def link(self, x: np.ndarray) -> np.ndarray:
        return np.log(np.fmax(x, LOG_LOWER_BOUND))

    def inverse(self, x: np.ndarray) -> np.ndarray:
        return np.fmax(
            np.exp(np.fmin(x, EXP_UPPER_BOUND)),
            LOG_LOWER_BOUND,
        )

    def inverse_derivative(self, x: np.ndarray) -> np.ndarray:
        return np.exp(np.fmin(x, EXP_UPPER_BOUND))

    def link_derivative(self, x: np.ndarray) -> np.ndarray:
        return 1 / x

    def link_second_derivative(self, x: np.ndarray) -> np.ndarray:
        return -1 / x**2


class IdentityLink:
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


class DistributionJSU:
    """
    Corresponds to GAMLSS JSUo()
    """

    def __init__(
        self,
        loc_link=IdentityLink(),
        scale_link=LogLink(),
        shape_link=IdentityLink(),
        tail_link=LogLink(),
    ):
        self.n_params = 4
        self.loc_link = loc_link
        self.scale_link = scale_link
        self.shape_link = shape_link
        self.tail_link = tail_link
        self.links = [
            self.loc_link,
            self.scale_link,
            self.shape_link,  # skew
            self.tail_link,  # tail
        ]

    def theta_to_params(self, theta):
        mu = theta[:, 0]
        sigma = theta[:, 1]
        nu = theta[:, 2]
        tau = theta[:, 3]
        return mu, sigma, nu, tau

    def dl1_dp1(self, y, theta, param=0):
        mu, sigma, nu, tau = self.theta_to_params(theta)
       
        if param == 0:
            # MU
            z = (y - mu) / sigma
            r = nu + tau * np.arcsinh(z)
            dldm = (z / (sigma * (z * z + 1))) + (
                (r * tau) / (sigma * np.sqrt(z * z + 1))
            )
            return dldm
       
        if param == 1:
            # SIGMA
            z = (y - mu) / sigma
            r = nu + tau * np.arcsinh(z)
            dldd = (-1 / (sigma * (z * z + 1))) + (
                (r * tau * z) / (sigma * np.sqrt(z * z + 1))
            )
            return dldd
       
        if param == 2:
            # nu
            z = (y - mu) / sigma
            r = nu + tau * np.arcsinh(z)
            dldv = -r
            return dldv
       
        if param == 3:
            # tau
            z = (y - mu) / sigma
            r = nu + tau * np.arcsinh(z)
            dldt = (1 + r * nu - r * r) / tau
            return dldt
       
    def dl2_dp2(self, y, theta, param=0):
        mu, sigma, nu, tau = self.theta_to_params(theta)
        if param == 0:
            # MU
            z = (y - mu) / sigma
            r = nu + tau * np.arcsinh(z)
            dldm = (z / (sigma * (z * z + 1))) + (
                (r * tau) / (sigma * np.sqrt(z * z + 1))
            )
            d2ldm2 = -dldm * dldm
            # d2ldm2 = np.max(1e-15, d2ldm2)
            return d2ldm2
       
        if param == 1:
            # SIGMA
            z = (y - mu) / sigma
            r = nu + tau * np.arcsinh(z)
            dldd = (-1 / (sigma * (z * z + 1))) + (
                (r * tau * z) / (sigma * np.sqrt(z * z + 1))
            )
            d2ldd2 = -(dldd * dldd)
            # d2ldd2 = np.max(d2ldd2, -1e-15)
            return d2ldd2
       
        if param == 2:
            # TAIL
            z = (y - mu) / sigma
            r = nu + tau * np.arcsinh(z)
            d2ldv2 = -(r * r)
            # d2ldv2 = np.max(d2ldv2 < -1e-15)
            return d2ldv2
        if param == 3:
            z = (y - mu) / sigma
            r = nu + tau * np.arcsinh(z)
            dldt = (1 + r * nu - r * r) / tau
            d2ldt2 = -dldt * dldt
            # d2ldt2 = np.max(d2ldt2, -1e-15)
            return d2ldt2
       
    def dl2_dpp(self, y, theta, params=(0, 1)):
        mu, sigma, nu, tau = self.theta_to_params(theta)
        if sorted(params) == [0, 1]:
            z = (y - mu) / sigma
            r = nu + tau * np.arcsinh(z)
            dldm = (z / (sigma * (z * z + 1))) + (
                (r * tau) / (sigma * np.sqrt(z * z + 1))
            )
            dldd = (-1 / (sigma * (z * z + 1))) + (
                (r * tau * z) / (sigma * np.sqrt(z * z + 1))
            )
            d2ldmdd = -(dldm * dldd)
            return d2ldmdd
       
        if sorted(params) == [0, 2]:
            z = (y - mu) / sigma
            r = nu + tau * np.arcsinh(z)
            dldm = (z / (sigma * (z * z + 1))) + (
                (r * tau) / (sigma * np.sqrt(z * z + 1))
            )
            dldv = -r
            d2ldmdv = -(dldm * dldv)
            return d2ldmdv
       
        if sorted(params) == [0, 3]:
            z = (y - mu) / sigma
            r = nu + tau * np.arcsinh(z)
            dldm = (z / (sigma * (z * z + 1))) + (
                (r * tau) / (sigma * np.sqrt(z * z + 1))
            )
            dldt = (1 + r * nu - r * r) / tau
            d2ldmdt = -(dldm * dldt)
            return d2ldmdt
       
        if sorted(params) == [1, 2]:
            z = (y - mu) / sigma
            r = nu + tau * np.arcsinh(z)
            dldd = (-1 / (sigma * (z * z + 1))) + (
                (r * tau * z) / (sigma * np.sqrt(z * z + 1))
            )
            dldv = -r
            d2ldddv = -(dldd * dldv)
            return d2ldddv
       
        if sorted(params) == [1, 3]:
            z = (y - mu) / sigma
            r = nu + tau * np.arcsinh(z)
            dldd = (-1 / (sigma * (z * z + 1))) + (
                (r * tau * z) / (sigma * np.sqrt(z * z + 1))
            )
            dldt = (1 + r * nu - r * r) / tau
            d2ldddt = -(dldd * dldt)
            return d2ldddt
       
        if sorted(params) == [2, 3]:
            z = (y - mu) / sigma
            r = nu + tau * np.arcsinh(z)
            dldv = -r
            dldt = (1 + r * nu - r * r) / tau
            d2ldvdt = -(dldv * dldt)
            return d2ldvdt
       
    def link_function(self, y, param=0):
        return self.links[param].link(y)
       
    def link_inverse(self, y, param=0):
        return self.links[param].inverse(y)
       
    def link_function_derivative(self, y: np.ndarray, param: int = 0) -> np.ndarray:
        return self.links[param].link_derivative(y)
       
    def link_inverse_derivative(self, y: np.ndarray, param: int = 0) -> np.ndarray:
        return self.links[param].inverse_derivative(y)
       
    def initial_values(self, y, param=0, axis=None):
        if param == 0:
            return np.repeat(np.mean(y, axis=None), y.shape[0])
        if param == 1:
            return np.repeat(np.std(y, axis=None), y.shape[0])
        if param == 2:
            return np.full_like(y, 0)
        if param == 3:
            return np.full_like(y, 10)
       
    def cdf(self, y, theta):
        mu, sigma, nu, tau = self.theta_to_params(theta)
        return st.johnsonsu(
            loc=mu,
            scale=sigma,
            a=nu,
            b=tau,
        ).cdf(y)
       
    def pdf(self, y, theta):
        mu, sigma, nu, tau = self.theta_to_params(theta)
        return st.johnsonsu(
            loc=mu,
            scale=sigma,
            a=nu,
            b=tau,
        ).pdf(y)