

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 09:40:26 2025

@author: samue
"""

import numpy as np
from distributions import JohnsonSU

class GAMLSS:
    """GAMLSS with active set CD and likelihood-based convergence"""
    
    def __init__(self, distribution= JohnsonSU(), max_iter_outer=50, max_iter_inner=100,
                 tol=1e-4, lambda_n=100, lambda_eps=1e-4):
        self.distribution = distribution
        self.max_iter_outer = max_iter_outer
        self.max_iter_inner = max_iter_inner
        self.tol = tol
        self.lambda_n = lambda_n
        self.lambda_eps = lambda_eps
        self.betas = {}
        self.bic_values = {}


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

    def _enhanced_lasso_path(self, X, y_, weights, current_beta, p):
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

    def _calculate_bic(self, y, X, theta, beta_path, p, weights):
        """BIC calculation using log-likelihood"""
        n_samples = X.shape[0]
        bic = []
        
        for beta in beta_path:
            # Temporary parameter update
            eta = X @ beta
            theta_temp = self.theta.copy()
            theta_temp[:, p] = self.distribution.link_inverse(eta, p)
            
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
            dll = self.distribution.dll(y, self.theta, p)
            ddll = self.distribution.ddll(y, self.theta, p)
            dr = 1 / self.distribution.link_inverse_derivative(eta, p)
            
            
            # Calculate weights and pseudo-response
            w = np.clip(-ddll / (dr*dr), 1e-10, 1e10)
            y_ = eta + dll / (dr * w)
        
            

            
            # Fit LASSO path with active set updates
            beta_path = self._enhanced_lasso_path(
                X, y_, weights, 
                current_beta=self.coef_.get(p, np.zeros(X.shape[1])),
                p=p
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

