# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 16:53:58 2025

@author: samue
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 09:40:26 2025

@author: samue
"""
import numpy as np
from distributions import JSUo
from sklearn.preprocessing import StandardScaler
from glmnet_lasso import _enhanced_lasso_path
from sklearn.datasets import load_diabetes

class GAMLSS:
    """GAMLSS with active set CD and likelihood-based convergence"""
    
    def __init__(self, distribution= JSUo(), max_iter_outer=30, max_iter_inner=30,
                 abs_tol=1e-4, rel_tol = 1e-5, lambda_n=100, lambda_eps=1e-4):
        self.distribution = distribution
        self.max_iter_outer = max_iter_outer
        self.max_iter_inner = max_iter_inner
        self.abs_tol = abs_tol
        self.rel_tol = rel_tol
        self.tol_cd = 1e-6  
        self.max_iter_cd = 1000
        self.lambda_n = lambda_n
        self.lambda_eps = lambda_eps
        self.betas = {p: None for p in range(self.distribution.n_of_p)}
        self.scaler = StandardScaler()
        self.fitted = False

    def _make_initial_fitted_values(self, y):
        out = np.stack(
            [self.distribution.initial_values(y, p=i)
             for i in range(self.distribution.n_of_p)],
            axis=1,
        )
        return out   
    

    def _log_likelihood(self, y, theta):
        """Compute total log-likelihood for convergence checking"""
        ll_ = -2 * np.log(np.clip(self.distribution.pdf(y, theta), 10e-50, None))
        return np.sum(ll_)

    def _calculate_bic(self, y, X, theta, beta_path, p):
        """BIC calculation using log-likelihood"""
        n_samples = y.shape[0]
        bic = []
        
        for beta in beta_path:
            # Temporary parameter update
            eta = X @ beta
            theta_temp = self.theta.copy()
            theta_temp[:, p] = self.distribution.link_inverse(eta, p)
            
            # Log-likelihood based BIC
            ll = self._log_likelihood(y, theta_temp)
            k = np.sum(beta != 0)
            bic_val = ll + k * np.log(n_samples)
            bic.append(bic_val)
            
        return np.array(bic)


    def fit(self, X, y):
        """Fit model with active set updates and likelihood convergence"""
        self.n_obs = y.shape[0]
        
            
        X_int = np.column_stack([np.ones(X.shape[0]), X])
        X_features = X_int[:, 1:]
        X_features_scaled = self.scaler.fit_transform(X_features)
        X_int[:, 1:] = X_features_scaled
       
        # Initialize parameters   
        self.theta = self._make_initial_fitted_values(y)
        self.ll, self.it_outer = self._outer(X, y)
        
        self.fitted = True
        return print("Fit is done")

    
    
    def _outer(self, X, y): 
        ll = self._log_likelihood(y, self.theta)
        ll_prev = ll
        
        for iter_outer in range(1, self.max_iter_outer+1):

            print(f"Outer iteration: {iter_outer}")
            for p in range(self.distribution.n_of_p):
                ll = self._inner(X=X, y=y, p=p)
                
            print(f"\t  Improvement outer ll {ll - ll_prev}")
        
            if np.abs(ll_prev - ll) / np.abs(ll_prev) < self.rel_tol:
                break
            if np.abs(ll_prev - ll) < self.abs_tol:
                break   
            
            ll_prev = ll
        return ll, iter_outer
    
    def _inner(self, X, y, p):
        ll_inner_prev = self._log_likelihood(y, self.theta)
        
        
        for iter_inner in range(1, self.max_iter_inner+1):
            print(f"\t \t {p} Inner iteration: {iter_inner}")
            eta = self.distribution.link_function(self.theta[:, p], p)
            dll = self.distribution.dll(y, self.theta, p)
            ddll = self.distribution.ddll(y, self.theta, p)
            dth = self.distribution.link_inverse_derivative(eta, p)
            
            v = dll*dth
            w = np.clip(-ddll*dth*dth, 1e-10, 1e10)
            
            y_ = eta + v/w
        
            
            # Fit LASSO path with active set updates
            beta_path = _enhanced_lasso_path(
                X, y_ , w, 
                self.betas[p], 
                self.lambda_eps, self.lambda_n, 
                self.max_iter_cd, self.tol_cd, 
                p=p
            )
            
            # Select best model using BIC
            bic = self._calculate_bic(y_, X, self.theta, beta_path, p)
            best_idx = np.argmin(bic)
            betas_old = self.betas[p].copy() if self.betas[p] is not None else None
            self.betas[p] = beta_path[best_idx]
            
            # Update theta and check inner convergence
            eta_new = np.matmul(X, self.betas[p])
            self.theta[:, p] = self.distribution.link_inverse(eta_new, p)
            ll_inner = self._log_likelihood(y, self.theta)
            
           
            
            if ll_inner > ll_inner_prev:
                print("\t \t \t Likelihood increased - rejecting step")
                if betas_old is not None:
                    self.betas[p] = 0.5 * (self.betas[p] + betas_old)  # Revert halfway
                    eta_new = np.matmul(X, self.betas[p])
                    self.theta[:, p] = self.distribution.link_inverse(eta_new, p)
                    ll_inner = self._log_likelihood(y, self.theta)
            
            print(f"\t \t \t Improvement inner ll {ll_inner}, {ll_inner - ll_inner_prev}")
            if np.abs(ll_inner_prev - ll_inner) / np.abs(ll_inner_prev) < self.rel_tol:
                break
            if np.abs(ll_inner_prev - ll_inner) < self.abs_tol:
                break
            ll_inner_prev = ll_inner
           
        return ll_inner
    
    
    def predict(self, X):
        """
        Predict distribution parameters for new data
        """
        if not self.fitted:
            raise ValueError("Model not fitted yet. Call fit() first")
            
        X_int = np.column_stack([np.ones(X.shape[0]), X])
        X_int[:, 1:] = self.scaler.transform(X_int[:, 1:])
        ps = np.zeros((X.shape[0], self.distribution.n_of_p))
        
        for p in range(self.distribution.n_of_p):
            # Get linear predictor
            eta = X_int @ self.betas[p]
            # Apply inverse link function
            ps[:, p] = self.distribution.link_inverse(eta, p)
        
        #TODO: Add option to return objective value/ error
        return ps
    

    @property
    def beta(self):
        """Return coefficients for each distribution parameter"""
        return {
            f"param_{p}": {
                "intercept": self.betas[p][0],
                "coefficients": self.betas[p][1:]
            } 
            for p in range(self.distribution.n_of_p)
        }

X, y = load_diabetes(return_X_y=True)
gamlss = GAMLSS()
gamlss.fit(X, y)