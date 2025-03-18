# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 15:12:47 2025

@author: samue

"""
import numpy as np

def _soft_threshold(x, lambd):
    return np.sign(x) * np.maximum(np.abs(x) - lambd, 0)

def _enhanced_lasso_path( X, y_, weights, current_beta,lambda_eps, lambda_n, max_iter_cd, tol_cd, p,  f = 1):
    """Advanced coordinate descent with active set and warm starts"""
    n_samples, n_features = X.shape

    # Lambda path setup
    lambda_max = np.max(np.abs(X[:, 1:].T @ y_)) / n_samples
    print(f"Lmabda max : {lambda_max}")
    lambda_path = np.geomspace(lambda_max, lambda_max*lambda_eps, lambda_n)
    
    beta_path = np.zeros((lambda_n, n_features))
    active_set = np.zeros(n_features, dtype=bool)
    
    # Warm start with previous beta
    if  current_beta is not None and np.any(current_beta != 0):
        beta = current_beta.copy()
        active_set = beta != 0
    else:
        beta = np.zeros(n_features)
    
    for i, lambd in enumerate(lambda_path):
            
        for _ in range(max_iter_cd):
            beta_prev = beta.copy()
            
            # Full cycle first iteration, then active set
            features = np.arange(n_features) if _ == 0 else np.where(active_set)[0]
            
            for j in features:
                x_j = X[:, j]
                r = y_- X @ beta 
                wr = weights * r
                wx = np.dot(weights, np.square(x_j)) / n_samples
                z = np.dot(x_j, wr) / n_samples + f*wx*beta[j]
                if j == 0:
                    # Intercept update (no L1 penalty)
                    beta[j] = z / (f * wx)
                    active_set[j] = True
        
                else:
                    # For other coefficients, apply soft-threshold
                    beta[j] = _soft_threshold(z, lambd) / (f * wx)
                    active_set[j] = (abs(beta[j]) > 1e-30)

            
            # Check convergence
            if np.linalg.norm(beta - beta_prev, ord = np.inf) < tol_cd:
                break
            
            #TODO: Early exit if no active features


            

        beta_path[i] = beta.copy()
            
    return beta_path


