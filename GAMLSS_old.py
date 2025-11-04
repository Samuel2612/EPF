# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 09:40:26 2025

@author: samue
"""
import numpy as np
from distributions import JSUo
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_diabetes
#from glmnet_lasso import _enhanced_lasso_path

class GAMLSS:
    """GAMLSS with active set CD and likelihood-based convergence"""
    
    def __init__(self, distribution= JSUo(), max_iter_outer=20, max_iter_inner=30):
        
        self.max_iter_outer = max_iter_outer
        self.max_iter_inner = max_iter_inner
        self.rel_tol = 1e-4
        self.tol_cd = 1e-6  
        self.max_iter_cd = 1000
        self.lambda_n = 100
        self.lambda_eps = 1e-4
        
        self.distribution = distribution
        self.betas = {p: None for p in range(self.distribution.n_of_p)}
        self.scaler = StandardScaler()
        self.fitted = False

    def _init_val(self, y):
        out = np.stack(
            [self.distribution.initial_values(y, p=i)
             for i in range(self.distribution.n_of_p)],
            axis=0,
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
        n_features = X_int.shape[1]

        
        # X = self.scaler.fit_transform(X)
        
       
        # # Initialize parameters   
        for p in range(self.distribution.n_of_p):
            self.betas[p] = np.zeros(n_features)

        self.theta = self.distribution.init_val(y)
        self.ll = self._outer(X_int, y)
        
        self.fitted = True
        print("Fit is done")
        
        return self

    
    
    def _outer(self, X, y): 
        ll = self._log_likelihood(y, self.theta)
        ll_prev = ll
        
        for iter_outer in range(1, self.max_iter_outer+1):

            print(f"Outer iteration: {iter_outer}")
            for p in range(self.distribution.n_of_p):
                ll = self._inner2(X=X, y=y, p=p, iter_outer=iter_outer)
                
           
        
            if np.abs(ll_prev - ll) / np.abs(ll_prev) < self.rel_tol:
                break  
            
            ll_prev = ll
        
        # for iter_outer in range(5):
        #     for p in range(self.distribution.n_of_p):
        #         eta =  self.distribution.link_function(self.theta[:, p], p)
        #         dr = 1 /  self.distribution.link_inverse_derivative(eta, p)
        #         ddr = self.distribution.link_inverse_second_derivative(eta, p)
        #         dl1dp1 = self.distribution.dll(y, self.theta, p)
        #         dl2dp2 = self.distribution.ddll(y, self.theta, p)
        #         wt = -(dl2dp2 / (dr * dr)) 
        #         wt = np.clip(wt, 1e-10, 1e10)
        #         y_ = eta + dl1dp1 / (dr * wt)
            
        #         # Fit y_ to X with prior weights wt using LASSO with sklearn
        #         lasso_cv = LassoCV(
        #             fit_intercept=False,  # We already added a column of 1's to X
        #             cv=5,                 #5-fold CV
        #             n_alphas=100,         # or choose your alpha grid/logspace
        #             random_state=1
        #         )
                
        #         lasso_cv.fit(X, y_, sample_weight=wt)
    
        #         self.betas[p] = lasso_cv.coef_.copy()
        #         eta_new = X @ self.betas[p]
        #         self.theta[:, p] = self.distribution.link_inverse(eta_new, p)
                
                
        #         self.betas[p] = lasso_cv.coef_.copy()
                
            
        ll = self._log_likelihood(y, self.theta)
        
        print(f"Final neg LL: {ll}")
            
        return ll
    
    def _inner(self, X, y, p, iter_outer):
        ll_inner_prev = self._log_likelihood(y, self.theta)
        
        
        for iter_inner in range(self.max_iter_inner):
            # Store old values in case we need to revert
            # old_beta_p = self.betas[p].copy()   # old coefficient vector for parameter p
            # old_theta_p = self.theta[:, p].copy()   # old theta values for parameter p
            # old_ll = ll_inner_prev
            
           
            # eta = self.distribution.link_function(self.theta[:, p], p)
            # dll = self.distribution.dll(y, self.theta, p)
            # ddll = self.distribution.ddll(y, self.theta, p)
            # dth = self.distribution.link_inverse_derivative(eta, p)
            
            # v = dll*dth
            # w = np.clip(-ddll*dth*dth, 1e-10, 1e10)
            
            # y_ = eta + v/w
            print(f"\t \t {p} Inner iteration: {iter_inner}")
            
            
            eta =  self.distribution.link_function(self.theta[:, p], p)
            dr = 1 /  self.distribution.link_inverse_derivative(eta, p)
            # ddr = self.distribution.link_inverse_second_derivative(eta, p)
            dl1dp1 = self.distribution.dll(y, self.theta, p)
            dl2dp2 = self.distribution.ddll(y, self.theta, p)
            wt = -(dl2dp2 / (dr * dr))
            wt = np.clip(wt, 1e-10, 1e10)
            y_ = eta + dl1dp1 / (dr * wt)
        
            # Fit y_ to X with prior weights wt using LASSO with sklearn
            lasso_cv = LassoCV(
                        fit_intercept=False,  # We already added a column of 1's to X
                        cv=5,                 #5-fold CV
                        n_alphas = 1000,
                        random_state=1
                    )
            
            lasso_cv.fit(X, y_, sample_weight=wt)
            self.betas[p] = lasso_cv.coef_.copy()
            
            # proposed_beta = lasso_cv.coef_.copy()

            # # Possibly do line search:
            # alpha = 1.0
            # for _ in range(10):
            #     # Proposed partial update
            #     new_beta_p = old_beta_p + alpha*(proposed_beta - old_beta_p)
            #     eta_new = X @ new_beta_p
            #     theta_new = self.distribution.link_inverse(eta_new, p)
                
            #     # Temporarily update self.theta
            #     self.theta[:, p] = theta_new
            #     ll_inner = self._log_likelihood(y, self.theta)
                
            #     if ll_inner < old_ll:
            #         # This is an improvement in the -2 log-likelihood sense
            #         # (or log-likelihood got better if you are using the sum(-2 ln(pdf)) as an objective).
            #         self.betas[p] = new_beta_p
            #         break
            #     else:
            #         # Step-halving: revert back but shrink alpha
            #         self.theta[:, p] = old_theta_p
            #         alpha /= 2.0
            
            # # after step halving, if no improvement found, revert
            # if ll_inner >= old_ll:
            #     # revert to old parameters
            #     self.betas[p] = old_beta_p
            #     self.theta[:, p] = old_theta_p
            #     ll_inner = old_ll

            
            
            # # Fit LASSO path with active set updates
            # beta_path = _enhanced_lasso_path(
            #     X, y_ , wt, 
            #     self.betas[p], 
            #     self.lambda_eps, self.lambda_n, 
            #     self.max_iter_cd, self.tol_cd, 
            #     p=p
            # )
            
            # # Select best model using BIC
            # bic = self._calculate_bic(y_, X, self.theta, beta_path, p)
            # best_idx = np.argmin(bic)
            # self.betas[p] = beta_path[best_idx]
            
            # Update theta and check inner convergence
            eta_new = X @ self.betas[p]
            self.theta[:, p] = self.distribution.link_inverse(eta_new, p)
            ll_inner = self._log_likelihood(y, self.theta)
            
            
            

        
            print(f"\t \t \t Improvement inner ll {ll_inner}, {ll_inner - ll_inner_prev}")
         
        
            # if ll_inner > ll_inner_prev:
            #     # The objective (which we want to minimize) got worse.
            #     # Revert to old values.
            #     self.betas[p] = old_beta_p
            #     self.theta[:, p] = old_theta_p
            #     ll_inner = old_ll
            #     print("\t     Reverting last update (objective increased).")
            #     break
                
            

            if (np.abs(ll_inner_prev - ll_inner) / np.abs(ll_inner_prev) < self.rel_tol or np.abs(ll_inner_prev - ll_inner) < self.abs_tol) and iter_outer + iter_inner >= 1:
                break
            
            ll_inner_prev = ll_inner
           
        return ll_inner
    
    def _inner2(self, X, y, p, iter_outer):
        ll_inner_prev = self._log_likelihood(y, self.theta)
    
        for iter_inner in range(1, self.max_iter_inner + 1):
            print(f"\t \t {p} Inner iteration: {iter_inner}")
            old_beta_p = self.betas[p].copy()
            old_theta_p = self.theta[:, p].copy()
            old_ll = ll_inner_prev
    
            eta = self.distribution.link_function(self.theta[:, p], p)
            dr = 1.0 / self.distribution.link_inverse_derivative(eta, p)
            # ddr = self.distribution.link_inverse_second_derivative(eta, p)
            dl1dp1 = self.distribution.dll(y, self.theta, p)
            dl2dp2 = self.distribution.ddll(y, self.theta, p)
    
            wt = -(dl2dp2 / (dr * dr))
            wt = np.clip(wt, 1e-10, 1e10)  # keep IRLS weights in a safe numeric range
            y_ = eta + dl1dp1 / (dr * wt)
    
            # lasso_cv = LassoCV(
            #     fit_intercept=False,
            #     cv=5,
            #     n_alphas=100,
            #     max_iter=1000,
            #     random_state=1
            # )
            
            lasso_cv = LinearRegression(
                fit_intercept=False,  # We already added a column of 1's to X
         
            )
            lasso_cv.fit(X, y_, sample_weight=wt)
            new_beta_p = lasso_cv.coef_.copy()
    
            self.betas[p] = new_beta_p
            eta_new = X @ self.betas[p]
            self.theta[:, p] = self.distribution.link_inverse(eta_new, p)
    
            # compute new deviance
            ll_inner = self._log_likelihood(y, self.theta)
    
            # Only break if (iter_inner>1 or iter_outer>1) AND ll got worse
            # Here, suppose we store the outer iteration in self.current_outer_iter
            # or pass it as a parameter. For example, let's assume we have
            # 'self.current_outer_iter' telling us which outer iteration we are in.
            if (iter_inner > 1 or iter_outer > 1):
                if ll_inner > old_ll*1.25:
                    print(f"Reverting update (LL worsened) at inner={iter_inner} p={p}")
                    self.betas[p] = old_beta_p
                    self.theta[:, p] = old_theta_p
                    ll_inner = old_ll
                    break  # do not proceed further in this inner loop
                    
            
            # print(f"\t \t \t Improvement inner ll {ll_inner}, {ll_inner - ll_inner_prev}")
    

            if iter_inner > 1:
                if (abs(ll_inner_prev - ll_inner) / abs(ll_inner_prev) )< self.rel_tol:
                    # print('breakkkkkkk')
                    break
    
            ll_inner_prev = ll_inner
    
        return ll_inner
       
    def predict(self, X):
        """
        Predict distribution parameters for new data
        """
        if not self.fitted:
            raise ValueError("Model not fitted yet. Call fit() first")
            
        X_int = np.concatenate(([1], X))
        X_int[1:] = self.scaler.transform(X_int[1:].reshape(1, -1))
        ps = np.zeros(self.distribution.n_of_p)
        
        for p in range(self.distribution.n_of_p):
            # Get linear predictor
            eta = X_int @ self.betas[p]
            # Apply inverse link function
            ps[p] = self.distribution.link_inverse(eta, p)
        
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



# X, y = load_diabetes(return_X_y=True)
# # X, y = fetch_california_housing(return_X_y=True)
# gamlss = GAMLSS()
# gamlss.fit(X, y)
# print(np.vstack([*gamlss.betas.values()]).T)
