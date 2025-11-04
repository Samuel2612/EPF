# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 09:40:26 2025

@author: samue
"""
import numpy as np
from distributions import JSUo, JSU, NO, NO2
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_diabetes
#from glmnet_lasso import _enhanced_lasso_path

class GAMLSS:
    """GAMLSS with active set CD and likelihood-based convergence"""
    
    def __init__(
            self, 
            distribution= JSUo(), 
            max_iter_outer=25, 
            max_iter_inner=25,
            step_shrink: float = 0.5,
            min_step: float = 1 / 1024,
            verbose = False
            ):
        
        self.max_iter_outer = max_iter_outer
        self.max_iter_inner = max_iter_inner
        self.rel_tol = 1e-4
        #self.tol_cd = 1e-6                 #only needed when using glmnet_lasso
        #self.max_iter_cd =                 #only needed when using glmnet_lasso
        #self.lambda_n = 100                #only needed when using glmnet_lasso            
        #self.lambda_eps = 1e-4             #only needed when using glmnet_lasso        
        
        self.step_shrink = step_shrink
        self.min_step = min_step
        
        self.distribution = distribution
        self.betas = {p: None for p in range(self.distribution.n_of_p)}
        self.scaler = StandardScaler()
        self.fitted = False
        self.verbose = verbose
        self.verboses = True

    

    def _log_likelihood(self, y, theta):
        """Compute total log-likelihood for convergence checking"""
        ll_ = -2*np.log(np.clip(self.distribution.pdf(y, theta), 10e-30, None))
        return np.sum(ll_)

    def _calculate_bic(self, y, X, theta, beta_path, p):
        """BIC calculation using log-likelihood (currently not used)"""
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
    
    def _damped_update_(self, p, beta_old, beta_new_p, X, y, theta_old):
        """Line‑search step factor

        """       
            
        
        step = 1.0
        ll_old = self._log_likelihood(y, theta_old)

        direction = beta_new_p - beta_old
        beta_new = beta_new_p.copy()

        theta_candidate = theta_old.copy()

        while step >= self.min_step:
            eta_new = X @ beta_new
            theta_candidate[:, p] = self.distribution.link_inverse(eta_new, p)
            ll_new = self._log_likelihood(y, theta_candidate)

            if ll_new <= ll_old:  
                if self.verbose:
                    print(
                        "Line_search effective"
                    )
                return beta_new, theta_candidate, ll_new

            # Otherwise shrink step and try again
            step *= self.step_shrink
            beta_new = beta_old + step * direction

        # If all else fails, keep the old values (should rarely happen)
        if self.verbose:
            print(
                f"[WARN] Parameter {p}: step fell below min_step; keeping old coefficients."
            )
        return beta_old, theta_old, ll_old
    


    def _damped_update(self, p, beta_old, beta_full, X, y, theta_old):
        """
        RS-style step control:
          – Start with step = 1.0
          – Blend new & old linear predictors:  η = step·η_new + (1-step)·η_old
          – Halve `step` until deviance falls or the cap on halvings is reached
          – Return *the best* candidate found (never reverts to β_old unless all fail)
        """
        max_halves = 5
        step = 1.0
    
        eta_old = X @ beta_old
        eta_full = X @ beta_full
        best_ll = self._log_likelihood(y, theta_old)
        best_beta, best_theta = beta_old, theta_old
    
        for _ in range(max_halves + 1):
            eta_old = step*eta_full +  (1 - step)*eta_old
            theta_cand = theta_old.copy()
            theta_cand[:, p] = self.distribution.link_inverse(eta_old, p)
            ll = self._log_likelihood(y, theta_cand)
    
            if ll < best_ll :    
                beta_cand = step * beta_full + (1.0 - step) * beta_old
                best_ll, best_beta, best_theta = ll, beta_cand, theta_cand
                break
    
            step *= 0.5                       # halve and try again
    
        return best_beta, best_theta, best_ll
    
   


    def fit(self, X, y):
        """Fit model with active set updates and likelihood convergence"""
        self.n_obs = y.shape[0]
      
        
            
        X_int = np.column_stack([np.ones(X.shape[0]), X])
        X_features = X_int[:, 1:]
        X_features_scaled = self.scaler.fit_transform(X_features)
        X_int[:, 1:] = X_features_scaled
        n_features = X_int.shape[1]
        
       
        # # Initialize parameters   
        for p in range(self.distribution.n_of_p):
            self.betas[p] = np.zeros(n_features)

        self.theta = self.distribution.init_val(y)
        
        self.init_ll= self._log_likelihood(y, self.theta)
        
        if self.verbose:
            print(f"Starting LL = {self.init_ll}")
        
        self.ll = self._outer(X_int, y)
        
        
        self.fitted = True
        # print("Fit is done")
        
        return self

    
    
    def _outer(self, X, y): 

        
        for iter_outer in range(1, self.max_iter_outer+1):
            ll_old = self._log_likelihood(y, self.theta)
            for p in range(self.distribution.n_of_p):
                ll = self._inner(X=X, y=y, p=p, iter_outer=iter_outer)
                # Convergence check every full sweep
            
            ll_curr = self._log_likelihood(y, self.theta)
            if self.verbose:
                print(
                    f"Outer {iter_outer:02d}: deviance = {ll_curr:.4f} (Δ = {ll_old - ll_curr:+.4e})"
                )
           
        
            if np.abs(ll_old - ll_curr) / np.abs(ll_old) < self.rel_tol:
                # print(iter_outer)
                break  
            
            ll_prev = ll
            
            
            
        # # for possible feature selection afterwards
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
        
        print(f"Final dev: {ll:.4f}. Δ = {self.init_ll - ll:.4f}")
            
        return ll
    
    def _rs_eta_halving(
        self, p: int,
        eta_old: np.ndarray,
        eta_prop: np.ndarray,
        theta_old: np.ndarray,
        y: np.ndarray,
        armijo_c: float = 1e-4,
        max_halving: int = 5,
    ):
        """
        Returns (step, theta_new, ll_new) with the first step that satisfies
        the Armijo proportional-decrease condition.  If none do, the best
        deviance found is used.
    
        Parameters
        ----------
        eta_old   : X @ beta_old
        eta_prop  : X @ beta_prop          (full Newton/IRLS step)
        """
        ll_old   = self._log_likelihood(y, theta_old)
        best_ll  = ll_old
        best_step = 0.0
    
        d_eta = eta_prop - eta_old
        # directional derivative  –u·g′·dη  at the *current* point
        score     = -self.distribution.dll(y, theta_old, p) * \
                     self.distribution.link_inverse_derivative(eta_old, p)
        dir_deriv = np.dot(score, d_eta)
    
        step = 1.0
        for _ in range(max_halving + 1):
            eta_new = eta_old + step * d_eta
            theta_new = theta_old.copy()
            theta_new[:, p] = self.distribution.link_inverse(eta_new, p)
    
            ll_new = self._log_likelihood(y, theta_new)
    
            if ll_new <= ll_old + armijo_c * step * dir_deriv:
                return step, theta_new, ll_new          # accepted
    
            if ll_new < best_ll:
                best_ll   = ll_new
                best_step = step
    
            step *= 0.5                                # RS “autostep”
    
        # all Armijo tests failed – fall back to the best deviance seen
        eta_best = eta_old + best_step * d_eta
        theta_best = theta_old.copy()
        theta_best[:, p] = self.distribution.link_inverse(eta_best, p)
        return best_step, theta_best, best_ll
    
    def _inner_(self, X, y, p, iter_outer):
       
        for iter_inner in range(1, self.max_iter_inner + 1):
        
            ll_inner_prev = self._log_likelihood(y, self.theta)
            beta_old = self.betas[p].copy()
            theta_old = self.theta.copy()
            
            eta = self.distribution.link_function(self.theta[:, p], p)
            dr = 1.0 / self.distribution.link_inverse_derivative(eta, p)
            # ddr =  1/self.distribution.link_inverse_second_derivative(eta, p)
            dl1dp1 = self.distribution.dll(y, self.theta, p)
            dl2dp2 = self.distribution.ddll(y, self.theta, p)
            dl2dp2 = np.clip(dl2dp2, None, -1e-15)
            
            wt = -(dl2dp2 / (dr * dr))
            wt = np.clip(wt, 1e-10, 1e10)  # keep IRLS weights in a safe numeric range
            y_ = eta + dl1dp1 / (dr * wt)
            
            y_ = np.nan_to_num(y_, nan=0.0, posinf=1e10, neginf=-1e10)
            wt = np.nan_to_num(wt, nan=1.0,  posinf=1e10, neginf=1e-10)
            
            irls = LinearRegression(fit_intercept=False)
            irls.fit(X, y_, sample_weight=wt)
            beta_p = irls.coef_.copy()              
            
            eta_old  = X @ beta_old
            eta_prop = X @ beta_p
            
            step, theta_new, ll_inner = self._rs_eta_halving(
            p, eta_old, eta_prop, theta_old, y
            )
            
            self.betas[p] = beta_p          # store the *unshrunk* coefficients
            self.theta    = theta_new
            
            
            if self.verbose:
                print(
                    f"    p={p}, inner={iter_inner:02d}: dev = {ll_inner:.4f}, step = "
                    f"{np.max(np.abs(beta_new - beta_old)):+.2e}"
                )
                
                           
            if iter_inner > 1:
                if (abs(ll_inner_prev - ll_inner) / abs(ll_inner_prev) ) < self.rel_tol:
                    break
            
            ll_inner_prev = ll_inner
        
        return ll_inner
    
    
    def _inner(self, X, y, p, iter_outer):
        
    
        for iter_inner in range(1, self.max_iter_inner + 1):

            ll_inner_prev = self._log_likelihood(y, self.theta)
            beta_old = self.betas[p].copy()
            theta_old = self.theta.copy()
    
            eta = self.distribution.link_function(self.theta[:, p], p)
            dr = 1.0 / self.distribution.link_inverse_derivative(eta, p)
            # ddr =  1/self.distribution.link_inverse_second_derivative(eta, p)
            dl1dp1 = self.distribution.dll(y, self.theta, p)
            dl2dp2 = self.distribution.ddll(y, self.theta, p)
            dl2dp2 = np.clip(dl2dp2, None, -1e-15)
    
            wt = -(dl2dp2 / (dr * dr))
            wt = np.clip(wt, 1e-10, 1e10)  # keep IRLS weights in a safe numeric range
            y_ = eta + dl1dp1 / (dr * wt)
            
            y_ = np.nan_to_num(y_, nan=0.0, posinf=1e10, neginf=-1e10)
            wt = np.nan_to_num(wt, nan=1.0,  posinf=1e10, neginf=1e-10)
            
            irls = LinearRegression(fit_intercept=False)
            irls.fit(X, y_, sample_weight=wt)
            beta_new = irls.coef_.copy()


            beta_new, theta_new, ll_inner = self._damped_update(
                p, beta_old, beta_new, X, y, theta_old
            )
            

            eta_new = X @ beta_new
            theta_new = theta_old.copy()
            theta_new[:, p] = self.distribution.link_inverse(eta_new, p)
            
            # self.betas[p] = beta_new
            # self.theta = theta_new


            ll_inner = self._log_likelihood(y, self.theta)
            
            
            if (iter_inner > 1 or iter_outer > 1):
                if ll_inner >  ll_inner_prev*1.25:
                    if self.verbose:
                        print(f"Reverting update (LL worsened) at outer = {iter_outer}, inner={iter_inner} p={p}")
                    self.betas[p] = beta_old
                    self.theta = theta_old
                    ll_inner = ll_inner_prev
                    break 



            if self.verbose:
                print(
                    f"    p={p}, inner={iter_inner:02d}: dev = {ll_inner:.4f}, step = "
                    f"{np.max(np.abs(beta_new - beta_old)):+.2e}"
                )
                
            # eta = self.distribution.link_function(self.theta[:, p], p)
            # # dr = 1.0 / self.distribution.link_inverse_derivative(eta, p)
            # # ddr =  -1*self.distribution.link_inverse_second_derivative(eta, p)/self.distribution.link_inverse_derivative(eta, p)**3
            # dr = self.distribution.link_function_derivative(self.theta[:, p], p)
            # ddr = self.distribution.link_second_derivative(self.theta[:, p], p)
            # dl1dp1 = self.distribution.dll(y, self.theta, p)
            # dl2dp2 = self.distribution.ddll(y, self.theta, p)
            
            # wt = -(dl2dp2*dr - dl1dp1*ddr) / (dr **3)
            # wt = np.clip(wt, 1e-10, 1e10)  # keep IRLS weights in a safe numeric range
            # y_ = eta + dl1dp1 / (dr * wt)
            
            # eta    = self.distribution.link_function(self.theta[:, p], p)
            # u      = self.distribution.dll(y, self.theta, p)  # dℓ/dθ
            # h      = self.distribution.ddll(y, self.theta, p)   # d²ℓ/dθ²
            
            
            # # dr     = self.distribution.link_function_derivative(self.theta[:, p], p)
            # dr     =  self.distribution.link_inverse_derivative(eta, p) 
            # ddr    = self.distribution.link_inverse_second_derivative(eta, p)
            
            # W      = -(h * dr**2 + u * ddr)
            # wt      = np.clip(W, 1e-10, 1e10)
            
            # u_eta  = u * dr
            # y_      = eta + u_eta / wt
            

    

            if iter_inner > 1:
                if (abs(ll_inner_prev - ll_inner) / abs(ll_inner_prev) ) < self.rel_tol:
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
            eta = X_int @ self.betas[p]
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


if __name__ == "__main__":
    X, y = load_diabetes(return_X_y=True)
    gamlss = GAMLSS()
    gamlss.fit(X, y)
    # print(np.vstack([*gamlss.betas.values()]).T)
