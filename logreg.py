# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 11:15:32 2025

@author: samue
"""

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class LassoLogisticBIC:
    """
    Implements a L1-regularized (LASSO) logistic regression, selecting
    the penalty parameter by Bayesian Information Criterion (BIC).
    
    Usage:
        1) Instantiate with a grid of Cs or lambdas you want to try.
        2) Call .fit(X, y).
        3) The best model is stored in self.best_model_,
           with self.best_C_ holding the chosen C.
        4) Use .predict() or .predict_proba() for inference.
    """
    def __init__(self, 
                 Cs=None,
                 max_iter=1000,
                 random_state=None,
                 fit_intercept=True):
        """
        Parameters
        ----------
        Cs : list or np.array
            Grid of C values (inverse of regularization strength) to try.
            If None, a default grid is used.
        max_iter : int
            Maximum number of iterations for the logistic solver.
        random_state : int or None
            Random seed for reproducibility (used for solver).
        fit_intercept : bool
            Whether to fit an intercept term.
        """
        if Cs is None:
            # Example log-spaced grid from 1e-3 to 1e2
            Cs = np.logspace(-4, 3, 100)
        self.Cs = Cs
        self.max_iter = max_iter
        self.random_state = random_state
        self.fit_intercept = fit_intercept

        # These get set after calling .fit()
        self.best_model_ = None
        self.best_C_ = None
        self.scaler_ = None

    def _bic_score(self, model, X, y):
        """
        Compute the Bayesian Information Criterion (BIC):
            BIC = -2 * logLik + k * ln(N)
        where
            logLik = sum(log p_i)
            k = number of estimated parameters (nonzero + intercept)
            N = number of observations
        """
        # Predicted probability for the true class:
        p = model.predict_proba(X)
        
        # log p_i for each observation i:
        # We need the probability of the correct class (0 or 1),
        # i.e., p[i, y[i]] if y[i] in {0,1}.
        # Vectorized approach:
        idx = (np.arange(len(y)), y.astype(int))  # row indices, col indices
        log_lik = np.log(p[idx])  # log probability of the correct label
        total_log_lik = np.sum(log_lik)
        
        N = len(y)
        
        # Count #nonzero in coefficients. We do not penalize the intercept,
        # but we do count it as a parameter for BIC:
        coefs = model.coef_.ravel()
        n_nonzero = np.count_nonzero(np.abs(coefs) > 1e-8)
        k = n_nonzero + (1 if self.fit_intercept else 0)
        
        bic = -2.0 * total_log_lik + k * np.log(N)
        return bic

    def fit(self, X, y):
        """
        Fits L1-penalized logistic regression models for each C in self.Cs,
        then picks the model with minimal BIC.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix.
        y : array-like, shape (n_samples,)
            Binary targets (0/1).
        """
        X = np.asarray(X)
        y = np.asarray(y, dtype=int)
        
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)
        
        best_bic = np.inf
        best_model = None
        best_C = None
        
        #Grid search over Cs
        for c in self.Cs:
            model = LogisticRegression(
                penalty='l1',
                C=c,
                solver='saga',
                max_iter=self.max_iter,
                fit_intercept=self.fit_intercept,
                random_state=self.random_state
            )
            model.fit(X_scaled, y)
            
            bic = self._bic_score(model, X_scaled, y)
            if bic < best_bic:
                best_bic = bic
                best_model = model
                best_C = c
        
        # Store the best model
        self.best_model_ = best_model
        self.best_C_ = best_C

    def predict_proba(self, X):
        """
        Predict probability of class=1 for each sample in X.
        Must call fit() first.
        """
        if self.best_model_ is None:
            raise RuntimeError("Must fit the model before calling predict_proba.")
        X_scaled = self.scaler_.transform(X)
        return self.best_model_.predict_proba(X_scaled)

    def predict(self, X, threshold=0.5):
        """
        Predict binary label (0 or 1) based on the predicted probability >= threshold.
        """
        probs = self.predict_proba(X)[:, 1]  # probability of class=1
        return (probs >= threshold).astype(int)

    def get_params(self):
        """
        Returns the model's best hyperparameter and coefficients.
        """
        if self.best_model_ is None:
            raise RuntimeError("Must fit the model before getting parameters.")
        return {
            "C": self.best_C_,
            "coef_": self.best_model_.coef_,
            "intercept_": self.best_model_.intercept_,
        }
