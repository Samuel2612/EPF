import numpy as np
from glmnet import glmnet
from glmnetPredict import glmnetPredict

class LogisticGlmnetModel:
    """
    A wrapper for penalized logistic regression via glmnet,
    set up to estimate log( pi / (1 - pi) ) from predictors X.
    """
    def __init__(self, alpha=0.5, nlambda=100, standardize=True, random_state=None):
        """
        Parameters
        ----------
        alpha : float, default=0.5
            The elastic-net mixing parameter (0 <= alpha <= 1).
            alpha=1 => lasso penalty, alpha=0 => ridge penalty.
        nlambda : int, default=100
            Number of lambda values to try.
        standardize : bool, default=True
            Standardize predictors.
        random_state : int or None
            Seed for reproducibility.
        """
  
        self.fitted = False
        
    def fit(self, X, y):
        """
        Fit the logistic glmnet model on the provided data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training predictors.
        y : array-like of shape (n_samples,)
            Binary targets (0 or 1).
        """
        # Convert X and y to numpy arrays if they are not already
        X = np.asarray(X)
        y = np.asarray(y)
        
        # Fit penalized logistic regression
        self.fit = glmnet(x = X.copy(), y = y.copy(), family = 'binomial', alpha = 1.0)
        
        self.fitted = True
        return self
    
    def predict_pi(self, X_new):
        """
        Predict the probability (pi) that y=1 for new data.
        
        Parameters
        ----------
        X_new : array-like of shape (m_samples, n_features)
            New data for which to predict pi.
        
        Returns
        -------
        pi_hat : ndarray of shape (m_samples,)
            The predicted probability for each row in X_new.
        """
        if not self.fitted:
            raise ValueError("Model must be fit before calling predict_pi.")
        
        X_new = np.asarray(X_new)
        
        # predict_proba returns two columns: P(y=0), P(y=1)
        # We want the second column => P(y=1)
        pi_hat = glmnetPredict(self.fit, newx = X_new, ptype='link')
        return pi_hat
    
    def sample_y(self, X_new, n_draws=1):
        """
        Draw Bernoulli samples for new data, using predicted pi.
        
        Parameters
        ----------
        X_new : array-like of shape (m_samples, n_features)
            New data for which to predict pi.
        n_draws : int, default=1
            How many times to sample from the predicted probabilities 
            for each row in X_new.
        
        Returns
        -------
        samples : ndarray of shape (m_samples, n_draws)
            Each column is one random draw for all rows in X_new.
        """
        if not self.fitted:
            raise ValueError("Model must be fit before calling sample_y.")
        
        X_new = np.asarray(X_new)
        pi_hat = self.predict_pi(X_new)  # shape (m_samples,)
        
        # For each row in X_new, sample n_draws times from Bernoulli(pi)
        # We'll store those results in (m_samples, n_draws)
        samples = np.random.binomial(n=1, p=pi_hat[:, None], size=(X_new.shape[0], n_draws))
        
        return samples
