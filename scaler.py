# -*- coding: utf-8 -*-
"""
Created on  Feb 12 15:55:05 2025

@author: samue
"""

import numpy as np


class MyScaler:
    def __init__(self, forget=0.0, to_scale=True):
        """The online scaler allows for incremental updating and scaling of matrices.

        """

        self.to_scale = to_scale

    def _prepare_estimator(self, X):
        """Add derived attributes to estimator."""
        if isinstance(self.to_scale, np.ndarray):
            self._selection = self.to_scale
            self._do_scale = True
        elif isinstance(self.to_scale, bool):
            if self.to_scale:
                self._selection = np.arange(X.shape[1])
                self._do_scale = True
            else:
                self._selection = False
                self._do_scale = False

        self.m = 0
        self.M = 0
        self.v = 0
        self.n_observations = X.shape[0]


    def fit(self, X):
        """Fit the OnlineScaler object for the first time.

        Args:
            X: Matrix of covariates.
        """
        self._prepare_estimator(X)
        if self._do_scale:
            self.m = np.mean(X[:, self._selection], axis=0)
            self.v = np.var(X[:, self._selection], axis=0)
            self.M = self.v * self.n_observations

    def update(self, X):
        """Wrapper for partial_fit to align API."""
        self.partial_fit(X)

    def partial_fit(self, X):
        """Update the OnlineScaler for new rows of X.

        Args:
            X: New data for X.
        """
        if self._do_scale:
            for i in range(X.shape[0]):
                self.n_observations += 1
                n_seen = calculate_effective_training_length(self.forget, self.n_observations)
                forget_scaled = self.forget * np.maximum(self.n_asymmptotic / n_seen, 1.0)
                diff = X[i, self._selection] - self.m
                incr = forget_scaled * diff

                if forget_scaled > 0:
                    self.m += incr
                    self.v = (1 - forget_scaled) * (self.v + forget_scaled * diff**2)
                else:
                    self.m += diff / self.n_observations
                    self.M += diff * (X[i, self._selection] - self.m)
                    self.v = self.M / self.n_observations

    def transform(self, X):
        """Transform X to a mean-std scaled matrix.

        Args:
            X: Matrix of covariates.

        Returns:
            The scaled X matrix.
        """
        if self._do_scale:
            out = np.copy(X)
            out[:, self._selection] = (X[:, self._selection] - self.m) / np.sqrt(self.v)
            return out
        else:
            return X

    def inverse_transform(self, X):
        """Back-transform a scaled X matrix to the original domain.

        Args:
            X: Scaled X matrix.

        Returns:
            X scaled back to the original scale.
        """
        if self._do_scale:
            out = np.copy(X)
            out[:, self._selection] = X[:, self._selection] * np.sqrt(self.v) + self.m
          
