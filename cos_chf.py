# -*- coding: utf-8 -*-
"""
Created on Wed Aug  6 16:44:17 2025

@author: samue
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Tuple, Optional, Dict

import numpy as np
import pandas as pd
from numpy.random import Generator, PCG64
from scipy.optimize import minimize
from scipy.stats import chi2, kendalltau, t as sps_t
from sklearn.covariance import ledoit_wolf
from dist_cos import StudentTModel
from ecf_einsum import ecf_grid_einsum

@dataclass
class COSStudentTFitter:
    model: StudentTModel
    a: np.ndarray                 # lower trunc. (d,)
    b: np.ndarray                 # upper trunc. (d,)
    N: int = 32                   # resolution per axis
    batch_frac: float = 0.01
    rng: Generator = field(default_factory=lambda: Generator(PCG64()))


    k_grid: np.ndarray = field(init=False)       # (K, d)
    u_grid: np.ndarray = field(init=False)       # (K, d)
    weights: np.ndarray = field(init=False)      # (K,)
    phi_emp: np.ndarray = field(init=False)      # (K,)

    def __post_init__(self):
        axes = [np.arange(self.N) for _ in range(self.model.d)]
        k_mesh = np.array(np.meshgrid(*axes, indexing="ij"))  # (d,N,…)
        self.k_grid = k_mesh.reshape(self.model.d, -1).T        # (K, d)
        self.u_grid = np.pi * self.k_grid / (self.b - self.a)
        self.weights = np.exp(-0.04 * np.linalg.norm(self.u_grid, axis=1))


    
    def prepare_empirical_cf(self, X: np.ndarray):
        K = np.full(self.model.d, self.N, dtype=int)
        s = np.ones(self.model.d)
        self.phi_emp = ecf_grid_einsum(X, self.a, self.b, s, K)


    def _batch_indices(self) -> np.ndarray:
        grid_size = len(self.k_grid)
        base = self.rng.choice(grid_size, int(grid_size * self.batch_frac), replace=False)
        # ensure each axis direction has at least one point
        axis_pts = [self.rng.choice(self.N) for _ in range(self.model.d)]
        axis_idx = [np.ravel_multi_index(tuple([axis_pts[i] if j == i else 0 for j in range(self.model.d)]), (self.N,) * self.model.d) for i in range(self.model.d)]
        return np.unique(np.concatenate([base, axis_idx]))

 

    def _loss(self, theta: np.ndarray) -> float:
        self._unpack(theta)
        idx = self._batch_indices()
        phi_mod = self.model.phi(self.u_grid[idx], self._tau.mean() * np.ones(len(idx)), np.zeros((len(idx), self.model.k)))
        diff = self.phi_emp[idx] - phi_mod
        return np.sum(self.weights[idx] * (diff.real**2 + diff.imag**2)) * (len(self.k_grid) / len(idx))

    
    def _grad(self, theta: np.ndarray) -> np.ndarray:
        
        eps = 1e-5
        grad = np.zeros_like(theta)
        f0 = self._loss(theta)
        for i in range(len(theta)):
            th = theta.copy(); th[i] += eps
            grad[i] = (self._loss(th) - f0) / eps
        return grad

    def fit(self, maxiter: int = 400) -> None:
        theta0 = self._pack()
        res = minimize(
            fun=self._loss,
            x0=theta0,
            method="L-BFGS-B",
            jac=self._grad,
            options=dict(maxiter=maxiter, maxls=20),
        )
        logging.info("Optimization success: %s", res.success)
        self._unpack(res.x)
        self.result_ = res


    def _pack(self) -> np.ndarray:
        L = np.linalg.cholesky(self.model.Sigma)
        tril = L[np.tril_indices(self.model.d)]
        return np.concatenate([
            self.model.mu,
            self.model.B.ravel(),
            tril,
            [self.model.alpha, self.model.nu],
        ])

    def _unpack(self, theta: np.ndarray) -> None:
        d, k = self.model.d, self.model.k
        self.model.mu = theta[:d]
        self.model.B = theta[d : d + d * k].reshape(d, k)
        tril_len = d * (d + 1) // 2
        tril = theta[d + d * k : d + d * k + tril_len]
        L = np.zeros((d, d))
        L[np.tril_indices(d)] = tril
        self.model.Sigma = L @ L.T
        self.model.alpha, self.model.nu = theta[-2:]


    def diagnostics(self) -> Dict[str, float]:
        K = len(self.k_grid)
        diff = self.phi_emp - self.model.phi(
            self.u_grid, self._tau.mean() * np.ones(K), np.zeros((K, self.model.k))
        )
        Q = np.sum(self.weights * (diff.real**2 + diff.imag**2))
        J = len(self._tau) * Q
        pval = 1 - chi2.cdf(J, 2 * K - len(self._pack()))
        cond = np.linalg.cond(self.model.Sigma)
        return {"J": J, "pval": pval, "cond(Sigma)": cond}



    def cos_price_moments(self, n_mom: int = 2) -> np.ndarray:
        """Return first n_mom moments of the price component using COS."""
        moments = np.zeros(n_mom)
        # build A_k only along price index k1, fixing others to 0 for speed
        price_axis = self.k_grid[:, 0]
        u = np.pi * price_axis / (self.b[0] - self.a[0])
        phi = self.model.phi(u[:, None], np.ones_like(u) * self._tau.mean(), np.zeros((len(u), self.model.k)))
        Ak = (2.0 / (self.b[0] - self.a[0])) * np.real(
            np.exp(-1j * u * self.a[0]) * phi)
        for m in range(n_mom):
            # analytic integral of cos against x^m on [a,b]
            if m == 0:
                C = np.where(price_axis == 0, 1.0, 0.0)
            elif m == 1:
                C = (
                    (np.where(price_axis == 0, 0.5, 0.0))
                    + (np.sin(price_axis * np.pi) / (price_axis * np.pi))
                )
            else:
                C = np.zeros_like(Ak)  # extend as needed
            moments[m] = np.sum(Ak * C)
        return moments
