# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 11:53:09 2025



@author: samue
"""
import numpy as np

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