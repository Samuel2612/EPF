# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 08:45:08 2025

@author: samue
"""

import numpy as np

def init_weight_vector(size):
    """Initialise an exponentially discounted vector of weights.

    The weight for the n-th observation is defined as (1 - forget)^(N - n).

    Assumes the first observation is the oldest and the last is the newest.
    """

    out = np.ones(size)
    
    return out

def init_gram(X, w):
    """Initialise the Gramian Matrix.

    """
    f = init_weight_vector(X.shape[0])
    gram = (X * np.expand_dims((w * f) ** 0.5, -1)).T @ (X * np.expand_dims((w * f) ** 0.5, -1))
    return gram

def init_y_gram(X, y, w):
    """Initialise the y-Gramian Matrix.


    """
    f = init_weight_vector(X.shape[0])
    gram = np.expand_dims((X * np.expand_dims((w * f) ** 0.5, -1)).T @ (y * (w * f) ** 0.5), axis=-1)
    return gram

def init_inverted_gram(X, w):
    """Initialise the inverted Gramian Matrix.

    """
    gram = init_gram(X, w)
    rank = np.linalg.matrix_rank(gram)
    if rank == gram.shape[0]:
        inv_gram = np.linalg.inv(gram)
        return inv_gram
    else:
        raise ValueError("Matrix is not invertible.")

def update_gram(gram, X, forget=0, w=1):
    """Update the Gramian Matrix with new observations.

    """
    if X.shape[0] == 1:
        new_gram =  gram + w * np.outer(X, X)
    else:
        batch_size = X.shape[0]
        f = init_weight_vector(batch_size)
        weights = np.expand_dims((w * f) ** 0.5, axis=-1)
        new_gram = gram + (X * weights).T @ (X * weights)
    return new_gram

def update_y_gram(gram, X, y, forget=0, w=1):
    """Update the y-Gramian Matrix with new observations.

    """
    if X.shape[0] == 1:
        new_gram =  gram + w * np.outer(X, y)
    else:
        batch_size = X.shape[0]
        f = init_weight_vector(batch_size)
        new_gram = gram   + np.expand_dims(
            ((X * np.expand_dims((w * f) ** 0.5, axis=-1)).T @ (y * (w * f) ** 0.5)), -1
        )
    return new_gram

def _update_inverted_gram(gram, X,  w=1):
    """Update the inverted Gramian for one step."""
 
    new_gram =  gram - (w * gram @ np.outer(X, X) @ gram) / (w * X @ gram @ X.T)
    return new_gram

def update_inverted_gram(gram, X,  w=1):
    """Update the inverted Gramian Matrix with new observations."""
    if X.shape[0] == 1:
        new_gram = _update_inverted_gram(gram, X, w=w)
    else:
        new_gram = _update_inverted_gram(gram, np.expand_dims(X[0, :], 0), w=w[0])
        for i in range(1, X.shape[0]):
            new_gram = _update_inverted_gram(new_gram, np.expand_dims(X[i, :], 0),  w=w[i])
    return new_gram
