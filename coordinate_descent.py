# -*- coding: utf-8 -*-
"""
Created on Thu Feb 08 15:40:46 2025

@author: samue
"""

import numpy as np

DEFAULT_ESTIMATOR_KWARGS = {
    "ic": {0: "bic", 1: "bic", 2: "bic", 3: "bic"},
    "lambda_n": 100,
    "lambda_eps": {0: 1e-4, 1: 1e-4, 2: 1e-8, 3: 1e-4},
    "start_value": "previous_fit",
    "tolerance": 1e-4,
    "max_iterations": 1000,
    "selection": "cyclic",
}


def soft_threshold(value, threshold):

    return np.sign(value) * np.maximum(np.abs(value) - threshold, 0)


def online_coordinate_descent(
    x_gram,
    y_gram,
    beta,
    regularization,
    is_regularized,
    beta_lower_bound,
    beta_upper_bound,
    selection="cyclic",
    tolerance=1e-4,
    max_iterations=1000,
):
    """The parameter update cycle of the online coordinate descent.
    """
    i = 0
    J = beta.shape[0]
    JJ = np.arange(J)
    beta_now = np.copy(beta)
    beta_star = np.copy(beta)

    while True:
        i += 1
        beta_star = np.copy(beta_now)

        if (selection == "random") and (i >= 2):
            JJ = np.random.permutation(J)

        for j in JJ:
            if (i < 2) or (beta_now[j] != 0):
                update = y_gram[j] - (x_gram[j, :] @ beta_now) + x_gram[j, j] * beta_now[j]
                if is_regularized[j]:
                    update = soft_threshold(update, regularization)
                beta_now[j] = min(max(update / x_gram[j, j], beta_lower_bound[j]), beta_upper_bound[j])
        if np.max(np.abs(beta_now - beta_star)) <= tolerance * np.max(np.abs(beta_now)):
            break
        if i > max_iterations:
            break
    return beta_now, i


def online_coordinate_descent_path(
    x_gram,
    y_gram,
    beta_path,
    lambda_path,
    is_regularized,
    beta_lower_bound,
    beta_upper_bound,
    which_start_value="previous_lambda",
    selection="cyclic",
    tolerance=1e-4,
    max_iterations=1000,
):
    """Run coordinate descent on a grid of regularization values.
    """
    beta_path_new = np.zeros_like(beta_path)
    iterations = np.zeros_like(lambda_path)

    for i, regularization in enumerate(lambda_path):
        # Select the appropriate starting value for the next coordinate descent update.
        if which_start_value == "average":
            beta = (beta_path_new[max(i - 1, 0), :] + beta_path[max(i, 0), :]) / 2
        elif which_start_value == "previous_lambda":
            beta = beta_path_new[max(i - 1, 0), :]
        else:
            beta = beta_path[max(i, 0), :]

        beta_path_new[i, :], iterations[i] = online_coordinate_descent(
            x_gram,
            y_gram,
            beta,
            regularization,
            is_regularized,
            beta_lower_bound,
            beta_upper_bound,
            selection=selection,
            tolerance=tolerance,
            max_iterations=max_iterations,
        )

    return beta_path_new, iterations
