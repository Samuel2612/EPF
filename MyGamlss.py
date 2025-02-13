# -*- coding: utf-8 -*-
"""
Created on Feb  6 10:12:20 2025

@author: samue
"""



import numpy as np
import copy

from gram import init_weight_vector
from information_criteria import select_best_model_by_information_criterion
from methods import get_estimation_method
from scaler import MyScaler



class MyGamlss:
    """The online/incremental GAMLSS class."""

    def __init__(self, distribution, equation, method="lasso", scale_inputs=True,
                 fit_intercept=True, regularize_intercept=False, ic="bic", max_it_outer=30,
                 max_it_inner=30, abs_tol_outer=1e-3, abs_tol_inner=1e-3, rel_tol_outer=1e-5,
                 rel_tol_inner=1e-5, rss_tol_inner=1.5):
        """The `OnlineGamlss()` provides the fit, update and predict methods for linear parametric GAMLSS models.

        For a response variable $Y$ which is distributed according to the distribution $\mathcal{F}(\theta)$
        with the distribution parameters $\theta$, we model:

        $$g_k(\theta_k) = \eta_k = X_k\beta_k$$

        
        """
        self.distribution = distribution
        self.equation = self._process_equation(equation)
        self._process_attribute(fit_intercept, True, "fit_intercept")
        self._process_attribute(regularize_intercept, False, "regularize_intercept")
        self._process_attribute(ic, "aic", "ic")

        # Get the estimation method
        self._process_attribute(method, "ols", "method")
        self._method = {p: get_estimation_method(m) for p, m in self.method.items()}

        self.scaler = OnlineScaler(to_scale=scale_inputs)
        self.do_scale = scale_inputs

        # These are global for all distribution parameters
        self.max_it_outer = max_it_outer
        self.max_it_inner = max_it_inner
        self.abs_tol_outer = abs_tol_outer
        self.abs_tol_inner = abs_tol_inner
        self.rel_tol_outer = rel_tol_outer
        self.rel_tol_inner = rel_tol_inner
        self.rss_tol_inner = rss_tol_inner


        self.is_regularized = {}

    @property
    def betas(self):
        return self.beta

   

    def _process_attribute(self, attribute, default, name):
        if isinstance(attribute, dict):
            for p in range(self.distribution.n_params):
                    if isinstance(default, dict):
                        attribute[p] = default[p]
                    else:
                        attribute[p] = default
        else:
            attribute = {p: attribute for p in range(self.distribution.n_params)}

        setattr(self, name, attribute)



    @staticmethod
    def _make_intercept(n_observations):
        """Make the intercept series as N x 1 array."""
        return np.ones((n_observations, 1))


    @staticmethod
    def _add_lags(y, x, lags):
        """
        Add lagged variables to the response and covariate matrices.

        """
        if lags == 0:
            return y, x

        if isinstance(lags, int):
            lags = np.arange(1, lags + 1, dtype=int)

        max_lag = np.max(lags)
        lagged = np.stack([np.roll(y, i) for i in lags], axis=1)[max_lag:, :]
        new_x = np.hstack((x, lagged))[max_lag:, :]
        new_y = y[max_lag:]
        return new_y, new_x

    def get_J_from_equation(self, X):
        J = {}
        for p in range(self.distribution.n_params):
            if isinstance(self.equation[p], str):
                if self.equation[p] == "all":
                    J[p] = X.shape[1] + int(self.fit_intercept[p])
                if self.equation[p] == "intercept":
                    J[p] = 1
            elif isinstance(self.equation[p], np.ndarray):
                if np.issubdtype(self.equation[p].dtype, bool):
                    J[p] = np.sum(self.equation[p]) + int(self.fit_intercept[p])
                elif np.issubdtype(self.equation[p].dtype, np.integer):
                    J[p] = self.equation[p].shape[0] + int(self.fit_intercept[p])
                

        return J

    def make_model_array(self, X, param):
        eq = self.equation[param]
        n = X.shape[0]

        if isinstance(eq, str) and (eq == "intercept"):
            out = self._make_intercept(n_observations=n)
        else:
            if isinstance(eq, str) and (eq == "all"):
                if isinstance(X, np.ndarray):
                    out = X
            elif isinstance(eq, np.ndarray) or isinstance(eq, list):
                if isinstance(X, np.ndarray):
                    out = X[:, eq]

            if self.fit_intercept[param]:
                out = np.hstack((self._make_intercept(n), out))

        return out

    def fit_beta_and_select_model(self, X, y, w, iteration_outer, iteration_inner, param):
        f = init_weight_vector(self.n_observations)

        if not self._method[param]._path_based_method:
            beta_path = None
            beta = self._method[param].fit_beta(
                x_gram=self.x_gram[param],
                y_gram=self.y_gram[param],
                is_regularized=self.is_regularized[param],
            )
            residuals = y - X @ beta.T
            rss = np.sum(residuals**2 * w * f) / np.mean(w * f)
        else:
            beta_path = self._method[param].fit_beta_path(
                x_gram=self.x_gram[param],
                y_gram=self.y_gram[param],
                is_regularized=self.is_regularized[param],
            )
            residuals = y[:, None] - X @ beta_path.T
            rss = np.sum(residuals**2 * w[:, None] * f[:, None], axis=0)
            rss = rss / np.mean(w * f)
            model_params_n = np.sum(~np.isclose(beta_path, 0), axis=1)
            best_ic = select_best_model_by_information_criterion(
                self.n_training[param], model_params_n, rss, self.ic[param]
            )
            beta = beta_path[best_ic, :]

            self.rss_iterations_inner[param][iteration_outer][iteration_inner] = rss
            self.ic_iterations_inner[param][iteration_outer][iteration_inner] = best_ic

        self.residuals[param] = residuals
        self.weights[param] = w

        return beta, beta_path, rss

    def update_beta_and_select_model(self, X, y, w, iteration_outer, iteration_inner, param):
        denom = online_mean_update(
            self.mean_of_weights[param], w, self.forget[param], self.n_observations
        )

        if not self._method[param]._path_based_method:
            beta_path = None
            beta = self._method[param].update_beta(
                x_gram=self.x_gram_inner[param],
                y_gram=self.y_gram_inner[param],
                beta=self.beta[param],
                is_regularized=self.is_regularized[param],
            )
            residuals = y - X @ beta.T
            rss = (
                (residuals**2).flatten() * w
                + (1 - self.forget[param])
                * (self.rss_old[param] * self.mean_of_weights[param])
            ) / denom
        else:
            beta_path = self._method[param].update_beta_path(
                x_gram=self.x_gram_inner[param],
                y_gram=self.y_gram_inner[param],
                beta_path=self.beta_path[param],
                is_regularized=self.is_regularized[param],
            )
            residuals = y - X @ beta_path.T
            rss = (
                (residuals**2).flatten() * w
                + (1 - self.forget[param])
                * (self.rss_old[param] * self.mean_of_weights[param])
            ) / denom
            model_params_n = np.sum(np.isclose(beta_path, 0), axis=1)
            best_ic = select_best_model_by_information_criterion(
                self.n_training[param], model_params_n, rss, self.ic[param]
            )
            self.rss_iterations_inner[param][iteration_outer][iteration_inner] = rss
            self.ic_iterations_inner[param][iteration_outer][iteration_inner] = best_ic
            beta = beta_path[best_ic, :]

        return beta, beta_path, rss

    def _make_initial_fitted_values(self, y):
        out = np.stack(
            [self.distribution.initial_values(y, param=i)
             for i in range(self.distribution.n_params)],
            axis=1,
        )
        return out

    def fit(self, X, y, sample_weight=None):
        """Fit the online GAMLSS model.

        Args:
            X: Data Matrix.
            y: Response variable.
            sample_weight: User-defined sample weights. Defaults to None.
        """
        self.n_observations = y.shape[0]
        self.n_training = {
            p: calculate_effective_training_length(self.forget[p], self.n_observations)
            for p in range(self.distribution.n_params)
        }

        if sample_weight is not None:
            w = sample_weight
        else:
            w = np.ones(y.shape[0])

        self.fv = self._make_initial_fitted_values(y=y)
        self.J = self.get_J_from_equation(X=X)

        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        X_dict = {
            p: self.make_model_array(X_scaled, param=p)
            for p in range(self.distribution.n_params)
        }

    

        self.rss = {i: 0 for i in range(self.distribution.n_params)}
        self.x_gram = {}
        self.y_gram = {}
        self.weights = {}
        self.residuals = {}

        for p in range(self.distribution.n_params):
            is_regularized = np.repeat(True, self.J[p])
            if self.fit_intercept[p] and not (self.regularize_intercept[p] or self._is_intercept_only(p)):
                is_regularized[0] = False
            self.is_regularized[p] = is_regularized

        self.beta_iterations = {i: {} for i in range(self.distribution.n_params)}
        self.beta_iterations_inner = {i: {} for i in range(self.distribution.n_params)}
        self.beta_path = {p: None for p in range(self.distribution.n_params)}
        self.beta = {p: np.zeros(self.J[p]) for p in range(self.distribution.n_params)}
        self.beta_path_iterations_inner = {i: {} for i in range(self.distribution.n_params)}
        self.beta_path_iterations = {i: {} for i in range(self.distribution.n_params)}
        self.rss_iterations_inner = {i: {} for i in range(self.distribution.n_params)}
        self.ic_iterations_inner = {i: {} for i in range(self.distribution.n_params)}
        self.sum_of_weights = {}
        self.mean_of_weights = {}

        message = "Starting fit call"
        self._print_message(message=message, level=1)
        self.global_dev, self.iteration_outer = self._outer_fit(X=X_dict, y=y, w=w)
        message = "Finished fit call"
        self._print_message(message=message, level=1)

    def update(self, X, y, sample_weight=None):
        """Update the fit for the online GAMLSS Model.

        Args:
            X: Data Matrix.
            y: Response variable.
            sample_weight: User-defined sample weights. Defaults to None.
        """
        if sample_weight is not None:
            w = sample_weight
        else:
            w = np.ones(y.shape[0])

        self.n_observations += y.shape[0]
        self.n_training = {
            p: calculate_effective_training_length(self.forget[p], self.n_observations)
            for p in range(self.distribution.n_params)
        }

        self.fv = self.predict(X, what="response")
        self.scaler.partial_fit(X)
        X_scaled = self.scaler.transform(X)
        X_dict = {
            p: self.make_model_array(X_scaled, param=p)
            for p in range(self.distribution.n_params)
        }

      

        self.rss_iterations_inner = {i: {} for i in range(self.distribution.n_params)}
        self.ic_iterations_inner = {i: {} for i in range(self.distribution.n_params)}
        self.x_gram_inner = copy.copy(self.x_gram)
        self.y_gram_inner = copy.copy(self.y_gram)
        self.rss_old = copy.copy(self.rss)
        self.sum_of_weights_inner = copy.copy(self.sum_of_weights)
        self.mean_of_weights_inner = copy.copy(self.mean_of_weights)

        message = "Starting update call"
        self._print_message(message=message, level=1)
        self.global_dev, self.iteration_outer = self._outer_update(X=X_dict, y=y, w=w)
        self.x_gram = copy.copy(self.x_gram_inner)
        self.y_gram = copy.copy(self.y_gram_inner)
        self.sum_of_weights = copy.copy(self.sum_of_weights_inner)
        self.mean_of_weights = copy.copy(self.mean_of_weights_inner)
        message = "Finished update call"
        self._print_message(message=message, level=1)

    def _outer_update(self, X, y, w):
        global_di = -2 * np.log(self.distribution.pdf(y, self.fv))
        global_dev = (1 - self.forget[0]) * self.global_dev + global_di
        global_dev_old = global_dev + 1000
        iteration_outer = 0

        while True:
            if np.abs(global_dev_old - global_dev) / np.abs(global_dev_old) < self.rel_tol_outer:
                break
            if np.abs(global_dev_old - global_dev) < self.abs_tol_outer:
                break
            if iteration_outer >= self.max_it_outer:
                break

            global_dev_old = global_dev
            iteration_outer += 1

            for param in range(self.distribution.n_params):
                self.beta_iterations_inner[param][iteration_outer] = {}
                self.beta_path_iterations_inner[param][iteration_outer] = {}
                self.rss_iterations_inner[param][iteration_outer] = {}
                self.ic_iterations_inner[param][iteration_outer] = {}

                global_dev = self._inner_update(X=X, y=y, w=w, iteration_outer=iteration_outer,
                                                param=param, dv=global_dev)
                message = f"Outer iteration {iteration_outer}: Fitted param {param}: Current LL {global_dev}"
                self._print_message(message=message, level=2)

            message = f"Outer iteration {iteration_outer}: Finished: current LL {global_dev}"
            self._print_message(message=message, level=1)

        return global_dev, iteration_outer

    def _outer_fit(self, X, y, w):
        global_di = -2 * np.log(self.distribution.pdf(y, self.fv))
        global_dev = np.sum(w * global_di)
        global_dev_old = global_dev + 1000
        iteration_outer = 0

        while True:
            if np.abs(global_dev_old - global_dev) / np.abs(global_dev_old) < self.rel_tol_outer:
                break
            if np.abs(global_dev_old - global_dev) < self.abs_tol_outer:
                break
            if iteration_outer >= self.max_it_outer:
                break

            global_dev_old = global_dev
            iteration_outer += 1

            for param in range(self.distribution.n_params):
                self.beta_iterations_inner[param][iteration_outer] = {}
                self.beta_path_iterations_inner[param][iteration_outer] = {}
                self.rss_iterations_inner[param][iteration_outer] = {}
                self.ic_iterations_inner[param][iteration_outer] = {}

                global_dev = self._inner_fit(X=X, y=y, w=w, param=param,
                                             iteration_outer=iteration_outer, dv=global_dev)
                self.beta_iterations[param][iteration_outer] = self.beta[param]
                self.beta_path_iterations[param][iteration_outer] = self.beta_path[param]

                message = f"Outer iteration {iteration_outer}: Fitted param {param}: current LL {global_dev}"
                self._print_message(message=message, level=2)

            message = f"Outer iteration {iteration_outer}: Finished. Current LL {global_dev}"
            self._print_message(message=message, level=1)

        return global_dev, iteration_outer

    def _inner_fit(self, X, y, w, iteration_outer, param, dv):
        di = -2 * np.log(self.distribution.pdf(y, self.fv))
        dv = np.sum(di * w)
        olddv = dv + 1
        iteration_inner = 0

        while True:
            if iteration_inner >= self.max_it_inner:
                break
            if (abs(olddv - dv) <= self.abs_tol_inner) and ((iteration_inner + iteration_outer) >= 2):
                break
            if (abs(olddv - dv) / abs(olddv) < self.rel_tol_inner) and ((iteration_inner + iteration_outer) >= 2):
                break

            iteration_inner += 1
            eta = self.distribution.link_function(self.fv[:, param], param=param)
            dr = 1 / self.distribution.link_inverse_derivative(eta, param=param)
            dl1dp1 = self.distribution.dl1_dp1(y, self.fv, param=param)
            dl2dp2 = self.distribution.dl2_dp2(y, self.fv, param=param)
            wt = -(dl2dp2 / (dr * dr))
            wt = np.clip(wt, 1e-10, 1e10)
            wv = eta + dl1dp1 / (dr * wt)

            self.x_gram[param] = self._method[param].init_x_gram(
                X=X[param], weights=(w * wt), forget=self.forget[param]
            )
            self.y_gram[param] = self._method[param].init_y_gram(
                X=X[param], y=wv, weights=(w * wt), forget=self.forget[param]
            )
            beta_new, beta_path_new, rss_new = self.fit_beta_and_select_model(
                X=X[param], y=wv, w=wt, param=param,
                iteration_inner=iteration_inner, iteration_outer=iteration_outer
            )

            if iteration_inner > 1 or iteration_outer > 1:
                if self.method[param] == "ols":
                    if rss_new > (self.rss_tol_inner * self.rss[param]):
                        break
                else:
                    ic_idx = self.ic_iterations_inner[param][iteration_outer][iteration_inner]
                    if rss_new[ic_idx] > (self.rss_tol_inner * self.rss[param][ic_idx]):
                        break

            self.beta[param] = beta_new
            self.beta_path[param] = beta_path_new
            self.rss[param] = rss_new

            eta = X[param] @ self.beta[param].T
            self.fv[:, param] = self.distribution.link_inverse(eta, param=param)

            di = -2 * np.log(self.distribution.pdf(y, self.fv))
            olddv = dv
            dv = np.sum(di * w)

            self.sum_of_weights[param] = np.sum(w * wt)
            self.mean_of_weights[param] = np.mean(w * wt)

            self.beta_iterations_inner[param][iteration_outer][iteration_inner] = beta_new
            self.beta_path_iterations_inner[param][iteration_outer][iteration_inner] = beta_path_new

            message = f"Outer iteration {iteration_outer}: Fitting Parameter {param}: Inner iteration {iteration_inner}: Current LL {dv}"
            self._print_message(message=message, level=3)

        return dv

    def _inner_update(self, X, y, w, iteration_outer, dv, param):
        di = -2 * np.log(self.distribution.pdf(y, self.fv))
        dv = (1 - self.forget[0]) * self.global_dev + np.sum(di * w)
        olddv = dv + 1
        iteration_inner = 0

        while True:
            if iteration_inner >= self.max_it_inner:
                break
            if (abs(olddv - dv) <= self.abs_tol_inner) and ((iteration_inner + iteration_outer) >= 2):
                break
            if (abs(olddv - dv) / abs(olddv) < self.rel_tol_inner) and ((iteration_inner + iteration_outer) >= 2):
                break

            iteration_inner += 1
            eta = self.distribution.link_function(self.fv[:, param], param=param)
            dr = 1 / self.distribution.link_inverse_derivative(eta, param=param)
            dl1dp1 = self.distribution.dl1_dp1(y, self.fv, param=param)
            dl2dp2 = self.distribution.dl2_dp2(y, self.fv, param=param)
            wt = -(dl2dp2 / (dr * dr))
            wt = np.clip(wt, -1e10, 1e10)
            wv = eta + dl1dp1 / (dr * wt)

            self.x_gram_inner[param] = self._method[param].update_x_gram(
                gram=self.x_gram[param], X=X[param],
                weights=(w * wt)
            )
            self.y_gram_inner[param] = self._method[param].update_y_gram(
                gram=self.y_gram[param], X=X[param], y=wv,
                weights=(w * wt)
            )
            beta_new, beta_path_new, rss_new = self.update_beta_and_select_model(
                X[param], y=wv, w=wt, iteration_inner=iteration_inner,
                iteration_outer=iteration_outer, param=param
            )
            if self.method[param] == "ols":
                if rss_new > (self.rss_tol_inner * self.rss[param]):
                    break
            else:
                ic_idx = self.ic_iterations_inner[param][iteration_outer][iteration_inner]
                if rss_new[ic_idx] > (self.rss_tol_inner * self.rss[param][ic_idx]):
                    break

            self.beta[param] = beta_new
            self.beta_path[param] = beta_path_new
            self.rss[param] = rss_new

            eta = X[param] @ self.beta[param].T
            self.fv[:, param] = self.distribution.link_inverse(eta, param=param)

            self.sum_of_weights_inner[param] = (np.sum(w * wt) +
                                                 self.sum_of_weights[param])
            self.mean_of_weights_inner[param] = (self.sum_of_weights_inner[param] /
                                                 self.n_training[param])

            olddv = dv
            di = -2 * np.log(self.distribution.pdf(y, self.fv))
            dv = np.sum(di * w) +   self.global_dev

            message = f"Outer iteration {iteration_outer}: Fitting Parameter {param}: Inner iteration {iteration_inner}: Current LL {dv}"
            self._print_message(message=message, level=3)

        return dv

    def predict(self, X, what="response", return_contributions=False):
        """Predict the distribution parameters given input data.

        Args:
            X: Design matrix.
            what: Predict the response or the link. Defaults to "response".
            return_contributions: Whether to return contributions of individual covariates. Defaults to False.

        Raises:
            ValueError: If 'what' is not 'link' or 'response'.

        Returns:
            Predicted values for the distribution.
        """
        X_scaled = self.scaler.transform(X=X)
        X_dict = {
            p: self.make_model_array(X_scaled, p)
            for p in range(self.distribution.n_params)
        }
        prediction = [x @ b.T for x, b in zip(X_dict.values(), self.beta.values())]

        if return_contributions:
            contribution = [x * b.T for x, b in zip(X_dict.values(), self.beta.values())]

        if what == "response":
            prediction = [
                self.distribution.link_inverse(p, param=i)
                for i, p in enumerate(prediction)
            ]
        elif what == "link":
            pass
        else:
            raise ValueError("Should be 'response' or 'link'.")

        prediction = np.stack(prediction, axis=1)

        if return_contributions:
            return prediction, contribution
        else:
            return prediction
