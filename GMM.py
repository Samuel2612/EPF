# -*- coding: utf-8 -*-
"""
Created on Fri Jun 27 12:42:23 2025

@author: samue
"""

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal

# ----- 1.  Build training matrix -------------------------------------------
df = pd.read_csv("df_2021.csv")
df_2022 = pd.read_csv("df_2022.csv")
# y_t is VWAP at t, X_t are exogenous features at t
Y  = df['VWAP'].shift(-1).dropna().values[:, None]   # target is t+1 price
X  = df[['solar_DA','wind_DA','mos','VWAP','trade']].iloc[:-1].values
YZ = np.hstack([Y, X])      # joint vector [price_next, features_now]


from sklearn.preprocessing import StandardScaler
scaler_y = StandardScaler().fit(Y)
scaler_x = StandardScaler().fit(X)
Yz = scaler_y.transform(Y)
Xz = scaler_x.transform(X)
YZz = np.hstack([Yz, Xz])


K = 3
gmm = GaussianMixture(n_components=K, covariance_type='full',
                      max_iter=500, random_state=0).fit(YZz)


def gmm_conditional_mean(x_now):
    x_now_z = scaler_x.transform(x_now.reshape(1,-1)).ravel()

    cond_num, cond_den = 0., 0.
    for k, (w, mu, cov) in enumerate(zip(gmm.weights_,
                                         gmm.means_,
                                         gmm.covariances_)):
        mu_y, mu_x  = mu[0], mu[1:]
        Sig_yy      = cov[0,0]
        Sig_yx      = cov[0,1:]
        Sig_xx      = cov[1:,1:]

        # conditional mean μ_y|x
        mu_cond = mu_y + Sig_yx @ np.linalg.solve(Sig_xx, x_now_z - mu_x)

        # weight the component by p_k(x_now)
        px_k = w * multivariate_normal.pdf(x_now_z, mean=mu_x, cov=Sig_xx)
        cond_num += px_k * mu_cond
        cond_den += px_k

    y_hat_z = cond_num / cond_den
    return scaler_y.inverse_transform([[y_hat_z]])[0,0]

# example:
x_now = df[['solar_DA','wind_DA','mos','VWAP','trade']].iloc[-1].values
price_next_hat = gmm_conditional_mean(x_now)
