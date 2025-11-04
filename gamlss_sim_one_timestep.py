# -*- coding: utf-8 -*-
"""
Created on Fri Oct 10 14:33:35 2025

@author: samue
"""

"""
One-step-ahead intraday forecasting with zero-inflated JSU:
- Train (per day, per hour) a Bernoulli logit for π_t and a JSU GAMLSS for ΔP | α=1
- For each TTD row: predict π_t and JSU(μ,σ,ν,τ), compute:
    cond_mean = π_t μ
    cond_var  = π_t σ^2 + π_t(1-π_t) μ^2
    log_score for the zero-inflated mixture
    CRPS via Monte-Carlo (fast, robust)
- Save everything to one CSV.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.linear_model import LogisticRegression
import scipy.stats as st

# Your modules
from GAMLSS_x import GAMLSS
from distributions import JSUo, JSU, jsu_reparam_to_original  # uses your reparam (μ,σ,ν,τ)->(μ1,σ1,ν1,τ1)

# -----------------------
# Config
# -----------------------
LOCAL_TZ = "Europe/Berlin"
N_MC = 1000            # Monte-Carlo samples for CRPS
RNG_SEED = 42          # reproducibility
HOURS = [h for h in range(24)]  # skip DST 02:00

# -----------------------
# Data loading
# -----------------------
df_2021 = pd.read_csv("df_2021.csv")
df_2022 = pd.read_csv("df_2022.csv")

df_2021['delivery_start'] = pd.to_datetime(df_2021['delivery_start'], utc=True).dt.tz_convert(LOCAL_TZ)
df_2022['delivery_start'] = pd.to_datetime(df_2022['delivery_start'], utc=True).dt.tz_convert(LOCAL_TZ)

df_all = pd.concat([df_2021, df_2022], ignore_index=True)
df_all['day'] = df_all['delivery_start'].dt.floor('D')

# helper agg (as in your code)
df_all['abs_vwap_changes_lag7_12'] = (
    df_all['abs_vwap_changes_lag7']  + df_all['abs_vwap_changes_lag8']  +
    df_all['abs_vwap_changes_lag9']  + df_all['abs_vwap_changes_lag10'] +
    df_all['abs_vwap_changes_lag11'] + df_all['abs_vwap_changes_lag12']
)

# subset for JSU fit (only α=1)
df_P = df_all[df_all['alpha'] == 1].copy()

min_day = df_P['day'].min()
max_day = df_P['day'].max()
first_possible_day = min_day + pd.Timedelta(days=365)
all_days = pd.date_range(start=first_possible_day, end=max_day, freq='D')

# -----------------------
# Feature sets
# -----------------------
col_gamlss = [
    'vwap_changes_lag1', 'vwap_changes_lag2', 'vwap_changes_lag3',
    'abs_vwap_changes_lag1', 'abs_vwap_changes_lag2', 'abs_vwap_changes_lag3',
    'abs_vwap_changes_lag4', 'abs_vwap_changes_lag5', 'abs_vwap_changes_lag6',
    'abs_vwap_changes_lag7_12', 'alpha_lag1', 'alpha_lag2', 'f_TTD',
    'MON', 'SAT', 'SUN', 'mo_slope_500', 'mo_slope_1000', 'mo_slope_2000',
    'wind_forecast', 'solar_forecast', 'da-id'
]

base_alpha = [
    'vwap_changes_lag1', 'vwap_changes_lag2', 'vwap_changes_lag3',
    'abs_vwap_changes_lag1', 'abs_vwap_changes_lag2', 'abs_vwap_changes_lag3',
    'abs_vwap_changes_lag4', 'abs_vwap_changes_lag5', 'abs_vwap_changes_lag6',
    'abs_vwap_changes_lag7_12',
    'alpha_lag1','alpha_lag2','alpha_lag3','alpha_lag4','alpha_lag5','alpha_lag6',
    'alpha_lag7','alpha_lag8','alpha_lag9','alpha_lag10','alpha_lag11','alpha_lag12',
    'MON','SAT','SUN','mo_slope_500','mo_slope_1000','mo_slope_2000',
    'wind_forecast','solar_forecast','da-id'
]
ttd_bin_feats = [f"TTD_bin_{i}" for i in range(1, 37)]
col_logreg = base_alpha + ttd_bin_feats

# -----------------------
# Helpers (JSU + scores)
# -----------------------

def reparam_to_scipy(mu, sigma, nu, tau):
    """
    Your GAMLSS returns JSU(μ, σ, ν, τ) with:
      - μ = mean of Y
      - σ = standard deviation of Y   (<< IMPORTANT: σ is SD, not var)
    Convert to SciPy johnsonsu params: a=ν1, b=τ1, loc=μ1, scale=σ1.
    """
    mu1, sigma1, nu1, tau1 = jsu_reparam_to_original(float(mu), float(sigma), float(nu), float(tau))
    return float(nu1), float(tau1), float(mu1), float(sigma1)

def mixture_moments(pi, mu, sigma_sd):
    """
    Moments of ΔP = α·Y, P(α=1)=pi, Y ~ JSU with mean=mu, sd=sigma_sd.
    """
    mu  = float(mu)
    var = float(sigma_sd) ** 2
    pi  = float(pi)
    cond_mean = pi * mu
    cond_var  = pi * var + pi * (1.0 - pi) * (mu ** 2)
    return cond_mean, cond_var

def logscore_zero_infl_jsu(y, pi, a, b, loc, scale):
    """
    Log p(y) for mixture (1-π)*δ0 + π*JSU(a,b,loc,scale).
    For y=0, only the point mass counts.
    """
    y = float(y); pi = float(pi)
    if np.isclose(y, 0.0):
        return np.log(max(1.0 - pi, 1e-300))
    pdf = st.johnsonsu(a, b, loc=loc, scale=scale).pdf(y)
    return np.log(max(pi * pdf, 1e-300))

def crps_mc_zero_infl_jsu(y, pi, a, b, loc, scale, M=1000, rng=None):
    """
    CRPS(F,y) = E|X-y| - 0.5 E|X-X'|
    Estimated from Monte-Carlo samples of the zero-inflated mixture.
    O(M log M) using the sorting trick for the pairwise term.
    """
    rng = np.random.default_rng(rng)
    alpha = rng.random(M) < pi
    x = np.zeros(M)
    n_pos = int(alpha.sum())
    if n_pos > 0:
        x[alpha] = st.johnsonsu(a, b, loc=loc, scale=scale).rvs(size=n_pos, random_state=rng)

    term1 = np.mean(np.abs(x - y))  # E|X - y|

    xs = np.sort(x)
    coeff = (2*np.arange(1, M+1) - M - 1)  # (2k - M - 1)
    sum_pair = np.sum(coeff * xs)
    e_abs_xx = (2.0 / (M**2)) * sum_pair
    return float(term1 - 0.5 * e_abs_xx)

def infer_ttd_from_bins(row):
    for k in range(1, 37):
        col = f"TTD_bin_{k}"
        if col in row and int(row[col]) == 1:
            return k
    return None

# -----------------------
# 1-step-ahead metrics for all TTD rows of (day, hour)
# -----------------------
def one_step_forecast_metrics(df_h_seq: pd.DataFrame,
                              model_logreg,
                              model_gamlss,
                              col_logreg: list,
                              col_gamlss: list,
                              n_mc: int,
                              rng_seed: int) -> pd.DataFrame:
    """
    df_h_seq: all TTD rows for this (day, hour) — already contains only 'current_day'.
    Returns per-row metrics without simulating paths.
    """
    # Use f_TTD if present to order far -> near (monotone in TTD).
    if 'f_TTD' in df_h_seq.columns:
        df_h_seq = df_h_seq.sort_values('f_TTD')

    rows = []
    for _, feats in df_h_seq.iterrows():
        # feature vectors
        Xg = feats[col_gamlss].values
        Xl = feats[col_logreg].values

        # predict π_t
        pi_hat = float(model_logreg.predict_proba(Xl.reshape(1, -1))[0, 1])

        # predict JSU(μ, σ (sd), ν, τ)
        mu_hat, sigma_hat, nu_hat, tau_hat = map(float, model_gamlss.predict(Xg))

        # conditional moments for ΔP
        cond_mean, cond_var = mixture_moments(pi_hat, mu_hat, sigma_hat)

        # point forecast for price using E[ΔP]
        price_prev = float(feats['vwap_lag1'])
        price_true = float(feats['vwap'])
        delta_true = float(feats['vwap_changes'])
        price_pred = price_prev + cond_mean
        err = price_true - price_pred

        # convert to SciPy johnsonsu for scoring
        a, b, loc, scale = reparam_to_scipy(mu_hat, sigma_hat, nu_hat, tau_hat)

        # scores
        log_sc = logscore_zero_infl_jsu(delta_true, pi_hat, a, b, loc, scale)
        crps   = crps_mc_zero_infl_jsu(delta_true, pi_hat, a, b, loc, scale, M=n_mc, rng=rng_seed)

        rows.append({
            'day': feats['day'],
            'hour': int(feats['delivery_start'].hour),
            'ttd': infer_ttd_from_bins(feats),
            'vwap_prev': price_prev,
            'price_true': price_true,
            'delta_true': delta_true,
            'alpha_true': float(feats['alpha']),
            # predictions / params
            'pi_hat': pi_hat,
            'mu_hat': mu_hat, 'sigma_hat_sd': sigma_hat, 'nu_hat': nu_hat, 'tau_hat': tau_hat,
            'cond_mean': cond_mean, 'cond_var': cond_var,
            'price_pred': price_pred,
            # errors & scores
            'err': err, 'abs_err': abs(err), 'sq_err': err**2,
            'log_score': log_sc, 'crps': crps
        })

    return pd.DataFrame(rows)

# -----------------------
# Main rolling-loop
# -----------------------
gamlss_results = {}   # {day: {hour: model_gamlss}}
logreg_results = {}   # {day: {hour: model_logreg}}
metrics_rows   = []   # collect per-step metrics

rng_seed = RNG_SEED

for current_day in all_days:
    print(current_day)
    window_start = current_day - pd.Timedelta(days=365)

    # rolling training windows (strictly before current_day)
    mask_all = (df_all['day'] >= window_start) & (df_all['day'] < current_day)
    mask_P   = (df_P['day']   >= window_start) & (df_P['day']   < current_day)

    df_train_logreg = df_all[mask_all].copy()
    df_train_gamlss = df_P[mask_P].copy()

    # rows to score on this current_day
    df_sim = df_all[df_all['day'] == current_day].copy()
    if df_sim.empty:
        continue

    day_dict_gamlss = {}
    day_dict_logreg = {}

    for hour in HOURS:
        # ----- Fit models for this hour -----
        # JSU GAMLSS (only α=1 rows)
        df_h_gamlss = df_train_gamlss[df_train_gamlss['delivery_start'].dt.hour == hour].copy()
        if df_h_gamlss.empty:
            continue
        X_gamlss = df_h_gamlss[col_gamlss].values
        y_gamlss = df_h_gamlss['vwap_changes'].values

        model_gamlss = GAMLSS(distribution=JSUo())
        model_gamlss.fit(X_gamlss, y_gamlss)
        day_dict_gamlss[hour] = model_gamlss

        # Logit for π_t
        df_h_logreg = df_train_logreg[df_train_logreg['delivery_start'].dt.hour == hour].copy()
        if df_h_logreg.empty:
            continue
        X_logreg = df_h_logreg[col_logreg].values
        y_logreg = df_h_logreg['alpha'].values

        model_logreg = LogisticRegression(
            penalty='l1',
            solver='liblinear',
            max_iter=1000
        )
        model_logreg.fit(X_logreg, y_logreg)
        day_dict_logreg[hour] = model_logreg

        # ----- Score 1-step-ahead for all TTD rows for this (day, hour) -----
        df_h_seq = df_sim[df_sim['delivery_start'].dt.hour == hour].copy()
        if df_h_seq.empty:
            continue

        metrics_df = one_step_forecast_metrics(
            df_h_seq=df_h_seq,
            model_logreg=model_logreg,
            model_gamlss=model_gamlss,
            col_logreg=col_logreg,
            col_gamlss=col_gamlss,
            n_mc=N_MC,
            rng_seed=rng_seed
        )
        metrics_rows.append(metrics_df)

    gamlss_results[current_day] = day_dict_gamlss
    logreg_results[current_day] = day_dict_logreg

# -----------------------
# Save results
# -----------------------
results_df = pd.concat(metrics_rows, ignore_index=True) if metrics_rows else pd.DataFrame()
results_df.to_csv("one_step_forecast_metrics.csv", index=False)
print("Saved metrics to one_step_forecast_metrics.csv")

# (optional) quick sanity stats
if not results_df.empty:
    print(results_df[['sq_err','crps','log_score']].describe())
