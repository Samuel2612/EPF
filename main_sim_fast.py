"""
Fully vectorised intraday Monte-Carlo simulation
===============================================

author :  <you>
date   :  2025-05-01
"""


import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import johnsonsu
from scipy.special import expit              # fast SIMD sigmoid
from sklearn.linear_model import LogisticRegression
from GAMLSS import GAMLSS                    # your own wrapper


N_SIMS     = 1_000                           
T          = 36                              
HOURS      = [h for h in range(24) if h != 2]  # 23 hours = index 0…22
DTYPE      = np.float32                      # less memory
SEED       = 42                              
rng        = np.random.default_rng(SEED)


print('loading CSV …')
df_2021 = pd.read_csv('df_2021.csv')
df_2022 = pd.read_csv('df_2022.csv')

for df in (df_2021, df_2022):
    df['delivery_start'] = (
        pd.to_datetime(df['delivery_start'], utc=True)
          .dt.tz_convert('Europe/Berlin')
    )

df_all = pd.concat([df_2021, df_2022], ignore_index=True)
df_all['day'] = df_all['delivery_start'].dt.floor('D')
df_all['abs_vwap_changes_lag7_12'] = (
      df_all['abs_vwap_changes_lag7']  + df_all['abs_vwap_changes_lag8']
    + df_all['abs_vwap_changes_lag9']  + df_all['abs_vwap_changes_lag10']
    + df_all['abs_vwap_changes_lag11'] + df_all['abs_vwap_changes_lag12']
)
df_P = df_all[df_all['alpha'] == 1].copy()


col_gamlss = [
    'vwap_changes_lag1', 'vwap_changes_lag2', 'vwap_changes_lag3',
    'abs_vwap_changes_lag1', 'abs_vwap_changes_lag2', 'abs_vwap_changes_lag3',
    'abs_vwap_changes_lag4', 'abs_vwap_changes_lag5', 'abs_vwap_changes_lag6',
    'abs_vwap_changes_lag7_12', 'alpha_lag1', 'alpha_lag2', 'f_TTD',
    'MON', 'SAT', 'SUN',
    'mo_slope_500', 'mo_slope_1000', 'mo_slope_2000',
    'wind_forecast', 'solar_forecast', 'da-id'
]
base_alpha = [
    'vwap_changes_lag1', 'vwap_changes_lag2', 'vwap_changes_lag3',
    'abs_vwap_changes_lag1', 'abs_vwap_changes_lag2', 'abs_vwap_changes_lag3',
    'abs_vwap_changes_lag4', 'abs_vwap_changes_lag5', 'abs_vwap_changes_lag6',
    'abs_vwap_changes_lag7_12',
] + [f'alpha_lag{i}' for i in range(1, 13)] + [
    'MON', 'SAT', 'SUN',
    'mo_slope_500', 'mo_slope_1000', 'mo_slope_2000',
    'wind_forecast', 'solar_forecast', 'da-id'
]
ttd_bin_feats = [f'TTD_bin_{i}' for i in range(1, 37)]
col_logreg = base_alpha + ttd_bin_feats

# -- every feature that is ever read or written must be in `col_full`
extra_runtime_feats = [
    'vwap_lag1', 'vwap_changes_lag1', 'vwap_changes_lag2', 'vwap_changes_lag3',
    *[f'abs_vwap_changes_lag{i}' for i in range(1, 13)],
    *[f'alpha_lag{i}' for i in range(1, 13)],
    'f_TTD', 'da-id', 'da_price'
]
col_full = list(dict.fromkeys(col_logreg + col_gamlss + extra_runtime_feats))

# mapping column name → integer position inside the “feature tensor”
pos = {c: i for i, c in enumerate(col_full)}
def idx(name: str) -> int:          
    return pos[name]

# indices for bulk updates
idx_bin_start = idx('TTD_bin_1')
idx_bin_end   = idx('TTD_bin_36') + 1      


min_day = df_P['day'].min()
max_day = df_P['day'].max()
first_day = min_day + pd.Timedelta(days=365)     # need 1-year history
days = np.arange(first_day, max_day + pd.Timedelta(days=1), dtype='datetime64[D]')
D, H = len(days), len(HOURS)

print(f'training models for {D} days × {H} hours …')

#storage for fitted parameters
F_LR  = len(col_logreg)
F_GS  = len(col_gamlss)
P_GS  = 4                               # Johnson-SU has 4 parameters

w_lr = np.empty((D, H, F_LR), dtype=DTYPE)
b_lr = np.empty((D, H),        dtype=DTYPE)
w_gs = np.empty((D, H, P_GS, F_GS), dtype=DTYPE)
c_gs = np.empty((D, H, P_GS),        dtype=DTYPE)

start_feat = np.empty((D, H), dtype=object)   # 1 row per (day, hour)

for d, day in enumerate(days):
    window_start = (day - np.timedelta64(365, 'D')).astype('datetime64[ns]')
    m_log = (df_all['day'] >= window_start) & (df_all['day'] < day)
    m_gam = m_log & (df_all['alpha'] == 1)

    df_train_log = df_all[m_log]
    df_train_gam = df_P[m_gam]

    df_today = df_all[df_all['day'] == day]

    for h, hour in enumerate(HOURS):
        #  logistic-regression  
        _log  = df_train_log[df_train_log['delivery_start'].dt.hour == hour]
        X_lr  = _log[col_logreg].to_numpy()
        y_lr  = _log['alpha'].to_numpy(dtype=int)
        lr    = LogisticRegression(penalty='l1', solver='liblinear',
                                   max_iter=1000).fit(X_lr, y_lr)
        w_lr[d, h] = lr.coef_.astype(DTYPE).ravel()
        b_lr[d, h] = lr.intercept_.astype(DTYPE)

        #  GAMLSS  
        _gam  = df_train_gam[df_train_gam['delivery_start'].dt.hour == hour]
        X_gs  = _gam[col_gamlss].to_numpy()
        y_gs  = _gam['vwap_changes'].to_numpy()
        gs    = GAMLSS()
        gs.fit(X_gs, y_gs)
        for k in range(P_GS):
            w_gs[d, h, k] = gs.coef_[k].astype(DTYPE)
            c_gs[d, h, k] = gs.intercept_[k].astype(DTYPE)


        start_feat[d, h] = (
            df_today[df_today['delivery_start'].dt.hour == hour].iloc[0]
        )

print('building tensors …')
P = N_SIMS
F = len(col_full)

paths = np.empty((D, H, T, P), dtype=DTYPE)          
feat  = np.empty((D, H, P, F), dtype=DTYPE)          # dynamic features

for d in range(D):
    for h in range(H):
        feat[d, h, :, :] = start_feat[d, h][col_full].to_numpy(DTYPE)

vwap_last = feat[..., idx('vwap_lag1')]               

print('running vectorised simulation …')
for ttd in range(T):                                  
    #  alpha_t via logistic-regression   --------------------------------------
    z = (feat[..., :F_LR] * w_lr[:, :, None, :]).sum(-1) + b_lr[:, :, None]
    p_alpha = expit(z, out=z)                         
    alpha = rng.random(p_alpha.shape, dtype=DTYPE) < p_alpha

    #  delta_t via Johnson-SU   -----------------------------------------------
    delta = np.zeros_like(vwap_last, dtype=DTYPE)
    if alpha.any():
        sel = np.nonzero(alpha)                       # tuple of 3 arrays
        f_sel = feat[sel][:, F_LR:].astype(DTYPE)     # (S, F_GS)
        pars = (f_sel @ w_gs[sel]) + c_gs[sel]        # broadcast
        mu,sig,nu,tau = pars.T
        delta[sel] = johnsonsu(mu,sig,nu,tau).rvs(random_state=rng)


    vwap_new = vwap_last + delta
    paths[..., ttd, :] = vwap_new                     # store (D, H, P)


    feat[..., idx('vwap_changes_lag3')] = feat[..., idx('vwap_changes_lag2')]
    feat[..., idx('vwap_changes_lag2')] = feat[..., idx('vwap_changes_lag1')]
    feat[..., idx('vwap_changes_lag1')] = delta

  
    sl_abs = slice(idx('abs_vwap_changes_lag1')-11, idx('abs_vwap_changes_lag1')+1)
    feat[..., sl_abs] = np.roll(feat[..., sl_abs], shift=1, axis=-1)
    feat[..., idx('abs_vwap_changes_lag1')] = np.abs(delta)


    sl_alpha = slice(idx('alpha_lag1')-11, idx('alpha_lag1')+1)
    feat[..., sl_alpha] = np.roll(feat[..., sl_alpha], shift=1, axis=-1)
    feat[..., idx('alpha_lag1')] = alpha.astype(DTYPE)


    feat[..., idx('vwap_lag1')] = vwap_last
    feat[..., idx('f_TTD')]     = 1.0 / np.sqrt(1.0 + (T - ttd - 1))
    feat[..., idx('da-id')]     = np.abs(feat[..., idx('da_price')] - vwap_last)


    feat[..., idx_bin_start:idx_bin_end] = 0
    feat[..., idx_bin_start + (T - ttd - 1)] = 1


    vwap_last = vwap_new

print('simulation done – tensor `paths` is ready!  shape =', paths.shape)

