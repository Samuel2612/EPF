"""
One-step-ahead intraday forecasting (15-min bins) with zero-inflated JSU + neighbor features

- Keep ONLY the last 4 hours before delivery (TTD <= 16 with 15-min bins).
- Train per (day, hour) on those rows from the past 365 days (rolling).
- Evaluate today using the last available information within this 4h window.
- Neighbor ΔVWAP features for offsets -2, -1, +1, +2 hours:
    * If neighbor is closed (n_ttd < 1): use TTD=1 (last pre-closure).
    * If 1 <= n_ttd <= 16 but exact value missing: use the last earlier obs (search k = n_ttd+1..16).
    * If n_ttd > 16 (not yet in last 4h): use 0.
- Scoring: log-score and CRPS (Monte-Carlo), conditional mean/variance.
"""

import re
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import scipy.stats as st

# Your modules
from GAMLSS_x import GAMLSS
from distributions import JSU, jsu_reparam_to_original

# -----------------------
# Config (15-min bins)
# -----------------------
LOCAL_TZ = "Europe/Berlin"
N_MC = 1000
RNG_SEED = 42
HOURS = list(range(24))
SLOTS_PER_HOUR = 4                  # 15-min bins
TTD_LAST = 4 * SLOTS_PER_HOUR       # last 4 hours => 16 bins
OFFSETS = [-2, -1, +1, +2]          # neighbor hours

# -----------------------
# Data loading
# -----------------------
df_2021 = pd.read_csv("df_2021_MVT_dst_10.csv")
df_2022 = pd.read_csv("df_2022_MVT_dst_10.csv")

df_2021['delivery_start'] = pd.to_datetime(df_2021['delivery_start'], utc=True).dt.tz_convert(LOCAL_TZ)
df_2022['delivery_start'] = pd.to_datetime(df_2022['delivery_start'], utc=True).dt.tz_convert(LOCAL_TZ)

df_all = pd.concat([df_2021, df_2022], ignore_index=True)
df_all['day']  = df_all['delivery_start'].dt.floor('D')
df_all['hour'] = df_all['delivery_start'].dt.hour

# helper agg
df_all['abs_vwap_changes_lag5_8'] = (
    df_all['abs_vwap_changes_lag5'] + df_all['abs_vwap_changes_lag6'] +
    df_all['abs_vwap_changes_lag7'] + df_all['abs_vwap_changes_lag8']
)

# -----------------------
# Detect ALL TTD columns; compute ttd_idx using all of them
# -----------------------
def get_ttd_cols(df: pd.DataFrame):
    patt = re.compile(r"^TTD_bin_(\d+)$")
    pairs = []
    for c in df.columns:
        m = patt.match(c)
        if m:
            pairs.append((int(m.group(1)), c))
    pairs.sort()
    idxs, cols = zip(*pairs)
    return list(cols), int(max(idxs))

TTD_COLS_ALL, T_TOTAL = get_ttd_cols(df_all)

def extract_ttd_idx_all(df: pd.DataFrame) -> pd.Series:
    idx = df[TTD_COLS_ALL].idxmax(axis=1)
    return idx.str.extract(r'(\d+)')[0].astype(int)

df_all['ttd_idx_all'] = extract_ttd_idx_all(df_all)

# Keep ONLY last 4 hours (TTD<=16)
df_all['in_last4h'] = df_all['ttd_idx_all'] <= TTD_LAST

# For α=1 subset
df_P = df_all[df_all['alpha'] == 1].copy()

min_day = df_P['day'].min()
max_day = df_P['day'].max()
first_possible_day = min_day + pd.Timedelta(days=365)
all_days = pd.date_range(start=first_possible_day, end=max_day, freq='D')

# -----------------------
# Features
# -----------------------
col_gamlss_base = [
    'vwap_changes_lag1', 'vwap_changes_lag2', 'vwap_changes_lag3',
    'abs_vwap_changes_lag1', 'abs_vwap_changes_lag2', 'abs_vwap_changes_lag3',
    'abs_vwap_changes_lag4', 'abs_vwap_changes_lag5_8',
    'alpha_lag1', 'alpha_lag2', 'f_TTD',
    'MON', 'SAT', 'SUN', 'mo_slope_500', 'mo_slope_1000', 'mo_slope_2000',
    'wind_forecast', 'solar_forecast', 'da-id'
]

base_alpha = [
    'vwap_changes_lag1', 'vwap_changes_lag2', 'vwap_changes_lag3',
    'abs_vwap_changes_lag1', 'abs_vwap_changes_lag2', 'abs_vwap_changes_lag3',
    'abs_vwap_changes_lag4', 'abs_vwap_changes_lag5_8',
    'alpha_lag1','alpha_lag2','alpha_lag3','alpha_lag4','alpha_lag5','alpha_lag6',
    'alpha_lag7','alpha_lag8',
    'MON','SAT','SUN','mo_slope_500','mo_slope_1000','mo_slope_2000',
    'wind_forecast','solar_forecast','da-id'
]

# Only the last-4h TTD indicator columns are relevant now
TTD_COLS_LAST4 = [f"TTD_bin_{i}" for i in range(1, TTD_LAST+1)]

# neighbor feature names
NB_VALS = { -2: 'nb_dchg_m2', -1: 'nb_dchg_m1', +1: 'nb_dchg_p1', +2: 'nb_dchg_p2' }
col_neighbors = list(NB_VALS.values())

col_gamlss = col_gamlss_base + col_neighbors
col_logreg = base_alpha + TTD_COLS_LAST4 + col_neighbors

# -----------------------
# Lookups & neighbor helpers (built on LAST-4H rows only)
# -----------------------
def build_last4h_lookups(df: pd.DataFrame):
    sub = df[df['in_last4h']].copy()
    sub['ttd_idx'] = sub['ttd_idx_all']          # alias
    # scalar lookup for exact (day, hour, ttd_idx) if present
    delta_exact = sub.groupby(['day','hour','ttd_idx'])['vwap_changes'].mean().sort_index()
    # scalar lookup for ttd==1 (last pre-closure) per (day, hour)
    last_known = sub[sub['ttd_idx'] == 1].groupby(['day','hour'])['vwap_changes'].mean().sort_index()
    return delta_exact, last_known

delta_exact, last_known = build_last4h_lookups(df_all)

def day_hour_shift(day_ts: pd.Timestamp, hour: int, offset_hours: int):
    total = hour + offset_hours
    shift_days, nhour = divmod(total, 24)
    nday = day_ts + pd.Timedelta(days=int(shift_days))
    return nday, int(nhour)

def get_neighbor_last_available(day, hour, ttd_idx_cur, offset):
    """
    Use LAST AVAILABLE ΔVWAP of neighbor before the forecast time, within last-4h window.
      n_ttd = ttd_idx_cur + offset*SLOTS_PER_HOUR
      - if n_ttd < 1    → use ttd=1 (closed case)
      - if 1 <= n_ttd <= 16:
            try exact n_ttd;
            else search back to earlier obs: k = n_ttd+1..16 (first found);
            if none → 0
      - if n_ttd > 16   → 0 (neighbor not yet in last-4h window)
    """
    nday, nhour = day_hour_shift(day, hour, offset)
    n_ttd = int(ttd_idx_cur + offset * SLOTS_PER_HOUR)

    if n_ttd < 1:
        try:
            return float(last_known.loc[(nday, nhour)])
        except KeyError:
            return 0.0

    if n_ttd > TTD_LAST:
        return 0.0

    # try exact first
    try:
        return float(delta_exact.loc[(nday, nhour, n_ttd)])
    except KeyError:
        # search earlier obs (larger ttd) within last-4h window
        for k in range(n_ttd + 1, TTD_LAST + 1):
            try:
                return float(delta_exact.loc[(nday, nhour, k)])
            except KeyError:
                continue
        return 0.0

def add_neighbor_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # ensure ttd within full range is known, then restrict later
    if 'ttd_idx_all' not in out:
        out['ttd_idx_all'] = extract_ttd_idx_all(out)
    if 'hour' not in out:
        out['hour'] = out['delivery_start'].dt.hour

    # compute neighbors using LAST-4H logic
    vals_dict = {NB_VALS[o]: [] for o in OFFSETS}
    for day, hour, ttd_all in zip(out['day'], out['hour'], out['ttd_idx_all']):
        for o in OFFSETS:
            vals_dict[NB_VALS[o]].append(get_neighbor_last_available(day, int(hour), int(ttd_all), o))
    for name, arr in vals_dict.items():
        out[name] = np.array(arr, dtype=float)

    # keep only last 4h rows
    out['ttd_idx'] = out['ttd_idx_all']
    out = out[out['ttd_idx'] <= TTD_LAST].copy()

    # f_TTD fallback if missing
    if 'f_TTD' not in out.columns or out['f_TTD'].isna().any():
        out['f_TTD'] = 1.0 / np.sqrt(1.0 + out['ttd_idx'])

    return out

# -----------------------
# Robust design matrices
# -----------------------
def ensure_columns(df: pd.DataFrame, cols: list):
    miss = [c for c in cols if c not in df.columns]
    if miss:
        for c in miss:
            df[c] = 0.0
    return df

def design_matrix(df: pd.DataFrame, cols: list) -> np.ndarray:
    df = ensure_columns(df, cols)
    X = (df[cols]
         .replace([np.inf, -np.inf], np.nan)
         .fillna(0.0)
         .to_numpy(dtype=float, copy=False))
    return X

def vector_from_row(row: pd.Series, cols: list) -> np.ndarray:
    vals = []
    for c in cols:
        v = row.get(c, 0.0)
        if pd.isna(v) or v in (np.inf, -np.inf):
            v = 0.0
        vals.append(float(v))
    return np.array(vals, dtype=float)

# -----------------------
# JSU helpers & scoring
# -----------------------
def reparam_to_scipy(mu, sigma_sd, nu, tau):
    mu1, sigma1, nu1, tau1 = jsu_reparam_to_original(float(mu), float(sigma_sd), float(nu), float(tau))
    return float(nu1), float(tau1), float(mu1), float(sigma1)

def mixture_moments(pi, mu, sigma_sd):
    mu = float(mu); var = float(sigma_sd)**2; pi = float(pi)
    cond_mean = pi * mu
    cond_var  = pi * var + pi * (1.0 - pi) * (mu ** 2)
    return cond_mean, cond_var

def logscore_zero_infl_jsu(y, pi, a, b, loc, scale):
    y = float(y); pi = float(pi)
    if np.isclose(y, 0.0):
        return np.log(max(1.0 - pi, 1e-300))
    pdf = st.johnsonsu(a, b, loc=loc, scale=scale).pdf(y)
    return np.log(max(pi * pdf, 1e-300))

def crps_mc_zero_infl_jsu(y, pi, a, b, loc, scale, M=1000, rng=None):
    rng = np.random.default_rng(rng)
    alpha = rng.random(M) < pi
    x = np.zeros(M)
    n_pos = int(alpha.sum())
    if n_pos > 0:
        x[alpha] = st.johnsonsu(a, b, loc=loc, scale=scale).rvs(size=n_pos, random_state=rng)
    term1 = np.mean(np.abs(x - y))
    xs = np.sort(x)
    coeff = (2*np.arange(1, M+1) - M - 1)
    sum_pair = np.sum(coeff * xs)
    e_abs_xx = (2.0 / (M**2)) * sum_pair
    return float(term1 - 0.5 * e_abs_xx)

# -----------------------
# 1-step metrics for a (day, hour)
# -----------------------
def one_step_forecast_metrics(df_h_seq: pd.DataFrame,
                              model_logreg,
                              model_gamlss,
                              col_logreg: list,
                              col_gamlss: list,
                              n_mc: int,
                              rng_seed: int) -> pd.DataFrame:
    # order far -> near if available
    if 'f_TTD' in df_h_seq.columns:
        df_h_seq = df_h_seq.sort_values('f_TTD')

    rows = []
    for _, feats in df_h_seq.iterrows():
        Xg = vector_from_row(feats, col_gamlss)
        Xl = vector_from_row(feats, col_logreg)

        pi_hat = float(model_logreg.predict_proba(Xl.reshape(1, -1))[0, 1])
        mu_hat, sigma_hat, nu_hat, tau_hat = map(float, model_gamlss.predict(Xg))

        cond_mean, cond_var = mixture_moments(pi_hat, mu_hat, sigma_hat)
        price_prev  = float(feats['vwap_lag1'])
        price_true  = float(feats['vwap'])
        delta_true  = float(feats['vwap_changes'])
        price_pred  = price_prev + cond_mean
        err         = price_true - price_pred

        a, b, loc, scale = reparam_to_scipy(mu_hat, sigma_hat, nu_hat, tau_hat)
        log_sc = logscore_zero_infl_jsu(delta_true, pi_hat, a, b, loc, scale)
        crps   = crps_mc_zero_infl_jsu(delta_true, pi_hat, a, b, loc, scale, M=n_mc, rng=rng_seed)

        row_out = {
            'day': feats['day'],
            'hour': int(feats['hour']),
            'ttd': int(feats['ttd_idx']),
            'vwap_prev': price_prev,
            'price_true': price_true,
            'delta_true': delta_true,
            'alpha_true': float(feats['alpha']),
            'pi_hat': pi_hat,
            'mu_hat': mu_hat, 'sigma_hat_sd': sigma_hat, 'nu_hat': nu_hat, 'tau_hat': tau_hat,
            'cond_mean': cond_mean, 'cond_var': cond_var,
            'price_pred': price_pred,
            'err': err, 'abs_err': abs(err), 'sq_err': err**2,
            'log_score': log_sc, 'crps': crps
        }
        # attach neighbor snapshot
        for name in col_neighbors:
            row_out[name] = float(feats.get(name, 0.0))

        rows.append(row_out)

    return pd.DataFrame(rows)

# -----------------------
# Main rolling loop (train per day/hour; last 365 days; last 4h only)
# -----------------------
gamlss_results = {}   # {day: {hour: model_gamlss}}
logreg_results = {}   # {day: {hour: model_logreg}}
metrics_rows   = []

for current_day in all_days[:30]:
    print(current_day)
    window_start = current_day - pd.Timedelta(days=365)

    # training windows: strictly before current_day AND last-4h rows only
    train_all = df_all[(df_all['day'] >= window_start) & (df_all['day'] < current_day)]
    train_all = add_neighbor_features(train_all)             # adds ttd_idx and filters to <=16
    train_P   = train_all[train_all['alpha'] == 1].copy()

    # rows to score for current_day: last-4h rows only, with neighbors
    df_sim = df_all[df_all['day'] == current_day].copy()
    df_sim = add_neighbor_features(df_sim)                   # adds ttd_idx and filters to <=16
    if df_sim.empty:
        continue

    day_dict_gamlss = {}
    day_dict_logreg = {}

    for hour in HOURS:
        df_h_gamlss = train_P[train_P['hour'] == hour]
        if df_h_gamlss.empty:
            continue
        X_gamlss = design_matrix(df_h_gamlss, col_gamlss)
        y_gamlss = df_h_gamlss['vwap_changes'].to_numpy(dtype=float, copy=False)
        mask = np.isfinite(y_gamlss)
        if mask.sum() < 5:
            continue
        model_gamlss = GAMLSS(distribution=JSU())
        model_gamlss.fit(X_gamlss[mask], y_gamlss[mask])
        day_dict_gamlss[hour] = model_gamlss

        df_h_logreg = train_all[train_all['hour'] == hour]
        if df_h_logreg.empty:
            continue
        X_logreg = design_matrix(df_h_logreg, col_logreg)
        y_logreg = df_h_logreg['alpha'].to_numpy(dtype=float, copy=False)
        mask_l = np.isfinite(y_logreg)
        if mask_l.sum() < 5:
            continue
        model_logreg = LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000)
        model_logreg.fit(X_logreg[mask_l], y_logreg[mask_l])
        day_dict_logreg[hour] = model_logreg

        # ----- Score 1-step-ahead for all TTD rows for this (day, hour) -----
        df_h_seq = df_sim[df_sim['hour'] == hour].copy()
        if df_h_seq.empty:
            continue

        metrics_df = one_step_forecast_metrics(
            df_h_seq=df_h_seq,
            model_logreg=model_logreg,
            model_gamlss=model_gamlss,
            col_logreg=col_logreg,
            col_gamlss=col_gamlss,
            n_mc=N_MC,
            rng_seed=RNG_SEED
        )
        metrics_rows.append(metrics_df)

    gamlss_results[current_day] = day_dict_gamlss
    logreg_results[current_day] = day_dict_logreg
    

# -----------------------
# Save results
# -----------------------
results_df = pd.concat(metrics_rows, ignore_index=True) if metrics_rows else pd.DataFrame()
results_df.to_csv("one_step_forecast_metrics_last4h_neighbors_365.csv", index=False)
print("Saved metrics to one_step_forecast_metrics_last4h_neighbors.csv")

if not results_df.empty:
    print(results_df[['sq_err','crps','log_score']].describe())
