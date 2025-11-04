# heatmaps_for_date_jsu.py
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
from sklearn.linear_model import LogisticRegression

# Your modules
from GAMLSS_x import GAMLSS
from distributions import JSU, JSUo

# -----------------------
# User inputs
# -----------------------
TARGET_DAY  = "2022-05-10"  # <-- set the date you want heatmaps for (YYYY-MM-DD)
WINDOW_DAYS = 365           # <-- how many days BEFORE TARGET_DAY to use for training

# -----------------------
# Constants (15-min bins)
# -----------------------
LOCAL_TZ = "Europe/Berlin"
HOURS = list(range(24))
SLOTS_PER_HOUR = 4
TTD_LAST = 4 * SLOTS_PER_HOUR  # last 4 hours => 16 bins
OFFSETS = [-2, -1, +1, +2]

# -----------------------
# Data
# -----------------------
df_2021 = pd.read_csv("df_2021_MVT_dst_10.csv")
df_2022 = pd.read_csv("df_2022_MVT_dst_10.csv")

# Parse timestamps
df_2021['delivery_start'] = pd.to_datetime(df_2021['delivery_start'], utc=True).dt.tz_convert(LOCAL_TZ)
df_2022['delivery_start'] = pd.to_datetime(df_2022['delivery_start'], utc=True).dt.tz_convert(LOCAL_TZ)

# <<< CHANGED: also ensure these two exist as proper datetimes for the neighbor self-join
df_2021['bin_timestamp'] = pd.to_datetime(df_2021['bin_timestamp'])
df_2022['bin_timestamp'] = pd.to_datetime(df_2022['bin_timestamp'])
df_2021['delivery_start_wall'] = pd.to_datetime(df_2021['delivery_start_wall'])
df_2022['delivery_start_wall'] = pd.to_datetime(df_2022['delivery_start_wall'])

df_all = pd.concat([df_2021, df_2022], ignore_index=True)
df_all['day']  = df_all['delivery_start'].dt.floor('D')
df_all['hour'] = df_all['delivery_start'].dt.hour

# helper aggregate
df_all['abs_vwap_changes_lag5_8'] = (
    df_all['abs_vwap_changes_lag5'] + df_all['abs_vwap_changes_lag6'] +
    df_all['abs_vwap_changes_lag7'] + df_all['abs_vwap_changes_lag8']
)

df_all['da-id_lag1'] = np.abs(df_all['da-id_lag1'])

# -----------------------
# TTD helpers
# -----------------------
def get_ttd_cols(df: pd.DataFrame):
    patt = re.compile(r"^TTD_bin_(\d+)$")
    pairs = []
    for c in df.columns:
        m = patt.match(c)
        if m:
            pairs.append((int(m.group(1)), c))
    if not pairs:
        raise ValueError("No TTD_bin_* columns found.")
    pairs.sort()
    idxs, cols = zip(*pairs)
    return list(cols), int(max(idxs))

TTD_COLS_ALL, _ = get_ttd_cols(df_all)

def extract_ttd_idx_all(df: pd.DataFrame) -> pd.Series:
    idx = df[TTD_COLS_ALL].idxmax(axis=1)
    return idx.str.extract(r'(\d+)')[0].astype(int)

df_all['ttd_idx_all'] = extract_ttd_idx_all(df_all)
df_all['in_last4h'] = df_all['ttd_idx_all'] <= TTD_LAST

# -----------------------
# Features (same as your one-step spec)
# -----------------------
col_gamlss_base = [
    'vwap_changes_lag1', 'vwap_changes_lag2', 'vwap_changes_lag3',
    'abs_vwap_changes_lag1', 'abs_vwap_changes_lag2', 'abs_vwap_changes_lag3',
    'abs_vwap_changes_lag4', 'abs_vwap_changes_lag5_8',
    'alpha_lag1', 'alpha_lag2', 'f_TTD',
    'MON', 'SAT', 'SUN', 'mo_slope_500', 'mo_slope_1000', 'mo_slope_2000',
    'wind_forecast', 'solar_forecast', 'da-id_lag1'
]
# Neighbor columns we will create
NB_COLS = { -2: 'nb_dchg_m2', -1: 'nb_dchg_m1', +1: 'nb_dchg_p1', +2: 'nb_dchg_p2' }  # <<< CHANGED
col_neighbors = list(NB_COLS.values())
col_gamlss = col_gamlss_base + col_neighbors

# -----------------------
# Neighbor creation via self-join on (bin_timestamp, delivery_start_wall)
# -----------------------
def add_neighbor_vwapchg_lag1(df: pd.DataFrame, default_val: float = 0.0) -> pd.DataFrame:
    """
    For each row, add neighbors' vwap_changes_lag1 from rows with the same bin_timestamp
    and delivery_start_wall shifted by -2h, -1h, +1h, +2h. If the neighbor row doesn't
    exist (e.g., previous product expired) or has non-finite values, fill with 0.0.
    """
    out = df.copy()

    # Ensure datetimes
    if not pd.api.types.is_datetime64_any_dtype(out['bin_timestamp']):
        out['bin_timestamp'] = pd.to_datetime(out['bin_timestamp'])
    if not pd.api.types.is_datetime64_any_dtype(out['delivery_start_wall']):
        out['delivery_start_wall'] = pd.to_datetime(out['delivery_start_wall'])

    # Unique lookup table (one row per (bin_timestamp, delivery_start_wall))
    base = out[['bin_timestamp', 'delivery_start_wall', 'vwap_changes_lag1']].copy()
    base = (base
            .dropna(subset=['bin_timestamp', 'delivery_start_wall'])
            .sort_values(['bin_timestamp', 'delivery_start_wall'])
            .drop_duplicates(['bin_timestamp', 'delivery_start_wall'], keep='last'))

    # Clean non-finite in source before merging
    base['vwap_changes_lag1'] = pd.to_numeric(base['vwap_changes_lag1'], errors='coerce')
    base['vwap_changes_lag1'] = base['vwap_changes_lag1'].where(np.isfinite(base['vwap_changes_lag1']), np.nan)

    for offset, colname in NB_COLS.items():
        right = base.copy()
        # A neighbor at delivery d' should provide values to rows at delivery d = d' - offset
        right['delivery_start_wall'] = right['delivery_start_wall'] - pd.Timedelta(hours=offset)
        right = right.rename(columns={'vwap_changes_lag1': colname})

        out = out.merge(
            right,
            on=['bin_timestamp', 'delivery_start_wall'],
            how='left'
        )

        # Fill missing or non-finite with default (0.0)
        out[colname] = pd.to_numeric(out[colname], errors='coerce')
        out[colname] = out[colname].where(np.isfinite(out[colname]), np.nan)
        out[colname] = out[colname].fillna(default_val).astype(float)

    return out  # <<< CHANGED (return was missing in your paste)

# -----------------------
# Utilities
# -----------------------
def ensure_columns(df: pd.DataFrame, cols: list):
    miss = [c for c in cols if c not in df.columns]
    for c in miss:
        df[c] = 0.0
    return df

def design_matrix(df: pd.DataFrame, cols: list) -> np.ndarray:
    df = ensure_columns(df, cols)
    return (df[cols]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .to_numpy(dtype=float, copy=False))

# -----------------------
# Core: train on window and make heatmaps for TARGET_DAY
# -----------------------
def make_heatmaps_for_date(target_day: str, window_days: int):
    target = pd.Timestamp(target_day, tz=LOCAL_TZ).floor('D')
    window_start = target - pd.Timedelta(days=window_days)

    # 1) Select training rows strictly before target day
    train_all = df_all[(df_all['day'] >= window_start) & (df_all['day'] < target)].copy()

    # 2) <<< CHANGED: keep ONLY last-4h rows (causality constraint)
    train_all = train_all[train_all['in_last4h']].copy()

    # 3) <<< CHANGED: add neighbor features via self-join on (bin_timestamp, delivery_start_wall)
    train_all = add_neighbor_vwapchg_lag1(train_all)

    # 4) α = 1 for the JSU GAMLSS training
    train_P = train_all[train_all['alpha'] == 1].copy()
    if train_P.empty:
        raise RuntimeError("No training rows found for the chosen window/day with α=1 within last-4h.")

    feature_names = list(col_gamlss)
    n_features = len(feature_names)

    # init coefficient tables (rows=features, cols=hours)
    df_mu    = pd.DataFrame(index=feature_names, columns=HOURS, data=0.0)
    df_sigma = df_mu.copy()
    df_nu    = df_mu.copy()
    df_tau   = df_mu.copy()

    for hour in HOURS:
        df_h = train_P[train_P['hour'] == hour]
        if df_h.empty:
            continue

        X = design_matrix(df_h, feature_names)
        y = df_h['vwap_changes'].to_numpy(dtype=float, copy=False)
        m = np.isfinite(y)
        if m.sum() < max(8, n_features + 1):  # sanity guard
            continue

        model = GAMLSS(distribution=JSU())
        model.fit(X[m], y[m])

        # SIMPLE: take coefficients directly (drop intercept)
        try:
            mu_coefs    = np.asarray(model.betas[0], dtype=float)[1:]
            sigma_coefs = np.asarray(model.betas[1], dtype=float)[1:]
            nu_coefs    = np.asarray(model.betas[2], dtype=float)[1:]
            tau_coefs   = np.asarray(model.betas[3], dtype=float)[1:]
        except Exception as e:
            raise RuntimeError(f"Unexpected model.betas structure for hour={hour}: {type(getattr(model,'betas',None))}") from e

        # pad/truncate just in case (defensive)
        def fit_len(v):
            v = np.asarray(v, dtype=float)
            if v.size < n_features:
                out = np.zeros(n_features, dtype=float); out[:v.size] = v; return out
            return v[:n_features]

        df_mu[hour]    = fit_len(mu_coefs)
        df_sigma[hour] = fit_len(sigma_coefs)
        df_nu[hour]    = fit_len(nu_coefs)
        df_tau[hour]   = fit_len(tau_coefs)

    # ---- Plot heatmaps (single day’s coefficients) ----
    cmap = mcolors.LinearSegmentedColormap.from_list("coef_map", ["red","green"])

    def plot_heatmap(df_coef, title):
        max_val = np.nanmax(np.abs(df_coef.values)) if df_coef.size else 0.0
        df_scaled = (df_coef.abs() / max_val).fillna(0.0) if max_val > 0 else df_coef.abs().fillna(0.0)

        plt.figure(figsize=(22, 18))
        ax = sns.heatmap(
            df_scaled, cmap=cmap, vmin=0, vmax=1,
            annot=df_coef.round(3), fmt="", cbar=True
        )
        ax.set_title(title)
        ax.set_xlabel("Hour of day")
        ax.set_ylabel("Feature")
        plt.tight_layout()
        plt.show()

    day_label = pd.Timestamp(target_day).date()
    plot_heatmap(df_mu,    f"μ coefficients (|color| normalized) — {day_label}, window={window_days}d")
    plot_heatmap(df_sigma, f"σ coefficients (|color| normalized) — {day_label}, window={window_days}d")
    plot_heatmap(df_nu,    f"ν coefficients (|color| normalized) — {day_label}, window={window_days}d")
    plot_heatmap(df_tau,   f"τ coefficients (|color| normalized) — {day_label}, window={window_days}d")

# -----------------------
# Run
# -----------------------
if __name__ == "__main__":
    make_heatmaps_for_date(TARGET_DAY, WINDOW_DAYS)
