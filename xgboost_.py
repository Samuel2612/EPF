import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import pandas as pd
import numpy as np
from xgboost import XGBClassifier, XGBRegressor
import shap
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# -----------------------------------------------------------------------------
# Data loading & preprocessing
# -----------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent

df_2021 = pd.read_csv(ROOT / "df_2021.csv")
df_2022 = pd.read_csv(ROOT / "df_2022.csv")

for df in (df_2021, df_2022):
    df["delivery_start"] = pd.to_datetime(df["delivery_start"], utc=True).dt.tz_convert("Europe/Berlin")

df_all = pd.concat([df_2021, df_2022], ignore_index=True)
df_all["day"] = df_all["delivery_start"].dt.floor("D")

# Aggregate 7‑12 lagged absolute changes into a single feature
vwap_lags = [f"abs_vwap_changes_lag{i}" for i in range(7, 13)]
df_all["abs_vwap_changes_lag7_12"] = df_all[vwap_lags].sum(axis=1)

# Only positive‑alpha rows for the delta model
df_P = df_all[df_all["alpha"] == 1].copy()

# -----------------------------------------------------------------------------
# Feature sets
# -----------------------------------------------------------------------------

col_reg = [
    "vwap_changes_lag1", "vwap_changes_lag2", "vwap_changes_lag3",
    "abs_vwap_changes_lag1", "abs_vwap_changes_lag2", "abs_vwap_changes_lag3",
    "abs_vwap_changes_lag4", "abs_vwap_changes_lag5", "abs_vwap_changes_lag6",
    "abs_vwap_changes_lag7_12", "alpha_lag1", "alpha_lag2", "f_TTD",
    "MON", "SAT", "SUN", "mo_slope_500", "mo_slope_1000", "mo_slope_2000",
    "wind_forecast", "solar_forecast", "da-id",
]

base_alpha = [
    "vwap_changes_lag1", "vwap_changes_lag2", "vwap_changes_lag3",
    "abs_vwap_changes_lag1", "abs_vwap_changes_lag2", "abs_vwap_changes_lag3",
    "abs_vwap_changes_lag4", "abs_vwap_changes_lag5", "abs_vwap_changes_lag6",
    "abs_vwap_changes_lag7_12",
] + [f"alpha_lag{i}" for i in range(1, 13)] + [
    "MON", "SAT", "SUN", "mo_slope_500", "mo_slope_1000", "mo_slope_2000",
    "wind_forecast", "solar_forecast", "da-id",
]

# Hour‑ahead distance encoded as one‑hot bins (1 … 36)
ttd_bin_feats = [f"TTD_bin_{i}" for i in range(1, 37)]
col_cls = base_alpha + ttd_bin_feats

# -----------------------------------------------------------------------------
# Helper: update_features (unchanged apart from typing hints)
# -----------------------------------------------------------------------------

def update_features(feats: pd.Series, vwap: float, vwap_change: float,
                    alpha: int, new_TTD: int) -> pd.Series:
    """Shift lagged features by one step and inject the newest realised values."""

    new_feats = feats.copy()
    new_feats["vwap_lag1"] = vwap

    # Shift VWAP change lags
    new_feats[[f"vwap_changes_lag{i}" for i in (3, 2)]] = feats[["vwap_changes_lag2", "vwap_changes_lag1"]]
    new_feats["vwap_changes_lag1"] = vwap_change

    # Shift |ΔVWAP| lags 12→2
    for i in range(12, 1, -1):
        new_feats[f"abs_vwap_changes_lag{i}"] = feats[f"abs_vwap_changes_lag{i-1}"]
    new_feats["abs_vwap_changes_lag1"] = abs(vwap_change)

    # Shift alpha lags 12→2
    for j in range(12, 1, -1):
        new_feats[f"alpha_lag{j}"] = feats[f"alpha_lag{j-1}"]
    new_feats["alpha_lag1"] = feats["alpha"]

    # Recompute time‑to‑delivery encodings
    new_feats["f_TTD"] = 1.0 / np.sqrt(1.0 + new_TTD)
    for i in range(1, 37):
        new_feats[f"TTD_bin_{i}"] = int(new_TTD == i)

    # day‑ahead ID spread
    new_feats["da-id"] = abs(new_feats["da_price"] - new_feats["vwap_lag1"])

    return new_feats



gamlss_results = {}      # now XGBRegressor
logreg_results = {}      # now XGBClassifier
shap_reg_results = {}
shap_cls_results = {}


min_day, max_day = df_P["day"].min(), df_P["day"].max()
first_possible_day = min_day + pd.Timedelta(days=365)
all_days = pd.date_range(first_possible_day, max_day, freq="D")
HOURS = [h for h in range(24) if h != 2]

for current_day in all_days:
    print(f"\n=== {current_day.date()} ===")

    # Rolling 1‑year window for training
    win_start = current_day - pd.Timedelta(days=365)
    train_mask = (df_all["day"] >= win_start) & (df_all["day"] < current_day)
    cls_train = df_all[train_mask].copy()
    reg_train = df_P[train_mask].copy()


    day_cls_models, day_reg_models = {}, {}
    day_cls_shap,   day_reg_shap   = {}, {}

    for hr in HOURS:
        # --- Regressor (ΔVWAP | alpha=1) --------------------------------------
        reg_hr = reg_train[reg_train["delivery_start"].dt.hour == hr]
        X_reg, y_reg = reg_hr[col_reg].values, reg_hr["vwap_changes"].values
        model_reg = XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            n_jobs=-1,
        )
        model_reg.fit(X_reg, y_reg)
        day_reg_models[hr] = model_reg

        # SHAP – mean|value| per feature
        expl_reg = shap.TreeExplainer(model_reg)
        shap_vals_reg = expl_reg.shap_values(X_reg)
        day_reg_shap[hr] = np.abs(shap_vals_reg).mean(axis=0)

        # --- Classifier (alpha) ---------------------------------------------
        cls_hr = cls_train[cls_train["delivery_start"].dt.hour == hr]
        X_cls, y_cls = cls_hr[col_cls].values, cls_hr["alpha"].values
        model_cls = XGBClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=-1,
        )
        model_cls.fit(X_cls, y_cls)
        day_cls_models[hr] = model_cls

        expl_cls = shap.TreeExplainer(model_cls)
        shap_vals_cls = expl_cls.shap_values(X_cls)[1]  # positive class
        day_cls_shap[hr] = np.abs(shap_vals_cls).mean(axis=0)

    # Store per‑day results
    gamlss_results[current_day] = day_reg_models
    logreg_results[current_day] = day_cls_models
    shap_reg_results[current_day] = day_reg_shap
    shap_cls_results[current_day] = day_cls_shap



def aggregate_shap(shap_dict, feature_names):
    """Return a pandas Series of mean absolute SHAP values across all models."""
    accum = np.zeros(len(feature_names))
    count = 0
    for _day, hour_dict in shap_dict.items():
        for _hr, vals in hour_dict.items():
            accum += vals
            count += 1
    return pd.Series(accum / count, index=feature_names).sort_values(ascending=False)

shap_alpha_global = aggregate_shap(shap_cls_results, col_cls)
shap_delta_global = aggregate_shap(shap_reg_results, col_reg)

print("\nTop 10 features – alpha probability (XGBClassifier):")
print(shap_alpha_global.head(10))

print("\nTop 10 features – ΔVWAP magnitude (XGBRegressor):")
print(shap_delta_global.head(10))

# The rest of your pipeline (simulation, backtesting, etc.) can continue to use
# `logreg_results` and `gamlss_results` (now XGBoost models) exactly as before.
