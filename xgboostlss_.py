import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from xgboostlss.model import *
from xgboostlss.distributions.StudentT import *
from xgboost import XGBClassifier  # unchanged for alpha
import shap


df_2021 = pd.read_csv("df_2021.csv")
df_2022 = pd.read_csv("df_2022.csv")

for df in (df_2021, df_2022):
    df["delivery_start"] = pd.to_datetime(df["delivery_start"], utc=True).dt.tz_convert("Europe/Berlin")

df_all = pd.concat([df_2021, df_2022], ignore_index=True)
df_all["day"] = df_all["delivery_start"].dt.floor("D")

vwap_lags = [f"abs_vwap_changes_lag{i}" for i in range(7, 13)]
df_all["abs_vwap_changes_lag7_12"] = df_all[vwap_lags].sum(axis=1)

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

ttd_bin_feats = [f"TTD_bin_{i}" for i in range(1, 37)]
col_cls = base_alpha + ttd_bin_feats

# -----------------------------------------------------------------------------
# Helper to shift features (unchanged)
# -----------------------------------------------------------------------------

def update_features(feats: pd.Series, vwap: float, vwap_change: float,
                    alpha: int, new_TTD: int) -> pd.Series:
    new_feats = feats.copy()
    new_feats["vwap_lag1"] = vwap
    new_feats[[f"vwap_changes_lag{i}" for i in (3, 2)]] = feats[["vwap_changes_lag2", "vwap_changes_lag1"]]
    new_feats["vwap_changes_lag1"] = vwap_change
    for i in range(12, 1, -1):
        new_feats[f"abs_vwap_changes_lag{i}"] = feats[f"abs_vwap_changes_lag{i-1}"]
    new_feats["abs_vwap_changes_lag1"] = abs(vwap_change)
    for j in range(12, 1, -1):
        new_feats[f"alpha_lag{j}"] = feats[f"alpha_lag{j-1}"]
    new_feats["alpha_lag1"] = feats["alpha"]
    new_feats["f_TTD"] = 1 / np.sqrt(1 + new_TTD)
    for i in range(1, 37):
        new_feats[f"TTD_bin_{i}"] = int(new_TTD == i)
    new_feats["da-id"] = abs(new_feats["da_price"] - new_feats["vwap_lag1"])
    return new_feats

# -----------------------------------------------------------------------------
# Containers for models + SHAP
# -----------------------------------------------------------------------------

gamlss_results = {}   # now XGBoostLSS
logreg_results = {}
shap_reg_results = {}
shap_cls_results = {}

# -----------------------------------------------------------------------------
# Main training loop
# -----------------------------------------------------------------------------

min_day, max_day = df_P["day"].min(), df_P["day"].max()
first_possible_day = min_day + pd.Timedelta(days=365)
all_days = pd.date_range(first_possible_day, max_day, freq="D")

HOURS = [h for h in range(24) if h != 2]

for current_day in all_days:
    print(f"\n=== {current_day.date()} ===")
    win_start = current_day - pd.Timedelta(days=365)
    cls_train = df_all[(df_all["day"] >= win_start) & (df_all["day"] < current_day)]
    reg_train = df_P[(df_P["day"] >= win_start) & (df_P["day"] < current_day)]

    day_cls_models, day_reg_models = {}, {}
    day_cls_shap,   day_reg_shap   = {}, {}

    for hr in HOURS:
        # ----- ΔVWAP distribution model --------------------------------------
        r_hr = reg_train[reg_train["delivery_start"].dt.hour == hr]
        Xr, yr = r_hr[col_reg].values, r_hr["vwap_changes"].values
        model_reg = XGBoostLSS(
            distribution="StudentT",
            n_estimators=400,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
        )
        model_reg.fit(Xr, yr)
        day_reg_models[hr] = model_reg


        expl_mu = shap.TreeExplainer(model_reg.models_["mu"])
        shap_vals_mu = expl_mu.shap_values(Xr)
        day_reg_shap[hr] = np.abs(shap_vals_mu).mean(axis=0)

       
        c_hr = cls_train[cls_train["delivery_start"].dt.hour == hr]
        Xc, yc = c_hr[col_cls].values, c_hr["alpha"].values
        model_cls = XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=-1,
        )
        model_cls.fit(Xc, yc)
        day_cls_models[hr] = model_cls

        expl_cls = shap.TreeExplainer(model_cls)
        shap_vals_cls = expl_cls.shap_values(Xc)[1]
        day_cls_shap[hr] = np.abs(shap_vals_cls).mean(axis=0)

    gamlss_results[current_day] = day_reg_models
    logreg_results[current_day] = day_cls_models
    shap_reg_results[current_day] = day_reg_shap
    shap_cls_results[current_day] = day_cls_shap

# -----------------------------------------------------------------------------
# Aggregate SHAP helpers (unchanged)
# -----------------------------------------------------------------------------

def aggregate_shap(shap_dict, feature_names):
    acc = np.zeros(len(feature_names))
    n = 0
    for d in shap_dict.values():
        for v in d.values():
            acc += v
            n += 1
    return pd.Series(acc / n, index=feature_names).sort_values(ascending=False)

shap_alpha_global = aggregate_shap(shap_cls_results, col_cls)
shap_delta_global = aggregate_shap(shap_reg_results, col_reg)

print("\nTop 10 features – alpha (XGBClassifier):\n", shap_alpha_global.head(10))
print("\nTop 10 features – ΔVWAP parameters μ (XGBoostLSS):\n", shap_delta_global.head(10))

# -----------------------------------------------------------------------------
# Simulation utilities (adapt sampling from JSUo distribution)
# -----------------------------------------------------------------------------

def simulate_intraday_paths(df_h_sim: pd.Series, model_cls, model_reg,
                            T: int = 36, n_sims: int = 1000):
    """Return array (n_sims, T) of simulated price paths from XGBoostLSS JSUo."""
    sim_prices = np.zeros((n_sims, T))
    start_price = df_h_sim["vwap_lag1"]

    for i in range(n_sims):
        feats = df_h_sim.copy()
        price_hist = []
        delta_t = 0.0
        p_new = start_price

        for ttd in range(T, 0, -1):
            Xc = feats[col_cls].values.reshape(1, -1)
            Xr = feats[col_reg].values.reshape(1, -1)

            p_alpha = model_reg.predict_proba(Xc)[0, 1]
            alpha_t = int(np.random.rand() < p_alpha)

            if alpha_t:
                dist = model_cls.predict(Xr, output="dist")[0]
                delta_t = dist.rvs()
            else:
                delta_t = 0.0

            p_new = p_new + delta_t
            price_hist.append(p_new)
            feats = update_features(feats, p_new, delta_t, alpha_t, ttd - 1)

        sim_prices[i, :] = price_hist
    return sim_prices
