# mdn_intraday_price_multi_hour.py  ➜ adds VWAP lag features
"""
*Adds richer price-history features*:

1. **Raw VWAP lags 1–3**              → `vwap_lag1, vwap_lag2, vwap_lag3`
2. **Absolute VWAP lags 1–6**          → `abs_vwap_lag1` … `abs_vwap_lag6`
3. **Mean abs‑VWAP of lags 7–12**      → `abs_vwap_lag7_12_mean`

Total new columns = 3 + 6 + 1 = **10**.  With a first‑layer width of 64,
that is **10 × 64 = 640 extra weights** – negligible compared to the ~20 k
parameters already in a 2‑layer MDN.

Run grid search as before:
```bash
python mdn_intraday_price_multi_hour.py intraday.csv --grid --folds 3
```
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from scikeras.wrappers import KerasRegressor

import tensorflow as tf
import tensorflow_probability as tfp

tfpl = tfp.layers
np.random.seed(42)
tf.random.set_seed(42)


# Data loading + advanced lag features


def load_dataframe(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=[0]).sort_index()

    # ------- calendar & τ features ----------------------------------------
    df["weekend"] = (df.index.dayofweek >= 5).astype(int)
    tgt_start = df.index + pd.Timedelta(minutes=5)
    tau = (tgt_start - df.index).total_seconds() / 60.0
    df["log_tau"] = np.log1p(tau)
    df["inv_sqrt_tau"] = 1.0 / np.sqrt(tau + 1.0)

    # ------- VWAP lag features --------------------------------------------
    for k in range(1, 4):
        df[f"vwap_lag{k}"] = df["VWAP"].shift(k)
    for k in range(1, 7):
        df[f"abs_vwap_lag{k}"] = df["VWAP"].shift(k).abs()
    # mean abs(VWAP) over lags 7–12
    abs_lags_7_12 = [df["VWAP"].shift(k).abs() for k in range(7, 13)]
    df["abs_vwap_lag7_12_mean"] = np.nanmean(abs_lags_7_12, axis=0)

    return df


# Feature list 

BASE_COLS = [
    "solar_DA", "wind_DA", "mos",
    "VWAP", "trade",
    "weekend", "log_tau", "inv_sqrt_tau",
]
LAG_COLS = (
    [f"vwap_lag{k}" for k in range(1, 4)] +
    [f"abs_vwap_lag{k}" for k in range(1, 7)] +
    ["abs_vwap_lag7_12_mean"]
)
FEATURE_COLS = BASE_COLS + LAG_COLS


# MDN 

def build_mdn_model(K=3, hidden=(64, 64), l1_lambda=1e-4, lr=1e-3):
    n_features = len(FEATURE_COLS)
    comp_size  = tfpl.IndependentNormal.params_size([1])
    param_size = tfpl.MixtureSameFamily.params_size(K, comp_size)

    x_in = tf.keras.Input(shape=(n_features,))
    h = x_in
    for i, units in enumerate(hidden):
        reg = tf.keras.regularizers.l1(l1_lambda) if i == 0 else None
        h   = tf.keras.layers.Dense(units, activation="elu", kernel_regularizer=reg)(h)
    mdn_params = tf.keras.layers.Dense(param_size)(h)
    gmm = tfpl.MixtureSameFamily(K, tfpl.IndependentNormal([1]))(mdn_params)

    model = tf.keras.Model(x_in, gmm)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                  loss=lambda y, rv: -rv.log_prob(y))
    return model


# 4.  Grid‑CV


def run_grid_cv(path: str | Path, n_folds: int = 3):
    df = load_dataframe(path)
    y = df["VWAP"].shift(-1)
    X = df[FEATURE_COLS]
    mask = (~y.isna()) & (~X.isna().any(axis=1))
    y, X = y[mask].to_numpy().reshape(-1, 1), X[mask].to_numpy().astype("float32")

    xs, ys = StandardScaler().fit(X), StandardScaler().fit(y)
    Xz, yz = xs.transform(X), ys.transform(y)

    est = KerasRegressor(model=build_mdn_model,
                         epochs=50, batch_size=512, verbose=0)

    param_grid = {
        "K": [2, 3, 4],
        "hidden": [(64,), (64, 64)],
        "l1_lambda": [0.0, 1e-6, 1e-5, 1e-4, 1e-3],
        "lr": [1e-3, 3e-4],
    }

    tscv = TimeSeriesSplit(n_splits=n_folds)
    gcv = GridSearchCV(est, param_grid, cv=tscv, scoring="neg_log_loss", n_jobs=-1, refit=True)
    gcv.fit(Xz, yz)
    print("Best params:", gcv.best_params_)
    print("Mean CV NLL:", -gcv.best_score_)


