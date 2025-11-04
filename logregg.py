import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors

from sklearn.linear_model import LogisticRegression

# -----------------------------------------------------------------------------
# 1) EXAMPLE DATA PREPARATION
# -----------------------------------------------------------------------------

# df_2021: your training set for the entire year 2021
# df_2022: your data for the year 2022

df_2021 = pd.read_csv("df_2021.csv")
df_2022 = pd.read_csv("df_2022.csv")

df_2021['delivery_start'] = pd.to_datetime(df_2021['delivery_start'], utc=True).dt.tz_convert("Europe/Berlin")
df_2022['delivery_start'] = pd.to_datetime(df_2022['delivery_start'], utc=True).dt.tz_convert("Europe/Berlin")


# -----------------------------------------------------------------------------
# 2) SELECT FEATURES
# -----------------------------------------------------------------------------
# Define your predictive features (COLUMNS_X).
#  - Remove "f_TTD" from the old feature list.
#  - Add alpha_lag3..alpha_lag12
#  - Add TTD_bin_1..TTD_bin_36

# Base features (like your original) but with "f_TTD" removed:
base_feats = [
    'vwap_changes_lag1',
    'vwap_changes_lag2',
    'vwap_changes_lag3',
    'abs_vwap_changes_lag1',
    'abs_vwap_changes_lag2',
    'abs_vwap_changes_lag3',
    'abs_vwap_changes_lag4',
    'abs_vwap_changes_lag5',
    'abs_vwap_changes_lag6',
    'abs_vwap_changes_lag7_12',
    'alpha_lag1',
    'alpha_lag2',
    'MON',
    'SAT',
    'SUN',
    'mo_slope_500',
    'mo_slope_1000',
    'mo_slope_2000',
    'wind_forecast',
    'solar_forecast',
    'da-id'
]

# alpha_lag3..alpha_lag12
alpha_lag_feats = [f"alpha_lag{i}" for i in range(3, 13)]

# TTD_bin_1..TTD_bin_36
ttd_bin_feats = [f"TTD_bin_{i}" for i in range(1, 37)]

COLUMNS_X = base_feats + alpha_lag_feats + ttd_bin_feats

# -----------------------------------------------------------------------------
# 3) FILTER TRAINING DATA (SAME TIME PERIODS FROM BOTH 2021, 2022)
# -----------------------------------------------------------------------------

start_date = pd.to_datetime("2021-10-05").tz_localize("Europe/Berlin")
end_date   = pd.to_datetime("2022-10-04").tz_localize("Europe/Berlin")

mask_2021 = (
    (df_2021['delivery_start'] >= start_date) &
    (df_2021['delivery_start'] <= end_date)
)
mask_2022 = (
    (df_2022['delivery_start'] >= start_date) &
    (df_2022['delivery_start'] <= end_date)
)

# If you still need the "abs_vwap_changes_lag7_12" column as before:
df_2021['abs_vwap_changes_lag7_12'] = (
    df_2021['abs_vwap_changes_lag7']  + df_2021['abs_vwap_changes_lag8']  +
    df_2021['abs_vwap_changes_lag9']  + df_2021['abs_vwap_changes_lag10'] +
    df_2021['abs_vwap_changes_lag11'] + df_2021['abs_vwap_changes_lag12']
)

df_2022['abs_vwap_changes_lag7_12'] = (
    df_2022['abs_vwap_changes_lag7']  + df_2022['abs_vwap_changes_lag8']  +
    df_2022['abs_vwap_changes_lag9']  + df_2022['abs_vwap_changes_lag10'] +
    df_2022['abs_vwap_changes_lag11'] + df_2022['abs_vwap_changes_lag12']
)

df_train_2021 = df_2021[mask_2021].copy()
df_train_2022 = df_2022[mask_2022].copy()

df_train = pd.concat([df_train_2021, df_train_2022], ignore_index=True)
df_train.dropna(subset=COLUMNS_X + ['alpha'], inplace=True)

# -----------------------------------------------------------------------------
# 4) CREATE HOUR-BY-HOUR (X, y) DICTIONARY
#    - The response vector y is now "alpha" (instead of "vwap_changes").
# -----------------------------------------------------------------------------

Xy_dict = {}
for hour in range(24):
    if hour == 2:
        # skip hour=2 if that's your requirement
        continue
    # Filter for that hour
    df_h = df_train[df_train['delivery_start'].dt.hour == hour].copy()
    
    # y is alpha
    y = df_h['alpha'].values  # 0/1 array
    
    # X is everything in COLUMNS_X
    X = df_h[COLUMNS_X].values
    
    Xy_dict[hour] = (X, y)

# -----------------------------------------------------------------------------
# 5) FIT A LOGISTIC REGRESSION PER HOUR & COLLECT COEFFICIENTS
# -----------------------------------------------------------------------------

fitted_models_logreg = {}
hours = [h for h in range(24) if h != 2]

# Prepare a DataFrame to store coefficients (rows = features, cols = hour)
df_coefs = pd.DataFrame(index=COLUMNS_X, columns=hours, dtype=float)

for h in range(24):
    if h == 2:
        continue
    
    X, y = Xy_dict[h]
    
    # You can tweak parameters or try regularization as needed:
    model = LogisticRegression(
        penalty='l1',  # or 'l2', etc.
        solver= 'liblinear',  # or your preferred solver
        max_iter=1000
    )
    model.fit(X, y)
    
    # model.coef_ has shape (1, n_features) because this is a binary classification
    coef_array = model.coef_[0]
    
    # Store coefficients in df_coefs
    df_coefs.loc[:, h] = coef_array
    
    # Save the model
    fitted_models_logreg[h] = model

