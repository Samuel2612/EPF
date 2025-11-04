import pandas as pd
import numpy as np
import rolch


import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors




df_2021 = pd.read_csv("df_2021.csv")
df_2022 = pd.read_csv("df_2022.csv")

df_2021['delivery_start'] = pd.to_datetime(df_2021['delivery_start'], utc=True).dt.tz_convert("Europe/Berlin")
df_2022['delivery_start'] = pd.to_datetime(df_2022['delivery_start'], utc=True).dt.tz_convert("Europe/Berlin")





# We assume both df_2021 and df_2022 contain at least these columns:
COLUMNS_X = [
    'vwap_changes_lag1', 'vwap_changes_lag2', 'vwap_changes_lag3',
    'abs_vwap_changes_lag1', 'abs_vwap_changes_lag2',
    'abs_vwap_changes_lag3', 'abs_vwap_changes_lag4', 'abs_vwap_changes_lag5',
    'abs_vwap_changes_lag6', 'abs_vwap_changes_lag7_12',
    'alpha_lag1', 'alpha_lag2', 'f_TTD', 'MON', 'SAT', 'SUN',
    'mo_slope_500', 'mo_slope_1000', 'mo_slope_2000',
    'wind_forecast', 'solar_forecast',
    'da-id'   
]

start_date = pd.to_datetime("2021-10-05").tz_localize("Europe/Berlin")
end_date   = pd.to_datetime("2022-10-04").tz_localize("Europe/Berlin")

mask_2021 = (
    (df_2021['delivery_start'] >= start_date) &
    (df_2021['delivery_start'] <= end_date) &
    (df_2021['alpha'] == 1)
)
mask_2022 = (
    (df_2022['delivery_start'] >= start_date) &
    (df_2022['delivery_start'] <= end_date) &
    (df_2022['alpha'] == 1)
)


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

# df_train['delivery_start'] = pd.to_datetime(df_train['delivery_start'])

Xy_dict = {}
for hour in range(24):
    if hour == 2:
        continue  # skip hour=2
    # Filter to this particular hour
    df_h = df_train[df_train['delivery_start'].dt.hour == hour].copy()
    
    # The response vector y is "vwap_changes"
    y = df_h['vwap_changes'].values  # NumPy array
    
    # The observation matrix X is the columns you specified
    X = df_h[COLUMNS_X].values      # NumPy array
    
    Xy_dict[hour] = (X, y)



# We will store results in a dictionary: {hour: fitted_model, ...}
fitted_models_gamlss_2 = {}
results_list = []  # to build a table of coefficients

# For example, we'll choose a Johnson's SU distribution. 
# (You can also try distributions.SkewStudent4(), etc.)
dist = rolch.DistributionNormal()

equation = {
    0: "all",  # Can also use: "intercept" or pass a numpy array with indices / boolean
    1: "all",
    # 2: "all",
    # 3: 'all',
}

intercept = {
    0: False,
    1: True,
    # 2: True,
    # 3: True,
}

to_scale = np.array([True, True, True, True, True, True, True, True, True, True, False, False, True, False, False, False, True, True, True, True, True, True ])

hours = [h for h in range(24) if h != 2]

# Create empty DataFrames (rows=feature_names, columns=hours)
df_mu    = pd.DataFrame(index=COLUMNS_X, columns=hours, data=0.0)
df_sigma = pd.DataFrame(index=COLUMNS_X, columns=hours, data=0.0)
df_nu    = pd.DataFrame(index=COLUMNS_X, columns=hours, data=0.0)
df_tau   = pd.DataFrame(index=COLUMNS_X, columns=hours, data=0.0)

for h in range(5, 23):
    if h == 2:
        continue
    X, y = Xy_dict[h]   # your pre‐filtered data for this hour
    
    # Build and fit the GAMLSS.  We'll do LASSO, but you can also do 'ols' or 'ridge'.
    model = rolch.OnlineGamlss(
        distribution=dist,
        equation=equation,
        method='ols',
        scale_inputs=to_scale,   
        fit_intercept=intercept,  
        ic='bic',             # pick BIC for model selection
        verbose =1
    )
    model.fit(X, y)
    fitted_models_gamlss_2[h] = model
     
    mu_coefs    = model.beta[0]    # For mu; expected length equals len(COLUMNS_X) if no intercept.
    sigma_coefs = model.beta[1]    # For sigma; expected length = len(COLUMNS_X) + 1 (intercept included)
    nu_coefs    = model.beta[2]    # For nu; expected length = len(COLUMNS_X) + 1
    tau_coefs   = model.beta[3]    # For tau; expected length = len(COLUMNS_X) + 1

    # For mu (parameter 0) intercept is False, so coefficients map directly.
    for i, feat_name in enumerate(COLUMNS_X):
        df_mu.loc[feat_name, h] = mu_coefs[i]
    
    # For parameters 1,2,3, intercept is True, so coefficients start with intercept at index 0.
    for i, feat_name in enumerate(COLUMNS_X):
        df_sigma.loc[feat_name, h] = sigma_coefs[i+1]
        df_nu.loc[feat_name, h]    = nu_coefs[i+1]
        df_tau.loc[feat_name, h]   = tau_coefs[i+1]

# --------------------------------------------------------------
# 3) Create 4 color‐scaled tables, one for each parameter
#    We color by absolute magnitude: 0 => red, 1 => green
# --------------------------------------------------------------
# Build a simple linear colormap from red to green:
cmap = mcolors.LinearSegmentedColormap.from_list("coef_map", ["red","green"])

def plot_heatmap_of_coeffs(df_coef, param_name):
    # Normalize the absolute values from 0..1
    # so that the largest absolute coefficient => 1 => green
    max_val = df_coef.abs().values.max()
    if max_val < 1e-12:
        # avoid division by zero if all coefs are zero
        df_scaled = df_coef.abs()  # all zeros
    else:
        df_scaled = df_coef.abs() / max_val

    plt.figure(figsize=(20, 18))
    sns.heatmap(df_scaled, cmap=cmap, vmin=0, vmax=1, 
                annot=True, fmt=".3f", # if you want numeric labels
                cbar=True)
    plt.title(f"{param_name} coefficient magnitudes (hour vs feature)")
    plt.xlabel("Hour of day")
    plt.ylabel("Feature")
    plt.show()

plot_heatmap_of_coeffs(df_mu,    "mu")
plot_heatmap_of_coeffs(df_sigma, "sigma")
plot_heatmap_of_coeffs(df_nu,    "nu")
plot_heatmap_of_coeffs(df_tau,   "tau")
