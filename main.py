import pandas as pd
import numpy as np
from GAMLSS_x import GAMLSS
from distributions import JSUo, JSU
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import scipy.stats as st
from datetime import timedelta
from tqdm import tqdm
from pathlib import Path
import os
from zoneinfo import ZoneInfo
import xgboost as xgb


np.random.seed(seed=1)

## inputs ###
sample_by_years = [2021, 2022]
calibration_sample_size= timedelta(days=365)
forward_rolling_step = timedelta(days=1)
use_features_in_the_paper_only = True
turn_on_simulation = False
output_regressors = True
output_model_results = True


local_time_zone = ZoneInfo("Europe/Amsterdam")
data_dir = Path(os.path.abspath("./data"))

result_dir = Path(os.path.abspath("./result"))
if not os.path.exists(result_dir):
    os.makedirs(result_dir)


temp_dir = Path(os.path.abspath("../temp"))
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)



df_all = pd.DataFrame()
for yr in sample_by_years:
    if use_features_in_the_paper_only:
        df_yr = pd.read_csv(data_dir/f"df_{yr}.csv")
    else:
        df_yr = pd.read_csv(data_dir / f"df_{yr}-extended.csv")
    df_yr['delivery_start'] = pd.to_datetime(df_yr['delivery_start'], utc=True).dt.tz_convert(local_time_zone)
    df_all = pd.concat([df_all, df_yr], ignore_index=True)


if 'timestamp' not in df_all.columns:
    df_all['timestamp'] = df_all['bin_timestamp']

df_all['timestamp'] = pd.to_datetime(df_all['timestamp'])

df_all['day'] = df_all['delivery_start'].dt.date

df_all['abs_vwap_changes_lag7_12'] = (
        df_all['abs_vwap_changes_lag7'] + df_all['abs_vwap_changes_lag8'] +
        df_all['abs_vwap_changes_lag9'] + df_all['abs_vwap_changes_lag10'] +
        df_all['abs_vwap_changes_lag11'] + df_all['abs_vwap_changes_lag12']
)

df_P = df_all[df_all['alpha'] == 1].copy()


# min_ts = df_P['timestamp'].min()
# max_ts = df_P['timestamp'].max()

min_day = df_P['day'].min()
max_day = df_P['day'].max()



fitting_sample_start, fitting_sample_end = (min_day + calibration_sample_size,  max_day)


all_fitting_ts = pd.date_range(start=fitting_sample_start, end=fitting_sample_end, freq=pd.to_timedelta(forward_rolling_step))



col_gamlss = [
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
    'f_TTD',
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

base_alpha = [
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
    'alpha_lag3',
    'alpha_lag4',
    'alpha_lag5',
    'alpha_lag6',
    'alpha_lag7',
    'alpha_lag8',
    'alpha_lag9',
    'alpha_lag10',
    'alpha_lag11',
    'alpha_lag12',
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

# TTD_bin_1..TTD_bin_36
ttd_bin_feats = [f"TTD_bin_{i}" for i in range(1, 37)]
col_logreg = base_alpha + ttd_bin_feats




gamlss_results = {}
logreg_results = {}

HOURS = [h for h in range(24) if h != 2]

distribution = JSU

for current_ts in tqdm(all_fitting_ts[:1]):

    current_ts = pd.to_datetime(current_ts)

    #  construct training and testing datasets
    calibration_window_start, calibration_window_end = (current_ts -  calibration_sample_size, current_ts)
    testing_window_start, testing_window_end  = (current_ts, current_ts + forward_rolling_step)

    #mask_train = (df_all['timestamp'] >= calibration_window_start) & (df_all['timestamp'] < calibration_window_end)
    
    mask_train = (df_all['day'] >= calibration_window_start.date()) & (df_all['day'] < calibration_window_end.date())
    mask_train_P = (df_P['day'] >= calibration_window_start.date()) & (df_P['day'] < calibration_window_end.date())
    
    df_train_logreg = df_all[mask_train].copy()
    df_train_gamlss = df_P[mask_train_P].copy()

    #mask_test = (df_all['timestamp'] >= testing_window_start) & (df_all['timestamp'] < testing_window_end)
    
    mask_test = (df_all['day'] >= testing_window_start.date()) & (df_all['day'] < testing_window_end.date())
    mask_test_P = (df_P['day'] >= testing_window_start.date()) & (df_P['day'] < testing_window_end.date())
    df_test_logreg = df_all[mask_test].copy()
    df_test_gamlss = df_P[mask_test_P].copy()

    # Fit each hour's model
    day_dict_gamlss = {}
    day_dict_logreg = {}


    for hour in HOURS:

        # Fit GAMLSS model
        df_h_gamlss = df_train_gamlss[df_train_gamlss['delivery_start'].dt.hour == hour].copy()
        train_X_gamlss = df_h_gamlss[col_gamlss].values
        train_Y_gamlss = df_h_gamlss['vwap_changes'].values


        if output_regressors:

            df_h_gamlss[col_gamlss].to_csv( temp_dir/
                f"X_gamlss_hour_{hour}_"
                f"calibration_start_{calibration_window_start.strftime('%Y%m%d-%H%M%S')}_"
                f"end_noninclusive_{calibration_window_end.strftime('%Y%m%d-%H%M%S')}.csv"
            )
            df_h_gamlss['vwap_changes'].to_csv( temp_dir/
                f"Y_gamlss_hour_{hour}_"
                f"calibration_start_{calibration_window_start.strftime('%Y%m%d-%H%M%S')}_"
                f"end_noninclusive_{calibration_window_end.strftime('%Y%m%d-%H%M%S')}.csv"
            )



        model_gamlss = GAMLSS(distribution=distribution())
        model_gamlss.fit(train_X_gamlss, train_Y_gamlss)
        day_dict_gamlss[hour] = model_gamlss

        print(f'val of loglikelihood:\n'
              f'{model_gamlss.fit_hist_ll}')

        # Fit logreg model

        df_h_logreg = df_train_logreg[df_train_logreg['delivery_start'].dt.hour == hour].copy()
        X_logreg = df_h_logreg[col_logreg].values
        y_logreg = df_h_logreg['alpha'].values

        if output_regressors:
            df_h_logreg[col_logreg].to_csv(temp_dir/
                f"X_logistic_reg_hour_{hour}_"
                f"calibration_start_{calibration_window_start.strftime('%Y%m%d-%H%M%S')}_"
                f"end_noninclusive_{calibration_window_end.strftime('%Y%m%d-%H%M%S')}.csv")

            df_h_logreg["alpha"].to_csv(temp_dir/
                f"Y_logistic_reg_hour_{hour}_"
                f"calibration_start_{calibration_window_start.strftime('%Y%m%d-%H%M%S')}_"
                f"end_noninclusive_{calibration_window_end.strftime('%Y%m%d-%H%M%S')}.csv")


        model_logreg = LogisticRegression(
            penalty='l1',
            solver='liblinear',
            max_iter=1000,
            random_state=1
        )
        model_logreg.fit(X_logreg, y_logreg)
        day_dict_logreg[hour] = model_logreg


        # fit XGBoost to compare against GAMLSS

        xgb_model = xgb.XGBRegressor(
            objective='reg:squarederror',  # For regression tasks
            n_estimators=100,  # Number of boosting rounds
            learning_rate=0.1,  # Step size shrinkage
            max_depth=3,  # Maximum tree depth
            subsample=0.8,  # Subsample ratio of training instances
            colsample_bytree=0.8,  # Subsample ratio of features
            random_state=42
        )

        xgb_model.fit(train_X_gamlss, train_Y_gamlss)



        if output_model_results:
            df_galmss_beta = pd.DataFrame.from_dict(model_gamlss.betas).T
            df_galmss_beta.columns = ['const'] + col_gamlss
            df_galmss_beta.to_csv(temp_dir /
                                    f"GAMLSS_beta_hour_{hour}_"
                                    f"calibration_start_{calibration_window_start.strftime('%Y%m%d-%H%M%S')}_"
                                    f"end_noninclusive_{calibration_window_end.strftime('%Y%m%d-%H%M%S')}.csv"
                                    )

            df_logistic_beta = pd.DataFrame(data = model_logreg.coef_)
            df_logistic_beta.columns = col_logreg

            df_logistic_beta.to_csv(temp_dir /
                                    f"Logistic_beta_hour_{hour}_"
                                    f"calibration_start_{calibration_window_start.strftime('%Y%m%d-%H%M%S')}_"
                                    f"end_noninclusive_{calibration_window_end.strftime('%Y%m%d-%H%M%S')}.csv"
                                    )



        ## metrics of fitting accuracy
        mu    = model_gamlss.theta[:, 0]
        sigma = model_gamlss.theta[:, 1]
        nu    = model_gamlss.theta[:, 2]
        tau   = model_gamlss.theta[:, 3]

        predicted_mean, predicted_std, predicted_skew, predicted_excess_kurtosis = distribution.moments(mu, sigma, nu, tau)
        rmsd_error = np.sqrt(np.mean((predicted_mean - train_Y_gamlss) ** 2))
        print(f" check - sum should be 0 : {np.sum(np.abs(predicted_mean - model_gamlss.predict(train_X_gamlss)[:, 0]))}")
        print(f"GAMLSS RSMD error of training data for hour {hour} calibration_start {calibration_window_start.strftime('%Y%m%d-%H%M%S')} : {rmsd_error}")


        xgb_fit = xgb_model.predict(train_X_gamlss)
        xgb_rmsd_error = np.sqrt(np.mean((xgb_fit - train_Y_gamlss) ** 2))
        print(
            f"XGBoost RSMD error of training data for hour {hour} calibration_start {calibration_window_start.strftime('%Y%m%d-%H%M%S')} : {xgb_rmsd_error}")

        df_accuracy = pd.DataFrame({"Y": train_Y_gamlss, "predicted_mean": predicted_mean, "predicted_sigma": predicted_std, "predicted_skew": predicted_skew, "predicted_excess_kurtosis": predicted_excess_kurtosis})

        suffix = "" if use_features_in_the_paper_only else "_extended_feature"
        df_accuracy.to_csv(result_dir /
            f"gamlss_fit_calibration_end_noninclusive_{calibration_window_end.date()}_hour_{hour}{suffix}.csv")


        ## metrics of the accuracy for the test set

        test_h_gamlss = df_test_gamlss[df_test_gamlss['delivery_start'].dt.hour == hour].copy()
        test_X_gamlss = test_h_gamlss[col_gamlss].values
        test_Y_gamlss = test_h_gamlss['vwap_changes'].values

        dist_params_test_sets = model_gamlss.predict(test_X_gamlss)

        mu    = dist_params_test_sets[:, 0]
        sigma = dist_params_test_sets[:, 1]
        nu    = dist_params_test_sets[:, 2]
        tau   = dist_params_test_sets[:, 3]


        predicted_mean, predicted_std, predicted_skew, predicted_excess_kurtosis = distribution.moments(mu, sigma, nu, tau)
        rmsd_error = np.sqrt(np.mean((predicted_mean - test_Y_gamlss) ** 2))
        print(f"GAMLSS RSMD error of test data for hour {hour} test date {testing_window_start.date()} : {rmsd_error}")

        xgb_pred = xgb_model.predict(test_X_gamlss)
        xgb_rmsd_error = np.sqrt(np.mean((xgb_pred - test_Y_gamlss) ** 2))
        print(
            f"XGBoost RSMD error of test data for hour {hour} calibration_start {calibration_window_start.strftime('%Y%m%d-%H%M%S')} : {xgb_rmsd_error}")

        df_accuracy = pd.DataFrame({"Y": test_Y_gamlss, "predicted_mean": predicted_mean,  "predicted_sigma": predicted_std, "predicted_skew": predicted_skew, "predicted_excess_kurtosis": predicted_excess_kurtosis,
                                    "XGBoost_pred": xgb_pred})

        suffix = "" if use_features_in_the_paper_only else "_extended_feature"
        df_accuracy.to_csv( result_dir /
            f"gamlss_test_date_{testing_window_start.date()}_hour_{hour}{suffix}.csv")





    # Store the set of hourly models for this day
    gamlss_results[current_ts] = day_dict_gamlss
    logreg_results[current_ts] = day_dict_logreg



print("Finish")
