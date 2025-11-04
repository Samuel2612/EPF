import pandas as pd
import numpy as np
from GAMLSS_x import GAMLSS
from distributions import JSUo, JSU, jsu_reparam_to_original
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import scipy.stats as st

df_2021 = pd.read_csv("df_2021.csv")
df_2022 = pd.read_csv("df_2022.csv")

df_2021['delivery_start'] = pd.to_datetime(df_2021['delivery_start'], utc=True).dt.tz_convert("Europe/Berlin")
df_2022['delivery_start'] = pd.to_datetime(df_2022['delivery_start'], utc=True).dt.tz_convert("Europe/Berlin")

df_all = pd.concat([df_2021, df_2022], ignore_index=True)
df_all['day'] = df_all['delivery_start'].dt.floor('D')
df_all['abs_vwap_changes_lag7_12'] = (
    df_all['abs_vwap_changes_lag7']  + df_all['abs_vwap_changes_lag8']  +
    df_all['abs_vwap_changes_lag9']  + df_all['abs_vwap_changes_lag10'] +
    df_all['abs_vwap_changes_lag11'] + df_all['abs_vwap_changes_lag12']
)




df_P = df_all[df_all['alpha'] == 1].copy()

min_day = df_P['day'].min()
max_day = df_P['day'].max()


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


# We need to start after at least 365 days from min_day
first_possible_day = min_day + pd.Timedelta(days=365)

# Build a daily range from first_possible_day to max_day inclusive
all_days = pd.date_range(start=first_possible_day, end=max_day, freq='D')


gamlss_results = {}
logreg_results = {}
sim_results = {}
err_results = {}

def update_features(feats, vwap, vwap_change, alpha, new_TTD) -> pd.Series:
    """
    Update dynamic features in a pandas Series and return a new Series.
    """
    
    new_feats = feats.copy()
    new_feats['vwap_lag1'] = vwap
    

    new_feats['vwap_changes_lag3'] = feats['vwap_changes_lag2']
    new_feats['vwap_changes_lag2'] = feats['vwap_changes_lag1']
    new_feats['vwap_changes_lag1'] = vwap_change
    

    for i in range(12, 1, -1):
        col_name = f'abs_vwap_changes_lag{i}'
        prev_col = f'abs_vwap_changes_lag{i-1}'
        new_feats[col_name] = feats[prev_col]
    new_feats['abs_vwap_changes_lag1'] = abs(vwap_change)
    

    for j in range(12, 1, -1):
        col_name = f'alpha_lag{j}'
        prev_col = f'alpha_lag{j-1}'
        new_feats[col_name] = feats[prev_col]
    new_feats['alpha_lag1'] = feats['alpha']

    new_feats['f_TTD'] = 1.0 / np.sqrt(1.0 + new_TTD)
    

    for i in range(1, 37):
        new_feats[f'TTD_bin_{i}'] = 1 if int(new_TTD) == i else 0
    

    new_feats['da-id'] = abs(new_feats['da_price'] - new_feats['vwap_lag1'])
    
    return new_feats


    
    

def simulate_intraday_paths(df_h_sim, col_gamlss, col_logreg,
                            model_logreg, model_gamlss,
                            n_sims=1000, T=36):
    """
    Perform the simulation-based forecast for the given day and hour.
    Returns an array of shape (n_sims, T+1) with the simulated price paths,
    or (n_sims, T) for the deltas, depending on your preference.
    """


    sim_prices = np.zeros((n_sims, T))
    start_price =df_h_sim['vwap_lag1']
    
    for i in range(n_sims):
        print(f"sim {i}")
        features = df_h_sim.copy()
        X_gamlss = features[col_gamlss].values
        X_logreg = features[col_logreg].values
        price_hist = [] 
        p_alpha = model_logreg.predict_proba(X_logreg.reshape(1, -1))[0,1]
        alpha_t = int(np.random.rand() < p_alpha)
          
        if alpha_t == 0:
            delta_t = 0.0
        else:
            ps = model_gamlss.predict(X_gamlss)
            # ps = jsu_reparam_to_original(*ps)
            # print(ps)
            delta_t = st.johnsonsu(a=ps[2], b=ps[3], loc=ps[0], scale=ps[1]).rvs()
        
        p_new =  start_price + delta_t
        price_hist.append(p_new)
        
        for ttd in range(T-1, 0, -1):
            features = update_features(features, p_new, delta_t, alpha_t, ttd) 
            X_gamlss = features[col_gamlss].values
            X_logreg = features[col_logreg].values


            p_alpha = model_logreg.predict_proba(X_logreg.reshape(1, -1))[0,1]
            alpha_t = np.random.rand() < p_alpha

            if not alpha_t:
                delta_t = 0.0
            else:
                ps = model_gamlss.predict(X_gamlss)
                # ps = jsu_reparam_to_original(*ps)
                print(ps)
                delta_t = st.johnsonsu(a=ps[2], b=ps[3], loc=ps[0], scale=ps[1]).rvs()
                print(f"delta t = {delta_t}")
                
            p_new = price_hist[-1] + delta_t
            print(f"p_new = {p_new}")
            price_hist.append(p_new)

        sim_prices[i, :] = price_hist

    return sim_prices

HOURS = [h for h in range(24) if h != 2]



for current_day in all_days:
    # The 1-year window is [current_day - 365 days, current_day]
    print(current_day)
    window_start = current_day - pd.Timedelta(days=365)
    
    # Filter data to this rolling window
    mask_all = (df_all['day'] >= window_start) & (df_all['day'] < current_day)
    mask_P  = (df_P['day'] >= window_start) & (df_P['day'] < current_day)
    df_train_logreg = df_all[mask_all].copy()
    df_train_gamlss = df_P[mask_P].copy()
    
    mask_sim = (df_all['day'] == current_day)
    df_sim= df_all[mask_sim].copy()


    # Fit each hour's model
    day_dict_gamlss = {}
    day_dict_logreg = {}
    day_dict_sim = {}
    day_dict_err = {}
    
    for hour in HOURS:
        df_h_gamlss = df_train_gamlss[df_train_gamlss['delivery_start'].dt.hour == hour].copy()
        X_gamlss = df_h_gamlss[col_gamlss].values
        y_gamlss = df_h_gamlss['vwap_changes'].values
        # Fit GAMLSS model
        model_gamlss = GAMLSS(distribution = JSUo())
        model_gamlss.fit(X_gamlss, y_gamlss)
        day_dict_gamlss[hour] = model_gamlss
        
        df_h_logreg = df_train_logreg[df_train_logreg['delivery_start'].dt.hour == hour].copy()
        X_logreg = df_h_logreg[col_logreg].values
        y_logreg = df_h_logreg['alpha'].values
        
        # Fit logreg model
        model_logreg = LogisticRegression(
            penalty='l1', 
            solver= 'liblinear',  
            max_iter=1000
        )
        model_logreg.fit(X_logreg, y_logreg)
        day_dict_logreg[hour] = model_logreg
        
        
        df_h_sim = df_sim[df_sim['delivery_start'].dt.hour == hour].iloc[0].copy()
        
        
        sim_prices = simulate_intraday_paths(
            df_h_sim,
            col_gamlss= col_gamlss,
            col_logreg= col_logreg,
            model_logreg=model_logreg,
            model_gamlss=model_gamlss,
            T=36,
            n_sims=1000
        )
        
        day_dict_sim[hour] = sim_prices
        
        price_path = df_sim[df_sim['delivery_start'].dt.hour == hour]['vwap'].values
        mean_path = sim_prices.mean(axis=0)
        err = price_path - mean_path
        day_dict_err[hour] = err

    # Store the set of hourly models for this day
    gamlss_results[current_day] = day_dict_gamlss
    logreg_results[current_day] = day_dict_logreg
    sim_results[current_day] = day_dict_sim
    err_results[current_day] = day_dict_err





