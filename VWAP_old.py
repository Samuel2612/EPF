import pandas as pd
from zipfile import ZipFile
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy import stats
import seaborn as sns
import types

class LimitOrderBookProcessor:
    def __init__(self, 
                 zip_file, 
                 time_interval="15min",
                 start_hours=5,
                 auction_spot_csv="auction_spot_prices_netherlands_2021.csv",
                 aggregator_zip="auction_aggregated_curves_netherlands_2021.zip"):
        """
        Parameters:
            zip_file (str): Path to the zip file containing CSV files of trades.
            time_interval (str): Resampling interval (e.g., '5min', '15min', '30min').
            start_hours (int/float): Look-back window size before delivery (in hours).
            auction_spot_csv (str): Path to the CSV file containing auction spot prices.
        """
        self.zip_file = zip_file
        self.time_interval = time_interval
        self.start_hours = start_hours
        
        if "2021" in zip_file:
            self.target_year = 2021
        elif "2022" in zip_file:
            self.target_year = 2022

        self.data = self._load_data()
        self._preprocess_data()

        self.auction_spot_prices = self._load_auction_spot_csv(auction_spot_csv)

        # Build the per-hour bin data using the configured window
        self._all_data_dict = self.extract_all_data(product_type='hourly', start_hours=self.start_hours)
        
        self.aggregator_zip = aggregator_zip
        self.aggregated_curves = self._load_aggregator_curves()

        # Precompute merit-order slopes aligned to the configured bins
        self.compute_and_store_all_mo_slopes()

    def _load_data(self):
        df_list = []
        rename_dict_2022 = {
            "DeliveryEnd"  : "Delivery End",
            "DeliveryStart": "Delivery Start",
            "ExecutionTime": "Execution time",
            "Volume"       : "Quantity (MW)",
            "TradeId"      : "TradeID",
            "SelfTrade"    : "Is Self Trade",
        }

        with ZipFile(self.zip_file, 'r') as top_zip:
            nested_zips = [f for f in top_zip.namelist() if f.lower().endswith('.zip')]
            if nested_zips:
                for nested_zip_name in nested_zips:
                    with ZipFile(top_zip.open(nested_zip_name)) as nested_zip:
                        csv_files = [x for x in nested_zip.namelist() if x.lower().endswith('.csv')]
                        for csv_file in csv_files:
                            tmp_df = pd.read_csv(nested_zip.open(csv_file), skiprows=1)
                            if any(c in tmp_df.columns for c in rename_dict_2022):
                                tmp_df.rename(columns=rename_dict_2022, inplace=True)
                            df_list.append(tmp_df)
            else:
                csv_files = [f for f in top_zip.namelist() if f.lower().endswith('.csv')]
                for csv_file in csv_files:
                    tmp_df = pd.read_csv(top_zip.open(csv_file))
                    if any(c in tmp_df.columns for c in rename_dict_2022):
                        tmp_df.rename(columns=rename_dict_2022, inplace=True)
                    df_list.append(tmp_df)

        if not df_list:
            raise ValueError("No CSV files found in the zip (or nested zips).")

        combined = pd.concat(df_list, ignore_index=True)
        return combined

    def _preprocess_data(self):
        cet_tz = 'Europe/Berlin'
        self.data['Delivery Start'] = pd.to_datetime(
            self.data['Delivery Start'], format='ISO8601', utc=True
        ).dt.tz_convert(cet_tz)
        self.data['Delivery End'] = pd.to_datetime(
            self.data['Delivery End'], format='ISO8601', utc=True
        ).dt.tz_convert(cet_tz)
        self.data['Execution time'] = pd.to_datetime(
            self.data['Execution time'], format='ISO8601', utc=True
        ).dt.tz_convert(cet_tz)
        
        if 'TradePhase' in self.data.columns:
            self.data = self.data[self.data['TradePhase'] == 'CONT']
        self.data = self.data[self.data['Is Self Trade'] != 'Y']
        self.data = self.data.drop_duplicates(subset='TradeID', keep='first')
        
        self.data['ProductDuration'] = (
            self.data['Delivery End'] - self.data['Delivery Start']
        ).dt.total_seconds() / 60
        
        self.data = self.data[self.data['Delivery Start'].dt.year == self.target_year]
        self.data = self.data[~self.data['Delivery Start'].dt.hour.isin([2])]
        
    def _load_auction_spot_csv(self, csv_file: str) -> pd.DataFrame:
        df = pd.read_csv(csv_file, skiprows=1)
        df = df.iloc[:, :26]

        def combine_3A_3B(row):
            a, b = row["Hour 3A"], row["Hour 3B"]
            if pd.isna(a) and pd.isna(b):
                return np.nan
            elif pd.isna(a):
                return b
            elif pd.isna(b):
                return a
            else:
                return (a + b) / 2
        
        df["Hour 3"] = df.apply(combine_3A_3B, axis=1)
        df.drop(["Hour 3A", "Hour 3B"], axis=1, inplace=True)
        
        cols = df.columns.tolist()
        cols = cols[:3] + cols[-1:] + cols[3:-1]
        df = df[cols]  
        
        cols = ["Delivery Date"] + [f"{int(i)}" for i in list(range(24))]
        df.columns = cols

        def fill_hours_with_interpolation(row):
            return row.interpolate(method="linear", limit_direction="both", axis=0)
        
        df["Delivery Date"] = pd.to_datetime(df["Delivery Date"], format="%d/%m/%Y").dt.tz_localize("Europe/Brussels")
        df.set_index("Delivery Date", inplace=True)
        df = df.apply(fill_hours_with_interpolation, axis=1)
        df.drop(columns=["2"], inplace=True)
        return df

    def _get_auction_spot_price(self, delivery_start: pd.Timestamp) -> float:
        day = delivery_start.normalize()
        hour = str(delivery_start.hour)
        return self.auction_spot_prices.loc[day].loc[hour]

    def _compute_vwap(self, group: pd.DataFrame):
        total_qty = group['Quantity (MW)'].sum()
        if total_qty > 0:
            return (group['Price'] * group['Quantity (MW)']).sum() / total_qty
        return None

    def extract_all_data(self, product_type='hourly', start_hours=None):
        """
        Build a fixed grid of bins in the last `start_hours` before delivery using `self.time_interval`,
        and compute per-bin VWAP, ΔVWAP, and alpha=1{trades>0}. Fallback to DA price where needed.
        """
        if start_hours is None:
            start_hours = self.start_hours

        duration_map = {'hourly': 60, 'half_hourly': 30, 'quarterly': 15}
        if product_type not in duration_map:
            raise ValueError("Invalid product_type. Choose from 'hourly', 'half_hourly', or 'quarterly'.")
        target_duration = duration_map[product_type]
        
        df_filtered = self.data[np.isclose(self.data['ProductDuration'], target_duration, atol=1e-6)].copy()
        
        if self.target_year == 2021:
            all_hours = pd.date_range(
                start="2021-01-01 00:00:00", 
                end="2021-12-31 23:00:00", 
                freq="h", 
                tz="Europe/Berlin"
            )
        elif self.target_year == 2022:
            all_hours = pd.date_range(
                start="2022-01-01 00:00:00", 
                end="2022-12-30 23:00:00", 
                freq="h", 
                tz="Europe/Berlin"
            )
        all_hours = [h for h in all_hours if h.hour != 2]
        
        interval_td = pd.Timedelta(self.time_interval)
        interval_minutes = interval_td.total_seconds() / 60.0
        num_intervals = int(round(start_hours * 60 / interval_minutes))
        
        results = {}
        for start_time in all_hours:
            df_sub = df_filtered[df_filtered['Delivery Start'] == start_time].copy()
            window_start = start_time - pd.Timedelta(hours=start_hours)
            full_index = pd.date_range(start=window_start, periods=num_intervals, freq=self.time_interval, tz=start_time.tz)

            if df_sub.empty:
                fallback_price = self._get_auction_spot_price(start_time)
                if np.isnan(fallback_price):
                    fallback_price = -9999
                df_result = pd.DataFrame({
                    'vwap': [fallback_price]*len(full_index),
                    'vwap_changes': [0]*len(full_index),
                    'alpha': [0]*len(full_index),
                }, index=full_index)
            else:
                df_sub.set_index('Execution time', inplace=True)
                df_sub.sort_index(inplace=True)
                
                vwap_series  = df_sub.resample(self.time_interval).apply(self._compute_vwap)
                alpha_series = df_sub.resample(self.time_interval).size().apply(lambda x: 1 if x > 0 else 0)
                
                vwap_series  = vwap_series.reindex(full_index)
                alpha_series = alpha_series.reindex(full_index, fill_value=0)

                if vwap_series.isna().all():
                    fallback_price = self._get_auction_spot_price(start_time)
                    vwap_series = pd.Series(fallback_price, index=full_index)
                    alpha_series = pd.Series(0, index=full_index)
                
                if pd.isna(vwap_series.iloc[0]):
                    vwap_series.iloc[0] = self._get_auction_spot_price(start_time)

                vwap_series = vwap_series.ffill()
                vwap_changes = vwap_series.diff().fillna(0)

                df_result = pd.DataFrame({
                    'vwap': vwap_series,
                    'vwap_changes': vwap_changes,
                    'alpha': alpha_series
                }, index=full_index)
            
            results[start_time] = df_result
        
        return results

    def aggregate_extracted_data(self, product_type='hourly', start_hours=None):
        data_dict = self.extract_all_data(product_type, start_hours)
        dfs = []
        for d_start, df in data_dict.items():
            df = df.copy()
            df['delivery_start'] = d_start
            dfs.append(df)
        return pd.concat(dfs)

    def _load_aggregator_curves(self):
        df_list = []
        with ZipFile(self.aggregator_zip, 'r') as z:
            csv_files = [f for f in z.namelist() if f.lower().endswith('.csv')]
            for csv_file in csv_files:
                agg_df = pd.read_csv(z.open(csv_file), skiprows=1)
                mask_3B = agg_df['Hour'].astype(str).str.lower().str.contains('3b')
                agg_df = agg_df[~mask_3B]
                agg_df["Hour"] = agg_df['Hour'].astype(int)
                agg_df["Hour"] = agg_df["Hour"] - 1
                agg_df = agg_df[agg_df['Hour'] != 2]
                agg_df['Date'] = pd.to_datetime(agg_df['Date'], dayfirst=True)
                df_list.append(agg_df)
        if not df_list:
            raise ValueError(f"No aggregator CSV found in {self.aggregator_zip}")
        combined = pd.concat(df_list, ignore_index=True)
        combined.sort_values(['Date','Hour','Sale/Purchase','Price'], inplace=True)
        return combined
 
    def get_p_id0_first_vwap(self, delivery_start_ts):
        df_res = self._all_data_dict[delivery_start_ts]
        return df_res['vwap'].iloc[0]
    
    def transform_supply_curve(self, df_hour, p_min=-500.0):
        supply = df_hour[df_hour["Sale/Purchase"].str.lower() == "sell"].sort_values(by="Price")
        demand = df_hour[df_hour["Sale/Purchase"].str.lower().isin(["buy","purchase"])].sort_values(by="Price")
        
        def supply_inv(p): return np.interp(p, supply["Price"], supply["Volume"], left=0, right=0)
        def demand_inv(p): return np.interp(p, demand["Price"], demand["Volume"], left=0, right=0)
        
        inelastic_dmd = demand_inv(p_min)
        def transformed_supply_inv(p):
            return supply_inv(p) + inelastic_dmd - demand_inv(p)

        price_range = np.linspace(supply["Price"].min(), supply["Price"].max(), 500)
        transf_vols = np.array([transformed_supply_inv(px) for px in price_range])
        return price_range, transf_vols

    def compute_merit_order_slope(self, price_range, transformed_volumes, p_id0, q_values=[500,1000,2000]):
        def supply_func(vol):
            return np.interp(vol, transformed_volumes, price_range,
                             left=price_range[0], right=price_range[-1])
        def supply_func_inv(price_val):
            return np.interp(price_val, price_range, transformed_volumes,
                             left=transformed_volumes[0], right=transformed_volumes[-1])
        
        dem_implied = supply_func_inv(p_id0)
        
        mo_dict = {}
        for q in q_values:
            vol_plus  = dem_implied + q
            vol_minus = max(dem_implied - q, 0)
            p_plus  = supply_func(vol_plus)
            p_minus = supply_func(vol_minus)
            mo_dict[q] = (p_plus - p_minus)/(2.0*q)
        return mo_dict

    def compute_and_store_all_mo_slopes(self, p_min=-500.0):
        """
        For each delivery_start window, compute merit-order slopes at each bin
        using the carry-forward VWAP for that bin (aligned to `self.time_interval`).
        """
        supply_cache = {}
        for (d, h), df_hour in self.aggregated_curves.groupby(['Date', 'Hour']):
            delivery_start = (pd.Timestamp(d) + pd.Timedelta(hours=int(h))
                              ).tz_localize('Europe/Brussels')
            supply_cache[delivery_start] = self.transform_supply_curve(df_hour, p_min=p_min)
    
        self.merit_order_slopes = {}
        for delivery_start, df_bins in self._all_data_dict.items():
            price_range, transf_vols = supply_cache[delivery_start]
            vwap_ff = df_bins['vwap'].ffill()
            if vwap_ff.isna().all():
                fallback_price = self._get_auction_spot_price(delivery_start)
                vwap_ff = pd.Series(fallback_price, index=df_bins.index)
    
            records = []
            prev_price = None
            prev_mo = None
    
            for ts, price in vwap_ff.items():
                if (prev_price is not None) and np.isclose(price, prev_price):
                    mo = prev_mo
                else:
                    mo = self.compute_merit_order_slope(price_range, transf_vols, price)
                    prev_price = price
                    prev_mo = mo
    
                records.append(
                    {
                        'bin_timestamp': ts,
                        'mo_slope_500':  mo[500],
                        'mo_slope_1000': mo[1000],
                        'mo_slope_2000': mo[2000],
                    }
                )
    
            self.merit_order_slopes[delivery_start] = (
                pd.DataFrame(records).set_index('bin_timestamp')
            )
            
    def merge_merit_order_slopes(self, df):
        flat_frames = []
        for delivery_start, df_slope in self.merit_order_slopes.items():
            tmp = (df_slope.reset_index()
                        .assign(delivery_start=delivery_start))
            flat_frames.append(tmp)
        df_mo = pd.concat(flat_frames, ignore_index=True)
        return df.merge(df_mo, on=['delivery_start', 'bin_timestamp'], how='left')
    
    def merge_res_forecasts(self, df, res_csv, res_type):
        res_df = pd.read_csv(res_csv, skiprows=[1])
        res_df["Date (CET)"] = res_df["Date (CET)"].str.strip("[]")
        res_df['Date (CET)'] = pd.to_datetime(res_df['Date (CET)'], format="%d/%m/%Y %H:%M", errors="coerce").dt.tz_localize('Europe/Brussels', ambiguous='infer')
        res_df = res_df[(res_df['Date (CET)'].dt.year == self.target_year) & (res_df['Date (CET)'].dt.hour != 2)]
        res_df.set_index("Date (CET)", inplace=True)
        res_df.index.name = "delivery_start"
        res_df = res_df[[f"{res_type.upper()} DA FORECAST (TENNET)"]].rename(columns={f'{res_type.upper()} DA FORECAST (TENNET)': f'{res_type.lower()}_forecast'})
        merged_df = df.merge(res_df, on="delivery_start", how="left")
        return merged_df

    @staticmethod
    def create_weekday_dummies(delivery_start_times):
        weekdays = delivery_start_times.dt.weekday
        return {
            'MON': (weekdays == 0).astype(int),
            'SAT': (weekdays == 5).astype(int),
            'SUN': (weekdays == 6).astype(int)
        }

    @staticmethod
    def prepare_hourly_arrays(df):
        hourly_data = {}
        for hour in range(24):
            hour_df = df[df['delivery_hour'] == hour]
            dtype = [
                ('vwap_changes', 'float64'),
                ('alpha', 'int64'),
                ('MON', 'int64'),
                ('SAT', 'int64'), 
                ('SUN', 'int64'),
                ('f_TTD', 'float64'),
            ]
            hourly_data[hour] = np.array(
                [tuple(x) for x in hour_df.values],
                dtype=dtype
            )
        return hourly_data
    
    def prepare_gamlss_data(self, product_type='hourly', start_hours=None, 
                            wind_csv='wind-renewables-elec.csv',
                            solar_csv='solar-renewables-elec.csv'):
        if start_hours is None:
            start_hours = self.start_hours

        # Aggregate using requested window
        agg_df = self.aggregate_extracted_data(product_type, start_hours)
        agg_df = agg_df.reset_index().rename(columns={'index': 'bin_timestamp'})
        
        # Bin index from delivery using actual interval length
        interval_td = pd.Timedelta(self.time_interval)
        interval_seconds = interval_td.total_seconds()
        agg_df['TTD_bins'] = (
            (agg_df['delivery_start'] - agg_df['bin_timestamp']).dt.total_seconds() / interval_seconds
        )
        agg_df['f_TTD'] = 1.0 / np.sqrt(1.0 + agg_df['TTD_bins'])
        
        agg_df['da_price'] = agg_df['delivery_start'].apply(self._get_auction_spot_price)
    
        weekdays = agg_df['delivery_start'].dt.weekday
        agg_df['MON'] = (weekdays == 0).astype(int)
        agg_df['SAT'] = (weekdays == 5).astype(int)
        agg_df['SUN'] = (weekdays == 6).astype(int)
    
        merged_df = self.merge_merit_order_slopes(agg_df)
        merged_df = self.merge_res_forecasts(merged_df, wind_csv, 'wind')
        merged_df = self.merge_res_forecasts(merged_df, solar_csv, 'solar')
    
        merged_df['delivery_date'] = merged_df['delivery_start'].dt.normalize()
        merged_df['delivery_hour'] = merged_df['delivery_start'].dt.hour
    
        merged_df = self.add_absolute_value_columns(merged_df, ['vwap_changes'])
        merged_df = self.add_lagged_variables(
            merged_df,
            {
                'vwap': [1, 2, 3], 
                'vwap_changes': [1, 2, 3], 
                'alpha': list(range(1, 13)),
                'abs_vwap_changes': list(range(1, 13))
            }
        )
        
        merged_df["da-id"] = merged_df['da_price'] - merged_df['vwap']
        merged_df["da-id_lag1"] = merged_df['da_price'] - merged_df['vwap_lag1']
        merged_df["da-id_lag2"] = merged_df['da_price'] - merged_df['vwap_lag2']
        
        # Keep only rows within the configured window
        max_bins = int(round(start_hours * 3600 / interval_seconds))
        merged_df = merged_df[merged_df['TTD_bins'] <= max_bins].copy()
    
        merged_df['TTD_bins_int'] = np.floor(merged_df['TTD_bins']).astype(int)
        ttd_dummies = pd.get_dummies(merged_df['TTD_bins_int'], prefix='TTD_bin').astype(int)
        merged_df = pd.concat([merged_df, ttd_dummies], axis=1)
    
        return merged_df

    @staticmethod
    def add_lagged_variables(df, lag_dict):
        df = df.sort_values(by=['delivery_start', 'bin_timestamp']).copy()
        for col in list(df.columns):
            if col in lag_dict:
                count_inserted = 0
                for lag in lag_dict[col]:
                    new_col_name = f"{col}_lag{lag}"
                    insertion_index = df.columns.get_loc(col) + 1 + count_inserted
                    new_series = df.groupby('delivery_start')[col].shift(lag)
                    if new_col_name in df.columns:
                        df.drop(columns=[new_col_name], inplace=True)
                    df.insert(insertion_index, new_col_name, new_series)
                    count_inserted += 1
        return df

    @staticmethod
    def add_absolute_value_columns(df, columns):
        for col in columns:
            if col in df.columns:
                insertion_index = df.columns.get_loc(col) + 1
                new_col = f"abs_{col}"
                if new_col in df.columns:
                    df.drop(columns=[new_col], inplace=True)
                df.insert(insertion_index, new_col, df[col].abs())
        return df
    
    def plot_price_changes_histogram(self, product_type='hourly', start_hours=None, delivery_hour=None, bins=51):
        if start_hours is None:
            start_hours = self.start_hours
        if bins % 2 == 0: 
            bins += 1

        agg_df = self.aggregate_extracted_data(product_type, start_hours)
        if delivery_hour is not None:
            agg_df = agg_df[agg_df['delivery_start'].dt.hour == delivery_hour]
        df_changes = agg_df[['vwap_changes', 'alpha']].dropna()
        changes = df_changes['vwap_changes'].values
        alphas = df_changes['alpha'].values
        
        max_val = max(abs(changes.min()), abs(changes.max()))
        bin_edges = np.linspace(-max_val, max_val, bins+1)
        counts, _ = np.histogram(changes, bins=bin_edges)
        bin_width = bin_edges[1] - bin_edges[0]
        bin_centers = bin_edges[:-1] + bin_width / 2
        
        zero_bin_index = None
        for i in range(len(bin_edges)-1):
            if bin_edges[i] <= 0 < bin_edges[i+1]:
                zero_bin_index = i
                break
        
        counts_alpha0, _ = np.histogram(changes[alphas == 0], bins=bin_edges)
        
        plt.figure(figsize=(10, 6))
        for i in range(len(counts)):
            if i == zero_bin_index:
                count_alpha0 = counts_alpha0[i]
                count_other = counts[i] - count_alpha0
                plt.bar(bin_centers[i], count_other, width=bin_width, color='blue',
                        edgecolor='black', align='center')
                plt.bar(bin_centers[i], count_alpha0, width=bin_width, color='red',
                        edgecolor='black', align='center', bottom=count_other)
            else:
                plt.bar(bin_centers[i], counts[i], width=bin_width, color='blue',
                        edgecolor='black', align='center')
        
        plt.xlabel("VWAP Price Changes")
        plt.ylabel("Frequency")
        title = f"Histogram of VWAP Price Changes (Last {start_hours}h before delivery)"
        if delivery_hour is not None:
            title += f"\nfor Delivery Hour = {delivery_hour:02d}:00"
        plt.title(title)
        plt.grid(True)

        blue_patch = mpatches.Patch(color='blue', label='Intervals with alpha != 0')
        red_patch = mpatches.Patch(color='red', label='Intervals with alpha = 0 (zero bin)')
        plt.legend(handles=[blue_patch, red_patch])
        plt.show()

    def fit_distributions_to_price_changes(self, product_type='hourly', start_hours=None ,delivery_hour=None, bins=51, plot=True):
        if start_hours is None:
            start_hours = self.start_hours

        agg_df = self.aggregate_extracted_data(product_type, start_hours)
        if delivery_hour is not None:
            agg_df = agg_df[agg_df['delivery_start'].dt.hour == delivery_hour]
        valid_df = agg_df[agg_df['alpha'] != 0]
        changes = valid_df['vwap_changes'].dropna()
        
        params_johnsonsu = stats.johnsonsu.fit(changes)
        params_t = stats.t.fit(changes)
        params_norm = stats.norm.fit(changes)
        
        ks_johnsonsu = stats.kstest(changes, 'johnsonsu', args=params_johnsonsu)
        ks_t = stats.kstest(changes, 't', args=params_t)
        ks_norm = stats.kstest(changes, 'norm', args=params_norm)
        
        if plot:
            x = np.linspace(changes.min(), changes.max(), 1000)
            pdf_johnsonsu = stats.johnsonsu.pdf(x, *params_johnsonsu)
            pdf_t = stats.t.pdf(x, *params_t)
            pdf_norm = stats.norm.pdf(x, *params_norm)
            
            plt.figure(figsize=(10, 6))
            plt.hist(changes, bins=np.linspace(changes.min(), changes.max(), bins+1),
                     density=True, edgecolor='black', alpha=0.6, label='Data histogram')
            plt.plot(x, pdf_johnsonsu, 'r-', lw=2, label="Johnson's SU")
            plt.plot(x, pdf_t, 'g-', lw=2, label="Student's t")
            plt.plot(x, pdf_norm, 'b-', lw=2, label="Normal")
            plt.xlabel("VWAP Price Changes")
            plt.ylabel("Density")
            title = "Fitted Distributions for VWAP Price Changes (alpha != 0)"
            if delivery_hour is not None:
                title += f"\nfor Delivery Hour = {delivery_hour:02d}:00"
            plt.title(title)
            plt.legend()
            plt.grid(True)
            plt.show()
        
        results = {
            'johnsonsu': {
                'params': params_johnsonsu,
                'ks_statistic': ks_johnsonsu.statistic,
                'p_value': ks_johnsonsu.pvalue
            },
            'student_t': {
                'params': params_t,
                'ks_statistic': ks_t.statistic,
                'p_value': ks_t.pvalue
            },
            'normal': {
                'params': params_norm,
                'ks_statistic': ks_norm.statistic,
                'p_value': ks_norm.pvalue
            }
        }
        return results

    def plot_data(self, delivery_start, product_type='hourly', start_hours=None):
        if start_hours is None:
            start_hours = self.start_hours

        data_dict = self.extract_all_data(product_type, start_hours)
        if delivery_start not in data_dict:
            print(f"No data found for Delivery Start: {delivery_start}")
            return
        df_result = data_dict[delivery_start]

        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        axes[0].step(df_result.index, df_result['vwap'], where='post', linestyle='-')
        axes[0].set_title("VWAP")
        axes[0].set_ylabel("Price")
        axes[0].grid(True)
        
        axes[1].step(df_result.index, df_result['vwap_changes'], where='post', linestyle='-', color='orange')
        axes[1].set_title("VWAP Price Changes")
        axes[1].set_ylabel("Price Change")
        axes[1].grid(True)
        
        axes[2].step(df_result.index, df_result['alpha'], where='post', color='green')
        axes[2].set_title("Alpha (Trade Indicator)")
        axes[2].set_ylabel("Alpha")
        axes[2].set_xlabel("Execution Time")
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.show()

    def plot_total_traded_volume_per_delivery_hour(self):
        df = self.data.copy()
        tol = 1e-6
        conditions = [
            np.abs(df['ProductDuration'] - 60) < tol,
            np.abs(df['ProductDuration'] - 30) < tol,
            np.abs(df['ProductDuration'] - 15) < tol
        ]
        choices = ['hourly', 'half_hourly', 'quarterly']
        df['product_type'] = np.select(conditions, choices, default='other')
        df = df[df['product_type'] != 'other']
        df['delivery_hour'] = df['Delivery Start'].dt.hour
        volume_pivot = df.groupby(['delivery_hour', 'product_type'])['Quantity (MW)'].sum().unstack()
        
        volume_pivot.plot(kind='bar', figsize=(10, 6))
        plt.xlabel("Delivery Hour")
        plt.ylabel("Total Traded Volume (MW)")
        plt.title("Total Traded Volume per Delivery Hour by Product Type")
        plt.legend(title="Product Type")
        plt.grid(True, axis='y')
        plt.tight_layout()
        plt.show()

# === Example Usage ===
if __name__ == '__main__':
    # 5-hour window, 5-minute bins
    lob_2021 = LimitOrderBookProcessor(
        zip_file="Continuous_Trades_NL_2021.zip",
        time_interval="15min",
        start_hours=6,
        auction_spot_csv='auction_spot_prices_netherlands_2021.csv',
        aggregator_zip="auction_aggregated_curves_netherlands_2021.zip"
    )
    datas_2021 = lob_2021.prepare_gamlss_data()  # uses start_hours=5 by default

    lob_2022 = LimitOrderBookProcessor(
        zip_file="Continuous_Trades_NL_2022.zip",
        time_interval="5min",
        start_hours=5,
        auction_spot_csv='auction_spot_prices_netherlands_2022.csv',
        aggregator_zip="auction_aggregated_curves_netherlands_2022.zip"
    )
    datas_2022 = lob_2022.prepare_gamlss_data()

    datas_2021.to_csv("df_2021_MVT_symm.csv", index=False)
    datas_2022.to_csv("df_2022_MVT_symm.csv", index=False)
