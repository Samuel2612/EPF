# -*- coding: utf-8 -*-
"""
Created on Thu Jun  5 10:35:29 2025

@author: samue
"""

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
                 time_interval="5min", 
                 auction_spot_csv="auction_spot_prices_netherlands_2021.csv",
                 aggregator_zip= "auction_aggregated_curves_netherlands_2021.zip",
                 wind_data = 'wind-renewables-elec.csv', solar_data = 'solar-renewables-elec.csv'):
        """
        Parameters:
            zip_file (str): Path to the zip file containing CSV files of trades.
            time_interval (str): Resampling interval (default is 5 minutes).
            auction_spot_csv (str): Path to the CSV file containing auction spot prices.
        """
        self.zip_file = zip_file
        self.wind_data = wind_data
        self.solar_data = solar_data
        self.time_interval = time_interval
        
        if "2021" in zip_file:
           self.target_year = 2021
        elif "2022" in zip_file:
           self.target_year = 2022
        elif "2023" in zip_file:
           self.target_year = 2023
        elif "2024" in zip_file:
           self.target_year = 2024

        self.data = self._load_data()
        self._preprocess_data()
       

        self.auction_spot_prices = self._load_auction_spot_csv(auction_spot_csv)
        self._all_data_dict = self.extract_all_data(product_type='hourly', start_hours=3)
        
        self.aggregator_zip = aggregator_zip
        self.aggregated_curves = self._load_aggregator_curves()
        self.compute_and_store_all_mo_slopes() 

    def _load_data(self):
        """
        """
        df_list = []

        # A dict mapping new 2022 column names -> old, standardized column names:
        rename_dict_2022 = {
            "DeliveryEnd"  :    "Delivery End",
            "DeliveryStart":    "Delivery Start",
            "ExecutionTime":    "Execution time",
            "Volume"       :    "Quantity (MW)",
            "TradeId"      :     "TradeID",
            "SelfTrade"    :     "Is Self Trade",
            # Keep "TradePhase" as is, or rename if you like:
            # "TradePhase":       "TradePhase",
            # "TradeID":          "TradeID",   # only needed if your new data has a different name
            # etc. 
        }

        with ZipFile(self.zip_file, 'r') as top_zip:
            # Check for nested zips
            nested_zips = [f for f in top_zip.namelist() if f.lower().endswith('.zip')]

            if nested_zips:
                # === "2022 style": read CSVs from inside each nested zip ===
                for nested_zip_name in nested_zips:
                    with ZipFile(top_zip.open(nested_zip_name)) as nested_zip:
                        csv_files = [
                            x for x in nested_zip.namelist() 
                            if x.lower().endswith('.csv')
                        ]
                        for csv_file in csv_files:
                            tmp_df = pd.read_csv(
                                nested_zip.open(csv_file),
                                skiprows=1)
                            # Rename columns if "tradePriceEUR" or similar is present
                            if any(c in tmp_df.columns for c in rename_dict_2022):
                                tmp_df.rename(columns=rename_dict_2022, inplace=True)

                            df_list.append(tmp_df)

            else:
                # === "2021 style": read CSVs directly from top-level zip ===
                csv_files = [f for f in top_zip.namelist() if f.lower().endswith('.csv')]
                for csv_file in csv_files:
                    tmp_df = pd.read_csv(
                        top_zip.open(csv_file),
                        # skiprows=1  # if needed
                    )
                    if any(c in tmp_df.columns for c in rename_dict_2022):
                        tmp_df.rename(columns=rename_dict_2022, inplace=True)

                    df_list.append(tmp_df)

        if not df_list:
            raise ValueError("No CSV files found in the zip (or nested zips).")

        combined = pd.concat(df_list, ignore_index=True)
        return combined

    def _preprocess_data(self):
        """
        Preprocess the data by:
          - Converting UTC columns to CET (Europe/Berlin).
          - Filtering out self-trades.
          - Dropping duplicate TradeIDs.
          - Computing product duration (in minutes) from 'Delivery Start' and 'Delivery End'.
        """
        cet_tz = 'Europe/Berlin'
        self.data['Delivery Start'] = pd.to_datetime(
            self.data['Delivery Start'], format='ISO8601', utc = True
        ).dt.tz_convert(cet_tz)
        self.data['Delivery End'] = pd.to_datetime(
            self.data['Delivery End'], format='ISO8601', utc = True
        ).dt.tz_convert(cet_tz)
        self.data['Execution time'] = pd.to_datetime(
            self.data['Execution time'], format='ISO8601', utc = True
        ).dt.tz_convert(cet_tz)
        
        #Keep only CONT trades if present
        if 'TradePhase' in self.data.columns:
            self.data = self.data[self.data['TradePhase'] == 'CONT']
            
        # Keep only non self-trades.
        self.data = self.data[self.data['Is Self Trade'] != 'Y']
        # Keep only rows with unique TradeID.
        self.data = self.data.drop_duplicates(subset='TradeID', keep='first')
        
        self.data['ProductDuration'] = (
            self.data['Delivery End'] - self.data['Delivery Start']
        ).dt.total_seconds() / 60
         
        self.data = self.data[self.data['Delivery Start'].dt.year == self.target_year]
        self.data = self.data[~self.data['Delivery Start'].dt.hour.isin([2])]
        
    def _load_auction_spot_csv(self, csv_file: str) -> pd.DataFrame:
        """
        Load a CSV containing daily rows and columns for each hour of the day.
        Some rows have '3a' and '3b' for DST changes. We aggregate them into a
        single hour '3', and if partial columns exist, we average whatever is present.
        
        Returns a DataFrame with a DateTimeIndex (tz='Europe/Berlin') and a single column:
            'auction_spot_price'
        """
        # Read raw CSV. 
        df = pd.read_csv(csv_file, skiprows=1)
        df = df.iloc[:, :26]


        def combine_3A_3B(row):
            """Average Hour 3A and Hour 3B if both present, 
               otherwise use whichever is present."""
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
        
        # Set aside the date column so we can apply rowwise interpolation
        df["Delivery Date"] = pd.to_datetime(df["Delivery Date"], format="%d/%m/%Y").dt.tz_localize("Europe/Brussels")
        df.set_index("Delivery Date", inplace=True)
        
        
        df = df.apply(fill_hours_with_interpolation, axis=1)
        
        df.drop(columns=["2"], inplace=True)
        

        return df

    def _get_auction_spot_price(self, delivery_start: pd.Timestamp) -> float:
        """
        Return the auction spot price for the product's delivery_start from the loaded CSV.
        """
        day = delivery_start.normalize()
        hour = str(delivery_start.hour)
        
        return self.auction_spot_prices.loc[day].loc[hour]
    
    
    

    def _compute_vwap(self, group: pd.DataFrame):
        """
        Compute the volume-weighted average price (VWAP).
        Uses 'Price' and 'Quantity (MW)' columns.
        """
        total_qty = group['Quantity (MW)'].sum()
        if total_qty > 0:
            return (group['Price'] * group['Quantity (MW)']).sum() / total_qty
        return None

    def extract_all_data(self, product_type='hourly', start_hours=3):
        """
        For each hour of 2021, extract data for the last <start_hours> hours before delivery,
        resampled every `self.time_interval`. If no trades exist in a window, fill VWAP with the 
        corresponding day-ahead price, set vwap_changes = 0, and alpha = 0.
        
        Returns:
            dict: 
                key:  delivery_start (Timestamp)
                val:  DataFrame(index=[bins in last 3 hours], columns=['vwap','vwap_changes','alpha'])
        """
        # 1) Map product_type to expected product duration.
        duration_map = {'hourly': 60, 'half_hourly': 30, 'quarterly': 15}
        if product_type not in duration_map:
            raise ValueError("Invalid product_type. Choose from 'hourly', 'half_hourly', or 'quarterly'.")
        target_duration = duration_map[product_type]
        
        # 2) Filter continuous trade data for the specified product duration.
        df_filtered = self.data[np.isclose(self.data['ProductDuration'], target_duration, atol=1e-6)].copy()
        
        # 3) Create a date range for every hour in 2021 (CET).
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
        elif self.target_year == 2023:
            all_hours = pd.date_range(
                start="2023-01-01 00:00:00", 
                end="2023-12-31 23:00:00", 
                freq="h", 
                tz="Europe/Berlin"
            )
            
        elif self.target_year == 2024:
            all_hours = pd.date_range(
                start="2024-01-01 00:00:00", 
                end="2024-12-31 23:00:00", 
                freq="h", 
                tz="Europe/Berlin"
            )
        
        all_hours = [h for h in all_hours if h.hour != 2]
        

        interval_minutes = pd.Timedelta(self.time_interval).total_seconds() / 60.0
        num_intervals = int(start_hours * 60 / interval_minutes)
        
        results = {}
        
        for start_time in all_hours:
            df_sub = df_filtered[df_filtered['Delivery Start'] == start_time].copy()
            
            # Define the uniform time index for the last <start_hours> hours
            window_start = start_time - pd.Timedelta(hours=start_hours)
            full_index = pd.date_range(start=window_start, periods=num_intervals, freq=self.time_interval)
            
            if df_sub.empty:
                # === No trades for this hour: fill with day-ahead price and zeros. ===
                fallback_price = self._get_auction_spot_price(start_time)
                if np.isnan(fallback_price):
                    fallback_price = -9999  
                
                # vwap = fallback_price, vwap_changes = 0, alpha = 0
                df_result = pd.DataFrame({
                    'vwap': [fallback_price]*len(full_index),
                    'vwap_changes': [0]*len(full_index),
                    'alpha': [0]*len(full_index),
                }, index=full_index)
                
            else:
                df_sub.set_index('Execution time', inplace=True)
                df_sub.sort_index(inplace=True)
                
                # Resample => compute VWAP and alpha
                vwap_series = df_sub.resample(self.time_interval).apply(self._compute_vwap)
                alpha_series= df_sub.resample(self.time_interval).size().apply(lambda x: 1 if x>0 else 0)
                
                    
                
                #Reindex to full time window
                vwap_series = vwap_series.reindex(full_index)
                alpha_series= alpha_series.reindex(full_index, fill_value=0)
                
                
    
                
                # If entire series is NaN => no trades at all in the window
                if vwap_series.isna().all():
                    fallback_price = self._get_auction_spot_price(start_time)
                    vwap_series.fillna(fallback_price, inplace=True)
                    alpha_series = 0
                    
                # If the earliest bins are NaN, fill with fallback
                if pd.isna(vwap_series.iloc[0]):
                    vwap_series.iloc[0]  = self._get_auction_spot_price(start_time)
                    
                
                vwap_series = vwap_series.ffill()
                vwap_changes = vwap_series.diff().fillna(0)

                # Build the final DataFrame for this hour
                df_result = pd.DataFrame({
                    'vwap': vwap_series,
                    'vwap_changes': vwap_changes,
                    'alpha': alpha_series
                }, index=full_index)
            
            # Store in the results dict
            results[start_time] = df_result
        
        return results

   

    def aggregate_extracted_data(self, product_type='hourly', start_hours=3):
        """
        Aggregate all individual DataFrames (for each Delivery Start) into one DataFrame.
        An additional column 'delivery_start' is added.
        
        Parameters:
            product_type (str): One of ['hourly', 'half_hourly', 'quarterly'].
        
        Returns:
            pd.DataFrame: Aggregated DataFrame with columns 'vwap', 'vwap_changes', 'alpha',
                          and 'delivery_start'.
        """
        data_dict = self.extract_all_data(product_type, start_hours)
        dfs = []
        for d_start, df in data_dict.items():
            df = df.copy()
            df['delivery_start'] = d_start
            dfs.append(df)

        return pd.concat(dfs)
       
        
    def _load_aggregator_curves(self):
        """
        Load the aggregator supply/demand curves for 2021 from a zip of CSVs.
        Returns one combined DataFrame with columns:
          Date (datetime), Hour, Sale/Purchase, Price, Volume, ...
        """
       
        df_list = []
        with ZipFile(self.aggregator_zip, 'r') as z:
            csv_files = [f for f in z.namelist() if f.lower().endswith('.csv')]
            for csv_file in csv_files:
                agg_df = pd.read_csv(z.open(csv_file), skiprows=1)
                mask_3B = agg_df['Hour'].astype(str).str.lower().str.contains('3b')
                agg_df = agg_df[~mask_3B]
                agg_df["Hour"] = agg_df['Hour'].astype(int)
                agg_df["Hour"] = agg_df["Hour"]-1
                agg_df = agg_df[agg_df['Hour'] != 2]
                agg_df['Date'] = pd.to_datetime(agg_df['Date'], dayfirst=True)
                df_list.append(agg_df)
                
                
        if not df_list:
            raise ValueError(f"No aggregator CSV found in {self.aggregator_zip}")
        combined = pd.concat(df_list, ignore_index=True)
        # Possibly sort or drop duplicates
        combined.sort_values(['Date','Hour','Sale/Purchase','Price'], inplace=True)
        return combined
 
    def get_p_id0_first_vwap(self, delivery_start_ts):
        """
        p_id0 = FIRST VWAP from the precomputed dict self._all_data_dict.
        No fallback logic: we assume the key is present in the dict.
        """
        df_res = self._all_data_dict[delivery_start_ts]
        return df_res['vwap'].iloc[0]  # first row's vwap
    
    
    def transform_supply_curve(self, df_hour, p_min=-500.0):
        """
        Based on aggregator data for a single day+hour.
        e.g. columns ["Sale/Purchase","Price","Volume"]
        Return (price_range, transformed_volumes).
        """
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
        """
        central difference around dem_implied = supply^-1(p_id0).
        """
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
        1) We already have self._all_data_dict from init.
        2) Loop over aggregator data by date/hour, transform supply, 
           get p_id0 from the FIRST VWAP, compute MO, store results.
        """
        
        self.merit_order_slopes = {}
        grouped = self.aggregated_curves.groupby(['Date','Hour'])
        
        for (the_date, the_hour), df_hour in grouped:
            # build a tz-aware timestamp
            dt_cet = (pd.Timestamp(the_date) + pd.Timedelta(hours=the_hour)).tz_localize("Europe/Brussels")
            
            if dt_cet not in self._all_data_dict:
                continue
        
            price_range, transf_vols = self.transform_supply_curve(df_hour, p_min=p_min)
            p_id0 = self.get_p_id0_first_vwap(dt_cet)  # FIRST vwap from the precomputed dict
            mo_dict = self.compute_merit_order_slope(price_range, transf_vols, p_id0)
            self.merit_order_slopes[dt_cet] = mo_dict
            
    def merge_merit_order_slopes(self, df):
        """
        Merge the precomputed merit order slopes into df, 
        based on the 'delivery_start' timestamp.
        """
        # 1) Convert dictionary to DataFrame
        df_mo_slopes = (
            pd.DataFrame
              .from_dict(self.merit_order_slopes, orient='index')
              .rename(columns={500: 'mo_slope_500',
                               1000: 'mo_slope_1000',
                               2000: 'mo_slope_2000'})
        )
        df_mo_slopes.index.name = 'delivery_start'
    
        # 2) Merge
        merged_df = df.merge(df_mo_slopes, on='delivery_start', how='left')
        return merged_df
    
    def merge_res_forecasts(self, df, res_csv):
        # 1. read raw file (skip 2nd header line that ENTSO-E often inserts)
        res_df = pd.read_csv(res_csv, skiprows=[1])

        res_df["Date (CET)"] = (
            res_df["Date (CET)"]
            .str.strip("[]")                                 
            .pipe(pd.to_datetime, format="%d/%m/%Y %H:%M", errors="coerce")
            .dt.tz_localize("Europe/Brussels", ambiguous="infer")
        )
    
        # 3. filter the year of interest and drop the duplicated 02:00 hour (CET->CEST switch)
        res_df = res_df[
            (res_df["Date (CET)"].dt.year == self.target_year)
            & (res_df["Date (CET)"].dt.hour != 2)
        ]
    
        # 4. build combined wind forecast
        res_df["WIND (FC)"] = (
            res_df["ONSHORE WIND (FC)"].fillna(0)
            + res_df["OFFSHORE WIND (FC)"].fillna(0)
        )
    
        # 5. select & rename the two RES forecast columns
        res_df = res_df[["Date (CET)", "SOLAR (FC)", "WIND (FC)"]].rename(
            columns={
                "Date (CET)": "delivery_start",
                "SOLAR (FC)": "solar_forecast",
                "WIND (FC)": "wind_forecast",
            }
        ).set_index("delivery_start")
    
        # 6. merge into your existing frame
        merged_df = df.merge(res_df, on="delivery_start", how="left")
    
        return merged_df

    @staticmethod
    def create_weekday_dummies(delivery_start_times):
        """
        Create weekday dummy variables (MON, SAT, SUN) for each delivery start time
        
        Args:
            delivery_start_times: Array/pd.Series of timestamps
            
        Returns:
            Dictionary with 'MON', 'SAT', 'SUN' arrays
        """
        weekdays = delivery_start_times.dt.weekday
        return {
            'MON': (weekdays == 0).astype(int),
            'SAT': (weekdays == 5).astype(int),
            'SUN': (weekdays == 6).astype(int)
        }
    
    

    @staticmethod
    def prepare_hourly_arrays(df):
        """
        Split data into 24 arrays, one for each delivery hour
        
        Args:
            df: DataFrame containing all processed data
            
        Returns:
            Dictionary with 24 numpy structured arrays
        """
        hourly_data = {}
        for hour in range(24):
            hour_df = df[df['delivery_hour'] == hour]
            
            # Convert to structured numpy array
            dtype = [
                ('vwap_changes', 'float64'),
                ('alpha', 'int64'),
                ('MON', 'int64'),
                ('SAT', 'int64'), 
                ('SUN', 'int64'),
                ('f_TTD', 'float64'),
                # Add other variables here
            ]
            
            hourly_data[hour] = np.array(
                [tuple(x) for x in hour_df.values],
                dtype=dtype
            )
        
        return hourly_data
    
    def prepare_gamlss_data(self, product_type='hourly', start_hours=3):
        # Get aggregated data
        agg_df = self.aggregate_extracted_data(product_type, start_hours+1)
        agg_df = agg_df.reset_index().rename(columns={'index': 'bin_timestamp'})
        
        agg_df['TTD_bins'] = (
            (agg_df['delivery_start'] - agg_df['bin_timestamp'])
            .dt.total_seconds() / (5*60.0)
        )
        agg_df['f_TTD'] = 1.0 / np.sqrt(1.0 + agg_df['TTD_bins'])
        
        agg_df['da_price'] = agg_df['delivery_start'].apply(self._get_auction_spot_price)
    
        weekdays = agg_df['delivery_start'].dt.weekday  # Monday=0, Sunday=6
        agg_df['MON'] = (weekdays == 0).astype(int)
        agg_df['SAT'] = (weekdays == 5).astype(int)
        agg_df['SUN'] = (weekdays == 6).astype(int)
    
        merged_df = self.merge_merit_order_slopes(agg_df)
        
        merged_df = self.merge_res_forecasts(merged_df, self.wind_data)

    
        merged_df['delivery_date'] = merged_df['delivery_start'].dt.normalize()
        merged_df['delivery_hour'] = merged_df['delivery_start'].dt.hour
    
        merged_df = self.add_absolute_value_columns(merged_df, ['vwap_changes'])
        merged_df = self.add_lagged_variables(
            merged_df,
            {
                'vwap': [1], 
                'vwap_changes': [1, 2, 3], 
                'alpha': list(range(1, 13)),
                'abs_vwap_changes': list(range(1, 13))
            }
        )
        merged_df["da-id"] = abs(merged_df['da_price'] - merged_df['vwap_lag1'])
    
        # Restrict to TTD_bins <= 36
        merged_df = merged_df[merged_df['TTD_bins'] <= 36].copy()
    
        # Create integer bins from TTD_bins
        merged_df['TTD_bins_int'] = np.floor(merged_df['TTD_bins']).astype(int)
    
        # Create dummy columns for each TTD_bins_int
        ttd_dummies = pd.get_dummies(merged_df['TTD_bins_int'], prefix='TTD_bin').astype(int)
        merged_df = pd.concat([merged_df, ttd_dummies], axis=1)
    
        return merged_df
    
    
    @staticmethod
    def add_lagged_variables(df, lag_dict):
        """
        Add lagged versions of specified columns to the DataFrame, inserting them immediately
        to the right of the original column.
        
        """
        # Ensure DataFrame is sorted appropriately
        df = df.sort_values(by=['delivery_start', 'bin_timestamp']).copy()
        
        # Iterate over a copy of the current columns to avoid issues during insertion.
        for col in list(df.columns):
            if col in lag_dict:
                count_inserted = 0
                for lag in lag_dict[col]:
                    new_col_name = f"{col}_lag{lag}"
                    # Compute insertion index: right after the original column plus any new columns already added.
                    insertion_index = df.columns.get_loc(col) + 1 + count_inserted
                    new_series = df.groupby('delivery_start')[col].shift(lag)
                    # If the column already exists, remove it to avoid ValueError.
                    if new_col_name in df.columns:
                        df.drop(columns=[new_col_name], inplace=True)
                    df.insert(insertion_index, new_col_name, new_series)
                    count_inserted += 1
        return df

    @staticmethod
    def add_absolute_value_columns(df, columns):
        """
        Add absolute value columns for the specified columns, inserting them immediately
        to the right of the original columns.

        """
        for col in columns:
            if col in df.columns:
                insertion_index = df.columns.get_loc(col) + 1
                new_col = f"abs_{col}"
                # If the column already exists, drop it before inserting.
                if new_col in df.columns:
                    df.drop(columns=[new_col], inplace=True)
                df.insert(insertion_index, new_col, df[col].abs())
        return df
    
    def plot_price_changes_histogram(self, product_type='hourly', start_hours = 3, delivery_hour=None, bins=51):
        """
        Plot a histogram of the VWAP price changes (from the last start_hours hours before delivery)
        for the specified product type and, optionally, for a specific delivery hour.
        The histogram uses symmetric bins such that one bin is centered at 0.
        In the bin containing zero, the portion corresponding to intervals with alpha == 0 is shown in red.
        
        Parameters:
            product_type (str): One of ['hourly', 'half_hourly', 'quarterly'].
            delivery_hour (int, optional): If specified (0-23), only use data where the delivery_start hour equals this.
            bins (int): Number of bins (must be odd to have a center bin at 0; default is 51).
        """
        # Ensure odd number of bins so that one bin is centered at 0.
        if bins % 2 == 0: 
            bins += 1

        agg_df = self.aggregate_extracted_data(product_type, start_hours)
        if delivery_hour is not None:
            agg_df = agg_df[agg_df['delivery_start'].dt.hour == delivery_hour]
        df_changes = agg_df[['vwap_changes', 'alpha']].dropna()
        changes = df_changes['vwap_changes'].values
        alphas = df_changes['alpha'].values
        
        # Create symmetric bins about zero.
        max_val = max(abs(changes.min()), abs(changes.max()))
        bin_edges = np.linspace(-max_val, max_val, bins+1)
        counts, _ = np.histogram(changes, bins=bin_edges)
        bin_width = bin_edges[1] - bin_edges[0]
        bin_centers = bin_edges[:-1] + bin_width / 2
        
        # Identify the bin that contains 0.
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

    def fit_distributions_to_price_changes(self, product_type='hourly', start_hours=3 ,delivery_hour=None, bins=51, plot=True):
        """
        Fit a Johnson's SU, Student's t, and Normal distribution to the VWAP price changes
        (from the last 3 hours before delivery) for the specified product type and delivery hour,
        omitting intervals where alpha == 0. Also compute goodness-of-fit via the KS test.
        
        Parameters:
            product_type (str): One of ['hourly', 'half_hourly', 'quarterly'].
            delivery_hour (int, optional): If specified, only use data for that delivery hour.
            bins (int): Number of bins (must be odd; default is 51).
            plot (bool): If True, plot the histogram with fitted PDFs.
        
        Returns:
            dict: Dictionary with keys 'johnsonsu', 'student_t', and 'normal'.
                  Each value is a dict with keys 'params', 'ks_statistic', and 'p_value'.
        """
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

    def plot_data(self, delivery_start, product_type='hourly', start_hours=3):
        """
        Plot the VWAP, its price changes, and the alpha indicator for a specific delivery start.
        The data plotted is restricted to the last 3 hours before delivery.
        
        Parameters:
            delivery_start (pd.Timestamp): The Delivery Start time (in CET) to plot.
            product_type (str): One of ['hourly', 'half_hourly', 'quarterly'].
            start_hours (float): How many hours before delivery to include (default is 3).
        """
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
        """
        Plot the total traded volume for each product type (quarterly, half_hourly, hourly)
        per delivery hour. The traded volume is summed over all trades (using 'Quantity (MW)').
        """
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
        
    def plot_pairwise_correlation_of_price_changes(self, product_type='hourly', start_hours = 3):
        """
        Compute and plot a 24x24 heatmap of pairwise Pearson correlations between
        the average "vwap_changes" time series of hourly products (last 3 hours before delivery)
        for each delivery hour (0-23).
        
        For each product in the aggregated data, a new column "time_to_delivery" is computed
        as the time difference (in minutes) between the delivery start and the execution time.
        Then, for each delivery hour, the vwap_changes are averaged over each time bin (of 5 minutes)
        to create a time series (from 180 minutes to 0). Finally, the correlation between these
        24 time series is computed and plotted as a heatmap.
        """
        agg_df = self.aggregate_extracted_data(product_type, start_hours)
     
     
        # Filter: only use data from 2021, but omit January 1, 2021.
        agg_df = agg_df[(agg_df['delivery_start'].dt.year == 2021) &
                        ~((agg_df['delivery_start'].dt.month == 1) & (agg_df['delivery_start'].dt.day == 1))]
                
        
        agg_df["delivery_hour"] = agg_df["delivery_start"].dt.hour
    
        # For each hour (0..23), gather all vwap_changes
        changes_by_hour = {}
        for hour in range(24):
            hour_subset = agg_df.loc[agg_df["delivery_hour"] == hour, "vwap_changes"]
            changes_by_hour[hour] = hour_subset.values


        corr_matrix = np.full((24, 24), np.nan)
        
        for h1 in range(24):
            for h2 in range(24):
                arr1 = changes_by_hour[h1]
                arr2 = changes_by_hour[h2]
                
                # If either is empty, correlation remains NaN
                if len(arr1) > 1 and len(arr2) > 1:
                    # Flatten and compute correlation
                    corr_val = np.corrcoef(arr1, arr2)[0, 1]
                    corr_matrix[h1, h2] = corr_val
                else:
                    corr_matrix[h1, h2] = np.nan
    
        # 4) Plot the heatmap using seaborn
        plt.figure(figsize=(10, 8))
        ax = sns.heatmap(corr_matrix, vmin=-1, vmax=1, center=0,
                         cmap=sns.diverging_palette(220, 20, as_cmap=True),
                         annot=True, fmt=".2f", square=True)
        ax.set_title(f"Pairwise Correlation of VWAP Changes Across Delivery Hours\n(last {start_hours}h of hourly products)")
        ax.set_xlabel("Delivery Hour")
        ax.set_ylabel("Delivery Hour")
        plt.tight_layout()
        plt.show()

# === Example Usage ===
if __name__ == '__main__':

    
     # 1) Instantiate your processor
    lob_2024 = LimitOrderBookProcessor(
        zip_file="Continuous_Trades-NL-2021.zip",
        auction_spot_csv='auction_spot_prices_netherlands_2021.csv',
        aggregator_zip = "auction_aggregated_curves_netherlands_2021.zip",
        wind_data = 'res_data.csv', solar_data = 'res_data.csv'
        
        
    )



    # 3) Compute & store MO slopes for all date/hour in 2021:
    datas_2023 = lob_2023.prepare_gamlss_data()
    datas_2023.to_csv('df_2023.csv' , index=False)
    
    # 1) Instantiate your processor
    lob_2024 = LimitOrderBookProcessor(
        zip_file="Continuous_Trades-NL-2024.zip",
        auction_spot_csv='auction_spot_prices_netherlands_2024.csv',
        aggregator_zip = "auction_aggregated_curves_netherlands_2024.zip",
        wind_data = 'res_data.csv', solar_data = 'res_data.csv'
        
    )



    # 3) Compute & store MO slopes for all date/hour in 2021:
    datas_2024 = lob_2024.prepare_gamlss_data()
    
    
    


    # # Plot histogram of VWAP price changes for hourly products delivered at 16:00.
    # processor.plot_price_changes_histogram(product_type='hourly', delivery_hour=16, bins=51)
    
    # # Fit distributions for hourly products delivered at 16:00.
    # results = processor.fit_distributions_to_price_changes(product_type='hourly', delivery_hour=16, bins=51, plot=True)
    # print("Goodness-of-Fit Results:")
    # for dist_name, res in results.items():
    #     print(f"\nDistribution: {dist_name.capitalize()}")
    #     print(f"Fitted Parameters: {res['params']}")
    #     print(f"KS Statistic: {res['ks_statistic']:.4f}")
    #     print(f"p-value: {res['p_value']:.4f}")
    
    # # Plot total traded volume per delivery hour for quarterly, half_hourly, and hourly products.
    # processor.plot_total_traded_volume_per_delivery_hour()
    
    # # Plot the VWAP, VWAP changes, and alpha for the product with Delivery Start 2021-01-02 15:00:00.
    # product_delivery = pd.Timestamp("2021-01-02 15:00:00", tz='Europe/Berlin')
    # print(f"\nPlotting data for Delivery Start: {product_delivery}")
    # processor.plot_data(product_delivery, product_type='hourly', start_hours=3)
    
    # processor.plot_pairwise_correlation_of_price_changes(product_type='hourly', start_hours=3)
    
    

