import pandas as pd
from zipfile import ZipFile
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy import stats
import seaborn as sns

class LimitOrderBookProcessor:
    def __init__(self, 
                 zip_file, 
                 time_interval='5min', 
                 auction_spot_csv='auction_spot_prices_netherlands_2021.csv'):
        """
        Parameters:
            zip_file (str): Path to the zip file containing CSV files of trades.
            time_interval (str): Resampling interval (default is 5 minutes).
            auction_spot_csv (str): Path to the CSV file containing auction spot prices.
        """
        self.zip_file = zip_file
        self.time_interval = time_interval
        
        # 1) Load your continuous-trades data.
        self.data = self._load_data()
        self._preprocess_data()
        
        # 2) Load Auction Spot Prices from the CSV, handling DST hour 3a/3b.
        self.auction_spot_prices = self._load_auction_spot_csv(auction_spot_csv)

    def _load_data(self):
        """Load all CSV files from the zip file (ignoring nested zips)."""
        df_list = []
        with ZipFile(self.zip_file, 'r') as z:
            csv_files = [f for f in z.namelist() if f.endswith('.csv')]
            for csv_file in csv_files:
                df = pd.read_csv(z.open(csv_file))
                df_list.append(df)
        if not df_list:
            raise ValueError("No CSV files found in the zip file.")
        return pd.concat(df_list, ignore_index=True)

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
            self.data['Delivery Start'], utc=True
        ).dt.tz_convert(cet_tz)
        self.data['Delivery End'] = pd.to_datetime(
            self.data['Delivery End'], utc=True
        ).dt.tz_convert(cet_tz)
        self.data['Execution time'] = pd.to_datetime(
            self.data['Execution time'], utc=True
        ).dt.tz_convert(cet_tz)
        
        # Keep only non self-trades.
        self.data = self.data[self.data['Is Self Trade'] != 'Y']
        # Keep only rows with unique TradeID.
        self.data = self.data.drop_duplicates(subset='TradeID', keep='first')
        
        # Compute product duration (in minutes).
        self.data['ProductDuration'] = (
            self.data['Delivery End'] - self.data['Delivery Start']
        ).dt.total_seconds() / 60
        
        # Keep only data from 2021 (based on 'Delivery Start').
        self.data = self.data[self.data['Delivery Start'].dt.year == 2021]

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
        

        return df

    def _get_auction_spot_price(self, delivery_start: pd.Timestamp) -> float:
        """
        Return the auction spot price for the product's delivery_start from the loaded CSV.
        """
        day = delivery_start.floor("D")
        hour = delivery_start.hour
        
        return self.auction_spot_prices.loc[day].iloc[hour]

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
        all_hours_2021 = pd.date_range(
            start="2021-01-01 00:00:00", 
            end="2021-12-31 23:00:00", 
            freq="h", 
            tz="Europe/Berlin"
        )
        
        # 4) Number of intervals in the last <start_hours> hours, given self.time_interval
        interval_minutes = pd.Timedelta(self.time_interval).total_seconds() / 60.0
        num_intervals = int(start_hours * 60 / interval_minutes)
        
        results = {}
        
        for start_time in all_hours_2021:
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
                # === We do have trades for this hour ===
                df_sub.set_index('Execution time', inplace=True)
                df_sub.sort_index(inplace=True)
                
                # Resample => compute VWAP and alpha
                vwap_series = df_sub.resample(self.time_interval).apply(self._compute_vwap)
                alpha_series= df_sub.resample(self.time_interval).size().apply(lambda x: 1 if x>0 else 0)
                
                # Price changes
                vwap_changes = vwap_series.diff().fillna(0)
                
                # Reindex to full time window
                vwap_series = vwap_series.reindex(full_index)
                alpha_series= alpha_series.reindex(full_index, fill_value=0)
                vwap_changes = vwap_changes.reindex(full_index)
                
                # Forward-fill from last known VWAP
                vwap_series = vwap_series.ffill()
                vwap_changes = vwap_changes.ffill()
                
                # If entire series is NaN => no trades at all in the window
                if vwap_series.isna().all():
                    fallback_price = self._get_auction_spot_price(start_time)
                    if np.isnan(fallback_price):
                        fallback_price = -9999
                    vwap_series.fillna(fallback_price, inplace=True)
                    vwap_changes = 0
                    alpha_series = 0
                    
                # If the earliest bins are NaN, fill with fallback
                if pd.isna(vwap_series.iloc[0]):
                    fallback_price = self._get_auction_spot_price(start_time)
                    if np.isnan(fallback_price):
                        fallback_price = -9999
                    vwap_series.fillna(fallback_price, inplace=True)
                    vwap_changes.fillna(0, inplace=True)
                
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
        if dfs:
            return pd.concat(dfs)
        else:
            return pd.DataFrame(columns=['vwap', 'vwap_changes', 'alpha', 'delivery_start'])

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
    zip_path = "Continuous_Trades_NL_2021.zip"
    processor = LimitOrderBookProcessor(zip_path)
    all_data = processor.extract_all_data()
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
