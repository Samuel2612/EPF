import pandas as pd
import seaborn as sns  
from zipfile import ZipFile
import matplotlib.pyplot as plt
import scipy.stats as st
import numpy as np

class LimitOrderBookData:
    def __init__(self, zip_file, time_interval='5min'):
        """
        Parameters:
            zip_file (str): Path to the zip file containing CSV files.
            time_interval (str): Resampling interval (default is 5 minutes).
        """
        self.zip_file = zip_file
        self.time_interval = time_interval
        self.data = self._load_data()
        self._preprocess_data()
        self.extracted_data = self.extract_all_data()

    def _load_data(self):
        """Load all CSV files from the zip file (ignoring any nested zip files)."""
        df_list = []
        with ZipFile(self.zip_file, 'r') as z:
            # Only consider CSV files
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
          - Converting date columns from UTC to CET (Europe/Berlin).
          - Filtering out self-trades (keeping only rows where Is Self Trade == 'N').
          - Computing product duration in minutes from "Delivery Start" and "Delivery End".
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
        self.data = self.data[self.data['Is Self Trade'] == 'N']
        
        # Compute product duration (in minutes)
        self.data['ProductDuration'] = (
            self.data['Delivery End'] - self.data['Delivery Start']
        ).dt.total_seconds() / 60

    def _compute_vwap(self, group):
        """
        Compute the volume-weighted average price (VWAP) for a group.
        Uses 'Price' and 'Quantity (MW)' columns.
        """
        total_qty = group['Quantity (MW)'].sum()
        if total_qty > 0:
            return (group['Price'] * group['Quantity (MW)']).sum() / total_qty
        return None

    def get_vwaps(self, product_type, delivery_start):
        """
        Compute VWAPs (volume weighted average prices) over fixed intervals for a specific product.
        
        For a given product (identified by its product type and its Delivery Start time),
        the trade period is assumed to be from 14:00 CET the day before the product's
        Delivery Start until the Delivery Start time.
        
        Parameters:
            product_type (str): 'hourly', 'half_hourly', or 'quarterly'
            delivery_start (pd.Timestamp): The product's Delivery Start time (in CET),
                e.g. pd.Timestamp("2021-01-02 15:00:00", tz='Europe/Berlin')
        
        Returns:
            pd.Series: VWAP for each interval, forward-filled where no trades occurred.
                       Intervals with no prior trades remain as NaN.
        """
        # Mapping from product type to expected duration in minutes.
        duration_mapping = {'hourly': 60, 'half_hourly': 30, 'quarterly': 15}
        target_duration = duration_mapping[product_type]
        
        # Compute market open time as 14:00 CET the day before delivery_start, ensuring tz-awareness.
        market_open = (pd.Timestamp(delivery_start.date()) - pd.Timedelta(days=1) +
                       pd.Timedelta(hours=14)).tz_localize(delivery_start.tz)
        
        # Filter trades for the given product: matching Delivery Start and product duration.
        df_product = self.data[
            (abs(self.data['ProductDuration'] - target_duration) < 1e-6) &
            (self.data['Delivery Start'] == delivery_start)
        ].copy()
        
        # Filter trades executed between market open and the product's Delivery Start.
        df_product = df_product[
            (df_product['Execution time'] >= market_open) & 
            (df_product['Execution time'] <= delivery_start)
        ]
        
        # Set 'Execution time' as the index for resampling.
        df_product.set_index('Execution time', inplace=True)
        
        # Resample over the defined interval and compute the VWAP.
        vwap_series = df_product.resample(self.time_interval).apply(self._compute_vwap)
        # Forward-fill intervals with no trades.
        vwap_series_filled = vwap_series.ffill()
        return vwap_series_filled

    def get_alpha(self, product_type, delivery_start):
        """
        Compute the boolean indicator 'alpha' for a given product.
        
        For the specified product (by product type and Delivery Start), alpha is 1 for an interval if
        at least one trade was executed between the market open (14:00 CET the day before Delivery Start)
        and the Delivery Start time; otherwise, it is 0.
        
        Parameters:
            product_type (str): 'hourly', 'half_hourly', or 'quarterly'
            delivery_start (pd.Timestamp): The product's Delivery Start time (in CET).
        
        Returns:
            pd.Series: A binary time series (1 if a trade occurred in the interval, 0 otherwise).
        """
        duration_mapping = {'hourly': 60, 'half_hourly': 30, 'quarterly': 15}
        target_duration = duration_mapping[product_type]
        
        market_open = (pd.Timestamp(delivery_start.date()) - pd.Timedelta(days=1) +
                       pd.Timedelta(hours=14)).tz_localize(delivery_start.tz)
        
        df_product = self.data[
            (abs(self.data['ProductDuration'] - target_duration) < 1e-6) &
            (self.data['Delivery Start'] == delivery_start)
        ].copy()
        
        df_product = df_product[
            (df_product['Execution time'] >= market_open) & 
            (df_product['Execution time'] <= delivery_start)
        ]
        df_product.set_index('Execution time', inplace=True)
        
        # Count trades per interval and map to 1 if any trade occurred.
        trade_counts = df_product.resample(self.time_interval).size()
        alpha = trade_counts.apply(lambda x: 1 if x > 0 else 0)
        return alpha

    def get_vwap_changes(self, product_type, delivery_start):
        """
        Compute the price changes of the VWAP per interval.
        Returns the difference in VWAP between consecutive intervals.
        """
        vwap_series = self.get_vwaps(product_type, delivery_start)
        vwap_changes = vwap_series.diff()
        return vwap_changes
    
    def extract_all_data(self, product_type='hourly'):
        """
        Extract for each unique Delivery Start (matching the specified product type),
        a DataFrame with columns 'vwap', 'vwap_changes', and 'alpha' over the last 3 hours
        before delivery, resampled at self.time_interval.
    
        Parameters:
            product_type (str): One of ['hourly', 'half_hourly', 'quarterly'].
    
        Returns:
            dict:
                Keys are unique Delivery Start timestamps (for the given product type).
                Values are DataFrames indexed by 'Execution time' (resampled),
                with columns:
                    - 'vwap'
                    - 'vwap_changes'
                    - 'alpha'
        """
        # 1) Map product types to expected duration (in minutes)
        duration_map = {
            'hourly': 60,
            'half_hourly': 30,
            'quarterly': 15
        }
        if product_type not in duration_map:
            raise ValueError("Invalid product_type. Choose from 'hourly', 'half_hourly', or 'quarterly'.")
    
        target_duration = duration_map[product_type]
    
        # 2) Filter data for this product type
        #    (We assume 'ProductDuration' is computed in _preprocess_data)
        df_filtered = self.data[abs(self.data['ProductDuration'] - target_duration) < 1e-6].copy()
    
        # Dictionary to store the results for each Delivery Start
        results = {}
    
        # 3) Get all unique Delivery Start timestamps for this product type
        unique_starts = df_filtered['Delivery Start'].dropna().unique()
        unique_starts.sort()  # sort them chronologically
    
        for start_time in unique_starts:
            # 4) Subset the data to this Delivery Start
            df_sub = df_filtered[df_filtered['Delivery Start'] == start_time].copy()
            if df_sub.empty:
                continue
    
            # 5) We only want the last 3 hours before this delivery start
            last_3h_start = start_time - pd.Timedelta(hours=3)
            
            # 6) Filter for Execution times in [last_3h_start, start_time]
            df_sub = df_sub[
                (df_sub['Execution time'] >= last_3h_start) &
                (df_sub['Execution time'] <= start_time)
            ].copy()
    
            if df_sub.empty:
                continue
    
            # 7) Set 'Execution time' as index, then resample
            df_sub.set_index('Execution time', inplace=True)
            df_sub.sort_index(inplace=True)
    
            # 8) Resample and compute VWAP
            vwap_series = df_sub.resample(self.time_interval).apply(self._compute_vwap).ffill()
    
            # 9) Alpha: 1 if any trade occurred in the interval, else 0
            alpha_series = df_sub.resample(self.time_interval).size().apply(lambda x: 1 if x > 0 else 0)
    
            # 10) VWAP changes as difference between consecutive intervals
            vwap_changes = vwap_series.diff()
    
            # 11) Build a DataFrame for this Delivery Start
            result_df = pd.DataFrame({
                'vwap': vwap_series,
                'vwap_changes': vwap_changes,
                'alpha': alpha_series
            })
    
            # 12) Store it in the dictionary keyed by this Delivery Start
            results[start_time] = result_df
    
        return results

    def plot_data(self, product_type, delivery_start, start_hours=None):
        """
        Plot the VWAP, its price changes, and the alpha indicator over time as step plots.
        
        Parameters:
            product_type (str): 'hourly', 'half_hourly', or 'quarterly'
            delivery_start (pd.Timestamp): The product's Delivery Start time (in CET).
            start_hours (float, optional): How many hours before delivery to start the plots.
                                           If not provided, the full period from market open is used.
        """
        vwap_series = self.get_vwaps(product_type, delivery_start)
        vwap_changes = self.get_vwap_changes(product_type, delivery_start)
        alpha_series = self.get_alpha(product_type, delivery_start)

        # If start_hours is specified, subset the series to only include data after that time.
        if start_hours is not None:
            start_time = delivery_start - pd.Timedelta(hours=start_hours)
            vwap_series = vwap_series.loc[vwap_series.index >= start_time]
            vwap_changes = vwap_changes.loc[vwap_changes.index >= start_time]
            alpha_series = alpha_series.loc[alpha_series.index >= start_time]
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        
        # Step plot for VWAP.
        axes[0].step(vwap_series.index, vwap_series, where='post', linestyle='-', label='VWAP')
        axes[0].set_title('VWAP')
        axes[0].set_ylabel('Price')
        axes[0].grid(True)
        
        # Step plot for VWAP changes.
        axes[1].step(vwap_changes.index, vwap_changes, where='post', linestyle='-', color='orange', label='VWAP Changes')
        axes[1].set_title('VWAP Price Changes')
        axes[1].set_ylabel('Price Change')
        axes[1].grid(True)
        
        # Step plot for alpha.
        axes[2].step(alpha_series.index, alpha_series, where='post', color='green', label='Alpha')
        axes[2].set_title('Alpha (Trade Indicator)')
        axes[2].set_ylabel('Alpha')
        axes[2].set_xlabel('Time')
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.show()
        
    
    
    def plot_price_changes(self, bins= 100):
        """
        Plot a histogram of the VWAP price changes over the entire dataset.
        In the bin that contains 0, the portion corresponding to intervals with alpha=0
        is shown in a different color.
        
        Parameters:
            bins (int): Number of bins for the histogram.
        """
        # Extract the full dataset and drop NaN differences.
        df = self.extracted_data.copy()
        df_changes = df[['vwap_changes', 'alpha']].dropna()
        changes = df_changes['vwap_changes'].values
        alphas = df_changes['alpha'].values

        # Compute histogram counts and bin edges.
        counts, bin_edges = np.histogram(changes, bins=bins)
        bin_width = bin_edges[1] - bin_edges[0]
        bin_centers = bin_edges[:-1] + bin_width / 2

        # Identify the bin that contains 0.
        zero_bin_index = None
        for i in range(len(bin_edges)-1):
            if bin_edges[i] <= 0 < bin_edges[i+1]:
                zero_bin_index = i
                break

        # Compute histogram counts for observations with alpha == 0.
        counts_alpha0, _ = np.histogram(changes[alphas == 0], bins=bin_edges)

        plt.figure(figsize=(10, 6))
        for i in range(len(counts)):
            if i == zero_bin_index:
                # For the zero bin, plot stacked bars:
                count_alpha0 = counts_alpha0[i]
                count_other = counts[i] - count_alpha0
                # Plot the base bar for intervals with alpha != 0.
                plt.bar(bin_centers[i], count_other, width=bin_width, color='blue',
                        edgecolor='black', align='center')
                # Plot on top the portion corresponding to alpha == 0 in a different color.
                plt.bar(bin_centers[i], count_alpha0, width=bin_width, color='red',
                        edgecolor='black', align='center', bottom=count_other)
            else:
                plt.bar(bin_centers[i], counts[i], width=bin_width, color='blue',
                        edgecolor='black', align='center')

        plt.xlabel("VWAP Price Changes")
        plt.ylabel("Frequency")
        plt.title("Histogram of VWAP Price Changes\n(Red portion in zero bin: alpha = 0)")
        plt.grid(True)

        # Add legend.
        import matplotlib.patches as mpatches
        blue_patch = mpatches.Patch(color='blue', label='Other intervals')
        red_patch = mpatches.Patch(color='red', label='Intervals with alpha = 0 (zero bin)')
        plt.legend(handles=[blue_patch, red_patch])
        plt.show()
        
    def fit_dist_to_price_changes(self, bins=100, plot=True):
        """
        Fit a Johnson's SU distribution to the VWAP price changes and optionally plot the fit.
        
        Parameters:
            bins (int): Number of bins for the histogram (default is 50).
            plot (bool): If True, plot the histogram with the fitted PDF.
        
        Returns:
            tuple: Fitted parameters (shape1, shape2, loc, scale) of the Johnson's SU distribution.
        """
        df = self.extracted_data.copy()
        # Omit data where alpha == 0.
        df = df[df['alpha'] != 0]
        df_changes = df['vwap_changes'].dropna()
        
        
 
        
        # Fit Johnson's SU distribution.
        params_johnsonsu = st.johnsonsu.fit(df_changes)
        # Fit Student's t distribution.
        params_t = st.t.fit(df_changes)
        # Fit Normal distribution.
        params_norm = st.norm.fit(df_changes)
        
        # Compute KS goodness-of-fit tests.
        ks_johnsonsu = st.kstest(df_changes, 'johnsonsu', args=params_johnsonsu)
        ks_t = st.kstest(df_changes, 't', args=params_t)
        ks_norm = st.kstest(df_changes, 'norm', args=params_norm)
        
        if plot:
            x = np.linspace(df_changes.min(), df_changes.max(), 1000)
            pdf_johnsonsu = st.johnsonsu.pdf(x, *params_johnsonsu)
            pdf_t = st.t.pdf(x, *params_t)
            pdf_norm = st.norm.pdf(x, *params_norm)
            
            plt.figure(figsize=(10, 6))
            plt.hist(df_changes, bins=bins, density=True, edgecolor='black', alpha=0.6, label='Data histogram')
            plt.plot(x, pdf_johnsonsu, 'r-', lw=2, label="Johnson's SU")
            plt.plot(x, pdf_t, 'g-', lw=2, label="Student's t")
            plt.plot(x, pdf_norm, 'b-', lw=2, label="Normal")
            plt.xlabel("VWAP Price Changes")
            plt.ylabel("Density")
            plt.title("Fitted Distributions for VWAP Price Changes (alpha != 0)")
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
    
    def compute_hourly_correlation(self):
        """
        1) Gather the last-3-hour price changes for each hour (0..23).
        2) Create a DataFrame with 24 columns (one per hour).
        3) Compute pairwise correlations and plot as a heatmap.
        """
        # Gather data in a dict: { hour: Series_of_changes }
        hour_data = {}
        for h in range(24):
            hour_data[h] = self.get_last_3h_price_changes(h)
    
        # Build a DataFrame where each column is the price-change distribution for that hour.
        # Rows do NOT align in time; we simply stack all changes for each hour.
        df_hours = pd.DataFrame(dict(hour_data))
        
        # Compute correlation matrix (24 x 24)
        corr_matrix = df_hours.corr()
    
        # Plot as a heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            corr_matrix,
            vmin=-1, vmax=1, center=0,
            cmap=sns.diverging_palette(220, 20, as_cmap=True),
            annot=True, fmt=".2f",
            square=True
        )
        plt.title("Correlation of Last-3h Price Changes Across Delivery Hours")
        plt.xlabel("Delivery Hour")
        plt.ylabel("Delivery Hour")
        plt.show()
    
    


# === Example Usage ===
if __name__ == '__main__':
    zip_path = "Continuous_Trades_NL_2021.zip"
    # Specify the product type: 'hourly', 'half_hourly', or 'quarterly'
    product_type = 'hourly'
    # Specify the product's Delivery Start time in CET.
    # For example, if an hourly product is delivered at 15:00 CET on January 2, 2021:
    delivery_start = pd.Timestamp("2021-01-02 04:00:00", tz='Europe/Berlin')
    
    LOB = LimitOrderBookData(zip_path)
    
    # Extract and print VWAP series.
    vwap_series = LOB.get_vwaps(product_type, delivery_start)
    print("VWAPs per interval:")
    print(vwap_series)
    
    # Extract and print Alpha indicator.
    alpha_series = LOB.get_alpha(product_type, delivery_start)
    print("\nAlpha (trade indicator) per interval:")
    print(alpha_series)
    
    # Extract and print VWAP price changes.
    vwap_changes = LOB.get_vwap_changes(product_type, delivery_start)
    print("\nVWAP Price Changes per interval:")
    print(vwap_changes)
    
    # Plot data starting from 6 hours before delivery.
    LOB.plot_data(product_type, delivery_start, start_hours=6)
    
    # Extract the complete dataset's VWAP, VWAP changes, and alpha into one DataFrame.
    full_data_df = LOB.extract_all_data()
    print("Extracted DataFrame:")
    print(full_data_df.head())
    
    # Plot a histogram of the VWAP price changes.
    LOB.plot_price_changes(bins=100)
    
    # Fit the distributions to the price changes (excluding intervals where alpha == 0) and plot.
    fit_results = LOB.fit_dist_to_price_changes(bins=100, plot=True)
    print("\nGoodness-of-Fit Results:\n" + "-"*30)
    for dist_name, result in fit_results.items():
        params = result['params']
        ks_stat = result['ks_statistic']
        p_val = result['p_value']
        print(f"\nDistribution: {dist_name.capitalize()}")
        print(f"Fitted Parameters: {params}")
        print(f"KS Statistic: {ks_stat:.4f}")
        print(f"p-value: {p_val:.4f}")
    
