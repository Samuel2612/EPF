import pandas as pd
from entsoe import EntsoePandasClient
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import brentq
import statsmodels.api as sm

class DataCollection:
    """
    Class to fetch day-ahead and actual wind+solar generation data for
    the Netherlands (NL), and combine them into one CSV file.
    """

    def __init__(self, api_key, country_code, start, end):
        """
        :param api_key:       (str) Your ENTSO-E API key
        :param country_code:  (str) e.g. 'NL'
        :param start:         (pd.Timestamp) UTC start time
        :param end:           (pd.Timestamp) UTC end time
        """
        self.client = EntsoePandasClient(api_key=api_key)
        self.country_code = country_code
        self.start = start
        self.end = end

    def fetch_combined_wind_solar_forecast(self):
        """
        Fetch combined day-ahead wind & solar forecasts (onshore, offshore, solar)
        in a single DataFrame by setting psr_type=None.
        """
        print("Fetching day-ahead wind & solar forecast (combined)...")
        return self.client.query_wind_and_solar_forecast(
            self.country_code,
            start=self.start,
            end=self.end,
            psr_type=None
        )

    def fetch_combined_actual_generation(self):
        """
        Fetch combined actual wind & solar generation (onshore, offshore, solar)
        in a single DataFrame by setting psr_type=None.
        """
        print("Fetching actual wind & solar generation (combined)...")
        return self.client.query_generation(
            self.country_code,
            start=self.start,
            end=self.end,
            psr_type=None
        )

    def run_collection(self):
        """
        1) Fetch combined forecast
        2) Fetch combined actual generation
        3) Merge them into one DataFrame
        4) Save to a single CSV file
        """
        try:
            # 1) Fetch forecast
            forecast_df = self.fetch_combined_wind_solar_forecast()
            print("\nForecast DataFrame:\n", forecast_df)

            # 2) Fetch actual
            actual_df = self.fetch_combined_actual_generation()
            print("\nActual DataFrame:\n", actual_df)

            # 3) Merge them
            # Option A: Use join with suffixes
            merged_df = forecast_df.join(actual_df, how='outer', lsuffix='_forecast', rsuffix='_actual')

            # Alternatively, you could do:
            # merged_df = pd.concat(
            #     [forecast_df.add_suffix('_forecast'), actual_df.add_suffix('_actual')],
            #     axis=1
            # )

            print("\nMerged DataFrame (Forecast + Actual):\n", merged_df)

            # 4) Save merged to a single CSV
            merged_df.to_csv("NL_wind_solar_forecast_and_actual.csv")
            print("\nData saved to 'NL_wind_solar_forecast_and_actual.csv'!")

        except Exception as e:
            print("An error occurred during data collection or merging:", e)
            
            
class RollingRegression:
    """
    Performs rolling-window OLS to predict P_ID, 
    supporting either:
      - a linear model
      - a quadratic model (feature + feature^2, no cross-terms).
    """

    def __init__(self, df: pd.DataFrame):
        """
        We assume df has:
          - 'P_ID' (the dependent variable)
          - some set of feature columns (e.g. W_delta, S_delta, P_DA, etc.)
        We'll define those feature columns in prepare_data().
        """
        self.df = df.copy()
        self.X_vars = []
        # Predictions will be stored in "P_ID_rolling_pred"

    def prepare_data(self):
        """
        Example: define X_vars, and y = df["P_ID"].
        Suppose we already computed deltas or whatever is needed.
        Adjust to match your actual column names/features.
        """
        if "P_ID" not in self.df.columns:
            raise ValueError("DataFrame has no 'P_ID' column. Cannot run regression.")
        # Let's pick an example set of features:
        self.X_vars = [
            "W_delta_min", "W_delta",
            "S_delta_min", "S_delta",
            "Wind_Actual", "Solar_Actual",
            "P_DA"
        ]
        # The dependent variable is y = P_ID. We'll handle it in each subset.
        # If there are NaNs, we'll drop them in each window.

    def _build_design_matrix(self, df_window: pd.DataFrame, model_type: str):
        """
        Build the design matrix X (and vector y) for either:
          - 'linear'
          - 'quadratic': each feature + its square, no cross-terms.
        We'll also add a constant column for OLS.
        Returns: (X, y, columns_list)
        """
        # Filter out rows with NaNs in X or y
        df_clean = df_window.dropna(subset=self.X_vars + ["P_ID"])
        if df_clean.empty:
            return None, None, []

        # Dependent variable
        y = df_clean["P_ID"].values

        # Base features
        X_base = df_clean[self.X_vars].values  # shape: (n_samples, n_features)

        if model_type.lower() == "linear":
            # X = [const, X1, X2, ..., Xn]
            X = sm.add_constant(X_base)  # shape: (n_samples, n_features+1)
            columns_list = ["const"] + self.X_vars

        elif model_type.lower() == "quadratic":
            # For each feature x_i, we create x_i^2 (no cross terms)
            # So if we have n features, final dimension is n + n + 1 = 2n+1
            squares = X_base ** 2
            # Combine base + squares
            X_full = np.hstack([X_base, squares])  # shape: (n_samples, 2*n_features)
            X = sm.add_constant(X_full)
            # Name the columns
            square_names = [f"{name}^2" for name in self.X_vars]
            columns_list = ["const"] + self.X_vars + square_names

        else:
            raise ValueError(f"Unknown model_type: {model_type}. Choose 'linear' or 'quadratic'.")

        return X, y, columns_list

    def _fit_one_model(self, df_window: pd.DataFrame, model_type: str = "linear"):
        """
        Fit OLS on the data subset (df_window) with the chosen model_type.
        Return the fitted model, or None if there's no valid data.
        """
        X, y, col_names = self._build_design_matrix(df_window, model_type=model_type)
        if X is None or y is None or len(y) == 0:
            return None
        # Fit OLS
        model = sm.OLS(y, X).fit()
        return model

    def _predict_one_hour(self, model, row, model_type: str = "linear"):
        """
        Given a fitted model and a single row (Series),
        predict P_ID for that row (if possible).
        """
        if model is None:
            return np.nan
        # Check if row has NaN in features
        if row[self.X_vars].isnull().any():
            return np.nan

        # Build the design vector for the single row
        x_base = row[self.X_vars].values.reshape(1, -1)

        if model_type.lower() == "linear":
            x_single = sm.add_constant(x_base)
        elif model_type.lower() == "quadratic":
            x_sq = x_base ** 2
            x_concat = np.hstack([x_base, x_sq])
            x_single = sm.add_constant(x_concat)
        else:
            return np.nan

        y_pred = model.predict(x_single)[0]
        return y_pred

    def rolling_forecast(self, window_hours: int, model_type: str = "linear"):
        """
        Rolling-window approach:
          - For hour i from window_hours to the end of df
          - Train on [i - window_hours, ..., i-1]
          - Predict row i
          - Store in df["P_ID_rolling_pred"]
        """
        # Ensure time-sorted
        self.df.sort_index(inplace=True)

        # Initialize predictions array
        preds = [np.nan] * len(self.df)

        for i in range(window_hours, len(self.df)):
            # Training subset
            train_start = i - window_hours
            train_end = i  # not inclusive
            df_train = self.df.iloc[train_start:train_end]

            # Fit model
            model = self._fit_one_model(df_train, model_type=model_type)

            # Predict hour i
            row_to_predict = self.df.iloc[i]
            pred_val = self._predict_one_hour(model, row_to_predict, model_type=model_type)
            preds[i] = pred_val

        self.df["P_ID_rolling_pred"] = preds
        return self.df
            
            
class TransformedSupplyCalculator:
    """
    Implements the method for constructing the transformed (inverse) supply curve:
      Sup_t^{p,-1}(z) = WS_sup_inv(z) + WS_dem_inv(q_inelastic) - WS_dem_inv(z),
    and provides:
      - a way to invert that relationship (Sup_t^p(x)) to get quantity for a given price
      - a method to compute the central-difference elasticity measure around an
        "implied demand" point, i.e. the quantity that corresponds to a reference price.
    
    References:
      - Kulakov & Ziel (2019, 2020)
      - Balardy (2022)
      - EPEX Spot data
    """

    def __init__(self,
                 supply_csv_path,
                 demand_csv_path,
                 hour_filter=None,
                 p_min=-500,
                 q_min=0,
                 q_max=50000):
        """
        Constructor. Automatically loads data from the given CSV paths
        and prepares the inverse supply/demand functions.

        :param supply_csv_path: path to EPEX aggregated supply data (CSV)
        :param demand_csv_path: path to EPEX aggregated demand data (CSV)
        :param hour_filter: if your CSV has a 'hour' column, filter to that hour
        :param p_min: The minimum price (e.g., -500 EUR/MWh from EPEX regulation)
        :param q_min: The lower bound for quantity searches (root-finding)
        :param q_max: The upper bound for quantity searches (root-finding)
        """
        self.p_min = p_min
        self.q_min = q_min
        self.q_max = q_max

        # Load supply & demand DataFrames
        self.supply_df, self.demand_df = self._load_epex_data(
            supply_csv_path, demand_csv_path, hour_filter
        )

        # Construct inverse supply/demand: WS_sup_inv(q), WS_dem_inv(q) -> price
        self.WS_sup_inv, self.WS_dem_inv = self._create_inverse_functions(
            self.supply_df, self.demand_df
        )

        # Solve for Q inelastic: the quantity where WS_dem_inv(Q) = p_min
        self.q_inelastic = self._find_inelastic_demand(
            self.WS_dem_inv, self.p_min, self.q_min, self.q_max
        )
        
        if self.q_inelastic is None:
            print("WARNING: Could not find inelastic demand quantity "
                  f"(WS_dem_inv(q) = {self.p_min} had no solution).")

    ######################################################################
    # 1) Load data
    ######################################################################
    @staticmethod
    def _load_epex_data(supply_csv_path, demand_csv_path, hour_filter=None):
        """
        Internal helper: loads the aggregated supply/demand curves from CSV.
        Sort them by 'quantity' in ascending order for consistent interpolation.
        """
        supply_df = pd.read_csv(supply_csv_path)
        demand_df = pd.read_csv(demand_csv_path)

        # If you have columns like 'hour', filter if an hour_filter was specified
        if hour_filter is not None and 'hour' in supply_df.columns:
            supply_df = supply_df[supply_df['hour'] == hour_filter]
        if hour_filter is not None and 'hour' in demand_df.columns:
            demand_df = demand_df[demand_df['hour'] == hour_filter]

        # Sort so that 'quantity' is ascending
        supply_df = supply_df.sort_values(by='quantity')
        demand_df = demand_df.sort_values(by='quantity')

        return supply_df, demand_df

    ######################################################################
    # 2) Construct inverse supply & demand: q -> p
    ######################################################################
    @staticmethod
    def _create_inverse_functions(supply_df, demand_df):
        """
        Create WS_sup_inv(q) and WS_dem_inv(q) by interpolation.
        These map a quantity (MWh) to a price (EUR/MWh).
        """
        # Supply inverse
        sup_q = supply_df['quantity'].values
        sup_p = supply_df['price'].values
        WS_sup_inv = interp1d(
            sup_q, sup_p,
            kind='linear', fill_value='extrapolate', bounds_error=False
        )

        # Demand inverse
        dem_q = demand_df['quantity'].values
        dem_p = demand_df['price'].values
        WS_dem_inv = interp1d(
            dem_q, dem_p,
            kind='linear', fill_value='extrapolate', bounds_error=False
        )

        return WS_sup_inv, WS_dem_inv

    ######################################################################
    # 3) Find the inelastic demand quantity
    ######################################################################
    @staticmethod
    def _find_inelastic_demand(WS_dem_inv, P_min, q_min, q_max):
        """
        Solve for Q in [q_min, q_max] such that WS_dem_inv(Q) = P_min.
        That is f(Q) = WS_dem_inv(Q) - P_min = 0, via brentq.
        """
        def f(q):
            return WS_dem_inv(q) - P_min

        try:
            return brentq(f, q_min, q_max)
        except ValueError:
            return None

    ######################################################################
    # 4) The transformed inverse supply: Sup_t^{p,-1}(z) = ...
    ######################################################################
    def transformed_inverse_supply(self, z):
        """
        Evaluate Sup_t^{p,-1}(z) = WS_sup_inv(z) + WS_dem_inv(q_inelastic) - WS_dem_inv(z).

        This returns a PRICE for a given quantity z (the transformed supply curve).
        """
        if self.q_inelastic is None:
            return np.nan

        return ( self.WS_sup_inv(z)
                 + self.WS_dem_inv(self.q_inelastic)
                 - self.WS_dem_inv(z) )

    ######################################################################
    # 5) Invert Sup_t^{p,-1}(z) => find z for a given price x
    ######################################################################
    def invert_transformed_supply(self, x):
        """
        Solve Sup_t^{p,-1}(z) = x for z in [q_min, q_max].
        i.e. find z such that:
          (WS_sup_inv(z) + WS_dem_inv(q_inelastic) - WS_dem_inv(z)) = x.
        """
        def g(z):
            return self.transformed_inverse_supply(z) - x

        try:
            sol = brentq(g, self.q_min, self.q_max)
            return sol
        except ValueError:
            return None

    ######################################################################
    # 6) Measure elasticity via finite central difference around a reference quantity
    ######################################################################
    def measure_elasticity(
        self, 
        q_center: float, 
        q_small: float
    ) -> float:
        r"""
        Computes the finite central difference quotient around q_center,
        as in Balardy (2022) and Kulakov & Ziel (2019):
        
        .. math::
        
           \mathrm{MO}^{d,s}_{q} 
           \;=\;\frac{\Sup^{d,s}(q_{\text{center}} + q_{\text{small}}) 
                     \;-\;\Sup^{d,s}(q_{\text{center}} - q_{\text{small}})}%
                   {\,2 \cdot q_{\text{small}}\,}\,.
        
        Here, \(\Sup^{d,s}\equiv \text{transformed_inverse_supply}\)
        returns the price for a given quantity.
        
        :param q_center: The reference quantity (e.g., implied demand)
        :param q_small: Small shift in quantity for the finite-difference
        :return: MO^{d,s}_q (float), or NaN if out of domain
        """
        try:
            p_plus = self.transformed_inverse_supply(q_center + q_small)
            p_minus = self.transformed_inverse_supply(q_center - q_small)
            return (p_plus - p_minus) / (2.0 * q_small)
        except ValueError:
            return float('nan')

    ######################################################################
    # (Optional) Helper: compute a grid of transformed-inverse-supply values
    ######################################################################
    def compute_transformed_inverse_supply_grid(self, z_grid):
        """
        Returns an array of Sup_t^{p,-1}(z) for a given array of z.
        """
        return np.array([self.transformed_inverse_supply(z) for z in z_grid])

    ######################################################################
    # (Optional) Helper: compute the "implied demand" quantity for intraday price
    ######################################################################
    def find_implied_demand_quantity(self, p_id_0):
        """
        For a given intraday price p_id_0, find the quantity z implied by
        the transformed inverse supply, i.e. solve for z s.t. Sup_t^{p,-1}(z) = p_id_0.
        """
        return self.invert_transformed_supply(p_id_0)


# ---------------------------------------------
# Example usage
# ---------------------------------------------
if __name__ == "__main__":
    # Replace with your ENTSO-E API key
    api_key = "YOUR_ENTSOE_API_KEY"

    # Netherlands country code
    country_code = "NL"

    # Define the date range (in UTC) for the data
    start_date = pd.Timestamp('2025-01-01 00:00', tz='UTC')
    end_date   = pd.Timestamp('2025-01-03 00:00', tz='UTC')

    # Create an instance of DataCollection
    data_collector = DataCollection(
        api_key=api_key,
        country_code=country_code,
        start=start_date,
        end=end_date
    )

    # Run data collection
    data_collector.run_collection()