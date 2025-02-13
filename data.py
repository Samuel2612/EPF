import pandas as pd
from entsoe import EntsoePandasClient
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

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
            
            
class SupplyDemandAnalysis:
    def __init__(self, data_path):
        # Load and clean the data
        self.data = pd.read_csv(data_path, skiprows=1, delimiter=',')

    def plot_supply_demand(self, hour):
        # Filter data for the specific hour
        hour_data = self.data[self.data["Hour"] == hour]
        supply = hour_data[hour_data["Sale/Purchase"] == "Sell"].sort_values(by="Price")
        demand = hour_data[hour_data["Sale/Purchase"] == "Purchase"].sort_values(by="Price")

        # Plot supply and demand curves
        plt.figure(figsize=(10, 6))
        plt.plot(supply["Volume"], supply["Price"], label="Supply Curve (Sell)", marker='o')
        plt.plot(demand["Volume"], demand["Price"], label="Demand Curve (Purchase)", marker='x')
        plt.title(f"Supply and Demand Curves for Hour {hour}")
        plt.xlabel("Volume (MWh)")
        plt.ylabel("Price (EUR/MWh)")
        plt.legend()
        plt.grid()
        plt.show()

    def transform_supply_curve(self, hour, p_min=-500.0):
        # Filter data for the specific hour
        hour_data = self.data[self.data["Hour"] == hour]
        supply = hour_data[hour_data["Sale/Purchase"] == "Sell"].sort_values(by="Price")
        demand = hour_data[hour_data["Sale/Purchase"] == "Purchase"].sort_values(by="Price")

        # Inverse curves (Price -> Volume interpolation functions)
        supply_inv = lambda p: np.interp(p, supply["Price"], supply["Volume"], left=0, right=0)
        demand_inv = lambda p: np.interp(p, demand["Price"], demand["Volume"], left=0, right=0)

        # Compute inelastic demand
        inelastic_demand = demand_inv(p_min)

        # Transformed supply curve
        def transformed_supply_inv(p):
            return supply_inv(p) + inelastic_demand - demand_inv(p)

        # Generate data for plotting
        price_range = np.linspace(supply["Price"].min(), supply["Price"].max(), 500)
        transformed_supply_volumes = [transformed_supply_inv(p) for p in price_range]

        return price_range, transformed_supply_volumes, inelastic_demand

    def plot_transformed_curve(self, hour, p_min=-500.0):
        price_range, transformed_supply_volumes, inelastic_demand = self.transform_supply_curve(hour, p_min)

        # Plot transformed supply curve
        plt.figure(figsize=(10, 6))
        plt.plot(transformed_supply_volumes, price_range, label="Transformed Supply Curve", marker='.')
        plt.axvline(x=inelastic_demand, color='r', linestyle='--', label=f"Inelastic Demand (Volume={inelastic_demand} MWh)")
        plt.title(f"Transformed Supply Curve and Inelastic Demand for Hour {hour}")
        plt.xlabel("Volume (MWh)")
        plt.ylabel("Price (EUR/MWh)")
        plt.legend()
        plt.grid()
        plt.show()

    def calculate_elasticity(self, hour, q_values=[500, 1000, 2000], p_min=-500.0):
        price_range, transformed_supply_volumes, _ = self.transform_supply_curve(hour, p_min)

        # Inverse transformed supply curve (Volume -> Price interpolation)
        transformed_supply = lambda v: np.interp(v, transformed_supply_volumes, price_range)

        # Calculate elasticity for each q in q_values
        elasticity = {}
        for q in q_values:
            implied_volume = transformed_supply(q)
            elasticity[q] = (transformed_supply(implied_volume + q) - transformed_supply(implied_volume - q)) / (2 * q)

        return elasticity


