import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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


analysis = SupplyDemandAnalysis("auction_aggregated_curves_netherlands_20200630.csv")
analysis.plot_supply_demand(hour=9)
analysis.plot_transformed_curve(hour=9)
elasticity = analysis.calculate_elasticity(hour=9)
print(elasticity)
