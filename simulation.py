# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 11:36:16 2025

@author: samue
"""

import numpy as np
import pandas as pd
from logreg import LassoLogisticBIC
from GAMLSS import GAMLSS
from scipy.stats import johnsonsu  
from distributions import JSUo



def simulate_paths(X_sim, initial_price, n_paths, gamlss_model, logreg_model):
    """
    Simulates price paths based on fitted regression models.
    
    Parameters:
      X_sim : np.array of shape (T, n_features)
          Simulation design matrix for T simulation intervals.
      initial_price : float
          The starting price P₀.
      n_paths : int
          Number of Monte Carlo simulation paths.
      gamlss_model : fitted GAMLSS model
          Model used to simulate the continuous ΔP (when a trade occurs).
      logreg_model : fitted logistic regression model
          Model used to predict the probability of a trade occurring.
    
    Returns:
      price_paths : np.array of shape (n_paths, T+1)
          Simulated price paths (each row is a path, first column is P₀).
    """
    T = X_sim.shape[0]
    price_paths = np.zeros((n_paths, T + 1))
    price_paths[:, 0] = initial_price
    
    for t in range(T):
        # Get feature vector for current simulation time (reshape to 2D)
        X_t = X_sim[t].reshape(1, -1)
        # Predict probability of a trade (α = 1) for this interval
        p_trade = logreg_model.predict_proba(X_t)[0, 1]
        
        # For the current time step, simulate ΔP for each path
        delta_t = np.zeros(n_paths)
        for i in range(n_paths):
            # Draw from a uniform random number to decide if a trade occurs
            if np.random.rand() < p_trade:
                # Get the distribution parameters for ΔP from the GAMLSS model.
                # Here we assume that the gamlss_model has a method `predict_params`
                # that returns a dictionary of parameters for the given features.
                params = gamlss_model.predict_params(X_t)
                delta_t[i] = gamlss_model.distribution.rsv(x ,theta)
            else:
                delta_t[i] = 0.0  # no trade ⇒ ΔP = 0
        
        # Update the price paths by adding the simulated ΔP
        price_paths[:, t + 1] = price_paths[:, t] + delta_t

    return price_paths

def run_simulation(data, X_sim, initial_price, n_paths):
    """
    Runs the full simulation by first performing regression on the training data,
    then simulating the price paths over the forecast horizon.
    
    Parameters:
      data : pd.DataFrame
          Training data containing the features and target variables.
          Expected columns include:
            - All feature columns (used for regression)
            - 'delta_P': the observed price differences (ΔP)
            - 'trade_indicator': binary indicator (α) for trade occurrence.
      X_sim : np.array of shape (T, n_features)
          Simulation design matrix for T simulation intervals.
      initial_price : float
          The starting price (P₀) for the simulation.
      n_paths : int
          Number of simulation paths to generate.
    
    Returns:
      sim_paths : np.array of shape (n_paths, T+1)
          The simulated price paths.
    """
    # Assume that the feature columns are all columns except for 'delta_P' and 'trade_indicator'
    feature_cols = [col for col in data.columns if col not in ['delta_P', 'trade_indicator']]
    X_train = data[feature_cols].values
    y_delta = data['delta_P'].values       # continuous target for ΔP
    y_trade = data['trade_indicator'].values  # binary target for trade occurrence
    
    # Fit logistic regression for the trade indicator
    logreg_model = LassoLogisticBIC()
    logreg_model.fit(X_train, y_trade)
    
    # Fit GAMLSS model for the price differences
    # (Assuming you want to use Johnson's SU as the continuous distribution;
    # replace 'YourDistribution' with the appropriate class from your codebase.)

    distribution = JSUo()  # instantiate your distribution object
    gamlss_model = GAMLSS(distribution=distribution, method="lasso")
    gamlss_model.fit(X_train, y_delta)
    
    # Now run the simulation
    sim_paths = simulate_paths(X_sim, initial_price, n_paths, gamlss_model, logreg_model)
    return sim_paths

# Example usage:
if __name__ == '__main__':
    # Load your training data (ensure it contains the required columns)
    data = pd.read_csv('training_data.csv')
    
    # Define the simulation horizon.
    # For example, T = 31 represents the 5-minute intervals from 185 to 30 minutes before delivery.
    T = 31
    # Here we simply use the first row of training features as a placeholder for the simulation features.
    # In practice, X_sim should be constructed based on your forecast inputs.
    dummy_features = data.drop(columns=['delta_P', 'trade_indicator']).iloc[0].values
    X_sim = np.tile(dummy_features, (T, 1))
    
    initial_price = 50.0   # example starting price
    n_paths = 1000         # number of simulation paths
    
    simulated_paths = run_simulation(data, X_sim, initial_price, n_paths)
    
    # Save the simulated paths for further analysis
    np.save('simulated_price_paths.npy', simulated_paths)
