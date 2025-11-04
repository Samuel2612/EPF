import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.preprocessing import StandardScaler


def _default_selector(columns):
    keep = []
    for c in columns:
        if (
            c.startswith("vwap_change")
            or c.startswith("abs_vwap_change")
            or c in {"solar_forecast", "wind_forecast", "mo_slope", "da-id", "f_TTD"}
        ):
            keep.append(c)
    return keep


class FeatureSelector:
    def __init__(self, use_columns=None, selector_fn=_default_selector):
        self.use_columns = use_columns
        self.selector_fn = selector_fn

    def select(self, df):
        if self.use_columns is not None:
            missing = [c for c in self.use_columns if c not in df.columns]
            if missing:
                raise KeyError(f"Missing columns: {missing}")
            return list(self.use_columns)
        cols = self.selector_fn(df.columns)
        if not cols:
            raise ValueError("No valid feature columns found.")
        return cols


class JointECF:
    """Rolling-window ECF keyed on delivery_start time (hourly product)."""

    def __init__(
        self,
        *,
        delivery_start_col="delivery_start",
        lookback_days=90,
        feature_selector=None,
    ):
        self.delivery_start_col = delivery_start_col
        self.lookback_days = int(lookback_days)
        self.feature_selector = feature_selector or FeatureSelector()


    def _slice_window(self, df, *, target_delivery_start):
        col = self.delivery_start_col
        if col not in df.columns:
            raise KeyError(f"{col!r} not found in DataFrame.")
        tgt_time = target_delivery_start.time()
        mask_product = df[col].dt.time == tgt_time
        start = target_delivery_start - timedelta(days=self.lookback_days)
        mask_window = (df[col] >= start) & (df[col] <= target_delivery_start)
        window = df.loc[mask_product & mask_window].sort_values(col)
        if window.empty:
            raise ValueError(
                f"No rows with delivery_start.time == {tgt_time} between "
                f"{start.date()} and {target_delivery_start.date()}."
            )
        return window

    @staticmethod
    def _standardise(X):
        scaler = StandardScaler()
        return scaler.fit_transform(X), scaler

    @staticmethod
    def _ecf_matrix(X_std, t):
        if t.ndim == 1:
            t = t[None, :]
        if X_std.shape[1] != t.shape[1]:
            raise ValueError(
                f"Frequency vectors dim = {t.shape[1]}, but feature space dim = {X_std.shape[1]}."
            )
        phase = X_std @ t.T               
        return np.mean(np.exp(1j * phase), axis=0)


    def ecf_for(self, df, *, target_delivery_start, t):
        window = self._slice_window(df, target_delivery_start=target_delivery_start)
        cols = self.feature_selector.select(window)
        X_std, scaler = self._standardise(window[cols].values)
        phi = self._ecf_matrix(X_std, t)
        return phi, scaler, cols



if __name__ == "__main__":
    date_cols = ['bin_timestamp', 'delivery_start']
    df = pd.read_csv("df_2021.csv")
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], utc=True).dt.tz_convert("Europe/Berlin")
        
    abs_grp = [f"abs_vwap_changes_lag{i}" for i in range(7, 13)]
    df["abs_vwap_changes_lag7_12"] = df[abs_grp].sum(axis=1)
    
    df = df.drop(columns=abs_grp)       

    target_tau = pd.Timestamp("2021-07-01 11:00", tz="Europe/Amsterdam")
    jecf = JointECF(lookback_days=90)
        
    used_cols = jecf.feature_selector.select(df)
    t = np.zeros((2, len(used_cols)))  
    t[1, 0] = 0.2                     
    
    phi, scaler, cols = jecf.ecf_for(df, target_delivery_start=target_tau, t=t)
    
    print("Columns used:", cols)
    print("φ(t) at t=0 : ", phi[0])     
    print("φ(t) probe  : ", phi[1])
