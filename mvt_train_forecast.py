# -*- coding: utf-8 -*-
"""
Two-phase refactor for MVT_forecasting_study:

1) study.train()  -> returns MVTTrainedModel (parameters only, no predictions).
   - Saves one row per (date, hour, which_hour_before)
   - Stores mu, Sigma, S, df, n, p, feature_names, and StandardScaler stats (mean, scale)
   - Easy persistence via: trained.save_parquet(path) / MVTTrainedModel.load_parquet(path)
   - Easy accessors: trained.get_params(...), trained.as_result(...)

2) study.forecast(trained) -> uses a trained model to produce predictions + metrics.
   - Leaves calibration untouched; only scores observations and computes metrics.

Also keeps study.run() as a convenience wrapper that calls train() then forecast().
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Tuple, Iterable
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy import special, optimize  
import matplotlib.pyplot as plt
from scipy.stats import norm, t as student_t
from time import perf_counter
import json

@dataclass
class MVTResult:
    mu: np.ndarray                 # location (scaled space)
    S: np.ndarray                  # sample covariance (scaled space) -- always reported
    df: Union[float, np.floating]  # ν (np.inf if Gaussian-like)
    Sigma: np.ndarray              # t scale (scaled space); Cov = ν/(ν-2) * Sigma
    n: int
    p: int
    feature_names: List[str]
    used_data: pd.DataFrame        # rows used (original units)
    scaler: StandardScaler         # fitted scaler


class MVTCalibrator:
    """
    Minimal, DST-neutral calibrator:
    - Treats timestamps as naive wall-clock (no tz conversion).
    - If columns are already datetime64, leaves them untouched.
    - If delivery_hour_col is missing, derives hour from delivery_start_wall.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        *,
        timestamp_col: str = "bin_timestamp",
        delivery_start_col: Optional[str] = "delivery_start_wall",
        delivery_hour_col: Optional[str] = None,
        tz: Optional[str] = None,                     # kept for API compatibility, ignored
        main_feature_col: str = "vwap_changes",
        main_feature_alias: str = "x",
        other_features: Optional[Dict[str, str]] = None,
        log1p_aliases: Optional[set] = None,
        parse_dayfirst: bool = True,                  # only used if we must parse strings
    ):
        self.df = df.copy()
        self._ts = timestamp_col


        if self._ts not in self.df.columns:
            raise ValueError(f"Missing timestamp column: {self._ts}")
        if not np.issubdtype(self.df[self._ts].dtype, np.datetime64):
            self.df[self._ts] = pd.to_datetime(
                self.df[self._ts], errors="coerce", dayfirst=parse_dayfirst
            )
        if self.df[self._ts].isna().any():
            bad = self.df.index[self.df[self._ts].isna()][:5].tolist()
            raise ValueError(
                f"Column '{self._ts}' has non-parsable timestamps. Example bad indices: {bad}"
            )

        # --- delivery hour:
        self._ds = None  # name of delivery_start col if used
        if delivery_hour_col is not None:
            # Use provided delivery_hour column
            if delivery_hour_col not in self.df.columns:
                raise ValueError(f"Missing delivery_hour column: {delivery_hour_col}")
            self._dh = delivery_hour_col
            self.df[self._dh] = self.df[self._dh].astype(int)
        else:
            # Derive hour from delivery_start_wall (already parsed by read_csv)
            if delivery_start_col is None or delivery_start_col not in self.df.columns:
                raise ValueError("Provide delivery_hour_col or a valid delivery_start_col.")
            ds_col = delivery_start_col
            if not np.issubdtype(self.df[ds_col].dtype, np.datetime64):
                # Only parse if it still isn't datetime
                self.df[ds_col] = pd.to_datetime(
                    self.df[ds_col], errors="coerce", dayfirst=parse_dayfirst
                )
            if self.df[ds_col].isna().any():
                print(self.df[ds_col].isna())
                bad = self.df.index[self.df[ds_col].isna()][:5].tolist()
                
                raise ValueError(
                    f"Column '{ds_col}' contains non-parsable delivery_start timestamps. "
                    f"Example bad indices: {bad}"
                
                )
            self.df["_delivery_hour"] = self.df[ds_col].dt.hour
            self._dh = "_delivery_hour"
            self._ds = ds_col

        # --- main feature
        self._main_col = main_feature_col
        if self._main_col not in self.df.columns:
            raise ValueError(f"Missing main_feature_col: {self._main_col}")
        self._main_alias = (main_feature_alias or "x").strip()

        # --- other features (alias -> column) with optional log1p
        self._other_features: Dict[str, str] = dict(other_features or {})
        for c in self._other_features.values():
            if c not in self.df.columns:
                raise ValueError(f"Missing other-feature column: {c}")

        self._log1p_aliases = set(log1p_aliases or {"mo"})
        for alias, col in list(self._other_features.items()):
            if alias in self._log1p_aliases:
                if (self.df[col].min(skipna=True) <= -1.0):
                    bad_idx = self.df.index[self.df[col] <= -1.0][:3].tolist()
                    raise ValueError(
                        f"Column '{col}' (alias '{alias}') has values ≤ -1; cannot apply log1p. "
                        f"Bad indices: {bad_idx}"
                    )
                new_col = f"__log1p__{col}"
                self.df[new_col] = np.log1p(self.df[col].astype(float))
                self._other_features[alias] = new_col

        # Sort once (naive, wall-clock)
        self.df.sort_values([self._ts, self._dh], inplace=True)

        # Result cache
        self._last_result: Optional[MVTResult] = None

    def _window_for_day(self, day: pd.Timestamp, H: int, which_hour_before: int, window_minutes: int):
        if day.tz is None and self.df[self._ts].dt.tz is not None:
            day = day.tz_localize(self.df[self._ts].dt.tz)
        delivery_start = day.replace(hour=H, minute=0, second=0, microsecond=0)
        start = delivery_start - pd.Timedelta(hours=which_hour_before)
        end = start + pd.Timedelta(minutes=window_minutes)
        return start, end

    def _offset_hour(self, H: int, offset: int) -> int:
        return (int(H) + int(offset)) % 24

    def _name_for(self, alias: str, offset: int, lag: int) -> str:
        if offset == 0:
            return f"{alias}_curr_now" if lag == 0 else f"{alias}_curr_lag{lag}"
        elif offset > 0:
            return f"{alias}_next{offset}_now" if lag == 0 else f"{alias}_next{offset}_lag{lag}"
        else:
            m = abs(offset)
            return f"{alias}_prev{m}_now" if lag == 0 else f"{alias}_prev{m}_lag{lag}"

    def _collect_feature_rows_for_day(
        self,
        day: pd.Timestamp,
        H: int,
        which_hour_before: int,
        window_minutes: int,
        *,
        product_lags: Optional[Dict[int, List[int]]] = None,
        lag_minutes: int = 15,
    ) -> pd.DataFrame:
        # Default: current now+lag1, next1 lag1, prev1 lag1
        if product_lags is None:
            product_lags = {0: [0, 1], 1: [1], -1: [1]}

        start, end = self._window_for_day(day, H, which_hour_before, window_minutes)

        # Base slice: **use delivery-start day if available**, then apply the time window.
        if self._ds is not None:
            day_norm = day.normalize()
            base_mask = (
                (self.df[self._dh] == int(H)) &
                (self.df[self._ds].dt.normalize() == day_norm) &    # <-- key change
                (self.df[self._ts] >= start) & (self.df[self._ts] < end)
            )
        else:
            base_mask = (
                (self.df[self._dh] == int(H)) &
                (self.df[self._ts] >= start) & (self.df[self._ts] < end)
            )

        base_cols = [self._ts, self._dh, self._main_col]
        base = self.df.loc[base_mask, base_cols].copy()
        if base.empty:
            return base

        feat = base.rename(columns={self._main_col: f"{self._main_alias}_curr_now"}).copy()

        def make_lag_df(hour: int, lag: int, new_col: str) -> pd.DataFrame:
            tmp = self.df[self.df[self._dh] == hour][[self._ts, self._main_col]].copy()
            if lag > 0:
                tmp[self._ts] = tmp[self._ts] + pd.Timedelta(minutes=lag_minutes * lag)
            tmp.rename(columns={self._main_col: new_col}, inplace=True)
            return tmp

        normalized = []
        for off, lags in (product_lags or {}).items():
            off = int(off)
            if off < 0 and int(which_hour_before) <= abs(off):
                continue
            L = sorted({int(l) for l in lags if int(l) >= 0})
            if L:
                normalized.append((off, L))
        normalized.sort(key=lambda t: (t[0] != 0, t[0] < 0, abs(t[0])))

        seen_curr_now = False
        for off, L in normalized:
            hour_off = self._offset_hour(H, off)
            for lag in L:
                colname = self._name_for(self._main_alias, off, lag)
                if off == 0 and lag == 0:
                    seen_curr_now = True
                    continue
                lag_df = make_lag_df(hour_off, lag, colname)
                feat = pd.merge(feat, lag_df, on=self._ts, how="inner")

        for alias, col in self._other_features.items():
            tmp = self.df[self.df[self._dh] == int(H)][[self._ts, col]].copy()
            tmp[self._ts] = tmp[self._ts] + pd.Timedelta(minutes=lag_minutes)  # lag1
            tmp.rename(columns={col: f"{alias}_lag1"}, inplace=True)
            feat = pd.merge(feat, tmp, on=self._ts, how="inner")

        ordered_main = []
        if (seen_curr_now or f"{self._main_alias}_curr_now" in feat.columns):
            ordered_main.append(f"{self._main_alias}_curr_now")
        curr_lags = [self._name_for(self._main_alias, 0, l)
                     for l in sorted({l for off, L in normalized if off == 0 for l in L if l > 0})]
        ordered_main += [c for c in curr_lags if c in feat.columns]
        prev_offsets = sorted({abs(off) for off, _ in normalized if off < 0})
        for m in prev_offsets:
            cols = [self._name_for(self._main_alias, -m, l)
                    for l in sorted({l for off, L in normalized if off == -m for l in L})]
            ordered_main += [c for c in cols if c in feat.columns]
        next_offsets = sorted({off for off, _ in normalized if off > 0})
        for m in next_offsets:
            cols = [self._name_for(self._main_alias, m, l)
                    for l in sorted({l for off, L in normalized if off == m for l in L})]
            ordered_main += [c for c in cols if c in feat.columns]

        cols_others = [f"{alias}_lag1" for alias in self._other_features.keys()]
        ordered = [self._ts] + [c for c in ordered_main if c in feat.columns] + cols_others
        return feat[ordered].dropna().reset_index(drop=True)

    def _collect_samples(
        self,
        target_date: Union[str, pd.Timestamp],
        H: int,
        which_hour_before: int,
        window_minutes: int,
        lookback_days: int,
        *,
        product_lags: Optional[Dict[int, List[int]]] = None,
        lag_minutes: int = 15,
    ) -> pd.DataFrame:
        if isinstance(target_date, str):
            target_date = pd.to_datetime(target_date)
        if target_date.tz is None and self.df[self._ts].dt.tz is not None:
            target_date = target_date.tz_localize(self.df[self._ts].dt.tz)

        # START ONE EXTRA DAY EARLIER to cover spillover into the "91st" day for early hours
        start_day = (target_date - pd.Timedelta(days=lookback_days)).normalize() - pd.Timedelta(days=1)
        end_day = (target_date - pd.Timedelta(days=1)).normalize()
        if end_day < start_day:
            raise ValueError("lookback_days must be ≥ 1.")

        chunks: List[pd.DataFrame] = []
        for d in pd.date_range(start_day, end_day, freq="D"):
            ch = self._collect_feature_rows_for_day(
                d, H, which_hour_before, window_minutes,
                product_lags=product_lags, lag_minutes=lag_minutes,
            )
            if not ch.empty:
                ch.insert(0, "day", d.date())
                chunks.append(ch)
        return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    
    @staticmethod
    def _mahalanobis2(X: np.ndarray, mu: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
        L = np.linalg.cholesky(Sigma)
        Y = np.linalg.solve(L, (X - mu).T)
        return np.sum(Y * Y, axis=0)
    
    @staticmethod
    def _mardia_kurtosis(X):
        mu = X.mean(axis=0, keepdims=True)
        Xc = X - mu
        S = np.cov(X, rowvar=False, ddof=1)
        S_inv = np.linalg.pinv(S)
        d2 = np.einsum("ni,ij,nj->n", Xc, S_inv, Xc)
        return float(np.mean(d2**2))

    @staticmethod
    def _df_from_mardia(beta_2p, p, eps=1e-9):
        k = beta_2p / (p * (p + 2))

        if k <= 1 + eps:
            return float(np.inf)
        nu_hat = (4.0 * k - 2.0) / (k - 1.0)
        return float(nu_hat)

    @staticmethod
    def _logdet(Sigma: np.ndarray) -> float:
        return 2.0 * np.sum(np.log(np.diag(np.linalg.cholesky(Sigma))))

    def _e_step(self, X: np.ndarray, mu: np.ndarray, Sigma: np.ndarray, nu: float) -> Tuple[np.ndarray, np.ndarray]:
        d2 = self._mahalanobis2(X, mu, Sigma)
        p = X.shape[1]
        w = (nu + p) / (nu + d2)
        return w, d2

    def _m_mu(self, X: np.ndarray, w: np.ndarray) -> np.ndarray:
        sw = float(np.sum(w))
        return (w[:, None] * X).sum(axis=0) / sw

    def _m_Sigma(self, X: np.ndarray, mu: np.ndarray, w: np.ndarray, ridge: float = 1e-8) -> np.ndarray:
        D = X - mu
        WD = D * w[:, None]
        S = (WD.T @ D) / X.shape[0]
        tr = float(np.trace(S)) or 1.0
        return 0.5 * (S + S.T) + ridge * (tr / X.shape[1]) * np.eye(X.shape[1])

    def _ll_nu_given_d2(self, nu: float, d2: np.ndarray, p: int) -> float:
        n = d2.size
        return (
            n * (special.gammaln(0.5 * (nu + p)) - special.gammaln(0.5 * nu) - 0.5 * p * np.log(nu))
            - 0.5 * (nu + p) * np.sum(np.log1p(d2 / nu))
        )

    def _update_nu_brent(self, d2: np.ndarray, p: int, bounds: Tuple[float, float] = (.1, 100.0)) -> float:
        def obj(nu):
            return -self._ll_nu_given_d2(nu, d2, p)
        res = optimize.minimize_scalar(obj, bounds=bounds, method="bounded",
                                       options={"xatol": 1e-6, "maxiter": 200})
        return float(res.x)

    def _loglik_full(self, X: np.ndarray, mu: np.ndarray, Sigma: np.ndarray, nu: float) -> float:
        n, p = X.shape
        d2 = self._mahalanobis2(X, mu, Sigma)
        return (
            n * (special.gammaln(0.5 * (nu + p)) - special.gammaln(0.5 * nu) - 0.5 * p * np.log(nu * np.pi))
            - 0.5 * n * self._logdet(Sigma)
            - 0.5 * (nu + p) * np.sum(np.log1p(d2 / nu))
        )

    def _fit_t_ecm_scaled(
        self,
        Xs: np.ndarray,
        nu_init: Union[str, Tuple[str, float]] = "mardia",
        max_iter: int = 200,
        tol: float = 1e-6,
        verbose: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        n, p = Xs.shape
        mu = Xs.mean(axis=0)
        S_emp = np.cov(Xs, rowvar=False, ddof=1)
        tr = float(np.trace(S_emp)) or 1.0
        Sigma = 0.5 * (S_emp + S_emp.T) + 1e-8 * (tr / p) * np.eye(p)

        if isinstance(nu_init, tuple) and nu_init[0] == "fixed":
            nu = float(max(nu_init[1], 0.6))
        else:
            try:
                beta_2p = self._mardia_kurtosis(Xs)
                nu = self._df_from_mardia(beta_2p, p=p)
                if not np.isfinite(nu):
                    nu = 30.0
                nu = float(max(nu, 0.6))
            except Exception:
                nu = 4.0

        ll_prev = self._loglik_full(Xs, mu, Sigma, nu)
        if verbose:
            print(f"[init] ll={ll_prev:.3f} nu={nu:.3f}")

        for it in range(1, max_iter + 1):
            w, d2 = self._e_step(Xs, mu, Sigma, nu)
            mu = self._m_mu(Xs, w)
            Sigma = self._m_Sigma(Xs, mu, w)
            d2 = self._mahalanobis2(Xs, mu, Sigma)
            nu = self._update_nu_brent(d2, p)

            ll = self._loglik_full(Xs, mu, Sigma, nu)
            if verbose and (it <= 3 or it % 10 == 0):
                print(f"[it {it:02d}] ll={ll:.3f} nu={nu:.4f}")
            if abs(ll - ll_prev) / (1.0 + abs(ll_prev)) < tol:
                break
            ll_prev = ll

        return mu, Sigma, nu, S_emp

    def calibrate(
        self,
        target_date: Union[str, pd.Timestamp],
        delivery_start_hour: int,
        which_hour_before: int,
        window_minutes: int = 60,
        lookback_days: int = 90,
        method: str = "ecm",                  # "ecm" or "mardia"
        nu_init: Union[str, Tuple[str, float]] = "mardia",  # for ECM only
        verbose_ecm: bool = True,
        *,
        product_lags: Optional[Dict[int, List[int]]] = None,  
        lag_minutes: int = 15,                                
    ) -> MVTResult:
        samples = self._collect_samples(
            target_date, delivery_start_hour, which_hour_before,
            window_minutes, lookback_days,
            product_lags=product_lags,
            lag_minutes=lag_minutes,
        )
        if samples.empty:
            raise ValueError("No samples found for this configuration.")

        feat_cols = [c for c in samples.columns if c not in ("day", self._ts)]

        X = samples[feat_cols].to_numpy(float)
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        n, p = Xs.shape

        S_emp = np.cov(Xs, rowvar=False, ddof=1)

        if method.lower() == "ecm":
            mu_hat, Sigma_hat, nu_hat, S_emp = self._fit_t_ecm_scaled(
                Xs, nu_init=nu_init, verbose=verbose_ecm
            )
            mu, Sigma, nu, S = mu_hat, Sigma_hat, nu_hat, S_emp
        elif method.lower() == "mardia":
            mu = Xs.mean(axis=0)
            S = S_emp
            beta_2p = self._mardia_kurtosis(Xs)
            nu = self._df_from_mardia(beta_2p, p=p)
            Sigma = S * (nu - 2.0) / nu if np.isfinite(nu) and nu > 2 else S.copy()
        else:
            raise ValueError("method must be 'ecm' or 'mardia'.")

        res = MVTResult(
            mu=mu,
            S=S,
            df=nu,
            Sigma=Sigma,
            n=n,
            p=p,
            feature_names=feat_cols,
            used_data=samples[[self._ts] + feat_cols].copy(),
            scaler=scaler,
        )
        self._last_result = res
        return res

    def predict(
        self,
        observed: Dict[str, float],
        result: Optional[MVTResult] = None,
        target_name: Optional[str] = None,
    ) -> Dict[str, float]:
        res = result if result is not None else self._last_result
        if res is None:
            raise RuntimeError("Run calibrate(...) first.")
        feat = res.feature_names
        # default target = main feature current
        if target_name is None:
            target_name = f"{self._main_alias}_curr_now"
        if target_name not in feat:
            raise ValueError(f"target_name '{target_name}' not in features {feat}")

        y_idx = feat.index(target_name)
        z_names = [n for i, n in enumerate(feat) if i != y_idx]
        missing = [k for k in z_names if k not in observed]
        if missing:
            raise ValueError(f"Missing observed values for: {missing}")

        mu = res.mu
        Sigma = res.Sigma
        scaler = res.scaler

        idx_y = [y_idx]
        idx_z = [i for i in range(len(feat)) if i != y_idx]

        mu_y = mu[y_idx]
        mu_z = mu[idx_z]
        Σ_yy = Sigma[np.ix_(idx_y, idx_y)][0, 0]
        Σ_yz = Sigma[np.ix_(idx_y, idx_z)][0, :]
        Σ_zy = Sigma[np.ix_(idx_z, idx_y)][:, 0]
        Σ_zz = Sigma[np.ix_(idx_z, idx_z)]

        z0 = np.array([observed[k] for k in z_names], dtype=float)
        z0_scaled = (z0 - scaler.mean_[idx_z]) / scaler.scale_[idx_z]

        Σ_zz_inv = np.linalg.pinv(Σ_zz)
        diff = z0_scaled - mu_z
        A = Σ_yz @ Σ_zz_inv
        schur = float(Σ_yy - A @ Σ_zy)

        if np.isfinite(res.df):
            ν = float(res.df)
            q = len(z_names)
            δ_z = float(diff @ Σ_zz_inv @ diff)
            ν_star = ν + q
            mean_scaled = float(mu_y + A @ diff)
            scale_cond = ((ν + δ_z) / (ν + q - 2.0)) * schur
            variance_scaled = (ν_star / (ν_star - 2.0)) * scale_cond if ν_star > 2 else float("nan")
            df_cond = ν_star
        else:
            mean_scaled = float(mu_y + A @ diff)
            variance_scaled = schur
            df_cond = float("inf")

        mean_orig = mean_scaled * scaler.scale_[y_idx] + scaler.mean_[y_idx]
        var_orig = variance_scaled * (scaler.scale_[y_idx] ** 2)

        return {
            "mean": float(mean_orig),
            "variance": float(var_orig),
            "mean_scaled": float(mean_scaled),
            "variance_scaled": float(variance_scaled),
            "df_cond": float(df_cond),
        }
    
    def ecm_covariance(self, result: Optional[MVTResult] = None, standardized: bool = True) -> np.ndarray:
        res = result if result is not None else self._last_result
        if res is None:
            raise RuntimeError("Run calibrate(...) first.")
        # ECM covariance in scaled space
        if np.isfinite(res.df) and res.df > 2.0:
            C = (float(res.df) / (float(res.df) - 2.0)) * res.Sigma
        else:
            C = res.Sigma.copy()  # fallback when df≤2 or Gaussian-like
        if standardized:
            return C
        # back to original units: D * C * D
        D = np.diag(res.scaler.scale_.astype(float))
        return D @ C @ D
    
    def sample_covariance(self, result: Optional[MVTResult] = None, standardized: bool = True) -> np.ndarray:
        res = result if result is not None else self._last_result
        if res is None:
            raise RuntimeError("Run calibrate(...) first.")
        S = res.S.copy()
        if standardized:
            return S
        D = np.diag(res.scaler.scale_.astype(float))
        return D @ S @ D


# ======================================================================================
# NEW: Trained model container with easy persistence & accessors
# ======================================================================================

@dataclass
class MVTTrainedModel:
    """Holds per-(date, hour, which_hour_before) calibration parameters.

    params: tidy DataFrame with columns
        ['date','hour','which_hour_before','method','calibration_ok','error',
         'df','n','p','feature_names','mu_scaled','Sigma_scaled','S_sample_scaled',
         'scaler_mean','scaler_scale']
    """
    cfg: Dict
    params: pd.DataFrame

    def save_parquet(self, path: str) -> None:
        df = self.params.copy()
        # Ensure Python lists (not numpy arrays) for nested columns
        for col in ["feature_names", "mu_scaled", "Sigma_scaled", "S_sample_scaled", "scaler_mean", "scaler_scale"]:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: x if isinstance(x, list) else (x.tolist() if x is not None else None))
        df.to_parquet(path, index=False)

    @staticmethod
    def load_parquet(path: str, cfg: Optional[Dict] = None) -> "MVTTrainedModel":
        df = pd.read_parquet(path)
        return MVTTrainedModel(cfg or {}, df)

    # -------------- Quick access --------------
    def _row(self, date, hour: int, which_hour_before: int) -> pd.Series:
        # Normalize input date to midnight
        date_norm = pd.to_datetime(date).normalize()
    
        # Robustly normalize the saved column
        col = self.params["date"]
        try:
            col_norm = pd.to_datetime(col, errors="coerce").dt.normalize()
        except Exception:
            # Last resort: string compare on ISO date
            col_norm = pd.to_datetime(col.astype(str), errors="coerce").dt.normalize()
    
        m = (
            (col_norm == date_norm) &
            (self.params["hour"].astype(int) == int(hour)) &
            (self.params["which_hour_before"].astype(int) == int(which_hour_before))
        )
        if not m.any():
            raise KeyError(f"No trained params for date={date_norm.date()}, hour={hour}, wh={which_hour_before}")
        return self.params.loc[m].iloc[0]

    def get_params(self, date, hour: int, which_hour_before: int) -> Dict:
        r = self._row(date, hour, which_hour_before)
        return r.to_dict()

    def as_result(self, date, hour: int, which_hour_before: int) -> MVTResult:
        r = self._row(date, hour, which_hour_before)
        if not bool(r["calibration_ok"]):
            raise RuntimeError(f"Calibration failed for {date} H{hour:02d} WH{which_hour_before}: {r['error']}")
        # Rebuild a minimal StandardScaler
        scaler = StandardScaler()
        scaler.mean_ = np.asarray(r["scaler_mean"], dtype=float)
        scaler.scale_ = np.asarray(r["scaler_scale"], dtype=float)
        scaler.var_ = scaler.scale_ ** 2
        scaler.n_features_in_ = scaler.mean_.shape[0]
        # Optional: attach names (not used by predict)
        try:
            scaler.feature_names_in_ = np.array(list(r["feature_names"]))
        except Exception:
            pass
        # Build MVTResult (used_data left empty; not required for forecasting)
        feat = list(r["feature_names"])
        mu = np.asarray(r["mu_scaled"], dtype=float)
        Sigma = np.asarray(r["Sigma_scaled"], dtype=float)
        S = np.asarray(r["S_sample_scaled"], dtype=float)
        df = float(r["df"]) if r.get("df", None) is not None else float("inf")
        used = pd.DataFrame(columns=["timestamp"] + feat)
        return MVTResult(
            mu=mu,
            S=S,
            df=df,
            Sigma=Sigma,
            n=int(r["n"]),
            p=int(r["p"]),
            feature_names=feat,
            used_data=used,
            scaler=scaler,
        )
    def save(self, basepath: str) -> None:
        """Writes basepath.parquet (params) and basepath.json (cfg)."""
        self.save_parquet(basepath + ".parquet")
        with open(basepath + ".json", "w", encoding="utf-8") as f:
            json.dump(self.cfg, f)
    
    @staticmethod
    def load(basepath: str) -> "MVTTrainedModel":
        """Loads basepath.parquet (params) and basepath.json (cfg, if present)."""
        params = pd.read_parquet(basepath + ".parquet")
        try:
            with open(basepath + ".json", "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except FileNotFoundError:
            cfg = {}  # fallback; you can still pass cfg manually to from_trained
        return MVTTrainedModel(cfg, params)


# ======================================================================================
# Two-phase study: train() and forecast(trained)
# ======================================================================================

class MVT_forecasting_study:
    """
    Rolling forecasting study for the MVTCalibrator.

    PHASE 1 — train():
        - 90-day rolling window (configurable)
        - hourly retraining for horizons in which_hours_before (e.g., 4,3,2,1)
        - stores parameters per (date, hour, wh) in an MVTTrainedModel

    PHASE 2 — forecast(trained):
        - loads parameters from MVTTrainedModel
        - builds observation rows for each (date, hour, wh) and scores them
        - computes MAE, RMSE, negative log-score, CRPS

    Convenience run(): train() then forecast().
    """

    @staticmethod
    def _crps_gaussian(mu: float, sigma: float, y: float) -> float:
        z = (y - mu) / max(sigma, 1e-12)
        return float(sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi)))

    @staticmethod
    def _crps_student_t_mc(mu: float, scale: float, df: float, y: float,
                           n: int = 1500, seed: int = 123) -> float:
        rng = np.random.default_rng(seed)
        x = student_t.rvs(df, loc=mu, scale=scale, size=n, random_state=rng)
        x2 = student_t.rvs(df, loc=mu, scale=scale, size=n, random_state=rng)
        return float(np.mean(np.abs(x - y)) - 0.5 * np.mean(np.abs(x - x2)))

    @staticmethod
    def _build_product_lags(
        which_hour_before: int,
        current_lags: int = 3,
        total_neighbors: int = 4,
        prefer_before: int = 2,
        neighbor_lags: Iterable[int] = (1,),
    ) -> Dict[int, List[int]]:
        """
        Returns {offset -> [lags]} for a given horizon.

        - current product: lags [0..current_lags]
        - neighbours: choose 'total_neighbors' offsets; prefer 'prefer_before' previous ones
        - each neighbour uses 'neighbor_lags' (>=1, no contemporaneous)
        Availability rule: previous offset -m only if which_hour_before > m.
        """
        pl: Dict[int, List[int]] = {0: list(range(0, int(current_lags) + 1))}
        nl = sorted({int(l) for l in neighbor_lags if int(l) >= 1})
        if not nl:
            raise ValueError("neighbor_lags must contain at least one integer >= 1")

        max_prev_available = max(0, int(which_hour_before) - 1)
        prev_to_use = min(prefer_before, max_prev_available)
        next_to_use = max(0, total_neighbors - prev_to_use)

        for m in range(1, prev_to_use + 1):
            pl[-m] = nl.copy()
        for m in range(1, next_to_use + 1):
            pl[m] = nl.copy()

        extra_needed = total_neighbors - (prev_to_use + next_to_use)
        k = next_to_use
        while extra_needed > 0:
            k += 1
            pl[k] = nl.copy()
            extra_needed -= 1
        return pl

    def __init__(
        self, 
        df: pd.DataFrame, 
        *,
        timestamp_col: str = "bin_timestamp",
        delivery_start_col: Optional[str] = "delivery_start_wall",  
        delivery_hour_col: Optional[str] = None,
        tz: Optional[str] = None,
        main_feature_col: str = "da-id",
        main_feature_alias: str = "daid",
        other_features: Optional[Dict[str, str]] = None,
        log1p_aliases: Optional[set] = None,
        parse_dayfirst: bool = True,

        # Study window & cadence
        lookback_days: int = 90,
        window_minutes: int = 60,
        which_hours_before: Iterable[int] = (4, 3, 2, 1),
        hours: Iterable[int] = tuple(range(24)),
        lag_minutes: int = 15,

        # Feature design
        current_lags: int = 3,
        total_neighbors: int = 4,
        prefer_before: int = 2,
        neighbor_lags: Iterable[int] = (1,),
        

        # Model choice
        method: str = "ecm",
        nu_init: Union[str, Tuple[str, float]] = "mardia",
        verbose_ecm: bool = False,

        # Metrics
        crps_mc_n: int = 1500,
        seed: int = 123,

        # Date bounds
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        

        # Logging
        verbose: bool = False,
    ):
        self.cfg = dict(
            timestamp_col=timestamp_col,
            delivery_start_col=delivery_start_col,
            delivery_hour_col=delivery_hour_col,
            tz=tz,
            main_feature_col=main_feature_col,
            main_feature_alias=main_feature_alias,
            other_features=other_features,
            log1p_aliases=log1p_aliases,
            parse_dayfirst=parse_dayfirst,
            lookback_days=lookback_days,
            window_minutes=window_minutes,
            which_hours_before=tuple(which_hours_before),
            hours=tuple(hours),
            lag_minutes=lag_minutes,
            current_lags=current_lags,
            total_neighbors=total_neighbors,
            prefer_before=prefer_before,
            neighbor_lags=tuple(neighbor_lags),
            method=method.lower(),
            nu_init=nu_init,
            verbose_ecm=verbose_ecm,
            crps_mc_n=crps_mc_n,
            seed=seed,
            start_date=start_date,
            end_date=end_date,
            verbose=verbose,
        )

        self.cal = MVTCalibrator(
            df,
            timestamp_col=timestamp_col,
            delivery_start_col=delivery_start_col,
            delivery_hour_col=delivery_hour_col,
            tz=tz,
            main_feature_col=main_feature_col,
            main_feature_alias=main_feature_alias,
            other_features=other_features,
            log1p_aliases=log1p_aliases,
            parse_dayfirst=parse_dayfirst, 
        )
        self.target_name = f"{main_feature_alias}_curr_now"
        self.predictions_ = None
        self.params_ = None

        # determine study dates
        if delivery_start_col is not None and delivery_start_col in self.cal.df.columns:
            day_source_col = delivery_start_col
        else:
            day_source_col = self.cal._ts

        day_series = self.cal.df[day_source_col].dropna()
        # Only tz-convert if series is tz-aware AND a tz was provided
        if tz is not None and getattr(day_series.dt, "tz", None) is not None:
            day_series = day_series.dt.tz_convert(tz)
        days_norm = day_series.dt.normalize()
        all_days = pd.DatetimeIndex(days_norm.unique()).sort_values()

        if start_date is None:
            start_date = (all_days.min() + pd.Timedelta(days=self.cfg["lookback_days"]))
        else:
            start_date = pd.to_datetime(start_date, dayfirst=parse_dayfirst)
        if end_date is None:
            end_date = all_days.max()
        else:
            end_date = pd.to_datetime(end_date, dayfirst=parse_dayfirst)

        # Keep naive if input is naive
        self.start_day = start_date.normalize()
        self.end_day   = end_date.normalize()

        # --- NEW: shrink the working frame early for speed ---
        self._prefilter_df()  
        
    def _prefilter_df(self) -> None:
        """
        Keep only the needed columns and only the timestamps that can influence
        the study window: [start_day - lookback - 1 day, end_day + 1 day).
        This can drastically reduce memory/CPU for training.
        """
        ts_col = self.cal._ts
        dh_col = self.cal._dh
        ds_col = self.cal._ds  # may be None

        # Columns we actually use downstream
        needed = {ts_col, dh_col, self.cfg["main_feature_col"]}
        needed.update(self.cal._other_features.values())  # after log1p remapping

        if ds_col is not None:
            needed.add(ds_col)

        # Time bounds (include lookback + small cushion for lag windows)
        lb = int(self.cfg["lookback_days"]) + 1
        min_ts = self.start_day - pd.Timedelta(days=lb)
        max_ts = self.end_day + pd.Timedelta(days=1)

        m = (self.cal.df[ts_col] >= min_ts) & (self.cal.df[ts_col] < max_ts)
        self.cal.df = (
            self.cal.df.loc[m, sorted(needed)]
                       .sort_values([ts_col, dh_col])
                       .reset_index(drop=True)
        )

    # ---------------------------- Helper: construct from trained cfg ----------------------------
    @classmethod
    def from_trained(cls, df: pd.DataFrame, trained: MVTTrainedModel) -> "MVT_forecasting_study":
        """Recreate a study with the exact configuration used during training.
        This guarantees feature design + scaling compatibility for forecasting.
        Usage:
            trained = MVTTrainedModel.load_parquet("mvt_trained.parquet")
            study   = MVT_forecasting_study.from_trained(df, trained)
            preds, summary = study.forecast(trained)
        """
        return cls(df, **trained.cfg)

    # ---------------------------- PHASE 1: TRAIN ----------------------------
    def train(self) -> MVTTrainedModel:
        cfg = self.cfg
        days = pd.date_range(self.start_day, self.end_day, freq="D")
        params_rows = []

        for day in days:
            for H in cfg["hours"]:
                for wh in cfg["which_hours_before"]:
                    pl = self._build_product_lags(
                        which_hour_before=int(wh),
                        current_lags=cfg["current_lags"],
                        total_neighbors=cfg["total_neighbors"],
                        prefer_before=cfg["prefer_before"],
                        neighbor_lags=cfg["neighbor_lags"],
                    )
                    if cfg["verbose"]:
                        print(f"[TRAIN {day.date()}] H={int(H):02d}, WH={int(wh)}", flush=True)
                    try:
                        time1 = perf_counter()
                        res = self.calibrate_once(
                            target_date=day, H=int(H), wh=int(wh),
                            window_minutes=cfg["window_minutes"],
                            lookback_days=cfg["lookback_days"],
                            method=cfg["method"],
                            nu_init=cfg["nu_init"],
                            verbose_ecm=cfg["verbose_ecm"],
                            product_lags=pl,
                            lag_minutes=cfg["lag_minutes"],
                        )
                        print(perf_counter() - time1)
                        cal_ok = True
                        cal_err = ""
                    except Exception as e:
                        res = None
                        cal_ok = False
                        cal_err = str(e)
                        if cfg["verbose"]:
                            print(f"  calibration failed: {cal_err}", flush=True)

                    row = {
                        "date": day.date(),
                        "hour": int(H),
                        "which_hour_before": int(wh),
                        "method": cfg["method"],
                        "calibration_ok": cal_ok,
                        "error": cal_err if not cal_ok else "",
                    }
                    if cal_ok:
                        row.update({
                            "df": float(res.df),
                            "n": res.n,
                            "p": res.p,
                            "feature_names": list(res.feature_names),
                            "mu_scaled": res.mu.tolist(),
                            "Sigma_scaled": res.Sigma.tolist(),
                            "S_sample_scaled": res.S.tolist(),
                            "scaler_mean": res.scaler.mean_.astype(float).tolist(),
                            "scaler_scale": res.scaler.scale_.astype(float).tolist(),
                        })
                    params_rows.append(row)

        params_df = pd.DataFrame(params_rows).sort_values(["date", "hour", "which_hour_before"]).reset_index(drop=True)
        self.params_ = params_df.copy()
        return MVTTrainedModel(cfg=self.cfg.copy(), params=params_df)

   
    def forecast(self, trained: MVTTrainedModel) -> Tuple[pd.DataFrame, pd.DataFrame]:
        cfg = self.cfg
        days = pd.date_range(self.start_day, self.end_day, freq="D")
        preds_rows = []
        mc_seed = int(cfg["seed"])

        for day in days:
            for H in cfg["hours"]:
                for wh in cfg["which_hours_before"]:
                    # Skip if no trained params or failed calibration
                    try:
                        res = trained.as_result(day, int(H), int(wh))
                    except Exception as e:
                        if cfg["verbose"]:
                            print(f"[FORECAST {day.date()}] H={int(H):02d}, WH={int(wh)} -> skip ({e})")
                        continue

                    # Build observation rows for this (day, H, wh)
                    pl = self._build_product_lags(
                        which_hour_before=int(wh),
                        current_lags=cfg["current_lags"],
                        total_neighbors=cfg["total_neighbors"],
                        prefer_before=cfg["prefer_before"],
                        neighbor_lags=cfg["neighbor_lags"],
                    )
                    obs_df = self.cal._collect_feature_rows_for_day(
                        day, int(H), which_hour_before=int(wh),
                        window_minutes=cfg["window_minutes"],
                        product_lags=pl, lag_minutes=cfg["lag_minutes"]
                    )
                    if obs_df.empty or self.target_name not in obs_df.columns:
                        if cfg["verbose"]:
                            print(f"  no scoring rows")
                        continue

                    z_all = [c for c in obs_df.columns if c not in (self.cal._ts, self.target_name)]
                    for _, row in obs_df.iterrows():
                        ts  = pd.to_datetime(row[self.cal._ts])
                        y_true = float(row[self.target_name])
                        observed = {k: float(row[k]) for k in z_all}

                        pred = self.cal.predict(observed=observed, result=res, target_name=self.target_name)
                        mu_y  = pred["mean"]
                        var_y = pred["variance"]
                        nu_c  = pred["df_cond"]

                        err  = y_true - mu_y
                        mae  = abs(err)
                        rmse = np.sqrt(err**2)

                        if np.isfinite(nu_c):
                            scale = float(np.sqrt(max(var_y, 0.0) * (nu_c - 2.0) / nu_c))
                            logp  = float(student_t.logpdf(y_true, df=nu_c, loc=mu_y, scale=scale))
                            crps  = self._crps_student_t_mc(mu_y, scale, nu_c, y_true,
                                                            n=cfg["crps_mc_n"], seed=mc_seed)
                        else:
                            sigma = float(np.sqrt(max(var_y, 0.0)))
                            logp  = float(norm.logpdf(y_true, loc=mu_y, scale=sigma))
                            crps  = self._crps_gaussian(mu_y, sigma, y_true)

                        preds_rows.append({
                            "timestamp": ts,
                            "date": ts.date(),
                            "hour": int(H),
                            "which_hour_before": int(wh),
                            "method": cfg["method"],
                            "y_true": y_true,
                            "mu_cond": mu_y,
                            "var_cond": var_y,
                            "df_cond": nu_c,
                            "error": err,
                            "abs_error": mae,
                            "sq_error": err**2,
                            "rmse_point": rmse,
                            "log_score": -logp,
                            "crps": crps,
                            "used_features": z_all,
                        })
                        mc_seed += 1

        preds_df = pd.DataFrame(preds_rows)
        sort_cols = [c for c in ["date", "hour", "which_hour_before", "timestamp"] if c in preds_df.columns]
        if sort_cols:
            preds_df = preds_df.sort_values(sort_cols).reset_index(drop=True)
        
        self.predictions_ = preds_df.copy()


        summary_df = self.summarize(preds_df)
        return preds_df, summary_df

    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame, MVTTrainedModel]:
        trained = self.train()
        preds, summary = self.forecast(trained)
        return preds, summary, trained

    def calibrate_once(
        self,
        *,
        target_date: pd.Timestamp,
        H: int,
        wh: int,
        window_minutes: int,
        lookback_days: int,
        method: str,
        nu_init: Union[str, Tuple[str, float]],
        verbose_ecm: bool,
        product_lags: Dict[int, List[int]],
        lag_minutes: int,
    ) -> MVTResult:
        return self.cal.calibrate(
            target_date=target_date,
            delivery_start_hour=H,
            which_hour_before=wh,
            window_minutes=window_minutes,
            lookback_days=lookback_days,
            method=method,
            nu_init=nu_init,
            verbose_ecm=verbose_ecm,
            product_lags=product_lags,
            lag_minutes=lag_minutes,
        )

    @staticmethod
    def summarize(predictions: pd.DataFrame) -> pd.DataFrame:
        if predictions is None or predictions.empty:
            return pd.DataFrame(columns=["which_hour_before","hour","method","mae","rmse","log_score","crps","n"])
        g = (predictions
             .groupby(["which_hour_before", "hour", "method"], as_index=False)
             .agg(mae=("abs_error","mean"),
                  rmse=("sq_error", lambda x: np.sqrt(np.mean(x))),
                  log_score=("log_score","mean"),
                  crps=("crps","mean"),
                  n=("y_true","count")))
        return g
    
    



# ======================================================================================
# Helpers
# ======================================================================================

def _infer_global_day_bounds(df: pd.DataFrame, *, delivery_start_col: Optional[str], timestamp_col: str) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    Prefer delivery_start_col if present (your study groups by delivery day),
    otherwise fall back to timestamp_col.
    """
    if delivery_start_col and delivery_start_col in df.columns:
        ser = df[delivery_start_col].dropna()
    else:
        ser = df[timestamp_col].dropna()

    if not np.issubdtype(ser.dtype, np.datetime64):
        ser = pd.to_datetime(ser, errors="coerce")
    ser = ser.dropna()

    days = ser.dt.normalize()
    return days.min(), days.max()


def _iter_day_blocks(start_day: pd.Timestamp, end_day: pd.Timestamp, *, chunk_days: int = 30):
    cur = start_day.normalize()
    end = end_day.normalize()
    one_day = pd.Timedelta(days=1)
    while cur <= end:
        blk_end = min(end, cur + pd.Timedelta(days=chunk_days - 1))
        yield cur, blk_end
        cur = blk_end + one_day


# ======================================================================================
# Chunked runner (saves concatenated predictions/summary as CSV)
# ======================================================================================

def run_study_in_chunks(
    df: pd.DataFrame,
    *,
    chunk_days: int = 30,
    out_preds_path: str = "mvt_preds_all.csv",
    out_summary_path: Optional[str] = None,
    # pass your existing study kwargs here (same names as MVT_forecasting_study.__init__)
    timestamp_col: str = "bin_timestamp",
    delivery_start_col: Optional[str] = "delivery_start_wall",
    delivery_hour_col: Optional[str] = None,
    tz: Optional[str] = None,
    main_feature_col: str = "da-id",
    main_feature_alias: str = "daid",
    other_features: Optional[Dict[str, str]] = None,
    log1p_aliases: Optional[set] = None,
    parse_dayfirst: bool = True,
    lookback_days: int = 90,
    window_minutes: int = 60,
    which_hours_before: Iterable[int] = (4, 3, 2, 1),
    hours: Iterable[int] = tuple(range(24)),
    lag_minutes: int = 15,
    current_lags: int = 3,
    total_neighbors: int = 4,
    prefer_before: int = 2,
    neighbor_lags: Iterable[int] = (1,),
    method: str = "ecm",
    nu_init: Union[str, Tuple[str, float]] = "mardia",
    verbose_ecm: bool = False,
    crps_mc_n: int = 1500,
    seed: int = 123,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Runs training+forecasting in day-blocks (default 30 days) and saves a single
    concatenated predictions CSV. Optionally also saves a CSV summary.
    Returns (preds_all, summary_all).
    """
    # 1) Global day bounds if not explicitly given
    g_start, g_end = _infer_global_day_bounds(
        df, delivery_start_col=delivery_start_col, timestamp_col=timestamp_col
    )

    if start_date is None:
        start_date = str(g_start.date())
    if end_date is None:
        end_date = str(g_end.date())

    start_day = pd.to_datetime(start_date, dayfirst=parse_dayfirst).normalize()
    end_day   = pd.to_datetime(end_date,   dayfirst=parse_dayfirst).normalize()

    # --- NEW: ensure enough lookback data ---
    effective_start = max(start_day, g_start.normalize() + pd.Timedelta(days=lookback_days))
    if effective_start > end_day:
        raise ValueError(
            f"Not enough data: need {lookback_days} days lookback before {start_day.date()}, "
            f"but data only starts at {g_start.date()}"
        )
    start_day = effective_start
    # ----------------------------------------

    # 2) Loop over blocks
    preds_parts: List[pd.DataFrame] = []

    for i, (blk_start, blk_end) in enumerate(_iter_day_blocks(start_day, end_day, chunk_days=chunk_days), start=1):
        if verbose:
            print(f"[Chunk {i}] {blk_start.date()} → {blk_end.date()}")

        study = MVT_forecasting_study(
            df,
            timestamp_col=timestamp_col,
            delivery_start_col=delivery_start_col,
            delivery_hour_col=delivery_hour_col,
            tz=tz,
            main_feature_col=main_feature_col,
            main_feature_alias=main_feature_alias,
            other_features=other_features,
            log1p_aliases=log1p_aliases,
            parse_dayfirst=parse_dayfirst,
            lookback_days=lookback_days,
            window_minutes=window_minutes,
            which_hours_before=which_hours_before,
            hours=hours,
            lag_minutes=lag_minutes,
            current_lags=current_lags,
            total_neighbors=total_neighbors,
            prefer_before=prefer_before,
            neighbor_lags=neighbor_lags,
            method=method,
            nu_init=nu_init,
            verbose_ecm=verbose_ecm,
            crps_mc_n=crps_mc_n,
            seed=seed,
            start_date=str(blk_start.date()),
            end_date=str(blk_end.date()),
            verbose=verbose,
        )

        preds_blk, summary_blk, _trained = study.run()
        if not preds_blk.empty:
            preds_blk = preds_blk.copy()
            preds_blk["chunk_id"] = i
            preds_parts.append(preds_blk)

    # 3) Concatenate predictions and save as CSV
    if preds_parts:
        preds_all = pd.concat(preds_parts, ignore_index=True)
        sort_cols = [c for c in ["date", "hour", "which_hour_before", "timestamp"] if c in preds_all.columns]
        if sort_cols:
            preds_all = preds_all.sort_values(sort_cols).reset_index(drop=True)
    else:
        preds_all = pd.DataFrame()

    preds_all.to_csv(out_preds_path, index=False)
    if verbose:
        print(f"[Done] Saved concatenated predictions → {out_preds_path}")

    # 4) Optional overall summary (also CSV if path provided)
    summary_all = MVT_forecasting_study.summarize(preds_all)
    if out_summary_path:
        summary_all.to_csv(out_summary_path, index=False)
        if verbose:
            print(f"[Done] Saved overall summary → {out_summary_path}")

    return preds_all, summary_all





# ======================================================================================
# Example usage (adjust paths as needed)
# ======================================================================================
if __name__ == "__main__":
    # Example: load and concatenate your years (adjust paths)
    df_2021 = pd.read_csv("df_2021_MVT_dst_10.csv",
                          parse_dates=["bin_timestamp", "delivery_start_wall"],
                          dayfirst=True)
    df_2022 = pd.read_csv("df_2022_MVT_dst_10.csv",
                          parse_dates=["bin_timestamp", "delivery_start_wall"],
                          dayfirst=True)
    df = pd.concat([df_2021, df_2022], ignore_index=True)
    
    df["delivery_start_wall"] = pd.to_datetime(
        df["delivery_start_wall"].astype(str).str.strip(),
        errors="coerce",
        format="%Y-%m-%d %H:%M:%S"
    )
    
    time1 = perf_counter()
    preds_all, summary_all = run_study_in_chunks(
        df,
        chunk_days=30,
        out_preds_path="mvt_preds_all_lb90-2.csv",
        out_summary_path="mvt_summary_all_lb90-2.csv",
        lookback_days=90,          # ← just this
        hours=range(24),
        which_hours_before=(4, 3, 2, 1),
        verbose=True,
    )
    time2 = perf_counter()
    
    preds_all2, summary_all2 = run_study_in_chunks(
        df,
        chunk_days=10,
        out_preds_path="mvt_preds_all_lb365.csv",
        out_summary_path="mvt_summary_all_lb365.csv",
        lookback_days=365,          # ← just this
        hours=range(24),
        which_hours_before=(4, 3, 2, 1),
        verbose=True,
    )
    time3 = perf_counter()