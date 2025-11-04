# -*- coding: utf-8 -*-
"""
Fast MVT calibration + rolling study with vectorized scoring and precomputed feature panel.
Keeps MC-CRPS for Student-t; uses one-time panel construction to avoid repeated merges.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Tuple, Iterable

import numpy as np
import pandas as pd

from scipy import special, optimize
from scipy.stats import norm, t as student_t
from sklearn.preprocessing import StandardScaler

from time import perf_counter


# ---------------------------
# Results container
# ---------------------------

@dataclass
class MVTResult:
    mu: np.ndarray                 # location (scaled space)
    S: np.ndarray                  # sample covariance (scaled space)
    df: Union[float, np.floating]  # ν (np.inf if Gaussian-like)
    Sigma: np.ndarray              # t scale (scaled space); Cov = ν/(ν-2) * Sigma
    n: int
    p: int
    feature_names: List[str]
    used_data: pd.DataFrame        # rows used (original units)
    scaler: StandardScaler         # fitted scaler


# ---------------------------
# Helpers
# ---------------------------

def _logdet_chol(L: np.ndarray) -> float:
    # log |Sigma| from its Cholesky factor L (Sigma = L L^T)
    return 2.0 * np.sum(np.log(np.diag(L)))


def _mahalanobis2_chol(X: np.ndarray, mu: np.ndarray, L: np.ndarray) -> np.ndarray:
    # with Sigma = L L^T
    Y = np.linalg.solve(L, (X - mu).T)
    return np.sum(Y * Y, axis=0)


def _mardia_kurtosis(X: np.ndarray) -> float:
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    S = np.cov(X, rowvar=False, ddof=1)
    S_inv = np.linalg.pinv(S)
    d2 = np.einsum("ni,ij,nj->n", Xc, S_inv, Xc)
    return float(np.mean(d2**2))


def _df_from_mardia(beta_2p: float, p: int, eps: float = 1e-9) -> float:
    k = beta_2p / (p * (p + 2))
    if k <= 1 + eps:
        return float(np.inf)
    return float((4.0 * k - 2.0) / (k - 1.0))


def _t_ll_given_d2(nu: float, d2: np.ndarray, p: int) -> float:
    n = d2.size
    return (
        n * (special.gammaln(0.5 * (nu + p)) - special.gammaln(0.5 * nu) - 0.5 * p * np.log(nu))
        - 0.5 * (nu + p) * np.sum(np.log1p(d2 / nu))
    )


def _update_nu_newton(d2: np.ndarray, p: int, nu0: float, maxit: int = 20, tol: float = 1e-6) -> float:
    """
    Fast Newton update for df (ν). Robustly clamps to [0.6, 1000].
    """
    nu = float(max(nu0, 0.6))
    n = d2.size
    # Precompute terms
    log1p_term = np.log1p(d2 / nu)  # depends on nu; updated each iter

    for _ in range(maxit):
        w = (nu + p) / (nu + d2)             # E[lambda | x] in EM
        g  = special.digamma(0.5 * (nu + p)) - special.digamma(0.5 * nu) - (np.mean(log1p_term)) + 0.5 * (np.mean(w) - 1)
        gp = 0.5 * special.polygamma(1, 0.5 * (nu + p)) - 0.5 * special.polygamma(1, 0.5 * nu) \
             - np.mean((d2 / (nu * (nu + d2)))) - 0.5 * (np.mean(w / (nu + d2)) * (nu + p)) + 0.5 * (np.mean(1 / (nu + d2)))
        # The above derivative is a cautious form; if gp ~ 0, fallback to Brent below
        if not np.isfinite(gp) or abs(gp) < 1e-12:
            break
        step = g / gp
        nu_new = nu - step
        nu_new = float(np.clip(nu_new, 0.6, 1000.0))
        if abs(nu_new - nu) / (1.0 + abs(nu)) < tol:
            nu = nu_new
            return nu
        nu = nu_new
        log1p_term = np.log1p(d2 / nu)
    # Fallback: short Brent
    def obj(x):
        return -_t_ll_given_d2(x, d2, p)
    res = optimize.minimize_scalar(obj, bounds=(0.6, 500.0), method="bounded",
                                   options={"xatol": 1e-4, "maxiter": 60})
    return float(res.x)


# ---------------------------
# Calibrator (with panel)
# ---------------------------

class MVTCalibrator:
    """
    - Build a single lagged panel (timestamp-indexed) once.
    - Calibrate ECM or Mardia on sliced windows from that panel.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        timestamp_col: str = "bin_timestamp",
        delivery_start_col: Optional[str] = "delivery_start_wall",
        delivery_hour_col: Optional[str] = None,
        tz: Optional[str] = None,  # keep None for wall-clock
        main_feature_col: str = "vwap_changes",
        main_feature_alias: str = "x",
        other_features: Optional[Dict[str, str]] = None,
        log1p_aliases: Optional[set] = None,

        # For panel construction
        global_product_lags: Optional[Dict[int, List[int]]] = None,  # union over all horizons
        lag_minutes: int = 15,
    ):
        self._ts = timestamp_col
        self._ds = delivery_start_col
        self._dh = delivery_hour_col
        self._main_col = main_feature_col
        self._main_alias = (main_feature_alias or "x").strip()
        self._lag_minutes = int(lag_minutes)

        if timestamp_col not in df.columns:
            raise ValueError(f"Missing timestamp column: {timestamp_col}")
        if self._dh is None:
            if delivery_start_col is None or delivery_start_col not in df.columns:
                raise ValueError("Provide delivery_hour_col or a valid delivery_start_col.")
            df = df.copy()
            df[delivery_start_col] = pd.to_datetime(df[delivery_start_col], errors="coerce")
            if df[delivery_start_col].isna().any():
                raise ValueError(f"Column '{delivery_start_col}' contains non-parsable timestamps.")
            if tz is not None and getattr(df[delivery_start_col].dt, "tz", None) is not None:
                df[delivery_start_col] = df[delivery_start_col].dt.tz_convert(tz)
            df["_delivery_hour"] = df[delivery_start_col].dt.hour
            self._dh = "_delivery_hour"
        else:
            if self._dh not in df.columns:
                raise ValueError(f"Missing delivery_hour column: {self._dh}")

        # timestamps
        df = df.copy()
        df[self._ts] = pd.to_datetime(df[self._ts], errors="coerce")
        if df[self._ts].isna().any():
            raise ValueError(f"Column '{self._ts}' has non-parsable timestamps.")
        if tz is not None and getattr(df[self._ts].dt, "tz", None) is not None:
            df[self._ts] = df[self._ts].dt.tz_convert(tz)

        if self._main_col not in df.columns:
            raise ValueError(f"Missing main_feature_col: {self._main_col}")

        # Other features
        self._other_features: Dict[str, str] = dict(other_features or {})
        for c in self._other_features.values():
            if c not in df.columns:
                raise ValueError(f"Missing other-feature column: {c}")

        # Optional log1p transforms
        self._log1p_aliases = set(log1p_aliases or {"mo"})
        for alias, col in list(self._other_features.items()):
            if alias in self._log1p_aliases:
                if (df[col].min(skipna=True) <= -1.0):
                    bad_idx = df.index[df[col] <= -1.0][:3].tolist()
                    raise ValueError(
                        f"Column '{col}' (alias '{alias}') has values ≤ -1; cannot apply log1p. Bad indices: {bad_idx}"
                    )
                new_col = f"__log1p__{col}"
                df[new_col] = np.log1p(df[col].astype(float))
                self._other_features[alias] = new_col

        # Save raw for later day filtering
        self.df_raw = df[[self._ts, self._dh] + ([self._ds] if self._ds else []) + [self._main_col] + list(self._other_features.values())].copy()

        # Build the union of product lags (offset -> [lags]) for the whole experiment
        if global_product_lags is None:
            # minimal default: current lag0..1 and neighbour ±1 with lag1
            global_product_lags = {0: [0, 1], -1: [1], +1: [1]}
        self._global_product_lags = {int(k): sorted({int(x) for x in v if int(x) >= 0}) for k, v in (global_product_lags or {}).items()}

        # One-time panel build
        self.panel = self._build_panel(df)

        # Will be filled by calibrate()
        self._last_result: Optional[MVTResult] = None

    # -------- Panel construction --------

    def _name_for(self, alias: str, offset: int, lag: int) -> str:
        if offset == 0:
            return f"{alias}_curr_now" if lag == 0 else f"{alias}_curr_lag{lag}"
        elif offset > 0:
            return f"{alias}_next{offset}_now" if lag == 0 else f"{alias}_next{offset}_lag{lag}"
        else:
            m = abs(offset)
            return f"{alias}_prev{m}_now" if lag == 0 else f"{alias}_prev{m}_lag{lag}"

    def _build_panel(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build a single long panel with all offset/lag columns once.
        Then we slice by (day, hour, window) later.
        """
        ts, dh = self._ts, self._dh
        alias = self._main_alias
        lag_minutes = self._lag_minutes

        # We will collect rows per current-hour h
        out_rows = []
        # Pre-index all per-hour series of the main feature for fast alignment
        per_hour = {
            h: (df[df[dh] == h][[ts, self._main_col]].sort_values(ts).set_index(ts)[self._main_col])
            for h in range(24)
        }

        # current-hour base rows (carry timestamp and hour)
        base = df[[ts, dh]].drop_duplicates().sort_values([dh, ts])

        # For each current-hour h, attach all requested offset/lag columns
        for h in range(24):
            base_h = base[base[dh] == h].set_index(ts)
            if base_h.empty:
                continue

            parts = [base_h.copy()]

            # current product lags (offset 0)
            for lag in self._global_product_lags.get(0, []):
                name = self._name_for(alias, 0, lag)
                s = per_hour[h]
                if lag > 0:
                    s = s.shift(lag, freq=f"{lag_minutes}min")
                parts.append(s.rename(name).to_frame())

            # neighbours (offset != 0)
            for off, L in self._global_product_lags.items():
                if off == 0:
                    continue
                h_off = (h + off) % 24
                s = per_hour.get(h_off)
                if s is None or s.empty:
                    continue
                for lag in L:
                    name = self._name_for(alias, off, lag)
                    s_l = s
                    if lag > 0:
                        s_l = s.shift(lag, freq=f"{lag_minutes}min")
                    parts.append(s_l.rename(name).to_frame())

            # other features (use lag1 by time)
            for alias2, col in self._other_features.items():
                # take values for the *current hour* then shift by lag_minutes
                s = df[df[dh] == h][[ts, col]].sort_values(ts).set_index(ts)[col].shift(1, freq=f"{lag_minutes}min")
                parts.append(s.rename(f"{alias2}_lag1").to_frame())

            # Join all aligned on timestamp index
            panel_h = parts[0]
            for tdf in parts[1:]:
                panel_h = panel_h.join(tdf, how="inner")

            # Bring timestamp back, keep hour
            panel_h = panel_h.reset_index().rename(columns={ts: self._ts})
            out_rows.append(panel_h)

        panel = pd.concat(out_rows, ignore_index=True) if out_rows else pd.DataFrame()
        # Attach delivery_start normalize if available (for day slicing)
        if self._ds and self._ds in self.df_raw.columns:
            ds_norm = (self.df_raw[[self._ts, self._ds]]
                       .drop_duplicates(self._ts)
                       .assign(_ds_day=lambda d: pd.to_datetime(d[self._ds]).dt.normalize())
                       [[self._ts, "_ds_day"]])
            panel = panel.merge(ds_norm, on=self._ts, how="left")
        else:
            panel["_ds_day"] = pd.to_datetime(panel[self._ts]).dt.normalize()

        return panel

    # -------- Data window slicer --------

    def _slice_window(
        self,
        day: pd.Timestamp,
        H: int,
        which_hour_before: int,
        window_minutes: int,
    ) -> pd.DataFrame:
        # Delivery start day constraint (if available)
        day_norm = pd.to_datetime(day).normalize()
        # Window in *wall-clock timestamps*
        delivery_start = day_norm + pd.Timedelta(hours=H)
        start = delivery_start - pd.Timedelta(hours=int(which_hour_before))
        end = start + pd.Timedelta(minutes=int(window_minutes))
        mask = (
            (self.panel[self._dh] == int(H)) &
            (self.panel["_ds_day"] == day_norm) &
            (self.panel[self._ts] >= start) & (self.panel[self._ts] < end)
        )
        return self.panel.loc[mask].copy()

    # -------- ECM internals --------

    def _ecm_fit_scaled(
        self,
        Xs: np.ndarray,
        nu_init: Union[str, Tuple[str, float]] = "mardia",
        max_iter: int = 200,
        tol: float = 1e-6,
        use_newton_nu: bool = True,
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
                beta_2p = _mardia_kurtosis(Xs)
                nu = _df_from_mardia(beta_2p, p=p)
                if not np.isfinite(nu):
                    nu = 30.0
                nu = float(max(nu, 0.6))
            except Exception:
                nu = 4.0

        # Initial loglik
        L = np.linalg.cholesky(Sigma)
        d2 = _mahalanobis2_chol(Xs, mu, L)
        ll_prev = (
            n * (special.gammaln(0.5 * (nu + p)) - special.gammaln(0.5 * nu) - 0.5 * p * np.log(nu * np.pi))
            - _logdet_chol(L) * 0.5 * n
            - 0.5 * (nu + p) * np.sum(np.log1p(d2 / nu))
        )
        if verbose:
            print(f"[init] ll={ll_prev:.3f} nu={nu:.3f}")

        for it in range(1, max_iter + 1):
            # E-step
            w = (nu + p) / (nu + d2)     # (n,)

            # M-step for mu
            sw = float(np.sum(w))
            mu = (w[:, None] * Xs).sum(axis=0) / sw

            # M-step for Sigma (weighted)
            D = Xs - mu
            WD = D * w[:, None]
            S = (WD.T @ D) / n
            tr = float(np.trace(S)) or 1.0
            Sigma = 0.5 * (S + S.T) + 1e-8 * (tr / p) * np.eye(p)

            # Refresh distances and Cholesky
            L = np.linalg.cholesky(Sigma)
            d2 = _mahalanobis2_chol(Xs, mu, L)

            # CM-step for nu
            if use_newton_nu:
                nu = _update_nu_newton(d2, p, nu0=nu, maxit=20, tol=5e-6)
            else:
                # fast Brent fallback
                def obj(x):
                    return -_t_ll_given_d2(x, d2, p)
                res = optimize.minimize_scalar(obj, bounds=(0.6, 200.0), method="bounded",
                                               options={"xatol": 1e-4, "maxiter": 60})
                nu = float(res.x)

            # Check convergence via loglik
            ll = (
                n * (special.gammaln(0.5 * (nu + p)) - special.gammaln(0.5 * nu) - 0.5 * p * np.log(nu * np.pi))
                - _logdet_chol(L) * 0.5 * n
                - 0.5 * (nu + p) * np.sum(np.log1p(d2 / nu))
            )
            if verbose and (it <= 3 or it % 10 == 0):
                print(f"[it {it:02d}] ll={ll:.3f} nu={nu:.4f}")
            if abs(ll - ll_prev) / (1.0 + abs(ll_prev)) < tol:
                break
            ll_prev = ll

        return mu, Sigma, nu, S_emp

    # -------- Public API --------

    def calibrate(
        self,
        *,
        target_date: Union[str, pd.Timestamp],
        delivery_start_hour: int,
        which_hour_before: int,
        window_minutes: int = 60,
        lookback_days: int = 90,
        method: str = "ecm",
        nu_init: Union[str, Tuple[str, float]] = "mardia",
        verbose_ecm: bool = False,
        product_lags: Optional[Dict[int, List[int]]] = None,
    ) -> MVTResult:
        """
        Slice the prebuilt panel over the last `lookback_days` and fit MVT.
        """
        if isinstance(target_date, str):
            target_date = pd.to_datetime(target_date)
        day = pd.to_datetime(target_date).normalize()

        # Build the training window by day loop (fast: just slices)
        start_day = (day - pd.Timedelta(days=lookback_days)).normalize()
        train_chunks = []
        for d in pd.date_range(start_day, day - pd.Timedelta(days=1), freq="D"):
            ch = self._slice_window(d, delivery_start_hour, which_hour_before, window_minutes)
            if not ch.empty:
                train_chunks.append(ch)

        if not train_chunks:
            raise ValueError("No samples found for this configuration.")
        samples = pd.concat(train_chunks, ignore_index=True)

        # Choose feature set for this horizon
        if product_lags is None:
            product_lags = {0: [0, 1], -1: [1], +1: [1]}
        feat_cols = []
        # main feature columns
        for off, L in sorted(product_lags.items(), key=lambda t: (t[0] != 0, t[0] < 0, abs(t[0]))):
            for lag in sorted(set(int(x) for x in L if int(x) >= 0)):
                name = self._name_for(self._main_alias, int(off), int(lag))
                if name in samples.columns:
                    feat_cols.append(name)
        # other features (lag1)
        feat_cols += [f"{alias}_lag1" for alias in self._other_features.keys()]
        feat_cols = [c for c in feat_cols if c in samples.columns]

        if not feat_cols:
            raise ValueError("No feature columns selected for this horizon.")

        X = samples[feat_cols].to_numpy(float)
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        n, p = Xs.shape
        S_emp = np.cov(Xs, rowvar=False, ddof=1)

        if method.lower() == "ecm":
            mu_hat, Sigma_hat, nu_hat, _ = self._ecm_fit_scaled(
                Xs, nu_init=nu_init, verbose=verbose_ecm
            )
            mu, Sigma, nu, S = mu_hat, Sigma_hat, nu_hat, S_emp
        elif method.lower() == "mardia":
            mu = Xs.mean(axis=0)
            S = S_emp
            beta_2p = _mardia_kurtosis(Xs)
            nu = _df_from_mardia(beta_2p, p=p)
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

    # Vectorized conditional prediction for a whole dataframe block
    def predict_block(
        self,
        df_block: pd.DataFrame,
        *,
        result: Optional[MVTResult] = None,
        target_name: Optional[str] = None,
    ) -> Dict[str, np.ndarray]:
        res = result if result is not None else self._last_result
        if res is None:
            raise RuntimeError("Run calibrate(...) first.")
        feat = res.feature_names
        if target_name is None:
            target_name = f"{self._main_alias}_curr_now"
        if target_name not in feat:
            raise ValueError(f"target_name '{target_name}' not in features {feat}")

        y_idx = feat.index(target_name)
        idx_z = np.array([i for i in range(len(feat)) if i != y_idx])
        z_names = [feat[i] for i in idx_z]

        # Build Z and standardize
        Z = df_block[z_names].to_numpy(float)
        scaler = res.scaler
        Zs = (Z - scaler.mean_[idx_z]) / scaler.scale_[idx_z]

        # Pull parameters
        mu = res.mu
        Sigma = res.Sigma

        mu_y = mu[y_idx]
        mu_z = mu[idx_z]

        Σ_yy = Sigma[y_idx, y_idx]
        Σ_yz = Sigma[y_idx, idx_z]
        Σ_zz = Sigma[np.ix_(idx_z, idx_z)]

        # Vectorized conditionals
        diff = Zs - mu_z                             # (m, p-1)
        # Use Cholesky for Σ_zz inverse and δ_z
        Lz = np.linalg.cholesky(Σ_zz)
        U = np.linalg.solve(Lz, diff.T)              # (p-1, m)
        delta_z = np.sum(U * U, axis=0)              # (m,)
        # Solve Σ_zz^{-1} Σ_zy once
        Σ_zy = Sigma[np.ix_(idx_z, [y_idx])][:, 0]
        v = np.linalg.solve(Lz, Σ_zy)                # Lz v1 = Σ_zy
        Σzz_inv_Σzy = np.linalg.solve(Lz.T, v)       # Σ_zz^{-1} Σ_zy
        A = (Σ_yz @ np.linalg.inv(Σ_zz)) if len(idx_z) <= 3 else (Σ_yz @ np.linalg.solve(Σ_zz, np.eye(len(idx_z))))
        # more stable with cho_solve:
        # from scipy.linalg import cho_solve
        # A = Σ_yz @ cho_solve((Lz, True), np.eye(len(idx_z)))

        mean_scaled = mu_y + diff @ A.T              # (m,)
        schur = float(Σ_yy - A @ Σ_zy)

        if np.isfinite(res.df):
            ν = float(res.df)
            q = len(idx_z)
            ν_star = ν + q
            scale_cond = ((ν + delta_z) / (ν + q - 2.0)) * schur
            var_scaled = (ν_star / (ν_star - 2.0)) * scale_cond
            df_cond = np.full(diff.shape[0], ν_star, dtype=float)
        else:
            var_scaled = np.full(diff.shape[0], schur, dtype=float)
            df_cond = np.full(diff.shape[0], np.inf, dtype=float)

        s_y = scaler.scale_[y_idx]
        m_y = scaler.mean_[y_idx]
        mean_orig = mean_scaled * s_y + m_y
        var_orig = var_scaled * (s_y ** 2)

        return {
            "mean": mean_orig,                # (m,)
            "variance": var_orig,             # (m,)
            "mean_scaled": mean_scaled,       # (m,)
            "variance_scaled": var_scaled,    # (m,)
            "df_cond": df_cond,               # (m,)
            "z_names": z_names
        }


# ---------------------------
# Rolling forecasting study (fast)
# ---------------------------

class MVT_forecasting_study:
    """
    Fast rolling study:
    - Prebuilt panel slicing
    - Vectorized prediction per block
    - Vectorized MC-CRPS for Student-t
    """

    @staticmethod
    def _crps_gaussian_vec(mu: np.ndarray, sigma: np.ndarray, y: np.ndarray) -> np.ndarray:
        z = (y - mu) / np.maximum(sigma, 1e-12)
        # scalar vectorized formula
        return sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))

    @staticmethod
    def _crps_student_t_mc_vec(mu: np.ndarray, scale: np.ndarray, df: float, y: np.ndarray,
                               n: int = 1500, seed: int = 123) -> np.ndarray:
        """
        Vectorized MC-CRPS when df is constant for the block (true for conditional t: df = ν+q).
        Draw (n, m) iid t, scale and shift by vectors, then compute CRPS per column.
        """
        if n <= 0:
            return np.zeros_like(mu, dtype=float)
        rng = np.random.default_rng(seed)
        m = mu.size
        # Draw base standard t(nu) for matrix (n, m)
        T1 = student_t.rvs(df, size=(n, m), random_state=rng)
        T2 = student_t.rvs(df, size=(n, m), random_state=rng)
        # scale/shift per column
        X1 = T1 * scale[None, :] + mu[None, :]
        X2 = T2 * scale[None, :] + mu[None, :]
        # CRPS = E|X - y| - 0.5 E|X - X'|
        term1 = np.mean(np.abs(X1 - y[None, :]), axis=0)
        term2 = 0.5 * np.mean(np.abs(X1 - X2), axis=0)
        return term1 - term2

    @staticmethod
    def _build_product_lags(
        which_hour_before: int,
        current_lags: int = 3,
        total_neighbors: int = 4,
        prefer_before: int = 2,
        neighbor_lags: Iterable[int] = (1,),
    ) -> Dict[int, List[int]]:
        pl: Dict[int, List[int]] = {0: list(range(0, int(current_lags) + 1))}
        nl = sorted({int(l) for l in neighbor_lags if int(l) >= 1})
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
        # Build the *union* of product lags needed in this study (for panel)
        union_pl: Dict[int, set] = {}
        for wh in which_hours_before:
            pl = self._build_product_lags(
                which_hour_before=int(wh),
                current_lags=current_lags,
                total_neighbors=total_neighbors,
                prefer_before=prefer_before,
                neighbor_lags=neighbor_lags,
            )
            for k, v in pl.items():
                union_pl.setdefault(k, set()).update(v)
        union_pl = {k: sorted(v) for k, v in union_pl.items()}

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
            global_product_lags=union_pl,
            lag_minutes=lag_minutes,
        )

        self.cfg = dict(
            lookback_days=lookback_days,
            window_minutes=window_minutes,
            which_hours_before=tuple(which_hours_before),
            hours=tuple(hours),
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
        self.target_name = f"{main_feature_alias}_curr_now"
        self.predictions_: Optional[pd.DataFrame] = None
        self.params_: Optional[pd.DataFrame] = None

        # Study date range from raw df (by delivery_start if present, else timestamps)
        date_col = delivery_start_col if (delivery_start_col and delivery_start_col in df.columns) else timestamp_col
        days_norm = pd.to_datetime(df[date_col]).dt.normalize()
        all_days = pd.DatetimeIndex(days_norm.unique()).sort_values()
        if start_date is None:
            start_date = (all_days.min() + pd.Timedelta(days=lookback_days)).date().isoformat()
        if end_date is None:
            end_date = all_days.max().date().isoformat()
        self.start_day = pd.to_datetime(start_date).normalize()
        self.end_day   = pd.to_datetime(end_date).normalize()

    # Single calibration wrapper
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
        )

    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        cfg = self.cfg
        days = pd.date_range(self.start_day, self.end_day, freq="D")
        preds_rows, params_rows = [], []
        rng_seed = int(cfg["seed"])

        for day in days:
            for H in cfg["hours"]:
                for wh in cfg["which_hours_before"]:
                    # Horizon-specific product lags (subset of union)
                    pl = self._build_product_lags(
                        which_hour_before=int(wh),
                        current_lags=cfg["current_lags"],
                        total_neighbors=cfg["total_neighbors"],
                        prefer_before=cfg["prefer_before"],
                        neighbor_lags=cfg["neighbor_lags"],
                    )
                    if cfg["verbose"]:
                        print(f"[{day.date()}] H={int(H):02d}, WH={int(wh)}", flush=True)

                    # Calibrate
                    try:
                        res = self.calibrate_once(
                            target_date=day, H=int(H), wh=int(wh),
                            window_minutes=cfg["window_minutes"],
                            lookback_days=cfg["lookback_days"],
                            method=cfg["method"],
                            nu_init=cfg["nu_init"],
                            verbose_ecm=cfg["verbose_ecm"],
                            product_lags=pl,
                        )
                        cal_ok = True
                        cal_err = ""
                    except Exception as e:
                        res = None
                        cal_ok = False
                        cal_err = str(e)

                    params_rows.append({
                        "date": day.date(),
                        "hour": int(H),
                        "which_hour_before": int(wh),
                        "method": cfg["method"],
                        "calibration_ok": cal_ok,
                        "error": cal_err if not cal_ok else "",
                        **({} if not cal_ok else {
                            "df": float(res.df),
                            "n": res.n,
                            "p": res.p,
                            "feature_names": list(res.feature_names),
                            "mu_scaled": res.mu.tolist(),
                            "Sigma_scaled": res.Sigma.tolist(),
                            "S_sample_scaled": res.S.tolist(),
                        })
                    })
                    if not cal_ok:
                        if cfg["verbose"]:
                            print(f"  calibration failed: {cal_err}", flush=True)
                        continue

                    # Scoring slice for this *same* (day,H,wh,window)
                    obs_df = self.cal._slice_window(day, int(H), int(wh), cfg["window_minutes"])
                    # Keep only the columns that the model expects
                    needed = [self.cal._ts, self.target_name] + [c for c in res.feature_names if c != self.target_name]
                    obs_df = obs_df[[c for c in needed if c in obs_df.columns]].dropna()
                    if obs_df.empty or self.target_name not in obs_df.columns:
                        if cfg["verbose"]:
                            print("  no scoring rows", flush=True)
                        continue

                    # Vectorized prediction
                    y_true = obs_df[self.target_name].to_numpy(float)
                    pred = self.cal.predict_block(obs_df, result=res, target_name=self.target_name)
                    mu_y  = pred["mean"]
                    var_y = pred["variance"]
                    df_c  = pred["df_cond"]

                    err  = y_true - mu_y
                    mae  = np.abs(err)
                    rmse = np.sqrt(err**2)

                    # Log-score (vectorized)
                    if np.isfinite(df_c).all():
                        df_unique = float(df_c[0])  # same across rows in a block (ν+q)
                        scale = np.sqrt(np.maximum(var_y, 0.0) * (df_unique - 2.0) / df_unique)
                        logp  = student_t.logpdf(y_true, df=df_unique, loc=mu_y, scale=scale)
                    else:
                        sigma = np.sqrt(np.maximum(var_y, 0.0))
                        logp  = norm.logpdf(y_true, loc=mu_y, scale=sigma)

                    # CRPS
                    if np.isfinite(df_c).all():
                        df_unique = float(df_c[0])
                        scale = np.sqrt(np.maximum(var_y, 0.0) * (df_unique - 2.0) / df_unique)
                        crps = self._crps_student_t_mc_vec(mu_y, scale, df_unique, y_true,
                                                           n=cfg["crps_mc_n"], seed=rng_seed)
                        rng_seed += 1
                    else:
                        sigma = np.sqrt(np.maximum(var_y, 0.0))
                        crps = self._crps_gaussian_vec(mu_y, sigma, y_true)

                    # Collect rows (vectorized to list of dicts)
                    block = pd.DataFrame({
                        "timestamp": pd.to_datetime(obs_df[self.cal._ts].to_numpy()),
                        "date": pd.to_datetime(obs_df[self.cal._ts]).dt.date.to_numpy(),
                        "hour": int(H),
                        "which_hour_before": int(wh),
                        "method": cfg["method"],
                        "y_true": y_true,
                        "mu_cond": mu_y,
                        "var_cond": var_y,
                        "df_cond": df_c,
                        "error": err,
                        "abs_error": mae,
                        "sq_error": err**2,
                        "rmse_point": rmse,
                        "log_score": -logp,
                        "crps": crps,
                    })
                    preds_rows.append(block)

        self.predictions_ = (pd.concat(preds_rows, ignore_index=True)
                             .sort_values(["date", "hour", "which_hour_before", "timestamp"]))
        self.params_ = (pd.DataFrame(params_rows)
                        .sort_values(["date", "hour", "which_hour_before"]))
        return self.predictions_, self.params_

    def summarize(self) -> pd.DataFrame:
        if self.predictions_ is None:
            raise RuntimeError("Run .run() first.")
        g = (self.predictions_
             .groupby(["which_hour_before", "hour", "method"])
             .agg(mae=("abs_error", "mean"),
                  rmse=("sq_error", lambda x: np.sqrt(np.mean(x))),
                  log_score=("log_score", "mean"),
                  crps=("crps", "mean"),
                  n=("y_true", "count"))
             .reset_index())
        return g


# ---------------------------
# Example main
# ---------------------------

if __name__ == "__main__":
    # Load data (wall-clock)
    df_2021 = pd.read_csv(
        "df_2021_MVT_dst_10.csv",
        parse_dates=["bin_timestamp", "delivery_start_wall"],
        dayfirst=True
    )
    df_2022 = pd.read_csv(
        "df_2022_MVT_dst_10.csv",
        parse_dates=["bin_timestamp", "delivery_start_wall"],
        dayfirst=True
    )
    df = pd.concat([df_2021, df_2022], ignore_index=True)

    # Single-day (ECM)
    t0 = perf_counter()
    study1 = MVT_forecasting_study(
        df,
        start_date="2022-05-01",
        end_date="2022-05-30",
        timestamp_col="bin_timestamp",
        delivery_start_col="delivery_start_wall",
        tz=None,
        main_feature_col="da-id",
        main_feature_alias="daid",
        lookback_days=90,
        window_minutes=60,
        which_hours_before=(4, 3, 2, 1),
        hours=range(24),
        current_lags=3,
        total_neighbors=4,
        prefer_before=2,
        neighbor_lags=(1,),
        method="ecm",
        nu_init="mardia",
        verbose_ecm=False,
        crps_mc_n=1000,
        seed=123,
        verbose=True,
    )
    preds1, params1 = study1.run()
    t1 = perf_counter()

    # Multi-day (Mardia)
    study2 = MVT_forecasting_study(
        df,
        start_date="2022-05-01",
        end_date="2022-05-31",
        timestamp_col="bin_timestamp",
        delivery_start_col="delivery_start_wall",
        tz=None,
        main_feature_col="da-id",
        main_feature_alias="daid",
        lookback_days=90,
        window_minutes=60,
        which_hours_before=(4, 3, 2, 1),
        hours=range(24),
        current_lags=3,
        total_neighbors=4,
        prefer_before=2,
        neighbor_lags=(1, 2),
        method="mardia",
        nu_init="mardia",
        verbose_ecm=False,
        crps_mc_n=1500,
        seed=103,
        verbose=True,
    )
    preds2, params2 = study2.run()
    t2 = perf_counter()

    # Multi-day (Mardia)
    study3 = MVT_forecasting_study(
        df,
        start_date="2022-05-01",
        end_date="2022-05-31",
        timestamp_col="bin_timestamp",
        delivery_start_col="delivery_start_wall",
        tz=None,
        main_feature_col="da-id",
        main_feature_alias="daid",
        lookback_days=365,
        window_minutes=60,
        which_hours_before=(4, 3, 2, 1),
        hours=range(24),
        current_lags=3,
        total_neighbors=4,
        prefer_before=2,
        neighbor_lags=(1, 2),
        method="ecm",
        nu_init="mardia",
        verbose_ecm=False,
        crps_mc_n=1500,
        seed=103,
        verbose=True,
    )
    preds3, params3 = study3.run()
    t3 = perf_counter()
    
    # Multi-day (Mardia)
    study4 = MVT_forecasting_study(
        df,
        start_date="2022-05-01",
        end_date="2022-05-31",
        timestamp_col="bin_timestamp",
        delivery_start_col="delivery_start_wall",
        tz=None,
        main_feature_col="da-id",
        main_feature_alias="daid",
        lookback_days=365,
        window_minutes=60,
        which_hours_before=(4, 3, 2, 1),
        hours=range(24),
        current_lags=3,
        total_neighbors=4,
        prefer_before=2,
        neighbor_lags=(1, 2),
        method="mardia",
        nu_init="mardia",
        verbose_ecm=False,
        crps_mc_n=1500,
        seed=103,
        verbose=True,
    )
    preds4, params4 = study4.run()
    t4 = perf_counter()

    
    # Quick summaries
    try:
        print("\nStudy1 summary:\n", study1.summarize().head())
    except Exception:
        pass
    try:
        print("\nStudy2 summary:\n", study2.summarize().head())
    except Exception:
        pass

    print(f"\nTiming: study1={(t1-t0):.2f}s, study2={(t2-t1):.2f}s")
