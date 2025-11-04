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
# from scipy.stats import gaussian_kde, t as student_t, norm


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
    def __init__(
        self,
        df: pd.DataFrame,
        *,
        timestamp_col: str = "bin_timestamp",
        delivery_start_col: Optional[str] = "delivery_start",
        delivery_hour_col: Optional[str] = None,
        tz: Optional[str] = "Europe/Amsterdam",
        main_feature_col: str = "vwap_changes",
        main_feature_alias: str = "x",
        other_features: Optional[Dict[str, str]] = None,
        log1p_aliases: Optional[set] = None,
    ):
        self.df = df.copy()
        self._ts = timestamp_col
        if self._ts not in self.df.columns:
            raise ValueError(f"Missing timestamp column: {self._ts}")

        # Parse timestamps and set timezone
        self.df[self._ts] = pd.to_datetime(self.df[self._ts], errors="coerce", utc=True)
        if self.df[self._ts].isna().any():
            bad = self.df[self._ts].isna()
            raise ValueError(f"Column '{self._ts}' has non-parsable timestamps. Bad indices sample: {self.df.index[bad][:3].tolist()}")
        if tz is not None:
            self.df[self._ts] = self.df[self._ts].dt.tz_convert(tz)

        # Delivery hour (and remember delivery_start if available)
        self._ds = None  # keep delivery_start column name if present
        if delivery_hour_col is not None:
            if delivery_hour_col not in self.df.columns:
                raise ValueError(f"Missing delivery_hour column: {delivery_hour_col}")
            self._dh = delivery_hour_col
            self.df[self._dh] = self.df[self._dh].astype(int)
        else:
            if delivery_start_col is None or delivery_start_col not in self.df.columns:
                raise ValueError("Provide delivery_hour_col or a valid delivery_start_col.")
            ds_col = delivery_start_col
            self.df[ds_col] = pd.to_datetime(self.df[ds_col], errors="coerce", utc=True)
            if self.df[ds_col].isna().any():
                raise ValueError(f"Column '{ds_col}' contains non-parsable delivery_start timestamps.")
            if tz is not None:
                self.df[ds_col] = self.df[ds_col].dt.tz_convert(tz)
            self.df["_delivery_hour"] = self.df[ds_col].dt.hour
            self._dh = "_delivery_hour"
            self._ds = ds_col  # <-- remember delivery_start

        # Feature configuration
        self._main_col = main_feature_col
        if self._main_col not in self.df.columns:
            raise ValueError(f"Missing main_feature_col: {self._main_col}")
        self._main_alias = (main_feature_alias or "x").strip()

        # Other features (alias -> column)
        self._other_features: Dict[str, str] = dict(other_features or {})
        for c in self._other_features.values():
            if c not in self.df.columns:
                raise ValueError(f"Missing other-feature column: {c}")

        # Optional log1p
        self._log1p_aliases = set(log1p_aliases or {"mo"})
        for alias, col in list(self._other_features.items()):
            if alias in self._log1p_aliases:
                if (self.df[col].min(skipna=True) <= -1.0):
                    bad_idx = self.df.index[self.df[col] <= -1.0][:3].tolist()
                    raise ValueError(
                        f"Column '{col}' (alias '{alias}') has values ≤ -1; cannot apply log1p. Bad indices: {bad_idx}"
                    )
                new_col = f"__log1p__{col}"
                self.df[new_col] = np.log1p(self.df[col].astype(float))
                self._other_features[alias] = new_col

        self.df.sort_values([self._ts, self._dh], inplace=True)
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
    
    def plot_marginal(
            self,
            feature: str,
            result: Optional[MVTResult] = None,
            *,
            standardized: bool = True,
            bins: Union[str, int] = "auto",
            kde_bw: Optional[float] = None,
            ax: Optional[plt.Axes] = None,
            title: Optional[str] = None,
            hist_kwargs: Optional[dict] = None,
            kde_kwargs: Optional[dict] = None,
            fit_kwargs: Optional[dict] = None,
            alt_dfs: Optional[List[float]] = None,   # <-- NEW
        ) -> plt.Axes:
        """
        Plot empirical marginal (hist + Gaussian KDE) and the fitted Student-t marginal.
        Optionally overlay additional Student-t marginals for custom degrees of freedom.
        """
        res = result if result is not None else self._last_result
        if res is None:
            raise RuntimeError("Run calibrate(...) first.")
        if feature not in res.feature_names:
            raise ValueError(f"feature '{feature}' not in {res.feature_names}")
    
        i = res.feature_names.index(feature)
        scaler = res.scaler
    
        # Raw series (original units as stored in used_data)
        x_raw = res.used_data[feature].to_numpy(float)
    
        # Choose plotting space and corresponding t parameters
        if standardized:
            x = (x_raw - scaler.mean_[i]) / scaler.scale_[i]
            mu_i = res.mu[i]
            sigma2_i = res.Sigma[i, i]
            scale_t = float(np.sqrt(max(sigma2_i, 0.0)))
            loc_t = float(mu_i)
        else:
            x = x_raw.copy()
            mu_i = res.mu[i]
            sigma2_i = res.Sigma[i, i]
            s_i = scaler.scale_[i]
            m_i = scaler.mean_[i]
            scale_t = float(np.sqrt(max(sigma2_i, 0.0)) * s_i)
            loc_t = float(mu_i * s_i + m_i)
    
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 3.2))
    
        # Histogram
        hk = {"bins": 100, "density": True, "alpha": 0.35, "edgecolor": "none", "label": "Histogram"}
        if hist_kwargs:
            hk.update(hist_kwargs)
        ax.hist(x, **hk)
    
        # # KDE
        # try:
        #     kde = gaussian_kde(x) if kde_bw is None else gaussian_kde(x, bw_method=kde_bw)
        #     xs = np.linspace(np.nanmin(x), np.nanmax(x), 400)
        #     ykde = kde(xs)
        #     kk = {"linewidth": 2.0, "label": "KDE"}
        #     if kde_kwargs:
        #         kk.update(kde_kwargs)
        #     ax.plot(xs, ykde, **kk)
        # except Exception:
        #     pass  # skip if KDE fails
    
        # Fitted marginal (from result.df)
        if np.isfinite(res.df):
            df_fit = float(res.df)
            fit_dist = student_t(df=df_fit, loc=loc_t, scale=scale_t)
            fit_label = f"Student-t fit (ν={df_fit:.1f})"
        else:
            fit_dist = norm(loc=loc_t, scale=np.sqrt(scale_t**2))
            fit_label = "Gaussian fit"
    
        xs_fit = np.linspace(np.nanmin(x), np.nanmax(x), 400)
        yf = fit_dist.pdf(xs_fit)
        fk = {"linewidth": 2.0, "linestyle": "--", "label": fit_label}
        if fit_kwargs:
            fk.update(fit_kwargs)
        ax.plot(xs_fit, yf, **fk)
    
        # Additional Student-t curves for requested alt_dfs
        if alt_dfs:
            for nu in alt_dfs:
                try:
                    nu = float(nu)
                    if nu <= 0:
                        continue
                    alt_dist = student_t(df=nu, loc=loc_t, scale=scale_t)
                    ya = alt_dist.pdf(xs_fit)
                    # Use a thin line so the main fit stands out; no explicit color
                    ax.plot(xs_fit, ya, linewidth=1.3, linestyle="-.", label=f"Student-t (ν={nu:g})")
                except Exception:
                    continue
    
        # Titles and labels
        if title is None:
            ttl = f"{feature} | {'standardized' if standardized else 'original units'}"
        else:
            ttl = title
        ax.set_title(ttl)
        ax.set_xlabel(f"{feature} {'(z-score)' if standardized else ''}")
        ax.set_ylabel("density")
        ax.grid(alpha=0.15, linewidth=0.6)
    
        # Legend
        ax.legend()
    
        return ax

    def plot_ll_vs_df(
        self,
        feature: str,
        result: Optional[MVTResult] = None,
        *,
        standardized: bool = True,
        nus: Optional[Union[List[float], np.ndarray]] = None,
        ax: Optional[plt.Axes] = None,
        title: Optional[str] = None,
        ) -> plt.Axes:
        """
        Plot the univariate log-likelihood of the data under a Student-t distribution
        as a function of ν (degrees of freedom).
    
        Shows two curves:
        - Using ECM/MLE μ, Σ (from result).
        - Using empirical μ, Σ (plain mean/cov).
    
        Additionally:
        - NO vertical line at the fitted ν.
        - Places a point at the maximum of each plotted curve (argmax over the ν grid).
        """
        res = result if result is not None else self._last_result
        if res is None:
            raise RuntimeError("Run calibrate(...) first.")
        if feature not in res.feature_names:
            raise ValueError(f"feature '{feature}' not in {res.feature_names}")
    
        i = res.feature_names.index(feature)
        scaler = res.scaler
        x_raw = res.used_data[feature].to_numpy(float)
    
        # Choose plotting space
        if standardized:
            x = (x_raw - scaler.mean_[i]) / scaler.scale_[i]
            mu_ecm = res.mu[i]
            sigma2_ecm = res.Sigma[i, i]
            loc_ecm = float(mu_ecm)
            scale_ecm = float(np.sqrt(max(sigma2_ecm, 0.0)))
            # empirical mean/cov in standardized space
            mu_emp = x.mean()
            sigma2_emp = np.var(x, ddof=1)
            loc_emp = float(mu_emp)
            scale_emp = float(np.sqrt(max(sigma2_emp, 0.0)))
        else:
            x = x_raw.copy()
            mu_ecm = res.mu[i]
            sigma2_ecm = res.Sigma[i, i]
            s_i, m_i = scaler.scale_[i], scaler.mean_[i]
            loc_ecm = float(mu_ecm * s_i + m_i)
            scale_ecm = float(np.sqrt(max(sigma2_ecm, 0.0)) * s_i)
            # empirical mean/cov in original space
            mu_emp = x.mean()
            sigma2_emp = np.var(x, ddof=1)
            loc_emp = float(mu_emp)
            scale_emp = float(np.sqrt(max(sigma2_emp, 0.0)))
    
        # ν grid
        if nus is None:
            nus = np.unique(
                np.concatenate([
                    np.linspace(1.05, 6.0, 150),
                    np.linspace(6.0, 20.0, 120),
                    np.linspace(20.0, 60.0, 80),
                ])
            )
        else:
            nus = np.asarray(nus, dtype=float)
            nus = nus[nus > 0]
            nus = np.unique(nus)
    
        # Log-likelihood curves
        ll_ecm = np.array([
            np.sum(student_t.logpdf(x, df=nu, loc=loc_ecm, scale=scale_ecm))
            for nu in nus
        ])
        ll_emp = np.array([
            np.sum(student_t.logpdf(x, df=nu, loc=loc_emp,  scale=scale_emp))
            for nu in nus
        ])
    
        # Plot
        if ax is None:
            fig, ax = plt.subplots(figsize=(6.5, 3.2))
    
        ax.plot(nus, ll_ecm, label="ECM μ,Σ (MLE)", linewidth=2.0)
        ax.plot(nus, ll_emp, label="Empirical μ,Σ", linewidth=2.0, linestyle="--")
    
        # --- NEW: mark maxima from the plotted data (argmax over grid) ---
        idx_ecm = int(np.nanargmax(ll_ecm))
        idx_emp = int(np.nanargmax(ll_emp))
        nu_ecm_best, ll_ecm_max = float(nus[idx_ecm]), float(ll_ecm[idx_ecm])
        nu_emp_best, ll_emp_max = float(nus[idx_emp]), float(ll_emp[idx_emp])
    
        ax.scatter([nu_ecm_best], [ll_ecm_max], marker="o", s=40,
                   label=f"ECM max  ν={nu_ecm_best:.2f}")
        ax.scatter([nu_emp_best], [ll_emp_max], marker="s", s=40,
                   label=f"Empirical max  ν={nu_emp_best:.2f}")
    
        # Labels
        ax.set_xlabel("ν (degrees of freedom)")
        ax.set_ylabel("log-likelihood")
        ax.grid(alpha=0.15, linewidth=0.6)
        ax.set_title(
            title if title is not None
            else f"Log-likelihood vs ν — {feature} | {'standardized' if standardized else 'original units'}"
        )
        ax.legend()
        return ax





class MVT_forecasting_study:
    """
    Rolling forecasting study for the MVTCalibrator.

    - 90-day rolling window (configurable)
    - hourly retraining for horizons in which_hours_before (e.g., 4,3,2,1)
    - feature design:
        * current product: lags [0..current_lags]
        * total_neighbors neighbouring products, prefer 'prefer_before' previous ones
        * each neighbour uses 'neighbor_lags' (e.g., (1,) or (1,2))
    - metrics: MAE, RMSE, negative log-score, CRPS
    - supports method="ecm" or "mardia"
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
        neighbor_lags: Iterable[int] = (1,),  # NEW
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
        delivery_start_col: Optional[str] = "delivery_start",
        delivery_hour_col: Optional[str] = None,
        tz: Optional[str] = "Europe/Amsterdam",
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
        neighbor_lags: Iterable[int] = (1,),   # NEW

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
        verbose: bool = False,                 # NEW
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
            lookback_days=lookback_days,
            window_minutes=window_minutes,
            which_hours_before=tuple(which_hours_before),
            hours=tuple(hours),
            lag_minutes=lag_minutes,
            current_lags=current_lags,
            total_neighbors=total_neighbors,
            prefer_before=prefer_before,
            neighbor_lags=tuple(neighbor_lags),    # NEW
            method=method.lower(),
            nu_init=nu_init,
            verbose_ecm=verbose_ecm,
            crps_mc_n=crps_mc_n,
            seed=seed,
            start_date=start_date,
            end_date=end_date,
            verbose=verbose,                        # NEW
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
        )
        self.target_name = f"{main_feature_alias}_curr_now"
        self.predictions_: Optional[pd.DataFrame] = None
        self.params_: Optional[pd.DataFrame] = None

        # determine study dates
        if delivery_start_col is not None and delivery_start_col in self.cal.df.columns:
            day_source_col = delivery_start_col
        else:
            day_source_col = self.cal._ts

        tzinfo = self.cal.df[day_source_col].dt.tz
        day_series = self.cal.df[day_source_col].dropna()
        if tz is not None and day_series.dt.tz is not None:
            day_series = day_series.dt.tz_convert(tz)
        days_norm = day_series.dt.normalize()
        all_days = pd.DatetimeIndex(days_norm.unique()).sort_values()

        if start_date is None:
            start_date = (all_days.min() + pd.Timedelta(days=self.cfg["lookback_days"])).date().isoformat()
        if end_date is None:
            end_date = all_days.max().date().isoformat()

        self.start_day = pd.Timestamp(start_date).tz_localize(tzinfo).normalize()
        self.end_day   = pd.Timestamp(end_date).tz_localize(tzinfo).normalize()

    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        cfg = self.cfg
        days = pd.date_range(self.start_day, self.end_day, freq="D")
        preds_rows, params_rows = [], []
        mc_seed = int(cfg["seed"])

        for day in days:
            for H in cfg["hours"]:
                for wh in cfg["which_hours_before"]:
                    # Build the features specification for this horizon (simple & explicit)
                    pl = self._build_product_lags(
                        which_hour_before=int(wh),
                        current_lags=cfg["current_lags"],
                        total_neighbors=cfg["total_neighbors"],
                        prefer_before=cfg["prefer_before"],
                        neighbor_lags=cfg["neighbor_lags"],   # NEW
                    )

                    if cfg["verbose"]:
                        print(f"[{day.date()}] H={int(H):02d}, WH={int(wh)}", flush=True)

                    try:
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
                        cal_ok = True
                        cal_err = ""
                    except Exception as e:
                        res = None
                        cal_ok = False
                        cal_err = str(e)
                        if cfg["verbose"]:
                            print(f"  calibration failed: {cal_err}", flush=True)

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
                        continue

                    obs_df = self.cal._collect_feature_rows_for_day(
                        day, int(H), which_hour_before=int(wh),
                        window_minutes=cfg["window_minutes"],
                        product_lags=pl, lag_minutes=cfg["lag_minutes"]
                    )
                    if obs_df.empty or self.target_name not in obs_df.columns:
                        if cfg["verbose"]:
                            print(f"  no scoring rows", flush=True)
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

        self.predictions_ = (pd.DataFrame(preds_rows)
                             .sort_values(["date", "hour", "which_hour_before", "timestamp"]))
        self.params_      = (pd.DataFrame(params_rows)
                             .sort_values(["date", "hour", "which_hour_before"]))
        return self.predictions_, self.params_

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

    def summarize(self) -> pd.DataFrame:
        if self.predictions_ is None:
            raise RuntimeError("Run .run() first.")
        g = (self.predictions_
             .groupby(["which_hour_before", "hour", "method"])
             .agg(mae=("abs_error","mean"),
                  rmse=("sq_error", lambda x: np.sqrt(np.mean(x))),
                  log_score=("log_score","mean"),
                  crps=("crps","mean"),
                  n=("y_true","count"))
             .reset_index())
        return g


    
    

if __name__ == "__main__":
    df_2021 = pd.read_csv("df_2021_MVT_dst.csv", parse_dates=["bin_timestamp", "delivery_start"])
    df_2022 = pd.read_csv("df_2021_MVT_dst.csv", parse_dates=["bin_timestamp", "delivery_start"])
    df = pd.concat([df_2021, df_2022], ignore_index=True)


        
    cal = MVTCalibrator(
            df,
            timestamp_col="bin_timestamp",
            delivery_start_col="delivery_start",    # OR: delivery_hour_col="delivery_hour"
            tz="Europe/Amsterdam",
            main_feature_col="da-id",
            main_feature_alias="daid",
            # other_features={
            #     "mo": "mo_slope_500",      # will use log1p(mo_slope_500) under the hood, feature: mo_lag1
            # },
        )


    METHOD = "ecm"       #  "ecm" or mardia"
    NU_INIT = "mardia"   # for ECM: "mardia" or ("fixed", 10.0)

    product_lags = {
        0:  [0, 1, 2],   # current product: now, lag1..lag3
       -2: [1],          # previous product (if available): now and lag1
       -1: [1],          # previous product (if available): now and lag1
        1:  [1],            # next product: lag1
        2:  [1],            # next2: lag1
    }
    
    res = cal.calibrate(
        target_date="2021-04-21",
        delivery_start_hour= 12,
        which_hour_before=3,
        window_minutes=60,
        lookback_days=90,
        method="ecm",
        product_lags=product_lags,
        lag_minutes=15,
    )
    print(res.feature_names)
    


    # print("ν (df):", res.df)
    # print("μ (scaled):", res.mu)
    # print("Σ (scale, scaled):", res.Sigma)
    # print("Samples used:", res.n, "dimension:", res.p)
    # print("Features:", res.feature_names)
    
    # # Standardized space (same space ECM is fit in)
    # C_ecm_std = cal.ecm_covariance(res, standardized=True)
    # S_std     = cal.sample_covariance(res, standardized=True)
    
    # # Original units (undo the StandardScaler)
    # C_ecm_orig = cal.ecm_covariance(res, standardized=False)
    # S_orig     = cal.sample_covariance(res, standardized=False)
    
    # # print("Sample cov (std):\n", S_orig)
    # print("ECM cov (std):\n", S_orig)



    # Plot DA–ID (as your main feature) in standardized space
    ax = cal.plot_marginal(
        "daid_curr_now",
        result=res,
        standardized=False,
        alt_dfs=[3, 6.5, 10]
      )

    
    cal.plot_ll_vs_df("daid_curr_now", result=res, standardized=True, nus=np.linspace(0, 25, 100))
    plt.show()


    obs = {
        "x_curr_lag1":  -0.01,
        "x_prev_lag1":  0.005,
        "x_next_lag1":  0.002,
        "da_id_spread": 1.75,
        "merit_order":  0.12,   
    }

    pred = cal.predict(observed=obs)
    print("Conditional mean:", pred["mean"])
    print("Conditional variance:", pred["variance"])
    print("Conditional df:", pred["df_cond"])
    
    # time1 = perf_counter()
    # study1 = MVT_forecasting_study(
    #     df,
    #     start_date="2021-04-02",
    #     end_date="2022-12-31",
    #     timestamp_col="bin_timestamp",
    #     delivery_start_col="delivery_start",
    #     tz="Europe/Amsterdam",
    #     main_feature_col="da-id",
    #     main_feature_alias="daid",
    #     lookback_days=90,
    #     window_minutes=60,
    #     which_hours_before=(4,3,2,1),
    #     hours=range(24),
    #     current_lags=3,
    #     total_neighbors=4,
    #     prefer_before=2,
    #     neighbor_lags=(1,2),   
    #     method="ecm",
    #     nu_init="mardia",
    #     verbose_ecm=False,
    #     crps_mc_n=1500,
    #     seed=123,
    #     verbose=True,            
    
    # )


    # preds1, params1 = study1.run()
    # time2 = perf_counter()
    
    # study2 = MVT_forecasting_study(
    #     df,
    #     start_date="2021-04-02",
    #     end_date="2022-12-31",
    #     timestamp_col="bin_timestamp",
    #     delivery_start_col="delivery_start",
    #     tz="Europe/Amsterdam",
    #     main_feature_col="da-id",
    #     main_feature_alias="daid",
    #     lookback_days=90,
    #     window_minutes=60,
    #     which_hours_before=(4,3,2,1),
    #     hours=range(24),
    #     current_lags=3,
    #     total_neighbors=4,
    #     prefer_before=2,
    #     neighbor_lags=(1, 2),   
    #     method="mardia",
    #     nu_init="mardia",
    #     verbose_ecm=False,
    #     crps_mc_n=1500,
    #     seed=123,
    #     verbose=True,            
    
    # )


    # preds2, params2 = study2.run()
    
    # time3 = perf_counter()
    
    # study3 = MVT_forecasting_study(
    #     df,
    #     start_date="2021-04-02",
    #     end_date="2022-12-31",
    #     timestamp_col="bin_timestamp",
    #     delivery_start_col="delivery_start",
    #     tz="Europe/Amsterdam",
    #     main_feature_col="da-id",
    #     main_feature_alias="daid",
    #     lookback_days=90,
    #     window_minutes=60,
    #     which_hours_before=(4,3,2,1),
    #     hours=range(24),
    #     current_lags=3,
    #     total_neighbors=4,
    #     prefer_before=2,
    #     neighbor_lags=(1),   
    #     method="ecm",
    #     nu_init="mardia",
    #     verbose_ecm=False,
    #     crps_mc_n=1500,
    #     seed=123,
    #     verbose=True,            
    
    # )


    # preds3, params3 = study3.run()
    
    # time4 = perf_counter()
    
    # study4 = MVT_forecasting_study(
    #     df,
    #     start_date="2021-04-02",
    #     end_date="2022-12-31",
    #     timestamp_col="bin_timestamp",
    #     delivery_start_col="delivery_start",
    #     tz="Europe/Amsterdam",
    #     main_feature_col="da-id",
    #     main_feature_alias="daid",
    #     lookback_days=90,
    #     window_minutes=60,
    #     which_hours_before=(4,3,2,1),
    #     hours=range(24),
    #     current_lags=3,
    #     total_neighbors=4,
    #     prefer_before=2,
    #     neighbor_lags=(1),   
    #     method="mardia",
    #     nu_init="mardia",
    #     verbose_ecm=False,
    #     crps_mc_n=1500,
    #     seed=123,
    #     verbose=True,            
    
    # )


    # preds4, params4 = study4.run()
    
    # time5 = perf_counter()
