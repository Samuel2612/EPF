# -*- coding: utf-8 -*-
"""
Created on Thu Sep 11 10:27:04 2025

@author: samue
"""

# ------------------------ helpers: metrics & product-lags ------------------------

from typing import Iterable
from scipy.stats import norm, t as student_t

def _build_product_lags(which_hour_before: int,
                        current_lags: int = 3,
                        total_neighbors: int = 4,
                        prefer_before: int = 2) -> Dict[int, List[int]]:
    """
    Build {offset -> [lags]} with:
      - current product: lags [0,1,2,...,current_lags]
      - neighbors: total_neighbors offsets, prefer `prefer_before` previous if available
      - neighbor lags = [1] (alignment like before)
    Availability: previous offset -m only if which_hour_before > m.
    """
    product_lags: Dict[int, List[int]] = {0: list(range(0, current_lags + 1))}

    max_prev_available = max(0, int(which_hour_before) - 1)  # prev m allowed if which_hour_before>m
    prev_to_use = min(prefer_before, max_prev_available)
    next_to_use = max(0, total_neighbors - prev_to_use)

    # previous: -1, -2, ...
    for m in range(1, prev_to_use + 1):
        product_lags[-m] = [1]

    # next: +1, +2, ...
    for m in range(1, next_to_use + 1):
        product_lags[m] = [1]

    # if not enough prev were available (due to availability), fill with more next
    extra_needed = total_neighbors - (prev_to_use + next_to_use)
    k = next_to_use
    while extra_needed > 0:
        k += 1
        product_lags[k] = [1]
        extra_needed -= 1

    return product_lags


def _crps_gaussian(mu: float, sigma: float, y: float) -> float:
    z = (y - mu) / max(sigma, 1e-12)
    return float(sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi)))


def _crps_student_t_mc(mu: float, scale: float, df: float, y: float, n: int = 2000, seed: int = 123) -> float:
    """
    Monte-Carlo CRPS:  E|X-y| - 0.5 E|X-X'|  with X ~ t_nu(mu, scale).
    Accurate enough for backtests; keep n moderate.
    """
    rng = np.random.default_rng(seed)
    x = student_t.rvs(df, loc=mu, scale=scale, size=n, random_state=rng)
    x2 = student_t.rvs(df, loc=mu, scale=scale, size=n, random_state=rng)
    return float(np.mean(np.abs(x - y)) - 0.5 * np.mean(np.abs(x - x2)))


# ---------------------------- main backtest function ----------------------------

def run_forecasting_study(
    df: pd.DataFrame,
    *,
    # Calibrator config (adapt if your columns differ)
    timestamp_col: str = "bin_timestamp",
    delivery_start_col: Optional[str] = "delivery_start",
    delivery_hour_col: Optional[str] = None,
    tz: Optional[str] = "Europe/Amsterdam",
    main_feature_col: str = "da-id",
    main_feature_alias: str = "daid",
    other_features: Optional[Dict[str, str]] = None,
    log1p_aliases: Optional[set] = None,

    # Study horizon
    start_date: Optional[str] = None,        # inclusive (YYYY-MM-DD); default = auto from data
    end_date: Optional[str] = None,          # inclusive; default = auto from data
    hours: Iterable[int] = tuple(range(24)), # which delivery_start_hour(s)

    # Rolling window + retraining cadence
    lookback_days: int = 90,
    window_minutes: int = 60,
    which_hours_before: Iterable[int] = (4, 3, 2, 1),

    # Feature design
    current_lags: int = 3,
    total_neighbors: int = 4,
    prefer_before: int = 2,
    lag_minutes: int = 15,

    # Model choice
    method: str = "ecm",                     # "ecm" or "mardia"
    nu_init: Union[str, Tuple[str, float]] = "mardia",
    verbose_ecm: bool = False,

    # Metrics
    crps_mc_n: int = 1500,                   # MC draws for t-CRPS
    seed: int = 123,

) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      predictions_df: one row per forecast (timestamp of obs inside the window),
        with errors & scores.
      params_df:      one row per calibration (per date-hour-which_before), with μ, Σ, ν.
    """
    # 1) Set up calibrator (once; it reuses the same df but re-fits each step)
    cal = MVTCalibrator(
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
    target_name = f"{main_feature_alias}_curr_now"

    # 2) Choose study date range
    all_days = pd.DatetimeIndex(df[timestamp_col].dropna()).tz_convert(tz).normalize().unique().sort_values()
    if start_date is None:
        start_date = (all_days.min() + pd.Timedelta(days=lookback_days)).date().isoformat()
    if end_date is None:
        end_date = all_days.max().date().isoformat()

    start_day = pd.Timestamp(start_date).tz_localize(df[timestamp_col].dt.tz)
    end_day   = pd.Timestamp(end_date).tz_localize(df[timestamp_col].dt.tz)

    days = pd.date_range(start_day.normalize(), end_day.normalize(), freq="D")
    if len(days) == 0:
        raise ValueError("No days in selected range.")

    preds_rows = []
    params_rows = []

    # reproducible seed for CRPS MC
    mc_seed = int(seed)

    for day in days:
        for H in hours:
            for wh in which_hours_before:
                # 3) Build features spec for this horizon
                pl = _build_product_lags(wh, current_lags=current_lags,
                                         total_neighbors=total_neighbors, prefer_before=prefer_before)

                # 4) Calibrate on a 90-day rolling window up to (day - 1)
                try:
                    res = cal.calibrate(
                        target_date=day,
                        delivery_start_hour=int(H),
                        which_hour_before=int(wh),
                        window_minutes=window_minutes,
                        lookback_days=lookback_days,
                        method=method,
                        nu_init=nu_init,
                        verbose_ecm=verbose_ecm,
                        product_lags=pl,
                        lag_minutes=lag_minutes,
                    )
                except Exception as e:
                    # store calibration failure
                    params_rows.append({
                        "date": day.date(), "hour": H, "which_hour_before": wh,
                        "method": method, "calibration_ok": False, "error": str(e)
                    })
                    continue

                params_rows.append({
                    "date": day.date(),
                    "hour": H,
                    "which_hour_before": wh,
                    "method": method,
                    "calibration_ok": True,
                    "df": float(res.df),
                    "n": res.n,
                    "p": res.p,
                    "feature_names": list(res.feature_names),
                    "mu_scaled": res.mu.tolist(),
                    "Sigma_scaled": res.Sigma.tolist(),  # t scale matrix
                    "S_sample_scaled": res.S.tolist(),
                })

                # 5) Build observation rows for the target day/hour at this horizon
                obs_df = cal._collect_feature_rows_for_day(
                    day, H, which_hour_before=int(wh),
                    window_minutes=window_minutes, product_lags=pl, lag_minutes=lag_minutes
                )
                if obs_df.empty or target_name not in obs_df.columns:
                    continue

                # Use the **last** timestamp in the window as the hourly forecast target
                row = obs_df.iloc[-1]
                ts  = pd.to_datetime(row[cal._ts])

                # (a) true target
                y_true = float(row[target_name])

                # (b) conditioning values (all except ts and target)
                z_names = [c for c in obs_df.columns if c not in (cal._ts, target_name)]
                observed = {k: float(row[k]) for k in z_names}

                # (c) predict conditional distribution
                pred = cal.predict(observed=observed, result=res, target_name=target_name)
                mu_y  = pred["mean"]
                var_y = pred["variance"]
                nu_c  = pred["df_cond"]

                # Metrics
                err   = y_true - mu_y
                mae   = abs(err)
                rmse  = np.sqrt(err**2)

                # log score & CRPS
                if np.isfinite(nu_c):
                    # t: derive "scale" from variance: var = scale^2 * nu/(nu-2)
                    scale = float(np.sqrt(max(var_y, 0.0) * (nu_c - 2.0) / nu_c))
                    logp  = float(student_t.logpdf(y_true, df=nu_c, loc=mu_y, scale=scale))
                    crps  = _crps_student_t_mc(mu_y, scale, nu_c, y_true, n=crps_mc_n, seed=mc_seed)
                else:
                    sigma = float(np.sqrt(max(var_y, 0.0)))
                    logp  = float(norm.logpdf(y_true, loc=mu_y, scale=sigma))
                    crps  = _crps_gaussian(mu_y, sigma, y_true)

                preds_rows.append({
                    "timestamp": ts,
                    "date": ts.date(),
                    "hour": H,
                    "which_hour_before": wh,
                    "method": method,
                    "y_true": y_true,
                    "mu_cond": mu_y,
                    "var_cond": var_y,
                    "df_cond": nu_c,
                    "error": err,
                    "abs_error": mae,
                    "sq_error": err**2,
                    "rmse_point": rmse,
                    "log_score": -logp,        # lower is better
                    "crps": crps,              # lower is better
                    "used_features": z_names,
                })

                mc_seed += 1  # vary seed slightly across steps

    predictions_df = pd.DataFrame(preds_rows).sort_values(["date", "hour", "which_hour_before"])
    params_df      = pd.DataFrame(params_rows).sort_values(["date", "hour", "which_hour_before"])
    return predictions_df, params_df
