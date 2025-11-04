import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Optional KDE overlay
try:
    from scipy.stats import gaussian_kde
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

# ── Load & timezone-normalize ─────────────────────────────────────────────
df = pd.read_csv("df_2021_MVT_symm.csv")
for ts_col in ["bin_timestamp", "delivery_start"]:
    dt = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
    df[ts_col] = dt.dt.tz_convert("Europe/Amsterdam")


def plot_marginal_densities_separate(
    df,
    features=("vwap", "vwap_changes", "da-id", "mo_slope_1000"),
    delivery_start_hours=(0, 8, 12, 18),
    which_hour_before=3,
    window_minutes=60,
    bins=150,
    also_scaled=True,
    transform_mo_slope=True,
    trim_upper_quantile=0.995,   # drop values above this quantile for mo_slope_*
    trim_upper_cap=None          # optional absolute cap (e.g., 1.0) applied after quantile step
):
    """
    One figure per (feature × delivery hour).
    For mo_slope_*: make an extra 'trim→log→z-score' plot (outlier removal + log1p + standardize).

    Window = [delivery_start - which_hour_before hours, + window_minutes].
    """

    # Ensure tz-aware datetime columns
    assert str(df["bin_timestamp"].dtype).startswith("datetime64[ns,")
    assert str(df["delivery_start"].dtype).startswith("datetime64[ns,")

    base = df[df["delivery_start"].dt.hour.isin(delivery_start_hours)].copy()
    deliveries = base["delivery_start"].dropna().unique()

    def _robust_xlim(x):
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        if x.size == 0:
            return None
        q1, q99 = np.percentile(x, [1, 99])
        if not np.isfinite(q1) or not np.isfinite(q99) or q1 == q99:
            return None
        pad = 0.05 * (q99 - q1) if q99 > q1 else 1.0
        return (q1 - pad, q99 + pad)

    def _std_scale(x):
        x = np.asarray(x, dtype=float)
        m, s = np.nanmean(x), np.nanstd(x)
        if not np.isfinite(s) or s == 0:
            return None
        return (x - m) / s

    def _plot_hist(data, title, xlabel):
        plt.figure(figsize=(7, 4.2))
        if "changes" in feat:
            plt.hist(data, bins=2*bins, density=True, alpha=0.6)
        else:
            plt.hist(data, bins=bins, density=True, alpha=0.6)
        xlim = _robust_xlim(data)
        if _HAS_SCIPY and data.size >= 50 and np.nanstd(data) > 0:
            kde = gaussian_kde(data)
            if xlim is None:
                lo, hi = np.min(data), np.max(data)
                pad = 0.05 * (hi - lo) if hi > lo else 1.0
                xlim = (lo - pad, hi + pad)
            xs = np.linspace(xlim[0], xlim[1], 512)
            plt.plot(xs, kde(xs), linewidth=2)
        if xlim is not None:
            plt.xlim(xlim)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel("density")
        plt.tight_layout()
        plt.show()

    def _trim_then_log_standardize(x):
        """
        For mo_slope_*:
          1) keep only finite x >= 0
          2) drop upper-tail outliers above the chosen quantile
          3) optionally drop values above an absolute cap
          4) apply log1p
          5) z-score
        """
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        x = x[x >= 0]  # mo_slope should be non-negative
        if x.size == 0:
            return None

        if trim_upper_quantile is not None:
            q_hi = np.quantile(x, trim_upper_quantile)
            x = x[x <= q_hi]
        if trim_upper_cap is not None:
            x = x[x <= float(trim_upper_cap)]
        if x.size == 0:
            return None

        z = _std_scale(np.log1p(x))
        return z

    for hour in delivery_start_hours:
        deliv_this_hour = [t for t in deliveries if pd.Timestamp(t).hour == hour]
        if len(deliv_this_hour) == 0:
            print(f"[Info] No deliveries found with start hour {hour:02d}.")
            continue

        for feat in features:
            vals = []
            for dstart in deliv_this_hour:
                dstart = pd.Timestamp(dstart)
                win_start = dstart - pd.Timedelta(hours=which_hour_before)
                win_end   = win_start + pd.Timedelta(minutes=window_minutes)
                mask = (df["bin_timestamp"] >= win_start) & (df["bin_timestamp"] < win_end)
                x = df.loc[mask, feat].to_numpy(dtype=float, copy=False)
                if x.size:
                    vals.append(x[np.isfinite(x)])

            if len(vals) == 0:
                print(f"[Info] No data for feature '{feat}' at {hour:02d}:00 window.")
                continue

            data = np.concatenate(vals) if len(vals) > 1 else vals[0]
            data = data[np.isfinite(data)]
            if data.size == 0:
                print(f"[Info] All-NaN/inf for feature '{feat}' at {hour:02d}:00.")
                continue

            # 1) RAW
            _plot_hist(
                data,
                f"{feat} | delivery {hour:02d}:00–{(hour+1)%24:02d}:00\nraw",
                feat
            )

            # 2) STANDARDIZED
            if also_scaled:
                z = _std_scale(data)
                if z is not None:
                    _plot_hist(
                        z,
                        f"{feat} | delivery {hour:02d}:00–{(hour+1)%24:02d}:00\nstandardized",
                        f"{feat} (z-score)"
                    )

            # 3) TRIM → LOG → STANDARDIZE for mo_slope_*
            if transform_mo_slope and "mo_slope" in feat:
                z_t = _trim_then_log_standardize(data)
                if z_t is not None:
                    tail = f"(q≤{trim_upper_quantile:.3f})" if trim_upper_quantile is not None else ""
                    cap  = f", cap≤{trim_upper_cap}" if trim_upper_cap is not None else ""
                    desc = f"trim upper {tail}{cap} → log(1+x) → z-score"
                    _plot_hist(
                        z_t,
                        f"{feat} | delivery {hour:02d}:00–{(hour+1)%24:02d}:00\n{desc}",
                        f"{feat} (trim+log z-score)"
                    )

# Example usage
plot_marginal_densities_separate(
    df,
    features=("vwap", "vwap_changes", "da-id", "mo_slope_1000"),
    delivery_start_hours=(11,12,13),
    which_hour_before=3,
    window_minutes=60,
    bins=500,
    also_scaled=True,
    transform_mo_slope=True,
    trim_upper_quantile=0.9,  # drop top 0.5%
    trim_upper_cap=None         # or e.g. 1.2 to add a hard cap
)
