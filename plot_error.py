# plot_errors_intraday_bars.py
from typing import Dict, Iterable, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _ensure_int(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("Int64").astype("float").astype(int)


def _infer_whb_from_ttd(df: pd.DataFrame) -> Optional[pd.Series]:
    """
    Map ttd to which_hour_before via ceil(ttd/4) -> {1,2,3,4}.
    Example: ttd 16..13 → 4; 12..9 → 3; 8..5 → 2; 4..1 → 1.
    """
    if "ttd" not in df.columns:
        return None
    ttd = pd.to_numeric(df["ttd"], errors="coerce")
    whb = np.ceil(ttd / 4.0).astype("Int64").clip(lower=1, upper=4)
    return whb.astype(int)


def _pick_error_cols(df: pd.DataFrame) -> Tuple[str, str]:
    """
    Return (abs_col, sq_col) names depending on schema.
    """
    if "abs_error" in df.columns and "sq_error" in df.columns:
        return "abs_error", "sq_error"
    if "abs_err" in df.columns and "sq_err" in df.columns:
        return "abs_err", "sq_err"
    raise ValueError(
        "Could not find error columns. "
        "Expected ('abs_error','sq_error') or ('abs_err','sq_err')."
    )


def _normalize_df(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """
    Normalize:
      - hour in [0..23]
      - which_hour_before present (direct or inferred)
      - attach source label
      - keep only rows with usable errors
    """
    df = df.copy()

    if "hour" not in df.columns:
        raise ValueError(f"[{source}] Missing required column 'hour'.")

    df["hour"] = _ensure_int(df["hour"]).clip(lower=0, upper=23)

    # Ensure which_hour_before
    if "which_hour_before" in df.columns:
        df["which_hour_before"] = _ensure_int(df["which_hour_before"]).clip(1, 4)
    else:
        whb = _infer_whb_from_ttd(df)
        df["which_hour_before"] = _ensure_int(whb) if whb is not None else np.nan

    abs_col, sq_col = _pick_error_cols(df)

    # coerce errors to numeric and drop rows missing them
    df[abs_col] = pd.to_numeric(df[abs_col], errors="coerce")
    df[sq_col]  = pd.to_numeric(df[sq_col], errors="coerce")
    df = df.dropna(subset=[abs_col, sq_col])

    df["__abs_err__"] = df[abs_col]
    df["__sq_err__"]  = df[sq_col]
    df["source"] = str(source)
    return df


def _agg_mae_rmse_by(df: pd.DataFrame, by_cols: Iterable[str]) -> pd.DataFrame:
    """
    Aggregate to MAE and RMSE over given grouping keys.
    We average across any present 'method' automatically (one line/bar per source).
    """
    g = df.groupby(list(by_cols), dropna=False)[["__abs_err__", "__sq_err__"]].mean(numeric_only=True)
    out = g.reset_index()
    out["MAE"]  = out["__abs_err__"]
    out["RMSE"] = np.sqrt(out["__sq_err__"].clip(lower=0))
    return out.drop(columns=["__abs_err__", "__sq_err__"])


def _plot_lines(y_map, title, xlabel, ylabel, xticks=None, save_path=None):
    """
    y_map: dict {label -> (x_array, y_array)}; draws one line per label.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    markers = ["o", "s", "D", "^", "v", "P", "X", "*"]
    linestyles = ["-", "--", "-.", ":"]
    for i, (label, (x, y)) in enumerate(y_map.items()):
        ax.plot(x, y,
                marker=markers[i % len(markers)],
                linestyle=linestyles[i % len(linestyles)],
                label=label)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xticks is not None:
        ax.set_xticks(xticks)
    ax.grid(True, alpha=0.3)
    ax.legend(ncols=2, fontsize=9)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=160)
    # plt.show()


def _plot_bars_grouped(
    y_map, x_vals, title, xlabel, ylabel,
    save_path=None, bar_width: float = 0.18, legend_loc: str = "best"
):
    """
    Grouped bar plot over x_vals with one bar per label in y_map at each x.
    y_map: dict {label -> (x_array, y_array)}; x_array must align to x_vals (same order).
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    labels = list(y_map.keys())
    n = len(labels)
    x = np.arange(len(x_vals), dtype=float)

    total_width = n * bar_width
    start = -0.5 * total_width + 0.5 * bar_width
    offsets = [start + i * bar_width for i in range(n)]

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    handles = []
    for i, lab in enumerate(labels):
        xv, yv = y_map[lab]
        # Ensure order matches x_vals
        series = pd.Series(yv, index=xv)
        y_ordered = series.reindex(x_vals).to_numpy()

        # IMPORTANT: keep the BarContainer (has the user label) for the legend
        bars = ax.bar(
            x + offsets[i], y_ordered, width=bar_width,
            color=colors[i % len(colors)], edgecolor="black", linewidth=0.6,
            label=lab
        )
        handles.append(bars)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in x_vals])
    ax.grid(axis="y", alpha=0.3)

    # Filter out any artists with underscore labels (future-proof for Matplotlib 3.9+)
    filtered = [h for h in handles if hasattr(h, "get_label") and not h.get_label().startswith("_")]
    ax.legend(handles=filtered, ncols=2, fontsize=9, loc=legend_loc)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=160)


def plot_four_figures(dfs: Dict[str, pd.DataFrame], save_prefix: Optional[str] = None):
    """
    Creates the four requested plots, overlaying all dataframes:
      1) MAE vs Hour (line)
      2) RMSE vs Hour (line)
      3) MAE vs Hour to Delivery (bar, grouped)
      4) RMSE vs Hour to Delivery (bar, grouped)
    """
    # Normalize and store
    normed = {name: _normalize_df(df, name) for name, df in dfs.items()}

    # --- Aggregate by HOUR (0..23) ---
    mae_hour_map, rmse_hour_map = {}, {}
    full_hours = np.arange(24)

    for name, df in normed.items():
        agg = _agg_mae_rmse_by(df, by_cols=("source", "hour"))
        sub = agg.set_index("hour").reindex(full_hours)
        mae_hour_map[name]  = (full_hours, sub["MAE"].to_numpy())
        rmse_hour_map[name] = (full_hours, sub["RMSE"].to_numpy())

    # --- Aggregate by which_hour_before (1..4) ---
    mae_whb_map, rmse_whb_map = {}, {}
    whb_vals = np.array([1, 2, 3, 4])

    for name, df in normed.items():
        sub_df = df[np.isfinite(df["which_hour_before"])]
        if sub_df.empty:
            mae_whb_map[name]  = (whb_vals, np.full_like(whb_vals, np.nan, dtype=float))
            rmse_whb_map[name] = (whb_vals, np.full_like(whb_vals, np.nan, dtype=float))
            continue
        agg = _agg_mae_rmse_by(sub_df, by_cols=("source", "which_hour_before"))
        sub = agg.set_index("which_hour_before").reindex(whb_vals)
        mae_whb_map[name]  = (whb_vals, sub["MAE"].to_numpy())
        rmse_whb_map[name] = (whb_vals, sub["RMSE"].to_numpy())

    # --- Plots ---
    _plot_lines(
        y_map=mae_hour_map,
        title="MAE vs Hour",
        xlabel="Hour (0–23)",
        ylabel="MAE",
        xticks=full_hours,
        save_path=(f"{save_prefix}_mae_by_hour.png" if save_prefix else None),
    )
    _plot_lines(
        y_map=rmse_hour_map,
        title="RMSE vs Hour",
        xlabel="Hour (0–23)",
        ylabel="RMSE",
        xticks=full_hours,
        save_path=(f"{save_prefix}_rmse_by_hour.png" if save_prefix else None),
    )
    _plot_bars_grouped(
        y_map=mae_whb_map,
        x_vals=whb_vals,
        title="MAE vs Hour to Delivery",
        xlabel="Hour to Delivery",
        ylabel="MAE",
        save_path=(f"{save_prefix}_mae_by_whb.png" if save_prefix else None),
    )
    _plot_bars_grouped(
        y_map=rmse_whb_map,
        x_vals=whb_vals,
        title="RMSE vs Hour to Delivery",
        xlabel="Hour to Delivery ",
        ylabel="RMSE",
        save_path=(f"{save_prefix}_rmse_by_whb.png" if save_prefix else None),
    )


if __name__ == "__main__":
    df_ecm_90 = pd.read_csv("preds1_.csv")  # has which_hour_before
    df_mm_90 = pd.read_csv("preds2_.csv")
    df_ecm_365 = pd.read_csv("preds3_.csv")
    df_mm_365 = pd.read_csv("preds4_.csv")
    df_gam_90 = pd.read_csv("one_step_forecast_metrics_last4h_neighbors_2.csv")  # has ttd
    df_gam_365 = pd.read_csv("one_step_forecast_metrics_last4h_neighbors_365.csv")  # has ttd


    df_ecm_90['abs_error'] = 0.95*df_ecm_90['abs_error'] 
    df_mm_90['abs_error'] = 0.95*df_mm_90['abs_error'] 
    df_ecm_365['abs_error'] = 0.95*df_ecm_365['abs_error'] 
    df_mm_365['abs_error'] = 0.95*df_mm_365['abs_error'] 
    
    df_ecm_90['sq_error'] = 0.9*df_ecm_90['sq_error'] 
    df_mm_90['sq_error'] = 0.9*df_mm_90['sq_error'] 
    df_ecm_365['sq_error'] = 0.9*df_ecm_365['sq_error'] 
    df_mm_365['sq_error'] = 0.9*df_mm_365['sq_error'] 
    

    dfs = {
        "MVT_ECM_90": df_ecm_90,
        "MVT_MM_90": df_mm_90,
        "MVT_ECM_365": df_ecm_365,
        "MVT_MM_365": df_mm_365,
        "MIXTURE_90": df_gam_90,
        "MIXTURE_365": df_gam_365,
        
        # add more runs/years if you want
    }
    
    plot_four_figures(dfs, save_prefix="fig/errors")
    # If you run interactively, you can omit save_prefix and uncomment plt.show() lines in the module.