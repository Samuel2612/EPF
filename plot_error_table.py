# plot_errors_intraday_bars.py
from typing import Dict, Iterable, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap


# ---------------------------
# Utilities
# ---------------------------

def _ensure_int(series: pd.Series) -> pd.Series:
    if series is None:
        return pd.Series(dtype="Int64")
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
      - coerce log_score and crps if present
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

    # uniform internals
    df["__abs_err__"] = df[abs_col]
    df["__sq_err__"]  = df[sq_col]

    # optional: log_score / crps (user said present & same across all)
    if "log_score" in df.columns:
        df["__log_score__"] = pd.to_numeric(df["log_score"], errors="coerce")
    if "crps" in df.columns:
        df["__crps__"] = pd.to_numeric(df["crps"], errors="coerce")

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


# ---------------------------
# Color logic (declutter + consistency)
# ---------------------------

def _is_mvt_label(label: str) -> bool:
    return str(label).upper().startswith("MVT")

def _color_for_label(label: str, idx_mvt: int, n_mvt: int, idx_other: int, n_other: int):
    """
    Pick a color from 'winter' (for MVT_*) or 'autumn' (others), spaced across the colormap.
    Avoid the very ends of the cmap for readability.
    """
    if _is_mvt_label(label):
        t = 0.15 + 0.7 * (idx_mvt + 1) / max(n_mvt + 1, 2)
        return get_cmap("winter")(t)
    else:
        t = 0.15 + 0.7 * (idx_other + 1) / max(n_other + 1, 2)
        return get_cmap("autumn")(t)


# ---------------------------
# Plotters
# ---------------------------

def _plot_lines(y_map, title, xlabel, ylabel, xticks=None, save_path=None):
    """
    y_map: dict {label -> (x_array, y_array)}; draws one line per label.
    Uses 'winter' for MVT_* labels and 'autumn' for the rest.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    labels = list(y_map.keys())
    mvt_labels   = [lab for lab in labels if _is_mvt_label(lab)]
    other_labels = [lab for lab in labels if not _is_mvt_label(lab)]
    ordered = mvt_labels + other_labels  # put MVT first for legend grouping

    for lab in ordered:
        x, y = y_map[lab]
        if _is_mvt_label(lab):
            color = _color_for_label(lab, idx_mvt=mvt_labels.index(lab), n_mvt=len(mvt_labels),
                                     idx_other=0, n_other=len(other_labels))
            ls = "-"
        else:
            color = _color_for_label(lab, idx_mvt=0, n_mvt=len(mvt_labels),
                                     idx_other=other_labels.index(lab), n_other=len(other_labels))
            ls = "--"

        ax.plot(x, y, label=lab, color=color, linestyle=ls, linewidth=2.0, marker=None)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xticks is not None:
        ax.set_xticks(xticks)
    ax.grid(True, alpha=0.25)

    # Legend outside to reduce clutter
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
              borderaxespad=0.0, fontsize=9, title="Series")
    fig.tight_layout(rect=(0, 0, 0.82, 1))
    if save_path:
        fig.savefig(save_path, dpi=160, bbox_inches="tight")


def _plot_bars_grouped(
    y_map, x_vals, title, xlabel, ylabel,
    save_path=None, bar_width: float = None, legend_loc: str = "outside"
):
    """
    Grouped bar plot over x_vals with one bar per label in y_map at each x.
    Colors: 'winter' for MVT_* labels, 'autumn' for others.
    Legend is placed outside to avoid clutter. Bar width auto-scales with #series.
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11.5, 5.5))

    labels = list(y_map.keys())
    mvt_labels   = [lab for lab in labels if _is_mvt_label(lab)]
    other_labels = [lab for lab in labels if not _is_mvt_label(lab)]
    ordered = mvt_labels + other_labels
    n = len(ordered)

    # Auto bar width to keep total group width <= 0.85
    if bar_width is None:
        bar_width = min(0.85 / max(n, 1), 0.16)

    x = np.arange(len(x_vals), dtype=float)
    total_width = n * bar_width
    start = -0.5 * total_width + 0.5 * bar_width
    offsets = [start + i * bar_width for i in range(n)]

    handles = []
    for i, lab in enumerate(ordered):
        xv, yv = y_map[lab]
        series = pd.Series(yv, index=xv)
        y_ordered = series.reindex(x_vals).to_numpy()

        if _is_mvt_label(lab):
            color = _color_for_label(lab, idx_mvt=mvt_labels.index(lab), n_mvt=len(mvt_labels),
                                     idx_other=0, n_other=len(other_labels))
        else:
            color = _color_for_label(lab, idx_mvt=0, n_mvt=len(mvt_labels),
                                     idx_other=other_labels.index(lab), n_other=len(other_labels))

        # Softer styling: no heavy edges, slight alpha
        bars = ax.bar(
            x + offsets[i], y_ordered, width=bar_width,
            color=color, edgecolor="none", alpha=0.95, label=lab
        )
        handles.append(bars)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in x_vals])
    ax.grid(axis="y", alpha=0.25)

    if legend_loc == "outside":
        ax.legend(handles=[h for h in handles], loc="center left", bbox_to_anchor=(1.02, 0.5),
                  borderaxespad=0.0, fontsize=9, ncols=1, title="Series")
        fig.tight_layout(rect=(0, 0, 0.82, 1))
    else:
        ax.legend(handles=[h for h in handles], ncols=2, fontsize=9, loc=legend_loc)
        fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=160, bbox_inches="tight")


# ---------------------------
# Summary Table
# ---------------------------

def build_summary_table(dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Returns a DataFrame with one row per model:
      Model | MAE | RMSE | LogScore | CRPS
    Means are taken over all rows of each model dataframe.
    """
    rows = []
    for name, df in dfs.items():
        nd = _normalize_df(df, name)

        mae  = nd["__abs_err__"].mean()
        rmse = np.sqrt(nd["__sq_err__"].mean())

        log_score = nd["__log_score__"].mean() if "__log_score__" in nd.columns else np.nan
        crps      = nd["__crps__"].mean()      if "__crps__" in nd.columns      else np.nan

        rows.append({
            "Model": name,
            "MAE": mae,
            "RMSE": rmse,
            "LogScore": log_score,
            "CRPS": crps,
        })

    tbl = pd.DataFrame(rows)
    # Nice ordering: MVT models first, then others
    tbl["_order"] = tbl["Model"].map(lambda s: (0, s) if _is_mvt_label(s) else (1, s))
    tbl = tbl.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)
    return tbl


def print_summary_table(tbl: pd.DataFrame, round_digits: int = 4):
    disp = tbl.copy()
    for col in ["MAE", "RMSE", "LogScore", "CRPS"]:
        if col in disp.columns:
            disp[col] = disp[col].astype(float).round(round_digits)
    print("\n=== Model Summary (MAE / RMSE / LogScore / CRPS) ===")
    print(disp.to_string(index=False))


# ---------------------------
# Public API (plots)
# ---------------------------

def plot_four_figures(dfs: Dict[str, pd.DataFrame], save_prefix: Optional[str] = None):
    """
    Creates the four requested plots, overlaying all dataframes:
      1) MAE vs Hour (line)
      2) RMSE vs Hour (line)
      3) MAE vs Hour to Delivery (bar, grouped)
      4) RMSE vs Hour to Delivery (bar, grouped)
    Also prints a model summary table; if save_prefix provided, saves it as CSV.
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
        xlabel="Hour to Delivery",
        ylabel="RMSE",
        save_path=(f"{save_prefix}_rmse_by_whb.png" if save_prefix else None),
    )

    # --- Summary Table (print + optional save) ---
    tbl = build_summary_table(dfs)
    print_summary_table(tbl)
    if save_prefix:
        tbl.to_csv(f"{save_prefix}_summary.csv", index=False)


# ---------------------------
# CLI entry
# ---------------------------

if __name__ == "__main__":
    # Example files (replace with your own)
    df_ecm_90   = pd.read_csv("preds1_.csv")  # has which_hour_before
    df_mm_90    = pd.read_csv("preds2_.csv")
    df_ecm_365  = pd.read_csv("preds3_.csv")
    df_mm_365   = pd.read_csv("preds4_.csv")
    df_gam_90   = pd.read_csv("one_step_forecast_metrics_last4h_neighbors_2.csv")   # has ttd
    df_gam_365  = pd.read_csv("one_step_forecast_metrics_last4h_neighbors_365.csv") # has ttd
    
    
 
    dfs = {
        "MVT_ECM_90": df_ecm_90,
        "MVT_MM_90": df_mm_90,
        "MVT_ECM_365": df_ecm_365,
        "MVT_MM_365": df_mm_365,
        "MIXTURE_90": df_gam_90,
        "MIXTURE_365": df_gam_365,
    }

    plot_four_figures(dfs, save_prefix="fig/errors")
