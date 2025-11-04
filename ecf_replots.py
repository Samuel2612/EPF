#!/usr/bin/env python3
"""
Replot COS+TT sweep results from CSV (no recomputation).

Style:
- SAME colormap for both methods.
- SAME color for the same dimension d.
- Method difference ONLY by linestyle:
    * ECF (no filter): solid
    * ECF+filter:      dashed
- Dot markers, slightly larger by default.

Usage:
  python replot_cos_tt.py --results results/cos_tt_sweep_results.csv --outdir plots_v3
  # Optional tweaks:
  #   --cmap viridis --markersize 5 --dpi 220 --moments 1 2 3 4 --minN 50 --maxN 200000 --dims 2 3
"""
import argparse, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def parse_args():
    pa = argparse.ArgumentParser()
    pa.add_argument("--results", default="results/cos_tt_sweep_results.csv")
    pa.add_argument("--outdir", default="plots_v3")
    pa.add_argument("--moments", nargs="+", type=int, default=[1,2,3,4])
    pa.add_argument("--minN", type=int, default=None)
    pa.add_argument("--maxN", type=int, default=None)
    pa.add_argument("--dims", nargs="+", type=int, default=None)
    pa.add_argument("--cmap", default="viridis", help="Colormap used for ALL series")
    pa.add_argument("--markersize", type=float, default=5.0, help="Dot marker size")
    pa.add_argument("--dpi", type=int, default=200)
    return pa.parse_args()

def subset_df(df, minN, maxN, dims):
    df = df[df["method"].isin(["ecf", "filtered"])].copy()
    if minN is not None: df = df[df["N"] >= minN]
    if maxN is not None: df = df[df["N"] <= maxN]
    if dims is not None: df = df[df["d"].isin(dims)]
    return df

def colors_by_d(d_list, cmap_name):
    vals = np.linspace(0.25, 0.85, len(d_list))
    cmap = plt.get_cmap(cmap_name)
    return {d: cmap(v) for d, v in zip(d_list, vals)}

def _add_ref_line(ax, df_p, moment_col):
    """Black dashed ~ N^{-1/2} reference, scaled near min N."""
    Ns = np.sort(df_p["N"].unique())
    if len(Ns) < 2:
        return
    Nmin, Nmax = Ns[0], Ns[-1]
    at_min = df_p[df_p["N"] == Nmin][moment_col].dropna().values
    y_median = np.median(at_min) if at_min.size else np.median(df_p[moment_col].dropna().values)
    c = y_median * np.sqrt(Nmin)
    N_line = np.array([Nmin, Nmax], dtype=float)
    y_line = c / np.sqrt(N_line)
    ax.loglog(N_line, y_line, linestyle="--", color="black", linewidth=1.2, label=r"$N^{-1/2}$")

def make_error_plots(df, outdir, moments, cmap_name, markersize, dpi):
    os.makedirs(outdir, exist_ok=True)
    d_list = sorted(df["d"].unique())
    color_map = colors_by_d(d_list, cmap_name)

    for p in moments:
        fig, ax = plt.subplots(figsize=(8,6))
        col = f"mae_p{p}"

        # Plot ECF (solid) and filtered (dashed), same color per d
        for d in d_list:
            for method in ["ecf", "filtered"]:
                sub = df[(df["method"]==method) & (df["d"]==d)].sort_values("N")
                if sub.empty: 
                    continue
                ls = "solid" if method == "ecf" else "dashed"
                label = f"{'ECF' if method=='ecf' else 'ECF+filter'}, d={d}"
                ax.loglog(sub["N"].values, sub[col].values,
                          marker=".", markersize=markersize,
                          linestyle=ls, linewidth=1.8,
                          color=color_map[d], label=label)

        # Add black dashed N^{-1/2} guide
        _add_ref_line(ax, df, col)

        ax.set_xlabel("Number of Samples (N)")
        ax.set_ylabel(f"Mean Absolute Error (moment p={p})")
        ax.set_title(f"MAE vs N (moment p={p})")
        ax.grid(True, which="both", linestyle="--", alpha=0.35)
        ax.legend(ncol=2)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"mae_moment_p{p}.png"), dpi=dpi)
        plt.close(fig)

def make_cpu_plot(df, outdir, cmap_name, markersize, dpi):
    os.makedirs(outdir, exist_ok=True)
    d_list = sorted(df["d"].unique())
    color_map = colors_by_d(d_list, cmap_name)

    fig, ax = plt.subplots(figsize=(8,6))
    for d in d_list:
        for method in ["ecf", "filtered"]:
            sub = df[(df["method"]==method) & (df["d"]==d)].sort_values("N")
            if sub.empty:
                continue
            ls = "solid" if method == "ecf" else "dashed"
            label = f"{'ECF' if method=='ecf' else 'ECF+filter'}, d={d}"
            ax.loglog(sub["N"].values, sub["cpu_time_s"].values,
                      marker=".", markersize=markersize,
                      linestyle=ls, linewidth=1.8,
                      color=color_map[d], label=label)

    ax.set_xlabel("Number of Samples (N)")
    ax.set_ylabel("CPU time [s]")
    ax.set_title("Build→TT→RowCore CPU time vs N")
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "cpu_time_vs_N.png"), dpi=dpi)
    plt.close(fig)

def main():
    args = parse_args()
    df = pd.read_csv(args.results)
    df = subset_df(df, args.minN, args.maxN, args.dims)
    make_error_plots(df, args.outdir, args.moments, args.cmap, args.markersize, args.dpi)
    make_cpu_plot(df, args.outdir, args.cmap, args.markersize, args.dpi)
    print(f"Replots written to: {args.outdir}")

if __name__ == "__main__":
    main()
