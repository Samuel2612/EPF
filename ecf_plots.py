#!/usr/bin/env python3
from __future__ import annotations
import math, time, os
from typing import List, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ecf_tensor_einsum import A_tensor_no_sign_blowup
from ecf_tensor_einsum_filter4 import build_cos_coefficients

# -------- utilities (same as before) --------
def cos_basis(x, a, b, N):
    k = np.arange(N)
    vec = np.cos(k * math.pi * (x - a) / (b - a))
    vec[0] *= 0.5
    return vec

def closed_form_moment_power(a, b, N, p):
    L = b - a
    J = np.zeros(N, dtype=np.float64)
    J[0] = 0.5 * (b**(p + 1) - a**(p + 1)) / (p + 1)
    for k in range(1, N):
        ilambda = 1j * k * math.pi
        pow_series = [(ilambda)**m / math.factorial(m) for m in range(p + 1)]
        acc = 0.0
        for j in range(p + 1):
            S_j = sum(pow_series[:j + 1])
            I_j = math.factorial(j) / ilambda**(j + 1) * (1 - np.exp(ilambda) * S_j)
            acc += math.comb(p, j) * a**(p - j) * L**(j + 1) * I_j.real
        J[k] = acc
    return J

def tt_svd_(tensor: np.ndarray, eps: float):
    T = tensor.copy()
    dims = T.shape
    d = len(dims)
    if d == 0:
        raise ValueError("Input must be at least 1-D.")
    norm2 = np.linalg.norm(T) ** 2
    thr = (eps / math.sqrt(max(d - 1, 1))) ** 2 * norm2
    cores: List[np.ndarray] = []
    ranks: List[int] = [1]
    unfold = T
    for k in range(d - 1):
        m = ranks[-1] * dims[k]
        unfold = unfold.reshape(m, -1)
        U, S, Vh = np.linalg.svd(unfold, full_matrices=False)
        tail = np.cumsum(S[::-1] ** 2)[::-1]
        r = len(S)
        mask = tail <= thr
        if mask.any():
            r = int(np.argmax(mask)) + 1
        r = max(1, r)
        cores.append(U[:, :r].reshape(ranks[-1], dims[k], r))
        ranks.append(r)
        unfold = (S[:r, None] * Vh[:r])
    cores.append(unfold.reshape(ranks[-1], dims[-1], 1))
    ranks.append(1)
    return cores, ranks

def build_row_core(G0_mat: np.ndarray, weights: np.ndarray):
    return (weights[:, None] * G0_mat).sum(0, keepdims=True)

def feature_contraction(x_feat: np.ndarray, cores: List[np.ndarray], a: np.ndarray, b: np.ndarray):
    vec = None
    for i, G in enumerate(cores[1:], start=1):
        N = G.shape[1]
        basis = cos_basis(x_feat[i-1], a[i], b[i], N)
        M = np.tensordot(G, basis, axes=([1], [0]))
        vec = M if vec is None else vec @ M
    return vec

def gaussian_conditional_moments(x_feat, mu, Sigma):
    Sigma_yx     = Sigma[0,1:]
    Sigma_xx_inv = np.linalg.inv(Sigma[1:,1:])
    mu_y, mu_x   = mu[0], mu[1:]
    sigma_cond   = float(Sigma[0,0] - Sigma_yx @ Sigma_xx_inv @ Sigma_yx)
    mu_c = mu_y + Sigma_yx @ Sigma_xx_inv @ (x_feat - mu_x)
    m1 = mu_c
    m2 = sigma_cond + mu_c**2
    m3 = mu_c**3 + 3*mu_c*sigma_cond
    m4 = mu_c**4 + 6*mu_c**2*sigma_cond + 3*sigma_cond**2
    return np.array([m1, m2, m3, m4])

# -------- config --------
NS = [10, 20, 50, 100, 200, 500, 1_000, 2_000, 5_000,
      10_000, 20_000, 50_000, 100_000, 200_000, 500_000, 1_000_000]
DIMENSIONS = [2, 3, 4]
EVAL_POINTS = 10000
EPS_TT = 1e-12
KS_PER_DIM = 64
RNG_SEED = 1234

MU_4 = np.array([0.0, -0.6, -0.1, 0.3])
SIGMA_4 = np.array([[1.0, 0.4, 0.3, 0.1],
                    [0.4, 1.0, 0.6, 0.5],
                    [0.3, 0.6, 1.0, 0.5],
                    [0.1, 0.5, 0.5, 1.0]])

def is_memory_error(exc: Exception) -> bool:
    if isinstance(exc, MemoryError):
        return True
    msg = str(exc).lower()
    # common numpy allocation messages
    return ("unable to allocate" in msg or
            "cannot allocate" in msg or
            "could not allocate" in msg or
            "out of memory" in msg or
            "oom" in msg)

def run_experiment():
    rng = np.random.default_rng(RNG_SEED)
    records: List[Dict] = []
    eval_timing: List[Dict] = []

    methods = [
        ("ecf", A_tensor_no_sign_blowup),     # summer
        ("filtered", build_cos_coefficients)  # winter
    ]

    for d in DIMENSIONS:
        mu = MU_4[:d]
        Sigma = SIGMA_4[:d, :d]
        std = np.sqrt(np.diag(Sigma))
        a = mu - 7 * std
        b = mu + 7 * std
        s = np.ones_like(mu)
        Ks = [KS_PER_DIM] * d

        # If we hit OOM at some N for this d, skip remaining Ns for this d
        oom_hit_for_d = False

        for N in NS:
            if oom_hit_for_d:
                break

            # samples
            X = rng.multivariate_normal(mean=mu, cov=Sigma, size=N)
            xs = rng.multivariate_normal(mean=mu[1:], cov=Sigma[1:,1:], size=EVAL_POINTS)

            try:
                for method_name, builder in methods:
                    t0 = time.perf_counter()
                    if method_name == "ecf":
                        A = builder(X, a, b, s, Ks)   # may OOM
                    else:
                        A = builder(X, a, b, Ks)      # may OOM

                    cores, ranks = tt_svd_(A, eps=EPS_TT)

                    G0_mat = cores[0].reshape(Ks[0], -1)
                    row_cores = {}
                    for p in range(5):
                        w = closed_form_moment_power(a[0], b[0], Ks[0], p)
                        row_cores[p] = build_row_core(G0_mat, w)
                    tilde_G0 = row_cores[0]
                    t1 = time.perf_counter()
                    cpu_time = t1 - t0

                    # errors + per-sample eval time
                    err_acc = {1: [], 2: [], 3: [], 4: []}
                    total_eval_time = 0.0

                    for x in xs:
                        t_e0 = time.perf_counter()
                        P = feature_contraction(x, cores, a, b)
                        norm = float((tilde_G0 @ P).squeeze())
                        m_tt = {}
                        for p in (1,2,3,4):
                            num_p = float((row_cores[p] @ P).squeeze())
                            m_tt[p] = num_p / norm
                        t_e1 = time.perf_counter()
                        total_eval_time += (t_e1 - t_e0)

                        m_ex = gaussian_conditional_moments(x, mu, Sigma)
                        for idx, p in enumerate((1,2,3,4)):
                            err_acc[p].append(abs(m_tt[p] - m_ex[idx]))

                    mae = {p: float(np.mean(err_acc[p])) for p in (1,2,3,4)}
                    avg_eval_time = total_eval_time / len(xs)

                    records.append({
                        "N": N, "d": d, "method": method_name,
                        "cpu_time_s": cpu_time,
                        "mae_p1": mae[1], "mae_p2": mae[2], "mae_p3": mae[3], "mae_p4": mae[4],
                        "avg_eval_time_s": avg_eval_time
                    })
                    eval_timing.append({
                        "N": N, "d": d, "method": method_name,
                        "avg_eval_time_s": avg_eval_time
                    })

                    print(f"d={d} N={N:>7} {method_name:9s} | cpu={cpu_time:8.3f}s | "
                          f"MAE p1..p4 = {mae[1]:.3e}, {mae[2]:.3e}, {mae[3]:.3e}, {mae[4]:.3e} | "
                          f"eval_avg={avg_eval_time:.3e}s")

            except Exception as e:
                if is_memory_error(e):
                    print(f"[OOM] d={d} N={N} — {e}. Skipping remaining N for d={d} and continuing with next dimension.")
                    oom_hit_for_d = True
                    # (Optional) mark a sentinel row so you can see where it stopped:
                    records.append({
                        "N": N, "d": d, "method": "oom",
                        "cpu_time_s": np.nan,
                        "mae_p1": np.nan, "mae_p2": np.nan, "mae_p3": np.nan, "mae_p4": np.nan,
                        "avg_eval_time_s": np.nan
                    })
                else:
                    # If it’s another exception, re-raise so you notice real bugs.
                    raise

    df = pd.DataFrame.from_records(records)
    df_eval = pd.DataFrame.from_records(eval_timing)
    return df, df_eval

def make_plots(df: pd.DataFrame, outdir: str = "plots"):
    os.makedirs(outdir, exist_ok=True)
    # drop OOM sentinel rows for plotting
    dfp = df[df["method"].isin(["ecf", "filtered"])].copy()

    cmap_ecf = plt.get_cmap("autumn")
    cmap_flt = plt.get_cmap("winter")

    d_list = sorted(dfp["d"].unique())
    vals = np.linspace(0.25, 0.85, len(d_list))
    colors_ecf = {d: cmap_ecf(v) for d, v in zip(d_list, vals)}
    colors_flt = {d: cmap_flt(v) for d, v in zip(d_list, vals)}

    # 1) Error plots per moment
    for p in (1, 2, 3, 4):
        plt.figure(figsize=(8, 6))
        for method in ["ecf", "filtered"]:
            for d in d_list:
                sub = dfp[(dfp["method"] == method) & (dfp["d"] == d)].sort_values("N")
                if sub.empty: 
                    continue
                color = colors_ecf[d] if method == "ecf" else colors_flt[d]
                label = f"{'ECF' if method=='ecf' else 'ECF+filter'}, d={d}"
                plt.loglog(sub["N"].values, sub[f"mae_p{p}"].values, marker="o", linewidth=1.5,
                           label=label, color=color)
        plt.xlabel("Number of Samples (N)")
        plt.ylabel(f"Mean Absolute Error (moment p={p})")
        plt.title(f"MAE vs N (moment p={p})")
        plt.grid(True, which="both", linestyle="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"mae_moment_p{p}.png"), dpi=160)
        plt.close()

    # 2) CPU time plot
    plt.figure(figsize=(8, 6))
    for method in ["ecf", "filtered"]:
        for d in d_list:
            sub = dfp[(dfp["method"] == method) & (dfp["d"] == d)].sort_values("N")
            if sub.empty:
                continue
            color = colors_ecf[d] if method == "ecf" else colors_flt[d]
            label = f"{'ECF' if method=='ecf' else 'ECF+filter'}, d={d}"
            plt.loglog(sub["N"].values, sub["cpu_time_s"].values, marker="o", linewidth=1.5,
                       label=label, color=color)
    plt.xlabel("Number of Samples (N)")
    plt.ylabel("CPU time [s]")
    plt.title("Build→TT→RowCore CPU time vs N")
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "cpu_time_vs_N.png"), dpi=160)
    plt.close()

def main():
    df, df_eval = run_experiment()
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/cos_tt_sweep_results.csv", index=False)
    df_eval.to_csv("results/cos_tt_eval_times.csv", index=False)
    make_plots(df, outdir="plots")

    print("\nAverage per-sample evaluation time (requested block):")
    grouped = df_eval.groupby(["N", "d", "method"])["avg_eval_time_s"].mean().reset_index()
    print(grouped.to_string(index=False))
    global_avg = df_eval["avg_eval_time_s"].mean()
    print(f"\nGlobal average per-sample evaluation time across all runs: {global_avg:.6e} s")

if __name__ == "__main__":
    main()
