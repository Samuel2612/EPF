#!/usr/bin/env python3
"""
Memory‑friendly Monte‑Carlo test of cond_moments_tt
with **sequential** empirical characteristic functions (ECFs).

* No ECF is held longer than one run.
* The ECF of each run is built in CHUNK_SIZE slices to cap RAM.
"""
import itertools, math, gc, time
import numpy as np
from cond_moments_tt import (
    tt_svd, index_grid, closed_form_moment_power,
    build_row_core, feature_contraction
)


Ks            = [8]*4      # COS modes per dimension
N_runs        = 1                   # how many independent ECFs
N_samples_ecf = 500              # Monte‑Carlo samples for each ECF
chunk_size    = 1000                   # t‑vectors processed per chunk
N_points      = 100                   # X‑locations per run
rng_seed      = 20250716


rng = np.random.default_rng(rng_seed)

mu    = np.array([1.0, 0.5, -1.1, 1.3])
Sigma = np.array([[1.0, 0.4, 0.3, 0.1],
                  [0.4, 1.0, 0.6, 0.5],
                  [0.3, 0.6, 1.0, 0.5],
                  [0.1, 0.5, 0.5, 1.0]])
std   = np.sqrt(np.diag(Sigma))
a     = mu - 7 * std
b     = mu + 7 * std
L     = b - a

# ----------------- analytic conditional moments --------------------------
def analytic_conditional_moments(x, mu, Sigma):
    Σ_yx     = Sigma[0, 1:]
    Σ_xx_inv = np.linalg.inv(Sigma[1:, 1:])
    μ_y, μ_x = mu[0], mu[1:]
    σ2_c     = float(Sigma[0, 0] - Σ_yx @ Σ_xx_inv @ Σ_yx)
    μ_c      = μ_y + Σ_yx @ Σ_xx_inv @ (x - μ_x)
    m1  = μ_c
    m2  = σ2_c + μ_c**2
    m3  = μ_c**3 + 3 * μ_c * σ2_c
    m4  = μ_c**4 + 6 * μ_c**2 * σ2_c + 3 * σ2_c**2
    return np.array([m1, m2, m3, m4])


def chf_empirical_chunked(t, samples, chunk):
    """Return φ_N(t) for an (*any‑shape*,4) array t, using chunked dot‑products."""
    shape = t.shape[:-1]
    tf    = t.reshape(-1, 4)          # (M,4)
    M     = tf.shape[0]
    out   = np.empty(M, dtype=np.complex128)

    for start in range(0, M, chunk):
        end   = min(start + chunk, M)
        batch = tf[start:end]                  # (chunk,4)
        # ‑‑ heavy part in manageable blocks (chunk × N_samples)
        expo  = np.exp(1j * batch @ samples.T) # (chunk,N_samples)
        out[start:end] = expo.mean(axis=1)

    return out.reshape(shape)

def build_cos_tensor_ecf(Ks, a, b, samples):
    n      = len(Ks)
    factor = 2.0 / np.prod(b - a)
    grid   = index_grid(Ks)                     # (*Ks,4)

    rest   = np.array(list(itertools.product([1, -1], repeat=n-1)), float)
    Svec   = np.c_[np.ones(rest.shape[0]), rest]  # (2^{d-1}, d)

    A = np.zeros(Ks, dtype=np.float64)
    for s in Svec:
        print("doing some")
        t     = math.pi * grid * s / (b - a)
        phi   = chf_empirical_chunked(t, samples, chunk_size)
        phase = np.exp(-1j * (t * a).sum(-1))
        A    += np.real(phase * phi)
    return factor * A


J_pow = {p: closed_form_moment_power(a[0], b[0], Ks[0], p) for p in range(5)}


abs_err = {p: [] for p in (1, 2, 3, 4)}
tic0    = time.time()

for run in range(1, N_runs + 1):

    samples = rng.multivariate_normal(mu, Sigma, size=N_samples_ecf)


    print("building tensor")
    A = build_cos_tensor_ecf(Ks, a, b, samples)
    print("done buikding tnesor")
    cores, _ = tt_svd(A, eps=1e-10)


    G0_mat   = cores[0].reshape(Ks[0], -1)
    row0     = build_row_core(G0_mat, J_pow[0])
    row      = {p: build_row_core(G0_mat, J_pow[p]) for p in (1, 2, 3, 4)}

    # 5) evaluate conditional moments at N_points fresh X’s
    xs = rng.normal(mu[1:], std[1:], size=(N_points, 3))
    for x in xs:
        P    = feature_contraction(x, cores, a, b)
        norm = float((row0 @ P).squeeze())
        for p in (1, 2, 3, 4):
            num   = float((row[p] @ P).squeeze())
            m_est = num / norm
            m_ref = analytic_conditional_moments(x, mu, Sigma)[p - 1]
            abs_err[p].append(abs(m_est - m_ref))


    del samples, A, cores, row, row0, xs, P
    gc.collect()

    if run % 10 == 0 or run == N_runs:
        print(f"[{run:3d}/{N_runs}] done – "
              f"elapsed {time.time() - tic0:7.1f} s")


print("\n=== Absolute‑error statistics over "
      f"{N_runs * N_points:,d} conditional moments ===")
for p in (1, 2, 3, 4):
    e = np.array(abs_err[p])
    print(f"  p={p}:  max {e.max():.3e}   mean {e.mean():.3e}   "
          f"median {np.median(e):.3e}")
