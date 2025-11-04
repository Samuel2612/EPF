"""
Check that the re-parameterised JSU (μ,σ,ν,τ)
and the original JSUo (μ₁,σ₁,ν₁,τ₁) give identical pdfs
after converting parameters with the helper functions.

Run with:  python check_jsu_equivalence.py
"""

import numpy as np
import scipy.stats as st

# ------------------------------------------------------------------
# 1.  Import your classes / helpers
# ------------------------------------------------------------------
# If everything lives in one file, comment these and rely on the
# definitions already present in the namespace.
from distributions import (
    JSU,
    JSUo,
    jsu_reparam_to_original,
)

# ------------------------------------------------------------------
# 2.  Instantiate the distribution wrappers
# ------------------------------------------------------------------
jsu  = JSU()   # re-parameterised
jsuo = JSUo()  # original


# ------------------------------------------------------------------
# 3.  Monte-Carlo test
# ------------------------------------------------------------------
rng = np.random.default_rng(2025)
n_param_sets = 20        # how many random (μ,σ,ν,τ) to try
n_y          = 0       # how many y-points per set

for k in range(n_param_sets):
    # --- draw a random but sensible parameter quadruple -------------
    mu    = rng.normal()
    sigma = np.exp(rng.uniform(-1, 1))      # > 0
    nu    = rng.normal()
    tau   = rng.uniform(0.25, 3.0)          # > 0

    # --- convert to the "original" parametrisation ------------------
    mu1, sigma1, nu1, tau1 = jsu_reparam_to_original(mu, sigma, nu, tau)

    # --- pick y-values around the centre + in the tails ------------
    y = rng.normal(mu, sigma, size=n_y)
    y = np.concatenate([y, mu + np.r_[-4, -2, 0, 2, 4]*sigma])  # a few fixed points

    # --- build θ-matrices expected by the pdf methods --------------
    theta_jsu  = np.column_stack([np.full_like(y, mu),
                                  np.full_like(y, sigma),
                                  np.full_like(y, nu),
                                  np.full_like(y, tau)])

    theta_jsuo = np.column_stack([np.full_like(y, mu1),
                                  np.full_like(y, sigma1),
                                  np.full_like(y, nu1),
                                  np.full_like(y, tau1)])

    # --- evaluate the two densities --------------------------------
    pdf_jsu  = jsu.pdf(y, theta_jsu)
    pdf_jsuo = jsuo.pdf(y, theta_jsuo)

    # --- check & report --------------------------------------------
    if not np.allclose(pdf_jsu, pdf_jsuo, rtol=1e-10, atol=1e-10):
        # Print a concise diagnostic then exit
        max_err = np.max(np.abs(pdf_jsu - pdf_jsuo))
        raise AssertionError(
            f"Mismatch detected for param-set {k+1}: max |Δ| = {max_err:.3e}"
        )

print(f"✅  All {n_param_sets} random parameter sets match to 1e-10 precision.")
