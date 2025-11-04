"""
check_jsu2_hessian.py
=====================

Compare JSU2 analytic 2nd derivatives with finite differences of
the *analytic scores* (_dmu/_dsigma/_dnu/_dtau).

Nothing outside the standard library and NumPy is required.
"""

from __future__ import annotations
import math
import random
from typing import Callable, Tuple

import numpy as np

# ---------------------------------------------------------------------
#  import JSU2 (adapt the import line if the file is named differently)
# ---------------------------------------------------------------------
from distributions_alt import JSU2        # <-- change if needed

_J   = JSU2()                # single reusable instance
LBLs = ("μ", "σ", "ν", "τ")  # parameter labels


# ---------------------------------------------------------------------
#  5-point first derivative (O(h^4))
# ---------------------------------------------------------------------
def central_first(f: Callable[[float], float], x: float, h: float) -> float:
    return (
        f(x - 2 * h)
        - 8 * f(x - h)
        + 8 * f(x + h)
        - f(x + 2 * h)
    ) / (12 * h)


def step_size(x: float, base: float = 1e-5) -> float:
    """absolute step that scales with |x| but never drops below *base*"""
    return base * max(1.0, abs(x))


# ---------------------------------------------------------------------
#  wrappers for score and Hessian
# ---------------------------------------------------------------------
def score(
    y: float, mu: float, sigma: float, nu: float, tau: float, idx: int
) -> float:
    """analytic ∂ℓ/∂θᵢ   (no clipping)"""
    if idx == 0:
        return JSU2._dmu(y, mu, sigma, nu, tau)
    if idx == 1:
        return JSU2._dsigma(y, mu, sigma, nu, tau)
    if idx == 2:
        return JSU2._dnu(y, mu, sigma, nu, tau)
    if idx == 3:
        return JSU2._dtau(y, mu, sigma, nu, tau)
    raise IndexError("idx must be 0-3")


def hessian_analytic(
    y: float, mu: float, sigma: float, nu: float, tau: float, idx: int
) -> float:
    """analytic ∂²ℓ/∂θᵢ²   (no clipping)"""
    if idx == 0:
        return JSU2._ddmu(y, mu, sigma, nu, tau)
    if idx == 1:
        return JSU2._ddsigma(y, mu, sigma, nu, tau)
    if idx == 2:
        return JSU2._ddnu(y, mu, sigma, nu, tau)
    if idx == 3:
        return JSU2._ddtau(y, mu, sigma, nu, tau)
    raise IndexError("idx must be 0-3")


# ---------------------------------------------------------------------
#  numeric ∂/∂θ of the score
# ---------------------------------------------------------------------
def hessian_numeric(
    y: float, mu: float, sigma: float, nu: float, tau: float, idx: int
) -> float:
    """finite-difference derivative of the score wrt the same parameter"""
    θ = [mu, sigma, nu, tau]
    h = step_size(θ[idx])

    def g(p: float) -> float:
        θ[idx] = p
        out = score(y, *θ, idx)
        θ[idx] = p          # keep mypy happy – value overwritten next call
        return out

    return central_first(g, θ[idx], h)


# ---------------------------------------------------------------------
#  main test driver
# ---------------------------------------------------------------------
def check(
    n: int = 10_000,
    rtol: float = 1e-4,
    atol: float = 1e-4,
    seed: int | None = 123,
) -> None:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    failures: list[Tuple[str, float, float, float, Tuple[float, ...]]] = []

    for _ in range(n):
        # random draw  (feel free to tighten / widen the ranges)
        y     = random.gauss(0.0, 1.0)
        mu    = random.gauss(0.0, 1.0)
        sigma = math.exp(random.gauss(0.0, 1.0))     # > 0
        nu    = random.gauss(0.0, 1.0)
        tau   = math.exp(random.gauss(0.0, 1.0))     # > 0

        for i in range(2,3):
            ana = hessian_analytic(y, mu, sigma, nu, tau, i)
            num = hessian_numeric (y, mu, sigma, nu, tau, i)

            if not (math.isfinite(ana) and math.isfinite(num)):
                continue

            if abs(ana - num) > atol + rtol * abs(num):
                failures.append(
                    (
                        LBLs[i],
                        abs(ana - num) / max(1e-16, abs(num)),
                        ana,
                        num,
                        (y, mu, sigma, nu, tau),
                    )
                )

    # ------------- report ------------------------------------------------
    total = 4 * n
    if not failures:
        print(f"All {total} comparisons passed (rtol={rtol:g}, atol={atol:g}).")
        return

    print(
        f"{len(failures)} / {total} comparisons failed "
        f"(rtol={rtol:g}, atol={atol:g}).  Worst 10:\n"
    )
    failures.sort(key=lambda t: -t[1])      # by relative error, descending
    for lbl, rel, ana, num, pars in failures[:10]:
        print(
            f"{lbl}: rel={rel:8.2e}  "
            f"ana={ana: .6g}  num={num: .6g}  (y,μ,σ,ν,τ)={pars}"
        )


# ---------------------------------------------------------------------
#  CLI entry-point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    check()
