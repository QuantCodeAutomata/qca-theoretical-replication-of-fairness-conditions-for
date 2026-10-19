"""
Simulate mean-reverting basis dynamics using an OU / AR(1) process.

Model:
    b_t = phi * b_{t-1} + epsilon_t,    epsilon_t ~ N(0, sigma_b^2)

where the AR(1) coefficient phi = exp(-lambda * dt) corresponds to an
Ornstein-Uhlenbeck mean-reversion with speed lambda.

Target parameters (hourly steps):
    phi  ≈ 0.72  =>  lambda ≈ ln(2) / 2.1 ≈ 0.33 / hour
    half-life T_{1/2} = ln(2) / lambda ≈ 2.1 hours.

The spot S_t follows geometric Brownian motion with correlated shocks.
The basis level e_t = b_t * S_t is returned as a separate column.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from typing import Dict, Tuple


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------

def simulate_basis_ou(
    n_steps: int = 72,
    phi_ar1: float = 0.72,           # hourly AR(1) coefficient
    sigma_b: float = 0.0006,         # 6 bp / hr basis volatility
    sigma_S: float = 0.20 / (252 * 24) ** 0.5,  # hourly spot log-vol
    rho: float = 0.0,                # W^S – W^b correlation
    b0: float = 0.0,                 # initial basis fraction
    S0: float = 100.0,               # initial spot price
    r: float = 0.05 / 252 / 24,     # hourly risk-free rate
    q: float = 0.02 / 252 / 24,     # hourly continuous yield
    seed: int = 42,
) -> pd.DataFrame:
    """
    Simulate an OU basis process correlated with a GBM spot price.

    The inferred OU speed is lambda = -ln(phi_ar1) per step.
    Correlated shocks are generated via Cholesky decomposition.

    Parameters
    ----------
    n_steps  : number of time steps.
    phi_ar1  : AR(1) persistence coefficient (0 < phi_ar1 < 1).
    sigma_b  : basis innovation std-dev per step.
    sigma_S  : spot log-vol per step.
    rho      : correlation between basis and spot innovations.
    b0       : initial basis fraction b_0 = e_0 / S_0.
    S0       : initial spot price.
    r, q     : risk-free rate and continuous dividend yield per step.
    seed     : RNG seed.

    Returns
    -------
    pd.DataFrame with columns:
      't' : integer step index 0, 1, ..., n_steps.
      'b' : basis fraction b_t = e_t / S_t.
      'S' : spot price S_t.
      'e' : basis level e_t = b_t * S_t.
    """
    rng = np.random.default_rng(seed)

    # Correlated innovations: eps_b, eps_S
    Z = rng.standard_normal((2, n_steps))
    eps_b = sigma_b * Z[0]
    eps_S = sigma_S * (rho * Z[0] + np.sqrt(max(1.0 - rho**2, 0.0)) * Z[1])

    b = np.empty(n_steps + 1)
    S = np.empty(n_steps + 1)
    b[0] = b0
    S[0] = S0

    for t in range(n_steps):
        b[t + 1] = phi_ar1 * b[t] + eps_b[t]
        S[t + 1] = S[t] * np.exp((r - q - 0.5 * sigma_S**2) + eps_S[t])

    e = b * S

    return pd.DataFrame({
        "t": np.arange(n_steps + 1),
        "b": b,
        "S": S,
        "e": e,
    })


# ---------------------------------------------------------------------------
# AR(1) estimation
# ---------------------------------------------------------------------------

def estimate_ar1_persistence(b_series: np.ndarray) -> float:
    """
    Estimate the AR(1) coefficient of a basis series via OLS.

    Fits:  b_t = phi * b_{t-1} + intercept + noise,
    using statsmodels OLS.  Returns the estimated phi.

    Parameters
    ----------
    b_series : 1-D array of basis observations, length T+1.

    Returns
    -------
    Estimated AR(1) persistence phi (slope coefficient).
    """
    y = b_series[1:]
    X = add_constant(b_series[:-1])
    model = OLS(y, X).fit()
    return float(model.params[1])   # slope = phi estimate


def estimate_ar1_full(b_series: np.ndarray) -> Dict[str, float]:
    """
    Full OLS AR(1) estimation: returns intercept, phi, R-squared, and std error.

    Parameters
    ----------
    b_series : 1-D array of basis observations.

    Returns
    -------
    Dict with 'intercept', 'phi', 'r_squared', 'phi_se', 'half_life'.
    """
    y = b_series[1:]
    X = add_constant(b_series[:-1])
    res = OLS(y, X).fit()
    phi_hat   = float(res.params[1])
    phi_se    = float(res.bse[1])
    half_life = compute_halflife(phi_hat)
    return {
        "intercept": float(res.params[0]),
        "phi"      : phi_hat,
        "r_squared": float(res.rsquared),
        "phi_se"   : phi_se,
        "half_life": half_life,
    }


# ---------------------------------------------------------------------------
# Half-life conversion
# ---------------------------------------------------------------------------

def compute_halflife(phi: float) -> float:
    """
    Convert an AR(1) persistence phi to a half-life (in steps).

    T_{1/2} = -ln(2) / ln(phi)

    Parameters
    ----------
    phi : AR(1) coefficient, must satisfy 0 < phi < 1.

    Returns
    -------
    Half-life in number of time steps.
    """
    if not (0.0 < phi < 1.0):
        raise ValueError(f"phi must be in (0, 1); got phi={phi}")
    return float(-np.log(2.0) / np.log(phi))


def infer_lambda_from_halflife(halflife_hours: float) -> float:
    """
    Compute the continuous-time mean-reversion speed lambda from a given half-life.

    lambda = ln(2) / halflife_hours

    Parameters
    ----------
    halflife_hours : target half-life in hours.

    Returns
    -------
    lambda (per hour).
    """
    return float(np.log(2.0) / halflife_hours)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_basis_simulation(
    n_steps: int = 10_000,
    phi_ar1: float = 0.72,
    sigma_b: float = 0.0006,
    seed: int = 0,
    phi_tol: float = 0.03,
    hl_tol: float = 0.5,
) -> Dict[str, object]:
    """
    Verify that a long simulated series recovers the target phi ≈ 0.72 and
    half-life ≈ 2.1 hours.

    Parameters
    ----------
    n_steps  : length of the verification path (longer => more accurate).
    phi_ar1  : true AR(1) coefficient.
    sigma_b  : innovation std-dev.
    seed     : RNG seed.
    phi_tol  : tolerance for phi estimation check.
    hl_tol   : tolerance for half-life check (in steps/hours).

    Returns
    -------
    Dict with 'phi_estimated', 'phi_target', 'phi_ok', 'hl_estimated', 'hl_target', 'hl_ok'.
    """
    df = simulate_basis_ou(
        n_steps=n_steps,
        phi_ar1=phi_ar1,
        sigma_b=sigma_b,
        seed=seed,
    )
    phi_est = estimate_ar1_persistence(df["b"].values)
    hl_est  = compute_halflife(phi_est)
    hl_true = compute_halflife(phi_ar1)

    return {
        "phi_estimated": phi_est,
        "phi_target"   : phi_ar1,
        "phi_ok"       : abs(phi_est - phi_ar1) < phi_tol,
        "hl_estimated" : hl_est,
        "hl_target"    : hl_true,
        "hl_ok"        : abs(hl_est - hl_true) < hl_tol,
    }


if __name__ == "__main__":
    # Quick smoke-test
    df = simulate_basis_ou(n_steps=72, phi_ar1=0.72, seed=42)
    print("Simulated columns:", list(df.columns))
    print(df.head())

    # Compute half-life from phi=0.72
    hl = compute_halflife(0.72)
    lam = infer_lambda_from_halflife(2.1)
    print(f"\nphi=0.72 => half-life = {hl:.3f} steps (target ≈ 2.1 hours)")
    print(f"Half-life=2.1 h => lambda = {lam:.4f} / hour")

    print("\n=== Verification on long path ===")
    v = verify_basis_simulation(n_steps=50_000, phi_ar1=0.72, seed=7)
    print(f"  phi estimated  = {v['phi_estimated']:.4f}  (target {v['phi_target']}), OK={v['phi_ok']}")
    print(f"  half-life est  = {v['hl_estimated']:.3f} h (target {v['hl_target']:.3f} h), OK={v['hl_ok']}")
