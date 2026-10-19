"""
Deployed and fair funding formulas for perpetual contracts.

Deployed (premium-index) rule:
    f_deploy(b_t) = (r - q) + kappa * b_t

Fair rule (equation 8 from the OU model):
    f_fair(b_t) = (r - q) - (lambda * b_t - rho * sigma_S * sigma_b) / (1 + b_t)

Small-b approximation of fair rule (rho = 0):
    f_fair(b_t) ≈ (r - q) - lambda * b_t

Key structural difference:
    df_deploy/db = kappa > 0   (positive slope)
    df_fair/db   = -(lambda + rho*sigma_S*sigma_b) / (1+b)^2  (negative for typical params)

A positive-premium environment (b > 0) triggers HIGHER deployed funding but
LOWER (more negative adjustment) fair funding — a qualitative sign mismatch.
"""

from __future__ import annotations

import numpy as np
from typing import Dict


# ---------------------------------------------------------------------------
# Funding-rate formulas
# ---------------------------------------------------------------------------

def deployed_funding_rule(
    b: np.ndarray,
    r: float,
    q: float,
    kappa: float = 0.05,
) -> np.ndarray:
    """
    Premium-index funding rule.

    f_deploy(b_t) = (r - q) + kappa * b_t

    Parameters
    ----------
    b     : basis fraction b_t = (P_t - S_t) / S_t, shape (T,).
    r     : risk-free rate per step.
    q     : continuous dividend yield per step.
    kappa : premium-index slope (> 0 by convention).

    Returns
    -------
    Deployed funding rate, shape (T,).
    """
    return (r - q) + kappa * np.asarray(b, dtype=float)


def fair_funding_rule(
    b: np.ndarray,
    r: float,
    q: float,
    lam: float,
    sigma_b: float = 0.0006,
    sigma_S: float = 0.0,
    rho: float = 0.0,
) -> np.ndarray:
    """
    Fair settlement funding from equation (8) of the OU model.

    f_fair(b_t) = (r - q) - (lambda * b_t - rho * sigma_S * sigma_b) / (1 + b_t)

    Parameters
    ----------
    b        : basis fraction b_t, shape (T,).
    r, q     : risk-free rate and continuous yield per step.
    lam      : OU mean-reversion speed lambda (> 0).
    sigma_b  : basis OU volatility.
    sigma_S  : spot log-volatility per step.
    rho      : W^S – W^b correlation.

    Returns
    -------
    Fair funding rate, shape (T,).
    """
    b_arr = np.asarray(b, dtype=float)
    return (r - q) - (lam * b_arr - rho * sigma_S * sigma_b) / (1.0 + b_arr)


def fair_funding_small_b_approx(
    b: np.ndarray,
    r: float,
    q: float,
    lam: float,
) -> np.ndarray:
    """
    Small-b approximation for the fair funding rate (rho = 0).

    f_fair(b_t) ≈ (r - q) - lambda * b_t   for |b_t| << 1.

    Parameters
    ----------
    b     : basis fraction b_t, shape (T,).
    r, q  : risk-free rate and continuous yield per step.
    lam   : OU mean-reversion speed.

    Returns
    -------
    Approximate fair funding rate, shape (T,).
    """
    return (r - q) - lam * np.asarray(b, dtype=float)


def cumulative_funding(
    funding_rates: np.ndarray,
    P: np.ndarray,
    dt: float = 1.0,
) -> np.ndarray:
    """
    Compute cumulative funding paid by the perp long.

    Cumulative_funding(t) = sum_{s=0}^{t-1} f_s * P_s * dt.

    Parameters
    ----------
    funding_rates : f_t, shape (T,).
    P             : perp mark prices at the start of each step, shape (T,).
    dt            : step size.

    Returns
    -------
    Cumulative funding, shape (T+1,); first element is 0.
    """
    return np.concatenate(
        [[0.0], np.cumsum(np.asarray(funding_rates) * np.asarray(P) * dt)]
    )


# ---------------------------------------------------------------------------
# Comparison functions
# ---------------------------------------------------------------------------

def compare_funding_rules_pointwise(
    b: np.ndarray,
    r: float,
    q: float,
    lam: float,
    sigma_b: float,
    sigma_S: float,
    rho: float,
    kappa: float,
) -> Dict[str, np.ndarray]:
    """
    Pointwise comparison of deployed and fair funding rates.

    Highlights cases where b > 0 gives positive deployed funding
    but a negative adjustment in the fair rule.

    Parameters
    ----------
    b                   : basis fractions, shape (T,).
    r, q                : risk-free rate and continuous yield.
    lam                 : OU mean-reversion speed.
    sigma_b, sigma_S    : basis and spot volatilities.
    rho                 : W^S–W^b correlation.
    kappa               : deployed rule premium-index slope.

    Returns
    -------
    Dict with 'b', 'f_deploy', 'f_fair', 'f_small_b', 'diff' (fair - deploy),
    'positive_premium_mask' (b > 0), 'sign_conflict_mask' (diff and f_deploy opposite signs).
    """
    b_arr   = np.asarray(b, dtype=float)
    f_dep   = deployed_funding_rule(b_arr, r, q, kappa)
    f_fair  = fair_funding_rule(b_arr, r, q, lam, sigma_b, sigma_S, rho)
    f_sb    = fair_funding_small_b_approx(b_arr, r, q, lam)
    diff    = f_fair - f_dep

    # Positive premium => positive deployed but negative fair adjustment
    pos_premium = b_arr > 0.0
    carry       = r - q
    sign_conflict = pos_premium & ((f_fair - carry) < 0) & ((f_dep - carry) > 0)

    return {
        "b"                   : b_arr,
        "f_deploy"            : f_dep,
        "f_fair"              : f_fair,
        "f_small_b"           : f_sb,
        "diff"                : diff,
        "positive_premium_mask": pos_premium,
        "sign_conflict_mask"  : sign_conflict,
    }


def compare_cumulative_funding(
    b_path: np.ndarray,
    P_path: np.ndarray,
    r: float,
    q: float,
    lam: float,
    sigma_b: float,
    sigma_S: float,
    rho: float,
    kappa: float,
    dt: float = 1.0,
) -> Dict[str, np.ndarray]:
    """
    Compare cumulative funding under deployed and fair rules along a simulated path.

    Parameters
    ----------
    b_path  : basis fractions at each step, shape (T,).
    P_path  : perp mark prices at start of each step, shape (T,).
    r, q    : risk-free rate and continuous yield.
    lam     : OU mean-reversion speed.
    sigma_b, sigma_S : basis and spot volatilities.
    rho     : correlation.
    kappa   : deployed rule slope.
    dt      : step size.

    Returns
    -------
    Dict with 'f_deploy', 'f_fair', 'cum_deploy', 'cum_fair', 'cum_diff'.
    """
    f_dep  = deployed_funding_rule(b_path, r, q, kappa)
    f_fair = fair_funding_rule(b_path, r, q, lam, sigma_b, sigma_S, rho)

    cum_dep  = cumulative_funding(f_dep, P_path, dt)
    cum_fair = cumulative_funding(f_fair, P_path, dt)

    return {
        "f_deploy" : f_dep,
        "f_fair"   : f_fair,
        "cum_deploy": cum_dep,
        "cum_fair"  : cum_fair,
        "cum_diff"  : cum_fair - cum_dep,
    }


def slope_comparison(
    b_grid: np.ndarray,
    r: float,
    q: float,
    lam: float,
    sigma_b: float,
    sigma_S: float,
    rho: float,
    kappa: float,
) -> Dict[str, np.ndarray]:
    """
    Return pointwise slopes of the deployed and fair funding rules.

    Deployed slope: df_deploy/db = kappa  (constant, positive).
    Fair slope:     df_fair/db   = -(lambda + rho*sigma_S*sigma_b) / (1+b)^2.

    Parameters
    ----------
    b_grid                  : grid of basis values, shape (K,).
    r, q, lam, sigma_b, sigma_S, rho, kappa : model parameters.

    Returns
    -------
    Dict with 'b_grid', 'slope_deploy', 'slope_fair', 'sign_mismatch'.
    """
    b_arr       = np.asarray(b_grid, dtype=float)
    slope_dep   = np.full_like(b_arr, kappa)
    slope_fair  = -(lam + rho * sigma_S * sigma_b) / (1.0 + b_arr) ** 2
    sign_mismatch = np.sign(slope_dep) != np.sign(slope_fair)

    return {
        "b_grid"       : b_arr,
        "slope_deploy" : slope_dep,
        "slope_fair"   : slope_fair,
        "sign_mismatch": sign_mismatch,
    }


if __name__ == "__main__":
    r, q    = 0.05 / 252 / 24, 0.02 / 252 / 24
    lam     = np.log(2) / 2.1
    sig_b   = 0.0006
    sig_S   = 0.20 / (252 * 24) ** 0.5
    rho     = 0.0
    kappa   = 0.05

    b_grid = np.linspace(-0.05, 0.05, 11)

    print("=== Pointwise comparison ===")
    pw = compare_funding_rules_pointwise(b_grid, r, q, lam, sig_b, sig_S, rho, kappa)
    print(f"  {'b':>8} {'f_deploy':>12} {'f_fair':>12} {'diff':>10}")
    for i in range(len(b_grid)):
        print(f"  {pw['b'][i]:8.4f} {pw['f_deploy'][i]:12.6f} "
              f"{pw['f_fair'][i]:12.6f} {pw['diff'][i]:10.6f}")

    print(f"\n  Sign conflicts (b>0, deploy>carry, fair<carry): "
          f"{pw['sign_conflict_mask'].sum()} / {len(b_grid)}")

    print("\n=== Slope comparison ===")
    sc = slope_comparison(b_grid, r, q, lam, sig_b, sig_S, rho, kappa)
    print(f"  Deployed slope (kappa): {sc['slope_deploy'][0]:.4f}")
    print(f"  Fair slope at b=0:      {sc['slope_fair'][len(b_grid)//2]:.4f}")
    print(f"  Sign mismatch everywhere: {bool(np.all(sc['sign_mismatch']))}")
