"""
OU / AR(1)-based fair funding rule derivation.

Model (continuous-time):
    dA_t        = q * S_t * dt                                    (dividend accrual)
    db_t        = -lambda * b_t * dt + sigma_b * dW^b_t           (OU basis fraction)
    dS_t        = (r - q) * S_t * dt + sigma_S * S_t * dW^S_t    (spot GBM)
    d<W^S, W^b>_t = rho * dt                                      (correlation)

Basis level: e_t = b_t * S_t,  perp mark: P_t = S_t * (1 + b_t).

By Itô's product rule:
    de_t = S_t * db_t + b_t * dS_t + d<b, S>_t
         = S_t [b_t*(r-q-lambda) + rho*sigma_b*sigma_S] dt  +  martingale part

Applying equation (4) (f*P = r*S + mu^e - a):

    f_fair(b_t) = (r-q) - (lambda*b_t - rho*sigma_S*sigma_b) / (1 + b_t)   [eq. 8]

Small-b approximation (rho=0):
    f_fair(b_t) ≈ (r-q) - lambda * b_t

Derivative:
    f'_fair(b) = -(lambda + rho*sigma_S*sigma_b) / (1+b)^2

Deployed premium-index rule:
    f_deploy(b) = (r-q) + kappa * b_t   (positive slope kappa > 0)

The sign mismatch (f_fair has negative slope, f_deploy has positive slope) is
the core structural discrepancy identified in the paper.
"""

from __future__ import annotations

import numpy as np
import sympy as sp
from scipy.stats import gaussian_kde
from typing import Dict, Tuple


# ---------------------------------------------------------------------------
# Symbolic derivation
# ---------------------------------------------------------------------------

def symbolic_derive_fair_funding() -> Dict[str, object]:
    """
    Symbolically derive equation (8) using SymPy and verify the derivative.

    Returns
    -------
    Dict with:
      'f_fair_expr'       : sympy Expr for f_fair(b).
      'f_fair_simplified' : simplified form.
      'small_b_approx'    : leading-order series in b (rho=0).
      'df_fair_db'        : derivative df/db.
      'df_fair_simplified': simplified derivative.
      'slope_sign'        : sympy expression whose sign determines slope.
      'zero_drift_eq'     : the zero-drift equation solved for f.
    """
    b, r, q, lam, sigma_b, sigma_S, rho = sp.symbols(
        'b r q lambda sigma_b sigma_S rho', real=True
    )

    # --- Derive from zero-drift condition: f*P = r*S + mu_e - a ----------
    # mu_e = drift of e_t = S_t*[b_t*(r-q-lambda) + rho*sigma_b*sigma_S]
    # a_t  = q*S_t
    # P_t  = S_t*(1+b_t)
    # => f = (r*S + mu_e - a) / P
    #      = [r + b*(r-q-lambda) + rho*sigma_b*sigma_S - q] / (1+b)

    S_sym = sp.Symbol('S', positive=True)
    mu_e  = S_sym * (b * (r - q - lam) + rho * sigma_b * sigma_S)
    a_sym = q * S_sym
    P_sym = S_sym * (1 + b)

    f_num   = r * S_sym + mu_e - a_sym
    f_expr  = sp.simplify(f_num / P_sym)   # S cancels

    # Closed-form equation (8)
    f_eq8   = (r - q) - (lam * b - rho * sigma_S * sigma_b) / (1 + b)
    equiv   = sp.simplify(f_expr - f_eq8)    # 0 if derivation is correct
    equiv_is_zero = bool(equiv == 0)

    # Small-b approximation (rho = 0): series around b=0
    f_rho0  = f_eq8.subs(rho, 0)
    small_b = sp.series(f_rho0, b, 0, n=3).removeO()

    # Derivative with respect to b
    df_db   = sp.diff(f_eq8, b)
    df_simplified = sp.simplify(df_db)
    # Expected: -(lambda + rho*sigma_S*sigma_b) / (1+b)^2
    slope_sign = lam + rho * sigma_S * sigma_b   # negative when > 0

    return {
        "f_fair_expr"       : f_expr,
        "f_eq8"             : f_eq8,
        "derivation_equiv"  : equiv_is_zero,   # True iff f_expr == f_eq8 symbolically
        "small_b_approx"    : small_b,
        "df_fair_db"        : df_db,
        "df_fair_simplified": df_simplified,
        "slope_sign"        : slope_sign,
    }


# ---------------------------------------------------------------------------
# Numerical funding-rule functions
# ---------------------------------------------------------------------------

def _compute_fair_funding_rule_ou(
    b: np.ndarray | float,
    r: float,
    q: float,
    lam: float,
    sigma_b: float,
    sigma_S: float,
    rho: float,
) -> np.ndarray:
    """
    Full OU fair funding rate from equation (8) including correlation term.

    f_fair(b_t) = (r - q) - (lambda*b_t - rho*sigma_S*sigma_b) / (1 + b_t)

    Parameters
    ----------
    b        : premium/spot basis fraction (scalar or array).
    r, q     : risk-free rate and continuous dividend yield.
    lam      : OU mean-reversion speed lambda (> 0).
    sigma_b  : OU basis volatility.
    sigma_S  : spot log-volatility.
    rho      : correlation d<W^S, W^b>/dt.

    Returns
    -------
    Fair funding rate, same shape as b.
    """
    b_arr = np.asarray(b, dtype=float)
    return (r - q) - (lam * b_arr - rho * sigma_S * sigma_b) / (1.0 + b_arr)


def compute_fair_funding_rule(
    r: float,
    q: float,
    lambda_ou: float,
    b: float | np.ndarray,
    rho: float = 0.0,
    sigma_b: float = 0.0,
    sigma_S: float = 0.0,
) -> float | np.ndarray:
    """Evaluate the OU fair funding rate — Eq. (8).

    Simplified public interface.  With rho=0 (default) the formula reduces to:

        f_fair(b) = (r - q) - lambda * b / (1 + b)

    For small b this approximates to (r - q) - lambda * b.

    The slope ∂f_fair/∂b = -lambda / (1+b)^2 < 0 for lambda > 0.

    Parameters
    ----------
    r         : risk-free rate (same time unit as lambda_ou).
    q         : continuous dividend yield.
    lambda_ou : OU mean-reversion speed (positive).
    b         : relative basis (r - q)(same unit); scalar or array.
    rho       : W^S–W^b correlation (default 0).
    sigma_b   : basis volatility (used only when rho != 0).
    sigma_S   : spot volatility (used only when rho != 0).

    Returns
    -------
    float or np.ndarray
        Fair funding rate, same shape as b.
    """
    scalar = np.ndim(b) == 0
    result = _compute_fair_funding_rule_ou(b, r, q, lambda_ou, sigma_b, sigma_S, rho)
    return float(result) if scalar else result


def fair_funding_derivative(
    b: np.ndarray | float,
    lam: float,
    sigma_b: float,
    sigma_S: float,
    rho: float,
) -> np.ndarray:
    """
    Derivative of f_fair with respect to b.

    f'_fair(b) = -(lambda + rho*sigma_S*sigma_b) / (1+b)^2

    Parameters
    ----------
    b        : premium/spot basis fraction.
    lam      : OU mean-reversion speed.
    sigma_b  : basis volatility.
    sigma_S  : spot volatility.
    rho      : W^S–W^b correlation.

    Returns
    -------
    df_fair/db, same shape as b.
    """
    b_arr = np.asarray(b, dtype=float)
    return -(lam + rho * sigma_S * sigma_b) / (1.0 + b_arr) ** 2


def fair_funding_small_b_approx(
    b: np.ndarray | float,
    r: float,
    q: float,
    lam: float,
) -> np.ndarray:
    """
    Small-b approximation for the fair funding rate (rho = 0).

    f_fair(b_t) ≈ (r - q) - lambda * b_t   for |b_t| << 1.

    Parameters
    ----------
    b     : premium/spot basis fraction.
    r, q  : risk-free rate and continuous yield.
    lam   : OU mean-reversion speed.

    Returns
    -------
    Approximate fair funding rate, same shape as b.
    """
    return (r - q) - lam * np.asarray(b, dtype=float)


def compare_fair_vs_deployed(
    b_grid: np.ndarray | None = None,
    r: float = 0.05 / 8760,
    q: float = 0.02 / 8760,
    lam: float = 0.329,
    sigma_b: float = 0.0006,
    sigma_S: float = 0.20 / (252 * 24) ** 0.5,
    rho: float = 0.0,
    kappa: float = 0.05,
) -> Dict[str, np.ndarray]:
    """Compare fair and deployed funding rules over a grid of basis values.

    Deployed premium-index rule: f_deploy(b) = (r-q) + kappa * b.
    Fair rule (eq. 8):           f_fair(b)   = (r-q) - (lambda*b - rho*sigma_S*sigma_b)/(1+b).

    Parameters
    ----------
    b_grid              : 1-D array of basis fraction values; defaults to
                          np.linspace(-0.05, 0.05, 201).
    r, q                : risk-free rate and continuous yield.
    lam                 : OU mean-reversion speed.
    sigma_b, sigma_S    : basis and spot volatilities.
    rho                 : correlation.
    kappa               : deployed rule premium-index slope (> 0 typically).

    Returns
    -------
    Dict with 'b_grid', 'f_deploy', 'f_fair', 'f_small_b', 'diff' (fair - deploy),
    'slope_deploy', 'slope_fair'.
    """
    if b_grid is None:
        b_grid = np.linspace(-0.05, 0.05, 201)
    b_grid = np.asarray(b_grid, dtype=float)
    f_dep   = (r - q) + kappa * b_grid
    f_fair  = _compute_fair_funding_rule_ou(b_grid, r, q, lam, sigma_b, sigma_S, rho)
    f_sb    = fair_funding_small_b_approx(b_grid, r, q, lam)
    df_dep  = np.full_like(b_grid, kappa)
    df_fair = fair_funding_derivative(b_grid, lam, sigma_b, sigma_S, rho)

    return {
        "b_grid"      : b_grid,
        "f_deploy"    : f_dep,
        "f_fair"      : f_fair,
        "f_small_b"   : f_sb,
        "diff"        : f_fair - f_dep,
        "slope_deploy": df_dep,
        "slope_fair"  : df_fair,
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
    Return pointwise slopes of deployed and fair funding rules.

    The deployed slope is +kappa (positive); the fair slope is
    -(lambda + rho*sigma_S*sigma_b)/(1+b)^2 (negative for typical parameters).

    Parameters
    ----------
    b_grid, r, q, lam, sigma_b, sigma_S, rho, kappa : see compare_fair_vs_deployed.

    Returns
    -------
    Dict with 'b_grid', 'slope_deploy', 'slope_fair', 'sign_mismatch' (bool array).
    """
    slope_dep  = np.full_like(b_grid, float(kappa))
    slope_fair = fair_funding_derivative(b_grid, lam, sigma_b, sigma_S, rho)
    sign_mismatch = np.sign(slope_dep) != np.sign(slope_fair)

    return {
        "b_grid"       : b_grid,
        "slope_deploy" : slope_dep,
        "slope_fair"   : slope_fair,
        "sign_mismatch": sign_mismatch,
    }


# ---------------------------------------------------------------------------
# Kernel-regression projection benchmark
# ---------------------------------------------------------------------------

def projection_nonneg(Y_t: float | np.ndarray) -> float | np.ndarray:
    """Project the fair funding target onto non-negative rates.

    f_clip(Y_t) = max(Y_t, 0)

    The smallest non-negative approximation to the fair funding target Y_t.

    Parameters
    ----------
    Y_t : float or np.ndarray
        Fair funding target value(s).

    Returns
    -------
    float or np.ndarray
        Non-negative projection max(Y_t, 0), same shape as Y_t.
    """
    scalar = np.ndim(Y_t) == 0
    result = np.maximum(np.asarray(Y_t, dtype=float), 0.0)
    return float(result) if scalar else result


def projection_benchmark(
    Y_samples: np.ndarray,
    P_samples: np.ndarray,
    e_samples: np.ndarray,
    x_grid: np.ndarray,
    bandwidth: float | None = None,
) -> np.ndarray:
    """P_t^2-weighted Nadaraya-Watson estimate of f*(x) = E[Y_t | e_t = x].

    Implements the projection benchmark from the paper:

        f*(x) = E[P_t^2 * Y_t | e_t = x] / E[P_t^2 | e_t = x]

    using a Gaussian kernel (Nadaraya-Watson) regression with P_t^2 weights.

    Parameters
    ----------
    Y_samples  : fair funding target Y_t = (r*S+mu_e-a)/P, shape (T,).
    P_samples  : perp mark prices P_t, shape (T,).
    e_samples  : basis levels e_t, shape (T,).
    x_grid     : evaluation points for f*, shape (K,).
    bandwidth  : kernel bandwidth h; defaults to Silverman's rule on e_samples.

    Returns
    -------
    np.ndarray
        f*(x_grid), shape (K,).
    """
    Y = np.asarray(Y_samples, dtype=float)
    P = np.asarray(P_samples, dtype=float)
    e = np.asarray(e_samples, dtype=float)
    x = np.asarray(x_grid, dtype=float)

    if bandwidth is None:
        n = len(e)
        bandwidth = 1.06 * np.std(e) * n ** (-0.2)

    W2 = P ** 2
    f_star = np.empty(len(x))

    for i, xi in enumerate(x):
        K = np.exp(-0.5 * ((e - xi) / bandwidth) ** 2)
        wK = W2 * K
        denom = wK.sum()
        f_star[i] = (wK * Y).sum() / denom if denom > 1e-300 else np.nan

    return f_star


# ---------------------------------------------------------------------------
# OU simulation helper for projection demo
# ---------------------------------------------------------------------------

def simulate_ou_fair_path(
    n_steps: int = 5_000,
    r: float = 0.05 / 252 / 24,
    q: float = 0.02 / 252 / 24,
    lam: float = np.log(2) / 2.1,
    sigma_b: float = 0.0006,
    sigma_S: float = 0.20 / (252 * 24) ** 0.5,
    rho: float = 0.0,
    b0: float = 0.0,
    S0: float = 100.0,
    seed: int = 0,
) -> Dict[str, np.ndarray]:
    """
    Simulate the OU-based spot/perp model and compute f_fair along the path.

    Parameters
    ----------
    n_steps, r, q, lam, sigma_b, sigma_S, rho, b0, S0, seed : model parameters.

    Returns
    -------
    Dict with 'b', 'S', 'e', 'P', 'Y' (= f_fair evaluated along the path),
    'mu_e' (drift of e), 'a' (dividend drift).
    """
    rng = np.random.default_rng(seed)
    phi_ar1 = np.exp(-lam)   # one step = one hour

    b = np.empty(n_steps + 1)
    S = np.empty(n_steps + 1)
    b[0] = b0
    S[0] = S0

    Z = rng.standard_normal((2, n_steps))
    # Cholesky: eps_b = sigma_b * Z[0],  eps_S uncorrelated direction
    eps_b = sigma_b * Z[0]
    eps_S = sigma_S * (rho * Z[0] + np.sqrt(max(1.0 - rho**2, 0.0)) * Z[1])

    for t in range(n_steps):
        b[t + 1] = phi_ar1 * b[t] + eps_b[t]
        S[t + 1] = S[t] * np.exp((r - q - 0.5 * sigma_S**2) + eps_S[t])

    e   = b * S
    P   = S * (1.0 + b)

    # Drift of e_t at each step (used for Y_t)
    mu_e = S[:-1] * (b[:-1] * (r - q - lam) + rho * sigma_b * sigma_S)
    a    = q * S[:-1]
    Y    = fair_funding_target_array(r, S[:-1], mu_e, a, P[:-1])

    return {"b": b, "S": S, "e": e, "P": P, "Y": Y, "mu_e": mu_e, "a": a}


def fair_funding_target_array(
    r: float,
    S: np.ndarray,
    mu_e: np.ndarray,
    a: np.ndarray,
    P: np.ndarray,
) -> np.ndarray:
    """
    Vectorised fair funding target Y_t = (r*S_t + mu^e_t - a_t) / P_t.

    Parameters
    ----------
    r    : scalar risk-free rate.
    S    : spot prices, shape (T,).
    mu_e : basis drift, shape (T,).
    a    : dividend accrual rate, shape (T,).
    P    : perp mark prices, shape (T,).

    Returns
    -------
    Y_t, shape (T,).
    """
    return (r * S + mu_e - a) / P


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Symbolic derivation ===")
    sym = symbolic_derive_fair_funding()
    print(f"  f_fair (eq. 8):     {sym['f_eq8']}")
    print(f"  Derivation equiv:   {sym['derivation_equiv']}")
    print(f"  Small-b approx:     {sym['small_b_approx']}")
    print(f"  df_fair/db:         {sym['df_fair_simplified']}")

    print("\n=== Numerical funding rule comparison ===")
    r, q   = 0.05 / 252 / 24, 0.02 / 252 / 24
    lam    = np.log(2) / 2.1
    sig_b, sig_S, rho, kappa = 0.0006, 0.20 / (252 * 24)**0.5, 0.0, 0.05

    b_grid = np.linspace(-0.05, 0.05, 11)
    cmp = compare_fair_vs_deployed(b_grid, r, q, lam, sig_b, sig_S, rho, kappa)
    print(f"  {'b':>8} {'f_deploy':>12} {'f_fair':>12} {'diff':>12}")
    for i in range(len(b_grid)):
        print(f"  {cmp['b_grid'][i]:8.4f} "
              f"{cmp['f_deploy'][i]:12.6f} "
              f"{cmp['f_fair'][i]:12.6f} "
              f"{cmp['diff'][i]:12.6f}")

    sc = slope_comparison(b_grid, r, q, lam, sig_b, sig_S, rho, kappa)
    print(f"\n  Deployed slope: {sc['slope_deploy'][0]:.4f} (positive)")
    print(f"  Fair slope at b=0: {sc['slope_fair'][5]:.4f} (negative)")
    print(f"  Sign mismatch everywhere: {bool(np.all(sc['sign_mismatch']))}")

    print("\n=== Projection benchmark ===")
    path = simulate_ou_fair_path(n_steps=2000, seed=42)
    x_grid  = np.linspace(np.percentile(path["e"][:-1], 5),
                           np.percentile(path["e"][:-1], 95), 30)
    # projection_benchmark: kernel-regression estimate of f*(x)=E[P^2*Y|e=x]/E[P^2|e=x]
    f_star = projection_benchmark(
        path["Y"], path["P"][:-1], path["e"][:-1], x_grid
    )
    b_approx = x_grid / np.median(path["S"][:-1])
    f_exact = compute_fair_funding_rule(r=r, q=q, lambda_ou=lam, b=b_approx,
                                        rho=rho, sigma_b=sig_b, sigma_S=sig_S)
    print(f"  Mean |f* - f_exact| ≈ {np.nanmean(np.abs(f_star - f_exact)):.2e}")

    # Non-negative projection clipping examples
    for Yt in [-0.01, 0.0, 0.01]:
        print(f"  projection_nonneg({Yt:+.4f}) = {projection_nonneg(Yt):.4f}")
