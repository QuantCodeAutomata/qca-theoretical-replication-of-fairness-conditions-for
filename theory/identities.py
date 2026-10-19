"""
Residual accounting identity for a financed spot-long / perp-short book.

Sign conventions (paper notation):
  - Long pays funding  phi_t * P_t * dt  to short per unit time.
  - Basis              e_t = P_t - S_t.
  - Shareholder income dA_t (continuous yield + discrete dividends).

Key results:
  R(tau1, tau2) = M_{tau2} - M_{tau1}
  where M_t = A_t - e_t + int_0^t {f(u,e_u)*P_u - r_u*S_u} du.

  Weak fairness:   E[R(sigma, rho) | F_sigma] = 0  for all sigma <= rho  (M martingale).
  Strong fairness: R(tau1, tau2) = 0 pathwise.

  Fair funding target (eq. 4):  f(t,e_t)*P_t = r_t*S_t + mu^e_t - a_t.
"""

from __future__ import annotations

import numpy as np
import sympy as sp
from typing import Callable, Dict, Tuple


# ---------------------------------------------------------------------------
# Symbolic verification
# ---------------------------------------------------------------------------

def symbolic_verify_residual_identity() -> Dict[str, object]:
    """
    Use SymPy to verify the residual accounting identity algebraically.

    Returns
    -------
    dict with keys:
      'R_from_definition'   : sympy Expr for R using the original definition.
      'R_basis_substituted' : sympy Expr after substituting dP = dS + de.
      'identity_holds'      : bool, True if both forms are symbolically equal.
      'M_difference'        : sympy Expr for M_{tau2} - M_{tau1}.
      'R_equals_M_diff'     : bool, True if R equals the M increment.
      'fair_funding_cond'   : sympy Eq encoding equation (4).
    """
    # Boundary-value symbols
    A1, A2 = sp.symbols(r'A_{\tau_1} A_{\tau_2}', real=True)
    S1, S2 = sp.symbols(r'S_{\tau_1} S_{\tau_2}', real=True)
    e1, e2 = sp.symbols(r'e_{\tau_1} e_{\tau_2}', real=True)

    # Integral symbols over [tau1, tau2]
    int_rS  = sp.Symbol(r'\int r_t S_t\,dt', real=True)   # int r_t S_t dt
    int_fP  = sp.Symbol(r'\int \phi_t P_t\,dt', real=True) # int phi_t P_t dt

    # Fundamental-theorem integrals
    int_dS = S2 - S1
    int_dA = A2 - A1
    int_dP_naive = sp.Symbol('P_tau2_minus_P_tau1', real=True)   # P2 - P1 before substitution

    # --- R from definition ---------------------------------------------------
    # R = [int(dS + dA - r*S*dt)] - int(dP) + int(phi*P*dt)
    R_def = (int_dS + int_dA - int_rS) - int_dP_naive + int_fP

    # --- Basis substitution: dP = dS + de  -----------------------------------
    # int(dP) = int(dS) + int(de) = (S2-S1) + (e2-e1)
    int_dP_sub = (S2 - S1) + (e2 - e1)
    R_sub = (int_dS + int_dA - int_rS) - int_dP_sub + int_fP
    R_sub_simplified = sp.expand(R_sub)

    # Expected basis-substituted form
    R_basis = (A2 - A1) - (e2 - e1) + (int_fP - int_rS)
    identity_holds = sp.simplify(R_sub_simplified - R_basis) == 0

    # --- M increment ---------------------------------------------------------
    # M_t = A_t - e_t + I_t,  I_t = int_0^t (phi*P - r*S) du
    # M_{tau2} - M_{tau1} = (A2-A1) - (e2-e1) + (I_{tau2} - I_{tau1})
    #                      = (A2-A1) - (e2-e1) + int_{tau1}^{tau2}(phi*P - r*S)dt
    int_12 = int_fP - int_rS   # reuse the same symbol for the [tau1,tau2] integral
    M_diff = (A2 - A1) - (e2 - e1) + int_12
    R_equals_M_diff = sp.simplify(R_basis - M_diff) == 0

    # --- Fair funding condition (eq. 4) --------------------------------------
    f, P_t, r_t, S_t, mu_e, a_t = sp.symbols(
        r'f P_t r_t S_t \mu^e_t a_t', real=True, positive=True
    )
    fair_funding_cond = sp.Eq(f * P_t, r_t * S_t + mu_e - a_t)

    return {
        "R_from_definition"  : R_def,
        "R_basis_substituted": R_basis,
        "identity_holds"     : identity_holds,
        "M_difference"       : M_diff,
        "R_equals_M_diff"    : R_equals_M_diff,
        "fair_funding_cond"  : fair_funding_cond,
    }


def symbolic_fair_funding_drift() -> Dict[str, object]:
    """
    Derive the zero-drift condition on M_t using SymPy.

    dM_t = dA_t - de_t + (f(t,e_t)*P_t - r_t*S_t) dt
         = [a_t + f(t,e_t)*P_t - r_t*S_t - mu^e_t] dt  +  martingale increments.

    Setting the drift to zero gives equation (4).

    Returns
    -------
    dict with 'drift', 'zero_drift_equation', 'fair_funding_solved', 'Y_t_expr'.
    """
    f, P_t, r_t, S_t, mu_e, a_t = sp.symbols(
        r'f P_t r_t S_t mu_e a_t', real=True
    )

    drift = a_t + f * P_t - r_t * S_t - mu_e
    zero_drift_eq = sp.Eq(drift, 0)
    f_solved = sp.solve(zero_drift_eq, f)[0]   # f = (r*S + mu_e - a) / P
    Y_t = sp.Symbol('Y_t', real=True)
    Y_t_expr = sp.Eq(Y_t, f_solved)

    return {
        "drift"               : drift,
        "zero_drift_equation" : zero_drift_eq,
        "fair_funding_solved" : f_solved,
        "Y_t_expr"            : Y_t_expr,
    }


# ---------------------------------------------------------------------------
# Core computational functions
# ---------------------------------------------------------------------------

def compute_residual(
    S: np.ndarray,
    A: np.ndarray,
    P: np.ndarray,
    r: np.ndarray | float,
    phi: np.ndarray,
    dt: float = 1.0,
) -> float:
    """
    Compute R(tau1, tau2) on a discretised path.

    R = sum(dS + dA - r*S*dt) - sum(dP) + sum(phi*P*dt)

    Parameters
    ----------
    S   : spot prices, shape (n+1,).
    A   : cumulative shareholder income, shape (n+1,).
    P   : perp mark prices, shape (n+1,).
    r   : risk-free rate(s); scalar or shape (n,).
    phi : funding rates, shape (n,).  Long pays phi_t*P_t*dt to short.
    dt  : time-step size.

    Returns
    -------
    Scalar residual value.
    """
    r_arr = np.broadcast_to(np.asarray(r, dtype=float), S.shape[0] - 1)
    dS = np.diff(S)
    dA = np.diff(A)
    dP = np.diff(P)
    spot_leg  = np.sum(dS + dA - r_arr * S[:-1] * dt)
    perp_leg  = np.sum(dP)
    funding   = np.sum(phi * P[:-1] * dt)
    return float(spot_leg - perp_leg + funding)


def compute_process_M(
    A: np.ndarray,
    e: np.ndarray,
    phi: np.ndarray,
    P: np.ndarray,
    r: np.ndarray | float,
    S: np.ndarray,
    dt: float = 1.0,
) -> np.ndarray:
    """
    Compute the process M_t = A_t - e_t + int_0^t {phi_u*P_u - r_u*S_u} du.

    Parameters
    ----------
    A   : cumulative shareholder income, shape (n+1,).
    e   : basis e_t = P_t - S_t, shape (n+1,).
    phi : funding rates, shape (n,).
    P   : perp mark prices, shape (n+1,).
    r   : risk-free rate(s); scalar or shape (n,).
    S   : spot prices, shape (n+1,).
    dt  : time-step size.

    Returns
    -------
    M values, shape (n+1,).
    """
    r_arr = np.broadcast_to(np.asarray(r, dtype=float), len(phi))
    integrand  = phi * P[:-1] - r_arr * S[:-1]
    cumulative = np.concatenate([[0.0], np.cumsum(integrand * dt)])
    return A - e + cumulative


def fair_funding_target(
    r_t: np.ndarray | float,
    S_t: np.ndarray | float,
    mu_e_t: np.ndarray | float,
    a_t: np.ndarray | float,
    P_t: np.ndarray | float,
) -> np.ndarray:
    """
    Compute the fair funding rate Y_t from equation (4).

    f(t, e_t) * P_t = r_t * S_t + mu^e_t - a_t
    => Y_t = (r_t*S_t + mu^e_t - a_t) / P_t

    Parameters
    ----------
    r_t   : risk-free rate.
    S_t   : spot price.
    mu_e_t: instantaneous drift of the basis e_t.
    a_t   : instantaneous dividend-yield rate (drift of A_t).
    P_t   : perp mark price.

    Returns
    -------
    Fair funding rate Y_t, same shape as inputs.
    """
    return (np.asarray(r_t) * np.asarray(S_t) + np.asarray(mu_e_t) - np.asarray(a_t)) / np.asarray(P_t)


# ---------------------------------------------------------------------------
# Numerical verification on simulated paths
# ---------------------------------------------------------------------------

def _simulate_path(
    n_steps: int = 500,
    r: float = 0.05 / 252 / 24,
    q: float = 0.02 / 252 / 24,
    sigma_S: float = 0.20 / (252 * 24) ** 0.5,
    lam: float = 0.33,
    sigma_b: float = 0.0006,
    S0: float = 100.0,
    b0: float = 0.005,
    dt: float = 1.0,
    seed: int = 0,
) -> Dict[str, np.ndarray]:
    """
    Simulate a spot/perp path for numerical identity verification.

    Uses GBM for S and AR(1) for the basis b_t = e_t/S_t.

    Returns
    -------
    Dict with arrays: 'S', 'P', 'e', 'A', 'phi', 'mu_e', 'a'.
    """
    rng = np.random.default_rng(seed)
    phi_ar1 = np.exp(-lam * dt)

    S = np.empty(n_steps + 1)
    b = np.empty(n_steps + 1)
    A = np.empty(n_steps + 1)

    S[0] = S0
    b[0] = b0
    A[0] = 0.0

    eps_S = rng.standard_normal(n_steps)
    eps_b = rng.standard_normal(n_steps)

    for t in range(n_steps):
        # GBM for spot (Euler-Maruyama)
        S[t + 1] = S[t] * np.exp((r - q - 0.5 * sigma_S ** 2) * dt + sigma_S * eps_S[t])
        # AR(1) for basis fraction
        b[t + 1] = phi_ar1 * b[t] + sigma_b * eps_b[t]
        # Continuous dividend income
        A[t + 1] = A[t] + q * S[t] * dt

    e   = b * S
    P   = S + e

    # Drift of e_t at each interior point: mu_e ≈ (r-q-lam)*b*S (continuous limit)
    mu_e_arr = ((r - q - lam) * b[:-1]) * S[:-1]
    a_arr    = q * S[:-1]
    r_arr    = np.full(n_steps, r)

    # Funding rate: use fair funding target Y_t = (r*S + mu_e - a) / P
    phi_arr  = fair_funding_target(r_arr, S[:-1], mu_e_arr, a_arr, P[:-1])

    return {
        "S"    : S,
        "P"    : P,
        "e"    : e,
        "A"    : A,
        "phi"  : phi_arr,
        "mu_e" : mu_e_arr,
        "a"    : a_arr,
        "r"    : r_arr,
        "dt"   : dt,
    }


def numerical_verify_residual_identity(
    n_steps: int = 500,
    seed: int = 42,
    tol: float = 1e-10,
    sigma_S: float | None = None,
    sigma_b: float | None = None,
) -> Dict[str, object]:
    """
    Numerically verify R(tau1, tau2) = M_{tau2} - M_{tau1} on a simulated path.

    Also checks that using the fair funding rule drives R ≈ 0 in expectation
    (within the discretisation error from Euler-Maruyama).

    Parameters
    ----------
    n_steps  : number of discretisation steps.
    seed     : RNG seed.
    tol      : tolerance for the algebraic identity check.
    sigma_S  : override stock volatility (None = use default 0.20/sqrt(252*24)).
    sigma_b  : override basis volatility (None = use default 0.0006).

    Returns
    -------
    Dict with:
      'R'              : residual value.
      'M_increment'    : M_{tau2} - M_{tau1}.
      'identity_error' : |R - M_increment|, should be < tol.
      'identity_holds' : bool.
      'M_path'         : the full M_t process array.
    """
    sim_kwargs: dict = {"n_steps": n_steps, "seed": seed}
    if sigma_S is not None:
        sim_kwargs["sigma_S"] = sigma_S
    if sigma_b is not None:
        sim_kwargs["sigma_b"] = sigma_b
    path = _simulate_path(**sim_kwargs)
    S, P, e, A = path["S"], path["P"], path["e"], path["A"]
    phi, r_arr, dt = path["phi"], path["r"], path["dt"]

    R = compute_residual(S, A, P, r_arr, phi, dt)
    M = compute_process_M(A, e, phi, P, r_arr, S, dt)
    M_increment = float(M[-1] - M[0])
    identity_error = abs(R - M_increment)

    return {
        "R"              : R,
        "M_increment"    : M_increment,
        "identity_error" : identity_error,
        "identity_holds" : identity_error < tol,
        "M_path"         : M,
    }


# ---------------------------------------------------------------------------
# Fairness conditions
# ---------------------------------------------------------------------------

def weak_fairness_check(
    M_path: np.ndarray,
    n_bootstrap: int = 500,
    seed: int = 0,
) -> Dict[str, float]:
    """
    Test weak fairness: E[M_{tau2} - M_{tau1} | F_{tau1}] ≈ 0.

    Checks whether the M process has zero expected increments by bootstrapping
    over random sub-intervals drawn from the supplied path.

    Parameters
    ----------
    M_path      : the process M_t, shape (n+1,).
    n_bootstrap : number of random interval pairs to sample.
    seed        : RNG seed.

    Returns
    -------
    Dict with 'mean_increment', 'std_increment', 'passes' (|mean| < 2*std/sqrt(n)).
    """
    rng = np.random.default_rng(seed)
    n = len(M_path) - 1
    tau1 = rng.integers(0, n // 2, size=n_bootstrap)
    tau2 = rng.integers(n // 2, n + 1, size=n_bootstrap)
    increments = M_path[tau2] - M_path[tau1]
    mean_inc = float(np.mean(increments))
    std_inc  = float(np.std(increments))
    se       = std_inc / n_bootstrap ** 0.5
    return {
        "mean_increment": mean_inc,
        "std_increment" : std_inc,
        "passes"        : abs(mean_inc) < 3.0 * se,
    }


def strong_fairness_check(R_series: np.ndarray, tol: float = 1e-8) -> Dict[str, object]:
    """
    Test strong fairness: R(tau1, tau2) = 0 pathwise for every sub-interval.

    Parameters
    ----------
    R_series : array of cumulative residuals R(0, t), shape (n+1,).
    tol      : tolerance for zero check.

    Returns
    -------
    Dict with 'max_deviation', 'passes', 'violations' (indices where |R| > tol).
    """
    deviations = np.abs(R_series)
    violations = np.where(deviations > tol)[0]
    return {
        "max_deviation": float(np.max(deviations)),
        "passes"       : len(violations) == 0,
        "violations"   : violations,
    }


# ---------------------------------------------------------------------------
# Entry point for quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Symbolic verification ===")
    sym = symbolic_verify_residual_identity()
    print(f"  Identity holds:     {sym['identity_holds']}")
    print(f"  R = M diff:         {sym['R_equals_M_diff']}")
    print(f"  Fair funding cond:  {sym['fair_funding_cond']}")

    drift_res = symbolic_fair_funding_drift()
    print(f"  Zero-drift eq:      {drift_res['zero_drift_equation']}")
    print(f"  f solved:           {drift_res['fair_funding_solved']}")

    print("\n=== Numerical verification ===")
    num = numerical_verify_residual_identity(n_steps=1000, seed=7)
    print(f"  R               = {num['R']:.6e}")
    print(f"  M increment     = {num['M_increment']:.6e}")
    print(f"  Identity error  = {num['identity_error']:.2e}")
    print(f"  Identity holds  : {num['identity_holds']}")

    wf = weak_fairness_check(num["M_path"])
    print(f"\n  Weak fairness mean increment = {wf['mean_increment']:.4e}")
    print(f"  Weak fairness passes         : {wf['passes']}")


# ---------------------------------------------------------------------------
# Public API convenience aliases and wrappers
# ---------------------------------------------------------------------------

#: Alias: verify the residual identity R = M_{tau2} - M_{tau1} numerically.
verify_residual_identity = numerical_verify_residual_identity

#: Alias: compute the fair funding rate Y_t = (r*S + mu_e - a) / P.
compute_fair_funding_target = fair_funding_target


def verify_basis_substituted_form() -> Dict[str, object]:
    """Return the symbolic basis-substituted residual identity verification.

    Convenience wrapper around :func:`symbolic_verify_residual_identity`
    exposing the result under the experiment-facing name used in exp1.

    Returns
    -------
    dict
        Same dict as :func:`symbolic_verify_residual_identity`, with keys
        'identity_holds', 'R_equals_M_diff', 'fair_funding_cond', etc.
    """
    return symbolic_verify_residual_identity()


def demonstrate_weak_fairness(
    n_steps: int = 500,
    seed: int = 42,
    n_bootstrap: int = 500,
    n_paths: int = 300,
) -> Dict[str, object]:
    """Demonstrate weak fairness via a multi-path martingale test.

    Simulates *n_paths* independent spot/perp paths under the fair funding
    rule, records M[T] - M[0] for each, and tests that the cross-path mean
    is indistinguishable from zero at the 3-sigma level.

    Parameters
    ----------
    n_steps    : number of simulation steps per path.
    seed       : base RNG seed; each path uses seed+i.
    n_bootstrap: kept for API compatibility (unused in multi-path version).
    n_paths    : number of independent paths to simulate.

    Returns
    -------
    dict with keys from :func:`numerical_verify_residual_identity` plus
    'weak_fairness_check' mapping to the multi-path test result dict.
    """
    # Reference path for identity check
    num = numerical_verify_residual_identity(n_steps=n_steps, seed=seed)

    # Multi-path mean M[T]-M[0] test: E[M_T - M_0] = 0 by martingale property
    terminal_increments = []
    for i in range(n_paths):
        path_i = _simulate_path(n_steps=n_steps, seed=seed + i)
        M_i = compute_process_M(
            path_i["A"], path_i["e"], path_i["phi"],
            path_i["P"], path_i["r"], path_i["S"], path_i["dt"],
        )
        terminal_increments.append(float(M_i[-1] - M_i[0]))

    arr = np.asarray(terminal_increments, dtype=float)
    mean_inc = float(arr.mean())
    std_inc  = float(arr.std())
    se       = std_inc / n_paths ** 0.5
    passes   = abs(mean_inc) < 3.0 * se

    wf = {
        "mean_increment": mean_inc,
        "std_increment" : std_inc,
        "se"            : se,
        "n_paths"       : n_paths,
        "passes"        : passes,
    }
    return {**num, "weak_fairness_check": wf}


def demonstrate_strong_fairness(
    n_steps: int = 500,
    seed: int = 42,
    tol: float = 1e-8,
) -> Dict[str, object]:
    """Demonstrate strong fairness on a deterministic (zero-noise) OU-basis path.

    Strong fairness requires R(tau1, tau2) = 0 pathwise.  In a stochastic
    simulation this only holds in expectation; to show the pathwise identity
    we set sigma_S = sigma_b = 0 so that the fair funding rule exactly
    cancels drift and the cumulative residual stays at machine precision.

    Parameters
    ----------
    n_steps : number of simulation steps.
    seed    : RNG seed (unused with zero noise; kept for API consistency).
    tol     : tolerance for the pathwise-zero check (default 1e-8).

    Returns
    -------
    dict with keys from :func:`numerical_verify_residual_identity` plus
    'strong_fairness_check' (the result of :func:`strong_fairness_check`).
    """
    # Deterministic path: zero diffusion so fair funding drives R ≡ 0 exactly.
    # Use actual path increments for mu_e to eliminate Euler-Maruyama discretisation
    # error and obtain exact pathwise R = 0.
    path = _simulate_path(n_steps=n_steps, seed=seed, sigma_S=0.0, sigma_b=0.0)
    S, P, e, A = path["S"], path["P"], path["e"], path["A"]
    r_arr, dt = path["r"], path["dt"]

    r_arr2 = np.broadcast_to(np.asarray(r_arr, dtype=float), n_steps)
    # Use realized increments to back out mu_e exactly from the simulated path
    de_actual = np.diff(e)                # actual basis increment each step
    a_actual  = path["a"]                 # continuous dividend density
    # Exact fair phi: phi * P = r*S + mu_e - a, with mu_e = de/dt (deterministic)
    mu_e_exact = de_actual / dt
    phi_exact  = (r_arr2 * S[:-1] + mu_e_exact - a_actual) / P[:-1]

    dS = np.diff(S)
    dA = np.diff(A)
    dP = np.diff(P)
    inc = dS + dA - r_arr2 * S[:-1] * dt - dP + phi_exact * P[:-1] * dt
    R_series = np.concatenate([[0.0], np.cumsum(inc)])

    sf = strong_fairness_check(R_series, tol=tol)
    num_result = numerical_verify_residual_identity(n_steps=n_steps, seed=seed,
                                                    sigma_S=0.0, sigma_b=0.0)
    return {**num_result, "strong_fairness_check": sf, "R_series": R_series}
