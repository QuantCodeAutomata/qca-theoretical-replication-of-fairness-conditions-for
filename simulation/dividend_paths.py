"""
Simulate three cumulative wealth paths for a spot long vs. perp long comparison
around an ex-dividend event.

Paths compared:
  1. Physical shareholder  — holds spot, receives discrete dividend D at tau.
  2. Standard perp long    — holds perp mark, pays/earns funding; NO event credit.
  3. Event-credit perp long— same as standard plus a discrete credit of D at tau.

Key result:
  standard_perp_wealth(t) = shareholder_wealth(t) - D   for all t >= tau,
  event_credit_wealth(t)  = shareholder_wealth(t)        for all t >= tau.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict


def simulate_dividend_paths(
    S0: float = 100.0,
    r: float = 0.05 / 252 / 24,      # hourly risk-free rate
    q: float = 0.02 / 252 / 24,      # hourly continuous dividend yield
    sigma_S: float = 0.20 / (252 * 24) ** 0.5,  # hourly GBM vol
    D_frac: float = 0.01,            # dividend = 1% of S_{tau-}
    tau: int = 36,                   # ex-dividend step (centre of 72-hour window)
    n_steps: int = 72,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Simulate spot and perp wealth paths around an ex-dividend event.

    Parameters
    ----------
    S0      : initial spot price.
    r       : risk-free rate per time-step (hourly).
    q       : continuous dividend yield per time-step (hourly).
    sigma_S : spot log-volatility per time-step (hourly).
    D_frac  : discrete dividend as a fraction of the spot price just before tau.
    tau     : time-step index of the ex-dividend event.
    n_steps : total number of steps in the simulation.
    seed    : RNG seed for reproducibility.

    Returns
    -------
    pd.DataFrame with columns:
      't'                   : integer time steps 0, 1, ..., n_steps.
      'S'                   : spot price.
      'P'                   : perp mark price (= S, zero-basis assumption).
      'shareholder_wealth'  : S_t - S_0 + accumulated dividends (continuous + discrete).
      'standard_perp_wealth': P_t - P_0 - cumulative_funding (no discrete credit).
      'event_credit_wealth' : standard_perp_wealth + D * 1{t >= tau}.
    """
    rng = np.random.default_rng(seed)
    dt = 1.0  # one step = one hour

    S = np.empty(n_steps + 1)
    A_cont = np.empty(n_steps + 1)   # continuous dividend income
    S[0] = S0
    A_cont[0] = 0.0

    eps = rng.standard_normal(n_steps)
    for t in range(n_steps):
        S[t + 1] = S[t] * np.exp((r - q - 0.5 * sigma_S**2) * dt + sigma_S * eps[t])
        A_cont[t + 1] = A_cont[t] + q * S[t] * dt

    # Discrete dividend at tau: D = D_frac * S_{tau-}
    D = D_frac * S[tau]

    # Ex-dividend spot-price drop at tau
    S[tau] = S[tau] - D

    # Perp mark = spot (zero-basis), so P also reflects the ex-div drop
    P = S.copy()

    # --- Funding rate: premium-index with f = (r - q) ----------------------
    # (zero basis => f = r - q, so funding accrual exactly cancels carry)
    phi = np.full(n_steps, r - q)
    cum_funding = np.concatenate([[0.0], np.cumsum(phi * P[:-1] * dt)])

    # Financing cost of holding the spot position (borrowed at rate r).
    # The residual identity subtracts int(r*S*dt) from the spot leg.
    cum_rS = np.concatenate([[0.0], np.cumsum(r * S[:-1] * dt)])

    # --- 1. Physical shareholder wealth (financed) -------------------------
    # Net P&L = change in spot + continuous dividends + discrete dividend
    #           - financing cost (borrow at r to hold spot)
    # => shareholder_net = (S_t-S_0) + A_cont + D*1{t>=tau} - cum_rS
    # Then difference vs standard perp = exactly D for all t >= tau.
    discrete_div = np.where(np.arange(n_steps + 1) >= tau, D, 0.0)
    shareholder_wealth = (S - S0) + A_cont + discrete_div - cum_rS

    # --- 2. Standard perp long (no event credit) ----------------------------
    # P&L = (P_t - P_0) - cumulative funding paid
    standard_perp_wealth = (P - P[0]) - cum_funding

    # --- 3. Event-credit perp long ------------------------------------------
    event_credit = np.where(np.arange(n_steps + 1) >= tau, D, 0.0)
    event_credit_wealth = standard_perp_wealth + event_credit

    return pd.DataFrame({
        "t"                  : np.arange(n_steps + 1),
        "S"                  : S,
        "P"                  : P,
        "shareholder_wealth" : shareholder_wealth,
        "standard_perp_wealth": standard_perp_wealth,
        "event_credit_wealth": event_credit_wealth,
    })


def verify_dividend_loss(
    results_df: pd.DataFrame,
    D: float,
    tau: int | None = None,
    tol: float = 1e-5,
) -> Dict[str, object]:
    """
    Assert that the standard perp trails the shareholder by exactly D post-tau,
    and that the event-credit perp matches the shareholder.

    Parameters
    ----------
    results_df : output of simulate_dividend_paths.
    D          : discrete dividend amount (must match the value used in simulation).
    tau        : ex-dividend step index; if None, inferred as the first step where
                 event_credit_wealth exceeds standard_perp_wealth.
    tol        : numerical tolerance for the equality checks.  The default 1e-5
                 accounts for floating-point accumulation over O(100) steps.

    Returns
    -------
    Dict with:
      'standard_deficit'      : array of (shareholder - standard_perp) values post-tau.
      'event_credit_error'    : array of (event_credit - shareholder) values post-tau.
      'standard_deficit_ok'   : bool, all deficits ≈ D.
      'event_credit_matches'  : bool, all errors ≈ 0.

    Raises
    ------
    AssertionError if either check fails.
    """
    df = results_df
    if tau is None:
        gap = df["event_credit_wealth"] - df["standard_perp_wealth"]
        tau = int(df.loc[gap > 0, "t"].iloc[0])
    # Identify post-dividend rows
    post_tau = df["t"] >= tau

    standard_deficit  = (df.loc[post_tau, "shareholder_wealth"]
                         - df.loc[post_tau, "standard_perp_wealth"]).values
    event_credit_error = (df.loc[post_tau, "event_credit_wealth"]
                          - df.loc[post_tau, "shareholder_wealth"]).values

    std_ok = bool(np.all(np.abs(standard_deficit - D) < tol))
    ec_ok  = bool(np.all(np.abs(event_credit_error) < tol))

    assert std_ok, (
        f"Standard perp deficit mismatch: max |deficit - D| = "
        f"{np.max(np.abs(standard_deficit - D)):.2e} > tol={tol}"
    )
    assert ec_ok, (
        f"Event-credit perp mismatch: max |error| = "
        f"{np.max(np.abs(event_credit_error)):.2e} > tol={tol}"
    )

    return {
        "standard_deficit"    : standard_deficit,
        "event_credit_error"  : event_credit_error,
        "standard_deficit_ok" : std_ok,
        "event_credit_matches": ec_ok,
        "D"                   : D,
    }


if __name__ == "__main__":
    _tau = 36
    df = simulate_dividend_paths(D_frac=0.01, tau=_tau, n_steps=72, seed=42)
    # D = gap between shareholder and standard perp immediately at tau
    D_est = float((df.loc[df["t"] == _tau, "shareholder_wealth"]
                   - df.loc[df["t"] == _tau, "standard_perp_wealth"]).iloc[0])
    print(f"Discrete dividend D ≈ {D_est:.6f}")
    print(f"Post-tau shareholder wealth (final): {df['shareholder_wealth'].iloc[-1]:.4f}")
    print(f"Post-tau standard perp  (final):     {df['standard_perp_wealth'].iloc[-1]:.4f}")
    print(f"Post-tau event-credit   (final):     {df['event_credit_wealth'].iloc[-1]:.4f}")
    chk = verify_dividend_loss(df, D=D_est, tau=_tau)
    print(f"Standard deficit OK:   {chk['standard_deficit_ok']}")
    print(f"Event-credit matches:  {chk['event_credit_matches']}")
