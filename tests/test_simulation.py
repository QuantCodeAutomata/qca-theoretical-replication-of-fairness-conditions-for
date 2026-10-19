"""Pytest tests for simulation/ modules.

Covers:
- Correct DataFrame shapes and column names from simulate_dividend_paths
- Standard perp trails shareholder by exactly D post ex-dividend
- Event-credit perp matches the physical shareholder path
- Simulated AR(1) coefficient is recoverable (≈ 0.72 ± 0.10)
- Half-life formula: ln(2) / lambda ≈ 2.1 hours for phi = 0.72
- Deployed funding rule has positive slope in basis b
- Fair funding rule has negative slope in b when lambda > 0 and rho = 0
- Cumulative funding is monotonically increasing for constant positive rate
"""

import numpy as np
import pandas as pd
import pytest
import sys

sys.path.insert(0, '/workspace/project')

from simulation.dividend_paths import simulate_dividend_paths, verify_dividend_loss
from simulation.basis_ou import simulate_basis_ou, estimate_ar1_persistence, compute_halflife
from simulation.funding_rules import (
    deployed_funding_rule,
    fair_funding_rule,
    cumulative_funding,
)


# ---------------------------------------------------------------------------
# Dividend path simulations
# ---------------------------------------------------------------------------

def test_dividend_path_shapes():
    """simulate_dividend_paths returns correct DataFrame shape and column names."""
    n_steps = 72
    df = simulate_dividend_paths(n_steps=n_steps, seed=0)

    expected_cols = {
        "t", "S", "P",
        "shareholder_wealth", "standard_perp_wealth", "event_credit_wealth",
    }
    assert set(df.columns) >= expected_cols, (
        f"Missing columns: {expected_cols - set(df.columns)}"
    )
    assert len(df) == n_steps + 1, (
        f"Expected {n_steps + 1} rows (0..{n_steps}), got {len(df)}"
    )
    assert df["t"].iloc[0] == 0 and df["t"].iloc[-1] == n_steps


def test_standard_perp_underperforms_by_dividend():
    """Standard perp wealth trails shareholder wealth by exactly D after tau."""
    tau, n_steps, D_frac = 36, 72, 0.01
    df = simulate_dividend_paths(tau=tau, n_steps=n_steps, D_frac=D_frac, seed=5)

    # Recover D from the simulation output: at any t >= tau,
    # event_credit_wealth - standard_perp_wealth == D exactly.
    D_exact = (df.loc[df["t"] == tau, "event_credit_wealth"].values[0]
               - df.loc[df["t"] == tau, "standard_perp_wealth"].values[0])

    post_tau = df["t"] >= tau
    deficit = (
        df.loc[post_tau, "shareholder_wealth"].values
        - df.loc[post_tau, "standard_perp_wealth"].values
    )
    assert np.allclose(deficit, D_exact, atol=1e-5), (
        f"Deficit should be ≈ D = {D_exact:.4f}; "
        f"got range [{deficit.min():.4e}, {deficit.max():.4e}]"
    )


def test_event_credit_matches_shareholder():
    """Event-credit perp matches the physical shareholder path up to 1e-5."""
    tau, n_steps, D_frac = 36, 72, 0.01
    df = simulate_dividend_paths(tau=tau, n_steps=n_steps, D_frac=D_frac, seed=9)

    # Read D directly from the output: event_credit - standard_perp == D at tau.
    D_exact = (df.loc[df["t"] == tau, "event_credit_wealth"].values[0]
               - df.loc[df["t"] == tau, "standard_perp_wealth"].values[0])

    result = verify_dividend_loss(df, D=D_exact, tau=tau)
    assert result["event_credit_matches"], (
        f"Event-credit perp does not match shareholder: "
        f"max error = {np.max(np.abs(result['event_credit_error'])):.2e}"
    )


# ---------------------------------------------------------------------------
# OU basis simulation
# ---------------------------------------------------------------------------

def test_basis_ou_persistence():
    """Simulated AR(1) coefficient is recoverable from the basis series."""
    phi_target = 0.72
    df = simulate_basis_ou(n_steps=20_000, phi_ar1=phi_target, sigma_b=0.0006, seed=99)
    phi_est = estimate_ar1_persistence(df["b"].values)
    assert abs(phi_est - phi_target) < 0.10, (
        f"AR(1) persistence estimate {phi_est:.4f} differs from target "
        f"{phi_target} by more than 0.10."
    )


def test_halflife_computation():
    """Half-life formula: -ln(2) / ln(phi) ≈ 2.1 hours for phi = 0.72."""
    phi = 0.72
    hl = compute_halflife(phi)
    hl_expected = -np.log(2.0) / np.log(phi)
    assert abs(hl - hl_expected) < 1e-10, (
        f"Half-life mismatch: got {hl:.6f}, expected {hl_expected:.6f}"
    )
    # Approximate check: ln(2) / (-ln(0.72)) ≈ 2.1 hours
    assert abs(hl - 2.1) < 0.5, (
        f"Half-life {hl:.3f} h is far from the expected ~2.1 h for phi=0.72"
    )


# ---------------------------------------------------------------------------
# Funding rules
# ---------------------------------------------------------------------------

def test_deployed_funding_positive_slope():
    """df_deploy/db = kappa > 0: deployed rule increases with premium."""
    r, q, kappa = 0.05 / 8760, 0.02 / 8760, 0.05
    b_lo = np.array([-0.02, -0.01, 0.0])
    b_hi = np.array([ 0.01,  0.02, 0.03])
    f_lo = deployed_funding_rule(b_lo, r=r, q=q, kappa=kappa)
    f_hi = deployed_funding_rule(b_hi, r=r, q=q, kappa=kappa)
    assert np.all(f_hi > f_lo), (
        "Deployed funding should be higher for higher basis (positive slope)."
    )


def test_fair_funding_negative_slope():
    """df_fair/db < 0 when lambda > 0 and rho = 0: fair rule decreases with premium."""
    r, q, lam = 0.05 / 8760, 0.02 / 8760, 0.329
    b_lo = np.array([-0.02, -0.01, 0.0])
    b_hi = np.array([ 0.01,  0.02, 0.03])
    f_lo = fair_funding_rule(b_lo, r=r, q=q, lam=lam, rho=0.0)
    f_hi = fair_funding_rule(b_hi, r=r, q=q, lam=lam, rho=0.0)
    assert np.all(f_hi < f_lo), (
        "Fair funding should be lower for higher basis (negative slope) when lambda > 0, rho=0."
    )


def test_cumulative_funding_monotone():
    """Cumulative funding is strictly increasing when rate and price are positive."""
    T = 50
    rate = np.full(T, 0.0001)        # constant positive rate
    P    = np.full(T, 100.0)         # constant mark price
    cum  = cumulative_funding(rate, P, dt=1.0)

    assert len(cum) == T + 1, f"Expected length {T + 1}, got {len(cum)}"
    assert cum[0] == 0.0, "First element must be zero."
    diffs = np.diff(cum)
    assert np.all(diffs > 0), (
        f"Cumulative funding should be strictly increasing; "
        f"min diff = {diffs.min():.4e}"
    )
