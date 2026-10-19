"""Pytest tests for panel/ modules.

Covers:
- Basis computation: e_t = P_t - S_t elementwise
- Relative basis range: b_t stays within [-0.1, 0.1] for typical equity perps
- AR(1) half-life formula: -ln(2)/ln(phi) for phi=0.72 ≈ 2.1 hours
- AR(1) lambda formula: -ln(0.72) ≈ 0.329 per hour
- AR(1) estimation on synthetic data with a known phi
- Panel AR(1) fitting returns one row per symbol in a DataFrame
- Venue slope classification: Hyperliquid, Binance, OKX all use positive slope
"""

import numpy as np
import pandas as pd
import pytest
import sys

sys.path.insert(0, '/workspace/project')

from panel.premium_series import compute_basis, compute_relative_basis
from panel.ar1_estimation import (
    compute_halflife,
    compute_lambda_per_hour,
    estimate_ar1,
    fit_panel_ar1,
)
from panel.funding_rule_docs import classify_slope_sign
from simulation.basis_ou import simulate_basis_ou


# ---------------------------------------------------------------------------
# Basis computation
# ---------------------------------------------------------------------------

def test_basis_computation():
    """e_t = P_t - S_t exactly, element-wise."""
    S = np.array([100.0, 100.5, 99.8, 101.2, 100.0])
    P = np.array([100.1, 100.7, 99.9, 101.3, 100.1])
    e = compute_basis(P, S)

    np.testing.assert_allclose(e, P - S, atol=1e-14,
        err_msg="compute_basis(P, S) must equal P - S element-wise.")
    assert e.shape == S.shape


def test_relative_basis_range():
    """b_t = (P_t - S_t) / S_t and the simulated series stays within [-0.1, 0.1]."""
    # Formula check
    S = np.array([100.0, 200.0, 50.0])
    P = S + np.array([0.5, -1.0, 0.1])
    b = compute_relative_basis(P, S)
    np.testing.assert_allclose(b, (P - S) / S, atol=1e-14,
        err_msg="compute_relative_basis must equal (P - S) / S.")

    # Range check on simulated OU basis (sigma_b=0.0006 ensures small basis)
    df = simulate_basis_ou(n_steps=5_000, phi_ar1=0.72, sigma_b=0.0006, seed=17)
    b_sim = compute_relative_basis(
        P=df['S'].values * (1.0 + df['b'].values),
        S=df['S'].values,
    )
    assert b_sim.min() >= -0.10, (
        f"Simulated b_t min {b_sim.min():.4f} is below -0.10."
    )
    assert b_sim.max() <= 0.10, (
        f"Simulated b_t max {b_sim.max():.4f} is above 0.10."
    )


# ---------------------------------------------------------------------------
# AR(1) half-life and lambda formulae
# ---------------------------------------------------------------------------

def test_ar1_halflife_formula():
    """-ln(2)/ln(phi) == 2.1 hours (±0.5 h) for phi = 0.72; formula exact."""
    phi = 0.72
    hl = compute_halflife(phi)
    hl_formula = -np.log(2.0) / np.log(phi)

    assert abs(hl - hl_formula) < 1e-10, (
        f"compute_halflife({phi}) = {hl:.6f} differs from formula {hl_formula:.6f}"
    )
    assert abs(hl - 2.1) < 0.5, (
        f"Half-life {hl:.3f} h is far from ~2.1 h for phi=0.72"
    )


def test_ar1_lambda_formula():
    """-ln(0.72) ≈ 0.329 per hour; formula exact."""
    phi = 0.72
    lam = compute_lambda_per_hour(phi)
    lam_formula = -np.log(phi)

    assert abs(lam - lam_formula) < 1e-10, (
        f"compute_lambda_per_hour({phi}) = {lam:.6f} differs from formula {lam_formula:.6f}"
    )
    assert abs(lam - 0.329) < 0.005, (
        f"Lambda {lam:.4f} /h is not close to 0.329 /h for phi=0.72"
    )


# ---------------------------------------------------------------------------
# AR(1) estimation on synthetic data
# ---------------------------------------------------------------------------

def test_ar1_estimation_synthetic():
    """estimate_ar1 recovers the known phi from a long synthetic AR(1) series."""
    phi_true = 0.72
    df = simulate_basis_ou(n_steps=15_000, phi_ar1=phi_true, sigma_b=0.0006, seed=314)
    result = estimate_ar1(df['b'].values, include_intercept=True)

    assert not np.isnan(result['phi']), "estimate_ar1 returned NaN phi."
    assert abs(result['phi'] - phi_true) < 0.05, (
        f"Estimated phi {result['phi']:.4f} deviates from true phi {phi_true} by more than 0.05."
    )
    # Derived quantities must be consistent
    assert abs(result['halflife_hours'] - compute_halflife(result['phi'])) < 1e-8
    assert abs(result['lambda_per_hour'] - compute_lambda_per_hour(result['phi'])) < 1e-8
    assert 0.0 <= result['r_squared'] <= 1.0
    assert result['nobs'] > 0


def test_ar1_estimation_returns_expected_keys():
    """estimate_ar1 result dict contains all required keys."""
    df = simulate_basis_ou(n_steps=2_000, phi_ar1=0.72, sigma_b=0.0006, seed=1)
    result = estimate_ar1(df['b'].values)
    required_keys = {
        'alpha', 'phi', 'halflife_hours', 'lambda_per_hour',
        'r_squared', 'nobs', 'std_err_phi', 't_stat_phi', 'p_value_phi',
    }
    assert required_keys <= set(result.keys()), (
        f"Missing keys: {required_keys - set(result.keys())}"
    )


# ---------------------------------------------------------------------------
# Panel AR(1) fitting
# ---------------------------------------------------------------------------

def test_fit_panel_ar1_shape():
    """fit_panel_ar1 returns one row per symbol with required columns."""
    symbols = ['AAPL', 'MSFT', 'NVDA']
    panel = {
        sym: simulate_basis_ou(n_steps=5_000, phi_ar1=0.72, sigma_b=0.0006,
                               seed=hash(sym) % 99999)['b'].values
        for sym in symbols
    }
    df = fit_panel_ar1(panel)

    assert len(df) == len(symbols), (
        f"Expected {len(symbols)} rows, got {len(df)}"
    )
    assert 'symbol' in df.columns
    assert 'phi' in df.columns
    assert 'halflife_hours' in df.columns
    assert set(df['symbol'].tolist()) == set(symbols)


# ---------------------------------------------------------------------------
# Venue slope classification
# ---------------------------------------------------------------------------

def test_classify_slope_sign():
    """Hyperliquid, Binance, and OKX all use positive slope (premium-index rule)."""
    venues = ['hyperliquid', 'binance', 'okx']
    for v in venues:
        sign = classify_slope_sign(v, b=0.01)
        assert sign == 'positive', (
            f"Venue '{v}' should classify as positive slope but got '{sign}'."
        )


def test_classify_slope_sign_zero_basis():
    """classify_slope_sign returns 'zero' when b == 0."""
    for v in ['hyperliquid', 'binance', 'okx']:
        assert classify_slope_sign(v, b=0.0) == 'zero', (
            f"Venue '{v}' at b=0 should return 'zero'."
        )


def test_classify_slope_sign_unknown_venue():
    """classify_slope_sign raises KeyError for unknown venue."""
    with pytest.raises(KeyError):
        classify_slope_sign('unknown_venue_xyz', b=0.01)
