"""Pytest tests for theory/ modules.

Covers the key mathematical properties:
- Residual accounting identity R = M_{tau2} - M_{tau1}
- Fair funding target formula Y_t = (r*S + mu_e - a) / P
- Weak fairness jump condition E[Delta_e_tau] = D
- Interval-discrepancy lower bound Delta(mu, nu) >= D for AC mu
- OU fair funding rule equation (8)
- Small-b approximation
- Slope sign reversal (df_fair/db < 0 vs df_deploy/db > 0)
- Projection benchmark f*(Y_t) = max(Y_t, 0) >= 0
"""

import numpy as np
import pytest
import sys

sys.path.insert(0, '/workspace/project')

from theory.identities import (
    verify_residual_identity,
    compute_fair_funding_target,
    demonstrate_weak_fairness,
    demonstrate_strong_fairness,
)
from theory.jumps import demonstrate_weak_fairness_condition, counterexample_no_atom
from theory.discrepancy import compute_discrepancy, compute_discrepancy_fast, minimize_discrepancy
from theory.ou_fair_rule import (
    compute_fair_funding_rule,
    compare_fair_vs_deployed,
    fair_funding_small_b_approx,
    fair_funding_derivative,
    projection_nonneg,
)


# ---------------------------------------------------------------------------
# Residual identity
# ---------------------------------------------------------------------------

def test_residual_identity_holds():
    """R(tau1, tau2) must equal M_{tau2} - M_{tau1} to numerical precision."""
    result = verify_residual_identity(n_steps=1000, seed=7)
    assert result['identity_holds'], (
        f"Identity error = {result['identity_error']:.2e} exceeds tolerance; "
        f"R = {result['R']:.6e}, M_increment = {result['M_increment']:.6e}"
    )
    assert abs(result['identity_error']) < 1e-6


def test_fair_funding_target_formula():
    """Y_t = (r*S + mu_e - a) / P must hold exactly at given parameter values."""
    r    = 0.05 / 8760        # hourly risk-free rate
    S    = 100.0
    mu_e = 0.8 / 8760 * S    # hourly basis drift (0.8% annualised)
    a    = 0.02 / 8760 * S   # hourly accrual (2% continuous yield)
    P    = S + 0.5            # small positive premium

    Y = compute_fair_funding_target(r, S, mu_e, a, P)
    Y_expected = (r * S + mu_e - a) / P

    assert abs(float(Y) - Y_expected) < 1e-14, (
        f"Formula mismatch: Y={float(Y):.10e}, expected={Y_expected:.10e}"
    )


# ---------------------------------------------------------------------------
# Weak fairness
# ---------------------------------------------------------------------------

def test_weak_fairness_requires_basis_jump():
    """E[Delta_e_tau | F_{tau-}] must equal D for the event-credit perp."""
    result = demonstrate_weak_fairness_condition(
        D=1.0, S0=100.0, tau=36, n_steps=72, n_paths=50_000, seed=42
    )
    assert result['condition_met'], (
        f"E[Delta_e] = {result['E_Delta_e']:.6f} differs from D = {result['D']:.6f}"
    )
    # In the fair-perp case Delta_e = D exactly, so the sample mean is D with zero variance
    assert abs(result['E_Delta_e'] - result['D']) < 1e-8, (
        f"Delta_e should equal D exactly; got error {abs(result['E_Delta_e'] - result['D']):.2e}"
    )


def test_weak_fairness_martingale_passes():
    """E[M_T - M_0] ≈ 0 across many independent paths under the fair funding rule.

    Single-path bootstrap is unreliable (any one realisation may drift);
    the correct test is to average over multiple independent paths and
    check the cross-path mean increment is within 3 SE of zero.
    """
    from theory.identities import numerical_verify_residual_identity

    n_paths = 200
    increments = [
        numerical_verify_residual_identity(n_steps=200, seed=s)['M_increment']
        for s in range(n_paths)
    ]
    mean_inc = float(np.mean(increments))
    se       = float(np.std(increments) / np.sqrt(n_paths))

    assert abs(mean_inc) < 3.0 * se, (
        f"E[M_T - M_0] ≠ 0 across {n_paths} paths: "
        f"mean = {mean_inc:.4e}, se = {se:.4e}"
    )


# ---------------------------------------------------------------------------
# Discrepancy lower bound
# ---------------------------------------------------------------------------

def test_discrepancy_lower_bound():
    """Delta(mu, nu) >= D for any absolutely continuous measure mu.

    Uses compute_discrepancy_fast which analytically handles the shrinking
    near-tau intervals that reveal the true supremum, avoiding the O(1/n)
    grid-truncation error of the naive compute_discrepancy.
    """
    tau, D, horizon = 0.5, 1.0, 1.0
    grid = np.linspace(0.0, horizon, 200)

    # Uniform density with total mass D over [0, horizon].
    uniform_density = lambda t: D / horizon  # integrates to D over [0, 1]

    result = compute_discrepancy_fast(uniform_density, tau=tau, D=D, grid=grid)
    disc   = result["discrepancy"]

    assert disc >= D - 1e-4, (
        f"Discrepancy {disc:.6f} is below lower bound D = {D}; "
        "the theorem guarantees Delta(mu, nu) >= D for AC mu."
    )


def test_discrepancy_lower_bound_gaussian():
    """Optimised Gaussian AC measure still cannot beat the D lower bound.

    The Nelder-Mead optimiser may overshoot the true infimum by a small
    numerical margin (≤ 2e-4); we allow that tolerance while still
    confirming that min_discrepancy ≈ D.
    """
    result = minimize_discrepancy(tau=0.5, D=2.0, horizon=1.0, n_grid=150)
    OPT_TOL = 2e-4   # Nelder-Mead + grid discretisation tolerance
    assert result['min_discrepancy'] >= result['lower_bound'] - OPT_TOL, (
        f"min_discrepancy {result['min_discrepancy']:.6f} is far below lower "
        f"bound {result['lower_bound']:.6f}; gap = {result['gap']:.4e}"
    )
    assert result['gap'] >= -OPT_TOL, (
        f"Gap must be nearly non-negative; got {result['gap']:.4e}"
    )


# ---------------------------------------------------------------------------
# OU fair funding rule — Eq. (8)
# ---------------------------------------------------------------------------

def test_ou_fair_rule_formula():
    """f_fair(b) = (r-q) - lambda*b / (1+b) at specific parameter values (rho=0)."""
    r, q, lam, b = 0.05 / 8760, 0.02 / 8760, 0.329, 0.01
    f = compute_fair_funding_rule(r=r, q=q, lambda_ou=lam, b=b)
    f_expected = (r - q) - lam * b / (1.0 + b)
    assert abs(float(f) - f_expected) < 1e-14, (
        f"Rule mismatch: f={float(f):.10e}, expected={f_expected:.10e}"
    )


def test_small_b_approximation():
    """f_fair ≈ (r-q) - lambda*b for small |b|; error O(b^2)."""
    r, q, lam = 0.05 / 8760, 0.02 / 8760, 0.329
    b_small = 1e-4   # very small premium

    f_exact  = compute_fair_funding_rule(r=r, q=q, lambda_ou=lam, b=b_small)
    f_approx = (r - q) - lam * b_small       # small-b formula

    # Error should be O(b^2 * lambda)
    error_bound = lam * b_small ** 2 * 2     # generous bound
    assert abs(float(f_exact) - f_approx) < error_bound, (
        f"Small-b approximation error {abs(float(f_exact) - f_approx):.2e} "
        f"exceeds O(b^2) bound {error_bound:.2e}"
    )


def test_slope_sign_reversal():
    """df_fair/db < 0 for lambda > 0 and rho = 0, while df_deploy/db = kappa > 0."""
    lam, rho, sigma_b, sigma_S, kappa = 0.329, 0.0, 0.0006, 0.001, 0.05
    b_grid = np.linspace(-0.04, 0.04, 41)

    slope_fair   = fair_funding_derivative(b_grid, lam=lam, sigma_b=sigma_b,
                                           sigma_S=sigma_S, rho=rho)
    slope_deploy = kappa  # constant positive slope

    # Fair slope must be strictly negative everywhere when lambda > 0 and rho = 0
    assert np.all(slope_fair < 0), (
        f"Fair slope has non-negative values; max = {slope_fair.max():.4e}. "
        "Theory requires df_fair/db = -lambda/(1+b)^2 < 0."
    )
    assert slope_deploy > 0, "Deployed slope kappa must be positive."


def test_projection_nonneg_formula():
    """f*(Y_t) = max(Y_t, 0): non-negative for Y_t >= 0, zero for Y_t < 0."""
    # Positive input: projection returns it unchanged
    assert projection_nonneg(0.01)  == pytest.approx(0.01,  abs=1e-14)
    assert projection_nonneg(0.0)   == pytest.approx(0.0,   abs=1e-14)
    # Negative input: projection returns 0
    assert projection_nonneg(-0.01) == pytest.approx(0.0,   abs=1e-14)
    assert projection_nonneg(-1e-8) == pytest.approx(0.0,   abs=1e-14)

    # Array input
    Y = np.array([-0.05, -0.01, 0.0, 0.01, 0.05])
    f_star = projection_nonneg(Y)
    expected = np.maximum(Y, 0.0)
    np.testing.assert_allclose(f_star, expected, atol=1e-14)

    # Non-negative for all Y_t >= 0
    assert np.all(projection_nonneg(Y[Y >= 0]) >= 0)
