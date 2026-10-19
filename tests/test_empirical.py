"""Pytest tests for empirical/ and data/ modules.

Covers:
- Expected atom formula: phi_atom = -D / (Mark - D)
- Atom match classification by 5 bp tolerance
- Residual decomposition: R = dividend_leg + basis_change_leg + funding - financing
- Straddling/non-straddling classification of intervals
- Synthetic atom insertion at the correct index
- Block-collapse direction: straddling mean moves toward zero after atom
- Event size in basis points: D / Mark * 10000
"""

import numpy as np
import pytest
import sys

sys.path.insert(0, '/workspace/project')

from data.dividends import compute_expected_atom, compute_event_size_bps
from empirical.atom_detection import validate_atom_match
from empirical.residuals import compute_residual, classify_straddling, build_residual_matrix
from empirical.counterfactual_atom import insert_synthetic_atom, block_collapse_analysis


# ---------------------------------------------------------------------------
# Expected atom formula
# ---------------------------------------------------------------------------

def test_expected_atom_formula():
    """phi_atom = -D / (Mark - D); check at D=0.27 over several mark prices."""
    D = 0.27
    for mark in [10.0, 50.0, 100.0, 200.0]:
        expected = -D / (mark - D)
        computed = compute_expected_atom(D=D, mark=mark)
        assert abs(computed - expected) < 1e-14, (
            f"Atom formula error at mark={mark}: computed={computed}, expected={expected}"
        )


def test_expected_atom_degenerate_raises():
    """mark <= D must raise ValueError (degenerate denominator)."""
    with pytest.raises(ValueError):
        compute_expected_atom(D=5.0, mark=4.0)
    with pytest.raises(ValueError):
        compute_expected_atom(D=1.0, mark=1.0)


# ---------------------------------------------------------------------------
# Atom match threshold
# ---------------------------------------------------------------------------

def test_atom_match_threshold_passes():
    """|observed - expected| < 5e-6 classifies as a match."""
    expected = compute_expected_atom(D=0.27, mark=100.0)
    # Perturb by well under 5e-6
    observed = expected + 1e-7
    assert validate_atom_match(expected, observed, tolerance=5e-6)


def test_atom_match_threshold_fails():
    """|observed - expected| >= 5e-6 should NOT classify as a match."""
    expected = compute_expected_atom(D=0.27, mark=100.0)
    observed = expected + 1e-5   # 10 bp – clearly over threshold
    assert not validate_atom_match(expected, observed, tolerance=5e-6)


# ---------------------------------------------------------------------------
# Residual decomposition
# ---------------------------------------------------------------------------

def _build_simple_window(N: int = 10, D: float = 0.5, event_idx: int = 5):
    """Helper: synthetic price/funding arrays for a short event window."""
    timestamps = np.arange(N, dtype=float) * 3_600_000     # hourly ms timestamps
    S = np.linspace(100.0, 102.0, N)
    P = S + 0.01 * np.sin(np.arange(N))                    # tiny basis
    funding_rates = np.full(N, 0.0001)                      # constant 1 bp/hr
    A_jumps = {event_idx: D}
    return timestamps, P, S, funding_rates, A_jumps


def test_residual_decomposition():
    """R = dividend_leg + basis_change_leg + funding_leg - financing_leg."""
    timestamps, P, S, funding_rates, A_jumps = _build_simple_window()
    for (t1, t2) in [(0, 5), (2, 8), (0, 9)]:
        res = compute_residual(
            t1_idx=t1, t2_idx=t2,
            timestamps=timestamps, P=P, S=S,
            funding_rates=funding_rates, r=0.0, A_jumps=A_jumps,
        )
        reconstructed = (
            res['dividend_leg']
            - res['basis_change_leg']
            + res['funding_leg']
            - res['financing_leg']
        )
        assert abs(res['R_total'] - reconstructed) < 1e-12, (
            f"R_total ({res['R_total']:.8e}) != reconstructed sum "
            f"({reconstructed:.8e}) for interval ({t1}, {t2}]"
        )


# ---------------------------------------------------------------------------
# Straddling classification
# ---------------------------------------------------------------------------

def test_straddling_classification():
    """classify_straddling returns True iff t1_idx < event_idx <= t2_idx."""
    event = 5

    # True cases: interval strictly straddles the event
    assert classify_straddling(0, 5, event)   # t1<event<=t2
    assert classify_straddling(4, 6, event)
    assert classify_straddling(3, 5, event)

    # False cases
    assert not classify_straddling(5, 9, event)   # t1 == event (not strictly before)
    assert not classify_straddling(0, 4, event)   # t2 < event
    assert not classify_straddling(6, 9, event)   # entirely after


# ---------------------------------------------------------------------------
# Synthetic atom insertion
# ---------------------------------------------------------------------------

def test_synthetic_atom_insertion():
    """insert_synthetic_atom adds phi_atom = -D/(Mark-D) at event_idx only."""
    N, event_idx, D = 12, 6, 0.5
    mark = 100.0
    funding_rates = np.zeros(N)
    timestamps    = np.arange(N, dtype=float)

    modified = insert_synthetic_atom(
        funding_rates=funding_rates,
        timestamps=timestamps,
        event_idx=event_idx,
        D=D,
        mark_at_event=mark,
    )
    phi_atom_expected = compute_expected_atom(D=D, mark=mark)

    # Original array unchanged
    assert np.all(funding_rates == 0.0), "Original array must not be mutated."

    # Atom placed at event_idx
    assert abs(modified[event_idx] - phi_atom_expected) < 1e-14, (
        f"Atom at event_idx: got {modified[event_idx]:.8e}, "
        f"expected {phi_atom_expected:.8e}"
    )

    # All other indices unchanged
    mask = np.ones(N, dtype=bool)
    mask[event_idx] = False
    assert np.all(modified[mask] == 0.0), "Non-event indices must not be modified."


# ---------------------------------------------------------------------------
# Block collapse direction
# ---------------------------------------------------------------------------

def test_block_collapse_direction():
    """After atom insertion, mean straddling residual moves closer to zero."""
    N, event_idx, D = 10, 5, 0.5
    timestamps, P, S, funding_rates, A_jumps = _build_simple_window(
        N=N, D=D, event_idx=event_idx
    )

    R_before, is_straddling, _ = build_residual_matrix(
        timestamps=timestamps, P=P, S=S,
        funding_rates=funding_rates, event_idx=event_idx,
        A_jumps=A_jumps, r=0.0,
    )

    mark_at_event = float(P[event_idx])
    funding_cf = insert_synthetic_atom(
        funding_rates, timestamps, event_idx, D, mark_at_event
    )
    R_after, _, _ = build_residual_matrix(
        timestamps=timestamps, P=P, S=S,
        funding_rates=funding_cf, event_idx=event_idx,
        A_jumps=A_jumps, r=0.0,
    )

    result = block_collapse_analysis(R_before, R_after, is_straddling)

    # The atom insertion should reduce the absolute straddling mean
    abs_before = abs(result['before_straddling_mean'])
    abs_after  = abs(result['after_straddling_mean'])
    assert abs_after <= abs_before + 1e-10, (
        f"|straddling mean| did not decrease after atom insertion: "
        f"before={abs_before:.4e}, after={abs_after:.4e}"
    )


# ---------------------------------------------------------------------------
# Event size in basis points
# ---------------------------------------------------------------------------

def test_compute_event_size_bps():
    """Event size = D / Mark * 10_000 (basis points)."""
    test_cases = [
        (0.27, 100.0, 0.27 / 100.0 * 10_000),
        (1.0,  50.0,  1.0  / 50.0  * 10_000),
        (0.5,  200.0, 0.5  / 200.0 * 10_000),
    ]
    for D, mark, expected_bps in test_cases:
        bps = compute_event_size_bps(D=D, mark=mark)
        assert abs(bps - expected_bps) < 1e-10, (
            f"Event size mismatch at D={D}, mark={mark}: "
            f"got {bps:.6f} bps, expected {expected_bps:.6f} bps"
        )
