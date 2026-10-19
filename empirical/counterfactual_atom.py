"""
Counterfactual fair-atom insertion — Experiment 4.

Tests the hypothesis that inserting a fair discrete atom
phi_atom = -D / (Mark - D) at the ex-dividend index collapses the
straddling residual block to near zero, while leaving the non-straddling
block unchanged.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from empirical.residuals import (
    build_residual_matrix,
    compute_straddling_stats,
)
from data.dividends import compute_expected_atom

logger = logging.getLogger(__name__)


def insert_synthetic_atom(
    funding_rates: np.ndarray,
    timestamps: np.ndarray,
    event_idx: int,
    D: float,
    mark_at_event: float,
) -> np.ndarray:
    """Insert a synthetic fair dividend atom into the funding rate series.

    The atom value is:

        phi_atom = -D / (mark_at_event - D)

    and is *added* to whatever funding rate is already recorded at
    ``event_idx``.  A copy of ``funding_rates`` is returned so the
    original array is not mutated.

    Parameters
    ----------
    funding_rates : np.ndarray of float
        Realized hourly funding rates, length N.
    timestamps : np.ndarray of int
        Unix millisecond timestamps, length N (used only for bounds checking).
    event_idx : int
        Index within ``funding_rates`` at which the atom is inserted.
    D : float
        Cash dividend per share (USD).
    mark_at_event : float
        Perpetual mark price immediately before the ex-date (USD).

    Returns
    -------
    np.ndarray of float
        Modified funding rate array with atom added at ``event_idx``.

    Raises
    ------
    IndexError
        If ``event_idx`` is out of bounds.
    ValueError
        If ``mark_at_event <= D`` (degenerate denominator).
    """
    n = len(funding_rates)
    if not (0 <= event_idx < n):
        raise IndexError(
            f"event_idx ({event_idx}) out of range for length {n}."
        )
    phi_atom = compute_expected_atom(D, mark_at_event)
    modified = funding_rates.copy().astype(float)
    modified[event_idx] += phi_atom
    logger.debug(
        "insert_synthetic_atom: event_idx=%d, phi_atom=%.6f, "
        "rate before=%.6f, rate after=%.6f",
        event_idx,
        phi_atom,
        float(funding_rates[event_idx]),
        float(modified[event_idx]),
    )
    return modified


def compute_counterfactual_residual_matrix(
    timestamps: np.ndarray,
    P: np.ndarray,
    S: np.ndarray,
    funding_rates: np.ndarray,
    event_idx: int,
    D: float,
    A_jumps: Optional[dict] = None,
    r: float = 0.0,
) -> np.ndarray:
    """Compute the residual matrix after inserting the synthetic fair atom.

    Convenience wrapper that calls :func:`insert_synthetic_atom` and then
    :func:`~empirical.residuals.build_residual_matrix`.

    Parameters
    ----------
    timestamps : np.ndarray of int
        Unix millisecond timestamps.
    P : np.ndarray of float
        Mark prices.
    S : np.ndarray of float
        Index prices.
    funding_rates : np.ndarray of float
        Realized funding rates *before* atom insertion.
    event_idx : int
        Ex-dividend event index.
    D : float
        Cash dividend (USD).
    A_jumps : dict, optional
        Discrete dividend jump map ``{timestamp_idx: D}``.
    r : float
        Financing rate per period.

    Returns
    -------
    np.ndarray, shape (N, N)
        Counterfactual residual matrix R_after.
    """
    mark_at_event = float(P[event_idx])
    funding_cf = insert_synthetic_atom(
        funding_rates, timestamps, event_idx, D, mark_at_event
    )
    R_after, _, _ = build_residual_matrix(
        timestamps=timestamps,
        P=P,
        S=S,
        funding_rates=funding_cf,
        event_idx=event_idx,
        A_jumps=A_jumps,
        r=r,
    )
    return R_after


def block_collapse_analysis(
    R_before: np.ndarray,
    R_after: np.ndarray,
    is_straddling: np.ndarray,
) -> dict:
    """Quantify the collapse of the straddling residual block after atom insertion.

    Compares summary statistics of the straddling and non-straddling cells
    between the original (before) and counterfactual (after) matrices.

    Parameters
    ----------
    R_before : np.ndarray, shape (N, N)
        Pairwise residual matrix before atom insertion.
    R_after : np.ndarray, shape (N, N)
        Pairwise residual matrix after counterfactual atom insertion.
    is_straddling : np.ndarray of bool, shape (N, N)
        Straddling mask.

    Returns
    -------
    dict
        Keys:
          * before_straddling_mean   – mean straddling residual before
          * after_straddling_mean    – mean straddling residual after
          * reduction                – before_mean - after_mean (positive = collapse)
          * reduction_pct            – reduction as % of |before_mean|
          * before_non_strad_mean    – mean non-straddling residual before
          * after_non_strad_mean     – mean non-straddling residual after
          * before_straddling_std    – std of straddling block before
          * after_straddling_std     – std of straddling block after
    """
    stats_before = compute_straddling_stats(R_before, is_straddling)
    stats_after = compute_straddling_stats(R_after, is_straddling)

    b_mean = stats_before['straddling_mean']
    a_mean = stats_after['straddling_mean']
    reduction = b_mean - a_mean

    if abs(b_mean) > 1e-12:
        reduction_pct = 100.0 * reduction / abs(b_mean)
    else:
        reduction_pct = float('nan')

    return {
        'before_straddling_mean': b_mean,
        'after_straddling_mean': a_mean,
        'reduction': reduction,
        'reduction_pct': reduction_pct,
        'before_non_strad_mean': stats_before['non_straddling_mean'],
        'after_non_strad_mean': stats_after['non_straddling_mean'],
        'before_straddling_std': stats_before['straddling_std'],
        'after_straddling_std': stats_after['straddling_std'],
    }
