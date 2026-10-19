"""
Cross-venue comparison and synthetic example generation.

Provides a summary table contrasting atom presence and residual statistics
across venues, and a fully self-contained synthetic simulation for
demonstration when live exchange data is unavailable.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from data.dividends import compute_expected_atom
from empirical.residuals import (
    build_residual_matrix,
    compute_straddling_stats,
)
from empirical.counterfactual_atom import (
    compute_counterfactual_residual_matrix,
    block_collapse_analysis,
)

logger = logging.getLogger(__name__)


def compare_venues(
    venue_results: dict,
    D: float,
    mark: float,
) -> pd.DataFrame:
    """Produce a summary table contrasting atom presence and residual statistics.

    Parameters
    ----------
    venue_results : dict
        Mapping of ``{venue_name: result_dict}`` where each result_dict has:
          * ``atom_present``         – bool
          * ``phi_observed``         – float
          * ``phi_expected``         – float
          * ``stats``                – dict from
            :func:`~empirical.residuals.compute_straddling_stats`
          * ``counterfactual_stats`` – dict from
            :func:`~empirical.counterfactual_atom.block_collapse_analysis`
            (optional, may be absent if not computed)
    D : float
        Cash dividend (USD).
    mark : float
        Approximate perpetual mark price (USD) used for expected atom reference.

    Returns
    -------
    pd.DataFrame
        Columns: venue, atom_present, phi_expected, phi_observed,
        absolute_error, straddling_mean, non_straddling_mean,
        cf_straddling_mean, cf_reduction_pct.
    """
    phi_expected = compute_expected_atom(D, mark)
    rows = []
    for venue, res in venue_results.items():
        stats = res.get('stats', {})
        cf_stats = res.get('counterfactual_stats', {})
        phi_obs = res.get('phi_observed', float('nan'))
        rows.append({
            'venue': venue,
            'atom_present': res.get('atom_present', False),
            'phi_expected': phi_expected,
            'phi_observed': phi_obs,
            'absolute_error': abs(phi_obs - phi_expected),
            'straddling_mean': stats.get('straddling_mean', float('nan')),
            'non_straddling_mean': stats.get('non_straddling_mean', float('nan')),
            'cf_straddling_mean': cf_stats.get('after_straddling_mean', float('nan')),
            'cf_reduction_pct': cf_stats.get('reduction_pct', float('nan')),
        })
    return pd.DataFrame(rows)


def generate_synthetic_example(
    D: float = 5.0,
    S0: float = 230.0,
    n_steps: int = 48,
    tau: int = 24,
    sigma_b: float = 0.0006,
    phi_ar1: float = 0.72,
    r: float = 0.0,
) -> dict:
    """Generate a fully synthetic residual-heatmap example.

    Simulates a ±24 h event window of mark price, index price, and funding
    for a single equity perpetual, both without (standard venue) and with
    (Binance-style) the fair dividend atom.  Useful for demonstration and
    unit-testing when live exchange data is unavailable.

    Simulation model
    ----------------
    * Index price S_t:  geometric random walk,  S_t = S_{t-1} * exp(eps_S)
    * Basis b_t:        AR(1),  b_t = phi_ar1 * b_{t-1} + eps_b
    * Mark price P_t:   P_t = S_t * (1 + b_t)
    * Funding rate f_t: f_t = b_t + eps_f  (simplified premium-index rule)
    * Dividend event:   discrete jump -D in S at index tau; P unchanged on
                        ex-date (perpetual does not gap unless atom applied)
    * Fair atom:        phi_atom = -D / (P_{tau} - D) inserted at tau in the
                        funded version

    Parameters
    ----------
    D : float
        Cash dividend per share (USD).
    S0 : float
        Initial index / reference price (USD).
    n_steps : int
        Total number of hourly steps in the simulation window (default 48 = ±24h).
    tau : int
        Index of the ex-dividend event within the window (default 24 = midpoint).
    sigma_b : float
        Basis volatility per step.
    phi_ar1 : float
        AR(1) persistence coefficient for basis dynamics.
    r : float
        Financing rate per period (approximated as 0 for crypto).

    Returns
    -------
    dict
        Keys:
          * timestamps         – np.ndarray[int], Unix ms
          * P                  – mark prices
          * S                  – index prices (post-dividend drop at tau)
          * funding_no_atom    – funding rate series without fair atom
          * funding_with_atom  – funding rate series with fair atom at tau
          * R_matrix_no_atom   – N×N residual matrix, standard venue
          * R_matrix_with_atom – N×N residual matrix, Binance-style
          * is_straddling      – N×N bool straddling mask
          * stats_no_atom      – straddling stats without atom
          * stats_with_atom    – straddling stats with atom
          * collapse           – block_collapse_analysis result
          * event_idx          – tau
          * D                  – D
          * phi_atom           – inserted atom value
    """
    rng = np.random.default_rng(seed=42)

    # --- Index price: geometric walk with dividend drop at tau ---------------
    log_ret_S = rng.normal(0, 0.0003, n_steps)
    S = np.zeros(n_steps)
    S[0] = S0
    for i in range(1, n_steps):
        S[i] = S[i - 1] * np.exp(log_ret_S[i])
    # Discrete dividend drop: S falls by D at tau
    S[tau:] -= D

    # --- Basis AR(1) ---------------------------------------------------------
    b = np.zeros(n_steps)
    for i in range(1, n_steps):
        b[i] = phi_ar1 * b[i - 1] + rng.normal(0, sigma_b)

    # --- Mark price P = S*(1+b) ----------------------------------------------
    P = S * (1 + b)

    # --- Funding (simplified positive-slope rule) ----------------------------
    # phi_t ≈ b_t (premium-index contribution)
    noise_f = rng.normal(0, 0.0001, n_steps)
    funding_no_atom = b + noise_f

    # --- Fair atom value -----------------------------------------------------
    mark_before_event = float(P[tau])
    phi_atom = compute_expected_atom(D, mark_before_event)
    funding_with_atom = funding_no_atom.copy()
    funding_with_atom[tau] += phi_atom

    # --- Timestamps: hourly starting from a round epoch ---------------------
    t0_ms = 1_748_649_600_000  # 2025-05-31 00:00 UTC in ms
    timestamps = np.array([t0_ms + i * 3_600_000 for i in range(n_steps)], dtype=np.int64)

    # --- Residual matrices ---------------------------------------------------
    R_no_atom, is_straddling, _ = build_residual_matrix(
        timestamps=timestamps,
        P=P,
        S=S,
        funding_rates=funding_no_atom,
        event_idx=tau,
        A_jumps=None,
        r=r,
    )
    R_with_atom, _, _ = build_residual_matrix(
        timestamps=timestamps,
        P=P,
        S=S,
        funding_rates=funding_with_atom,
        event_idx=tau,
        A_jumps=None,
        r=r,
    )

    stats_no_atom = compute_straddling_stats(R_no_atom, is_straddling)
    stats_with_atom = compute_straddling_stats(R_with_atom, is_straddling)
    collapse = block_collapse_analysis(R_no_atom, R_with_atom, is_straddling)

    return {
        'timestamps': timestamps,
        'P': P,
        'S': S,
        'funding_no_atom': funding_no_atom,
        'funding_with_atom': funding_with_atom,
        'R_matrix_no_atom': R_no_atom,
        'R_matrix_with_atom': R_with_atom,
        'is_straddling': is_straddling,
        'stats_no_atom': stats_no_atom,
        'stats_with_atom': stats_with_atom,
        'collapse': collapse,
        'event_idx': tau,
        'D': D,
        'phi_atom': phi_atom,
    }
