"""
Discrete residual computation — Experiment 4.

Implements the discrete analogue of the paper's no-arbitrage residual
equation (2) for perpetual funding contracts:

    R(t1, t2) = [A_t2 - A_t1]               # cumulative dividend leg
              - [e_t2 - e_t1]               # basis change
              + sum_{s in (t1,t2]} phi_s     # funding leg
              - sum_{s in (t1,t2]} r_s       # financing leg (approx 0)

where e_t = P_t - S_t is the absolute basis and A_t is the cumulative
dividend-adjustment process.

Under the fair-atom condition R ≈ 0 for all intervals; systematic
non-zero values in the straddling block reveal the missing atom.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core residual computation
# ---------------------------------------------------------------------------

def compute_residual(
    t1_idx: int,
    t2_idx: int,
    timestamps: np.ndarray,
    P: np.ndarray,
    S: np.ndarray,
    funding_rates: np.ndarray,
    r: float = 0.0,
    A_jumps: Optional[dict] = None,
) -> dict:
    """Compute the discrete no-arbitrage residual over interval (t1, t2].

    R(t1, t2) = (A_t2 - A_t1) - (e_t2 - e_t1) + sum_{funding} - sum_{financing}

    where e_t = P_t - S_t and A_t accumulates any discrete dividend jumps
    that fall strictly inside (t1, t2].

    Parameters
    ----------
    t1_idx : int
        Index of the entry timestamp in ``timestamps``.
    t2_idx : int
        Index of the exit timestamp in ``timestamps``.
    timestamps : np.ndarray of int
        Unix millisecond timestamps (length N).
    P : np.ndarray of float
        Mark prices aligned to ``timestamps``.
    S : np.ndarray of float
        Index / reference prices aligned to ``timestamps``.
    funding_rates : np.ndarray of float
        Realized hourly funding rates aligned to ``timestamps``.
        funding_rates[i] is the rate settled *at* timestamps[i].
    r : float
        Constant financing rate per period (set to 0.0 when unavailable;
        document this approximation in any published result).
    A_jumps : dict, optional
        Mapping ``{timestamp_idx: D_value}`` for discrete dividend events.
        A_jumps[k] = D means a cash dividend of D is paid at index k.
        If None no discrete dividends are applied.

    Returns
    -------
    dict
        Keys:
          * R_total          – total residual (float)
          * dividend_leg     – cumulative dividend adjustment A_t2 - A_t1
          * basis_change_leg – basis change e_t2 - e_t1
          * funding_leg      – cumulative funding sum_{phi}
          * financing_leg    – cumulative financing sum_{r}
    """
    if t2_idx <= t1_idx:
        raise ValueError(f"t2_idx ({t2_idx}) must be > t1_idx ({t1_idx}).")
    n = len(timestamps)
    if not (0 <= t1_idx < n and 0 <= t2_idx < n):
        raise IndexError(f"Indices ({t1_idx}, {t2_idx}) out of range for length {n}.")

    # Basis at endpoints
    e_t1 = float(P[t1_idx] - S[t1_idx])
    e_t2 = float(P[t2_idx] - S[t2_idx])
    basis_change_leg = e_t2 - e_t1

    # Funding and financing legs — sum over (t1, t2] exclusive of t1
    indices = range(t1_idx + 1, t2_idx + 1)
    funding_leg = float(np.sum(funding_rates[list(indices)]))
    financing_leg = r * len(indices)

    # Dividend leg — cumulative dividends with timestamps strictly inside (t1, t2]
    dividend_leg = 0.0
    if A_jumps:
        for k, d_val in A_jumps.items():
            if t1_idx < k <= t2_idx:
                dividend_leg += float(d_val)

    R_total = dividend_leg - basis_change_leg + funding_leg - financing_leg

    return {
        'R_total': R_total,
        'dividend_leg': dividend_leg,
        'basis_change_leg': basis_change_leg,
        'funding_leg': funding_leg,
        'financing_leg': financing_leg,
    }


# ---------------------------------------------------------------------------
# Matrix builders
# ---------------------------------------------------------------------------

def build_residual_matrix(
    timestamps: np.ndarray,
    P: np.ndarray,
    S: np.ndarray,
    funding_rates: np.ndarray,
    event_idx: int,
    A_jumps: Optional[dict] = None,
    r: float = 0.0,
) -> tuple:
    """Compute the N×N pairwise residual matrix in an event window.

    For each ordered pair (i, j) with i < j the matrix cell R[i, j]
    stores R(i, j) as defined in :func:`compute_residual`.  The upper
    triangle (i ≥ j) is filled with NaN.

    Parameters
    ----------
    timestamps : np.ndarray of int
        Unix millisecond timestamps of length N (event window).
    P : np.ndarray of float
        Mark prices, length N.
    S : np.ndarray of float
        Index prices, length N.
    funding_rates : np.ndarray of float
        Realized funding rates, length N.
    event_idx : int
        Index of the ex-dividend event within the window (0-based).
    A_jumps : dict, optional
        Discrete dividend jump map ``{timestamp_idx: D}``.
    r : float
        Financing rate per period.

    Returns
    -------
    tuple of (R_matrix, is_straddling_matrix, timestamps)
        * R_matrix           – np.ndarray shape (N, N), float
        * is_straddling_matrix – np.ndarray shape (N, N), bool
        * timestamps         – the input timestamps array (pass-through)
    """
    N = len(timestamps)
    R_matrix = np.full((N, N), np.nan)
    is_straddling_matrix = np.zeros((N, N), dtype=bool)

    for i in range(N):
        for j in range(i + 1, N):
            res = compute_residual(
                i, j,
                timestamps=timestamps,
                P=P,
                S=S,
                funding_rates=funding_rates,
                r=r,
                A_jumps=A_jumps,
            )
            R_matrix[i, j] = res['R_total']
            is_straddling_matrix[i, j] = classify_straddling(i, j, event_idx)

    return R_matrix, is_straddling_matrix, timestamps


# ---------------------------------------------------------------------------
# Straddling classification
# ---------------------------------------------------------------------------

def classify_straddling(
    t1_idx: int,
    t2_idx: int,
    event_idx: int,
) -> bool:
    """Return True if interval (t1_idx, t2_idx] strictly straddles event_idx.

    An interval straddles the event when:
        t1_idx < event_idx <= t2_idx

    i.e. the position was entered before the event and exited on or after it.

    Parameters
    ----------
    t1_idx : int
        Entry index.
    t2_idx : int
        Exit index.
    event_idx : int
        Index of the dividend event.

    Returns
    -------
    bool
    """
    return t1_idx < event_idx <= t2_idx


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def compute_straddling_stats(
    R_matrix: np.ndarray,
    is_straddling: np.ndarray,
) -> dict:
    """Summarize residuals split by straddling vs non-straddling cells.

    Parameters
    ----------
    R_matrix : np.ndarray, shape (N, N)
        Pairwise residual matrix (NaN for i ≥ j).
    is_straddling : np.ndarray of bool, shape (N, N)
        Straddling mask as returned by :func:`build_residual_matrix`.

    Returns
    -------
    dict
        Keys: straddling_mean, non_straddling_mean, difference,
        straddling_std, non_straddling_std, n_straddling, n_non_straddling.
    """
    valid_mask = ~np.isnan(R_matrix)
    strad_vals = R_matrix[valid_mask & is_straddling]
    non_strad_vals = R_matrix[valid_mask & ~is_straddling]

    def _safe_mean(arr: np.ndarray) -> float:
        return float(np.mean(arr)) if len(arr) > 0 else float('nan')

    def _safe_std(arr: np.ndarray) -> float:
        return float(np.std(arr, ddof=1)) if len(arr) > 1 else float('nan')

    strad_mean = _safe_mean(strad_vals)
    non_strad_mean = _safe_mean(non_strad_vals)

    return {
        'straddling_mean': strad_mean,
        'non_straddling_mean': non_strad_mean,
        'difference': strad_mean - non_strad_mean,
        'straddling_std': _safe_std(strad_vals),
        'non_straddling_std': _safe_std(non_strad_vals),
        'n_straddling': int(len(strad_vals)),
        'n_non_straddling': int(len(non_strad_vals)),
    }
