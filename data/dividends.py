"""
Synthetic dividend calendar and atom-size helpers.

All mark prices used in `compute_expected_atom` and `compute_event_size_bps`
are approximate placeholder values inferred from public sources at the time of
writing (2025-Q2). They are NOT queried from any exchange in real-time. Any
caller that has access to a live or historical mark price should pass it
explicitly rather than relying on the defaults embedded here.

Reference: paper Table 1 column "expected_atom_approx".
"""

from __future__ import annotations

import pandas as pd

# ---------------------------------------------------------------------------
# Canonical dividend calendar used throughout the experiments
# ---------------------------------------------------------------------------
# Fields
#   symbol            : equity ticker
#   ex_date           : ex-dividend date (YYYY-MM-DD)
#   D                 : cash dividend per share (USD)
#   expected_atom_approx : phi_expected = -D / (Mark - D) evaluated at the
#                          approximate mark price stated in the paper; provided
#                          for quick cross-checks only — do not use as ground truth
DIVIDEND_CALENDAR: list[dict] = [
    {
        'symbol': 'AAPL',
        'ex_date': '2026-05-11',
        'D': 0.27,
        'expected_atom_approx': -0.00092,
    },
    {
        'symbol': 'MSFT',
        'ex_date': '2026-05-21',
        'D': 0.91,
        'expected_atom_approx': -0.00218,
    },
    {
        'symbol': 'NVDA',
        'ex_date': '2026-06-04',
        'D': 0.25,
        'expected_atom_approx': -0.00117,
    },
    {
        'symbol': 'GOOGL',
        'ex_date': '2026-06-08',
        'D': 0.22,
        'expected_atom_approx': -0.00060,
    },
    {
        'symbol': 'META',
        'ex_date': '2026-06-15',
        'D': 0.525,
        'expected_atom_approx': -0.00091,
    },
]


def get_dividend_calendar() -> pd.DataFrame:
    """Return the paper's dividend calendar as a tidy DataFrame.

    Returns
    -------
    pd.DataFrame
        Columns: symbol (str), ex_date (datetime64[ns]), D (float64),
        expected_atom_approx (float64).
    """
    df = pd.DataFrame(DIVIDEND_CALENDAR)
    df['ex_date'] = pd.to_datetime(df['ex_date'])
    return df


def compute_expected_atom(D: float, mark: float) -> float:
    """Compute the fair discrete funding atom for an ex-dividend event.

    The atom is the single discrete funding payment phi_atom that makes the
    total-return basis exactly zero across the ex-date:

        phi_atom = -D / (Mark - D)

    Parameters
    ----------
    D : float
        Cash dividend per share (USD).
    mark : float
        Perpetual mark price immediately *before* the ex-dividend date (USD).
        NOTE: caller is responsible for supplying an accurate mark price;
        the values in DIVIDEND_CALENDAR are approximations only.

    Returns
    -------
    float
        Expected funding atom (dimensionless fraction, typically negative).

    Raises
    ------
    ValueError
        If mark <= D (degenerate / zero-denominator case).
    """
    if mark <= D:
        raise ValueError(
            f"mark ({mark}) must be strictly greater than D ({D}) to avoid "
            "a zero or negative denominator."
        )
    return -D / (mark - D)


def compute_event_size_bps(D: float, mark: float) -> float:
    """Compute the ex-dividend event size in basis points relative to mark.

    The raw event size is:

        size = D / Mark  (dimensionless)

    Returned in basis points (×10,000).

    Parameters
    ----------
    D : float
        Cash dividend per share (USD).
    mark : float
        Perpetual mark price immediately before the ex-dividend date (USD).
        NOTE: placeholder values will produce approximate results only.

    Returns
    -------
    float
        Event size in basis points.

    Raises
    ------
    ValueError
        If mark <= 0.
    """
    if mark <= 0:
        raise ValueError(f"mark ({mark}) must be positive.")
    return (D / mark) * 10_000
