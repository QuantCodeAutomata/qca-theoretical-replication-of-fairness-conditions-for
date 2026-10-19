"""
Venue funding rule documentation and symbol-level summary table generation.

Documents the deployed funding rate formulas for each covered venue, as
sourced from publicly available exchange documentation.  Provides helper
functions to classify the sign of the funding rule's sensitivity to the
perpetual basis.

Key fact: all three covered venues (Binance, Hyperliquid, OKX) use a
*positive-slope* premium-index mechanism, meaning a positive basis (P > S)
generates positive funding (longs pay shorts).  Only Binance additionally
posts a discrete negative atom at equity ex-dividend dates.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Venue funding rule registry
# ---------------------------------------------------------------------------

VENUE_FUNDING_RULES: dict[str, dict] = {
    'hyperliquid': {
        'form': 'premium_index_plus_interest',
        'formula': 'f = clamp((P - S)/S, -0.05, 0.05) + interest_rate',
        'slope_on_premium': 'positive',
        'settlement_frequency': '1h',
        'clamp_lower': -0.0500,
        'clamp_upper':  0.0500,
        'interest_rate_basis': 0.0,   # typically zero for equity perps
        'discrete_dividend_atom': False,
        'description': (
            'Hyperliquid uses a premium index equal to the relative basis '
            '(P-S)/S, clamped symmetrically at ±5 %, plus an optional '
            'interest-rate component (typically zero for equity perpetuals). '
            'The slope on the premium is strictly positive: positive basis '
            'generates positive funding, mean-reverting the basis toward zero.'
        ),
        'source_url': 'https://hyperliquid.gitbook.io/hyperliquid-docs/trading/funding',
    },
    'binance': {
        'form': 'premium_index_plus_interest',
        'formula': (
            'f = P_avg + clamp(interest_rate - P_avg, -0.0075, 0.0075); '
            'plus discrete dividend atom phi_D = -D/(Mark-D) at ex-date'
        ),
        'slope_on_premium': 'positive',
        'settlement_frequency': '8h',
        'clamp_lower': -0.0075,
        'clamp_upper':  0.0075,
        'interest_rate_basis': 0.0001,  # ~0.01% per 8-hour period
        'discrete_dividend_atom': True,
        'description': (
            'Binance USDT-M futures use a mark/index premium average (P_avg) '
            'with a clamped interest-rate adjustment (±0.75 % per 8 h). '
            'The slope on the premium is positive. '
            'Uniquely, Binance also posts a one-off discrete funding atom '
            'phi_D = -D/(Mark-D) at the equity ex-dividend date to prevent '
            'basis disruption from the index drop.'
        ),
        'source_url': 'https://www.binance.com/en/support/faq/what-is-funding-rate-360033525031',
    },
    'okx': {
        'form': 'premium_index_plus_interest',
        'formula': 'f = clamp(Premium Index, -0.005, 0.005) + interest_rate',
        'slope_on_premium': 'positive',
        'settlement_frequency': '8h',
        'clamp_lower': -0.0050,
        'clamp_upper':  0.0050,
        'interest_rate_basis': 0.0001,
        'discrete_dividend_atom': False,
        'description': (
            'OKX uses a premium index (relative basis clamped at ±0.5 % per '
            '8 h) plus a fixed interest-rate component. The slope on the '
            'premium is positive. No discrete dividend atom is applied at '
            'equity ex-dividend dates.'
        ),
        'source_url': 'https://www.okx.com/help/how-is-the-funding-fee-calculated',
    },
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_funding_rule(venue: str) -> dict:
    """Return the documented funding rule for a venue.

    Parameters
    ----------
    venue : str
        Venue name: ``'hyperliquid'``, ``'binance'``, or ``'okx'``.

    Returns
    -------
    dict
        Full funding rule metadata dict from :data:`VENUE_FUNDING_RULES`.

    Raises
    ------
    KeyError
        If the venue is not in the registry.
    """
    if venue not in VENUE_FUNDING_RULES:
        raise KeyError(
            f"Venue '{venue}' not found in VENUE_FUNDING_RULES. "
            f"Available venues: {sorted(VENUE_FUNDING_RULES)}"
        )
    return VENUE_FUNDING_RULES[venue]


def classify_slope_sign(venue: str, b: float) -> str:
    """Return the slope classification of the funding rule at basis b.

    All three covered venues apply a premium-index rule with a positive
    slope, meaning the direction of funding is the same as the sign of the
    basis everywhere within the clamp band.

    Parameters
    ----------
    venue : str
        Venue name.
    b : float
        Current relative basis (dimensionless fraction).

    Returns
    -------
    str
        ``'positive'`` — funding moves in the same direction as the basis.
        ``'negative'`` — funding moves opposite to the basis.
        ``'zero'``     — basis is exactly zero (degenerate case).

    Notes
    -----
    The clamp band boundary is not explicitly handled here; even at the
    boundary the incremental slope is non-negative (positive within the
    band, zero at and beyond the boundary).  We return ``'positive'`` for
    all non-zero basis values for all registered venues.
    """
    rule = get_funding_rule(venue)
    if b == 0.0:
        return 'zero'
    slope = rule.get('slope_on_premium', 'positive')
    return str(slope)


def generate_symbol_summary_table(
    ar1_results: pd.DataFrame,
    universe: pd.DataFrame,
) -> pd.DataFrame:
    """Produce a symbol-level summary combining universe and AR(1) metadata.

    Parameters
    ----------
    ar1_results : pd.DataFrame
        Output of :func:`~panel.ar1_estimation.fit_panel_ar1`.
        Must contain at minimum: symbol, phi, halflife_hours, r_squared.
    universe : pd.DataFrame
        Output of :func:`~panel.universe.build_panel_universe`.
        Must contain at minimum: coin, venue, listing_date_approx.

    Returns
    -------
    pd.DataFrame
        Columns: symbol, venue, listing_start, sample_len_obs, avg_premium,
        ar1_phi, halflife_hours, r_squared, funding_rule_form,
        slope_sign, discrete_dividend_atom.
        Rows sorted by halflife_hours ascending.
    """
    # Rename universe 'coin' to 'symbol' for join
    uni = universe.rename(columns={'coin': 'symbol'}).copy()

    if ar1_results.empty:
        logger.warning("generate_symbol_summary_table: ar1_results is empty.")
        merged = uni.copy()
        merged['ar1_phi'] = float('nan')
        merged['halflife_hours'] = float('nan')
        merged['r_squared'] = float('nan')
        merged['sample_len_obs'] = 0
    else:
        merged = uni.merge(
            ar1_results[['symbol', 'phi', 'halflife_hours', 'r_squared', 'nobs']].rename(
                columns={'phi': 'ar1_phi', 'nobs': 'sample_len_obs'}
            ),
            on='symbol',
            how='left',
        )

    # Attach funding rule metadata
    def _rule_form(venue: str) -> str:
        try:
            return VENUE_FUNDING_RULES[venue]['form']
        except KeyError:
            return 'unknown'

    def _slope_sign(venue: str) -> str:
        try:
            return VENUE_FUNDING_RULES[venue]['slope_on_premium']
        except KeyError:
            return 'unknown'

    def _has_atom(venue: str) -> bool:
        try:
            return bool(VENUE_FUNDING_RULES[venue]['discrete_dividend_atom'])
        except KeyError:
            return False

    merged['funding_rule_form'] = merged['venue'].apply(_rule_form)
    merged['slope_sign'] = merged['venue'].apply(_slope_sign)
    merged['discrete_dividend_atom'] = merged['venue'].apply(_has_atom)
    merged['listing_start'] = merged.get('listing_date_approx', pd.NaT)
    merged['avg_premium'] = float('nan')  # populated externally from premium series

    cols = [
        'symbol', 'venue', 'listing_start', 'sample_len_obs', 'avg_premium',
        'ar1_phi', 'halflife_hours', 'r_squared',
        'funding_rule_form', 'slope_sign', 'discrete_dividend_atom',
    ]
    existing = [c for c in cols if c in merged.columns]
    result = merged[existing].copy()

    if 'halflife_hours' in result.columns:
        result = result.sort_values('halflife_hours').reset_index(drop=True)

    return result
