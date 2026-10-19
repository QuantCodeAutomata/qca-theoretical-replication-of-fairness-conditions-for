"""
Equity perpetual universe discovery.

Provides the paper's canonical 13-ticker Hyperliquid equity perpetual list,
a live-discovery function that queries the venue API, and a helper to build
the descriptive panel universe DataFrame.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from data.download import get_hyperliquid_perp_universe

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Static universe (paper's Table 2 / panel section)
# ---------------------------------------------------------------------------

# The 13 Hyperliquid equity perpetuals used in the panel analysis.
# Tickers match Hyperliquid's native coin names as of the paper's data window.
HYPERLIQUID_EQUITY_PERPS: list[str] = [
    'AAPL',
    'MSFT',
    'NVDA',
    'GOOGL',
    'META',
    'AMZN',
    'TSLA',
    'NFLX',
    'AMD',
    'INTC',
    'ORCL',
    'BABA',
    'TSM',
]

# Approximate listing dates on Hyperliquid (year-month; exact day unknown).
# These are best-effort estimates from public forum announcements.
_HL_APPROX_LISTING: dict[str, str] = {
    'AAPL': '2023-10',
    'MSFT': '2023-10',
    'NVDA': '2023-10',
    'GOOGL': '2023-11',
    'META': '2023-11',
    'AMZN': '2023-11',
    'TSLA': '2023-10',
    'NFLX': '2024-01',
    'AMD': '2024-01',
    'INTC': '2024-02',
    'ORCL': '2024-02',
    'BABA': '2024-03',
    'TSM': '2024-03',
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def discover_equity_perps(venue: str = 'hyperliquid') -> pd.DataFrame:
    """Query a venue's API to enumerate available equity perpetuals.

    Equity perps are identified by matching the coin name against known
    equity ticker patterns (all-caps alphabetical, length 1–5, not a
    recognised crypto ticker).

    Parameters
    ----------
    venue : str
        Venue to query; currently only ``'hyperliquid'`` is supported.

    Returns
    -------
    pd.DataFrame
        Columns: coin (str), sz_decimals (int or NaN), max_leverage (float or NaN),
        is_equity_perp (bool).
        Falls back to the static HYPERLIQUID_EQUITY_PERPS list if the API
        call fails.
    """
    if venue != 'hyperliquid':
        logger.warning("discover_equity_perps: venue '%s' not supported; returning empty.", venue)
        return pd.DataFrame(columns=['coin', 'sz_decimals', 'max_leverage', 'is_equity_perp'])

    # Known crypto tickers to exclude from equity detection
    _CRYPTO_TICKERS = frozenset({
        'BTC', 'ETH', 'SOL', 'BNB', 'DOGE', 'XRP', 'ADA', 'AVAX',
        'LINK', 'DOT', 'UNI', 'LTC', 'MATIC', 'SHIB', 'FTM', 'ARB',
        'OP', 'INJ', 'SUI', 'APT', 'SEI', 'TIA', 'PYTH', 'JTO',
        'WIF', 'BONK', 'JUP', 'STRK', 'DYM', 'MANTA', 'ALT', 'PIXEL',
        'PORTAL', 'PENDLE', 'W', 'ETHFI', 'ENA', 'EIGEN', 'IO', 'ZK',
        'LISTA', 'ZRO', 'RENDER', 'ONDO', 'AAVE', 'MKR', 'CRV', 'SNX',
        'IMX', 'LDO', 'RPL', 'FXS', 'COMP', 'SUSHI', 'BLUR', 'ENS',
        '1INCH', 'SAND', 'MANA', 'AXS', 'GALA', 'ILV', 'GMT', 'STEPN',
        'KAVA', 'NEAR', 'ALGO', 'VET', 'HBAR', 'FLOW', 'ICP', 'ATOM',
        'LUNA', 'LUNC', 'USTC', 'CELR', 'SKL', 'ANKR', 'ONE', 'ZIL',
        'WAVES', 'ROSE', 'KSM', 'RUNE', 'THETA', 'FTT', 'DYDX',
        'PERP', 'GMX', 'GNS', 'RDNT', 'VELA', 'HOOK',
    })

    universe = get_hyperliquid_perp_universe()

    if not universe:
        logger.warning(
            "discover_equity_perps: API returned empty; using static list."
        )
        rows = [
            {
                'coin': c,
                'sz_decimals': float('nan'),
                'max_leverage': float('nan'),
                'is_equity_perp': True,
            }
            for c in HYPERLIQUID_EQUITY_PERPS
        ]
        return pd.DataFrame(rows)

    rows = []
    for entry in universe:
        coin = entry.get('name', '')
        # Heuristic: equity tickers are 1–5 uppercase alpha chars not in crypto list
        is_equity = (
            coin.isalpha()
            and coin.isupper()
            and 1 <= len(coin) <= 5
            and coin not in _CRYPTO_TICKERS
        )
        rows.append({
            'coin': coin,
            'sz_decimals': entry.get('szDecimals', float('nan')),
            'max_leverage': entry.get('maxLeverage', float('nan')),
            'is_equity_perp': is_equity,
        })

    return pd.DataFrame(rows)


def build_panel_universe(venue: str = 'hyperliquid') -> pd.DataFrame:
    """Build a descriptive panel universe with listing dates and availability.

    Merges the discovered or static equity perpetual list with known
    approximate listing dates.

    Parameters
    ----------
    venue : str
        Venue to build the universe for.

    Returns
    -------
    pd.DataFrame
        Columns: coin (str), venue (str), listing_month (str),
        listing_date_approx (datetime64[ns]),
        sz_decimals (float), max_leverage (float),
        in_paper_universe (bool).
        Rows are sorted by listing_date_approx then coin.
    """
    discovered = discover_equity_perps(venue)
    equity_df = discovered[discovered['is_equity_perp']].copy()

    paper_set = set(HYPERLIQUID_EQUITY_PERPS)
    equity_df['venue'] = venue
    equity_df['in_paper_universe'] = equity_df['coin'].isin(paper_set)
    equity_df['listing_month'] = equity_df['coin'].map(
        lambda c: _HL_APPROX_LISTING.get(c, 'unknown')
    )
    equity_df['listing_date_approx'] = pd.to_datetime(
        equity_df['listing_month'].apply(
            lambda m: m + '-01' if m != 'unknown' else pd.NaT
        ),
        errors='coerce',
    )

    cols = [
        'coin', 'venue', 'listing_month', 'listing_date_approx',
        'sz_decimals', 'max_leverage', 'in_paper_universe',
    ]
    # Ensure all columns exist
    for c in cols:
        if c not in equity_df.columns:
            equity_df[c] = float('nan')

    return (
        equity_df[cols]
        .sort_values(['listing_date_approx', 'coin'])
        .reset_index(drop=True)
    )
