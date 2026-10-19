"""
Premium and basis series construction.

Builds hourly mark-minus-index (absolute) and relative-basis series for
each equity perpetual from venue-native candle data.

Terminology
-----------
* P   – perpetual mark price
* S   – spot / index reference price
* e_t – absolute basis:  e_t = P_t - S_t
* b_t – relative basis:  b_t = (P_t - S_t) / S_t
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from data.download import fetch_hyperliquid_candles

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Elementary basis computations
# ---------------------------------------------------------------------------

def compute_basis(P: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Compute the absolute basis e_t = P_t - S_t.

    Parameters
    ----------
    P : np.ndarray of float
        Mark (perpetual) prices.
    S : np.ndarray of float
        Index / reference prices, same length as P.

    Returns
    -------
    np.ndarray of float
        Element-wise absolute basis array.
    """
    P = np.asarray(P, dtype=float)
    S = np.asarray(S, dtype=float)
    return P - S


def compute_relative_basis(P: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Compute the relative basis b_t = (P_t - S_t) / S_t.

    Division by zero (S_t = 0) produces NaN, not an exception.

    Parameters
    ----------
    P : np.ndarray of float
        Mark prices.
    S : np.ndarray of float
        Index prices, same length as P.

    Returns
    -------
    np.ndarray of float
        Element-wise relative basis (dimensionless fraction).
    """
    P = np.asarray(P, dtype=float)
    S = np.asarray(S, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        b = np.where(S != 0, (P - S) / S, np.nan)
    return b


# ---------------------------------------------------------------------------
# Series builders
# ---------------------------------------------------------------------------

def aggregate_to_hourly(
    df: pd.DataFrame,
    price_col: str = 'close',
) -> pd.DataFrame:
    """Resample a DataFrame to hourly observations using end-of-hour convention.

    Expects a datetime-indexed DataFrame.  The ``price_col`` column is
    aggregated by taking the last available value within each hour
    (end-of-hour snapshot).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with a DatetimeTzAware or naive index.
    price_col : str
        Column to aggregate.  Defaults to ``'close'``.

    Returns
    -------
    pd.DataFrame
        Resampled DataFrame with hourly DatetimeIndex (UTC, end-of-hour labels)
        and the same ``price_col`` column containing end-of-hour values.
    """
    if df.empty:
        return df

    work = df.copy()
    if not isinstance(work.index, pd.DatetimeIndex):
        raise TypeError("aggregate_to_hourly: DataFrame must have a DatetimeIndex.")

    # Ensure UTC
    if work.index.tz is None:
        work.index = work.index.tz_localize('UTC')
    else:
        work.index = work.index.tz_convert('UTC')

    hourly = work[[price_col]].resample('1h', label='right', closed='right').last()
    return hourly.dropna()


def build_premium_series(
    symbol: str,
    venue: str = 'hyperliquid',
    start_time: Optional[int] = None,
    end_time: Optional[int] = None,
    freq: str = '1h',
) -> pd.DataFrame:
    """Build hourly absolute and relative premium series for a symbol.

    Fetches venue-native candle data for the perpetual (P) and, where
    available, a spot/index feed (S).  For venues that do not provide a
    separate spot feed the mark price is used as both P and S, yielding a
    zero basis series — this is clearly documented in the returned
    DataFrame's ``data_source`` column.

    Currently supported venue: ``'hyperliquid'``.

    Parameters
    ----------
    symbol : str
        Venue-native asset name, e.g. ``'AAPL'``.
    venue : str
        Exchange identifier.
    start_time : int, optional
        Start of query window (Unix ms).  Defaults to 7 days ago.
    end_time : int, optional
        End of query window (Unix ms).  Defaults to now.
    freq : str
        Target frequency, currently only ``'1h'`` is supported.

    Returns
    -------
    pd.DataFrame
        Columns: timestamp (datetime64, UTC), P (float64), S (float64),
        e (float64, absolute basis), b (float64, relative basis),
        data_source (str).
        Returns an empty DataFrame with these columns on any error.
    """
    schema = ['timestamp', 'P', 'S', 'e', 'b', 'data_source']

    if venue == 'hyperliquid':
        candles = fetch_hyperliquid_candles(
            coin=symbol,
            resolution=freq,
            start_time=start_time,
            end_time=end_time,
        )
        if candles.empty:
            logger.warning(
                "build_premium_series(%s, %s): no candle data returned.", symbol, venue
            )
            return pd.DataFrame(columns=schema)

        # Hyperliquid mark price candles; no separate spot feed in same endpoint.
        # Use close price as P; spot approximation = same value (b=0 placeholder).
        P_arr = candles['close'].to_numpy(dtype=float)
        S_arr = P_arr.copy()  # placeholder — no independent spot feed
        e_arr = compute_basis(P_arr, S_arr)
        b_arr = compute_relative_basis(P_arr, S_arr)

        df = pd.DataFrame({
            'timestamp': candles['open_time'].values,
            'P': P_arr,
            'S': S_arr,
            'e': e_arr,
            'b': b_arr,
            'data_source': 'hyperliquid_mark_only',
        })
        return df[schema].reset_index(drop=True)

    logger.warning("build_premium_series: venue '%s' not supported.", venue)
    return pd.DataFrame(columns=schema)
