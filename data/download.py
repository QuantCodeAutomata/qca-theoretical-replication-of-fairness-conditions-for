"""
Exchange data download utilities — direct REST API calls only.

All HTTP calls use the `requests` library with a 30-second timeout.
No third-party exchange SDKs or Massive API wrappers are used.

Supported venues
----------------
* Binance USDT-M Futures  (https://fapi.binance.com)
* Hyperliquid             (https://api.hyperliquid.xyz/info)

Each public function returns an empty DataFrame with the documented schema
on any network error, HTTP error, or data-unavailable response so that
callers can handle missing data uniformly.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

_BINANCE_FAPI = "https://fapi.binance.com"
_HL_INFO = "https://api.hyperliquid.xyz/info"
_TIMEOUT = 30  # seconds


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _empty(columns: list[str]) -> pd.DataFrame:
    """Return an empty DataFrame with the given columns."""
    return pd.DataFrame(columns=columns)


def _ms_to_utc(ms_series: pd.Series) -> pd.Series:
    """Convert millisecond-epoch integers to UTC datetime."""
    return pd.to_datetime(ms_series, unit='ms', utc=True)


# ---------------------------------------------------------------------------
# Binance USDT-M Futures
# ---------------------------------------------------------------------------

def fetch_binance_funding_history(
    symbol: str,
    start_time: int,
    end_time: int,
    limit: int = 1000,
) -> pd.DataFrame:
    """Fetch Binance USDT-M futures funding rate history via REST API.

    Endpoint: GET https://fapi.binance.com/fapi/v1/fundingRate

    Parameters
    ----------
    symbol : str
        Binance futures symbol, e.g. ``'AAPLUSDT'``.
    start_time : int
        Start of the query window as Unix milliseconds.
    end_time : int
        End of the query window as Unix milliseconds.
    limit : int
        Maximum rows per request (Binance cap: 1 000).

    Returns
    -------
    pd.DataFrame
        Columns: symbol (str), funding_time (datetime64, UTC),
        funding_rate (float64), mark_price (float64).
        Empty DataFrame with these columns on any error.
    """
    schema = ['symbol', 'funding_time', 'funding_rate', 'mark_price']
    url = f"{_BINANCE_FAPI}/fapi/v1/fundingRate"
    params = {
        'symbol': symbol,
        'startTime': start_time,
        'endTime': end_time,
        'limit': limit,
    }
    try:
        resp = requests.get(url, params=params, timeout=_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            return _empty(schema)
        df = pd.DataFrame(data)
        df = df.rename(columns={
            'fundingTime': 'funding_time',
            'fundingRate': 'funding_rate',
            'markPrice': 'mark_price',
        })
        df['funding_time'] = _ms_to_utc(df['funding_time'].astype('int64'))
        df['funding_rate'] = df['funding_rate'].astype(float)
        df['mark_price'] = df['mark_price'].astype(float)
        return df[schema].reset_index(drop=True)
    except Exception as exc:
        logger.warning("fetch_binance_funding_history(%s): %s", symbol, exc)
        return _empty(schema)


def fetch_binance_mark_price(
    symbol: str,
    start_time: int,
    end_time: int,
    interval: str = '1h',
) -> pd.DataFrame:
    """Fetch Binance mark price klines for a USDT-M futures contract.

    Endpoint: GET https://fapi.binance.com/fapi/v1/markPriceKlines

    Each kline row represents: open_time, open, high, low, close, volume,
    close_time, quote_volume, n_trades, taker_buy_base, taker_buy_quote,
    ignore.  Only the close (end-of-bar mark price) is retained.

    Parameters
    ----------
    symbol : str
        Binance futures symbol, e.g. ``'AAPLUSDT'``.
    start_time : int
        Start of the query window as Unix milliseconds.
    end_time : int
        End of the query window as Unix milliseconds.
    interval : str
        Kline interval, e.g. ``'1h'``, ``'4h'``, ``'1d'``.

    Returns
    -------
    pd.DataFrame
        Columns: open_time (datetime64, UTC), close_time (datetime64, UTC),
        open (float64), high (float64), low (float64), close (float64).
        Empty DataFrame with these columns on any error.
    """
    schema = ['open_time', 'close_time', 'open', 'high', 'low', 'close']
    url = f"{_BINANCE_FAPI}/fapi/v1/markPriceKlines"
    params = {
        'symbol': symbol,
        'interval': interval,
        'startTime': start_time,
        'endTime': end_time,
        'limit': 1500,
    }
    try:
        resp = requests.get(url, params=params, timeout=_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            return _empty(schema)
        df = pd.DataFrame(data, columns=[
            'open_time', 'open', 'high', 'low', 'close',
            'volume', 'close_time', 'quote_volume', 'n_trades',
            'taker_buy_base', 'taker_buy_quote', 'ignore',
        ])
        df['open_time'] = _ms_to_utc(df['open_time'].astype('int64'))
        df['close_time'] = _ms_to_utc(df['close_time'].astype('int64'))
        for col in ('open', 'high', 'low', 'close'):
            df[col] = df[col].astype(float)
        return df[schema].reset_index(drop=True)
    except Exception as exc:
        logger.warning("fetch_binance_mark_price(%s): %s", symbol, exc)
        return _empty(schema)


def fetch_binance_index_price(
    pair: str,
    start_time: int,
    end_time: int,
    interval: str = '1h',
) -> pd.DataFrame:
    """Fetch Binance index price klines for a USDT-M futures pair.

    Endpoint: GET https://fapi.binance.com/fapi/v1/indexPriceKlines

    Parameters
    ----------
    pair : str
        Index pair name, e.g. ``'AAPLUSD'``.  Note: Binance index pairs for
        USDT-M contracts typically drop the ``'T'`` suffix (``'USD'`` not
        ``'USDT'``).
    start_time : int
        Start of the query window as Unix milliseconds.
    end_time : int
        End of the query window as Unix milliseconds.
    interval : str
        Kline interval, e.g. ``'1h'``.

    Returns
    -------
    pd.DataFrame
        Columns: open_time (datetime64, UTC), close_time (datetime64, UTC),
        open (float64), high (float64), low (float64), close (float64).
        Empty DataFrame with these columns on any error.
    """
    schema = ['open_time', 'close_time', 'open', 'high', 'low', 'close']
    url = f"{_BINANCE_FAPI}/fapi/v1/indexPriceKlines"
    params = {
        'pair': pair,
        'interval': interval,
        'startTime': start_time,
        'endTime': end_time,
        'limit': 1500,
    }
    try:
        resp = requests.get(url, params=params, timeout=_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            return _empty(schema)
        df = pd.DataFrame(data, columns=[
            'open_time', 'open', 'high', 'low', 'close',
            'volume', 'close_time', 'quote_volume', 'n_trades',
            'taker_buy_base', 'taker_buy_quote', 'ignore',
        ])
        df['open_time'] = _ms_to_utc(df['open_time'].astype('int64'))
        df['close_time'] = _ms_to_utc(df['close_time'].astype('int64'))
        for col in ('open', 'high', 'low', 'close'):
            df[col] = df[col].astype(float)
        return df[schema].reset_index(drop=True)
    except Exception as exc:
        logger.warning("fetch_binance_index_price(%s): %s", pair, exc)
        return _empty(schema)


# ---------------------------------------------------------------------------
# Hyperliquid
# ---------------------------------------------------------------------------

def fetch_hyperliquid_funding_history(
    coin: str,
    n_samples: int = 500,
) -> pd.DataFrame:
    """Fetch Hyperliquid perpetual funding rate history via REST API.

    Endpoint: POST https://api.hyperliquid.xyz/info
    Body: ``{"type": "fundingHistory", "coin": coin, "startTime": <ms>}``

    Hyperliquid funding is settled every hour.  The API returns up to
    ``n_samples`` rows starting from ``startTime``.  We back-calculate
    ``startTime`` as ``now - n_samples * 3_600_000`` ms.

    Parameters
    ----------
    coin : str
        Hyperliquid asset name, e.g. ``'AAPL'``, ``'BTC'``.
    n_samples : int
        Approximate number of hourly rows to request (controls ``startTime``).

    Returns
    -------
    pd.DataFrame
        Columns: coin (str), funding_time (datetime64, UTC),
        funding_rate (float64), premium (float64).
        Empty DataFrame with these columns on any error.
    """
    schema = ['coin', 'funding_time', 'funding_rate', 'premium']
    start_ms = int((time.time() - n_samples * 3600) * 1000)
    payload = {
        'type': 'fundingHistory',
        'coin': coin,
        'startTime': start_ms,
    }
    try:
        resp = requests.post(_HL_INFO, json=payload, timeout=_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            return _empty(schema)
        rows = []
        for entry in data:
            rows.append({
                'coin': entry.get('coin', coin),
                'funding_time': int(entry['time']),
                'funding_rate': float(entry['fundingRate']),
                'premium': float(entry.get('premium', float('nan'))),
            })
        df = pd.DataFrame(rows)
        df['funding_time'] = _ms_to_utc(df['funding_time'])
        return df[schema].reset_index(drop=True)
    except Exception as exc:
        logger.warning("fetch_hyperliquid_funding_history(%s): %s", coin, exc)
        return _empty(schema)


def fetch_hyperliquid_candles(
    coin: str,
    resolution: str = '1h',
    start_time: Optional[int] = None,
    end_time: Optional[int] = None,
) -> pd.DataFrame:
    """Fetch Hyperliquid candlestick (mark price) data via REST API.

    Endpoint: POST https://api.hyperliquid.xyz/info
    Body: ``{"type": "candleSnapshot", "req": {"coin": coin, "interval": resolution,
             "startTime": start_time, "endTime": end_time}}``

    Parameters
    ----------
    coin : str
        Hyperliquid asset name, e.g. ``'AAPL'``.
    resolution : str
        Candle interval string accepted by Hyperliquid, e.g. ``'1h'``, ``'4h'``.
    start_time : int, optional
        Start of query window as Unix milliseconds.  Defaults to 7 days ago.
    end_time : int, optional
        End of query window as Unix milliseconds.  Defaults to now.

    Returns
    -------
    pd.DataFrame
        Columns: open_time (datetime64, UTC), open (float64), high (float64),
        low (float64), close (float64), volume (float64).
        Empty DataFrame with these columns on any error.
    """
    schema = ['open_time', 'open', 'high', 'low', 'close', 'volume']
    now_ms = int(time.time() * 1000)
    if start_time is None:
        start_time = now_ms - 7 * 24 * 3600 * 1000
    if end_time is None:
        end_time = now_ms
    payload = {
        'type': 'candleSnapshot',
        'req': {
            'coin': coin,
            'interval': resolution,
            'startTime': start_time,
            'endTime': end_time,
        },
    }
    try:
        resp = requests.post(_HL_INFO, json=payload, timeout=_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            return _empty(schema)
        rows = []
        for c in data:
            rows.append({
                'open_time': int(c['t']),
                'open': float(c['o']),
                'high': float(c['h']),
                'low': float(c['l']),
                'close': float(c['c']),
                'volume': float(c.get('v', float('nan'))),
            })
        df = pd.DataFrame(rows)
        df['open_time'] = _ms_to_utc(df['open_time'])
        return df[schema].reset_index(drop=True)
    except Exception as exc:
        logger.warning("fetch_hyperliquid_candles(%s): %s", coin, exc)
        return _empty(schema)


def get_hyperliquid_perp_universe() -> list:
    """Return the list of all Hyperliquid perpetual market metadata dicts.

    Endpoint: POST https://api.hyperliquid.xyz/info
    Body: ``{"type": "meta"}``

    The returned list contains one dict per perpetual with at minimum the
    key ``'name'`` (the coin ticker) plus exchange-provided metadata such as
    ``szDecimals``, ``maxLeverage``, etc.

    Returns
    -------
    list[dict]
        List of universe metadata dicts.  Returns an empty list on any error.
    """
    try:
        resp = requests.post(
            _HL_INFO,
            json={'type': 'meta'},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        meta = resp.json()
        # The response is {"universe": [...], ...}
        return meta.get('universe', [])
    except Exception as exc:
        logger.warning("get_hyperliquid_perp_universe(): %s", exc)
        return []
