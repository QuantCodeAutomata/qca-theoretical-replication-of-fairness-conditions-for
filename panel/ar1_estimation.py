"""
AR(1) estimation for hourly relative-basis dynamics.

Model:   b_{t+1} = alpha + phi * b_t + epsilon_{t+1}

Fitted via OLS (statsmodels).  Half-life and mean-reversion speed are
derived analytically from the estimated phi coefficient.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Elementary helpers
# ---------------------------------------------------------------------------

def compute_halflife(phi: float) -> float:
    """Implied mean-reversion half-life in hours.

    h = -ln(2) / ln(phi),  valid when 0 < phi < 1.

    Parameters
    ----------
    phi : float
        AR(1) autoregressive coefficient.

    Returns
    -------
    float
        Half-life in hours.  Returns +inf if phi >= 1, NaN if phi <= 0.
    """
    if phi <= 0:
        return float('nan')
    if phi >= 1:
        return float('inf')
    return -np.log(2) / np.log(phi)


def compute_lambda_per_hour(phi: float) -> float:
    """Continuous-time mean-reversion speed per hour.

    lambda = -ln(phi)

    Parameters
    ----------
    phi : float
        AR(1) autoregressive coefficient.

    Returns
    -------
    float
        Mean-reversion speed (positive implies reversion).  Returns NaN if
        phi <= 0.
    """
    if phi <= 0:
        return float('nan')
    return float(-np.log(phi))


# ---------------------------------------------------------------------------
# Core estimation
# ---------------------------------------------------------------------------

def estimate_ar1(
    b_series: np.ndarray,
    include_intercept: bool = True,
) -> dict:
    """Estimate AR(1) dynamics for a relative-basis series via OLS.

    Regresses b_{t+1} on (1, b_t) using statsmodels OLS.

    Parameters
    ----------
    b_series : np.ndarray of float
        Hourly relative-basis observations.  Must have length >= 3 after
        dropping NaNs.
    include_intercept : bool
        If True (default), include an intercept (alpha) in the regression.

    Returns
    -------
    dict
        Keys:
          * alpha           – intercept (0.0 if include_intercept=False)
          * phi             – AR(1) coefficient on b_t
          * halflife_hours  – implied half-life (hours)
          * lambda_per_hour – mean-reversion speed per hour
          * r_squared       – in-sample R²
          * nobs            – number of observations used
          * std_err_phi     – standard error of phi estimate
          * t_stat_phi      – t-statistic for phi
          * p_value_phi     – two-sided p-value for phi

        Returns a dict of NaN values with nobs=0 if estimation fails.
    """
    _nan_result = {
        'alpha': float('nan'),
        'phi': float('nan'),
        'halflife_hours': float('nan'),
        'lambda_per_hour': float('nan'),
        'r_squared': float('nan'),
        'nobs': 0,
        'std_err_phi': float('nan'),
        't_stat_phi': float('nan'),
        'p_value_phi': float('nan'),
    }

    b = np.asarray(b_series, dtype=float)
    valid = ~np.isnan(b)
    b_clean = b[valid]

    if len(b_clean) < 3:
        logger.warning("estimate_ar1: fewer than 3 valid observations; skipping.")
        return _nan_result

    y = b_clean[1:]      # b_{t+1}
    x = b_clean[:-1]     # b_t

    if include_intercept:
        X = sm.add_constant(x, prepend=True, has_constant='add')
    else:
        X = x.reshape(-1, 1)

    try:
        ols = sm.OLS(y, X).fit()
    except Exception as exc:
        logger.warning("estimate_ar1: OLS failed: %s", exc)
        return _nan_result

    if include_intercept:
        alpha = float(ols.params[0])
        phi = float(ols.params[1])
        std_err_phi = float(ols.bse[1])
        t_stat_phi = float(ols.tvalues[1])
        p_value_phi = float(ols.pvalues[1])
    else:
        alpha = 0.0
        phi = float(ols.params[0])
        std_err_phi = float(ols.bse[0])
        t_stat_phi = float(ols.tvalues[0])
        p_value_phi = float(ols.pvalues[0])

    return {
        'alpha': alpha,
        'phi': phi,
        'halflife_hours': compute_halflife(phi),
        'lambda_per_hour': compute_lambda_per_hour(phi),
        'r_squared': float(ols.rsquared),
        'nobs': int(ols.nobs),
        'std_err_phi': std_err_phi,
        't_stat_phi': t_stat_phi,
        'p_value_phi': p_value_phi,
    }


# ---------------------------------------------------------------------------
# Panel-level fitting
# ---------------------------------------------------------------------------

def fit_panel_ar1(
    panel: dict,
) -> pd.DataFrame:
    """Fit AR(1) for each symbol in a panel dictionary.

    Parameters
    ----------
    panel : dict
        Mapping ``{symbol: b_series_array}`` where each value is a 1-D
        array-like of hourly relative-basis observations.

    Returns
    -------
    pd.DataFrame
        One row per symbol with columns matching the keys returned by
        :func:`estimate_ar1` plus a leading ``symbol`` column.
        Rows are sorted by ``halflife_hours`` ascending (fastest mean
        reversion first).
    """
    rows = []
    for symbol, b_series in panel.items():
        result = estimate_ar1(np.asarray(b_series, dtype=float))
        result['symbol'] = symbol
        rows.append(result)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    col_order = [
        'symbol', 'alpha', 'phi', 'halflife_hours', 'lambda_per_hour',
        'r_squared', 'nobs', 'std_err_phi', 't_stat_phi', 'p_value_phi',
    ]
    existing_cols = [c for c in col_order if c in df.columns]
    return (
        df[existing_cols]
        .sort_values('halflife_hours')
        .reset_index(drop=True)
    )
