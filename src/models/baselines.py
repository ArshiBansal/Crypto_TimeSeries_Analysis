"""
src/models/baselines.py

Econometric / naïve baseline models.

Baselines implemented
---------------------
1. Naïve / Persistence  : forecast[t] = actual[t-1]
2. ARIMA                : via statsmodels
3. GARCH(1,1)           : via arch library
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA as _ARIMA
from arch import arch_model


# ── 1. Naïve / Persistence ────────────────────────────────────────────────────

def naive_persistence(series: pd.Series, horizon: int = 1) -> pd.Series:
    """
    Persistence (random-walk) forecast: predict last observed value.

    Parameters
    ----------
    series : pd.Series
        Observed target series (e.g. log returns).
    horizon : int
        Steps ahead.

    Returns
    -------
    pd.Series of forecasts (shifted by horizon).
    """
    return series.shift(horizon)


# ── 2. ARIMA ──────────────────────────────────────────────────────────────────

def fit_arima(
    train: pd.Series,
    order: tuple[int, int, int] = (1, 0, 1),
    forecast_steps: int = 1,
) -> dict:
    """
    Fit an ARIMA model on ``train`` and return in-sample fitted values
    plus out-of-sample forecast for ``forecast_steps`` steps.

    Parameters
    ----------
    train : pd.Series
    order : (p, d, q)
    forecast_steps : int

    Returns
    -------
    dict with keys: 'model', 'fitted', 'forecast', 'aic', 'bic'
    """
    model = _ARIMA(train, order=order).fit()
    forecast = model.forecast(steps=forecast_steps)
    return {
        "model": model,
        "fitted": model.fittedvalues,
        "forecast": forecast,
        "aic": model.aic,
        "bic": model.bic,
    }


# ── 3. GARCH(1,1) ─────────────────────────────────────────────────────────────

def fit_garch(
    train: pd.Series,
    p: int = 1,
    q: int = 1,
    dist: str = "normal",
    forecast_horizon: int = 1,
) -> dict:
    """
    Fit a GARCH(p,q) model on ``train`` log-return series.

    Parameters
    ----------
    train : pd.Series
        Log-return series (stationary; in percentage units works best).
    p : int
        GARCH lag order (volatility).
    q : int
        ARCH lag order (innovation).
    dist : str
        Innovation distribution: 'normal', 't', 'skewt'.
    forecast_horizon : int
        Steps-ahead variance forecast.

    Returns
    -------
    dict with keys: 'model', 'result', 'conditional_vol', 'variance_forecast'
    """
    am = arch_model(train * 100, vol="Garch", p=p, q=q, dist=dist)
    res = am.fit(disp="off")
    forecast = res.forecast(horizon=forecast_horizon, reindex=False)
    vol_forecast = np.sqrt(forecast.variance.values[-1]) / 100  # back to raw scale
    return {
        "model": am,
        "result": res,
        "conditional_vol": res.conditional_volatility / 100,
        "variance_forecast": vol_forecast,
        "aic": res.aic,
        "bic": res.bic,
    }
