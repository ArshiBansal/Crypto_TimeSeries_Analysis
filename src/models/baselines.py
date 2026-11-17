"""
src/models/baselines.py

Econometric, naive, and linear baseline models.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ── 1. Naive / Persistence ──────────────────────────────────────────

def naive_persistence(series: pd.Series, horizon: int = 1) -> pd.Series:
    """Persistence (random-walk) forecast: predict last observed value."""
    return series.shift(horizon)


# ── 2. ARIMA ────────────────────────────────────────────────────────

def fit_arima(
    train: pd.Series,
    order: tuple[int, int, int] = (1, 0, 1),
    forecast_steps: int = 1,
) -> dict:
    """Fit an ARIMA model and return fitted values and out-of-sample forecast."""
    try:
        from statsmodels.tsa.arima.model import ARIMA
    except ImportError as exc:
        raise ImportError(
            "statsmodels is required for fit_arima. Install requirements-research.txt."
        ) from exc

    model = ARIMA(train, order=order).fit()
    forecast = model.forecast(steps=forecast_steps)
    return {
        "model": model,
        "fitted": model.fittedvalues,
        "forecast": forecast,
        "aic": float(model.aic),
        "bic": float(model.bic),
    }


# ── 3. GARCH / volatility baseline ──────────────────────────────────

def fit_garch(
    train: pd.Series,
    p: int = 1,
    q: int = 1,
    dist: str = "t",
    vol: str = "Garch",
    forecast_horizon: int = 1,
) -> dict:
    """
    Fit a (G)ARCH-family volatility model on ``train`` log-return series.

    Defaults to Student-t innovations, which better capture fat-tailed crypto
    returns than a normal distribution.
    """
    try:
        from arch import arch_model
    except ImportError as exc:
        raise ImportError(
            "arch is required for fit_garch. Install requirements-research.txt."
        ) from exc

    clean = pd.to_numeric(train, errors="coerce").dropna()
    if clean.empty:
        raise ValueError("train series has no valid observations for GARCH fitting.")

    am = arch_model(clean * 100, vol=vol, p=p, q=q, dist=dist)
    res = am.fit(disp="off")
    forecast = res.forecast(horizon=forecast_horizon, reindex=False)
    vol_forecast = np.sqrt(np.asarray(forecast.variance.values[-1], dtype=float)) / 100
    return {
        "model": am,
        "result": res,
        "conditional_vol": res.conditional_volatility / 100,
        "variance_forecast": vol_forecast,
        "aic": float(res.aic),
        "bic": float(res.bic),
        "distribution": dist,
        "vol_model": vol,
    }


# ── 4. Regularized linear baseline ──────────────────────────────────

def fit_linear_baseline(
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    X_val: pd.DataFrame | np.ndarray | None = None,
    y_val: pd.Series | np.ndarray | None = None,
    model_type: str = "ridge",
    alpha: float = 1.0,
) -> dict:
    """
    Fit a regularized linear baseline for tabular return forecasting.

    This provides a simple, interpretable benchmark alongside tree models.
    """
    try:
        from sklearn.linear_model import ElasticNet, Lasso, Ridge
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for fit_linear_baseline."
        ) from exc

    model_type = model_type.lower()
    if model_type == "ridge":
        estimator = Ridge(alpha=alpha)
    elif model_type == "lasso":
        estimator = Lasso(alpha=alpha, max_iter=10000)
    elif model_type in {"elasticnet", "elastic_net"}:
        estimator = ElasticNet(alpha=alpha, l1_ratio=0.5, max_iter=10000)
    else:
        raise ValueError("model_type must be one of: ridge, lasso, elasticnet")

    model = Pipeline([("scaler", StandardScaler()), ("model", estimator)])
    model.fit(X_train, y_train)

    result = {"model": model, "model_type": model_type, "alpha": alpha}
    if X_val is not None:
        result["val_pred"] = model.predict(X_val)
        if y_val is not None:
            result["val_residuals"] = np.asarray(y_val) - np.asarray(result["val_pred"])
    return result
