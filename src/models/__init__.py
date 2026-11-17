"""
src/models/__init__.py
Models sub-package.

Heavy optional dependencies are imported lazily so lightweight utilities
(splitter/evaluator) remain usable without the full research stack.
"""
from __future__ import annotations

from importlib import import_module

__all__ = [
    "chronological_split",
    "naive_persistence", "fit_arima", "fit_garch",
    "log_run",
    "forecast_metrics", "volatility_metrics", "anomaly_metrics",
    "train_lgbm", "train_catboost",
    "fit_gmm", "fit_hmm",
    "fit_isolation_forest", "compute_matrix_profile", "detect_changepoints_cusum",
]

_EXPORTS = {
    "chronological_split": (".splitter", "chronological_split"),
    "naive_persistence": (".baselines", "naive_persistence"),
    "fit_arima": (".baselines", "fit_arima"),
    "fit_garch": (".baselines", "fit_garch"),
    "log_run": (".experiment_tracker", "log_run"),
    "forecast_metrics": (".evaluator", "forecast_metrics"),
    "volatility_metrics": (".evaluator", "volatility_metrics"),
    "anomaly_metrics": (".evaluator", "anomaly_metrics"),
    "train_lgbm": (".lgbm_model", "train_lgbm"),
    "train_catboost": (".catboost_model", "train_catboost"),
    "fit_gmm": (".regime_models", "fit_gmm"),
    "fit_hmm": (".regime_models", "fit_hmm"),
    "fit_isolation_forest": (".anomaly_models", "fit_isolation_forest"),
    "compute_matrix_profile": (".anomaly_models", "compute_matrix_profile"),
    "detect_changepoints_cusum": (".anomaly_models", "detect_changepoints_cusum"),
}


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value