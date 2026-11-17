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
    "naive_persistence", "fit_arima", "fit_garch", "fit_linear_baseline",
    "log_run",
    "forecast_metrics", "volatility_metrics", "anomaly_metrics",
    "directional_accuracy", "forecast_correlation", "mase", "naive_scaled_error", "diebold_mariano",
    "train_lgbm", "train_catboost",
    "fit_gmm", "fit_hmm", "gmm_diagnostics", "hmm_diagnostics",
    "fit_isolation_forest", "compute_matrix_profile", "detect_changepoints_cusum",
    "rolling_origin_splits", "walk_forward_forecast",
]

_EXPORTS = {
    "chronological_split": (".splitter", "chronological_split"),
    "naive_persistence": (".baselines", "naive_persistence"),
    "fit_arima": (".baselines", "fit_arima"),
    "fit_garch": (".baselines", "fit_garch"),
    "fit_linear_baseline": (".baselines", "fit_linear_baseline"),
    "log_run": (".experiment_tracker", "log_run"),
    "forecast_metrics": (".evaluator", "forecast_metrics"),
    "volatility_metrics": (".evaluator", "volatility_metrics"),
    "anomaly_metrics": (".evaluator", "anomaly_metrics"),
    "directional_accuracy": (".evaluator", "directional_accuracy"),
    "forecast_correlation": (".evaluator", "forecast_correlation"),
    "mase": (".evaluator", "mase"),
    "naive_scaled_error": (".evaluator", "naive_scaled_error"),
    "diebold_mariano": (".evaluator", "diebold_mariano"),
    "train_lgbm": (".lgbm_model", "train_lgbm"),
    "train_catboost": (".catboost_model", "train_catboost"),
    "fit_gmm": (".regime_models", "fit_gmm"),
    "fit_hmm": (".regime_models", "fit_hmm"),
    "gmm_diagnostics": (".regime_models", "gmm_diagnostics"),
    "hmm_diagnostics": (".regime_models", "hmm_diagnostics"),
    "fit_isolation_forest": (".anomaly_models", "fit_isolation_forest"),
    "compute_matrix_profile": (".anomaly_models", "compute_matrix_profile"),
    "detect_changepoints_cusum": (".anomaly_models", "detect_changepoints_cusum"),
    "rolling_origin_splits": (".rolling_validator", "rolling_origin_splits"),
    "walk_forward_forecast": (".rolling_validator", "walk_forward_forecast"),
}


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value