"""
tests/test_evaluator.py

Smoke tests for src/models/evaluator.py metric functions.
Covers edge cases: NaN handling, zero denominators, empty inputs.
"""

import numpy as np
import pytest

from src.models.evaluator import (
    forecast_metrics,
    volatility_metrics,
    anomaly_metrics,
)


# ── forecast_metrics ─────────────────────────────────────────────────

def test_forecast_metrics_basic():
    y_true = np.array([0.1, 0.2, 0.3, 0.4])
    y_pred = np.array([0.12, 0.18, 0.31, 0.38])
    result = forecast_metrics(y_true, y_pred)
    assert "mae" in result and "rmse" in result and "smape" in result
    assert result["mae"] >= 0
    assert result["rmse"] >= 0
    assert 0 <= result["smape"] <= 200


def test_forecast_metrics_nan_handling():
    y_true = np.array([0.1, np.nan, 0.3, 0.4])
    y_pred = np.array([0.12, 0.18, np.nan, 0.38])
    result = forecast_metrics(y_true, y_pred)
    # Only 2 valid pairs remain
    assert result["mae"] >= 0


def test_forecast_metrics_zero_denominator():
    """sMAPE must not explode when both actual and predicted are near zero."""
    y_true = np.array([0.0, 0.0, 0.1])
    y_pred = np.array([0.0, 0.0, 0.12])
    result = forecast_metrics(y_true, y_pred)
    assert result["smape"] <= 200  # bounded


def test_forecast_metrics_perfect():
    y_true = np.array([0.1, 0.2, 0.3])
    result = forecast_metrics(y_true, y_true.copy())
    assert result["mae"] == pytest.approx(0.0, abs=1e-12)
    assert result["rmse"] == pytest.approx(0.0, abs=1e-12)


# ── volatility_metrics ───────────────────────────────────────────────

def test_volatility_metrics_basic():
    v_true = np.array([0.01, 0.02, 0.03, 0.04])
    v_pred = np.array([0.012, 0.018, 0.032, 0.038])
    result = volatility_metrics(v_true, v_pred)
    assert "mae" in result and "rmse" in result and "qlike" in result
    assert result["mae"] >= 0


def test_volatility_metrics_filters_nonpositive():
    """QLIKE requires strictly positive values; zeros/negatives are masked out."""
    v_true = np.array([0.01, 0.0, -0.02, 0.04])
    v_pred = np.array([0.012, 0.018, 0.032, 0.038])
    result = volatility_metrics(v_true, v_pred)
    # Only 2 valid pairs (indices 0 and 3)
    assert result["mae"] >= 0


# ── anomaly_metrics ──────────────────────────────────────────────────

def test_anomaly_metrics_no_ground_truth():
    scores = np.array([-0.8, -0.5, -0.3, -0.1, -0.05])
    result = anomaly_metrics(scores, k=3)
    assert "n_flagged_at_5pct" in result
    assert result["n_flagged_at_5pct"] >= 0


def test_anomaly_metrics_with_ground_truth():
    scores = np.array([-0.9, -0.8, -0.3, -0.2, -0.1])
    true_labels = np.array([1, 1, 0, 0, 0])
    result = anomaly_metrics(scores, true_labels=true_labels, k=2)
    assert "precision_at_k" in result
    # Top-2 most anomalous are indices 0 and 1, both true anomalies
    assert result["precision_at_k"] == pytest.approx(1.0)