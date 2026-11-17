"""
src/models/evaluator.py

Centralised evaluation metrics for forecasting, volatility, and anomaly tasks.
"""
from __future__ import annotations

import numpy as np


def _valid_pairs(*arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    """Return arrays filtered to positions where all inputs are finite."""
    cleaned = [np.asarray(array, float) for array in arrays]
    mask = np.ones(cleaned[0].shape, dtype=bool)
    for array in cleaned:
        mask &= np.isfinite(array)
    return tuple(array[mask] for array in cleaned)


def forecast_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label: str = "",
) -> dict:
    """
    MAE, RMSE, sMAPE for a forecast series.

    sMAPE note: when both actual and predicted are near zero the denominator
    shrinks and sMAPE becomes unreliable — interpret cautiously for log-returns.
    """
    y_true, y_pred = _valid_pairs(y_true, y_pred)
    if y_true.size == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "smape": float("nan")}

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        smape_val = np.where(denom == 0, 0, np.abs(y_true - y_pred) / denom)
    smape = float(np.mean(smape_val) * 100)

    result = {"mae": mae, "rmse": rmse, "smape": smape}
    if label:
        print(f"  Forecast [{label}]  MAE={mae:.6f}  RMSE={rmse:.6f}  sMAPE={smape:.2f}%")
    return result


def volatility_metrics(
    v_true: np.ndarray,
    v_pred: np.ndarray,
    label: str = "",
) -> dict:
    """MAE, RMSE, and QLIKE for volatility forecasts using positive pairs only."""
    v_true, v_pred = _valid_pairs(v_true, v_pred)
    mask = (v_true > 0) & (v_pred > 0)
    v_true, v_pred = v_true[mask], v_pred[mask]
    if v_true.size == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "qlike": float("nan")}

    mae = float(np.mean(np.abs(v_true - v_pred)))
    rmse = float(np.sqrt(np.mean((v_true - v_pred) ** 2)))
    ratio = v_true / v_pred
    qlike = float(np.mean(np.log(v_pred) + ratio))

    result = {"mae": mae, "rmse": rmse, "qlike": qlike}
    if label:
        print(f"  Volatility [{label}]  MAE={mae:.6f}  RMSE={rmse:.6f}  QLIKE={qlike:.6f}")
    return result


def anomaly_metrics(
    scores: np.ndarray,
    true_labels: np.ndarray | None = None,
    k: int = 10,
) -> dict:
    """Return anomaly summary metrics and precision-at-k when labels are provided."""
    scores = np.asarray(scores, float)
    scores = scores[np.isfinite(scores)]
    if scores.size == 0:
        result = {"n_flagged_at_5pct": 0}
        if true_labels is not None:
            result["precision_at_k"] = float("nan")
        return result

    threshold = float(np.percentile(scores, 5))
    result = {"n_flagged_at_5pct": int(np.sum(scores <= threshold))}

    if true_labels is not None:
        labels = np.asarray(true_labels)
        valid = np.isfinite(np.asarray(scores, float)[: labels.size])
        labels = labels[valid]
        finite_scores = np.asarray(scores, float)[: labels.size][valid]
        if labels.size == 0 or k <= 0:
            result["precision_at_k"] = float("nan")
            return result

        top_k = min(int(k), labels.size)
        ranked = np.argsort(finite_scores)[:top_k]
        positives = labels[ranked] == 1
        result["precision_at_k"] = float(np.mean(positives)) if positives.size else float("nan")

    return result