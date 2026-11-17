"""
src/models/evaluator.py

Centralised evaluation metrics for forecasting, volatility, and anomaly tasks.
"""
from __future__ import annotations

import math

import numpy as np


def _valid_pairs(*arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    """Return arrays filtered to positions where all inputs are finite."""
    cleaned = [np.asarray(array, dtype=float).reshape(-1) for array in arrays]
    if not cleaned:
        return ()
    size = min(array.size for array in cleaned)
    cleaned = [array[:size] for array in cleaned]
    mask = np.ones(size, dtype=bool)
    for array in cleaned:
        mask &= np.isfinite(array)
    return tuple(array[mask] for array in cleaned)


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Share of predictions matching the sign of the actual return (zeros ignored)."""
    y_true, y_pred = _valid_pairs(y_true, y_pred)
    if y_true.size == 0:
        return float("nan")
    mask = (y_true != 0) & (y_pred != 0)
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.sign(y_true[mask]) == np.sign(y_pred[mask])))


def forecast_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Pearson correlation / information coefficient between forecast and actual."""
    y_true, y_pred = _valid_pairs(y_true, y_pred)
    if y_true.size < 2:
        return float("nan")
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def mase(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray | None = None, seasonality: int = 1) -> float:
    """
    Mean Absolute Scaled Error.

    Scales MAE by the in-sample naive forecast MAE. When ``y_train`` is not
    provided, falls back to a one-step naive baseline on ``y_true``.
    """
    y_true, y_pred = _valid_pairs(y_true, y_pred)
    if y_true.size == 0:
        return float("nan")
    mae = float(np.mean(np.abs(y_true - y_pred)))

    if y_train is not None:
        train = np.asarray(y_train, dtype=float)
        train = train[np.isfinite(train)]
        if train.size > seasonality:
            scale = float(np.mean(np.abs(train[seasonality:] - train[:-seasonality])))
        else:
            scale = float("nan")
    else:
        if y_true.size > seasonality:
            scale = float(np.mean(np.abs(y_true[seasonality:] - y_true[:-seasonality])))
        else:
            scale = float("nan")

    if not np.isfinite(scale) or scale == 0:
        return float("nan")
    return float(mae / scale)


def naive_scaled_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Model MAE divided by naive (persistence) MAE on the same series; < 1 beats naive."""
    y_true, y_pred = _valid_pairs(y_true, y_pred)
    if y_true.size < 2:
        return float("nan")
    naive_pred = np.roll(y_true, 1)[1:]
    actual = y_true[1:]
    pred = y_pred[1:]
    naive_mae = float(np.mean(np.abs(actual - naive_pred)))
    model_mae = float(np.mean(np.abs(actual - pred)))
    if naive_mae == 0:
        return float("nan")
    return float(model_mae / naive_mae)


def diebold_mariano(y_true: np.ndarray, y_pred: np.ndarray, benchmark_pred: np.ndarray, power: int = 2) -> dict:
    """
    Simple Diebold-Mariano style test against a benchmark forecast.

    Uses a HAC-free normal approximation on the loss differential; intended as a
    lightweight diagnostic, not a definitive statistical test.
    """
    y_true, y_pred, benchmark_pred = _valid_pairs(y_true, y_pred, benchmark_pred)
    if y_true.size < 3:
        return {"dm_stat": float("nan"), "p_value": float("nan")}
    err_m = y_true - y_pred
    err_b = y_true - benchmark_pred
    if power == 1:
        d = np.abs(err_m) - np.abs(err_b)
    else:
        d = (err_m ** 2) - (err_b ** 2)
    mean_d = float(np.mean(d))
    var_d = float(np.var(d, ddof=1))
    if var_d <= 0:
        return {"dm_stat": float("nan"), "p_value": float("nan")}
    dm_stat = mean_d / math.sqrt(var_d / d.size)
    # two-sided p-value via normal CDF approximation
    p_value = float(math.erfc(abs(dm_stat) / math.sqrt(2)))
    return {"dm_stat": float(dm_stat), "p_value": p_value}


def forecast_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label: str = "",
    y_train: np.ndarray | None = None,
    naive_pred: np.ndarray | None = None,
) -> dict:
    """
    Forecast evaluation metrics.

    Primary: MAE, RMSE, directional accuracy, correlation.
    Secondary/diagnostic: sMAPE (unstable near zero), MASE, naive-scaled MAE.
    """
    y_true, y_pred = _valid_pairs(y_true, y_pred)
    if y_true.size == 0:
        return {
            "mae": float("nan"), "rmse": float("nan"), "smape": float("nan"),
            "directional_accuracy": float("nan"), "correlation": float("nan"),
            "mase": float("nan"), "naive_scaled_mae": float("nan"),
        }

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        smape_val = np.where(denom == 0, 0, np.abs(y_true - y_pred) / denom)
    smape = float(np.mean(smape_val) * 100)

    result = {
        "mae": mae,
        "rmse": rmse,
        "smape": smape,
        "directional_accuracy": directional_accuracy(y_true, y_pred),
        "correlation": forecast_correlation(y_true, y_pred),
        "mase": mase(y_true, y_pred, y_train=y_train),
        "naive_scaled_mae": naive_scaled_error(y_true, y_pred),
    }
    if naive_pred is not None:
        result.update(diebold_mariano(y_true, y_pred, naive_pred))
    if label:
        da = result["directional_accuracy"]
        corr = result["correlation"]
        print(
            f"  Forecast [{label}]  MAE={mae:.6f}  RMSE={rmse:.6f}  "
            f"sMAPE={smape:.2f}%  DA={da:.3f}  IC={corr:.3f}"
        )
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
    scores = np.asarray(scores, dtype=float).reshape(-1)
    scores = scores[np.isfinite(scores)]
    if scores.size == 0:
        result = {"n_flagged_at_5pct": 0}
        if true_labels is not None:
            result["precision_at_k"] = float("nan")
        return result

    threshold = float(np.percentile(scores, 5))
    result = {"n_flagged_at_5pct": int(np.sum(scores <= threshold))}

    if true_labels is not None:
        labels = np.asarray(true_labels).reshape(-1)
        finite_scores = np.asarray(scores, dtype=float).reshape(-1)
        size = min(labels.size, finite_scores.size)
        labels = labels[:size]
        finite_scores = finite_scores[:size]
        if labels.size == 0 or k <= 0:
            result["precision_at_k"] = float("nan")
            return result

        top_k = min(int(k), labels.size)
        ranked = np.argsort(finite_scores)[:top_k]
        positives = labels[ranked] == 1
        result["precision_at_k"] = float(np.mean(positives)) if positives.size else float("nan")

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