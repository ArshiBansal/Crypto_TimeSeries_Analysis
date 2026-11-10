"""
src/models/evaluator.py

Centralised evaluation metrics for all task types defined in the proposal.

Task                | Metrics
--------------------|------------------------------------------------------------
Price / Return Forecast | MAE, RMSE, sMAPE
Volatility Forecast     | MAE, RMSE, QLIKE
Market-State Classification | Macro F1, Balanced Accuracy, PR-AUC
Regime Clustering       | Silhouette Score, Davies-Bouldin Index
Change-Point Detection  | Detection delay, False alarms per period
Anomaly Detection       | Precision@K (if annotated), case count
Forecast Uncertainty    | Coverage (prediction interval), Mean Interval Width
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ── 1. Price / Return Forecast ────────────────────────────────────────────────

def forecast_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    label:  str = "",
) -> dict:
    """
    MAE, RMSE, sMAPE for a forecast series.

    sMAPE note: when both actual and predicted are near zero the denominator
    shrinks and sMAPE becomes unreliable — interpret cautiously for log-returns.
    """
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    mask   = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]

    mae  = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    smape = float(np.mean(np.where(denom == 0, 0, np.abs(y_true - y_pred) / denom)) * 100)

    result = {"mae": mae, "rmse": rmse, "smape": smape}
    if label:
        print(f"  Forecast [{label}]  MAE={mae:.6f}  RMSE={rmse:.6f}  sMAPE={smape:.2f}%")
    return result


# ── 2. Volatility Forecast ────────────────────────────────────────────────────

def volatility_metrics(
    vol_true: np.ndarray | pd.Series,
    vol_pred: np.ndarray | pd.Series,
    label:    str = "",
) -> dict:
    """
    MAE, RMSE, QLIKE for a volatility (or variance) forecast.

    QLIKE = mean( vol_true/vol_pred - log(vol_true/vol_pred) - 1 )
    Lower QLIKE → better conditional coverage of tail risk.
    """
    v_true = np.asarray(vol_true, float)
    v_pred = np.asarray(vol_pred, float)
    mask   = ~(np.isnan(v_true) | np.isnan(v_pred)) & (v_pred > 0) & (v_true > 0)
    v_true, v_pred = v_true[mask], v_pred[mask]

    mae   = float(np.mean(np.abs(v_true - v_pred)))
    rmse  = float(np.sqrt(np.mean((v_true - v_pred) ** 2)))
    qlike = float(np.mean(v_true / v_pred - np.log(v_true / v_pred) - 1))

    result = {"mae": mae, "rmse": rmse, "qlike": qlike}
    if label:
        print(f"  Volatility [{label}]  MAE={mae:.6f}  RMSE={rmse:.6f}  QLIKE={qlike:.6f}")
    return result


# ── 3. Market-State Classification ───────────────────────────────────────────

def classification_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    y_score: np.ndarray | pd.Series | None = None,
    label:  str = "",
) -> dict:
    """
    Macro F1, Balanced Accuracy, and optionally PR-AUC.

    Parameters
    ----------
    y_score : np.ndarray | None
        Predicted probability scores (required for PR-AUC; binary or multiclass OvR).
    """
    from sklearn.metrics import (
        f1_score, balanced_accuracy_score,
        average_precision_score,
    )
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    bal_acc  = float(balanced_accuracy_score(y_true, y_pred))

    result = {"macro_f1": macro_f1, "balanced_accuracy": bal_acc}

    if y_score is not None:
        y_score = np.asarray(y_score)
        try:
            pr_auc = float(average_precision_score(y_true, y_score, average="macro"))
            result["pr_auc"] = pr_auc
        except Exception:
            result["pr_auc"] = float("nan")

    if label:
        print(f"  Classification [{label}]  "
              f"MacroF1={macro_f1:.4f}  BalAcc={bal_acc:.4f}"
              + (f"  PR-AUC={result.get('pr_auc', 'N/A'):.4f}" if y_score is not None else ""))
    return result


# ── 4. Regime Clustering ─────────────────────────────────────────────────────

def clustering_metrics(
    X:      np.ndarray,
    labels: np.ndarray,
    label:  str = "",
) -> dict:
    """
    Silhouette Score and Davies-Bouldin Index for regime/cluster quality.

    Parameters
    ----------
    X      : feature array used for clustering (standardised).
    labels : cluster assignments.
    """
    from sklearn.metrics import silhouette_score, davies_bouldin_score

    n_clusters = len(np.unique(labels))
    if n_clusters < 2:
        print(f"  [{label}] Only 1 cluster — metrics not meaningful.")
        return {"silhouette": float("nan"), "davies_bouldin": float("nan")}

    sil = float(silhouette_score(X, labels, sample_size=min(5000, len(X))))
    db  = float(davies_bouldin_score(X, labels))

    result = {"silhouette": sil, "davies_bouldin": db}
    if label:
        print(f"  Clustering [{label}]  Silhouette={sil:.4f}  Davies-Bouldin={db:.4f}")
    return result


# ── 5. Change-Point Detection ────────────────────────────────────────────────

def changepoint_metrics(
    detected: list[int],
    true_bkps: list[int] | None = None,
    n_total:   int = 0,
    tolerance: int = 5,
    label:     str = "",
) -> dict:
    """
    Change-point detection quality metrics.

    Parameters
    ----------
    detected   : list of detected breakpoint indices.
    true_bkps  : list of ground-truth breakpoint indices (None → qualitative only).
    n_total    : total number of time steps (for false-alarm rate).
    tolerance  : bar tolerance for a hit (detected within ±tolerance of true).

    Returns
    -------
    dict with 'n_detected', 'false_alarms', optionally 'precision', 'recall', 'f1', 'mean_delay'
    """
    result: dict = {"n_detected": len(detected)}

    if true_bkps is None or len(true_bkps) == 0:
        result["note"] = "No ground truth provided — qualitative inspection required."
        if label:
            print(f"  Change-Point [{label}]  Detected={len(detected)} (no ground truth)")
        return result

    hits, delays = [], []
    used = set()
    for tb in true_bkps:
        for db in detected:
            if abs(db - tb) <= tolerance and db not in used:
                hits.append(db)
                delays.append(abs(db - tb))
                used.add(db)
                break

    tp  = len(hits)
    fp  = len(detected) - tp
    fn  = len(true_bkps) - tp
    prec   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0.0
    fa_rate = fp / n_total if n_total > 0 else float("nan")

    result.update({
        "precision": prec, "recall": recall, "f1": f1,
        "false_alarms": fp, "false_alarm_rate": fa_rate,
        "mean_delay": float(np.mean(delays)) if delays else float("nan"),
    })
    if label:
        print(f"  Change-Point [{label}]  "
              f"Precision={prec:.3f}  Recall={recall:.3f}  "
              f"F1={f1:.3f}  FalseAlarms={fp}  MeanDelay={result['mean_delay']:.1f}")
    return result


# ── 6. Anomaly Detection ─────────────────────────────────────────────────────

def anomaly_metrics(
    scores:      np.ndarray,
    true_labels: np.ndarray | None = None,
    k:           int = 50,
    label:       str = "",
) -> dict:
    """
    Anomaly detection metrics.

    Parameters
    ----------
    scores      : anomaly scores (lower = more anomalous for Isolation Forest).
    true_labels : binary ground-truth (1 = anomaly) if available.
    k           : top-k most anomalous points for Precision@K.
    """
    n_flagged = int((scores <= np.percentile(scores, 5)).sum())
    result: dict = {"n_flagged_at_5pct": n_flagged, "k": k}

    if true_labels is not None:
        true_labels = np.asarray(true_labels)
        # Rank by ascending score (most anomalous first)
        top_k_idx = np.argsort(scores)[:k]
        prec_at_k = float(true_labels[top_k_idx].mean())
        result["precision_at_k"] = prec_at_k
        if label:
            print(f"  Anomaly [{label}]  Precision@{k}={prec_at_k:.4f}  "
                  f"Flagged(5%)={n_flagged}")
    else:
        result["note"] = "No ground-truth — qualitative case review required."
        if label:
            print(f"  Anomaly [{label}]  Flagged(5%)={n_flagged}  "
                  "(no ground truth — inspect flagged timestamps manually)")
    return result


# ── 7. Forecast Uncertainty ───────────────────────────────────────────────────

def uncertainty_metrics(
    y_true:  np.ndarray | pd.Series,
    y_lower: np.ndarray | pd.Series,
    y_upper: np.ndarray | pd.Series,
    alpha:   float = 0.90,
    label:   str   = "",
) -> dict:
    """
    Evaluate prediction interval quality.

    Parameters
    ----------
    y_true          : actual target values.
    y_lower, y_upper: lower/upper bounds of the prediction interval.
    alpha           : nominal coverage level (e.g. 0.90 for 90% PI).

    Returns
    -------
    dict with 'coverage' (empirical), 'mean_interval_width', 'winkler_score'.
    """
    y_true  = np.asarray(y_true,  float)
    y_lower = np.asarray(y_lower, float)
    y_upper = np.asarray(y_upper, float)

    in_interval  = (y_true >= y_lower) & (y_true <= y_upper)
    coverage     = float(in_interval.mean())
    mean_width   = float(np.mean(y_upper - y_lower))

    # Winkler score: penalises width and coverage misses
    width  = y_upper - y_lower
    pen    = 2 / (1 - alpha)
    lower_pen = np.where(y_true < y_lower, pen * (y_lower - y_true), 0)
    upper_pen = np.where(y_true > y_upper, pen * (y_true - y_upper), 0)
    winkler   = float(np.mean(width + lower_pen + upper_pen))

    result = {
        "nominal_coverage": alpha,
        "empirical_coverage": coverage,
        "mean_interval_width": mean_width,
        "winkler_score": winkler,
    }
    if label:
        print(f"  Uncertainty [{label}]  "
              f"Coverage={coverage:.3f} (target={alpha})  "
              f"Width={mean_width:.6f}  Winkler={winkler:.6f}")
    return result
