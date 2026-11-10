"""
src/models/anomaly_models.py

Anomaly and change-point detection.

Models
------
- Isolation Forest   : multi-feature outlier detection
- Matrix Profile     : subsequence discord / motif discovery (via stumpy)
- CUSUM              : sequential change-point detection (via ruptures)
"""
from __future__ import annotations

from pathlib import Path as _Path
_PROJECT_ROOT   = _Path(__file__).resolve().parent.parent.parent
_DEFAULT_CONFIG = str(_PROJECT_ROOT / "configs" / "pipeline.yaml")


import yaml
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import IsolationForest

import stumpy
import ruptures as rpt


# ── Isolation Forest ──────────────────────────────────────────────────────────

def fit_isolation_forest(
    X_train: np.ndarray,
    X_full: np.ndarray,
    config_path: str = _DEFAULT_CONFIG,
    artifact_name: str = "isolation_forest",
) -> dict:
    """
    Fit Isolation Forest on training data; score the full dataset.

    IMPORTANT: fit on X_train only; threshold inferred from X_train scores.

    Returns
    -------
    dict with 'model', 'train_scores', 'full_scores', 'anomaly_flags', 'artifact_path'
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    if_cfg  = cfg["models"]["isolation_forest"]
    art_dir = Path(cfg["artifacts"]["models_dir"])
    art_dir.mkdir(parents=True, exist_ok=True)

    model = IsolationForest(
        n_estimators=if_cfg["n_estimators"],
        contamination=if_cfg["contamination"],
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train)

    full_scores   = model.score_samples(X_full)       # lower = more anomalous
    anomaly_flags = model.predict(X_full) == -1        # True = anomaly

    artifact_path = art_dir / f"{artifact_name}.pkl"
    joblib.dump(model, artifact_path)
    print(f"  Isolation Forest artifact saved → {artifact_path}")

    return {
        "model": model,
        "train_scores": model.score_samples(X_train),
        "full_scores": full_scores,
        "anomaly_flags": anomaly_flags,
        "artifact_path": str(artifact_path),
    }


# ── Matrix Profile ────────────────────────────────────────────────────────────

def compute_matrix_profile(
    series: np.ndarray,
    window: int = 30,
) -> dict:
    """
    Compute the Matrix Profile of a 1-D time series using stumpy.

    Parameters
    ----------
    series : np.ndarray  (1-D)
        Time series values (e.g. log_return).
    window : int
        Subsequence length (number of bars).

    Returns
    -------
    dict with 'mp' (matrix profile values), 'mp_index', 'discord_idx'
    """
    mp = stumpy.stump(series.astype(np.float64), m=window)
    mp_values  = mp[:, 0].astype(float)
    mp_indices = mp[:, 1].astype(int)
    discord_idx = int(np.argmax(mp_values))   # most unusual subsequence

    return {
        "mp": mp_values,
        "mp_index": mp_indices,
        "discord_idx": discord_idx,
        "discord_value": float(mp_values[discord_idx]),
    }


# ── CUSUM Change-Point Detection ──────────────────────────────────────────────

def detect_changepoints_cusum(
    series: np.ndarray,
    n_bkps: int = 5,
    model: str = "l2",
    max_samples: int = 10_000,
) -> list[int]:
    """
    Detect structural change-points using ruptures (PELT algorithm).

    IMPORTANT: Call this only on training partition for threshold calibration.

    Parameters
    ----------
    series : np.ndarray  (1-D or 2-D [n_samples, n_features])
    n_bkps : int
        Number of breakpoints to detect (used only for reference; PELT with
        a penalty selects the count automatically).
    model : str
        Cost function: 'l2' (default, O(n) memory, detects mean shifts),
        'normal' (mean+variance shifts), 'rbf' (kernel, avoid on large series
        — O(n²) memory).
    max_samples : int
        Cap on series length before fitting.  'rbf' requires O(n²) memory;
        keep this small (≤10 000) when using rbf.

    Returns
    -------
    list[int] – indices of detected change-points (last element = len(series)).
    """
    if series.ndim == 1:
        series = series.reshape(-1, 1)

    # Subsample to avoid OOM on large series (especially for rbf kernel)
    if len(series) > max_samples:
        step = len(series) // max_samples
        series = series[::step]

    algo = rpt.Pelt(model=model).fit(series)
    breakpoints = algo.predict(pen=10)   # pen controls sensitivity
    return breakpoints
