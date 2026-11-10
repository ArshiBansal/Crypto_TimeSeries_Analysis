"""
src/models/regime_models.py

Unsupervised regime discovery.

Models
------
- GMM  : Gaussian Mixture Model (static probabilistic clustering)
- HMM  : Hidden Markov Model (sequential latent-state transitions)
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
from sklearn.mixture import GaussianMixture
from hmmlearn.hmm import GaussianHMM


def fit_gmm(
    X_train: np.ndarray,
    config_path: str = _DEFAULT_CONFIG,
    artifact_name: str = "gmm",
) -> dict:
    """
    Fit a Gaussian Mixture Model on training features to discover market regimes.

    Parameters
    ----------
    X_train : np.ndarray  (shape: [n_samples, n_features])
        Feature matrix from training partition ONLY.

    Returns
    -------
    dict with 'model', 'labels', 'probabilities', 'artifact_path'
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    gmm_cfg = cfg["models"]["gmm"]
    art_dir = Path(cfg["artifacts"]["models_dir"])
    art_dir.mkdir(parents=True, exist_ok=True)

    model = GaussianMixture(
        n_components=gmm_cfg["n_components"],
        covariance_type=gmm_cfg["covariance_type"],
        max_iter=gmm_cfg["max_iter"],
        random_state=42,
    )
    model.fit(X_train)

    labels = model.predict(X_train)
    probs  = model.predict_proba(X_train)

    artifact_path = art_dir / f"{artifact_name}.pkl"
    joblib.dump(model, artifact_path)
    print(f"  GMM artifact saved → {artifact_path}")

    return {
        "model": model,
        "labels": labels,
        "probabilities": probs,
        "artifact_path": str(artifact_path),
    }


def fit_hmm(
    X_train: np.ndarray,
    config_path: str = _DEFAULT_CONFIG,
    artifact_name: str = "hmm",
) -> dict:
    """
    Fit a Gaussian HMM on training features.

    Parameters
    ----------
    X_train : np.ndarray  (shape: [n_samples, n_features])
        MUST be ordered chronologically (time-series).

    Returns
    -------
    dict with 'model', 'states', 'artifact_path'
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    hmm_cfg = cfg["models"]["hmm"]
    art_dir = Path(cfg["artifacts"]["models_dir"])
    art_dir.mkdir(parents=True, exist_ok=True)

    model = GaussianHMM(
        n_components=hmm_cfg["n_components"],
        covariance_type=hmm_cfg["covariance_type"],
        n_iter=hmm_cfg["n_iter"],
        random_state=42,
    )
    model.fit(X_train)
    states = model.predict(X_train)

    artifact_path = art_dir / f"{artifact_name}.pkl"
    joblib.dump(model, artifact_path)
    print(f"  HMM artifact saved → {artifact_path}")

    return {"model": model, "states": states, "artifact_path": str(artifact_path)}
