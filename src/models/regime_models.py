"""
src/models/regime_models.py

Unsupervised regime discovery (GMM, HMM) with basic diagnostics.
"""
from __future__ import annotations

from pathlib import Path as _Path

_PROJECT_ROOT = _Path(__file__).resolve().parent.parent.parent
_DEFAULT_CONFIG = str(_PROJECT_ROOT / "configs" / "pipeline.yaml")

import yaml
import joblib
import numpy as np
from pathlib import Path
from sklearn.mixture import GaussianMixture
from hmmlearn.hmm import GaussianHMM


def _coerce_2d(X: np.ndarray) -> np.ndarray:
    array = np.asarray(X, dtype=float)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    return array


def gmm_diagnostics(X: np.ndarray, model: GaussianMixture, labels: np.ndarray) -> dict:
    """Basic model-selection and state-quality diagnostics for GMM."""
    X = _coerce_2d(X)
    labels = np.asarray(labels).reshape(-1)
    n = int(X.shape[0])
    k = int(getattr(model, "n_components", 0))
    d = int(X.shape[1])
    cov_type = str(getattr(model, "covariance_type", "full"))

    if cov_type == "full":
        cov_params = k * d * (d + 1) / 2
    elif cov_type == "diag":
        cov_params = k * d
    elif cov_type in {"tied", "spherical"}:
        cov_params = d * (d + 1) / 2 if cov_type == "tied" else k
    else:
        cov_params = k * d
    mean_params = k * d
    weight_params = max(k - 1, 0)
    n_params = int(mean_params + cov_params + weight_params)

    log_likelihood = float(model.score(X) * n) if n else float("nan")
    bic = float(np.nan) if n == 0 else float(n_params * np.log(n) - 2 * log_likelihood)
    aic = float(np.nan) if n == 0 else float(2 * n_params - 2 * log_likelihood)
    counts = np.bincount(labels, minlength=max(k, 1)).astype(int)
    shares = counts / n if n else counts.astype(float)
    return {
        "n_components": k,
        "log_likelihood": log_likelihood,
        "aic": aic,
        "bic": bic,
        "state_counts": counts.tolist(),
        "state_shares": shares.tolist(),
    }


def hmm_diagnostics(X: np.ndarray, model: GaussianHMM, states: np.ndarray) -> dict:
    """Basic diagnostics for HMM regimes."""
    X = _coerce_2d(X)
    states = np.asarray(states).reshape(-1)
    n = int(X.shape[0])
    log_likelihood = float(model.score(X)) if n else float("nan")
    transmat = np.asarray(getattr(model, "transmat_", []), dtype=float)
    persistence = np.diag(transmat).tolist() if transmat.size else []
    counts = np.bincount(states, minlength=int(getattr(model, "n_components", 1))).astype(int)
    shares = counts / n if n else counts.astype(float)
    switches = int(np.sum(states[1:] != states[:-1])) if n > 1 else 0
    return {
        "n_components": int(getattr(model, "n_components", len(counts))),
        "log_likelihood": log_likelihood,
        "state_counts": counts.tolist(),
        "state_shares": shares.tolist(),
        "state_persistence": persistence,
        "transition_count": switches,
        "switch_rate": float(switches / max(n - 1, 1)),
    }


def fit_gmm(
    X_train: np.ndarray,
    config_path: str = _DEFAULT_CONFIG,
    artifact_name: str = "gmm",
) -> dict:
    """Fit a Gaussian Mixture Model on training features to discover regimes."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    gmm_cfg = cfg["models"]["gmm"]
    art_dir = Path(cfg["artifacts"]["models_dir"])
    art_dir.mkdir(parents=True, exist_ok=True)

    X_train = _coerce_2d(X_train)
    model = GaussianMixture(
        n_components=gmm_cfg["n_components"],
        covariance_type=gmm_cfg["covariance_type"],
        max_iter=gmm_cfg.get("max_iter", 200),
        random_state=42,
    )
    model.fit(X_train)

    labels = model.predict(X_train)
    probs = model.predict_proba(X_train)

    artifact_path = art_dir / f"{artifact_name}.pkl"
    joblib.dump(model, artifact_path)
    print(f"  GMM artifact saved → {artifact_path}")

    return {
        "model": model,
        "labels": labels,
        "probabilities": probs,
        "diagnostics": gmm_diagnostics(X_train, model, labels),
        "artifact_path": str(artifact_path),
    }


def fit_hmm(
    X_train: np.ndarray,
    config_path: str = _DEFAULT_CONFIG,
    artifact_name: str = "hmm",
) -> dict:
    """Fit a Gaussian HMM on chronologically ordered training features."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    hmm_cfg = cfg["models"]["hmm"]
    art_dir = Path(cfg["artifacts"]["models_dir"])
    art_dir.mkdir(parents=True, exist_ok=True)

    X_train = _coerce_2d(X_train)
    model = GaussianHMM(
        n_components=hmm_cfg["n_components"],
        covariance_type=hmm_cfg["covariance_type"],
        n_iter=hmm_cfg.get("n_iter", 200),
        random_state=42,
    )
    model.fit(X_train)
    states = model.predict(X_train)

    artifact_path = art_dir / f"{artifact_name}.pkl"
    joblib.dump(model, artifact_path)
    print(f"  HMM artifact saved → {artifact_path}")

    return {
        "model": model,
        "states": states,
        "diagnostics": hmm_diagnostics(X_train, model, states),
        "artifact_path": str(artifact_path),
    }
