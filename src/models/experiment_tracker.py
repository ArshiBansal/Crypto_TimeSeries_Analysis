"""
src/models/experiment_tracker.py

Lightweight experiment logging to configs/experiments.yaml.
Optionally integrates with MLflow when available.
"""
from __future__ import annotations

import yaml
import datetime
from pathlib import Path


EXPERIMENTS_FILE = "configs/experiments.yaml"


def log_run(
    model: str,
    target: str,
    train_end: str,
    val_end: str,
    test_end: str,
    metrics: dict,
    artifact_path: str,
    notes: str = "",
) -> str:
    """
    Append an experiment run entry to configs/experiments.yaml.

    Parameters
    ----------
    model : str
        Model identifier (e.g. 'lightgbm', 'arima', 'hmm').
    target : str
        Target variable (e.g. 'log_return_1').
    train_end / val_end / test_end : str
        ISO date strings.
    metrics : dict
        Evaluation metrics dict (e.g. {'mae': 0.002, 'rmse': 0.003}).
    artifact_path : str
        Path to saved model artifact.
    notes : str

    Returns
    -------
    str : run_id
    """
    run_id = f"run_{datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

    entry = {
        "run_id": run_id,
        "model": model,
        "target": target,
        "train_end": train_end,
        "val_end": val_end,
        "test_end": test_end,
        "metrics": metrics,
        "artifact_path": artifact_path,
        "notes": notes,
        "logged_at": datetime.datetime.utcnow().isoformat(),
    }

    exp_path = Path(EXPERIMENTS_FILE)
    with open(exp_path, "r") as f:
        data = yaml.safe_load(f) or {"runs": []}

    data["runs"].append(entry)

    with open(exp_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    print(f"  Run logged → {run_id}")
    return run_id
