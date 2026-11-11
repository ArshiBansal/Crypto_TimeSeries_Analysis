"""
src/models/experiment_tracker.py

Lightweight experiment logging to configs/experiments.yaml.
Optionally integrates with MLflow when available.
"""
from __future__ import annotations

import yaml
import datetime
from pathlib import Path

# Always resolve relative to project root (this file lives at src/models/)
_PROJECT_ROOT    = Path(__file__).resolve().parent.parent.parent
EXPERIMENTS_FILE = str(_PROJECT_ROOT / "configs" / "experiments.yaml")


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
    Record an experiment run in configs/experiments.yaml.

    Repeated logging of the same model, target, split, metrics, and artifact
    is idempotent; changed metrics remain a separate run.

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
    logged_at = datetime.datetime.now(datetime.timezone.utc)
    run_id = f"run_{logged_at.strftime('%Y%m%d_%H%M%S_%f')}"

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
        "logged_at": logged_at.isoformat(),
    }

    exp_path = Path(EXPERIMENTS_FILE)
    exp_path.parent.mkdir(parents=True, exist_ok=True)   # create configs/ if missing

    # Load existing data, or start fresh if file doesn't exist yet
    if exp_path.exists():
        with open(exp_path, "r") as f:
            data = yaml.safe_load(f) or {"runs": []}
    else:
        data = {"runs": []}

    if not isinstance(data, dict):
        raise ValueError(f"Experiment log must contain a YAML mapping: {exp_path}")
    runs = data.setdefault("runs", [])
    if not isinstance(runs, list):
        raise ValueError(f"Experiment log 'runs' must be a YAML list: {exp_path}")

    identity_fields = (
        "model", "target", "train_end", "val_end", "test_end",
        "metrics", "artifact_path", "notes",
    )
    existing = next(
        (run for run in runs
         if isinstance(run, dict)
         and all(run.get(field) == entry[field] for field in identity_fields)),
        None,
    )
    if existing is not None:
        print(f"  Run already logged → {existing['run_id']}")
        return existing["run_id"]

    runs.append(entry)

    with open(exp_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    print(f"  Run logged → {run_id}")
    return run_id
