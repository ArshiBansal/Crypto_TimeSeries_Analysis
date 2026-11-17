"""
scripts/prepare_data.py

Provisioning script for notebook stages 01-04.

Produces processed Parquet data and model/evaluation artifacts (baselines and
tree forecasts, volatility/regime/anomaly/change-point diagnostics, and
walk-forward/metric exports) as implemented in the notebooks.

Usage
-----
    python scripts/prepare_data.py [--full] [--no-notebooks]

By default this runs a streaming/sample audit path in notebook 01 logic as
configured there. Pass --full to run notebooks as-is for fuller materialization.

The script does not modify Java/Spark/Python path configuration; it reads
paths and settings from configs/pipeline.yaml and notebook/config code.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
REPORTS_DIR = PROJECT_ROOT / "reports"


def _ensure_dirs() -> None:
    for directory in (PROCESSED_DIR, ARTIFACTS_DIR, REPORTS_DIR, ARTIFACTS_DIR / "models", ARTIFACTS_DIR / "scalers", REPORTS_DIR / "metrics", REPORTS_DIR / "figures"):
        directory.mkdir(parents=True, exist_ok=True)


def _check_python() -> None:
    if sys.version_info < (3, 10):
        raise RuntimeError("Python 3.10 or newer is required.")


def _check_dependencies() -> None:
    try:
        import yaml  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            f"Missing dependency: {exc.name}. Run: pip install -r requirements-research.txt"
        ) from exc


def _run_notebooks() -> None:
    notebooks = [
        "notebooks/01_data_pipeline.ipynb",
        "notebooks/02_feature_engineering.ipynb",
        "notebooks/03_modelling.ipynb",
        "notebooks/04_evaluation_and_export.ipynb",
    ]
    for nb in notebooks:
        print(f"\n{'='*60}\n  Running {nb} ...\n{'='*60}")
        result = subprocess.run(
            [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace", str(PROJECT_ROOT / nb)],
            cwd=str(PROJECT_ROOT),
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Notebook {nb} failed with exit code {result.returncode}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Provision processed data and model/evaluation artifacts via notebooks 01-04.")
    parser.add_argument("--full", action="store_true", help="Run notebooks for fuller materialization as configured.")
    parser.add_argument("--no-notebooks", action="store_true", help="Only create directories; do not execute notebooks.")
    args = parser.parse_args()

    _check_python()
    _check_dependencies()
    _ensure_dirs()

    if args.no_notebooks:
        print("Directories created. Skipping notebook execution.")
        return

    print(f"Provisioning (full={args.full}) ...")
    _run_notebooks()
    print("\nDone. Run `streamlit run Home.py` to launch the dashboard.")


if __name__ == "__main__":
    main()
