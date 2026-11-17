"""
scripts/prepare_data.py

Provisioning script: downloads HuggingFace datasets and runs the four
notebook stages to produce the Parquet files and model artifacts that
the Streamlit dashboard requires.

Usage
-----
    python scripts/prepare_data.py [--full] [--no-notebooks]

By default this runs a streaming sample (fast, low memory) to create
the audit Parquet.  Pass --full to materialise the entire dataset.

The script does NOT modify any Java / Spark / Python path configuration.
It reads paths from configs/pipeline.yaml only.
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
    for directory in (PROCESSED_DIR, ARTIFACTS_DIR, REPORTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def _check_python() -> None:
    """Verify Python >= 3.10."""
    if sys.version_info < (3, 10):
        raise RuntimeError("Python 3.10 or newer is required.")


def _check_dependencies() -> None:
    """Verify the research dependencies are installed."""
    try:
        import pyspark  # noqa: F401
        import datasets  # noqa: F401
        import yaml  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            f"Missing dependency: {exc.name}. "
            "Run: pip install -r requirements-research.txt"
        ) from exc


def _run_notebooks() -> None:
    """Execute the four notebook stages in order via nbconvert."""
    notebooks = [
        "notebooks/01_data_pipeline.ipynb",
        "notebooks/02_feature_engineering.ipynb",
        "notebooks/03_modelling.ipynb",
        "notebooks/04_evaluation_and_export.ipynb",
    ]
    for nb in notebooks:
        print(f"\n{'='*60}")
        print(f"  Running {nb} ...")
        print(f"{'='*60}")
        result = subprocess.run(
            [
                sys.executable, "-m", "jupyter", "nbconvert",
                "--to", "notebook",
                "--execute",
                "--inplace",
                str(PROJECT_ROOT / nb),
            ],
            cwd=str(PROJECT_ROOT),
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Notebook {nb} failed with exit code {result.returncode}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Provision processed Parquet and model artifacts for the dashboard."
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Materialise the full dataset (default: streaming sample).",
    )
    parser.add_argument(
        "--no-notebooks",
        action="store_true",
        help="Only create directories; do not run notebooks.",
    )
    args = parser.parse_args()

    _check_python()
    _check_dependencies()
    _ensure_dirs()

    if args.no_notebooks:
        print("Directories created. Skipping notebook execution.")
        return

    print(f"Provisioning data (full={args.full}) ...")
    _run_notebooks()
    print("\nDone. Run `streamlit run Home.py` to launch the dashboard.")


if __name__ == "__main__":
    main()