"""
src/models/dl_models.py

Deep Learning model wrappers — TCN and PatchTST via darts.

Stubs are structured and ready to run once the DL environment is confirmed
(PyTorch + darts installed). Each function follows the same interface as
the ML model wrappers: fit on training, evaluate on val, save artifact.

Activation checklist
--------------------
1. pip install torch darts
2. Remove comment markers (#) from the import block below.
3. Set DL_AVAILABLE = True.
4. Call train_tcn() / train_patchtst() from notebook 03.
"""
from __future__ import annotations

from pathlib import Path as _Path
_PROJECT_ROOT   = _Path(__file__).resolve().parent.parent.parent
_DEFAULT_CONFIG = str(_PROJECT_ROOT / "configs" / "pipeline.yaml")


import yaml
import numpy as np
import pandas as pd
from pathlib import Path

DL_AVAILABLE = False   # flip to True after verifying PyTorch / darts env

try:
    import torch
    from darts import TimeSeries
    from darts.models import TCNModel, PatchTST as PatchTSTModel
    from darts.dataprocessing.transformers import Scaler as DartsScaler
    DL_AVAILABLE = True
except ImportError:
    pass   # stubs still importable; functions will raise informatively


def _require_dl(fn_name: str) -> None:
    if not DL_AVAILABLE:
        raise RuntimeError(
            f"{fn_name}() requires PyTorch and darts.\n"
            "  pip install torch darts\n"
            "Then set DL_AVAILABLE = True in src/models/dl_models.py."
        )


def _build_timeseries(df: pd.DataFrame, ts_col: str, value_col: str) -> "TimeSeries":
    """Convert a pandas DataFrame column to a darts TimeSeries."""
    series = df[[ts_col, value_col]].dropna().copy()
    series[ts_col] = pd.to_datetime(series[ts_col])
    return TimeSeries.from_dataframe(series, time_col=ts_col, value_cols=value_col)


# ── TCN ───────────────────────────────────────────────────────────────────────

def train_tcn(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    ts_col:   str = "timestamp",
    value_col: str = "log_return",
    config_path: str = _DEFAULT_CONFIG,
    artifact_name: str = "tcn_model",
) -> dict:
    """
    Train a Temporal Convolutional Network (TCN) on the training partition.

    Architecture
    ------------
    - Dilated causal convolutions extract local temporal features.
    - input_chunk_length  : look-back window (bars).
    - output_chunk_length : forecast horizon (bars ahead).

    Parameters
    ----------
    train_df, val_df : pandas DataFrames (sorted chronologically).
    ts_col    : timestamp column name.
    value_col : target series column (e.g. 'log_return').

    Returns
    -------
    dict with 'model', 'artifact_path', 'val_pred'
    """
    _require_dl("train_tcn")

    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    art_dir = Path(cfg["artifacts"]["models_dir"])
    art_dir.mkdir(parents=True, exist_ok=True)

    ts_train = _build_timeseries(train_df, ts_col, value_col)
    ts_val   = _build_timeseries(val_df,   ts_col, value_col)

    model = TCNModel(
        input_chunk_length  = 60,
        output_chunk_length = cfg["features"]["target_horizon"],
        num_filters  = 32,
        kernel_size  = 3,
        dilation_base= 2,
        dropout      = 0.1,
        n_epochs     = 20,
        random_state = 42,
        pl_trainer_kwargs={"enable_progress_bar": True},
    )
    model.fit(ts_train, val_series=ts_val, verbose=True)

    artifact_path = art_dir / artifact_name
    model.save(str(artifact_path))
    print(f"TCN artifact saved → {artifact_path}")

    val_pred = model.predict(len(ts_val))
    return {"model": model, "artifact_path": str(artifact_path), "val_pred": val_pred}


# ── PatchTST ──────────────────────────────────────────────────────────────────

def train_patchtst(
    train_df:  pd.DataFrame,
    val_df:    pd.DataFrame,
    ts_col:    str = "timestamp",
    value_col: str = "log_return",
    config_path: str = _DEFAULT_CONFIG,
    artifact_name: str = "patchtst_model",
) -> dict:
    """
    Train a PatchTST model on the training partition.

    Architecture
    ------------
    - Splits the input sequence into non-overlapping patches.
    - Each patch is projected and fed to a Transformer encoder.
    - Provides better long-range dependency capture than pure TCN.

    Parameters
    ----------
    Same as train_tcn().

    Returns
    -------
    dict with 'model', 'artifact_path', 'val_pred'
    """
    _require_dl("train_patchtst")

    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    art_dir = Path(cfg["artifacts"]["models_dir"])
    art_dir.mkdir(parents=True, exist_ok=True)

    ts_train = _build_timeseries(train_df, ts_col, value_col)
    ts_val   = _build_timeseries(val_df,   ts_col, value_col)

    model = PatchTSTModel(
        input_chunk_length  = 60,
        output_chunk_length = cfg["features"]["target_horizon"],
        d_model      = 64,
        nhead        = 4,
        num_encoder_layers = 2,
        dropout      = 0.1,
        patch_len    = 16,
        stride       = 8,
        n_epochs     = 20,
        random_state = 42,
        pl_trainer_kwargs={"enable_progress_bar": True},
    )
    model.fit(ts_train, val_series=ts_val, verbose=True)

    artifact_path = art_dir / artifact_name
    model.save(str(artifact_path))
    print(f"PatchTST artifact saved → {artifact_path}")

    val_pred = model.predict(len(ts_val))
    return {"model": model, "artifact_path": str(artifact_path), "val_pred": val_pred}
