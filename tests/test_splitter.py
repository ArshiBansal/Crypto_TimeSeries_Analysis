"""
tests/test_splitter.py

Smoke tests for src/models/splitter.py chronological split logic.
Verifies: no shuffling, correct ratios, contiguous partitions.
"""

import numpy as np
import pandas as pd
import pytest

from src.models.splitter import chronological_split


def _make_frame(n: int) -> pd.DataFrame:
    return pd.DataFrame({
        "timestamp": pd.date_range("2020-01-01", periods=n, freq="h"),
        "value": np.arange(n),
    })


def test_split_ratios():
    df = _make_frame(1000)
    train, val, test = chronological_split(df)
    assert len(train) == 700
    assert len(val) == 150
    assert len(test) == 150


def test_split_is_chronological():
    df = _make_frame(1000)
    train, val, test = chronological_split(df)
    assert train["timestamp"].max() < val["timestamp"].min()
    assert val["timestamp"].max() < test["timestamp"].min()


def test_split_no_shuffling():
    df = _make_frame(1000)
    train, _, _ = chronological_split(df)
    # Values should be monotonically increasing (no shuffle)
    assert (train["value"].diff().dropna() >= 0).all()


def test_split_small_frame():
    df = _make_frame(10)
    train, val, test = chronological_split(df)
    assert len(train) == 7
    assert len(val) == 1 or len(val) == 2  # int(10*0.15)=1
    assert len(test) == 2 or len(test) == 1


def test_split_custom_ratios(tmp_path):
    """Verify that a custom config overrides defaults."""
    import yaml
    cfg = {"splits": {"train_ratio": 0.6, "val_ratio": 0.2, "test_ratio": 0.2}}
    cfg_path = tmp_path / "pipeline.yaml"
    cfg_path.write_text(yaml.dump(cfg))
    df = _make_frame(1000)
    train, val, test = chronological_split(df, config_path=str(cfg_path))
    assert len(train) == 600
    assert len(val) == 200
    assert len(test) == 200