"""
src/models/splitter.py

Chronological train / validation / test split.

RULE: NO random shuffling of time-series data.
      Splits are contiguous in time.
"""
from __future__ import annotations

from pathlib import Path as _Path
_PROJECT_ROOT   = _Path(__file__).resolve().parent.parent.parent
_DEFAULT_CONFIG = str(_PROJECT_ROOT / "configs" / "pipeline.yaml")


import yaml
import pandas as pd
from pyspark.sql import DataFrame


def chronological_split(
    df: DataFrame | pd.DataFrame,
    config_path: str = _DEFAULT_CONFIG,
    ts_col: str = "timestamp",
) -> tuple:
    """
    Split a time-ordered DataFrame into (train, val, test) using
    ratios from pipeline.yaml.

    Works with both Spark DataFrames (returns Spark DFs) and
    pandas DataFrames (returns pandas DFs).

    Parameters
    ----------
    df : DataFrame | pd.DataFrame
        Must be sorted ascending by ts_col.
    config_path : str
    ts_col : str

    Returns
    -------
    (train, val, test) of the same type as ``df``.
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)["splits"]

    train_ratio: float = cfg["train_ratio"]
    val_ratio: float   = cfg["val_ratio"]

    if isinstance(df, pd.DataFrame):
        n = len(df)
        train_end = int(n * train_ratio)
        val_end   = int(n * (train_ratio + val_ratio))
        train = df.iloc[:train_end].copy()
        val   = df.iloc[train_end:val_end].copy()
        test  = df.iloc[val_end:].copy()
        print(f"  Train : {train[ts_col].min()} → {train[ts_col].max()}  ({len(train):,} rows)")
        print(f"  Val   : {val[ts_col].min()} → {val[ts_col].max()}  ({len(val):,} rows)")
        print(f"  Test  : {test[ts_col].min()} → {test[ts_col].max()}  ({len(test):,} rows)")
        return train, val, test

    # Spark path: get boundary timestamps
    from pyspark.sql import functions as F
    ts_vals = df.select(ts_col).orderBy(ts_col).rdd.map(lambda r: r[0]).collect()
    n = len(ts_vals)
    train_end_ts = ts_vals[int(n * train_ratio) - 1]
    val_end_ts   = ts_vals[int(n * (train_ratio + val_ratio)) - 1]

    train = df.filter(F.col(ts_col) <= train_end_ts)
    val   = df.filter((F.col(ts_col) > train_end_ts) & (F.col(ts_col) <= val_end_ts))
    test  = df.filter(F.col(ts_col) > val_end_ts)
    return train, val, test
