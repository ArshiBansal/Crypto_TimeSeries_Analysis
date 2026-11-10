"""
src/models/rolling_validator.py

Rolling-origin (walk-forward) cross-validation for time-series models.

Purpose
-------
Used for hyperparameter tuning to prevent look-ahead bias.
Each fold trains on all data up to t, validates on the next h steps.
Preprocessing (scaler) is re-fitted on each training window.

This implements the proposal requirement:
  "Rolling-origin or expanding-window validation for hyperparameter tuning."
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Callable, Iterator
from sklearn.preprocessing import StandardScaler


def rolling_origin_splits(
    df: pd.DataFrame,
    min_train_size: int,
    val_size: int,
    step: int | None = None,
) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Generate rolling-origin (expanding-window) train/val splits.

    Parameters
    ----------
    df            : DataFrame sorted chronologically.
    min_train_size: Minimum number of rows in first training window.
    val_size      : Number of rows per validation window.
    step          : How far to advance the origin each fold.
                    Defaults to val_size (non-overlapping validation windows).

    Yields
    ------
    (train_fold, val_fold) DataFrames — no data leaks across folds.
    """
    step = step or val_size
    n = len(df)
    start = min_train_size

    while start + val_size <= n:
        train_fold = df.iloc[:start]
        val_fold   = df.iloc[start : start + val_size]
        yield train_fold, val_fold
        start += step


def cross_validate(
    df:             pd.DataFrame,
    feature_cols:   list[str],
    target_col:     str,
    model_fn:       Callable,
    metric_fn:      Callable,
    min_train_size: int,
    val_size:       int,
    step:           int | None = None,
    scale:          bool = True,
) -> pd.DataFrame:
    """
    Run rolling-origin cross-validation for a model.

    Parameters
    ----------
    df            : Full feature DataFrame (chronologically sorted).
    feature_cols  : List of input feature column names.
    target_col    : Target column name.
    model_fn      : Callable(X_train, y_train, X_val, y_val) → predictions array.
    metric_fn     : Callable(y_true, y_pred) → dict of metric scalars.
    min_train_size: Starting training window size (rows).
    val_size      : Validation window size per fold.
    step          : Fold advancement step (default = val_size).
    scale         : If True, fit StandardScaler on each training fold only.

    Returns
    -------
    pd.DataFrame with one row per fold:
      fold | train_size | val_start_idx | val_end_idx | <metric columns>
    """
    records = []
    for fold_idx, (train_fold, val_fold) in enumerate(
        rolling_origin_splits(df, min_train_size, val_size, step)
    ):
        X_tr = train_fold[feature_cols].values
        y_tr = train_fold[target_col].values
        X_va = val_fold[feature_cols].values
        y_va = val_fold[target_col].values

        # Remove NaN rows
        tr_mask = ~(np.isnan(X_tr).any(axis=1) | np.isnan(y_tr))
        va_mask = ~(np.isnan(X_va).any(axis=1) | np.isnan(y_va))
        X_tr, y_tr = X_tr[tr_mask], y_tr[tr_mask]
        X_va, y_va = X_va[va_mask], y_va[va_mask]

        if len(X_tr) < 10 or len(X_va) < 1:
            continue

        # Scaler fitted on training fold ONLY
        if scale:
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_va = scaler.transform(X_va)

        try:
            preds   = model_fn(X_tr, y_tr, X_va, y_va)
            metrics = metric_fn(y_va, preds)
        except Exception as e:
            print(f"  Fold {fold_idx} error: {e}")
            continue

        records.append({
            "fold":          fold_idx,
            "train_size":    len(X_tr),
            "val_start_idx": df.index.get_loc(val_fold.index[0]) if hasattr(df.index, 'get_loc') else 0,
            "val_end_idx":   df.index.get_loc(val_fold.index[-1]) if hasattr(df.index, 'get_loc') else 0,
            **metrics,
        })

    results_df = pd.DataFrame(records)
    if not results_df.empty:
        metric_cols = [c for c in results_df.columns
                       if c not in ("fold", "train_size", "val_start_idx", "val_end_idx")]
        print(f"\nRolling-origin CV summary ({len(results_df)} folds):")
        print(results_df[metric_cols].agg(["mean", "std"]).round(6))
    return results_df
