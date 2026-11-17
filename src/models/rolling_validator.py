"""
src/models/rolling_validator.py

Utilities for walk-forward / rolling-origin evaluation of time-series models.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator

import numpy as np
import pandas as pd

from .evaluator import forecast_metrics


@dataclass(frozen=True)
class RollingSplit:
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    fold: int


def rolling_origin_splits(
    n_samples: int,
    initial_train_size: int,
    test_size: int,
    step_size: int | None = None,
    expanding: bool = True,
    max_folds: int | None = None,
) -> Iterator[RollingSplit]:
    """
    Yield chronological walk-forward splits.

    Parameters
    ----------
    n_samples : int
        Total available observations.
    initial_train_size : int
        Number of observations in the first training window.
    test_size : int
        Number of observations in each test window.
    step_size : int | None
        Advance between folds; defaults to ``test_size``.
    expanding : bool
        If True, train window grows from 0..train_end. If False, it slides with
        fixed size ``initial_train_size``.
    """
    step = int(test_size if step_size is None else step_size)
    if n_samples <= 0 or initial_train_size <= 0 or test_size <= 0 or step <= 0:
        raise ValueError("n_samples, initial_train_size, test_size, and step_size must be positive.")

    fold = 0
    train_end = int(initial_train_size)
    while train_end + test_size <= n_samples:
        test_start = train_end
        test_end = test_start + test_size
        train_start = 0 if expanding else max(0, train_end - initial_train_size)
        yield RollingSplit(train_start, train_end, test_start, test_end, fold)
        fold += 1
        if max_folds is not None and fold >= max_folds:
            break
        train_end += step


def walk_forward_forecast(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    fit_predict: Callable[[pd.DataFrame | np.ndarray, pd.Series | np.ndarray, pd.DataFrame | np.ndarray], np.ndarray],
    initial_train_size: int,
    test_size: int,
    step_size: int | None = None,
    expanding: bool = True,
    max_folds: int | None = None,
) -> dict:
    """
    Run walk-forward forecasting and aggregate metrics across folds.

    ``fit_predict`` must accept (X_train, y_train, X_test) and return predictions
    aligned to X_test.
    """
    if isinstance(X, pd.DataFrame):
        X_values = X.reset_index(drop=True)
    else:
        X_values = pd.DataFrame(np.asarray(X))
    y_values = pd.Series(np.asarray(y)).reset_index(drop=True)
    n_samples = len(y_values)

    fold_results = []
    all_true, all_pred = [], []
    for split in rolling_origin_splits(
        n_samples=n_samples,
        initial_train_size=initial_train_size,
        test_size=test_size,
        step_size=step_size,
        expanding=expanding,
        max_folds=max_folds,
    ):
        X_train = X_values.iloc[split.train_start:split.train_end]
        y_train = y_values.iloc[split.train_start:split.train_end]
        X_test = X_values.iloc[split.test_start:split.test_end]
        y_test = y_values.iloc[split.test_start:split.test_end]

        pred = np.asarray(fit_predict(X_train, y_train, X_test), dtype=float).reshape(-1)
        true = np.asarray(y_test, dtype=float).reshape(-1)
        metrics = forecast_metrics(true, pred)
        fold_results.append({"fold": split.fold, **metrics, "n_test": int(true.size)})
        all_true.append(true)
        all_pred.append(pred)

    if not fold_results:
        return {"folds": [], "aggregate": forecast_metrics([], [])}

    aggregate = forecast_metrics(np.concatenate(all_true), np.concatenate(all_pred))
    aggregate["n_folds"] = len(fold_results)
    return {"folds": fold_results, "aggregate": aggregate}
