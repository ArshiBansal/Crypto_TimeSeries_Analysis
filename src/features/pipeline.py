"""
src/features/pipeline.py

Orchestrates all feature-engineering steps in the correct order:
  returns → volatility → volume → market_structure → target label

LEAKAGE SAFETY
--------------
All window functions are past-only (rowsBetween(-(w-1), 0)).
Target labels are created by LEADING the return series (shift back),
and rows without a valid future label are dropped before model export.
"""
from __future__ import annotations

from pathlib import Path as _Path
_PROJECT_ROOT   = _Path(__file__).resolve().parent.parent.parent
_DEFAULT_CONFIG = str(_PROJECT_ROOT / "configs" / "pipeline.yaml")


import yaml
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from .returns import add_return_features
from .volatility import add_volatility_features
from .volume import add_volume_features
from .market_structure import add_market_structure_features


def build_feature_pipeline(
    df: DataFrame,
    config_path: str = _DEFAULT_CONFIG,
    ts_col: str = "timestamp",
    close_col: str = "close",
) -> DataFrame:
    """
    Run the full feature-engineering pipeline on a cleaned Spark DataFrame.

    Steps
    -----
    1. Returns (simple, log, lagged, rolling)
    2. Volatility (rolling std, Parkinson, Garman-Klass)
    3. Volume / activity
    4. Market structure (range, drawdown, momentum, trend)
    5. Target label: forward log-return over ``target_horizon`` bars
       (leakage-safe: shift the return series backward in time)
    6. Drop rows with any null in feature or target columns.

    Parameters
    ----------
    df : DataFrame
        Cleaned, sorted Spark DataFrame.
    config_path : str
        Path to pipeline.yaml (reads rolling_windows, lags, target_horizon).
    ts_col : str
    close_col : str

    Returns
    -------
    Feature-rich DataFrame ready for train/val/test split.
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)["features"]

    rolling_windows: list[int] = cfg.get("rolling_windows", [7, 14, 30, 60])
    return_lags: list[int] = cfg.get("return_lags", [1, 2, 3, 5, 10])
    target_horizon: int = cfg.get("target_horizon", 1)
    has_ohlc = all(c in df.columns for c in ["open", "high", "low", "close"])

    # Step 0 – Cast any string-typed numeric columns to DoubleType.
    # The HuggingFace dataset stores OHLCV values as strings (e.g. "4164.02000000").
    # Spark 4 ANSI mode refuses to compare STRING vs integer 0, so we normalise
    # all numeric-looking columns to double before any arithmetic.
    _NUMERIC_COLS = [
        "open", "high", "low", "close", "volume",
        "MA_20", "MA_50", "MA_200", "RSI", "%K", "%D", "ADX", "ATR",
        "Trendline", "MACD", "BL_Upper", "BL_Lower", "Signal", "Histogram",
        "MN_Upper", "MN_Lower", "month",
    ]
    for _col in _NUMERIC_COLS:
        if _col in df.columns:
            col_type = dict(df.dtypes).get(_col, "")
            if col_type == "string":
                df = df.withColumn(_col, F.col(_col).cast("double"))

    # Step 1 – Returns
    df = add_return_features(
        df, close_col=close_col, lags=return_lags,
        rolling_windows=rolling_windows, ts_col=ts_col,
    )

    # Step 2 – Volatility
    df = add_volatility_features(
        df, ts_col=ts_col, rolling_windows=rolling_windows, has_ohlc=has_ohlc,
    )

    # Step 3 – Volume
    if "volume" in df.columns:
        df = add_volume_features(df, ts_col=ts_col, rolling_windows=rolling_windows)

    # Step 4 – Market structure
    if has_ohlc:
        df = add_market_structure_features(
            df, ts_col=ts_col, close_col=close_col,
            rolling_windows=rolling_windows,
        )

    # Step 5 – Target label (future log-return; NO look-ahead in features)
    w_order = Window.orderBy(ts_col)
    df = df.withColumn(
        f"target_log_return_{target_horizon}",
        F.lead("log_return", target_horizon).over(w_order),
    )

    # Step 6 – Drop rows with nulls in target (last `target_horizon` rows)
    df = df.filter(F.col(f"target_log_return_{target_horizon}").isNotNull())

    return df
