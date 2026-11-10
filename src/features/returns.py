"""
src/features/returns.py

Return feature engineering using PySpark window functions.

Features produced (all use PAST-ONLY window – no look-ahead bias)
-----------------------------------------------------------------
- simple_return      : (close - prev_close) / prev_close
- log_return         : log(close / prev_close)
- lag_return_{n}     : log_return shifted n bars into the past
- rolling_return_{w} : mean log_return over w bars
"""
from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

TS_COL = "timestamp"


def _order_window() -> Window:
    return Window.orderBy(TS_COL)


def add_return_features(
    df: DataFrame,
    close_col: str = "close",
    lags: list[int] | None = None,
    rolling_windows: list[int] | None = None,
    ts_col: str = TS_COL,
) -> DataFrame:
    """
    Append return-family features to ``df``.

    Parameters
    ----------
    df : DataFrame (must be sorted by ``ts_col``)
    close_col : str
    lags : list[int]
        Past lags for lagged log returns.  Defaults to [1, 2, 3, 5, 10].
    rolling_windows : list[int]
        Bar windows for rolling mean log-return.  Defaults to [7, 14, 30].
    ts_col : str

    Returns
    -------
    DataFrame with new feature columns appended.
    """
    lags = lags or [1, 2, 3, 5, 10]
    rolling_windows = rolling_windows or [7, 14, 30]

    w = Window.orderBy(ts_col)

    # Simple return  — guard against prev_close == 0
    prev_close = F.lag(close_col, 1).over(w)
    df = df.withColumn("prev_close", prev_close)
    df = df.withColumn(
        "simple_return",
        F.when(
            F.col("prev_close").isNotNull() & (F.col("prev_close") != 0),
            (F.col(close_col) - F.col("prev_close")) / F.col("prev_close"),
        ).otherwise(F.lit(None).cast("double")),
    )
    # Log return  — guard against prev_close <= 0 (log undefined)
    df = df.withColumn(
        "log_return",
        F.when(
            F.col("prev_close").isNotNull() & (F.col("prev_close") > 0) & (F.col(close_col) > 0),
            F.log(F.col(close_col) / F.col("prev_close")),
        ).otherwise(F.lit(None).cast("double")),
    )
    df = df.drop("prev_close")

    # Lagged log returns
    for lag in lags:
        df = df.withColumn(
            f"lag_return_{lag}",
            F.lag("log_return", lag).over(w),
        )

    # Rolling mean log return
    for win in rolling_windows:
        rw = Window.orderBy(ts_col).rowsBetween(-(win - 1), 0)
        df = df.withColumn(
            f"rolling_return_{win}",
            F.avg("log_return").over(rw),
        )

    return df
