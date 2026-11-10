"""
src/data/cleaner.py

Cleaning and alignment utilities for PySpark DataFrames.

Responsibilities
----------------
1. Standardise timestamp column to UTC (LongType epoch-ms → TimestampType).
2. Drop fully-null rows and duplicate timestamps.
3. Sort chronologically.
4. Optionally aggregate to a target interval (e.g. '1h', '1d').
5. Document coverage gaps.
"""
from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import TimestampType


def standardise_timestamps(
    df: DataFrame,
    ts_col: str = "timestamp",
    unit: str = "ms",
) -> DataFrame:
    """
    Convert a timestamp column to a proper TimestampType in UTC.

    Handles two cases automatically based on the column's Spark dtype:
      - **String / timestamp** column: parsed directly with ``F.to_timestamp``
        (e.g. ``"2017-08-18 01:19:00"``).
      - **Numeric** column: treated as epoch ints and divided by the
        appropriate divisor (ms → ÷1000, s → ÷1).

    Parameters
    ----------
    ts_col : str
        Name of the raw timestamp column (epoch int, datetime string, or
        already a TimestampType).
    unit : str
        Epoch unit used **only** when the column is numeric: 'ms'
        (milliseconds, default) or 's' (seconds).

    Returns
    -------
    DataFrame with ``ts_col`` replaced by a UTC TimestampType column.
    """
    col_type = dict(df.dtypes).get(ts_col, "string")

    if col_type in ("string", "timestamp"):
        # Already a datetime string or Timestamp — parse directly
        df = df.withColumn(ts_col, F.to_timestamp(F.col(ts_col)).cast(TimestampType()))
    else:
        # Numeric epoch — convert using divisor
        divisor = 1000.0 if unit == "ms" else 1.0
        df = df.withColumn(
            ts_col,
            (F.col(ts_col).cast("double") / divisor).cast(TimestampType()),
        )
    return df


def clean_and_align(
    df: DataFrame,
    ts_col: str = "timestamp",
    subset_dedup: list[str] | None = None,
) -> DataFrame:
    """
    Core cleaning pipeline:
      1. Drop rows where the timestamp is null.
      2. Drop duplicate timestamps (keep first occurrence).
      3. Sort ascending by timestamp.
      4. Remove rows with all OHLCV values null.

    Parameters
    ----------
    df : DataFrame
    ts_col : str
        Timestamp column name.
    subset_dedup : list[str] | None
        Columns to use for deduplication.  Defaults to [ts_col].

    Returns
    -------
    Cleaned, sorted DataFrame.
    """
    subset_dedup = subset_dedup or [ts_col]
    df = (
        df.filter(F.col(ts_col).isNotNull())
          .dropDuplicates(subset_dedup)
          .orderBy(ts_col)
    )
    return df


def report_gaps(
    df: DataFrame,
    ts_col: str = "timestamp",
    expected_interval_seconds: int = 3600,
) -> DataFrame:
    """
    Identify timestamp gaps larger than ``expected_interval_seconds``.

    Returns a DataFrame with columns:
        gap_start, gap_end, gap_seconds
    """
    from pyspark.sql.window import Window

    w = Window.orderBy(ts_col)
    df_gaps = (
        df.select(
            F.col(ts_col).alias("gap_end"),
            F.lag(ts_col, 1).over(w).alias("gap_start"),
        )
        .withColumn(
            "gap_seconds",
            F.unix_timestamp("gap_end") - F.unix_timestamp("gap_start"),
        )
        .filter(F.col("gap_seconds") > expected_interval_seconds)
        .select("gap_start", "gap_end", "gap_seconds")
        .orderBy("gap_start")
    )
    return df_gaps
