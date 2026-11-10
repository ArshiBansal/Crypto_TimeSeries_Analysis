"""
src/data/profiler.py

Schema and quality profiling utilities for PySpark DataFrames.
Run after ingestion, before any cleaning/feature engineering.
"""
from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def profile_dataframe(df: DataFrame, name: str = "DataFrame") -> dict:
    """
    Compute a comprehensive quality profile of a Spark DataFrame.

    Checks performed
    ----------------
    - Row count
    - Schema (column names, dtypes)
    - Null / missing percentages per column
    - Duplicate row count
    - Min / max of numeric columns
    - Timestamp range (auto-detected column named 'timestamp'/'date'/'open_time')

    Parameters
    ----------
    df : pyspark.sql.DataFrame
    name : str
        Label used in the printed report.

    Returns
    -------
    dict
        Profile dictionary (can be serialised to YAML / JSON for the report).
    """
    print(f"\n{'='*60}")
    print(f"  PROFILE: {name}")
    print(f"{'='*60}")

    row_count = df.count()
    print(f"  Rows        : {row_count:,}")
    print(f"  Columns     : {len(df.columns)}")
    print(f"\n  Schema:")
    df.printSchema()

    # Null percentages
    print("  Null percentages:")
    null_exprs = [
        F.round(F.sum(F.col(c).isNull().cast("int")) / row_count * 100, 2)
        .alias(c)
        for c in df.columns
    ]
    null_df = df.select(null_exprs).collect()[0].asDict()
    for col, pct in null_df.items():
        flag = " ⚠" if pct and pct > 0 else ""
        print(f"    {col:<35} {pct}%{flag}")

    # Duplicates
    dup_count = row_count - df.dropDuplicates().count()
    print(f"\n  Duplicate rows : {dup_count:,}")

    # Timestamp range (look for common timestamp column names)
    ts_candidates = [c for c in df.columns if c.lower() in
                     ("timestamp", "date", "open_time", "time", "close_time")]
    ts_range = {}
    if ts_candidates:
        ts_col = ts_candidates[0]
        bounds = df.select(F.min(ts_col).alias("min"), F.max(ts_col).alias("max")).collect()[0]
        ts_range = {"column": ts_col, "min": str(bounds["min"]), "max": str(bounds["max"])}
        print(f"\n  Timestamp range ({ts_col}):")
        print(f"    min : {ts_range['min']}")
        print(f"    max : {ts_range['max']}")

    # Numeric summary
    numeric_cols = [f.name for f in df.schema.fields
                    if str(f.dataType) in ("DoubleType()", "FloatType()",
                                           "LongType()", "IntegerType()")]
    if numeric_cols:
        print(f"\n  Numeric summary (selected columns):")
        df.select(numeric_cols).describe().show(truncate=False)

    profile = {
        "name": name,
        "row_count": row_count,
        "columns": df.columns,
        "null_pct": null_df,
        "duplicate_rows": dup_count,
        "timestamp_range": ts_range,
    }
    return profile
