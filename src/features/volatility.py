"""
src/features/volatility.py

Volatility feature engineering (past-only window functions).

Features produced
-----------------
- rolling_vol_{w}      : Rolling std of log_return over w bars
- rolling_vol_ann_{w}  : Annualised rolling vol (× sqrt(365) for daily bars)
- parkinson_vol_{w}    : Parkinson range-based vol proxy  [if high/low present]
- garman_klass_vol_{w} : Garman-Klass vol proxy           [if OHLC present]
- lag_vol_{w}_{l}      : Lagged rolling volatility (l bars back)
"""
from __future__ import annotations

import math
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

ANN_FACTOR_DAILY = math.sqrt(365)


def add_volatility_features(
    df: DataFrame,
    ts_col: str = "timestamp",
    rolling_windows: list[int] | None = None,
    vol_lags: list[int] | None = None,
    has_ohlc: bool = True,
    ann_factor: float = ANN_FACTOR_DAILY,
) -> DataFrame:
    """
    Append volatility-family features to ``df``.

    Parameters
    ----------
    df : DataFrame (sorted by ts_col; must contain 'log_return')
    rolling_windows : list[int]
        Bar-count windows for rolling std.  Defaults to [7, 14, 30, 60].
    vol_lags : list[int]
        Lags to apply to each rolling vol series.  Defaults to [1, 2].
    has_ohlc : bool
        If True, also compute Parkinson and Garman-Klass proxies from
        'open', 'high', 'low', 'close' columns.
    ann_factor : float
        Annualisation multiplier (sqrt of trading periods per year).

    Returns
    -------
    DataFrame with new volatility columns appended.
    """
    rolling_windows = rolling_windows or [7, 14, 30, 60]
    vol_lags = vol_lags or [1, 2]
    ln2 = math.log(2)

    for win in rolling_windows:
        rw = Window.orderBy(ts_col).rowsBetween(-(win - 1), 0)

        # Close-to-close rolling std
        col_name = f"rolling_vol_{win}"
        df = df.withColumn(col_name, F.stddev("log_return").over(rw))

        # Annualised
        df = df.withColumn(
            f"rolling_vol_ann_{win}",
            F.col(col_name) * ann_factor,
        )

        # Lagged volatility
        for lag in vol_lags:
            df = df.withColumn(
                f"lag_vol_{win}_{lag}",
                F.lag(col_name, lag).over(Window.orderBy(ts_col)),
            )

        if has_ohlc:
            # Parkinson proxy: sqrt(1/(4*ln2) * mean(ln(H/L))^2)
            # Guard: low > 0 required for log(H/L)
            safe_hl_log = F.when(
                (F.col("low") > 0) & (F.col("high") > 0),
                F.pow(F.log(F.col("high") / F.col("low")), 2),
            ).otherwise(F.lit(None).cast("double"))

            df = df.withColumn(
                f"parkinson_vol_{win}",
                F.sqrt(F.avg(safe_hl_log).over(rw) / (4 * ln2)),
            )

            # Garman-Klass proxy
            # Guard: open > 0 and low > 0 required
            safe_hl_sq = F.when(
                (F.col("low") > 0) & (F.col("high") > 0),
                0.5 * F.pow(F.log(F.col("high") / F.col("low")), 2),
            ).otherwise(F.lit(None).cast("double"))

            safe_co_sq = F.when(
                (F.col("open") > 0) & (F.col("close") > 0),
                (2 * ln2 - 1) * F.pow(F.log(F.col("close") / F.col("open")), 2),
            ).otherwise(F.lit(None).cast("double"))

            df = df.withColumn("_gk_term1", safe_hl_sq)
            df = df.withColumn("_gk_term2", safe_co_sq)
            df = df.withColumn(
                f"garman_klass_vol_{win}",
                F.sqrt(F.avg(F.col("_gk_term1") - F.col("_gk_term2")).over(rw)),
            )

    # Drop helper columns
    df = df.drop("_gk_term1", "_gk_term2")
    return df
