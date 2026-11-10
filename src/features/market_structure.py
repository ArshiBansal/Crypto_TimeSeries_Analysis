"""
src/features/market_structure.py

Market-structure feature engineering (past-only windows).

Features produced
-----------------
- price_range_{w}       : Rolling mean of (high - low) / close
- rolling_drawdown_{w}  : Rolling drawdown from rolling peak
- momentum_{w}          : close / rolling-mean-close - 1
- trend_slope_{w}       : OLS slope proxy = (close - lag_close_w) / w
"""
from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window


def add_market_structure_features(
    df: DataFrame,
    ts_col: str = "timestamp",
    close_col: str = "close",
    high_col: str = "high",
    low_col: str = "low",
    rolling_windows: list[int] | None = None,
) -> DataFrame:
    """
    Append market-structure features to ``df``.

    Parameters
    ----------
    df : DataFrame (sorted by ts_col; must contain close, high, low columns)
    rolling_windows : list[int]
        Defaults to [7, 14, 30, 60].

    Returns
    -------
    DataFrame with new market-structure columns appended.
    """
    rolling_windows = rolling_windows or [7, 14, 30, 60]
    w_order = Window.orderBy(ts_col)

    for win in rolling_windows:
        rw = Window.orderBy(ts_col).rowsBetween(-(win - 1), 0)

        # Price range (normalised by close) — guard close == 0
        df = df.withColumn(
            f"price_range_{win}",
            F.avg(
                F.when(
                    F.col(close_col) != 0,
                    (F.col(high_col) - F.col(low_col)) / F.col(close_col),
                ).otherwise(F.lit(None).cast("double"))
            ).over(rw),
        )

        # Rolling peak and drawdown — guard rolling_max == 0
        rolling_max = F.max(close_col).over(rw)
        df = df.withColumn(
            f"rolling_drawdown_{win}",
            F.when(
                rolling_max.isNotNull() & (rolling_max != 0),
                (F.col(close_col) - rolling_max) / rolling_max,
            ).otherwise(F.lit(None).cast("double")),
        )

        # Momentum: close vs rolling mean — guard avg_close == 0
        avg_close = F.avg(close_col).over(rw)
        df = df.withColumn(
            f"momentum_{win}",
            F.when(
                avg_close.isNotNull() & (avg_close != 0),
                F.col(close_col) / avg_close - 1,
            ).otherwise(F.lit(None).cast("double")),
        )

        # Trend slope proxy — division by constant `win` is always safe
        lag_close = F.lag(close_col, win).over(w_order)
        df = df.withColumn(
            f"trend_slope_{win}",
            (F.col(close_col) - lag_close) / win,
        )

    return df
