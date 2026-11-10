"""
src/features/volume.py

Volume and trading-activity feature engineering (past-only windows).

Features produced
-----------------
- rolling_volume_{w}    : Rolling mean volume
- volume_surprise_{w}   : volume / rolling_volume - 1  (deviation from mean)
- rolling_trades_{w}    : Rolling mean trade count     [if 'number_of_trades' present]
- taker_imbalance       : taker_buy_volume / volume    [if taker_buy_base_asset_volume present]
- lag_volume_{l}        : Lagged raw volume
"""
from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window


def add_volume_features(
    df: DataFrame,
    ts_col: str = "timestamp",
    volume_col: str = "volume",
    rolling_windows: list[int] | None = None,
    lag_volumes: list[int] | None = None,
) -> DataFrame:
    """
    Append volume/activity features to ``df``.

    Parameters
    ----------
    df : DataFrame (sorted by ts_col)
    volume_col : str
        Name of the raw volume column.
    rolling_windows : list[int]
        Defaults to [7, 14, 30].
    lag_volumes : list[int]
        Defaults to [1, 2, 3].

    Returns
    -------
    DataFrame with new volume columns appended.
    """
    rolling_windows = rolling_windows or [7, 14, 30]
    lag_volumes = lag_volumes or [1, 2, 3]
    w_order = Window.orderBy(ts_col)

    for win in rolling_windows:
        rw = Window.orderBy(ts_col).rowsBetween(-(win - 1), 0)
        roll_col = f"rolling_volume_{win}"
        df = df.withColumn(roll_col, F.avg(volume_col).over(rw))
        df = df.withColumn(
            f"volume_surprise_{win}",
            F.col(volume_col) / F.col(roll_col) - 1,
        )

    for lag in lag_volumes:
        df = df.withColumn(f"lag_volume_{lag}", F.lag(volume_col, lag).over(w_order))

    # Trade count features (Binance-style 'number_of_trades' column)
    if "number_of_trades" in df.columns:
        for win in rolling_windows:
            rw = Window.orderBy(ts_col).rowsBetween(-(win - 1), 0)
            df = df.withColumn(
                f"rolling_trades_{win}",
                F.avg("number_of_trades").over(rw),
            )

    # Taker buy imbalance (Binance aggregate bars expose this)
    taker_col = "taker_buy_base_asset_volume"
    if taker_col in df.columns:
        df = df.withColumn(
            "taker_imbalance",
            F.col(taker_col) / F.col(volume_col),
        )

    return df
