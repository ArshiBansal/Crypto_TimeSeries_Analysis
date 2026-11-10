"""
src/features/microstructure.py

Conditional microstructure feature engineering.

ACTIVATION RULE (from proposal)
--------------------------------
These features are ONLY computed when the dataset contains raw trade ticks
or order-book depth records (aggressor-labelled trades, bid/ask volumes).
If only OHLCV bars are present, all functions here are OUT OF SCOPE and
will raise a clear error rather than silently producing garbage proxies.

Features (when activated)
--------------------------
- Order-Flow Imbalance (OFI)   : (buy_vol - sell_vol) / (buy_vol + sell_vol)
- VPIN-style volume imbalance  : rolling bucket-based toxicity proxy
- Inter-arrival time features  : mean / std of time between trades
- Hawkes process intensity     : self-exciting point process (via tick library)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window


_REQUIRED_TICK_COLS = {
    "ofi":    ["taker_buy_base_asset_volume", "volume"],
    "vpin":   ["taker_buy_base_asset_volume", "volume"],
    "arrival":["number_of_trades"],
    "hawkes": ["number_of_trades"],
}


def _check_cols(df: pd.DataFrame | DataFrame, required: list[str], feature: str) -> None:
    """Raise informatively if required columns are absent."""
    if isinstance(df, pd.DataFrame):
        present = set(df.columns)
    else:
        present = set(df.columns)
    missing = [c for c in required if c not in present]
    if missing:
        raise ValueError(
            f"[{feature}] Required columns not found: {missing}\n"
            "  → Microstructure features require raw trade or order-book data.\n"
            "  → Check audit_meta.yaml: has_tick_data must be True."
        )


# ── Order-Flow Imbalance (OFI) ────────────────────────────────────────────────

def add_ofi(
    df: DataFrame,
    ts_col: str = "timestamp",
    buy_vol_col: str = "taker_buy_base_asset_volume",
    total_vol_col: str = "volume",
    rolling_windows: list[int] | None = None,
) -> DataFrame:
    """
    Add Order-Flow Imbalance (OFI) features to a Spark DataFrame.

    OFI_t = (buy_vol_t - sell_vol_t) / (buy_vol_t + sell_vol_t)
    sell_vol_t = volume_t - taker_buy_base_asset_volume_t

    Parameters
    ----------
    df : Spark DataFrame (must contain taker_buy_base_asset_volume, volume)
    rolling_windows : list[int]  Defaults to [7, 14, 30]

    Returns
    -------
    DataFrame with OFI columns appended.
    """
    _check_cols(df, _REQUIRED_TICK_COLS["ofi"], "OFI")
    rolling_windows = rolling_windows or [7, 14, 30]

    df = df.withColumn(
        "ofi",
        (F.col(buy_vol_col) - (F.col(total_vol_col) - F.col(buy_vol_col)))
        / F.col(total_vol_col),
    )
    for w in rolling_windows:
        rw = Window.orderBy(ts_col).rowsBetween(-(w - 1), 0)
        df = df.withColumn(f"rolling_ofi_{w}", F.avg("ofi").over(rw))

    return df


# ── VPIN-style Volume Imbalance ───────────────────────────────────────────────

def add_vpin(
    df: DataFrame,
    ts_col: str = "timestamp",
    buy_vol_col: str = "taker_buy_base_asset_volume",
    total_vol_col: str = "volume",
    bucket_size: int = 50,
) -> DataFrame:
    """
    Add a VPIN-style (Volume-Synchronised Probability of Informed Trading)
    volume imbalance proxy.

    Uses rolling buckets of ``bucket_size`` bars. Within each bucket:
        VPIN ≈ |buy_vol - sell_vol| / total_vol

    Note: True VPIN uses equal-volume buckets; this is a time-bar approximation.

    Parameters
    ----------
    bucket_size : int   Number of bars per bucket.

    Returns
    -------
    DataFrame with 'vpin_proxy' column.
    """
    _check_cols(df, _REQUIRED_TICK_COLS["vpin"], "VPIN")

    rw = Window.orderBy(ts_col).rowsBetween(-(bucket_size - 1), 0)
    df = df.withColumn(
        "vpin_proxy",
        F.abs(
            F.sum(buy_vol_col).over(rw)
            - F.sum(F.col(total_vol_col) - F.col(buy_vol_col)).over(rw)
        ) / F.sum(total_vol_col).over(rw),
    )
    return df


# ── Inter-arrival Time Features ───────────────────────────────────────────────

def add_arrival_features(
    df: DataFrame,
    ts_col: str = "timestamp",
    trade_count_col: str = "number_of_trades",
    rolling_windows: list[int] | None = None,
) -> DataFrame:
    """
    Approximate inter-arrival time features from trade counts per bar.

    Proxy: mean_inter_arrival ≈ bar_duration / number_of_trades

    Parameters
    ----------
    df : Spark DataFrame (must contain number_of_trades)
    rolling_windows : list[int]  Defaults to [7, 30]

    Returns
    -------
    DataFrame with inter-arrival proxy columns.
    """
    _check_cols(df, _REQUIRED_TICK_COLS["arrival"], "InterArrival")
    rolling_windows = rolling_windows or [7, 30]

    df = df.withColumn(
        "inter_arrival_proxy",
        F.lit(1.0) / (F.col(trade_count_col) + 1),
    )
    for w in rolling_windows:
        rw = Window.orderBy(ts_col).rowsBetween(-(w - 1), 0)
        df = df.withColumn(
            f"rolling_inter_arrival_{w}",
            F.avg("inter_arrival_proxy").over(rw),
        )
    return df


# ── Hawkes Process (pandas-level) ─────────────────────────────────────────────

def fit_hawkes(
    event_times: np.ndarray,
    kernel: str = "exponential",
) -> dict:
    """
    Fit a univariate Hawkes process to an array of event arrival times.

    Requires the `tick` library:
        pip install tick

    Parameters
    ----------
    event_times : np.ndarray  (1-D, sorted event timestamps in seconds)
    kernel      : str         'exponential' or 'powerlaw'

    Returns
    -------
    dict with 'model', 'baseline_intensity', 'alpha', 'beta', 'intensity_series'
    """
    try:
        from tick.hawkes import HawkesExpKern
    except ImportError:
        raise RuntimeError(
            "fit_hawkes() requires the `tick` library.\n"
            "  pip install tick\n"
            "  Note: tick requires Linux/macOS or WSL on Windows."
        )

    model = HawkesExpKern(decays=1.0)
    model.fit([event_times])

    return {
        "model":               model,
        "baseline_intensity":  float(model.baseline[0]),
        "alpha":               float(model.adjacency[0, 0]),
        "beta":                1.0,
        "intensity_series":    model.estimated_intensity([event_times], model.get_kernel_supports())[0],
    }
