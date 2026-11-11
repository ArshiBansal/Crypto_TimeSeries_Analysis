"""
src/visualisation/regime_plots.py

Regime and change-point visualisation.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
import plotly.graph_objects as go


_REGIME_COLORS = ["#4FC3F7", "#F7931A", "#66BB6A", "#EF5350", "#AB47BC"]


def plot_regime_states(
    df: pd.DataFrame,
    close_col: str = "close",
    ts_col: str = "timestamp",
    states: np.ndarray | None = None,
    state_col: str | None = None,
    title: str = "Market Regimes",
) -> go.Figure:
    """
    Plot BTC close price coloured by regime state.

    Parameters
    ----------
    states : np.ndarray | None
        Regime label array aligned with ``df``.
    state_col : str | None
        Alternatively, name of a column in ``df`` containing regime labels.
    """
    if states is None and state_col:
        states = df[state_col].values
    fig = go.Figure()
    if states is not None:
        for s in np.unique(states):
            mask = states == s
            fig.add_trace(go.Scatter(
                x=df[ts_col][mask], y=df[close_col][mask],
                mode="markers", marker=dict(size=3, color=_REGIME_COLORS[int(s) % len(_REGIME_COLORS)]),
                name=f"Regime {s}",
            ))
    else:
        fig.add_trace(go.Scatter(x=df[ts_col], y=df[close_col],
                                 name="Close", line=dict(color="#F7931A")))
    fig.update_layout(title=title, template="plotly_dark")
    return fig


def plot_volatility_regimes(
    df: pd.DataFrame,
    vol_col: str = "rolling_vol_30",
    ts_col: str = "timestamp",
    changepoints: list[int] | None = None,
    title: str = "Volatility + Change-Points",
) -> go.Figure:
    """Plot rolling volatility with optional CUSUM change-point markers."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[ts_col], y=df[vol_col],
                             name="Rolling Vol", line=dict(color="#4FC3F7")))
    if changepoints:
        for cp in changepoints[:-1]:
            if cp < len(df):
                cp_ts = df[ts_col].iloc[cp]
                fig.add_vline(x=cp_ts, line_color="#EF5350", line_dash="dot")
                fig.add_annotation(
                    x=cp_ts, y=1, yref="paper", text="CP",
                    showarrow=False, yshift=8,
                )
    fig.update_layout(title=title, template="plotly_dark")
    return fig
