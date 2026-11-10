"""
src/visualisation/eda_plots.py

Reusable EDA plotting routines (Matplotlib / Plotly).
All functions accept pandas DataFrames and return Plotly figures
(also saveable as PNG/HTML to reports/figures/).
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_price_volume(
    df: pd.DataFrame,
    ts_col: str = "timestamp",
    close_col: str = "close",
    volume_col: str = "volume",
    title: str = "BTC Price & Volume",
) -> go.Figure:
    """Dual-axis candlestick / close price + volume bar chart."""
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.7, 0.3],
        subplot_titles=("Price (Close)", "Volume"),
    )
    fig.add_trace(
        go.Scatter(x=df[ts_col], y=df[close_col], name="Close", line=dict(color="#F7931A")),
        row=1, col=1,
    )
    if volume_col in df.columns:
        fig.add_trace(
            go.Bar(x=df[ts_col], y=df[volume_col], name="Volume", marker_color="#aaa"),
            row=2, col=1,
        )
    fig.update_layout(title=title, template="plotly_dark", height=600, showlegend=False)
    return fig


def plot_returns_distribution(
    df: pd.DataFrame,
    return_col: str = "log_return",
    title: str = "Log-Return Distribution",
) -> go.Figure:
    """Histogram + KDE overlay for return distribution."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=df[return_col].dropna(), nbinsx=200,
        name="Log Return", histnorm="probability density",
        marker_color="#F7931A", opacity=0.7,
    ))
    fig.update_layout(title=title, template="plotly_dark",
                      xaxis_title="Log Return", yaxis_title="Density")
    return fig


def plot_correlation_heatmap(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    title: str = "Feature Correlation Matrix",
) -> go.Figure:
    """Plotly heatmap of Pearson correlations."""
    cols = columns or df.select_dtypes("number").columns.tolist()
    corr = df[cols].corr()
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index,
        colorscale="RdBu", zmid=0,
    ))
    fig.update_layout(title=title, template="plotly_dark", height=700)
    return fig
