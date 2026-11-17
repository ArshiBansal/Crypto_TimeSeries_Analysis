"""
src/visualisation/model_plots.py

Model evaluation and forecast plots.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_forecast_vs_actual(
    actual: pd.Series,
    forecast: pd.Series,
    title: str = "Forecast vs Actual",
    index: pd.Index | None = None,
    benchmark: pd.Series | None = None,
) -> go.Figure:
    """Line chart comparing actual vs predicted values and optional benchmark."""
    idx = index if index is not None else actual.index
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=idx, y=actual.values, name="Actual",
                             line=dict(color="#F7931A")))
    fig.add_trace(go.Scatter(x=idx, y=forecast.values, name="Forecast",
                             line=dict(color="#4FC3F7", dash="dash")))
    if benchmark is not None:
        bidx = index if index is not None else benchmark.index
        fig.add_trace(go.Scatter(x=bidx, y=benchmark.values, name="Naive benchmark",
                                 line=dict(color="#9AA7B2", dash="dot")))
    fig.update_layout(title=title, template="plotly_dark",
                      yaxis_title="Value", xaxis_title="Time")
    return fig


def plot_residuals(
    residuals: pd.Series | np.ndarray,
    index: pd.Index | None = None,
    title: str = "Forecast residuals",
) -> go.Figure:
    """Residual time series and distribution for quick diagnostics."""
    values = pd.Series(residuals).dropna()
    x = index[: len(values)] if index is not None else np.arange(len(values))
    fig = make_subplots(rows=1, cols=2, column_widths=[0.65, 0.35],
                        subplot_titles=("Residuals over time", "Residual distribution"))
    fig.add_trace(go.Scatter(x=x, y=values.values, mode="lines",
                             name="Residual", line=dict(color="#EF5350")), row=1, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="#9AA7B2", row=1, col=1)
    fig.add_trace(go.Histogram(x=values.values, nbinsx=40, name="Distribution",
                               marker_color="#4FC3F7", opacity=0.8), row=1, col=2)
    fig.update_layout(title=title, template="plotly_dark", showlegend=False, height=360)
    return fig


def plot_prediction_scatter(
    actual: pd.Series | np.ndarray,
    predicted: pd.Series | np.ndarray,
    title: str = "Predicted vs actual",
) -> go.Figure:
    """Scatter plot with identity line for forecast calibration sanity check."""
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    mask = np.isfinite(a) & np.isfinite(p)
    a, p = a[mask], p[mask]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=a, y=p, mode="markers", name="Predictions",
                             marker=dict(color="#4FC3F7", size=5, opacity=0.6)))
    if a.size:
        lo, hi = float(min(a.min(), p.min())), float(max(a.max(), p.max()))
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                                 name="Ideal", line=dict(color="#9AA7B2", dash="dot")))
    fig.update_layout(title=title, template="plotly_dark",
                      xaxis_title="Actual", yaxis_title="Predicted", height=380)
    return fig


def plot_feature_importance(
    feature_names: list[str],
    importances: np.ndarray,
    top_n: int = 30,
    title: str = "Feature Importance",
) -> go.Figure:
    """Horizontal bar chart of model feature importances."""
    idx = np.argsort(importances)[-top_n:]
    fig = go.Figure(go.Bar(
        x=importances[idx],
        y=[feature_names[i] for i in idx],
        orientation="h",
        marker_color="#F7931A",
    ))
    fig.update_layout(title=title, template="plotly_dark",
                      xaxis_title="Importance", height=600)
    return fig
