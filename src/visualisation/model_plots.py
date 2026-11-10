"""
src/visualisation/model_plots.py

Model evaluation and forecast plots.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
import plotly.graph_objects as go


def plot_forecast_vs_actual(
    actual: pd.Series,
    forecast: pd.Series,
    title: str = "Forecast vs Actual",
    index: pd.Index | None = None,
) -> go.Figure:
    """Line chart comparing actual vs predicted values."""
    idx = index if index is not None else actual.index
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=idx, y=actual.values, name="Actual",
                             line=dict(color="#F7931A")))
    fig.add_trace(go.Scatter(x=idx, y=forecast.values, name="Forecast",
                             line=dict(color="#4FC3F7", dash="dash")))
    fig.update_layout(title=title, template="plotly_dark",
                      yaxis_title="Value", xaxis_title="Time")
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
