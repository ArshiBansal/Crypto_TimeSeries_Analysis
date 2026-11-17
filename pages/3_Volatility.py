"""Volatility and drawdown diagnostics."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.dashboard import (
    ACCENT,
    BLUE,
    CORAL,
    PURPLE,
    ROOT,
    apply_app_style,
    chart_template,
    load_csv,
    load_dashboard_data,
    sample_for_chart,
)
from src.models.evaluator import volatility_metrics

st.set_page_config(page_title="Risk & Volatility", page_icon="~", layout="wide")
frame, timestamp_column = load_dashboard_data()
apply_app_style()
if frame is None:
    st.stop()

st.markdown(
    """
    <div class="hero">
      <div class="hero-kicker">03 / RISK LENS</div>
      <h1>Volatility has a rhythm.</h1>
      <p>Compare rolling feature estimates with exported conditional volatility.
         Treat these as descriptive/diagnostic measures, not guarantees.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

volatility_columns = [c for c in frame.columns if "rolling_vol" in c.lower()]
if volatility_columns:
    selected_columns = st.multiselect("Rolling volatility features", volatility_columns,
                                      default=volatility_columns[: min(3, len(volatility_columns))])
    chart_frame = sample_for_chart(frame, st.session_state.get("max_pts", 5000))
    if selected_columns:
        chart = go.Figure()
        colors = [ACCENT, BLUE, PURPLE, CORAL]
        for color, column in zip(colors * 4, selected_columns):
            chart.add_trace(go.Scatter(x=chart_frame[timestamp_column],
                                       y=pd.to_numeric(chart_frame[column], errors="coerce"),
                                       name=column, mode="lines", line=dict(color=color, width=1.8)))
        chart.update_layout(template=chart_template(), height=420, margin=dict(l=12, r=12, t=18, b=12),
                            hovermode="x unified", yaxis_title="Feature value")
        chart.update_xaxes(showgrid=False)
        st.plotly_chart(chart, use_container_width=True, config={"displaylogo": False})
else:
    st.info("No rolling-volatility features were found in this dataset.")

garch_path = ROOT / "artifacts" / "models" / "garch_cond_vol.csv"
if garch_path.exists():
    garch_frame = load_csv(str(garch_path))
    if {"timestamp", "cond_vol"}.issubset(garch_frame.columns):
        garch_frame["timestamp"] = pd.to_datetime(garch_frame["timestamp"], errors="coerce", utc=True)
        garch_frame["cond_vol"] = pd.to_numeric(garch_frame["cond_vol"], errors="coerce")
        garch_frame = garch_frame.dropna(subset=["timestamp", "cond_vol"])
        start, end = frame[timestamp_column].min(), frame[timestamp_column].max()
        garch_frame = garch_frame[garch_frame["timestamp"].between(start, end)]
        if not garch_frame.empty:
            gcf = sample_for_chart(garch_frame, st.session_state.get("max_pts", 5000))
            garch_chart = go.Figure(go.Scatter(x=gcf["timestamp"], y=gcf["cond_vol"], name="Conditional volatility",
                                               mode="lines", line=dict(color=BLUE, width=1.8),
                                               fill="tozeroy", fillcolor="rgba(131,169,212,0.10)"))
            garch_chart.update_layout(template=chart_template(), height=340, margin=dict(l=12, r=12, t=18, b=12),
                                      hovermode="x unified", yaxis_title="Conditional volatility")
            garch_chart.update_xaxes(showgrid=False)
            st.markdown("#### Exported conditional volatility (e.g. GARCH)")
            st.plotly_chart(garch_chart, use_container_width=True, config={"displaylogo": False})

            realized_col = next((c for c in ("rolling_vol_30", "realized_vol", "rv_30") if c in frame.columns), None)
            if realized_col is None:
                non_ann = [c for c in volatility_columns if "ann" not in c.lower()]
                realized_col = non_ann[0] if non_ann else None
            if realized_col:
                merged = pd.merge_asof(
                    garch_frame.sort_values("timestamp"),
                    frame[[timestamp_column, realized_col]].sort_values(timestamp_column).rename(columns={timestamp_column: "timestamp"}),
                    on="timestamp", direction="nearest", tolerance=pd.Timedelta("1D"),
                ).dropna(subset=["cond_vol", realized_col])
                if len(merged) >= 10:
                    pred = pd.to_numeric(merged["cond_vol"], errors="coerce").to_numpy(dtype=float)
                    truth = pd.to_numeric(merged[realized_col], errors="coerce").to_numpy(dtype=float)
                    pred_cmp = pred.copy()
                    if np.nanmedian(truth) > 0 and np.nanmedian(pred) > 0:
                        scale = np.nanmedian(truth) / np.nanmedian(pred)
                        if 0.2 < scale < 5:
                            pred_cmp = pred * scale
                    vm = volatility_metrics(truth, np.abs(pred_cmp))
                    cols = st.columns(3)
                    cols[0].metric("Realized proxy", realized_col)
                    cols[1].metric("MAE", f"{vm['mae']:.4g}" if np.isfinite(vm.get('mae', np.nan)) else "-")
                    cols[2].metric("QLIKE", f"{vm['qlike']:.4g}" if np.isfinite(vm.get('qlike', np.nan)) else "-",
                                   help="QLIKE is preferred for volatility forecast comparison on positive values.")
                    st.caption("Comparison is diagnostic only (proxy realized vol and possible scale/unit differences).")
        else:
            st.info("The exported volatility series does not overlap this selected date range.")
    else:
        st.error("Volatility export must contain `timestamp` and `cond_vol` columns.")
else:
    st.info("No conditional volatility export is available. Run notebook 03 to generate this series.")

price_column = next((c for c in ("close", "price", "adj_close") if c in frame.columns), None)
if price_column:
    close = pd.to_numeric(frame[price_column], errors="coerce")
    drawdown = close / close.cummax() - 1
    drawdown_frame = sample_for_chart(frame, st.session_state.get("max_pts", 5000))
    drawdown_chart = go.Figure(go.Scatter(
        x=drawdown_frame[timestamp_column], y=drawdown.loc[drawdown_frame.index] * 100,
        name="Drawdown", mode="lines", line=dict(color=CORAL, width=1.8),
        fill="tozeroy", fillcolor="rgba(217,142,135,0.12)",
    ))
    drawdown_chart.update_layout(template=chart_template(), height=300, margin=dict(l=12, r=12, t=18, b=12),
                                 hovermode="x unified", yaxis_title="Drawdown from window high (%)")
    drawdown_chart.update_xaxes(showgrid=False)
    st.markdown("#### Drawdown from the selected window high")
    st.plotly_chart(drawdown_chart, use_container_width=True, config={"displaylogo": False})

st.caption("Rolling volatility depends on bar interval and is not annualized unless named otherwise. "
           "For crypto, fat-tailed innovations (e.g. Student-t) are often more appropriate than normal GARCH assumptions.")
