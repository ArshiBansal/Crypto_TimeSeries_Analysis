"""Volatility and drawdown diagnostics."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.dashboard import (
    ACCENT,
    BLUE,
    CORAL,
    ROOT,
    apply_app_style,
    chart_template,
    load_dashboard_data,
    load_csv,
    sample_for_chart,
)


st.set_page_config(page_title="Risk & Volatility", page_icon="∿", layout="wide")
frame, timestamp_column = load_dashboard_data()
apply_app_style()

st.markdown(
    """
    <div class="hero">
      <div class="hero-kicker">03 &nbsp; / &nbsp; RISK LENS</div>
      <h1>Volatility has a rhythm.</h1>
      <p>Compare rolling feature estimates with exported GARCH conditional
         volatility. All measures describe the selected historical interval.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

volatility_columns = [
    column for column in frame.columns
    if "rolling_vol" in column.lower()
]

if volatility_columns:
    selected_columns = st.multiselect(
        "Rolling volatility features",
        volatility_columns,
        default=volatility_columns[: min(3, len(volatility_columns))],
    )
    chart_frame = sample_for_chart(
        frame, st.session_state.get("max_pts", 5000)
    )
    if selected_columns:
        chart = go.Figure()
        colors = [ACCENT, BLUE, "#A69AC8", CORAL]
        for color, column in zip(colors * 2, selected_columns):
            chart.add_trace(
                go.Scatter(
                    x=chart_frame[timestamp_column],
                    y=pd.to_numeric(chart_frame[column], errors="coerce"),
                    name=column,
                    mode="lines",
                    line=dict(color=color, width=1.8),
                )
            )
        chart.update_layout(
            template=chart_template(),
            height=440,
            margin=dict(l=12, r=12, t=18, b=12),
            hovermode="x unified",
            yaxis_title="Feature value",
        )
        chart.update_xaxes(showgrid=False)
        st.plotly_chart(chart, use_container_width=True, config={"displaylogo": False})
else:
    st.info("No rolling-volatility features were found in this dataset.")

garch_path = ROOT / "artifacts" / "models" / "garch_cond_vol.csv"
if garch_path.exists():
    garch_frame = load_csv(str(garch_path))
    required = {"timestamp", "cond_vol"}
    if required.issubset(garch_frame.columns):
        garch_frame["timestamp"] = pd.to_datetime(
            garch_frame["timestamp"], errors="coerce", utc=True
        )
        garch_frame["cond_vol"] = pd.to_numeric(
            garch_frame["cond_vol"], errors="coerce"
        )
        garch_frame = garch_frame.dropna(subset=["timestamp", "cond_vol"])
        start = frame[timestamp_column].min()
        end = frame[timestamp_column].max()
        garch_frame = garch_frame[
            garch_frame["timestamp"].between(start, end)
        ]
        if not garch_frame.empty:
            garch_chart_frame = sample_for_chart(
                garch_frame, st.session_state.get("max_pts", 5000)
            )
            garch_chart = go.Figure(
                go.Scatter(
                    x=garch_chart_frame["timestamp"],
                    y=garch_chart_frame["cond_vol"],
                    name="GARCH conditional volatility",
                    mode="lines",
                    line=dict(color=BLUE, width=1.8),
                    fill="tozeroy",
                    fillcolor="rgba(131,169,212,0.10)",
                )
            )
            garch_chart.update_layout(
                template=chart_template(),
                height=360,
                margin=dict(l=12, r=12, t=18, b=12),
                hovermode="x unified",
                yaxis_title="Conditional volatility",
            )
            garch_chart.update_xaxes(showgrid=False)
            st.markdown("#### GARCH(1,1) conditional volatility")
            st.plotly_chart(
                garch_chart,
                use_container_width=True,
                config={"displaylogo": False},
            )
        else:
            st.info("The exported GARCH series does not overlap this selected date range.")
    else:
        st.error("GARCH export must contain `timestamp` and `cond_vol` columns.")
else:
    st.info("No GARCH export is available. Run notebook 03 to generate this series.")

price_column = next(
    (column for column in ("close", "price", "adj_close") if column in frame.columns),
    None,
)
if price_column:
    close = pd.to_numeric(frame[price_column], errors="coerce")
    drawdown = close / close.cummax() - 1
    drawdown_frame = sample_for_chart(
        frame, st.session_state.get("max_pts", 5000)
    )
    drawdown_chart = go.Figure(
        go.Scatter(
            x=drawdown_frame[timestamp_column],
            y=drawdown.loc[drawdown_frame.index] * 100,
            name="Drawdown",
            mode="lines",
            line=dict(color=CORAL, width=1.8),
            fill="tozeroy",
            fillcolor="rgba(217,142,135,0.12)",
        )
    )
    drawdown_chart.update_layout(
        template=chart_template(),
        height=320,
        margin=dict(l=12, r=12, t=18, b=12),
        hovermode="x unified",
        yaxis_title="Drawdown from window high (%)",
    )
    drawdown_chart.update_xaxes(showgrid=False)
    st.markdown("#### Drawdown from the selected window high")
    st.plotly_chart(
        drawdown_chart,
        use_container_width=True,
        config={"displaylogo": False},
    )

st.caption(
    "Rolling volatility depends on the bar interval and is not annualized unless "
    "the feature name indicates otherwise."
)
