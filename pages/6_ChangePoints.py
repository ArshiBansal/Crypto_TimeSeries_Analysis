"""Explore exported CUSUM change points in the selected market window."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

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


st.set_page_config(page_title="Change Points", page_icon="⌁", layout="wide")
frame, timestamp_column = load_dashboard_data()
apply_app_style()

st.markdown(
    """
    <div class="hero">
      <div class="hero-kicker">06 &nbsp; / &nbsp; STRUCTURAL BREAKS</div>
      <h1>Mark the moments the texture shifted.</h1>
      <p>Inspect exported CUSUM breakpoints against the selected price and
         volatility history. Markers retain their exported timestamps.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

change_points_path = ROOT / "artifacts" / "models" / "changepoints.csv"
if not change_points_path.exists():
    st.info(
        "No change-point export is available. Run the CUSUM section in "
        "notebook 03 to generate `artifacts/models/changepoints.csv`."
    )
    st.stop()

try:
    change_points = load_csv(str(change_points_path))
except pd.errors.EmptyDataError:
    st.warning("The change-point export is empty. Run the CUSUM section in notebook 03.")
    st.stop()
except pd.errors.ParserError as error:
    st.error("Could not parse the change-point export.")
    st.exception(error)
    st.stop()

required_columns = {"timestamp"}
if not required_columns.issubset(change_points.columns):
    st.error("The change-point export must contain a `timestamp` column.")
    st.stop()

change_points["timestamp"] = pd.to_datetime(
    change_points["timestamp"], errors="coerce", utc=True
)
change_points = change_points.dropna(subset=["timestamp"]).drop_duplicates(
    subset=["timestamp"]
)
start = frame[timestamp_column].min()
end = frame[timestamp_column].max()
visible_points = change_points[
    change_points["timestamp"].between(start, end)
].sort_values("timestamp")

metrics = st.columns(3)
metrics[0].metric("Exported breakpoints", f"{len(change_points):,}")
metrics[1].metric("In selected window", f"{len(visible_points):,}")
metrics[2].metric(
    "Window",
    f"{start.strftime('%d %b %Y')} – {end.strftime('%d %b %Y')}",
)

if visible_points.empty:
    st.info("No exported breakpoints overlap the selected date range.")
    st.stop()

volatility_columns = [
    column
    for column in frame.columns
    if "rolling_vol" in column.lower() and "ann" not in column.lower()
]
if volatility_columns:
    volatility_column = st.selectbox(
        "Volatility context",
        volatility_columns,
        help="Choose a rolling-volatility feature to show beneath the price series.",
    )
else:
    volatility_column = next(
        (column for column in ("log_return", "simple_return") if column in frame.columns),
        None,
    )

price_column = next(
    (column for column in ("close", "price", "adj_close") if column in frame.columns),
    None,
)
if price_column is None and volatility_column is None:
    st.warning("This dataset has no price or volatility/return column to plot.")
    st.stop()

chart_frame = sample_for_chart(frame, st.session_state.get("max_pts", 5000))
rows = 2 if price_column is not None and volatility_column is not None else 1
figure = make_subplots(
    rows=rows,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.08,
    row_heights=[0.68, 0.32] if rows == 2 else [1],
)

if price_column is not None:
    figure.add_trace(
        go.Scatter(
            x=chart_frame[timestamp_column],
            y=pd.to_numeric(chart_frame[price_column], errors="coerce"),
            name=price_column,
            mode="lines",
            line=dict(color=ACCENT, width=1.8),
            connectgaps=False,
        ),
        row=1,
        col=1,
    )
    figure.update_yaxes(title_text=price_column, row=1, col=1)

if volatility_column is not None:
    volatility_row = 2 if rows == 2 else 1
    figure.add_trace(
        go.Scatter(
            x=chart_frame[timestamp_column],
            y=pd.to_numeric(chart_frame[volatility_column], errors="coerce"),
            name=volatility_column,
            mode="lines",
            line=dict(color=BLUE if rows == 2 else CORAL, width=1.6),
            connectgaps=False,
        ),
        row=volatility_row,
        col=1,
    )
    figure.update_yaxes(title_text=volatility_column, row=volatility_row, col=1)

for point in visible_points["timestamp"]:
    figure.add_vline(
        x=point,
        line_color=CORAL,
        line_dash="dot",
        line_width=1.5,
        opacity=0.85,
    )

figure.update_layout(
    template=chart_template(),
    height=560 if rows == 2 else 430,
    margin=dict(l=12, r=12, t=22, b=12),
    hovermode="x unified",
    legend=dict(orientation="h", y=1.04, x=0),
)
figure.update_xaxes(showgrid=False)
grid_color = "#E0E7EB" if st.session_state.get("light_mode", False) else "#293743"
figure.update_yaxes(gridcolor=grid_color)
st.plotly_chart(
    figure,
    use_container_width=True,
    config={"displaylogo": False, "scrollZoom": True},
)

st.markdown("#### Breakpoint dates")
display_columns = ["timestamp"]
if "index" in visible_points.columns:
    display_columns.append("index")
st.dataframe(
    visible_points[display_columns].assign(
        timestamp=visible_points["timestamp"].dt.strftime("%Y-%m-%d %H:%M UTC")
    ),
    use_container_width=True,
    hide_index=True,
)
st.download_button(
    "Download visible breakpoints",
    data=visible_points.to_csv(index=False).encode("utf-8"),
    file_name="btc_change_points_in_view.csv",
    mime="text/csv",
)
st.caption(
    "Breakpoints were detected on the training partition. This view filters "
    "the exported timestamps to the selected window; it does not refit detection "
    "on the current selection."
)
