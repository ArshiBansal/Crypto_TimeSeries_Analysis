"""Explore exported CUSUM change points in the selected market window."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.dashboard import ACCENT, BLUE, CORAL, ROOT, apply_app_style, chart_template, load_csv, load_dashboard_data, sample_for_chart

st.set_page_config(page_title="Change Points", page_icon="#", layout="wide")
frame, timestamp_column = load_dashboard_data()
apply_app_style()
if frame is None:
    st.stop()

st.markdown(
    """
    <div class="hero">
      <div class="hero-kicker">06 / STRUCTURAL BREAKS</div>
      <h1>Mark the moments the texture shifted.</h1>
      <p>Inspect exported change-point breakpoints against selected price and volatility history.
         Breakpoint counts depend on penalty/model choices-interpret as sensitivity-aware signals.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

change_points_path = ROOT / "artifacts" / "models" / "changepoints.csv"
if not change_points_path.exists():
    st.info("No change-point export is available. Run the CUSUM section in notebook 03 to generate `artifacts/models/changepoints.csv`.")
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

cp_ts_col = next((c for c in [timestamp_column, "timestamp", "datetime", "date", "time"] if c in change_points.columns), None)
if cp_ts_col is None:
    st.error("The change-point export must contain a recognizable timestamp column.")
    st.stop()

change_points[cp_ts_col] = pd.to_datetime(change_points[cp_ts_col], errors="coerce", utc=True)
change_points = change_points.dropna(subset=[cp_ts_col]).drop_duplicates(subset=[cp_ts_col])
start, end = frame[timestamp_column].min(), frame[timestamp_column].max()
visible_points = change_points[change_points[cp_ts_col].between(start, end)].sort_values(cp_ts_col)

metrics = st.columns(3)
metrics[0].metric("Exported breakpoints", f"{len(change_points):,}")
metrics[1].metric("In selected window", f"{len(visible_points):,}")
metrics[2].metric("Window", f"{start.strftime('%d %b %Y')} - {end.strftime('%d %b %Y')}")

if visible_points.empty:
    st.info("No exported breakpoints overlap the selected date range.")
    st.stop()

volatility_columns = [c for c in frame.columns if "rolling_vol" in c.lower() and "ann" not in c.lower()]
if volatility_columns:
    volatility_column = st.selectbox("Volatility context", volatility_columns, help="Choose a rolling-volatility feature to show beneath the price series.")
else:
    volatility_column = next((c for c in ("log_return", "simple_return") if c in frame.columns), None)

price_column = next((c for c in ("close", "price", "adj_close") if c in frame.columns), None)
if price_column is None and volatility_column is None:
    st.warning("This dataset has no price or volatility/return column to plot.")
    st.stop()

chart_frame = sample_for_chart(frame, st.session_state.get("max_pts", 5000))
rows = 2 if price_column is not None and volatility_column is not None else 1
figure = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.68, 0.32] if rows == 2 else [1])

if price_column is not None:
    figure.add_trace(go.Scatter(x=chart_frame[timestamp_column], y=pd.to_numeric(chart_frame[price_column], errors="coerce"),
                                name=price_column, mode="lines", line=dict(color=ACCENT, width=1.8), connectgaps=False), row=1, col=1)
    figure.update_yaxes(title_text=price_column, row=1, col=1)
if volatility_column is not None:
    r = 2 if rows == 2 else 1
    figure.add_trace(go.Scatter(x=chart_frame[timestamp_column], y=pd.to_numeric(chart_frame[volatility_column], errors="coerce"),
                                name=volatility_column, mode="lines", line=dict(color=BLUE if rows == 2 else CORAL, width=1.6), connectgaps=False), row=r, col=1)
    figure.update_yaxes(title_text=volatility_column, row=r, col=1)
for point in visible_points[cp_ts_col]:
    figure.add_vline(x=point, line_color=CORAL, line_dash="dot", line_width=1.5, opacity=0.85)

figure.update_layout(template=chart_template(), height=540 if rows == 2 else 420, margin=dict(l=12, r=12, t=22, b=12),
                     hovermode="x unified", legend=dict(orientation="h", y=1.04, x=0))
figure.update_xaxes(showgrid=False)
figure.update_yaxes(gridcolor="#293743")
st.plotly_chart(figure, use_container_width=True, config={"displaylogo": False, "scrollZoom": True})

summary_col = volatility_column or next((c for c in ("log_return", "simple_return") if c in frame.columns), None)
if summary_col:
    st.markdown("#### Local pre/post summary")
    series = frame[[timestamp_column, summary_col]].copy()
    series[timestamp_column] = pd.to_datetime(series[timestamp_column], errors="coerce", utc=True)
    series[summary_col] = pd.to_numeric(series[summary_col], errors="coerce")
    series = series.dropna().sort_values(timestamp_column)
    summaries = []
    points = visible_points[cp_ts_col].sort_values().tolist()
    for i, pt in enumerate(points[:20]):
        prev_pt = points[i - 1] if i > 0 else series[timestamp_column].min()
        next_pt = points[i + 1] if i + 1 < len(points) else series[timestamp_column].max()
        pre = series[(series[timestamp_column] > prev_pt) & (series[timestamp_column] <= pt)][summary_col]
        post = series[(series[timestamp_column] > pt) & (series[timestamp_column] <= next_pt)][summary_col]
        if len(pre) >= 3 and len(post) >= 3:
            summaries.append({"breakpoint": pd.Timestamp(pt).strftime("%Y-%m-%d %H:%M UTC"),
                              "pre_mean": float(pre.mean()), "post_mean": float(post.mean()), "mean_shift": float(post.mean() - pre.mean()),
                              "pre_std": float(pre.std()), "post_std": float(post.std()), "pre_n": int(len(pre)), "post_n": int(len(post))})
    if summaries:
        st.dataframe(pd.DataFrame(summaries), use_container_width=True, hide_index=True)
        st.caption("Pre/post windows are segments between consecutive exported breakpoints; interpret shifts cautiously.")
    else:
        st.info("Not enough data around visible breakpoints to compute pre/post summaries.")

st.markdown("#### Breakpoint dates")
display_columns = [cp_ts_col] + [c for c in ["index", "penalty", "model"] if c in visible_points.columns]
table = visible_points[display_columns].copy()
table[cp_ts_col] = table[cp_ts_col].dt.strftime("%Y-%m-%d %H:%M UTC")
st.dataframe(table, use_container_width=True, hide_index=True)
st.download_button("Download visible breakpoints", data=visible_points.to_csv(index=False).encode("utf-8"),
                   file_name="btc_change_points_in_view.csv", mime="text/csv")
st.caption("Breakpoints were detected on an exported run/partition. This view filters timestamps to the selected window and does not refit detection or test penalty sensitivity.")
