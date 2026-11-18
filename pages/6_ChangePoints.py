"""Explore exported CUSUM change points with enhanced visualizations."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy.stats import norm

from src.dashboard import ACCENT, BLUE, CORAL, PURPLE, ROOT, apply_app_style, chart_template, load_csv, load_dashboard_data, sample_for_chart

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
         Breakpoint counts depend on penalty/model choices—interpret as sensitivity-aware signals.</p>
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

tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Price & Volatility", "🔍 Pre/Post Analysis", "📋 Breakpoint Details"])

with tab1:
    metrics = st.columns(4)
    metrics[0].metric("Exported breakpoints", f"{len(change_points):,}")
    metrics[1].metric("In selected window", f"{len(visible_points):,}")
    metrics[2].metric("Window", f"{start.strftime('%d %b %Y')} – {end.strftime('%d %b %Y')}")
    if len(visible_points) > 1:
        avg_interval = (visible_points[cp_ts_col].diff().dt.total_seconds() / 3600).mean()
        metrics[3].metric("Avg interval (hrs)", f"{avg_interval:.1f}")
    else:
        metrics[3].metric("Avg interval (hrs)", "—")

    if visible_points.empty:
        st.info("No exported breakpoints overlap the selected date range.")
        st.stop()

    st.markdown("#### Breakpoint Timeline")
    tl_fig = go.Figure()
    tl_fig.add_trace(go.Scatter(x=visible_points[cp_ts_col], y=[1] * len(visible_points),
                                mode="markers", name="Breakpoints",
                                marker=dict(color=CORAL, size=12, symbol="line-ns-open",
                                           line=dict(width=2, color="white")),
                                hovertemplate="%{x}<br>Breakpoint<extra></extra>"))
    tl_fig.update_layout(template=chart_template(), height=150,
                         margin=dict(l=12, r=12, t=25, b=12),
                         yaxis=dict(visible=False), showlegend=False,
                         hovermode="x unified")
    tl_fig.update_xaxes(showgrid=False)
    st.plotly_chart(tl_fig, use_container_width=True, config={"displaylogo": False})

    st.markdown("#### Breakpoint Frequency Over Time")
    if len(visible_points) > 1:
        freq_fig = go.Figure()
        freq_fig.add_trace(go.Histogram(x=visible_points[cp_ts_col].dt.date,
                                        nbinsx=min(30, len(visible_points) * 2),
                                        marker_color=BLUE, opacity=0.7))
        freq_fig.update_layout(template=chart_template(), height=300,
                               margin=dict(l=12, r=12, t=25, b=12),
                               xaxis_title="Date", yaxis_title="Breakpoints per day")
        freq_fig.update_xaxes(showgrid=False)
        st.plotly_chart(freq_fig, use_container_width=True, config={"displaylogo": False})

with tab2:
    if visible_points.empty:
        st.stop()

    volatility_columns = [c for c in frame.columns if "rolling_vol" in c.lower() and "ann" not in c.lower()]
    if volatility_columns:
        volatility_column = st.selectbox("Volatility context", volatility_columns,
                                         help="Choose a rolling-volatility feature to show beneath the price series.")
    else:
        volatility_column = next((c for c in ("log_return", "simple_return") if c in frame.columns), None)

    price_column = next((c for c in ("close", "price", "adj_close") if c in frame.columns), None)
    if price_column is None and volatility_column is None:
        st.warning("This dataset has no price or volatility/return column to plot.")
        st.stop()

    chart_frame = sample_for_chart(frame, st.session_state.get("max_pts", 5000))
    rows = 2 if price_column is not None and volatility_column is not None else 1
    figure = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                           row_heights=[0.68, 0.32] if rows == 2 else [1],
                           subplot_titles=("Price", volatility_column) if rows == 2 else (volatility_column or "Price",))

    if price_column is not None:
        figure.add_trace(go.Scatter(x=chart_frame[timestamp_column],
                                    y=pd.to_numeric(chart_frame[price_column], errors="coerce"),
                                    name=price_column, mode="lines",
                                    line=dict(color=ACCENT, width=1.8), connectgaps=False), row=1, col=1)
        figure.update_yaxes(title_text=price_column, row=1, col=1)
    if volatility_column is not None:
        r = 2 if rows == 2 else 1
        figure.add_trace(go.Scatter(x=chart_frame[timestamp_column],
                                    y=pd.to_numeric(chart_frame[volatility_column], errors="coerce"),
                                    name=volatility_column, mode="lines",
                                    line=dict(color=BLUE if rows == 2 else CORAL, width=1.6), connectgaps=False), row=r, col=1)
        figure.update_yaxes(title_text=volatility_column, row=r, col=1)

    for point in visible_points[cp_ts_col]:
        figure.add_vline(x=point, line_color=CORAL, line_dash="dot", line_width=1.5, opacity=0.85)

    figure.update_layout(template=chart_template(), height=540 if rows == 2 else 420,
                         margin=dict(l=12, r=12, t=40, b=12),
                         hovermode="x unified", legend=dict(orientation="h", y=1.04, x=0))
    figure.update_xaxes(showgrid=False)
    figure.update_yaxes(gridcolor="#293743")
    st.plotly_chart(figure, use_container_width=True, config={"displaylogo": False, "scrollZoom": True})

    if price_column and volatility_column:
        st.markdown("#### Breakpoint Return/Volatility Scatter")
        price_data = frame[[timestamp_column, price_column]].copy()
        price_data[timestamp_column] = pd.to_datetime(price_data[timestamp_column], errors="coerce", utc=True)
        price_data[price_column] = pd.to_numeric(price_data[price_column], errors="coerce")
        price_data = price_data.dropna().sort_values(timestamp_column)
        price_data["return"] = price_data[price_column].pct_change() * 100

        vol_data = frame[[timestamp_column, volatility_column]].copy()
        vol_data[timestamp_column] = pd.to_datetime(vol_data[timestamp_column], errors="coerce", utc=True)
        vol_data[volatility_column] = pd.to_numeric(vol_data[volatility_column], errors="coerce")
        vol_data = vol_data.dropna().sort_values(timestamp_column)

        merged = pd.merge_asof(price_data, vol_data, on=timestamp_column, direction="nearest", tolerance=pd.Timedelta("1D"))
        merged = merged.dropna()

        bp_data = []
        for cp in visible_points[cp_ts_col]:
            nearest = merged.iloc[(merged[timestamp_column] - cp).abs().argsort()[:1]]
            if len(nearest) > 0:
                bp_data.append({
                    "breakpoint": cp,
                    "return": nearest["return"].values[0],
                    "volatility": nearest[volatility_column].values[0]
                })
        
        if bp_data:
            bp_df = pd.DataFrame(bp_data)
            scatter_fig = go.Figure()
            scatter_fig.add_trace(go.Scatter(x=bp_df["return"], y=bp_df["volatility"],
                                             mode="markers+text", name="Breakpoints",
                                             marker=dict(color=CORAL, size=10, symbol="diamond",
                                                        line=dict(width=1, color="white")),
                                             text=[pd.Timestamp(cp).strftime("%m-%d") for cp in bp_df["breakpoint"]],
                                             textposition="top center",
                                             hovertemplate="%{text}<br>Return: %{x:.2f}%<br>Vol: %{y:.4f}<extra></extra>"))
            scatter_fig.add_hline(y=merged[volatility_column].mean(), line_dash="dash", line_color="#9AA7B2")
            scatter_fig.add_vline(x=0, line_dash="dash", line_color="#9AA7B2")
            scatter_fig.update_layout(template=chart_template(), height=350,
                                     margin=dict(l=12, r=12, t=25, b=12),
                                     xaxis_title="Return (%)", yaxis_title="Volatility")
            st.plotly_chart(scatter_fig, use_container_width=True, config={"displaylogo": False})

with tab3:
    summary_col = volatility_column or next((c for c in ("log_return", "simple_return") if c in frame.columns), None)
    if summary_col:
        st.markdown("#### Local Pre/Post Summary")
        series = frame[[timestamp_column, summary_col]].copy()
        series[timestamp_column] = pd.to_datetime(series[timestamp_column], errors="coerce", utc=True)
        series[summary_col] = pd.to_numeric(series[summary_col], errors="coerce")
        series = series.dropna().sort_values(timestamp_column)

        points = visible_points[cp_ts_col].sort_values().tolist()
        summaries = []
        for i, pt in enumerate(points[:30]):
            prev_pt = points[i - 1] if i > 0 else series[timestamp_column].min()
            next_pt = points[i + 1] if i + 1 < len(points) else series[timestamp_column].max()
            pre = series[(series[timestamp_column] > prev_pt) & (series[timestamp_column] <= pt)][summary_col]
            post = series[(series[timestamp_column] > pt) & (series[timestamp_column] <= next_pt)][summary_col]
            if len(pre) >= 3 and len(post) >= 3:
                from scipy.stats import ttest_ind, mannwhitneyu
                t_stat, t_p = ttest_ind(pre, post, equal_var=False)
                u_stat, u_p = mannwhitneyu(pre, post, alternative="two-sided")
                summaries.append({
                    "breakpoint": pd.Timestamp(pt).strftime("%Y-%m-%d %H:%M UTC"),
                    "pre_mean": float(pre.mean()), "post_mean": float(post.mean()),
                    "mean_shift": float(post.mean() - pre.mean()),
                    "pre_std": float(pre.std()), "post_std": float(post.std()),
                    "shift_pct": float((post.mean() - pre.mean()) / abs(pre.mean()) * 100) if pre.mean() != 0 else 0,
                    "t_pvalue": float(t_p), "mw_pvalue": float(u_p),
                    "pre_n": int(len(pre)), "post_n": int(len(post))
                })

        if summaries:
            sum_df = pd.DataFrame(summaries)
            st.dataframe(sum_df, use_container_width=True, hide_index=True)
            
            st.markdown("#### Shift Magnitude Visualization")
            shift_fig = go.Figure()
            colors = [CORAL if x > 0 else BLUE for x in sum_df["mean_shift"]]
            shift_fig.add_trace(go.Bar(x=sum_df["breakpoint"], y=sum_df["mean_shift"],
                                       marker_color=colors,
                                       text=[f"{x:+.3f}" for x in sum_df["mean_shift"]],
                                       textposition="outside",
                                       hovertemplate="%{x}<br>Shift: %{y:.4f}<br>t-test p: %{customdata[0]:.3f}<br>MW p: %{customdata[1]:.3f}<extra></extra>",
                                       customdata=list(zip(sum_df["t_pvalue"], sum_df["mw_pvalue"]))))
            shift_fig.add_hline(y=0, line_dash="dash", line_color="#9AA7B2")
            shift_fig.update_layout(template=chart_template(), height=350,
                                   margin=dict(l=12, r=12, t=25, b=12),
                                   xaxis_title="Breakpoint", yaxis_title="Mean Shift")
            shift_fig.update_xaxes(tickangle=45)
            st.plotly_chart(shift_fig, use_container_width=True, config={"displaylogo": False})
            
            st.caption("Pre/post windows are segments between consecutive exported breakpoints; interpret shifts cautiously. "
                       "t-test and Mann-Whitney p-values indicate statistical significance of the shift.")
        else:
            st.info("Not enough data around visible breakpoints to compute pre/post summaries.")
    else:
        st.info("No return or volatility column available for pre/post analysis.")

with tab4:
    st.markdown("#### Breakpoint Dates & Metadata")
    display_columns = [cp_ts_col] + [c for c in ["index", "penalty", "model", "score"] if c in visible_points.columns]
    table = visible_points[display_columns].copy()
    table[cp_ts_col] = table[cp_ts_col].dt.strftime("%Y-%m-%d %H:%M UTC")
    st.dataframe(table, use_container_width=True, hide_index=True)
    
    st.download_button("Download visible breakpoints", data=visible_points.to_csv(index=False).encode("utf-8"),
                       file_name="btc_change_points_in_view.csv", mime="text/csv")
    
    if "penalty" in visible_points.columns:
        st.markdown("#### Penalty Distribution")
        pen_fig = go.Figure(go.Histogram(x=visible_points["penalty"], nbinsx=20,
                                          marker_color=PURPLE, opacity=0.7))
        pen_fig.update_layout(template=chart_template(), height=250,
                             margin=dict(l=12, r=12, t=25, b=12),
                             xaxis_title="Penalty", yaxis_title="Count")
        st.plotly_chart(pen_fig, use_container_width=True, config={"displaylogo": False})
    
    if "model" in visible_points.columns:
        st.markdown("#### Model Distribution")
        model_counts = visible_points["model"].value_counts()
        model_fig = go.Figure(go.Bar(x=model_counts.index, y=model_counts.values,
                                      marker_color=ACCENT, text=model_counts.values,
                                      textposition="outside"))
        model_fig.update_layout(template=chart_template(), height=250,
                               margin=dict(l=12, r=12, t=25, b=12),
                               xaxis_title="Model", yaxis_title="Count")
        st.plotly_chart(model_fig, use_container_width=True, config={"displaylogo": False})

st.caption("Breakpoints were detected on an exported run/partition. This view filters timestamps to the selected window "
           "and does not refit detection or test penalty sensitivity.")