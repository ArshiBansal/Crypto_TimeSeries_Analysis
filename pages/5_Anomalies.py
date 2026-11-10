"""Explore exported Isolation Forest anomaly scores."""

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
)


st.set_page_config(page_title="Anomaly Review", page_icon="⌖", layout="wide")
frame, timestamp_column = load_dashboard_data()
apply_app_style()

st.markdown(
    """
    <div class="hero">
      <div class="hero-kicker">05 &nbsp; / &nbsp; OUTLIER REVIEW</div>
      <h1>Find the bars that stand apart.</h1>
      <p>Review saved Isolation Forest scores against the current date window.
         Lower scores indicate more unusual observations in the exported model.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

scores_path = ROOT / "artifacts" / "models" / "anomaly_scores.csv"
if not scores_path.exists():
    st.info("No anomaly score export is available. Run notebook 03 to generate it.")
    st.stop()

scores = pd.read_csv(scores_path)
required = {timestamp_column, "score"}
if not required.issubset(scores.columns):
    st.error(
        "The anomaly score export must contain "
        f"`{timestamp_column}` and `score` columns."
    )
    st.stop()

scores[timestamp_column] = pd.to_datetime(
    scores[timestamp_column], errors="coerce", utc=True
)
scores["score"] = pd.to_numeric(scores["score"], errors="coerce")
scores = scores.dropna(subset=[timestamp_column, "score"])
start = frame[timestamp_column].min()
end = frame[timestamp_column].max()
scores = scores[scores[timestamp_column].between(start, end)].sort_values(timestamp_column)
if scores.empty:
    st.warning("No anomaly scores overlap the selected date range.")
    st.stop()

quantile = st.slider(
    "Flag the lowest score percentile",
    min_value=0.5,
    max_value=20.0,
    value=5.0,
    step=0.5,
    help="Isolation Forest scores near the lower tail are treated as unusual.",
)
threshold = float(scores["score"].quantile(quantile / 100))
scores["flagged"] = scores["score"] <= threshold
flagged = scores[scores["flagged"]]

summary = st.columns(4)
summary[0].metric("Scored bars", f"{len(scores):,}")
summary[1].metric("Flagged", f"{len(flagged):,}")
summary[2].metric("Share flagged", f"{len(flagged) / len(scores):.2%}")
summary[3].metric("Score threshold", f"{threshold:.4g}")

score_chart = go.Figure()
score_chart.add_trace(
    go.Scattergl(
        x=scores[timestamp_column],
        y=scores["score"],
        name="Score",
        mode="lines",
        line=dict(color=BLUE, width=1),
        opacity=0.8,
    )
)
score_chart.add_trace(
    go.Scatter(
        x=flagged[timestamp_column],
        y=flagged["score"],
        name="Flagged observations",
        mode="markers",
        marker=dict(color=CORAL, size=7, symbol="x"),
    )
)
score_chart.add_hline(
    y=threshold,
    line_dash="dot",
    line_color=ACCENT,
    annotation_text=f"{quantile:.1f}th percentile",
)
score_chart.update_layout(
    template=chart_template(),
    height=430,
    margin=dict(l=12, r=12, t=25, b=12),
    hovermode="x unified",
    yaxis_title="Isolation Forest score",
)
score_chart.update_xaxes(showgrid=False)
st.plotly_chart(
    score_chart,
    use_container_width=True,
    config={"displaylogo": False},
)

left, right = st.columns([1, 1], gap="large")
with left:
    score_histogram = go.Figure()
    score_histogram.add_trace(
        go.Histogram(
            x=scores["score"],
            nbinsx=60,
            name="All scores",
            marker_color=BLUE,
            opacity=0.72,
        )
    )
    if not flagged.empty:
        score_histogram.add_trace(
            go.Histogram(
                x=flagged["score"],
                nbinsx=60,
                name="Flagged",
                marker_color=CORAL,
                opacity=0.82,
            )
        )
    score_histogram.update_layout(
        template=chart_template(),
        barmode="overlay",
        height=300,
        margin=dict(l=12, r=12, t=15, b=12),
        xaxis_title="Score",
        yaxis_title="Observations",
    )
    st.plotly_chart(
        score_histogram,
        use_container_width=True,
        config={"displaylogo": False},
    )

with right:
    if "is_anomaly" in scores.columns:
        exported = pd.to_numeric(scores["is_anomaly"], errors="coerce").fillna(0).astype(bool)
        exported_count = int(exported.sum())
        st.metric("Notebook-exported flags", f"{exported_count:,}")
        st.caption(
            "The slider-based flags are recalculated from the score distribution "
            "in this date window. Exported flags retain the notebook threshold."
        )
    else:
        st.info("This export contains scores only; flags are derived from the selected percentile.")

if not flagged.empty:
    st.markdown("#### Lowest-scoring observations")
    st.dataframe(
        flagged.nsmallest(100, "score").sort_values(timestamp_column, ascending=False),
        use_container_width=True,
        hide_index=True,
    )

st.download_button(
    "Download flagged observations",
    data=flagged.to_csv(index=False).encode("utf-8"),
    file_name="btc_flagged_observations.csv",
    mime="text/csv",
)
st.caption(
    "Anomaly flags indicate statistical unusualness under one fitted model; "
    "they are not evidence of an error or a future event."
)
