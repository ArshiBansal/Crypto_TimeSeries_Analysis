"""Explore exported Isolation Forest anomaly scores."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.dashboard import ACCENT, BLUE, CORAL, ROOT, apply_app_style, chart_template, load_csv, load_dashboard_data, sample_for_chart

st.set_page_config(page_title="Anomaly Review", page_icon="!", layout="wide")
frame, timestamp_column = load_dashboard_data()
apply_app_style()
if frame is None:
    st.stop()

st.markdown(
    """
    <div class="hero">
      <div class="hero-kicker">05 / OUTLIER REVIEW</div>
      <h1>Find the bars that stand apart.</h1>
      <p>Review saved Isolation Forest scores against the current date window.
         Flags indicate statistical unusualness under one fitted model-not errors or future events.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

scores_path = ROOT / "artifacts" / "models" / "anomaly_scores.csv"
if not scores_path.exists():
    st.info("No anomaly score export is available. Run notebook 03 to generate it.")
    st.stop()

scores = load_csv(str(scores_path))
ts_col = next((c for c in [timestamp_column, "timestamp", "datetime", "date", "time"] if c in scores.columns), None)
if ts_col is None or "score" not in scores.columns:
    st.error("The anomaly score export must contain a timestamp column and `score`.")
    st.stop()

scores[ts_col] = pd.to_datetime(scores[ts_col], errors="coerce", utc=True)
scores["score"] = pd.to_numeric(scores["score"], errors="coerce")
scores = scores.dropna(subset=[ts_col, "score"])
start, end = frame[timestamp_column].min(), frame[timestamp_column].max()
scores = scores[scores[ts_col].between(start, end)].sort_values(ts_col)
if scores.empty:
    st.warning("No anomaly scores overlap the selected date range.")
    st.stop()

quantile = st.slider("Flag the lowest score percentile", 0.5, 20.0, 5.0, 0.5,
                     help="Lower Isolation Forest scores are treated as more unusual; validate sensitivity to this threshold.")
threshold = float(scores["score"].quantile(quantile / 100))
scores["flagged"] = scores["score"] <= threshold
flagged = scores[scores["flagged"]]

sens = [{"percentile": q, "threshold": float(scores['score'].quantile(q/100)),
         "flagged": int((scores['score'] <= scores['score'].quantile(q/100)).sum()),
         "share": float((scores['score'] <= scores['score'].quantile(q/100)).mean())} for q in [1, 2, 5, 10]]

summary = st.columns(4)
summary[0].metric("Scored bars", f"{len(scores):,}")
summary[1].metric("Flagged", f"{len(flagged):,}")
summary[2].metric("Share flagged", f"{len(flagged) / len(scores):.2%}")
summary[3].metric("Score threshold", f"{threshold:.4g}")

chart_scores = sample_for_chart(scores, st.session_state.get("max_pts", 5000))
chart_flagged = sample_for_chart(flagged, st.session_state.get("max_pts", 5000))
score_chart = go.Figure()
score_chart.add_trace(go.Scattergl(x=chart_scores[ts_col], y=chart_scores["score"], name="Score (sampled)",
                                   mode="lines", line=dict(color=BLUE, width=1), opacity=0.8))
score_chart.add_trace(go.Scatter(x=chart_flagged[ts_col], y=chart_flagged["score"], name="Flagged (sampled)",
                                 mode="markers", marker=dict(color=CORAL, size=7, symbol="x")))
score_chart.add_hline(y=threshold, line_dash="dot", line_color=ACCENT, annotation_text=f"{quantile:.1f}th percentile")
score_chart.update_layout(template=chart_template(), height=380, margin=dict(l=12, r=12, t=25, b=12),
                          hovermode="x unified", yaxis_title="Isolation Forest score")
score_chart.update_xaxes(showgrid=False)
st.plotly_chart(score_chart, use_container_width=True, config={"displaylogo": False})

price_col = next((c for c in ("close", "price", "adj_close") if c in frame.columns), None)
ret_col = next((c for c in ("log_return", "simple_return") if c in frame.columns), None)
overlay_col = price_col or ret_col
if overlay_col:
    overlay = frame[[timestamp_column, overlay_col]].copy()
    overlay[timestamp_column] = pd.to_datetime(overlay[timestamp_column], errors="coerce", utc=True)
    overlay[overlay_col] = pd.to_numeric(overlay[overlay_col], errors="coerce")
    overlay = overlay.dropna().sort_values(timestamp_column)
    merged_flags = pd.merge_asof(flagged.sort_values(ts_col).rename(columns={ts_col: "ts"}),
                                 overlay.rename(columns={timestamp_column: "ts"}), on="ts", direction="nearest", tolerance=pd.Timedelta("1D"))
    ov_sample = sample_for_chart(overlay, st.session_state.get("max_pts", 5000))
    mf_sample = sample_for_chart(merged_flags.dropna(subset=["ts", overlay_col]), st.session_state.get("max_pts", 5000))
    ov = go.Figure()
    ov.add_trace(go.Scatter(x=ov_sample[timestamp_column], y=ov_sample[overlay_col], name=overlay_col, mode="lines", line=dict(color=ACCENT, width=1.5)))
    if not mf_sample.empty:
        ov.add_trace(go.Scatter(x=mf_sample["ts"], y=mf_sample[overlay_col], name="Flagged bars", mode="markers", marker=dict(color=CORAL, size=7, symbol="x")))
    ov.update_layout(template=chart_template(), height=320, margin=dict(l=12, r=12, t=20, b=12), hovermode="x unified", yaxis_title=overlay_col)
    ov.update_xaxes(showgrid=False)
    st.markdown("#### Flagged bars in market context")
    st.plotly_chart(ov, use_container_width=True, config={"displaylogo": False})

left, right = st.columns([1, 1], gap="large")
with left:
    score_histogram = go.Figure()
    score_histogram.add_trace(go.Histogram(x=scores["score"], nbinsx=60, name="All scores", marker_color=BLUE, opacity=0.72))
    if not flagged.empty:
        score_histogram.add_trace(go.Histogram(x=flagged["score"], nbinsx=60, name="Flagged", marker_color=CORAL, opacity=0.82))
    score_histogram.update_layout(template=chart_template(), barmode="overlay", height=280, margin=dict(l=12, r=12, t=15, b=12), xaxis_title="Score", yaxis_title="Observations")
    st.plotly_chart(score_histogram, use_container_width=True, config={"displaylogo": False})
with right:
    st.markdown("**Threshold sensitivity (selected window)**")
    st.dataframe(pd.DataFrame(sens), use_container_width=True, hide_index=True)
    if "is_anomaly" in scores.columns:
        exported = pd.to_numeric(scores["is_anomaly"], errors="coerce").fillna(0).astype(bool)
        st.metric("Notebook-exported flags", f"{int(exported.sum()):,}")
        st.caption("Slider flags are recalculated in-view; exported flags keep the notebook threshold.")

if not flagged.empty:
    st.markdown("#### Lowest-scoring observations")
    st.dataframe(flagged.nsmallest(100, "score").sort_values(ts_col, ascending=False), use_container_width=True, hide_index=True)

st.download_button("Download flagged observations", data=flagged.to_csv(index=False).encode("utf-8"), file_name="btc_flagged_observations.csv", mime="text/csv")
st.caption("Anomaly flags indicate statistical unusualness under one fitted model; they are not evidence of an error or a future event. Review sensitivity across thresholds.")


mp_path = ROOT / "artifacts" / "models" / "matrix_profile.csv"
mp_meta = ROOT / "artifacts" / "models" / "matrix_profile_meta.json"
if mp_path.exists():
    st.markdown("#### Matrix profile (motif/discord, exploratory)")
    try:
        mp = load_csv(str(mp_path))
        ts_c = next((c for c in [score_timestamp_column, timestamp_column, "timestamp", "datetime", "date", "time"] if c in mp.columns), None)
        val_c = next((c for c in ["matrix_profile", "mp", "value"] if c in mp.columns), None)
        if ts_c and val_c:
            mp[ts_c] = pd.to_datetime(mp[ts_c], errors="coerce", utc=True)
            mp[val_c] = pd.to_numeric(mp[val_c], errors="coerce")
            mp = mp.dropna(subset=[ts_c, val_c])
            mp = mp[mp[ts_c].between(start, end)]
            if not mp.empty:
                mps = sample_for_chart(mp, st.session_state.get("max_pts", 5000))
                fig = go.Figure(go.Scatter(x=mps[ts_c], y=mps[val_c], mode="lines", name="Matrix profile", line=dict(color=ACCENT, width=1.2)))
                if mp_meta.exists():
                    meta = json.loads(mp_meta.read_text(encoding="utf-8"))
                    disc_ts = pd.to_datetime(meta.get("discord_timestamp"), errors="coerce", utc=True)
                    if pd.notna(disc_ts):
                        fig.add_vline(x=disc_ts, line_dash="dot", line_color=CORAL)
                        st.caption(f"Discord: {meta.get('discord_timestamp')} (value={meta.get('discord_value')}) window={meta.get('window')}")
                fig.update_layout(template=chart_template(), height=260, margin=dict(l=12,r=12,t=20,b=12), yaxis_title="Matrix profile")
                fig.update_xaxes(showgrid=False)
                st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
    except Exception:
        st.info("Matrix profile export found but could not be rendered.")
