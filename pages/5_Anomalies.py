"""Explore exported Isolation Forest anomaly scores with enhanced visualizations."""

import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy.stats import norm

from src.dashboard import ACCENT, BLUE, CORAL, PURPLE, ROOT, apply_app_style, chart_template, load_csv, load_dashboard_data, sample_for_chart

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
         Flags indicate statistical unusualness under one fitted model—not errors or future events.</p>
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

tab1, tab2, tab3, tab4 = st.tabs(["📊 Score Timeline", "🔍 Market Context", "📈 Distribution", "🔬 Deep Analysis"])

with tab1:
    quantile = st.slider("Flag the lowest score percentile", 0.5, 20.0, 5.0, 0.5,
                         help="Lower Isolation Forest scores are treated as more unusual; validate sensitivity to this threshold.")
    threshold = float(scores["score"].quantile(quantile / 100))
    scores["flagged"] = scores["score"] <= threshold
    flagged = scores[scores["flagged"]]
    
    sens = [{"percentile": q, "threshold": float(scores['score'].quantile(q/100)),
             "flagged": int((scores['score'] <= scores['score'].quantile(q/100)).sum()),
             "share": float((scores['score'] <= scores['score'].quantile(q/100)).mean())} for q in [1, 2, 5, 10]]
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Scored bars", f"{len(scores):,}")
    col2.metric("Flagged", f"{len(flagged):,}")
    col3.metric("Share flagged", f"{len(flagged) / len(scores):.2%}")
    col4.metric("Score threshold", f"{threshold:.4g}")
    
    chart_scores = sample_for_chart(scores, st.session_state.get("max_pts", 5000))
    chart_flagged = sample_for_chart(flagged, st.session_state.get("max_pts", 5000))
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        row_heights=[0.7, 0.3],
                        subplot_titles=("Anomaly Scores Over Time", "Score Density"))
    
    fig.add_trace(go.Scattergl(x=chart_scores[ts_col], y=chart_scores["score"],
                               mode="lines", name="Score", line=dict(color=BLUE, width=1), opacity=0.8), row=1, col=1)
    fig.add_trace(go.Scatter(x=chart_flagged[ts_col], y=chart_flagged["score"],
                             mode="markers", name="Flagged", marker=dict(color=CORAL, size=8, symbol="x")), row=1, col=1)
    fig.add_hline(y=threshold, line_dash="dot", line_color=ACCENT,
                  annotation_text=f"{quantile:.1f}th percentile: {threshold:.4f}", row=1, col=1)
    
    fig.add_trace(go.Histogram(x=scores["score"], nbinsx=60, name="All scores",
                               marker_color=BLUE, opacity=0.5), row=2, col=1)
    if not flagged.empty:
        fig.add_trace(go.Histogram(x=flagged["score"], nbinsx=60, name="Flagged",
                                   marker_color=CORAL, opacity=0.7), row=2, col=1)
    fig.add_vline(x=threshold, line_dash="dash", line_color=ACCENT, row=2, col=1)
    
    fig.update_layout(template=chart_template(), height=600, margin=dict(l=12, r=12, t=40, b=12),
                      hovermode="x unified", yaxis_title="Isolation Forest Score", yaxis2_title="Frequency",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_xaxes(showgrid=False)
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
    
    st.markdown("**Threshold sensitivity (selected window)**")
    st.dataframe(pd.DataFrame(sens), use_container_width=True, hide_index=True)
    if "is_anomaly" in scores.columns:
        exported = pd.to_numeric(scores["is_anomaly"], errors="coerce").fillna(0).astype(bool)
        st.metric("Notebook-exported flags", f"{int(exported.sum()):,}")
        st.caption("Slider flags are recalculated in-view; exported flags keep the notebook threshold.")

with tab2:
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
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                            row_heights=[0.6, 0.4],
                            subplot_titles=(f"{overlay_col} with Anomalies", "Anomaly Score vs Market Move"))
        
        fig.add_trace(go.Scatter(x=ov_sample[timestamp_column], y=ov_sample[overlay_col],
                                 name=overlay_col, mode="lines", line=dict(color=ACCENT, width=1.5)), row=1, col=1)
        if not mf_sample.empty:
            fig.add_trace(go.Scatter(x=mf_sample["ts"], y=mf_sample[overlay_col],
                                     name="Flagged bars", mode="markers",
                                     marker=dict(color=CORAL, size=10, symbol="x", line=dict(width=1, color="white"))), row=1, col=1)
        
        if not flagged.empty and ret_col:
            flagged_returns = pd.merge_asof(flagged.sort_values(ts_col).rename(columns={ts_col: "ts"}),
                                            frame[[timestamp_column, ret_col]].rename(columns={timestamp_column: "ts"}).dropna(),
                                            on="ts", direction="nearest", tolerance=pd.Timedelta("1D"))
            if not flagged_returns.empty:
                fig.add_trace(go.Scatter(x=flagged_returns["score"], y=flagged_returns[ret_col] * 100,
                                         mode="markers", name="Flagged returns",
                                         marker=dict(color=CORAL, size=8, symbol="diamond",
                                                    line=dict(width=1, color="white"))), row=2, col=1)
                fig.add_trace(go.Scatter(x=scores["score"], y=pd.merge_asof(
                    scores.sort_values(ts_col).rename(columns={ts_col: "ts"}),
                    frame[[timestamp_column, ret_col]].rename(columns={timestamp_column: "ts"}).dropna(),
                    on="ts", direction="nearest", tolerance=pd.Timedelta("1D"))[ret_col] * 100,
                    mode="markers", name="All returns",
                    marker=dict(color=BLUE, size=4, opacity=0.3)), row=2, col=1)
                fig.add_hline(y=0, line_dash="dash", line_color="#9AA7B2", row=2, col=1)
        
        fig.update_layout(template=chart_template(), height=650, margin=dict(l=12, r=12, t=40, b=12),
                          hovermode="x unified", yaxis_title=overlay_col, yaxis2_title="Return (%)")
        fig.update_xaxes(showgrid=False)
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
        
        st.markdown("#### Flagged Bar Details")
        if not flagged.empty:
            display_cols = [ts_col, "score"] + [c for c in ["is_anomaly"] if c in flagged.columns]
            st.dataframe(flagged[display_cols].sort_values("score").head(50), use_container_width=True, hide_index=True)
    else:
        st.info("No price or return column available for market context overlay.")

with tab3:
    st.markdown("#### Score Distribution Analysis")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        dist_fig = make_subplots(rows=2, cols=2, subplot_titles=("Histogram", "Box Plot", "ECDF", "Violin"))
        
        dist_fig.add_trace(go.Histogram(x=scores["score"], nbinsx=80, marker_color=BLUE, opacity=0.7), row=1, col=1)
        dist_fig.add_vline(x=threshold, line_dash="dash", line_color=ACCENT, row=1, col=1)
        
        dist_fig.add_trace(go.Box(y=scores["score"], name="All", marker_color=BLUE), row=1, col=2)
        if not flagged.empty:
            dist_fig.add_trace(go.Box(y=flagged["score"], name="Flagged", marker_color=CORAL), row=1, col=2)
        
        ecdf_x = np.sort(scores["score"])
        ecdf_y = np.arange(1, len(ecdf_x) + 1) / len(ecdf_x)
        dist_fig.add_trace(go.Scatter(x=ecdf_x, y=ecdf_y, mode="lines", name="ECDF", line=dict(color=PURPLE, width=2)), row=2, col=1)
        dist_fig.add_hline(y=quantile/100, line_dash="dash", line_color=ACCENT, row=2, col=1)
        
        dist_fig.add_trace(go.Violin(y=scores["score"], name="All", side="positive", width=0.8,
                                     line_color=BLUE, fillcolor=BLUE, opacity=0.4, box_visible=True, meanline_visible=True), row=2, col=2)
        if not flagged.empty:
            dist_fig.add_trace(go.Violin(y=flagged["score"], name="Flagged", side="positive", width=0.8,
                                         line_color=CORAL, fillcolor=CORAL, opacity=0.4, box_visible=True, meanline_visible=True), row=2, col=2)
        
        dist_fig.update_layout(template=chart_template(), height=500, margin=dict(l=12, r=12, t=40, b=12), showlegend=False)
        st.plotly_chart(dist_fig, use_container_width=True, config={"displaylogo": False})
    
    with col2:
        st.markdown("**Distribution Statistics**")
        stats = scores["score"].describe()
        for k, v in stats.items():
            if pd.notna(v):
                st.metric(k.capitalize(), f"{v:.4f}")
        
        st.markdown("**Percentiles**")
        pcts = scores["score"].quantile([0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999])
        for p, v in pcts.items():
            st.metric(f"{p:.1%}", f"{v:.4f}")
        
        if not flagged.empty:
            st.markdown("**Flagged Statistics**")
            fstats = flagged["score"].describe()
            for k, v in fstats.items():
                if pd.notna(v):
                    st.metric(f"Flagged {k.capitalize()}", f"{v:.4f}")

with tab4:
    st.markdown("#### Anomaly Clustering")
    if len(flagged) > 1:
        flagged = flagged.sort_values(ts_col).reset_index(drop=True)
        flagged["time_diff"] = flagged[ts_col].diff().dt.total_seconds()
        clusters = []
        cluster_id = 0
        for i, row in flagged.iterrows():
            if i == 0 or pd.isna(row["time_diff"]) or row["time_diff"] > 3600:
                cluster_id += 1
            clusters.append(cluster_id)
        flagged["cluster"] = clusters
        
        cluster_summary = flagged.groupby("cluster").agg(
            start=(ts_col, "min"),
            end=(ts_col, "max"),
            count=("score", "count"),
            min_score=("score", "min"),
            avg_score=("score", "mean")
        ).reset_index()
        cluster_summary["duration"] = (cluster_summary["end"] - cluster_summary["start"]).dt.total_seconds() / 60
        
        st.dataframe(cluster_summary.sort_values("min_score"), use_container_width=True, hide_index=True)
        
        cluster_fig = go.Figure()
        for cid in flagged["cluster"].unique():
            cdata = flagged[flagged["cluster"] == cid]
            cluster_fig.add_trace(go.Scatter(x=cdata[ts_col], y=cdata["score"],
                                             mode="markers+lines", name=f"Cluster {cid}",
                                             marker=dict(size=8), line=dict(width=1)))
        cluster_fig.update_layout(template=chart_template(), height=350,
                                  margin=dict(l=12, r=12, t=25, b=12),
                                  xaxis_title="Time", yaxis_title="Anomaly Score")
        st.plotly_chart(cluster_fig, use_container_width=True, config={"displaylogo": False})
    
    st.markdown("#### Lowest-Scoring Observations")
    if not flagged.empty:
        st.dataframe(flagged.nsmallest(100, "score").sort_values(ts_col, ascending=False), use_container_width=True, hide_index=True)
    
    st.download_button("Download flagged observations", data=flagged.to_csv(index=False).encode("utf-8"),
                       file_name="btc_flagged_observations.csv", mime="text/csv")
    
    st.caption("Anomaly flags indicate statistical unusualness under one fitted model; they are not evidence of an error or a future event. Review sensitivity across thresholds.")

mp_path = ROOT / "artifacts" / "models" / "matrix_profile.csv"
mp_meta = ROOT / "artifacts" / "models" / "matrix_profile_meta.json"
if mp_path.exists():
    st.markdown("#### Matrix Profile (Motif/Discord, Exploratory)")
    try:
        mp = load_csv(str(mp_path))
        ts_c = next((c for c in [ts_col, timestamp_column, "timestamp", "datetime", "date", "time"] if c in mp.columns), None)
        val_c = next((c for c in ["matrix_profile", "mp", "value"] if c in mp.columns), None)
        if ts_c and val_c:
            mp[ts_c] = pd.to_datetime(mp[ts_c], errors="coerce", utc=True)
            mp[val_c] = pd.to_numeric(mp[val_c], errors="coerce")
            mp = mp.dropna(subset=[ts_c, val_c])
            mp = mp[mp[ts_c].between(start, end)]
            if not mp.empty:
                mps = sample_for_chart(mp, st.session_state.get("max_pts", 5000))
                
                mp_fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                                       row_heights=[0.7, 0.3],
                                       subplot_titles=("Matrix Profile", "Profile Distribution"))
                
                mp_fig.add_trace(go.Scatter(x=mps[ts_c], y=mps[val_c], mode="lines",
                                            name="Matrix profile", line=dict(color=ACCENT, width=1.2)), row=1, col=1)
                mp_fig.add_trace(go.Histogram(x=mp[val_c], nbinsx=60, marker_color=ACCENT, opacity=0.7), row=2, col=1)
                
                if mp_meta.exists():
                    meta = json.loads(mp_meta.read_text(encoding="utf-8"))
                    disc_ts = pd.to_datetime(meta.get("discord_timestamp"), errors="coerce", utc=True)
                    if pd.notna(disc_ts):
                        mp_fig.add_vline(x=disc_ts, line_dash="dot", line_color=CORAL, row=1, col=1)
                        mp_fig.add_vline(x=disc_ts, line_dash="dot", line_color=CORAL, row=2, col=1)
                        st.caption(f"Discord: {meta.get('discord_timestamp')} (value={meta.get('discord_value')}) window={meta.get('window')}")
                
                mp_fig.update_layout(template=chart_template(), height=500, margin=dict(l=12, r=12, t=40, b=12),
                                     hovermode="x unified", yaxis_title="Matrix Profile", yaxis2_title="Frequency")
                mp_fig.update_xaxes(showgrid=False)
                st.plotly_chart(mp_fig, use_container_width=True, config={"displaylogo": False})
    except Exception:
        st.info("Matrix profile export found but could not be rendered.")