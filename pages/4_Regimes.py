"""Explore market states from exported Gaussian or hidden Markov models with enhanced visualizations."""

import json

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml
from plotly.subplots import make_subplots
from scipy.stats import norm

from src.dashboard import ACCENT, BLUE, CORAL, PURPLE, ROOT, apply_app_style, chart_template, load_dashboard_data, sample_for_chart

st.set_page_config(page_title="Market Regimes", page_icon="*", layout="wide")
frame, timestamp_column = load_dashboard_data()
apply_app_style()
if frame is None:
    st.stop()

st.markdown(
    """
    <div class="hero">
      <div class="hero-kicker">04 / MARKET STRUCTURE</div>
      <h1>Different states. Different texture.</h1>
      <p>Inspect how saved GMM or HMM models partition the selected market window.
         States are unsupervised clusters—interpret with diagnostics, not as ground truth.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

artifact_dir = ROOT / "artifacts" / "models"
model_paths = sorted([p for p in artifact_dir.glob("*.pkl") if "gmm" in p.stem.lower() or "hmm" in p.stem.lower()]) if artifact_dir.exists() else []
if not model_paths:
    st.info("No regime model artifacts are available. Run notebook 03 to export GMM/HMM models.")
    st.stop()

metadata_path = ROOT / "processed" / "split_meta.yaml"
scaler_path = ROOT / "artifacts" / "scalers" / "regime_scaler.pkl"
if not metadata_path.exists() or not scaler_path.exists():
    st.warning("Regime feature metadata or scaler is missing. Run notebook 03 to export `split_meta.yaml` and the regime scaler.")
    st.stop()

with metadata_path.open(encoding="utf-8") as file:
    metadata = yaml.safe_load(file) or {}
feature_columns = [c for c in metadata.get("feature_cols", [])
                   if any(t in c.lower() for t in ("rolling_vol", "log_return", "momentum", "rolling_return"))]
missing = [c for c in feature_columns if c not in frame.columns]
if not feature_columns or missing:
    st.error("The selected dataset does not contain the regime features required by the saved model.")
    if missing:
        st.caption("Missing: " + ", ".join(missing[:12]))
    st.stop()

choice = st.selectbox("State model", model_paths, format_func=lambda p: p.stem.upper())
try:
    model = joblib.load(choice)
    scaler = joblib.load(scaler_path)
except Exception as error:
    st.error("Could not load the regime model or scaler. Use trusted compatible artifacts.")
    st.exception(error)
    st.stop()

model_input = frame[feature_columns].dropna()
if model_input.empty:
    st.warning("No complete regime feature rows are available in this date range.")
    st.stop()
try:
    scaled = scaler.transform(model_input)
    states = model.predict(scaled)
except Exception as error:
    st.error("The saved regime model is incompatible with the selected feature data.")
    st.exception(error)
    st.stop()

full_state_frame = frame.loc[model_input.index].copy()
full_state_frame["state"] = states.astype(int)
counts = full_state_frame["state"].value_counts().sort_index()
state_frame = sample_for_chart(full_state_frame, st.session_state.get("max_pts", 5000))
palette = [ACCENT, BLUE, PURPLE, CORAL, "#D4B483", "#8BC6A8"]

tab1, tab2, tab3, tab4 = st.tabs(["📊 State Overview", "📈 Price & State", "🔄 Transitions", "📋 Regime Analytics"])

with tab1:
    st.markdown("#### State Distribution")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        state_columns = st.columns(min(max(len(counts), 1), 4))
        for idx, (state, count) in enumerate(counts.items()):
            state_columns[idx % len(state_columns)].metric(
                f"State {state}", f"{count:,} bars",
                delta=f"{count / len(full_state_frame) * 100:.1f}% of view"
            )
    
    with col2:
        dist_fig = go.Figure(go.Bar(
            x=[f"State {s}" for s in counts.index],
            y=counts.values,
            marker_color=[palette[i % len(palette)] for i in range(len(counts))],
            text=[f"{c / len(full_state_frame):.1%}" for c in counts.values],
            textposition="outside",
            hovertemplate="State %{x}<br>Count: %{y}<br>Share: %{text}<extra></extra>"
        ))
        dist_fig.update_layout(template=chart_template(), height=300,
                               margin=dict(l=12, r=12, t=15, b=12), yaxis_title="Bars")
        st.plotly_chart(dist_fig, use_container_width=True, config={"displaylogo": False})
    
    st.markdown("#### Model Diagnostics")
    diag_cols = st.columns(4)
    diag_cols[0].metric("Components/States", str(int(getattr(model, "n_components", len(counts)))))
    try:
        diag_cols[1].metric("Avg Log-Likelihood", f"{float(model.score(scaled)):.3f}")
    except Exception:
        diag_cols[1].metric("Avg Log-Likelihood", "—")
    try:
        diag_cols[2].metric("BIC", f"{float(model.bic(scaled)):.1f}" if hasattr(model, "bic") else "—")
    except Exception:
        diag_cols[2].metric("BIC", "—")
    if hasattr(model, "transmat_"):
        persistence = np.diag(np.asarray(model.transmat_, dtype=float))
        diag_cols[3].metric("Mean Persistence", f"{float(np.nanmean(persistence)):.2%}" if persistence.size else "—")
    else:
        switches = int((full_state_frame["state"].diff().fillna(0) != 0).sum())
        diag_cols[3].metric("State Switches", f"{switches:,}")
    
    if hasattr(model, "means_") and hasattr(model, "covariances_"):
        st.markdown("#### Gaussian Components (Feature Space)")
        n_components = model.n_components
        n_features = len(feature_columns)
        
        if n_features <= 3:
            for i in range(min(3, n_features)):
                feat_fig = go.Figure()
                for state in range(n_components):
                    mean = model.means_[state, i]
                    cov = model.covariances_[state]
                    if cov.ndim == 1:
                        std = np.sqrt(cov[i])
                    else:
                        std = np.sqrt(cov[i, i])
                    
                    x = np.linspace(mean - 4*std, mean + 4*std, 200)
                    y = norm.pdf(x, mean, std) * counts.iloc[state] / len(full_state_frame)
                    feat_fig.add_trace(go.Scatter(x=x, y=y, mode="lines",
                                                  name=f"State {state}", line=dict(color=palette[state], width=2),
                                                  fill="tozeroy", fillcolor=f"rgba{palette[state][1:-1]},0.15)"))
                feat_fig.update_layout(template=chart_template(), height=250,
                                      margin=dict(l=12, r=12, t=25, b=12),
                                      xaxis_title=feature_columns[i], yaxis_title="Density")
                st.plotly_chart(feat_fig, use_container_width=True, config={"displaylogo": False})

with tab2:
    price_column = next((c for c in ("close", "price", "adj_close") if c in state_frame.columns), None)
    if price_column:
        st.markdown("#### Price Colored by Regime")
        state_chart = go.Figure()
        for idx, (state, group) in enumerate(state_frame.groupby("state", sort=True)):
            state_chart.add_trace(go.Scatter(
                x=group[timestamp_column],
                y=pd.to_numeric(group[price_column], errors="coerce"),
                mode="markers",
                name=f"State {state}",
                marker=dict(color=palette[idx % len(palette)], size=6, opacity=0.85,
                           line=dict(width=0.5, color="white")),
                hovertemplate=f"State {state}<br>%{{x}}<br>Price: %{{y:$.2f}}<extra></extra>"
            ))
        state_chart.update_layout(template=chart_template(), height=450,
                                  margin=dict(l=12, r=12, t=25, b=12),
                                  hovermode="x unified", yaxis_title=price_column,
                                  legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        state_chart.update_xaxes(showgrid=False)
        st.plotly_chart(state_chart, use_container_width=True, config={"displaylogo": False})
        
        st.markdown("#### State Timeline Ribbon")
        ribbon = sample_for_chart(full_state_frame[[timestamp_column, "state"]].sort_values(timestamp_column), 5000)
        ribbon_fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                                   row_heights=[0.15, 0.85],
                                   subplot_titles=("Regime Sequence", "Price"))
        
        ribbon_fig.add_trace(go.Scatter(
            x=ribbon[timestamp_column], y=ribbon["state"],
            mode="lines", line=dict(shape="hv", color=BLUE, width=1.5),
            name="State", hovertemplate="%{x}<br>State: %{y}<extra></extra>"
        ), row=1, col=1)
        
        for idx, (state, group) in enumerate(full_state_frame.groupby("state", sort=True)):
            ribbon_fig.add_trace(go.Scatter(
                x=group[timestamp_column],
                y=pd.to_numeric(group[price_column], errors="coerce"),
                mode="markers", name=f"State {state}",
                marker=dict(color=palette[idx % len(palette)], size=4, opacity=0.7,
                           line=dict(width=0)),
                showlegend=False,
                hovertemplate=f"State {state}<br>%{{x}}<br>Price: %{{y:$.2f}}<extra></extra>"
            ), row=2, col=1)
        
        ribbon_fig.update_layout(template=chart_template(), height=550,
                                margin=dict(l=12, r=12, t=35, b=12),
                                hovermode="x unified", yaxis_title="State", yaxis2_title=price_column)
        ribbon_fig.update_xaxes(showgrid=False)
        ribbon_fig.update_yaxes(dtick=1, row=1, col=1)
        st.plotly_chart(ribbon_fig, use_container_width=True, config={"displaylogo": False})
        
        st.markdown("#### State-Conditioned Return Distribution")
        ret_col = next((c for c in ("log_return", "simple_return") if c in full_state_frame.columns), None)
        if ret_col:
            ret_fig = go.Figure()
            for state in sorted(full_state_frame["state"].unique()):
                returns = pd.to_numeric(full_state_frame[full_state_frame["state"] == state][ret_col], errors="coerce").dropna()
                if len(returns) > 10:
                    ret_fig.add_trace(go.Violin(
                        y=returns * 100,
                        name=f"State {state}",
                        side="positive",
                        width=0.8,
                        line_color=palette[state % len(palette)],
                        fillcolor=palette[state % len(palette)],
                        opacity=0.4,
                        box_visible=True,
                        meanline_visible=True,
                        hovertemplate=f"State {state}<br>Return: %{{y:.2f}}%<extra></extra>"
                    ))
            ret_fig.add_hline(y=0, line_dash="dash", line_color="#9AA7B2")
            ret_fig.update_layout(template=chart_template(), height=350,
                                 margin=dict(l=12, r=12, t=25, b=12),
                                 yaxis_title="Return (%)", xaxis_title="Regime")
            st.plotly_chart(ret_fig, use_container_width=True, config={"displaylogo": False})
    else:
        st.warning("No price column is available to plot the inferred states.")

with tab3:
    if hasattr(model, "transmat_"):
        st.markdown("#### Transition Probability Matrix (HMM)")
        transition = pd.DataFrame(model.transmat_, index=[f"State {i}" for i in range(model.transmat_.shape[0])],
                                  columns=[f"State {i}" for i in range(model.transmat_.shape[1])])
        
        tc = go.Figure(go.Heatmap(
            z=transition.values,
            x=transition.columns,
            y=transition.index,
            zmin=0, zmax=1,
            colorscale=[[0, "#253440"], [0.5, "#3D5A6B"], [1, "#63A99E"]],
            colorbar=dict(title="Probability"),
            hovertemplate="%{y} → %{x}<br>P = %{z:.2%}<extra></extra>",
            text=[[f"{v:.1%}" for v in row] for row in transition.values],
            texttemplate="%{text}",
            textfont=dict(size=12, color="white")
        ))
        tc.update_layout(template=chart_template(), height=400,
                         margin=dict(l=12, r=12, t=25, b=12))
        st.plotly_chart(tc, use_container_width=True, config={"displaylogo": False})
        
        st.markdown("#### State Persistence & Expected Duration")
        persistence = np.diag(transition.values)
        expected_duration = 1 / (1 - persistence + 1e-10)
        
        pers_fig = go.Figure()
        pers_fig.add_trace(go.Bar(x=[f"State {i}" for i in range(len(persistence))],
                                  y=persistence, name="Self-transition prob.",
                                  marker_color=ACCENT, text=[f"{p:.1%}" for p in persistence],
                                  textposition="outside"))
        pers_fig.add_trace(go.Bar(x=[f"State {i}" for i in range(len(expected_duration))],
                                  y=expected_duration, name="Expected duration (bars)",
                                  marker_color=BLUE, text=[f"{d:.1f}" for d in expected_duration],
                                  textposition="outside", yaxis="y2"))
        pers_fig.update_layout(template=chart_template(), height=350,
                              margin=dict(l=12, r=12, t=25, b=12),
                              yaxis=dict(title="Persistence"), yaxis2=dict(title="Duration", overlaying="y", side="right"),
                              barmode="group", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(pers_fig, use_container_width=True, config={"displaylogo": False})
        
        st.markdown("#### Transition Flow (Sankey-style)")
        if len(transition) <= 5:
            flow_fig = go.Figure()
            for i in range(len(transition)):
                for j in range(len(transition)):
                    prob = transition.iloc[i, j]
                    if prob > 0.05:
                        flow_fig.add_annotation(
                            x=0.2, y=0.8 - i*0.15, xref="paper", yref="paper",
                            text=f"State {i} → State {j}: {prob:.1%}",
                            showarrow=False, font=dict(size=11, color=palette[i % len(palette)])
                        )
            flow_fig.update_layout(template=chart_template(), height=300,
                                  margin=dict(l=12, r=12, t=10, b=10),
                                  xaxis=dict(visible=False), yaxis=dict(visible=False))
            st.info("Strong transitions (>5%): " + 
                    ", ".join([f"{transition.index[i]}→{transition.columns[j]} ({transition.iloc[i,j]:.1%})" 
                              for i in range(len(transition)) for j in range(len(transition)) if transition.iloc[i,j] > 0.05]))
    else:
        st.info("Transition probabilities and persistence are available for HMM models.")
        switches = int((full_state_frame["state"].diff().fillna(0) != 0).sum())
        st.metric("State Switches in View", f"{switches:,}")
        
        state_duration = full_state_frame.groupby((full_state_frame["state"] != full_state_frame["state"].shift()).cumsum()).size()
        dur_fig = go.Figure(go.Histogram(x=state_duration, nbinsx=30, marker_color=BLUE, opacity=0.7))
        dur_fig.update_layout(template=chart_template(), height=300,
                             margin=dict(l=12, r=12, t=25, b=12),
                             xaxis_title="State Duration (bars)", yaxis_title="Frequency")
        st.plotly_chart(dur_fig, use_container_width=True, config={"displaylogo": False})

with tab4:
    st.markdown("#### Regime-Conditioned Summary Statistics")
    ret_col = next((c for c in ("log_return", "simple_return") if c in full_state_frame.columns), None)
    vol_candidates = [c for c in full_state_frame.columns if "rolling_vol" in c.lower() and "ann" not in c.lower()]
    vol_col = vol_candidates[0] if vol_candidates else None
    
    rows = []
    for state, grp in full_state_frame.groupby("state", sort=True):
        row = {"State": int(state), "Bars": int(len(grp)), "Share": f"{len(grp) / len(full_state_frame):.1%}"}
        if ret_col:
            r = pd.to_numeric(grp[ret_col], errors="coerce")
            row["Mean Return"] = f"{r.mean():.4f}" if r.notna().any() else "—"
            row["Return Std"] = f"{r.std():.4f}" if r.notna().any() else "—"
            row["Skew"] = f"{r.skew():.3f}" if r.notna().any() else "—"
            row["Kurtosis"] = f"{r.kurtosis():.3f}" if r.notna().any() else "—"
            row["Pos %"] = f"{(r > 0).mean():.1%}" if r.notna().any() else "—"
        if vol_col:
            v = pd.to_numeric(grp[vol_col], errors="coerce")
            row["Mean Vol"] = f"{v.mean():.4f}" if v.notna().any() else "—"
            row["Vol Std"] = f"{v.std():.4f}" if v.notna().any() else "—"
        rows.append(row)
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    
    st.markdown("#### Feature Distributions by Regime")
    if len(feature_columns) > 0:
        feat_select = st.selectbox("Select feature to visualize by regime", feature_columns)
        feat_data = full_state_frame[feat_select].dropna()
        
        if len(feat_data) > 0:
            feat_fig = go.Figure()
            for state in sorted(full_state_frame["state"].unique()):
                vals = pd.to_numeric(full_state_frame[full_state_frame["state"] == state][feat_select], errors="coerce").dropna()
                if len(vals) > 5:
                    feat_fig.add_trace(go.Violin(
                        y=vals, name=f"State {state}",
                        side="positive", width=0.8,
                        line_color=palette[state % len(palette)],
                        fillcolor=palette[state % len(palette)], opacity=0.4,
                        box_visible=True, meanline_visible=True,
                        hovertemplate=f"State {state}<br>{feat_select}: %{{y:.4f}}<extra></extra>"
                    ))
            feat_fig.update_layout(template=chart_template(), height=350,
                                  margin=dict(l=12, r=12, t=25, b=12),
                                  yaxis_title=feat_select, xaxis_title="Regime")
            st.plotly_chart(feat_fig, use_container_width=True, config={"displaylogo": False})
    
    st.markdown("#### Regime Sequence Analysis")
    state_seq = full_state_frame["state"].values
    unique_states = sorted(full_state_frame["state"].unique())
    
    seq_stats = []
    for state in unique_states:
        state_mask = state_seq == state
        runs = np.diff(np.concatenate(([0], state_mask.astype(int), [0])))
        run_starts = np.where(runs == 1)[0]
        run_ends = np.where(runs == -1)[0]
        durations = run_ends - run_starts
        
        if len(durations) > 0:
            seq_stats.append({
                "State": state,
                "Episodes": len(durations),
                "Avg Duration": f"{durations.mean():.1f}",
                "Median Duration": f"{np.median(durations):.1f}",
                "Max Duration": f"{durations.max()}",
                "Total Bars": int(state_mask.sum())
            })
    
    if seq_stats:
        st.dataframe(pd.DataFrame(seq_stats), use_container_width=True, hide_index=True)

st.caption(f"Model: {choice.stem.upper()} · Input features: {len(feature_columns)} · Predicted sample: {len(full_state_frame):,} bars. "
           "Use BIC/log-likelihood, stability, and interpretability to judge regime quality; do not treat labels as ground truth.")

diag_path = ROOT / "reports" / "metrics" / "regime_diagnostics.json"
if diag_path.exists():
    try:
        st.markdown("#### Exported Regime Diagnostics")
        st.json(json.loads(diag_path.read_text(encoding="utf-8")))
    except Exception:
        pass