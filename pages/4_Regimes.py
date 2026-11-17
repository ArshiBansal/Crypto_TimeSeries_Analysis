"""Explore market states from exported Gaussian or hidden Markov models."""

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml

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
         States are unsupervised clusters-interpret with diagnostics, not as ground truth.</p>
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

state_columns = st.columns(min(max(len(counts), 1), 4))
for idx, (state, count) in enumerate(counts.items()):
    state_columns[idx % len(state_columns)].metric(f"State {state}", f"{count:,} bars",
                                                   delta=f"{count / len(full_state_frame) * 100:.1f}% of view")

diag_cols = st.columns(4)
diag_cols[0].metric("Components/states", str(int(getattr(model, "n_components", len(counts)))))
try:
    diag_cols[1].metric("Avg log-likelihood", f"{float(model.score(scaled)):.3f}")
except Exception:
    diag_cols[1].metric("Avg log-likelihood", "-")
try:
    diag_cols[2].metric("BIC", f"{float(model.bic(scaled)):.1f}" if hasattr(model, "bic") else "-")
except Exception:
    diag_cols[2].metric("BIC", "-")
if hasattr(model, "transmat_"):
    persistence = np.diag(np.asarray(model.transmat_, dtype=float))
    diag_cols[3].metric("Mean persistence", f"{float(np.nanmean(persistence)):.2%}" if persistence.size else "-")
else:
    switches = int((full_state_frame["state"].diff().fillna(0) != 0).sum())
    diag_cols[3].metric("State switches", f"{switches:,}")

price_column = next((c for c in ("close", "price", "adj_close") if c in state_frame.columns), None)
if price_column:
    state_chart = go.Figure()
    for idx, (state, group) in enumerate(state_frame.groupby("state", sort=True)):
        state_chart.add_trace(go.Scatter(x=group[timestamp_column], y=pd.to_numeric(group[price_column], errors="coerce"),
                                         mode="markers", name=f"State {state}",
                                         marker=dict(color=palette[idx % len(palette)], size=5, opacity=0.82)))
    state_chart.update_layout(template=chart_template(), height=420, margin=dict(l=12, r=12, t=18, b=12),
                              hovermode="x unified", yaxis_title=price_column)
    state_chart.update_xaxes(showgrid=False)
    st.plotly_chart(state_chart, use_container_width=True, config={"displaylogo": False})

    ribbon = sample_for_chart(full_state_frame[[timestamp_column, "state"]].sort_values(timestamp_column), 5000)
    ribbon_chart = go.Figure(go.Scatter(x=ribbon[timestamp_column], y=ribbon["state"], mode="lines",
                                        line=dict(shape="hv", color=BLUE, width=1.2), name="State over time"))
    ribbon_chart.update_layout(template=chart_template(), height=180, margin=dict(l=12, r=12, t=10, b=10),
                               yaxis_title="State", hovermode="x unified")
    ribbon_chart.update_yaxes(dtick=1)
    st.plotly_chart(ribbon_chart, use_container_width=True, config={"displaylogo": False})
else:
    st.warning("No price column is available to plot the inferred states.")

left, right = st.columns([1, 1], gap="large")
with left:
    distribution = go.Figure(go.Bar(x=[f"State {s}" for s in counts.index], y=counts.values,
                                    marker_color=[palette[i % len(palette)] for i in range(len(counts))],
                                    text=[f"{c / len(full_state_frame):.1%}" for c in counts.values], textposition="outside"))
    distribution.update_layout(template=chart_template(), height=300, margin=dict(l=12, r=12, t=15, b=12), yaxis_title="Bars")
    st.plotly_chart(distribution, use_container_width=True, config={"displaylogo": False})
with right:
    if hasattr(model, "transmat_"):
        transition = pd.DataFrame(model.transmat_, index=[f"State {i}" for i in range(model.transmat_.shape[0])],
                                  columns=[f"State {i}" for i in range(model.transmat_.shape[1])])
        tc = go.Figure(go.Heatmap(z=transition.values, x=transition.columns, y=transition.index, zmin=0, zmax=1,
                                  colorscale=[[0, "#253440"], [1, "#63A99E"]], colorbar=dict(title="Probability"),
                                  hovertemplate="%{y} -> %{x}<br>%{z:.1%}<extra></extra>"))
        tc.update_layout(template=chart_template(), height=300, margin=dict(l=12, r=12, t=15, b=12))
        st.plotly_chart(tc, use_container_width=True, config={"displaylogo": False})
    else:
        st.info("Transition probabilities and persistence are available for HMM models.")

st.markdown("#### Regime-conditioned summary")
ret_col = next((c for c in ("log_return", "simple_return") if c in full_state_frame.columns), None)
vol_candidates = [c for c in full_state_frame.columns if "rolling_vol" in c.lower() and "ann" not in c.lower()]
vol_col = vol_candidates[0] if vol_candidates else None
rows = []
for state, grp in full_state_frame.groupby("state", sort=True):
    row = {"state": int(state), "bars": int(len(grp)), "share": float(len(grp) / len(full_state_frame))}
    if ret_col:
        r = pd.to_numeric(grp[ret_col], errors="coerce")
        row["mean_return"] = float(r.mean()) if r.notna().any() else np.nan
        row["return_std"] = float(r.std()) if r.notna().any() else np.nan
    if vol_col:
        v = pd.to_numeric(grp[vol_col], errors="coerce")
        row["mean_vol"] = float(v.mean()) if v.notna().any() else np.nan
    rows.append(row)
if rows:
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
else:
    st.info("No return/volatility columns available for regime-conditioned summaries.")

st.caption(f"Model: {choice.stem.upper()} · Input features: {len(feature_columns)} · Predicted sample: {len(full_state_frame):,} bars. "
           "Use BIC/log-likelihood, stability, and interpretability to judge regime quality; do not treat labels as ground truth.")


diag_path = ROOT / "reports" / "metrics" / "regime_diagnostics.json"
if diag_path.exists():
    try:
        st.markdown("#### Exported regime diagnostics")
        st.json(json.loads(diag_path.read_text(encoding="utf-8")))
    except Exception:
        pass
