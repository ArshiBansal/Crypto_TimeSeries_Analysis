"""Explore market states from exported Gaussian or hidden Markov models."""

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml

from src.dashboard import (
    ACCENT,
    BLUE,
    CORAL,
    PURPLE,
    ROOT,
    apply_app_style,
    chart_template,
    load_dashboard_data,
    sample_for_chart,
)


st.set_page_config(page_title="Market Regimes", page_icon="◌", layout="wide")
frame, timestamp_column = load_dashboard_data()
apply_app_style()

st.markdown(
    """
    <div class="hero">
      <div class="hero-kicker">04 &nbsp; / &nbsp; MARKET STRUCTURE</div>
      <h1>Different states. Different texture.</h1>
      <p>Inspect how saved GMM or HMM models partition the selected market
         window using the same return and volatility features used in training.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

artifact_dir = ROOT / "artifacts" / "models"
model_paths = sorted(
    path for path in artifact_dir.glob("*.pkl")
    if "gmm" in path.stem.lower() or "hmm" in path.stem.lower()
) if artifact_dir.exists() else []
if not model_paths:
    st.info("No regime model artifacts are available. Run notebook 03 to export GMM/HMM models.")
    st.stop()

metadata_path = ROOT / "processed" / "split_meta.yaml"
scaler_path = ROOT / "artifacts" / "scalers" / "regime_scaler.pkl"
if not metadata_path.exists() or not scaler_path.exists():
    st.warning(
        "Regime feature metadata or scaler is missing. Run notebook 03 to "
        "export `split_meta.yaml` and the regime scaler."
    )
    st.stop()

with metadata_path.open(encoding="utf-8") as file:
    metadata = yaml.safe_load(file)
feature_columns = [
    column
    for column in metadata.get("feature_cols", [])
    if any(token in column.lower() for token in ("rolling_vol", "log_return", "momentum", "rolling_return"))
]
missing = [column for column in feature_columns if column not in frame.columns]
if not feature_columns or missing:
    st.error(
        "The selected dataset does not contain the regime features required by "
        "the saved model."
    )
    if missing:
        st.caption("Missing: " + ", ".join(missing))
    st.stop()

choice = st.selectbox(
    "State model",
    model_paths,
    format_func=lambda path: path.stem.upper(),
)
try:
    model = joblib.load(choice)
    scaler = joblib.load(scaler_path)
except Exception as error:
    st.error("Could not load the regime model or scaler. Use trusted compatible artifacts.")
    st.exception(error)
    st.stop()

model_input = frame[feature_columns].dropna()
model_input = sample_for_chart(model_input, st.session_state.get("max_pts", 5000))
if model_input.empty:
    st.warning("No complete regime feature rows are available in this date range.")
    st.stop()

try:
    states = model.predict(scaler.transform(model_input))
except Exception as error:
    st.error("The saved regime model is incompatible with the selected feature data.")
    st.exception(error)
    st.stop()

state_frame = frame.loc[model_input.index].copy()
state_frame["state"] = states.astype(int)
counts = state_frame["state"].value_counts().sort_index()
state_columns = st.columns(min(len(counts), 4))
palette = [ACCENT, BLUE, PURPLE, CORAL]
for index, (state, count) in enumerate(counts.items()):
    state_columns[index % len(state_columns)].metric(
        f"State {state}",
        f"{count:,} bars",
        delta=f"{count / len(state_frame) * 100:.1f}% of view",
    )

price_column = next(
    (column for column in ("close", "price", "adj_close") if column in state_frame.columns),
    None,
)
if price_column:
    state_chart = go.Figure()
    for index, (state, group) in enumerate(state_frame.groupby("state", sort=True)):
        state_chart.add_trace(
            go.Scatter(
                x=group[timestamp_column],
                y=pd.to_numeric(group[price_column], errors="coerce"),
                mode="markers",
                name=f"State {state}",
                marker=dict(color=palette[index % len(palette)], size=5, opacity=0.82),
            )
        )
    state_chart.update_layout(
        template=chart_template(),
        height=440,
        margin=dict(l=12, r=12, t=18, b=12),
        hovermode="x unified",
        yaxis_title=price_column,
    )
    state_chart.update_xaxes(showgrid=False)
    st.plotly_chart(
        state_chart,
        use_container_width=True,
        config={"displaylogo": False},
    )
else:
    st.warning("No price column is available to plot the inferred states.")

left, right = st.columns([1, 1], gap="large")
with left:
    distribution = go.Figure(
        go.Bar(
            x=[f"State {state}" for state in counts.index],
            y=counts.values,
            marker_color=[palette[index % len(palette)] for index in range(len(counts))],
            text=[f"{count / len(state_frame):.1%}" for count in counts.values],
            textposition="outside",
        )
    )
    distribution.update_layout(
        template=chart_template(),
        height=310,
        margin=dict(l=12, r=12, t=15, b=12),
        yaxis_title="Bars",
    )
    st.plotly_chart(
        distribution,
        use_container_width=True,
        config={"displaylogo": False},
    )

with right:
    if hasattr(model, "transmat_"):
        transition = pd.DataFrame(
            model.transmat_,
            index=[f"State {i}" for i in range(model.transmat_.shape[0])],
            columns=[f"State {i}" for i in range(model.transmat_.shape[1])],
        )
        transition_chart = go.Figure(
            go.Heatmap(
                z=transition.values,
                x=transition.columns,
                y=transition.index,
                zmin=0,
                zmax=1,
                colorscale=[[0, "#253440"], [1, "#63A99E"]],
                colorbar=dict(title="Probability"),
                hovertemplate="%{y} → %{x}<br>%{z:.1%}<extra></extra>",
            )
        )
        transition_chart.update_layout(
            template=chart_template(),
            height=310,
            margin=dict(l=12, r=12, t=15, b=12),
        )
        st.plotly_chart(
            transition_chart,
            use_container_width=True,
            config={"displaylogo": False},
        )
    else:
        st.info("Transition probabilities are available for the HMM model.")

st.caption(
    f"Model: {choice.stem.upper()} · Input features: {len(feature_columns)} · "
    f"Predicted sample: {len(state_frame):,} bars"
)
