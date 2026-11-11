"""Forecast diagnostics using exported, trusted model artifacts."""

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.dashboard import (
    ACCENT,
    BLUE,
    ROOT,
    apply_app_style,
    chart_template,
    load_dashboard_data,
    sample_for_chart,
)


st.set_page_config(page_title="Forecasts", page_icon="⌁", layout="wide")
frame, timestamp_column = load_dashboard_data()
apply_app_style()

st.markdown(
    """
    <div class="hero">
      <div class="hero-kicker">02 &nbsp; / &nbsp; FORECAST REVIEW</div>
      <h1>Predictions, beside the observations.</h1>
      <p>Replay an exported forecasting model over the selected historical
         window and inspect its errors and learned feature weights.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

artifact_dir = ROOT / "artifacts" / "models"
model_files = sorted(artifact_dir.glob("*.pkl")) if artifact_dir.exists() else []
forecast_files = [
    path for path in model_files
    if "lgbm" in path.stem.lower() or "catboost" in path.stem.lower()
]

if not forecast_files:
    st.info(
        "No LightGBM or CatBoost forecast artifacts are available in "
        "`artifacts/models/`. Run notebook 03 to export them."
    )
    st.stop()

if st.session_state.get("resolution", "Raw") != "Raw":
    st.warning(
        "Forecast replay requires raw bars. Set **Aggregate bars** to **Raw** "
        "in the sidebar; aggregation changes the trained feature and target values."
    )
    st.stop()

split_path = ROOT / "processed" / "split_meta.yaml"
if not split_path.exists():
    st.warning(
        "Feature metadata is missing. Run notebook 02 to create "
        "`processed/split_meta.yaml`."
    )
    st.stop()

with split_path.open(encoding="utf-8") as file:
    split_metadata = yaml.safe_load(file)
target_column = split_metadata.get("target_col")
if not target_column or target_column not in frame.columns:
    st.error(
        "The selected dataset does not contain the model's target listed in "
        "`split_meta.yaml`. Choose the matching feature dataset or rerun notebooks 01–02."
    )
    st.stop()

control, context = st.columns([1, 2], gap="large")
with control:
    selected_path = st.selectbox(
        "Model artifact",
        forecast_files,
        format_func=lambda path: path.stem.replace("_", " ").title(),
    )
    st.caption(f"Observed target: `{target_column}`")
    forecast_points = st.slider(
        "Maximum evaluation points",
        min_value=100,
        max_value=10000,
        value=min(5000, max(100, len(frame))),
        step=100,
    )

feature_columns = split_metadata.get("feature_cols", [])
if not feature_columns:
    st.error("No feature columns are listed in `split_meta.yaml`.")
    st.stop()

missing_features = [column for column in feature_columns if column not in frame.columns]
if missing_features:
    st.error(
        "The selected dataset is missing features required by this model: "
        + ", ".join(missing_features[:12])
    )
    st.stop()

try:
    artifact = joblib.load(selected_path)
except Exception as error:
    st.error(f"Could not load `{selected_path.name}`. Use a trusted, compatible artifact.")
    st.exception(error)
    st.stop()

model = artifact.get("model") if isinstance(artifact, dict) else None
scaler = artifact.get("scaler") if isinstance(artifact, dict) else None
if model is None or scaler is None:
    st.error("The selected artifact must contain both `model` and `scaler` entries.")
    st.stop()

features = frame[feature_columns]
target = pd.to_numeric(frame[target_column], errors="coerce")
valid = features.notna().all(axis=1) & target.notna()
features = features.loc[valid]
target = target.loc[valid]
if features.empty:
    st.warning("No complete feature/target rows are available in this date range.")
    st.stop()

sampled_features = sample_for_chart(features, forecast_points)
sampled_target = target.loc[sampled_features.index]
try:
    predictions = model.predict(scaler.transform(sampled_features))
except Exception as error:
    st.error("The model could not predict from the selected feature set.")
    st.exception(error)
    st.stop()

predicted = pd.Series(predictions, index=sampled_target.index, name="Forecast")
mae = mean_absolute_error(sampled_target, predicted)
rmse = mean_squared_error(sampled_target, predicted) ** 0.5
with context:
    st.caption(
        "This page replays the saved model across the selected dataset window. "
        "It is not a new forecast and is not a substitute for the notebook's "
        "chronological holdout evaluation."
    )

metric_columns = st.columns(4)
metric_columns[0].metric("Rows evaluated", f"{len(sampled_target):,}")
metric_columns[1].metric("MAE", f"{mae:.6g}")
metric_columns[2].metric("RMSE", f"{rmse:.6g}")
metric_columns[3].metric("Features", f"{len(feature_columns):,}")

figure = go.Figure()
figure.add_trace(
    go.Scatter(
        x=frame.loc[sampled_target.index, timestamp_column],
        y=sampled_target,
        name="Observed",
        mode="lines",
        line=dict(color=ACCENT, width=1.8),
    )
)
figure.add_trace(
    go.Scatter(
        x=frame.loc[sampled_target.index, timestamp_column],
        y=predicted,
        name="Model replay",
        mode="lines",
        line=dict(color=BLUE, width=1.6, dash="dot"),
    )
)
figure.update_layout(
    template=chart_template(),
    height=470,
    margin=dict(l=12, r=12, t=25, b=12),
    hovermode="x unified",
    yaxis_title=target_column,
)
figure.update_xaxes(showgrid=False)
st.plotly_chart(figure, use_container_width=True, config={"displaylogo": False})

importance = getattr(model, "feature_importances_", None)
if importance is not None and len(importance) == len(feature_columns):
    importance_frame = (
        pd.DataFrame({"feature": feature_columns, "importance": importance})
        .sort_values("importance")
        .tail(25)
    )
    importance_chart = go.Figure(
        go.Bar(
            x=importance_frame["importance"],
            y=importance_frame["feature"],
            orientation="h",
            marker_color=ACCENT,
        )
    )
    importance_chart.update_layout(
        template=chart_template(),
        height=max(350, 22 * len(importance_frame)),
        margin=dict(l=12, r=12, t=25, b=12),
        xaxis_title="Model importance",
    )
    st.markdown("#### What the model used")
    st.plotly_chart(
        importance_chart,
        use_container_width=True,
        config={"displaylogo": False},
    )
