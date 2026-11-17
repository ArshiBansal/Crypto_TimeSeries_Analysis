"""Forecast diagnostics using exported, trusted model artifacts."""

import json

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml
from plotly.subplots import make_subplots

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
from src.models.evaluator import forecast_metrics


st.set_page_config(page_title="Forecasts", page_icon="⌁", layout="wide")
frame, timestamp_column = load_dashboard_data()
apply_app_style()

if frame is None:
    st.stop()

st.markdown(
    """
    <div class="hero">
      <div class="hero-kicker">02 &nbsp; / &nbsp; FORECAST REVIEW</div>
      <h1>Predictions, beside the observations.</h1>
      <p>Replay an exported forecasting model over the selected historical
         window and inspect error, direction, residuals, and feature weights.</p>
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
    split_metadata = yaml.safe_load(file) or {}
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

work = frame[[timestamp_column] + feature_columns + [target_column]].copy()
work[timestamp_column] = pd.to_datetime(work[timestamp_column], errors="coerce", utc=True)
work[target_column] = pd.to_numeric(work[target_column], errors="coerce")
valid = work[feature_columns].notna().all(axis=1) & work[target_column].notna() & work[timestamp_column].notna()
work = work.loc[valid].sort_values(timestamp_column)
if work.empty:
    st.warning("No complete feature/target rows are available in this date range.")
    st.stop()

sampled = sample_for_chart(work, forecast_points)
try:
    predictions = model.predict(scaler.transform(sampled[feature_columns]))
except Exception as error:
    st.error("The model could not predict from the selected feature set.")
    st.exception(error)
    st.stop()

predicted = pd.Series(np.asarray(predictions).reshape(-1), index=sampled.index, name="Forecast")
observed = sampled[target_column].astype(float)
naive = observed.shift(1)
eval_mask = naive.notna()
obs_eval = observed.loc[eval_mask]
pred_eval = predicted.loc[eval_mask]
naive_eval = naive.loc[eval_mask]
metrics = forecast_metrics(obs_eval.to_numpy(), pred_eval.to_numpy(), naive_pred=naive_eval.to_numpy())
naive_metrics = forecast_metrics(obs_eval.to_numpy(), naive_eval.to_numpy())

split_dates = {k: pd.to_datetime(split_metadata.get(k), errors="coerce", utc=True)
               for k in ("train_end", "val_end", "test_end") if split_metadata.get(k)}
with context:
    st.caption(
        "This page replays the saved model across the selected dataset window. "
        "It is not a new forward forecast or a substitute for chronological holdout evaluation."
    )
    if split_dates:
        parts = []
        for key in ("train_end", "val_end", "test_end"):
            if pd.notna(split_dates.get(key)):
                parts.append(f"{key.replace('_', ' ')}: {split_dates[key].date()}")
        st.caption("Split boundaries from metadata: " + " · ".join(parts))

metric_columns = st.columns(6)
metric_columns[0].metric("Rows evaluated", f"{len(obs_eval):,}")
metric_columns[1].metric("MAE", f"{metrics['mae']:.4g}")
metric_columns[2].metric("RMSE", f"{metrics['rmse']:.4g}")
metric_columns[3].metric("Direction acc.", f"{metrics['directional_accuracy']*100:.1f}%" if np.isfinite(metrics.get('directional_accuracy', np.nan)) else "—")
metric_columns[4].metric("Correlation (IC)", f"{metrics['correlation']:.3f}" if np.isfinite(metrics.get('correlation', np.nan)) else "—")
metric_columns[5].metric("MAE vs naive", f"{metrics['naive_scaled_mae']:.3f}x" if np.isfinite(metrics.get('naive_scaled_mae', np.nan)) else "—")

secondary = st.columns(4)
secondary[0].metric("sMAPE (diagnostic)", f"{metrics['smape']:.2f}%" if np.isfinite(metrics.get('smape', np.nan)) else "—",
                    help="Unstable near zero for log returns; do not use as the primary selector.")
secondary[1].metric("MASE", f"{metrics['mase']:.3f}" if np.isfinite(metrics.get('mase', np.nan)) else "—")
secondary[2].metric("Naive MAE", f"{naive_metrics['mae']:.4g}" if np.isfinite(naive_metrics.get('mae', np.nan)) else "—")
secondary[3].metric("Naive RMSE", f"{naive_metrics['rmse']:.4g}" if np.isfinite(naive_metrics.get('rmse', np.nan)) else "—")

figure = go.Figure()
figure.add_trace(go.Scatter(x=sampled[timestamp_column], y=observed, name="Observed",
                             mode="lines", line=dict(color=ACCENT, width=1.8)))
figure.add_trace(go.Scatter(x=sampled[timestamp_column], y=predicted, name="Model replay",
                             mode="lines", line=dict(color=BLUE, width=1.6, dash="dot")))
figure.add_trace(go.Scatter(x=sampled[timestamp_column], y=naive, name="Naive (t-1)",
                             mode="lines", line=dict(color=PURPLE, width=1.2, dash="dash")))
for label, key, color in [("Train end", "train_end", "#9AA7B2"), ("Val end", "val_end", "#D4B483"), ("Test end", "test_end", CORAL)]:
    ts = split_dates.get(key)
    if pd.notna(ts) and sampled[timestamp_column].min() <= ts <= sampled[timestamp_column].max():
        figure.add_vline(x=ts, line_dash="dash", line_color=color, opacity=0.7)
figure.update_layout(template=chart_template(), height=450, margin=dict(l=12, r=12, t=25, b=12),
                      hovermode="x unified", yaxis_title=target_column)
figure.update_xaxes(showgrid=False)
st.plotly_chart(figure, use_container_width=True, config={"displaylogo": False})

residuals = (obs_eval - pred_eval)
res_fig = make_subplots(rows=1, cols=2, column_widths=[0.62, 0.38],
                        subplot_titles=("Residuals over time (obs - pred)", "Predicted vs actual"))
res_fig.add_trace(go.Scatter(x=sampled.loc[obs_eval.index, timestamp_column], y=residuals,
                             mode="lines", name="Residual", line=dict(color=CORAL, width=1.2)), row=1, col=1)
res_fig.add_hline(y=0, line_dash="dot", line_color="#9AA7B2", row=1, col=1)
res_fig.add_trace(go.Scatter(x=obs_eval, y=pred_eval, mode="markers", name="Pred vs actual",
                             marker=dict(color=BLUE, size=4, opacity=0.45)), row=1, col=2)
if len(obs_eval):
    lo = float(min(obs_eval.min(), pred_eval.min()))
    hi = float(max(obs_eval.max(), pred_eval.max()))
    res_fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Ideal",
                                 line=dict(color="#9AA7B2", dash="dot")), row=1, col=2)
res_fig.update_layout(template=chart_template(), height=360, margin=dict(l=12, r=12, t=30, b=12), showlegend=False)
st.plotly_chart(res_fig, use_container_width=True, config={"displaylogo": False})

importance = getattr(model, "feature_importances_", None)
if importance is not None and len(importance) == len(feature_columns):
    importance_frame = (
        pd.DataFrame({"feature": feature_columns, "importance": importance})
        .sort_values("importance")
        .tail(25)
    )
    importance_chart = go.Figure(go.Bar(x=importance_frame["importance"], y=importance_frame["feature"],
                                        orientation="h", marker_color=ACCENT))
    importance_chart.update_layout(template=chart_template(), height=max(350, 22 * len(importance_frame)),
                                   margin=dict(l=12, r=12, t=25, b=12), xaxis_title="Model importance")
    st.markdown("#### What the model used")
    st.plotly_chart(importance_chart, use_container_width=True, config={"displaylogo": False})


wf_path = ROOT / "reports" / "metrics" / "walk_forward_summary.json"
test_pred_path = ROOT / "artifacts" / "models" / f"{'lgbm' if 'lgbm' in selected_path.stem.lower() else 'catboost'}_test_predictions.csv"
if wf_path.exists() or test_pred_path.exists():
    st.markdown("#### Exported evaluation artifacts")
    c1, c2 = st.columns(2)
    if wf_path.exists():
        try:
            wf = json.loads(wf_path.read_text(encoding="utf-8"))
            with c1:
                st.caption("Walk-forward summary (reports/metrics/walk_forward_summary.json)")
                st.json({k: wf.get(k) for k in ["initial_train_size","test_size","step_size","expanding","ridge","naive_proxy"] if k in wf})
        except Exception:
            pass
    if test_pred_path.exists():
        try:
            tp = pd.read_csv(test_pred_path)
            with c2:
                st.caption(f"Exported test predictions: {test_pred_path.name}")
                if {"actual","predicted"}.issubset(tp.columns):
                    tm = forecast_metrics(pd.to_numeric(tp["actual"], errors="coerce").to_numpy(),
                                          pd.to_numeric(tp["predicted"], errors="coerce").to_numpy())
                    st.write({k: tm.get(k) for k in ["mae","rmse","directional_accuracy","correlation","mase","naive_scaled_mae"]})
        except Exception:
            pass

st.caption(
    "Direction accuracy, correlation/IC, MASE, and naive-scaled MAE are better decision signals than sMAPE "
    "for near-zero return targets. Use chronological/walk-forward holdout evaluation for model selection."
)
