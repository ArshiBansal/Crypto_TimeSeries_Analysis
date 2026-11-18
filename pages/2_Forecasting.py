"""Forecast diagnostics using exported, trusted model artifacts with SHAP explainability."""

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
    target_candidates = [
        c for c in frame.columns
        if c.startswith("target_") or c in ("log_return", "simple_return", "return")
    ]
    if target_candidates:
        st.warning(
            f"Default target `{target_column}` not in this dataset. "
            f"Available target-like columns: {', '.join(target_candidates)}"
        )
        target_column = st.selectbox(
            "Select target column for evaluation",
            target_candidates,
            index=0,
            help="Choose the column to use as the ground-truth target for forecast evaluation.",
        )
    else:
        st.error(
            "The selected dataset does not contain the model's target listed in "
            "`split_meta.yaml` and no target-like columns were found. "
            "Choose the matching feature dataset or rerun notebooks 01–02."
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
    X_scaled = scaler.transform(sampled[feature_columns].to_numpy())
    predictions = model.predict(X_scaled)
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

tab1, tab2, tab3, tab4 = st.tabs(["📈 Replay", "🔍 Residuals", "🧠 SHAP Explainability", "📊 Feature Analysis"])

with tab1:
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=sampled[timestamp_column], y=observed, name="Observed",
                                 mode="lines", line=dict(color=ACCENT, width=2)))
    figure.add_trace(go.Scatter(x=sampled[timestamp_column], y=predicted, name="Model replay",
                                 mode="lines", line=dict(color=BLUE, width=1.8, dash="dot")))
    figure.add_trace(go.Scatter(x=sampled[timestamp_column], y=naive, name="Naive (t-1)",
                                 mode="lines", line=dict(color=PURPLE, width=1.4, dash="dash")))
    for label, key, color in [("Train end", "train_end", "#9AA7B2"), ("Val end", "val_end", "#D4B483"), ("Test end", "test_end", CORAL)]:
        ts = split_dates.get(key)
        if pd.notna(ts) and sampled[timestamp_column].min() <= ts <= sampled[timestamp_column].max():
            figure.add_vline(x=ts, line_dash="dash", line_color=color, opacity=0.7,
                             annotation_text=label, annotation_position="top")
    figure.update_layout(template=chart_template(), height=500, margin=dict(l=12, r=12, t=30, b=12),
                          hovermode="x unified", yaxis_title=target_column,
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    figure.update_xaxes(showgrid=False)
    st.plotly_chart(figure, use_container_width=True, config={"displaylogo": False})

    st.markdown("#### Error distribution")
    errors = obs_eval - pred_eval
    error_fig = go.Figure()
    error_fig.add_trace(go.Histogram(x=errors, nbinsx=50, name="Prediction error",
                                      marker_color=ACCENT, opacity=0.8))
    error_fig.add_vline(x=0, line_dash="dash", line_color="#9AA7B2")
    error_fig.add_vline(x=errors.mean(), line_dash="dot", line_color=CORAL,
                        annotation_text=f"Mean: {errors.mean():.4f}")
    error_fig.update_layout(template=chart_template(), height=300, margin=dict(l=12, r=12, t=25, b=12),
                            xaxis_title="Error (obs - pred)", yaxis_title="Frequency", showlegend=False)
    st.plotly_chart(error_fig, use_container_width=True, config={"displaylogo": False})

with tab2:
    residuals = (obs_eval - pred_eval)
    res_fig = make_subplots(rows=2, cols=2, column_widths=[0.6, 0.4], row_heights=[0.6, 0.4],
                            subplot_titles=("Residuals over time", "Predicted vs Actual",
                                            "Residual distribution", "Q-Q plot"),
                            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                                   [{"secondary_y": False}, {"secondary_y": False}]])
    
    res_fig.add_trace(go.Scatter(x=sampled.loc[obs_eval.index, timestamp_column], y=residuals,
                                 mode="lines", name="Residual", line=dict(color=CORAL, width=1.2)), row=1, col=1)
    res_fig.add_hline(y=0, line_dash="dot", line_color="#9AA7B2", row=1, col=1)
    
    res_fig.add_trace(go.Scatter(x=obs_eval, y=pred_eval, mode="markers", name="Pred vs actual",
                                 marker=dict(color=BLUE, size=5, opacity=0.5, line=dict(width=0))), row=1, col=2)
    if len(obs_eval):
        lo = float(min(obs_eval.min(), pred_eval.min()))
        hi = float(max(obs_eval.max(), pred_eval.max()))
        res_fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Ideal",
                                     line=dict(color="#9AA7B2", dash="dot")), row=1, col=2)
    
    res_fig.add_trace(go.Histogram(x=residuals, nbinsx=40, marker_color=PURPLE, opacity=0.7,
                                    name="Residuals"), row=2, col=1)
    res_fig.add_vline(x=0, line_dash="dash", line_color="#9AA7B2", row=2, col=1)
    
    from scipy import stats
    if len(residuals) > 3:
        qq_x, qq_y = stats.probplot(residuals.dropna(), dist="norm")
        res_fig.add_trace(go.Scatter(x=qq_x[0], y=qq_x[1], mode="markers",
                                     marker=dict(color=ACCENT, size=4, opacity=0.6),
                                     name="Sample"), row=2, col=2)
        res_fig.add_trace(go.Scatter(x=qq_x[0], y=qq_x[0], mode="lines",
                                     line=dict(color="#9AA7B2", dash="dot"), name="Normal"), row=2, col=2)
    
    res_fig.update_layout(template=chart_template(), height=600, margin=dict(l=12, r=12, t=40, b=12), showlegend=False)
    st.plotly_chart(res_fig, use_container_width=True, config={"displaylogo": False})

with tab3:
    st.markdown("#### SHAP Feature Importance & Explanations")
    st.caption("SHAP values show how each feature pushes predictions away from the baseline.")
    
    try:
        import shap
        with st.spinner("Computing SHAP values..."):
            X_scaled = scaler.transform(sampled[feature_columns].to_numpy())
            
            if hasattr(model, "predict"):
                if hasattr(model, "feature_importances_"):
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X_scaled)
                else:
                    explainer = shap.Explainer(model, X_scaled[:100])
                    shap_values = explainer(X_scaled)
                
                if isinstance(shap_values, list):
                    shap_values = shap_values[0]
                
                mean_abs_shap = np.abs(shap_values).mean(axis=0)
                shap_importance = pd.DataFrame({
                    "feature": feature_columns,
                    "mean_abs_shap": mean_abs_shap
                }).sort_values("mean_abs_shap", ascending=False)
                
                top_n = st.slider("Top features to show", 5, min(30, len(feature_columns)), 15)
                top_features = shap_importance.head(top_n)
                
                fig_importance = go.Figure(go.Bar(
                    x=top_features["mean_abs_shap"][::-1],
                    y=top_features["feature"][::-1],
                    orientation="h",
                    marker_color=ACCENT,
                    text=[f"{v:.4f}" for v in top_features["mean_abs_shap"][::-1]],
                    textposition="outside"
                ))
                fig_importance.update_layout(template=chart_template(), 
                                            height=max(400, 25 * top_n),
                                            margin=dict(l=12, r=12, t=25, b=12),
                                            xaxis_title="Mean |SHAP|", yaxis_title="")
                st.plotly_chart(fig_importance, use_container_width=True, config={"displaylogo": False})
                
                st.markdown("#### SHAP Beeswarm (feature impact distribution)")
                sample_size = min(500, len(shap_values))
                sample_idx = np.random.choice(len(shap_values), sample_size, replace=False)
                shap_sample = shap_values[sample_idx]
                
                beeswarm_data = []
                for i, feat in enumerate(shap_importance.head(top_n)["feature"]):
                    feat_idx = feature_columns.index(feat)
                    vals = shap_sample[:, feat_idx]
                    beeswarm_data.append(pd.DataFrame({
                        "feature": feat,
                        "shap_value": vals,
                        "feature_value": X_scaled[sample_idx, feat_idx]
                    }))
                beeswarm_df = pd.concat(beeswarm_data)
                
                beeswarm_fig = go.Figure()
                for i, feat in enumerate(shap_importance.head(top_n)["feature"][::-1]):
                    feat_data = beeswarm_df[beeswarm_df["feature"] == feat]
                    beeswarm_fig.add_trace(go.Violin(
                        x=feat_data["shap_value"],
                        y=[feat] * len(feat_data),
                        orientation="h",
                        side="positive",
                        width=0.8,
                        line_color=ACCENT,
                        fillcolor=ACCENT,
                        opacity=0.4,
                        showlegend=False
                    ))
                    beeswarm_fig.add_trace(go.Violin(
                        x=-feat_data["shap_value"],
                        y=[feat] * len(feat_data),
                        orientation="h",
                        side="negative",
                        width=0.8,
                        line_color=BLUE,
                        fillcolor=BLUE,
                        opacity=0.4,
                        showlegend=False
                    ))
                beeswarm_fig.add_vline(x=0, line_dash="dash", line_color="#9AA7B2")
                beeswarm_fig.update_layout(template=chart_template(), height=max(500, 30 * top_n),
                                          margin=dict(l=12, r=12, t=25, b=12),
                                          xaxis_title="SHAP value", yaxis_title="")
                st.plotly_chart(beeswarm_fig, use_container_width=True, config={"displaylogo": False})
                
                st.markdown("#### SHAP Dependence (top feature interactions)")
                top_feat = shap_importance.iloc[0]["feature"]
                feat_idx = feature_columns.index(top_feat)
                
                dep_fig = go.Figure()
                dep_fig.add_trace(go.Scatter(
                    x=X_scaled[:, feat_idx],
                    y=shap_values[:, feat_idx],
                    mode="markers",
                    marker=dict(color=shap_values[:, feat_idx], colorscale="RdBu", 
                               size=4, opacity=0.5, showscale=True, colorbar=dict(title="SHAP")),
                    name=top_feat
                ))
                dep_fig.update_layout(template=chart_template(), height=400,
                                     margin=dict(l=12, r=12, t=25, b=12),
                                     xaxis_title=f"Feature value: {top_feat}",
                                     yaxis_title="SHAP value")
                st.plotly_chart(dep_fig, use_container_width=True, config={"displaylogo": False})
                
    except ImportError:
        st.warning("SHAP not installed. Install with `pip install shap` for explainability.")
    except Exception as e:
        st.error(f"SHAP computation failed: {e}")

with tab4:
    st.markdown("#### Model Feature Importance (built-in)")
    importance = getattr(model, "feature_importances_", None)
    if importance is not None and len(importance) == len(feature_columns):
        importance_frame = (
            pd.DataFrame({"feature": feature_columns, "importance": importance})
            .sort_values("importance")
            .tail(30)
        )
        importance_chart = go.Figure(go.Bar(
            x=importance_frame["importance"],
            y=importance_frame["feature"],
            orientation="h",
            marker_color=ACCENT,
            text=[f"{v:.4f}" for v in importance_frame["importance"]],
            textposition="outside"
        ))
        importance_chart.update_layout(template=chart_template(), height=max(500, 22 * len(importance_frame)),
                                       margin=dict(l=12, r=12, t=25, b=12), xaxis_title="Model importance")
        st.plotly_chart(importance_chart, use_container_width=True, config={"displaylogo": False})
    else:
        st.info("Built-in feature importance not available for this model type.")
    
    st.markdown("#### Feature correlation with target")
    target_corr = sampled[feature_columns + [target_column]].corr()[target_column].drop(target_column)
    target_corr = target_corr.sort_values(key=abs, ascending=False).head(20)
    
    corr_fig = go.Figure(go.Bar(
        x=target_corr.values[::-1],
        y=target_corr.index[::-1],
        orientation="h",
        marker_color=[CORAL if v < 0 else ACCENT for v in target_corr.values[::-1]],
        text=[f"{v:.3f}" for v in target_corr.values[::-1]],
        textposition="outside"
    ))
    corr_fig.update_layout(template=chart_template(), height=500,
                          margin=dict(l=12, r=12, t=25, b=12),
                          xaxis_title=f"Correlation with {target_column}", yaxis_title="")
    corr_fig.add_vline(x=0, line_dash="dash", line_color="#9AA7B2")
    st.plotly_chart(corr_fig, use_container_width=True, config={"displaylogo": False})
    
    st.markdown("#### Feature correlation heatmap (top 15)")
    top_feats = target_corr.head(15).index.tolist()
    corr_matrix = sampled[top_feats].corr()
    heatmap = go.Figure(go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        zmin=-1, zmax=1, zmid=0,
        colorscale=[[0, "#B66F73"], [0.5, "#273743"], [1, "#63A99E"]],
        colorbar=dict(title="Pearson r"),
        hovertemplate="%{y} × %{x}<br>r = %{z:.3f}<extra></extra>",
    ))
    heatmap.update_layout(template=chart_template(), height=500,
                         margin=dict(l=12, r=12, t=25, b=12))
    st.plotly_chart(heatmap, use_container_width=True, config={"displaylogo": False})

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