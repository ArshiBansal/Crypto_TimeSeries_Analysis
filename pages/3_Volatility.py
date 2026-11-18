"""Volatility and drawdown diagnostics with enhanced visualizations."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.dashboard import (
    ACCENT,
    BLUE,
    CORAL,
    PURPLE,
    ROOT,
    apply_app_style,
    chart_template,
    load_csv,
    load_dashboard_data,
    sample_for_chart,
)
from src.models.evaluator import volatility_metrics


st.set_page_config(page_title="Risk & Volatility", page_icon="~", layout="wide")
frame, timestamp_column = load_dashboard_data()
apply_app_style()
if frame is None:
    st.stop()

st.markdown(
    """
    <div class="hero">
      <div class="hero-kicker">03 / RISK LENS</div>
      <h1>Volatility has a rhythm.</h1>
      <p>Compare rolling feature estimates with exported conditional volatility.
         Treat these as descriptive/diagnostic measures, not guarantees.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

volatility_columns = [c for c in frame.columns if "rolling_vol" in c.lower()]

tab1, tab2, tab3, tab4 = st.tabs(["📊 Rolling Volatility", "🔮 Conditional Vol (GARCH)", "📉 Drawdown Analysis", "📈 Vol Regime & Stats"])

with tab1:
    if volatility_columns:
        selected_columns = st.multiselect("Rolling volatility features", volatility_columns,
                                          default=volatility_columns[: min(3, len(volatility_columns))])
        chart_frame = sample_for_chart(frame, st.session_state.get("max_pts", 5000))
        if selected_columns:
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                                row_heights=[0.7, 0.3],
                                subplot_titles=("Rolling Volatility Over Time", "Volatility Distribution"))
            
            colors = [ACCENT, BLUE, PURPLE, CORAL, "#D4B483", "#8BC6A8"]
            for color, column in zip(colors * 4, selected_columns):
                fig.add_trace(go.Scatter(x=chart_frame[timestamp_column],
                                         y=pd.to_numeric(chart_frame[column], errors="coerce"),
                                         name=column, mode="lines", line=dict(color=color, width=1.8)), row=1, col=1)
            
            for color, column in zip(colors * 4, selected_columns):
                vals = pd.to_numeric(chart_frame[column], errors="coerce").dropna()
                fig.add_trace(go.Histogram(x=vals, nbinsx=40, name=column,
                                           marker_color=color, opacity=0.5, showlegend=False), row=2, col=1)
            
            fig.update_layout(template=chart_template(), height=600, margin=dict(l=12, r=12, t=40, b=12),
                              hovermode="x unified", yaxis_title="Volatility", yaxis2_title="Frequency")
            fig.update_xaxes(showgrid=False)
            st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
            
            st.markdown("#### Volatility Statistics")
            stats_cols = st.columns(len(selected_columns) if selected_columns else 1)
            for i, col in enumerate(selected_columns):
                vals = pd.to_numeric(frame[col], errors="coerce").dropna()
                with stats_cols[i % len(stats_cols)]:
                    st.metric(col, f"{vals.mean():.4f}",
                              delta=f"σ={vals.std():.4f} | max={vals.max():.4f}")
        else:
            st.info("Select volatility features to visualize.")
    else:
        st.info("No rolling-volatility features were found in this dataset.")

with tab2:
    garch_path = ROOT / "artifacts" / "models" / "garch_cond_vol.csv"
    if garch_path.exists():
        garch_frame = load_csv(str(garch_path))
        if {"timestamp", "cond_vol"}.issubset(garch_frame.columns):
            garch_frame["timestamp"] = pd.to_datetime(garch_frame["timestamp"], errors="coerce", utc=True)
            garch_frame["cond_vol"] = pd.to_numeric(garch_frame["cond_vol"], errors="coerce")
            garch_frame = garch_frame.dropna(subset=["timestamp", "cond_vol"])
            start, end = frame[timestamp_column].min(), frame[timestamp_column].max()
            garch_frame = garch_frame[garch_frame["timestamp"].between(start, end)]
            if not garch_frame.empty:
                gcf = sample_for_chart(garch_frame, st.session_state.get("max_pts", 5000))
                
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                                    row_heights=[0.7, 0.3],
                                    subplot_titles=("GARCH Conditional Volatility", "Volatility Distribution"))
                
                fig.add_trace(go.Scatter(x=gcf["timestamp"], y=gcf["cond_vol"], name="Conditional volatility",
                                         mode="lines", line=dict(color=BLUE, width=1.8),
                                         fill="tozeroy", fillcolor="rgba(131,169,212,0.10)"), row=1, col=1)
                
                fig.add_trace(go.Histogram(x=gcf["cond_vol"], nbinsx=50, name="Distribution",
                                           marker_color=BLUE, opacity=0.7), row=2, col=1)
                
                fig.update_layout(template=chart_template(), height=600, margin=dict(l=12, r=12, t=40, b=12),
                                  hovermode="x unified", yaxis_title="Conditional Volatility", yaxis2_title="Frequency")
                fig.update_xaxes(showgrid=False)
                st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
                
                realized_col = next((c for c in ("rolling_vol_30", "realized_vol", "rv_30") if c in frame.columns), None)
                if realized_col is None:
                    non_ann = [c for c in volatility_columns if "ann" not in c.lower()]
                    realized_col = non_ann[0] if non_ann else None
                if realized_col:
                    merged = pd.merge_asof(
                        garch_frame.sort_values("timestamp"),
                        frame[[timestamp_column, realized_col]].sort_values(timestamp_column).rename(columns={timestamp_column: "timestamp"}),
                        on="timestamp", direction="nearest", tolerance=pd.Timedelta("1D"),
                    ).dropna(subset=["cond_vol", realized_col])
                    if len(merged) >= 10:
                        pred = pd.to_numeric(merged["cond_vol"], errors="coerce").to_numpy(dtype=float)
                        truth = pd.to_numeric(merged[realized_col], errors="coerce").to_numpy(dtype=float)
                        pred_cmp = pred.copy()
                        if np.nanmedian(truth) > 0 and np.nanmedian(pred) > 0:
                            scale = np.nanmedian(truth) / np.nanmedian(pred)
                            if 0.2 < scale < 5:
                                pred_cmp = pred * scale
                        vm = volatility_metrics(truth, np.abs(pred_cmp))
                        
                        st.markdown("#### GARCH vs Realized Volatility Comparison")
                        comp_fig = go.Figure()
                        comp_fig.add_trace(go.Scatter(x=merged["timestamp"], y=truth, name=realized_col,
                                                      mode="lines", line=dict(color=ACCENT, width=1.6)))
                        comp_fig.add_trace(go.Scatter(x=merged["timestamp"], y=pred_cmp, name="GARCH (scaled)",
                                                      mode="lines", line=dict(color=BLUE, width=1.6, dash="dot")))
                        comp_fig.update_layout(template=chart_template(), height=350,
                                              margin=dict(l=12, r=12, t=25, b=12), hovermode="x unified",
                                              yaxis_title="Volatility")
                        comp_fig.update_xaxes(showgrid=False)
                        st.plotly_chart(comp_fig, use_container_width=True, config={"displaylogo": False})
                        
                        cols = st.columns(4)
                        cols[0].metric("Realized proxy", realized_col)
                        cols[1].metric("MAE", f"{vm['mae']:.4g}" if np.isfinite(vm.get('mae', np.nan)) else "—")
                        cols[2].metric("RMSE", f"{vm['rmse']:.4g}" if np.isfinite(vm.get('rmse', np.nan)) else "—")
                        cols[3].metric("QLIKE", f"{vm['qlike']:.4g}" if np.isfinite(vm.get('qlike', np.nan)) else "—",
                                       help="QLIKE is preferred for volatility forecast comparison on positive values.")
                        st.caption("Comparison is diagnostic only (proxy realized vol and possible scale/unit differences).")
                        
                        scatter_fig = go.Figure()
                        scatter_fig.add_trace(go.Scatter(x=truth, y=pred_cmp, mode="markers",
                                                         marker=dict(color=BLUE, size=5, opacity=0.5),
                                                         name="Observations"))
                        lo = min(truth.min(), pred_cmp.min())
                        hi = max(truth.max(), pred_cmp.max())
                        scatter_fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                                                         line=dict(color="#9AA7B2", dash="dot"), name="Ideal"))
                        scatter_fig.update_layout(template=chart_template(), height=350,
                                                 margin=dict(l=12, r=12, t=25, b=12),
                                                 xaxis_title=realized_col, yaxis_title="GARCH Conditional Vol")
                        st.plotly_chart(scatter_fig, use_container_width=True, config={"displaylogo": False})
            else:
                st.info("The exported volatility series does not overlap this selected date range.")
        else:
            st.error("Volatility export must contain `timestamp` and `cond_vol` columns.")
    else:
        st.info("No conditional volatility export is available. Run notebook 03 to generate this series.")

with tab3:
    price_column = next((c for c in ("close", "price", "adj_close") if c in frame.columns), None)
    if price_column:
        close = pd.to_numeric(frame[price_column], errors="coerce")
        drawdown = close / close.cummax() - 1
        drawdown_frame = sample_for_chart(frame, st.session_state.get("max_pts", 5000))
        
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                            row_heights=[0.4, 0.3, 0.3],
                            subplot_titles=("Price", "Drawdown from Peak", "Underwater Plot"))
        
        fig.add_trace(go.Scatter(x=drawdown_frame[timestamp_column],
                                 y=pd.to_numeric(drawdown_frame[price_column], errors="coerce"),
                                 name="Price", mode="lines", line=dict(color=ACCENT, width=1.5)), row=1, col=1)
        
        fig.add_trace(go.Scatter(x=drawdown_frame[timestamp_column],
                                 y=drawdown.loc[drawdown_frame.index] * 100,
                                 name="Drawdown", mode="lines", line=dict(color=CORAL, width=1.5),
                                 fill="tozeroy", fillcolor="rgba(217,142,135,0.12)"), row=2, col=1)
        
        underwater = drawdown * 100
        fig.add_trace(go.Scatter(x=drawdown_frame[timestamp_column],
                                 y=underwater.loc[drawdown_frame.index],
                                 name="Underwater", mode="lines", line=dict(color=PURPLE, width=1.2),
                                 fill="tozeroy", fillcolor="rgba(166,154,200,0.1)"), row=3, col=1)
        
        fig.update_layout(template=chart_template(), height=700, margin=dict(l=12, r=12, t=40, b=12),
                          hovermode="x unified", showlegend=False)
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown %", row=2, col=1)
        fig.update_yaxes(title_text="Underwater %", row=3, col=1)
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
        
        st.markdown("#### Drawdown Statistics")
        dd_vals = drawdown.dropna() * 100
        dd_cols = st.columns(5)
        dd_cols[0].metric("Max Drawdown", f"{dd_vals.min():.2f}%")
        dd_cols[1].metric("Current Drawdown", f"{dd_vals.iloc[-1]:.2f}%")
        dd_cols[2].metric("Avg Drawdown", f"{dd_vals.mean():.2f}%")
        dd_cols[3].metric("Time in Drawdown", f"{(dd_vals < 0).mean()*100:.1f}%")
        dd_cols[4].metric("Worst 5%", f"{dd_vals.quantile(0.05):.2f}%")
        
        st.markdown("#### Top 10 Drawdown Periods")
        dd_periods = []
        in_dd = False
        start_idx = None
        for i, (ts, val) in enumerate(drawdown.items()):
            if val < 0 and not in_dd:
                in_dd = True
                start_idx = i
            elif val >= 0 and in_dd:
                in_dd = False
                end_idx = i - 1
                period_dd = drawdown.iloc[start_idx:end_idx+1]
                dd_periods.append({
                    "Start": period_dd.index[0],
                    "End": period_dd.index[-1],
                    "Duration": len(period_dd),
                    "Max DD": period_dd.min() * 100,
                    "Recovery": period_dd.iloc[-1] * 100 if period_dd.iloc[-1] >= 0 else "—"
                })
        if in_dd and start_idx is not None:
            period_dd = drawdown.iloc[start_idx:]
            dd_periods.append({
                "Start": period_dd.index[0],
                "End": period_dd.index[-1],
                "Duration": len(period_dd),
                "Max DD": period_dd.min() * 100,
                "Recovery": "Ongoing"
            })
        
        if dd_periods:
            dd_df = pd.DataFrame(dd_periods).sort_values("Max DD").head(10)
            st.dataframe(dd_df, use_container_width=True, hide_index=True)
    else:
        st.info("No price column available for drawdown analysis.")

with tab4:
    st.markdown("#### Volatility Regime Analysis")
    if volatility_columns:
        vol_col = st.selectbox("Select volatility feature for regime analysis", volatility_columns, index=0)
        vol_series = pd.to_numeric(frame[vol_col], errors="coerce").dropna()
        
        if len(vol_series) > 20:
            from sklearn.mixture import GaussianMixture
            
            n_regimes = st.slider("Number of volatility regimes", 2, 5, 3)
            X = vol_series.values.reshape(-1, 1)
            gmm = GaussianMixture(n_components=n_regimes, random_state=42, covariance_type="full")
            labels = gmm.fit_predict(X)
            
            regime_colors = [ACCENT, BLUE, PURPLE, CORAL, "#D4B483"]
            
            fig = go.Figure()
            for i in range(n_regimes):
                mask = labels == i
                regime_data = vol_series.iloc[mask]
                fig.add_trace(go.Scatter(x=regime_data.index, y=regime_data.values,
                                         mode="markers", name=f"Regime {i}",
                                         marker=dict(color=regime_colors[i % len(regime_colors)], 
                                                     size=4, opacity=0.6)))
            fig.update_layout(template=chart_template(), height=400,
                             margin=dict(l=12, r=12, t=25, b=12),
                             xaxis_title="Time Index", yaxis_title=vol_col,
                             hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
            
            st.markdown("#### Regime Statistics")
            regime_stats = []
            for i in range(n_regimes):
                mask = labels == i
                regime_data = vol_series.iloc[mask]
                regime_stats.append({
                    "Regime": i,
                    "Count": int(mask.sum()),
                    "Share": f"{mask.mean()*100:.1f}%",
                    "Mean": f"{regime_data.mean():.4f}",
                    "Std": f"{regime_data.std():.4f}",
                    "Min": f"{regime_data.min():.4f}",
                    "Max": f"{regime_data.max():.4f}",
                    "Persistence": f"{(mask == np.roll(mask, 1)).mean():.3f}"
                })
            st.dataframe(pd.DataFrame(regime_stats), use_container_width=True, hide_index=True)
            
            st.markdown("#### Volatility Term Structure (Rolling Windows)")
            vol_windows = [c for c in volatility_columns if c.startswith("rolling_vol_")]
            if len(vol_windows) > 1:
                vol_windows = sorted(vol_windows, key=lambda x: int(x.split("_")[-1]))
                term_data = frame[vol_windows].dropna()
                if len(term_data) > 0:
                    term_fig = go.Figure()
                    for i, w in enumerate(vol_windows):
                        term_fig.add_trace(go.Scatter(x=term_data.index, y=term_data[w],
                                                      mode="lines", name=w,
                                                      line=dict(color=regime_colors[i % len(regime_colors)], width=1.2)))
                    term_fig.update_layout(template=chart_template(), height=350,
                                          margin=dict(l=12, r=12, t=25, b=12),
                                          xaxis_title="Time", yaxis_title="Volatility",
                                          hovermode="x unified")
                    st.plotly_chart(term_fig, use_container_width=True, config={"displaylogo": False})
                    
                    st.markdown("#### Average Term Structure")
                    avg_term = term_data.mean()
                    term_bar = go.Figure(go.Bar(x=avg_term.index, y=avg_term.values,
                                                marker_color=ACCENT, text=[f"{v:.4f}" for v in avg_term.values],
                                                textposition="outside"))
                    term_bar.update_layout(template=chart_template(), height=300,
                                          margin=dict(l=12, r=12, t=25, b=12),
                                          xaxis_title="Window", yaxis_title="Average Volatility")
                    st.plotly_chart(term_bar, use_container_width=True, config={"displaylogo": False})
    else:
        st.info("No volatility features available for regime analysis.")

st.caption("Rolling volatility depends on bar interval and is not annualized unless named otherwise. "
           "For crypto, fat-tailed innovations (e.g. Student-t) are often more appropriate than normal GARCH assumptions.")