"""Deep feature exploration and data-quality diagnostics with enhanced visualizations."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy.stats import norm

from src.dashboard import (
    ACCENT,
    BLUE,
    CORAL,
    PURPLE,
    apply_app_style,
    chart_template,
    load_dashboard_data,
    sample_for_chart,
)


st.set_page_config(page_title="Data Lab", page_icon="◫", layout="wide")
frame, timestamp_column = load_dashboard_data()
apply_app_style()

if frame is None:
    st.stop()

numeric_columns = frame.select_dtypes(include="number").columns.tolist()
selected_file = st.session_state.get("_selected_file", "prepared dataset")
max_points = st.session_state.get("max_pts", 5000)
grid_color = "#293743"

st.markdown(
    """
    <div class="hero">
      <div class="hero-kicker">01 &nbsp; / &nbsp; DATA LAB</div>
      <h1>Find the shape of a feature.</h1>
      <p>Inspect distributions, compare scales, test relationships and check
         data quality without leaving the selected market window.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


def feature_group(column: str) -> str:
    name = column.lower()
    if name.startswith("target_"):
        return "Targets"
    if any(token in name for token in ("return", "log_return")):
        return "Returns"
    if "vol" in name or "risk" in name:
        return "Volatility"
    if "volume" in name:
        return "Volume"
    if any(token in name for token in ("momentum", "slope", "drawdown", "range")):
        return "Trend & structure"
    if name in {"open", "high", "low", "close", "price"}:
        return "Market bars"
    return "Other features"


grouped_features: dict[str, list[str]] = {}
for column in numeric_columns:
    grouped_features.setdefault(feature_group(column), []).append(column)

overview_columns = st.columns(4)
missing_cells = int(frame.isna().sum().sum())
duplicate_times = int(frame[timestamp_column].duplicated().sum())
completeness = (
    100 * (1 - missing_cells / (frame.shape[0] * frame.shape[1]))
    if frame.size
    else 0
)
overview_columns[0].metric("Numeric features", f"{len(numeric_columns):,}")
overview_columns[1].metric("Completeness", f"{completeness:.2f}%")
overview_columns[2].metric("Missing cells", f"{missing_cells:,}")
overview_columns[3].metric("Repeated timestamps", f"{duplicate_times:,}")

tab1, tab2, tab3 = st.tabs(["📊 Feature Studio", "🔗 Relationships", "🏥 Data Health"])

with tab1:
    controls, chart_area = st.columns([1, 2.5], gap="large")
    with controls:
        category_options = list(grouped_features)
        active_groups = st.multiselect(
            "Feature families",
            category_options,
            default=category_options,
        )
        search = st.text_input("Find a feature", placeholder="e.g. volatility, lag_1")
        candidates = [
            column
            for group in active_groups
            for column in grouped_features[group]
            if search.casefold() in column.casefold()
        ]
        if not candidates:
            st.info("No features match these filters.")
            selected_feature = None
        else:
            selected_feature = st.selectbox("Feature", candidates)
        chart_kind = st.radio(
            "Lens",
            ["Timeline", "Distribution", "Both"],
            horizontal=True,
        )

    if selected_feature is not None:
        chart_frame = sample_for_chart(frame, max_points)
        values = pd.to_numeric(chart_frame[selected_feature], errors="coerce")
        full_values = pd.to_numeric(frame[selected_feature], errors="coerce").dropna()
        
        if chart_kind == "Timeline":
            feature_chart = go.Figure()
            feature_chart.add_trace(
                go.Scatter(
                    x=chart_frame[timestamp_column],
                    y=values,
                    mode="lines",
                    name=selected_feature,
                    line=dict(color=ACCENT, width=1.5),
                    connectgaps=False,
                )
            )
            feature_chart.update_xaxes(title_text="Observation time")
            feature_chart.update_yaxes(title_text=selected_feature)
            feature_chart.update_layout(
                template=chart_template(),
                height=450,
                margin=dict(l=12, r=12, t=25, b=12),
                showlegend=False,
                hovermode="x unified",
            )
            feature_chart.update_xaxes(showgrid=False)
            feature_chart.update_yaxes(gridcolor=grid_color)
            with chart_area:
                st.plotly_chart(
                    feature_chart,
                    use_container_width=True,
                    config={"displaylogo": False},
                )

        elif chart_kind == "Distribution":
            dist_fig = make_subplots(rows=2, cols=2, subplot_titles=("Histogram", "Box Plot", "ECDF", "Violin"))
            
            dist_fig.add_trace(go.Histogram(x=full_values, nbinsx=60, name=selected_feature,
                                            marker_color=BLUE, opacity=0.7), row=1, col=1)
            dist_fig.add_vline(x=full_values.mean(), line_dash="dash", line_color=ACCENT, row=1, col=1)
            dist_fig.add_vline(x=full_values.median(), line_dash="dot", line_color=PURPLE, row=1, col=1)
            
            dist_fig.add_trace(go.Box(y=full_values, name=selected_feature,
                                      marker_color=BLUE, boxpoints="outliers"), row=1, col=2)
            
            ecdf_x = np.sort(full_values)
            ecdf_y = np.arange(1, len(ecdf_x) + 1) / len(ecdf_x)
            dist_fig.add_trace(go.Scatter(x=ecdf_x, y=ecdf_y, mode="lines", name="ECDF",
                                          line=dict(color=PURPLE, width=2)), row=2, col=1)
            
            dist_fig.add_trace(go.Violin(y=full_values, name=selected_feature, side="positive",
                                         width=0.8, line_color=ACCENT, fillcolor=ACCENT, opacity=0.4,
                                         box_visible=True, meanline_visible=True), row=2, col=2)
            
            dist_fig.update_layout(template=chart_template(), height=550,
                                   margin=dict(l=12, r=12, t=40, b=12), showlegend=False)
            with chart_area:
                st.plotly_chart(dist_fig, use_container_width=True, config={"displaylogo": False})

        else:  # Both
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                                row_heights=[0.6, 0.4],
                                subplot_titles=(f"{selected_feature} - Timeline", f"{selected_feature} - Distribution"))
            
            fig.add_trace(go.Scatter(x=chart_frame[timestamp_column], y=values,
                                     mode="lines", name=selected_feature,
                                     line=dict(color=ACCENT, width=1.5), connectgaps=False), row=1, col=1)
            
            fig.add_trace(go.Histogram(x=full_values, nbinsx=60, name=selected_feature,
                                       marker_color=BLUE, opacity=0.7), row=2, col=1)
            fig.add_vline(x=full_values.mean(), line_dash="dash", line_color=ACCENT, row=2, col=1)
            fig.add_vline(x=full_values.median(), line_dash="dot", line_color=PURPLE, row=2, col=1)
            
            fig.update_layout(template=chart_template(), height=650,
                             margin=dict(l=12, r=12, t=40, b=12), hovermode="x unified",
                             showlegend=False)
            fig.update_xaxes(showgrid=False)
            fig.update_yaxes(gridcolor=grid_color)
            with chart_area:
                st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

        stats = full_values.describe()
        stat_columns = st.columns(4)
        for column, label in zip(
            stat_columns,
            ["Mean", "Std dev", "Median", "Min"],
        ):
            key = {
                "Mean": "mean",
                "Std dev": "std",
                "Median": "50%",
                "Min": "min",
            }[label]
            value = stats.get(key, float("nan"))
            column.metric(label, f"{value:,.4g}" if pd.notna(value) else "—")
        
        st.markdown("#### Extended Statistics")
        ext_cols = st.columns(4)
        ext_cols[0].metric("Max", f"{stats.get('max', float('nan')):,.4g}")
        ext_cols[1].metric("Skew", f"{full_values.skew():.3f}")
        ext_cols[2].metric("Kurtosis", f"{full_values.kurtosis():.3f}")
        ext_cols[3].metric("IQR", f"{(stats.get('75%', 0) - stats.get('25%', 0)):,.4g}")
        
        st.markdown("#### Percentiles")
        pct_cols = st.columns(5)
        pcts = full_values.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
        for i, (p, v) in enumerate(pcts.items()):
            pct_cols[i % 5].metric(f"{p:.0%}", f"{v:,.4g}")

    st.markdown("#### Compare Feature Shapes")
    compare_candidates = [
        column for column in numeric_columns if not column.startswith("target_")
    ]
    compared = st.multiselect(
        "Overlay up to four features",
        compare_candidates,
        default=[],
        max_selections=4,
        help="Each selected series is standardized so shape—not unit—drives the comparison.",
    )
    if compared:
        compare_frame = sample_for_chart(frame, max_points)
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                            row_heights=[0.6, 0.4],
                            subplot_titles=("Standardized Timeline", "Distribution Overlay"))
        colors = [ACCENT, BLUE, PURPLE, CORAL]
        for color, column in zip(colors, compared):
            series = pd.to_numeric(compare_frame[column], errors="coerce")
            deviation = series.std()
            normalized = (
                (series - series.mean()) / deviation
                if pd.notna(deviation) and deviation > 0
                else series * 0
            )
            fig.add_trace(
                go.Scatter(
                    x=compare_frame[timestamp_column],
                    y=normalized,
                    name=column,
                    mode="lines",
                    line=dict(color=color, width=1.8),
                    connectgaps=False,
                ), row=1, col=1)
            
            full_series = pd.to_numeric(frame[column], errors="coerce").dropna()
            if len(full_series) > 0:
                norm_full = (full_series - full_series.mean()) / full_series.std()
                fig.add_trace(go.Histogram(x=norm_full, nbinsx=40, name=column,
                                           marker_color=color, opacity=0.4), row=2, col=1)
        
        fig.update_layout(template=chart_template(), height=600,
                         margin=dict(l=12, r=12, t=40, b=12),
                         hovermode="x unified", yaxis_title="Std deviations", yaxis2_title="Frequency")
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(gridcolor=grid_color)
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

with tab2:
    if len(numeric_columns) < 2:
        st.info("At least two numeric features are needed to calculate correlations.")
    else:
        safe_default = numeric_columns[: min(12, len(numeric_columns))]
        correlation_features = st.multiselect(
            "Choose up to 20 numeric features",
            numeric_columns,
            default=safe_default,
            max_selections=20,
        )
        if len(correlation_features) >= 2:
            correlation_sample = sample_for_chart(
                frame[correlation_features],
                20000,
            )
            correlation = correlation_sample.corr()
            
            tab_heatmap, tab_network, tab_target = st.tabs(["Heatmap", "Correlation Network", "Target Correlation"])
            
            with tab_heatmap:
                heatmap = go.Figure(
                    go.Heatmap(
                        z=correlation.values,
                        x=correlation.columns,
                        y=correlation.index,
                        zmin=-1,
                        zmax=1,
                        zmid=0,
                        colorscale=[
                            [0, "#B66F73"],
                            [0.5, "#273743"],
                            [1, "#63A99E"],
                        ],
                        colorbar=dict(title="Pearson r"),
                        hovertemplate="%{y} × %{x}<br>r = %{z:.3f}<extra></extra>",
                    )
                )
                heatmap.update_layout(
                    template=chart_template(),
                    height=max(500, 34 * len(correlation_features)),
                    margin=dict(l=12, r=12, t=20, b=12),
                )
                st.plotly_chart(
                    heatmap,
                    use_container_width=True,
                    config={"displaylogo": False},
                )
                st.caption(
                    "Correlations are pairwise and descriptive. The chart samples at "
                    "most 20,000 rows for responsiveness."
                )
            
            with tab_network:
                st.markdown("#### Strong Correlations (|r| > 0.5)")
                strong_pairs = []
                for i, col1 in enumerate(correlation.columns):
                    for j, col2 in enumerate(correlation.columns):
                        if i < j:
                            r = correlation.iloc[i, j]
                            if abs(r) > 0.5:
                                strong_pairs.append({"Feature 1": col1, "Feature 2": col2, "Correlation": r})
                if strong_pairs:
                    strong_df = pd.DataFrame(strong_pairs).sort_values("Correlation", key=abs, ascending=False)
                    st.dataframe(strong_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No correlations above 0.5 threshold.")
            
            with tab_target:
                target_cols = [c for c in correlation.columns if c.startswith("target_")]
                if target_cols:
                    for target in target_cols:
                        target_corr = correlation[target].drop(target).sort_values(key=abs, ascending=False)
                        targ_fig = go.Figure(go.Bar(
                            x=target_corr.values[:20][::-1],
                            y=target_corr.index[:20][::-1],
                            orientation="h",
                            marker_color=[CORAL if v < 0 else ACCENT for v in target_corr.values[:20][::-1]],
                            text=[f"{v:.3f}" for v in target_corr.values[:20][::-1]],
                            textposition="outside"
                        ))
                        targ_fig.update_layout(template=chart_template(), height=400,
                                              margin=dict(l=12, r=12, t=25, b=12),
                                              xaxis_title=f"Correlation with {target}", yaxis_title="")
                        targ_fig.add_vline(x=0, line_dash="dash", line_color="#9AA7B2")
                        st.plotly_chart(targ_fig, use_container_width=True, config={"displaylogo": False})
                else:
                    st.info("No target columns found in correlation matrix.")
        else:
            st.info("Select at least two features to draw the correlation map.")

with tab3:
    st.markdown("#### Missingness Map")
    missing_percent = frame.isna().mean().mul(100)
    missing_percent = missing_percent[missing_percent > 0].sort_values().tail(25)
    if missing_percent.empty:
        st.success("No missing cells in the current filtered view.")
    else:
        missing_chart = go.Figure(
            go.Bar(
                x=missing_percent.values,
                y=missing_percent.index,
                orientation="h",
                marker_color=PURPLE,
                hovertemplate="%{y}: %{x:.2f}% missing<extra></extra>",
                text=[f"{v:.1f}%" for v in missing_percent.values],
                textposition="outside"
            )
        )
        missing_chart.update_layout(
            template=chart_template(),
            height=max(300, 24 * len(missing_percent)),
            margin=dict(l=12, r=12, t=12, b=12),
            xaxis_title="Missing values (%)",
            yaxis_title="",
        )
        missing_chart.update_xaxes(gridcolor=grid_color)
        st.plotly_chart(
            missing_chart,
            use_container_width=True,
            config={"displaylogo": False},
        )

    if numeric_columns:
        st.markdown("#### Numeric Profile")
        profile = frame[numeric_columns].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T
        profile.insert(0, "missing %", frame[numeric_columns].isna().mean() * 100)
        profile.insert(1, "skew", frame[numeric_columns].skew())
        profile.insert(2, "kurtosis", frame[numeric_columns].kurtosis())
        profile.index.name = "Feature"
        st.dataframe(profile, use_container_width=True)

    st.markdown("#### Schema")
    schema = pd.DataFrame(
        {
            "Column": frame.columns,
            "Type": [str(frame[column].dtype) for column in frame.columns],
            "Missing": [int(frame[column].isna().sum()) for column in frame.columns],
            "Unique": [int(frame[column].nunique(dropna=True)) for column in frame.columns],
        }
    )
    st.dataframe(schema, use_container_width=True, hide_index=True)

    with st.expander("Preview filtered observations"):
        preview_rows = st.select_slider(
            "Rows to preview",
            options=[25, 100, 250, 500],
            value=100,
        )
        st.dataframe(frame.head(preview_rows), use_container_width=True, hide_index=True)

        export_signature = (
            selected_file,
            st.session_state.get("date_filter"),
            st.session_state.get("resolution"),
        )
        if st.session_state.get("_data_lab_export_signature") != export_signature:
            st.session_state.pop("_data_lab_export_data", None)
            st.session_state["_data_lab_export_signature"] = export_signature

        if st.button(
            "Prepare CSV export",
            help="Builds a CSV of the current filtered view only when requested.",
        ):
            with st.spinner("Preparing the filtered dataset…"):
                st.session_state["_data_lab_export_data"] = frame.to_csv(
                    index=False
                ).encode("utf-8")

        export_data = st.session_state.get("_data_lab_export_data")
        if export_data is not None:
            st.download_button(
                "Download prepared CSV",
                data=export_data,
                file_name=f"{selected_file.removesuffix('.parquet')}_filtered.csv",
                mime="text/csv",
            )