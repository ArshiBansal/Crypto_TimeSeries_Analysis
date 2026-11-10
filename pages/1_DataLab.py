"""Deep feature exploration and data-quality diagnostics."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

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

numeric_columns = frame.select_dtypes(include="number").columns.tolist()
selected_file = st.session_state.get("_selected_file", "prepared dataset")
max_points = st.session_state.get("max_pts", 5000)
light_canvas = st.session_state.get("light_mode", False)
grid_color = "#E0E7EB" if light_canvas else "#293743"

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

feature_tab, relationship_tab, quality_tab = st.tabs(
    ["FEATURE STUDIO", "RELATIONSHIPS", "DATA HEALTH"]
)

with feature_tab:
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
            ["Timeline", "Distribution"],
            horizontal=True,
        )

    if selected_feature is not None:
        chart_frame = sample_for_chart(frame, max_points)
        values = pd.to_numeric(chart_frame[selected_feature], errors="coerce")
        feature_chart = go.Figure()
        if chart_kind == "Timeline":
            feature_chart.add_trace(
                go.Scatter(
                    x=chart_frame[timestamp_column],
                    y=values,
                    mode="lines",
                    name=selected_feature,
                    line=dict(color=ACCENT, width=2),
                    connectgaps=False,
                )
            )
            feature_chart.update_xaxes(title_text="Observation time")
            feature_chart.update_yaxes(title_text=selected_feature)
        else:
            feature_chart.add_trace(
                go.Histogram(
                    x=values.dropna(),
                    nbinsx=65,
                    name=selected_feature,
                    marker_color=BLUE,
                    opacity=0.82,
                )
            )
            feature_chart.update_xaxes(title_text=selected_feature)
            feature_chart.update_yaxes(title_text="Observations")

        feature_chart.update_layout(
            template=chart_template(),
            height=390,
            margin=dict(l=12, r=12, t=18, b=12),
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
            stats = pd.to_numeric(frame[selected_feature], errors="coerce").describe()
            stat_columns = st.columns(5)
            for column, label in zip(
                stat_columns,
                ["Mean", "Std dev", "Median", "Min", "Max"],
            ):
                key = {
                    "Mean": "mean",
                    "Std dev": "std",
                    "Median": "50%",
                    "Min": "min",
                    "Max": "max",
                }[label]
                value = stats.get(key, float("nan"))
                column.metric(label, f"{value:,.4g}" if pd.notna(value) else "—")

    st.markdown("#### Compare feature shapes")
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
        compare_chart = go.Figure()
        colors = [ACCENT, BLUE, PURPLE, CORAL]
        for color, column in zip(colors, compared):
            series = pd.to_numeric(compare_frame[column], errors="coerce")
            deviation = series.std()
            normalized = (
                (series - series.mean()) / deviation
                if pd.notna(deviation) and deviation > 0
                else series * 0
            )
            compare_chart.add_trace(
                go.Scatter(
                    x=compare_frame[timestamp_column],
                    y=normalized,
                    name=column,
                    mode="lines",
                    line=dict(color=color, width=1.8),
                    connectgaps=False,
                )
            )
        compare_chart.update_layout(
            template=chart_template(),
            height=340,
            margin=dict(l=12, r=12, t=20, b=10),
            yaxis_title="Standard deviations from mean",
            hovermode="x unified",
        )
        compare_chart.update_xaxes(showgrid=False)
        compare_chart.update_yaxes(gridcolor=grid_color)
        st.plotly_chart(
            compare_chart,
            use_container_width=True,
            config={"displaylogo": False},
        )

with relationship_tab:
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
                height=max(440, 34 * len(correlation_features)),
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
        else:
            st.info("Select at least two features to draw the correlation map.")

with quality_tab:
    st.markdown("#### Missingness map")
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
        st.markdown("#### Numeric profile")
        profile = frame[numeric_columns].describe(percentiles=[0.25, 0.5, 0.75]).T
        profile.insert(0, "missing %", frame[numeric_columns].isna().mean() * 100)
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
