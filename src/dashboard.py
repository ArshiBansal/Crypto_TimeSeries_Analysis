"""Shared data loading, filters, and visual styling for Streamlit pages."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "processed"
ARTIFACTS_DIR = ROOT / "artifacts"

ACCENT = "#6CB6AD"
BLUE = "#83A9D4"
PURPLE = "#A69AC8"
CORAL = "#D98E87"


@st.cache_data(show_spinner="Reading prepared market data…")
def load_parquet(path: str) -> pd.DataFrame:
    """Load a prepared Parquet file once and cache it for the current app process."""
    return pd.read_parquet(path)


def detect_timestamp_column(frame: pd.DataFrame) -> str | None:
    for candidate in ("timestamp", "datetime", "date", "time"):
        if candidate in frame.columns:
            return candidate

    return next(
        (
            column
            for column in frame.columns
            if pd.api.types.is_datetime64_any_dtype(frame[column])
        ),
        None,
    )


def _apply_date_preset() -> None:
    preset = st.session_state.get("date_preset", "All data")
    bounds = st.session_state.get("_date_bounds")
    if not bounds or preset == "Custom":
        return

    first, last = bounds
    if preset == "All data":
        start = first
    elif preset == "YTD":
        start = max(first, last.replace(month=1, day=1))
    else:
        days = {"1M": 30, "3M": 90, "6M": 180, "1Y": 365}.get(preset)
        if days is None:
            return
        start = max(first, (pd.Timestamp(last) - pd.Timedelta(days=days)).date())

    st.session_state["date_filter"] = (start, last)


def resample_frame(frame: pd.DataFrame, timestamp_column: str, rule: str) -> pd.DataFrame:
    if rule == "Raw":
        return frame

    aggregations: dict[str, str] = {}
    ohlcv_aggregations = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }

    for column in frame.columns:
        if column == timestamp_column:
            continue
        if column in ohlcv_aggregations:
            aggregations[column] = ohlcv_aggregations[column]
        elif column.startswith("target_"):
            aggregations[column] = "last"
        elif pd.api.types.is_numeric_dtype(frame[column]):
            aggregations[column] = "mean"
        else:
            aggregations[column] = "first"

    if not aggregations:
        return frame[[timestamp_column]].drop_duplicates().reset_index(drop=True)

    return (
        frame.set_index(timestamp_column)
        .resample(rule)
        .agg(aggregations)
        .dropna(how="all")
        .reset_index()
    )


def sample_for_chart(frame: pd.DataFrame, limit: int = 5000) -> pd.DataFrame:
    if len(frame) <= limit:
        return frame
    return frame.iloc[:: math.ceil(len(frame) / limit)]


def load_dashboard_data() -> tuple[pd.DataFrame, str]:
    """Render global controls and publish the selected frame to session state."""
    with st.sidebar:
        st.markdown("## BTC / FIELDNOTES")
        st.caption("Historical market observatory")
        st.toggle("Light canvas", key="light_mode")
        st.divider()

        parquet_files = (
            sorted(PROCESSED_DIR.glob("*.parquet"))
            if PROCESSED_DIR.exists()
            else []
        )
        if not parquet_files:
            st.session_state["df"] = None
            st.session_state["ts_col"] = "timestamp"
            st.info("No prepared Parquet data is available.")
            st.caption("Run notebooks 01 and 02 to create the dashboard datasets.")
            st.warning("Open Home after preparing data to see the workspace overview.")
            st.stop()

        file_names = [path.name for path in parquet_files]
        default_file = next(
            (name for name in file_names if "features" in name.lower()),
            file_names[0],
        )
        if st.session_state.get("dataset_file") not in file_names:
            st.session_state["dataset_file"] = default_file

        st.markdown("### Workspace")
        selected_file = st.selectbox(
            "Prepared dataset",
            file_names,
            key="dataset_file",
            label_visibility="collapsed",
        )
        try:
            raw_df = load_parquet(str(PROCESSED_DIR / selected_file))
        except Exception as error:
            st.error(f"Could not read `{selected_file}`.")
            st.exception(error)
            st.stop()

        timestamp_column = detect_timestamp_column(raw_df)
        if timestamp_column is None:
            st.error(f"`{selected_file}` does not contain a recognizable timestamp.")
            st.stop()

        raw_df[timestamp_column] = pd.to_datetime(
            raw_df[timestamp_column], errors="coerce", utc=True
        )
        invalid_timestamps = int(raw_df[timestamp_column].isna().sum())
        raw_df = (
            raw_df.dropna(subset=[timestamp_column])
            .sort_values(timestamp_column)
            .reset_index(drop=True)
        )
        if raw_df.empty:
            st.error(f"`{selected_file}` contains no valid timestamps.")
            st.stop()
        if invalid_timestamps:
            st.warning(f"Skipped {invalid_timestamps:,} rows with invalid timestamps.")

        earliest = raw_df[timestamp_column].min().date()
        latest = raw_df[timestamp_column].max().date()
        bounds = (earliest, latest)
        if (
            st.session_state.get("_filter_dataset") != selected_file
            or st.session_state.get("_date_bounds") != bounds
        ):
            st.session_state["_filter_dataset"] = selected_file
            st.session_state["_date_bounds"] = bounds
            st.session_state["date_preset"] = "All data"
            st.session_state["date_filter"] = bounds

        st.markdown("### Observation window")
        st.radio(
            "Quick range",
            ["All data", "1M", "3M", "6M", "1Y", "YTD", "Custom"],
            horizontal=True,
            key="date_preset",
            on_change=_apply_date_preset,
            label_visibility="collapsed",
        )
        date_range = st.date_input(
            "From / to",
            min_value=earliest,
            max_value=latest,
            key="date_filter",
        )

        st.markdown("### Resolution")
        resolution = st.selectbox(
            "Aggregate bars",
            ["Raw", "5min", "15min", "1h", "4h", "1D"],
            key="data_resolution",
        )
        max_points = st.slider(
            "Chart detail",
            min_value=1000,
            max_value=10000,
            value=5000,
            step=1000,
            key="chart_detail",
            help="Limits the number of plotted points; table and metrics use all filtered rows.",
        )
        st.divider()
        st.caption("No live feeds · No trade execution")

    if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
        start_date, end_date = date_range
        filtered_df = raw_df[
            (raw_df[timestamp_column].dt.date >= start_date)
            & (raw_df[timestamp_column].dt.date <= end_date)
        ]
    else:
        filtered_df = raw_df

    if filtered_df.empty:
        st.warning("No observations match the selected date range.")
        st.stop()

    frame = resample_frame(filtered_df, timestamp_column, resolution)
    if frame.empty:
        st.warning("No observations remain at this time resolution.")
        st.stop()

    st.session_state.update(
        {
            "df": frame,
            "df_raw": filtered_df,
            "ts_col": timestamp_column,
            "resolution": resolution,
            "max_pts": max_points,
            "_root": ROOT,
            "_selected_file": selected_file,
        }
    )
    return frame, timestamp_column


def apply_app_style() -> None:
    light = st.session_state.get("light_mode", False)
    background = "#F2F5F7" if light else "#0D141C"
    panel = "#FFFFFF" if light else "#151F29"
    border = "#DCE4E9" if light else "#263441"
    text = "#202D37" if light else "#E6EDF1"
    muted = "#60717D" if light else "#9AABB5"
    hero_background = (
        "linear-gradient(120deg, #FFFFFF 8%, #E8F1F1 100%)"
        if light
        else "linear-gradient(120deg, #1B2A36 0%, #17221F 100%)"
    )

    st.markdown(
        f"""
        <style>
        .stApp {{ background: {background}; color: {text}; }}
        [data-testid="stSidebar"] {{
            background: {panel}; border-right: 1px solid {border};
        }}
        .block-container {{ padding-top: 2rem; padding-bottom: 3rem; }}
        .hero {{
            padding: 1.8rem 2rem; margin: 0.2rem 0 1.25rem;
            border: 1px solid {border}; border-radius: 22px;
            background: {hero_background};
        }}
        .hero-kicker {{
            color: {ACCENT}; font-size: .75rem; font-weight: 700;
            letter-spacing: .16em; text-transform: uppercase;
        }}
        .hero h1 {{ margin: .45rem 0 .4rem; font-size: 2.25rem; }}
        .hero p {{ color: {muted}; margin: 0; max-width: 760px; }}
        .eyebrow {{
            color: {muted}; font-size: .73rem; letter-spacing: .14em;
            text-transform: uppercase; font-weight: 700;
        }}
        div[data-testid="stMetric"] {{
            background: {panel}; border: 1px solid {border};
            padding: .8rem 1rem; border-radius: 15px;
        }}
        div[data-testid="stMetricLabel"] {{ color: {muted}; }}
        div[data-testid="stMetricValue"] {{ color: {text}; }}
        div[data-testid="stTabs"] button {{ font-weight: 600; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def chart_template() -> str:
    return "plotly_white" if st.session_state.get("light_mode", False) else "plotly_dark"
