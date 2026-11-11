"""BTC / FIELDNOTES — an interactive historical-market observatory."""

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
    apply_app_style,
    chart_template,
    load_dashboard_data,
    sample_for_chart,
)


st.set_page_config(
    page_title="BTC / FIELDNOTES",
    page_icon="◒",
    layout="wide",
    initial_sidebar_state="expanded",
)

frame, timestamp_column = load_dashboard_data()
apply_app_style()

numeric_columns = frame.select_dtypes(include="number").columns.tolist()
price_candidates = [
    column for column in ("close", "price", "adj_close") if column in frame.columns
]
price_column = price_candidates[0] if price_candidates else None
selected_file = st.session_state.get("_selected_file", "prepared dataset")
resolution = st.session_state.get("resolution", "Raw")
max_points = st.session_state.get("max_pts", 5000)
light_canvas = st.session_state.get("light_mode", False)
muted_color = "#60717D" if light_canvas else "#9AABB5"
grid_color = "#E0E7EB" if light_canvas else "#293743"

start_label = frame[timestamp_column].min().strftime("%d %b %Y")
end_label = frame[timestamp_column].max().strftime("%d %b %Y")
st.markdown(
    f"""
    <div class="hero">
      <div class="hero-kicker">BTC / FIELDNOTES &nbsp;·&nbsp; MARKET OBSERVATORY</div>
      <h1>A closer look at the tape.</h1>
      <p>Historical BTC, viewed through price, participation and risk.
         Select a window in the sidebar; the workspace follows.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

header_left, header_right = st.columns([3, 1])
with header_left:
    st.markdown(
        f'<div class="eyebrow">{selected_file} &nbsp; / &nbsp; {start_label} — {end_label} '
        f'&nbsp; / &nbsp; {resolution}</div>',
        unsafe_allow_html=True,
    )
with header_right:
    st.markdown(
        f'<div style="text-align:right;color:{muted_color};font-size:.82rem;">'
        f'{len(frame):,} observations in view</div>',
        unsafe_allow_html=True,
    )

if price_column is None:
    st.warning(
        "No `close` or `price` column was found. Open **Data Lab** from the sidebar "
        "to explore the available features."
    )
    st.stop()

prices = pd.to_numeric(frame[price_column], errors="coerce")
valid_prices = prices.dropna()
returns = prices.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
valid_returns = returns.dropna()
first_price = float(valid_prices.iloc[0]) if not valid_prices.empty else float("nan")
latest_price = float(valid_prices.iloc[-1]) if not valid_prices.empty else float("nan")
previous_price = float(valid_prices.iloc[-2]) if len(valid_prices) > 1 else float("nan")
period_return = (
    (latest_price / first_price - 1) * 100
    if first_price and pd.notna(first_price)
    else float("nan")
)
daily_change = (
    (latest_price / previous_price - 1) * 100
    if previous_price and pd.notna(previous_price)
    else float("nan")
)
rolling_volatility = valid_returns.rolling(30, min_periods=5).std() * 100
latest_volatility = (
    float(rolling_volatility.iloc[-1])
    if not rolling_volatility.empty and pd.notna(rolling_volatility.iloc[-1])
    else float("nan")
)
max_drawdown = (
    float(((valid_prices / valid_prices.cummax()) - 1).min() * 100)
    if not valid_prices.empty
    else float("nan")
)
total_volume = (
    pd.to_numeric(frame["volume"], errors="coerce").sum(min_count=1)
    if "volume" in frame.columns
    else float("nan")
)


def format_price(value: float) -> str:
    if pd.isna(value):
        return "—"
    if abs(value) >= 1000:
        return f"${value:,.0f}"
    if abs(value) >= 1:
        return f"${value:,.2f}"
    return f"${value:.6f}"


metrics = st.columns(6)
metrics[0].metric(
    "Last close",
    format_price(latest_price),
    delta=f"{daily_change:+.2f}% vs prior bar" if pd.notna(daily_change) else None,
)
metrics[1].metric(
    "Window return",
    f"{period_return:+.2f}%" if pd.notna(period_return) else "—",
)
metrics[2].metric(
    "30-bar volatility",
    f"{latest_volatility:.3f}%" if pd.notna(latest_volatility) else "—",
    help="Standard deviation of close-to-close returns; not annualized.",
)
metrics[3].metric(
    "Worst drawdown",
    f"{max_drawdown:.2f}%" if pd.notna(max_drawdown) else "—",
)
metrics[4].metric(
    "Volume in view",
    f"{total_volume:,.4g}" if pd.notna(total_volume) else "—",
)
metrics[5].metric("Bars", f"{len(frame):,}")

st.write("")
overview_tab, rhythm_tab = st.tabs(["THE TAPE", "CALENDAR RHYTHM"])

with overview_tab:
    controls, note = st.columns([2, 3])
    with controls:
        display_mode = st.radio(
            "Price lens",
            ["Candles", "Close", "Indexed"],
            horizontal=True,
            help="Indexed rebases the selected window to 100.",
        )
    with note:
        if pd.notna(period_return):
            direction = "higher" if period_return >= 0 else "lower"
            st.markdown(
                f'<div style="text-align:right;padding-top:.45rem;color:{muted_color};">'
                f'Window closed <b>{direction}</b> by {abs(period_return):.2f}% · '
                f'illustrative history, not a forecast</div>',
                unsafe_allow_html=True,
            )

    chart_frame = sample_for_chart(frame, max_points)
    has_ohlc = all(
        column in chart_frame.columns for column in ("open", "high", "low", "close")
    )
    use_candles = display_mode == "Candles" and has_ohlc
    show_volume = "volume" in chart_frame.columns
    rows = 2 if show_volume else 1
    chart = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.055,
        row_heights=[0.76, 0.24] if show_volume else [1],
    )

    if use_candles:
        chart.add_trace(
            go.Candlestick(
                x=chart_frame[timestamp_column],
                open=chart_frame["open"],
                high=chart_frame["high"],
                low=chart_frame["low"],
                close=chart_frame["close"],
                name="BTC",
                increasing_line_color=ACCENT,
                decreasing_line_color=CORAL,
                whiskerwidth=0.4,
            ),
            row=1,
            col=1,
        )
    else:
        shown_prices = pd.to_numeric(chart_frame[price_column], errors="coerce")
        if display_mode == "Indexed" and shown_prices.notna().any():
            base = shown_prices.dropna().iloc[0]
            shown_prices = shown_prices.div(base).mul(100) if base else shown_prices
            price_axis = "Window start = 100"
        else:
            price_axis = (
                "Price (USD)"
                if price_column in ("close", "price", "adj_close")
                else price_column
            )
        chart.add_trace(
            go.Scatter(
                x=chart_frame[timestamp_column],
                y=shown_prices,
                name="Indexed price" if display_mode == "Indexed" else price_column,
                mode="lines",
                line=dict(color=ACCENT, width=2.2),
            ),
            row=1,
            col=1,
        )
        chart.update_yaxes(title_text=price_axis, row=1, col=1)

    if show_volume:
        open_values = pd.to_numeric(
            chart_frame["open"] if "open" in chart_frame.columns else chart_frame[price_column],
            errors="coerce",
        )
        close_values = pd.to_numeric(chart_frame[price_column], errors="coerce")
        volume_color = [
            ACCENT if pd.notna(open_) and pd.notna(close) and close >= open_ else CORAL
            for open_, close in zip(open_values, close_values)
        ]
        chart.add_trace(
            go.Bar(
                x=chart_frame[timestamp_column],
                y=pd.to_numeric(chart_frame["volume"], errors="coerce"),
                name="Volume",
                marker_color=volume_color,
                opacity=0.7,
            ),
            row=2,
            col=1,
        )
        chart.update_yaxes(title_text="Volume", row=2, col=1)

    chart.update_layout(
        template=chart_template(),
        height=510 if show_volume else 440,
        margin=dict(l=12, r=12, t=18, b=8),
        hovermode="x unified",
        showlegend=False,
        xaxis_rangeslider_visible=False,
        dragmode="pan",
    )
    chart.update_xaxes(showgrid=False, rangeslider_visible=False)
    chart.update_yaxes(gridcolor=grid_color, zerolinecolor=grid_color)
    st.plotly_chart(
        chart,
        use_container_width=True,
        config={"displaylogo": False, "scrollZoom": True},
    )

    lower_left, lower_right = st.columns([1.1, 0.9], gap="large")
    with lower_left:
        st.markdown("#### Return texture")
        return_values = valid_returns.to_numpy(dtype=float) * 100
        return_values = return_values[np.isfinite(return_values)]
        return_counts, return_edges = np.histogram(return_values, bins=65)
        return_centers = (return_edges[:-1] + return_edges[1:]) / 2
        return_chart = go.Figure()
        return_chart.add_trace(
            go.Bar(
                x=return_centers,
                y=return_counts,
                name="Bar returns",
                marker_color=BLUE,
                opacity=0.82,
            )
        )
        return_chart.add_vline(x=0, line_color=muted_color, line_dash="dot")
        return_chart.update_layout(
            template=chart_template(),
            height=310,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Close-to-close return (%)",
            yaxis_title="Bars",
            showlegend=False,
        )
        return_chart.update_xaxes(gridcolor=grid_color)
        return_chart.update_yaxes(gridcolor=grid_color)
        st.plotly_chart(return_chart, use_container_width=True, config={"displaylogo": False})

    with lower_right:
        st.markdown("#### Risk in the window")
        vol_column = next(
            (
                column
                for column in frame.columns
                if "rolling_vol" in column.lower() and "ann" not in column.lower()
            ),
            None,
        )
        if vol_column:
            vol_values = pd.to_numeric(frame[vol_column], errors="coerce")
            volatility_label = vol_column
        else:
            vol_values = returns.rolling(30, min_periods=5).std()
            volatility_label = "30-bar realized volatility"

        risk_chart = go.Figure()
        risk_frame = sample_for_chart(frame, max_points)
        risk_chart.add_trace(
            go.Scatter(
                x=risk_frame[timestamp_column],
                y=vol_values.loc[risk_frame.index],
                name=volatility_label,
                mode="lines",
                line=dict(color=PURPLE, width=2),
                fill="tozeroy",
                fillcolor="rgba(166,154,200,0.12)",
            )
        )
        risk_chart.update_layout(
            template=chart_template(),
            height=310,
            margin=dict(l=10, r=10, t=10, b=10),
            yaxis_title=volatility_label,
            showlegend=False,
            hovermode="x unified",
        )
        risk_chart.update_xaxes(showgrid=False)
        risk_chart.update_yaxes(gridcolor=grid_color)
        st.plotly_chart(risk_chart, use_container_width=True, config={"displaylogo": False})

with rhythm_tab:
    st.markdown(
        "### Monthly close-to-close returns",
        help="Each cell is the return from the prior month-end close to this month-end close.",
    )
    monthly_close = (
        pd.DataFrame(
            {
                "date": frame[timestamp_column],
                "close": pd.to_numeric(frame[price_column], errors="coerce"),
            }
        )
        .dropna()
        .assign(
            month=lambda data: data["date"].dt.tz_localize(None).dt.to_period("M")
        )
        .groupby("month")["close"]
        .last()
        .dropna()
    )
    monthly_returns = monthly_close.pct_change().mul(100).dropna()
    if monthly_returns.empty:
        st.info("A longer date window is needed to build a monthly return calendar.")
    else:
        monthly_table = monthly_returns.to_frame("return")
        monthly_table["year"] = monthly_table.index.year
        monthly_table["month"] = monthly_table.index.month
        calendar = monthly_table.pivot(
            index="month", columns="year", values="return"
        ).reindex(range(1, 13))

        calendar_chart = go.Figure(
            go.Heatmap(
                z=calendar.values,
                x=[str(year) for year in calendar.columns],
                y=["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
                colorscale=[
                    [0, "#B66F73"],
                    [0.5, "#283541"],
                    [1, "#63A99E"],
                ],
                zmid=0,
                colorbar=dict(title="Return %"),
                hovertemplate="%{y} %{x}<br>Return %{z:.2f}%<extra></extra>",
            )
        )
        calendar_chart.update_layout(
            template=chart_template(),
            height=430,
            margin=dict(l=12, r=12, t=20, b=12),
            xaxis_title="Year",
            yaxis_title="",
        )
        st.plotly_chart(
            calendar_chart,
            use_container_width=True,
            config={"displaylogo": False},
        )
        st.caption(
            "Calendar cells summarize historical monthly movement. "
            "They are descriptive, not predictive."
        )

st.divider()
left_footer, right_footer = st.columns([2, 1])
with left_footer:
    st.caption(
        f"Source: {selected_file} · {len(numeric_columns)} numeric features · "
        "Historical observations only"
    )
with right_footer:
    st.markdown(
        f'<div style="text-align:right;color:{muted_color};font-size:.78rem;">'
        "Explore deeper: Data Lab · Forecasts · Risk · Regimes · Anomalies · Breaks"
        "</div>",
        unsafe_allow_html=True,
    )
