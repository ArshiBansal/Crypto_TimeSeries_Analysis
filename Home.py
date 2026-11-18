"""BTC / FIELDNOTES — an interactive historical-market observatory."""

import numpy as np
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


st.set_page_config(
    page_title="BTC / FIELDNOTES",
    page_icon="◒",
    layout="wide",
    initial_sidebar_state="expanded",
)

frame, timestamp_column = load_dashboard_data()
apply_app_style()

if frame is None:
    st.stop()

numeric_columns = frame.select_dtypes(include="number").columns.tolist()
price_candidates = [
    column for column in ("close", "price", "adj_close") if column in frame.columns
]
price_column = price_candidates[0] if price_candidates else None
selected_file = st.session_state.get("_selected_file", "prepared dataset")
resolution = st.session_state.get("resolution", "Raw")
max_points = st.session_state.get("max_pts", 5000)
muted_color = "#9AABB5"
grid_color = "#293743"

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


metrics = st.columns(5)
metrics[0].metric(
    "Last close",
    format_price(latest_price),
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
    f"{total_volume:,.0f}" if pd.notna(total_volume) else "—",
)

st.write("")

tab_main, tab_dist, tab_calendar = st.tabs(["📈 THE TAPE", "📊 DISTRIBUTIONS", "📅 CALENDAR"])

with tab_main:
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
                increasing_line_color=BLUE,
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
                line=dict(color=BLUE, width=2.2),
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
            BLUE if pd.notna(open_) and pd.notna(close) and close >= open_ else CORAL
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

    col_left, col_right = st.columns([1.1, 0.9], gap="large")
    with col_left:
        st.markdown("#### Return Distribution")
        return_values = valid_returns.to_numpy(dtype=float) * 100
        return_values = return_values[np.isfinite(return_values)]
        
        ret_fig = make_subplots(rows=2, cols=2, subplot_titles=("Histogram", "Box Plot", "ECDF", "Violin"))
        
        ret_fig.add_trace(go.Histogram(x=return_values, nbinsx=65, name="Bar returns",
                                        marker_color=BLUE, opacity=0.8), row=1, col=1)
        ret_fig.add_vline(x=0, line_dash="dot", line_color=muted_color, row=1, col=1)
        ret_fig.add_vline(x=return_values.mean(), line_dash="dash", line_color=ACCENT, row=1, col=1)
        
        ret_fig.add_trace(go.Box(y=return_values, name="Returns", marker_color=BLUE,
                                  boxpoints="outliers"), row=1, col=2)
        
        ecdf_x = np.sort(return_values)
        ecdf_y = np.arange(1, len(ecdf_x) + 1) / len(ecdf_x)
        ret_fig.add_trace(go.Scatter(x=ecdf_x, y=ecdf_y, mode="lines", name="ECDF",
                                      line=dict(color=PURPLE, width=2)), row=2, col=1)
        
        ret_fig.add_trace(go.Violin(y=return_values, name="Returns", side="positive",
                                     width=0.8, line_color=ACCENT, fillcolor=ACCENT, opacity=0.4,
                                     box_visible=True, meanline_visible=True), row=2, col=2)
        
        ret_fig.update_layout(template=chart_template(), height=450,
                             margin=dict(l=12, r=12, t=40, b=12), showlegend=False)
        st.plotly_chart(ret_fig, use_container_width=True, config={"displaylogo": False})
        
        st.markdown("#### Return Statistics")
        rcols = st.columns(5)
        rcols[0].metric("Mean", f"{return_values.mean():.3f}%")
        rcols[1].metric("Std", f"{return_values.std():.3f}%")
        rcols[2].metric("Skew", f"{pd.Series(return_values).skew():.3f}")
        rcols[3].metric("Kurtosis", f"{pd.Series(return_values).kurtosis():.3f}")
        rcols[4].metric("Sharpe (est.)", f"{return_values.mean() / return_values.std() * np.sqrt(365*24):.2f}")

    with col_right:
        st.markdown("#### Risk in the Window")
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

        risk_fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                                 row_heights=[0.7, 0.3],
                                 subplot_titles=(volatility_label, "Volatility Distribution"))
        
        risk_frame = sample_for_chart(frame, max_points)
        risk_fig.add_trace(go.Scatter(
            x=risk_frame[timestamp_column],
            y=vol_values.loc[risk_frame.index],
            name=volatility_label,
            mode="lines",
            line=dict(color=PURPLE, width=2),
            fill="tozeroy",
            fillcolor="rgba(166,154,200,0.12)",
        ), row=1, col=1)
        
        vol_clean = vol_values.dropna()
        if len(vol_clean) > 0:
            risk_fig.add_trace(go.Histogram(x=vol_clean, nbinsx=50, name="Distribution",
                                            marker_color=PURPLE, opacity=0.7), row=2, col=1)
            risk_fig.add_vline(x=vol_clean.mean(), line_dash="dash", line_color=ACCENT, row=2, col=1)
        
        risk_fig.update_layout(template=chart_template(), height=500,
                              margin=dict(l=12, r=12, t=40, b=12),
                              hovermode="x unified", showlegend=False)
        risk_fig.update_xaxes(showgrid=False)
        risk_fig.update_yaxes(gridcolor=grid_color)
        st.plotly_chart(risk_fig, use_container_width=True, config={"displaylogo": False})

with tab_dist:
    st.markdown("#### Return vs Volatility Scatter")
    ret_vol_frame = frame[[timestamp_column, price_column]].copy()
    ret_vol_frame[timestamp_column] = pd.to_datetime(ret_vol_frame[timestamp_column], errors="coerce", utc=True)
    ret_vol_frame[price_column] = pd.to_numeric(ret_vol_frame[price_column], errors="coerce")
    ret_vol_frame = ret_vol_frame.dropna().sort_values(timestamp_column)
    ret_vol_frame["return"] = ret_vol_frame[price_column].pct_change() * 100
    ret_vol_frame["abs_return"] = ret_vol_frame["return"].abs()
    ret_vol_frame["rolling_vol"] = ret_vol_frame["return"].rolling(30, min_periods=5).std() * 100
    
    vol_cols = [c for c in frame.columns if "rolling_vol" in c.lower() and "ann" not in c.lower()]
    if vol_cols:
        vol_col = vol_cols[0]
        vol_data = frame[[timestamp_column, vol_col]].copy()
        vol_data[timestamp_column] = pd.to_datetime(vol_data[timestamp_column], errors="coerce", utc=True)
        vol_data[vol_col] = pd.to_numeric(vol_data[vol_col], errors="coerce")
        vol_data = vol_data.dropna().sort_values(timestamp_column)
        merged = pd.merge_asof(ret_vol_frame, vol_data, on=timestamp_column, direction="nearest", tolerance=pd.Timedelta("1D"))
        merged = merged.dropna()
        
        if len(merged) > 10:
            sc_fig = go.Figure()
            sc_fig.add_trace(go.Scattergl(x=merged["return"], y=merged[vol_col],
                                           mode="markers", name="Bars",
                                           marker=dict(color=BLUE, size=4, opacity=0.4)))
            sc_fig.add_hline(y=merged[vol_col].mean(), line_dash="dash", line_color="#9AA7B2")
            sc_fig.add_vline(x=0, line_dash="dash", line_color="#9AA7B2")
            sc_fig.update_layout(template=chart_template(), height=400,
                                margin=dict(l=12, r=12, t=25, b=12),
                                xaxis_title="Return (%)", yaxis_title=vol_col,
                                hovermode="x unified")
            st.plotly_chart(sc_fig, use_container_width=True, config={"displaylogo": False})
    
    st.markdown("#### Rolling Statistics")
    if len(ret_vol_frame) > 30:
        roll_fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                                 row_heights=[0.4, 0.3, 0.3],
                                 subplot_titles=("Rolling Mean Return", "Rolling Volatility", "Rolling Sharpe"))
        
        roll_fig.add_trace(go.Scatter(x=ret_vol_frame[timestamp_column], 
                                      y=ret_vol_frame["return"].rolling(30, min_periods=5).mean(),
                                      mode="lines", name="30-bar mean", line=dict(color=ACCENT, width=1.5)), row=1, col=1)
        roll_fig.add_trace(go.Scatter(x=ret_vol_frame[timestamp_column], 
                                      y=ret_vol_frame["return"].rolling(90, min_periods=10).mean(),
                                      mode="lines", name="90-bar mean", line=dict(color=BLUE, width=1.5, dash="dot")), row=1, col=1)
        roll_fig.add_hline(y=0, line_dash="dash", line_color="#9AA7B2", row=1, col=1)
        
        roll_fig.add_trace(go.Scatter(x=ret_vol_frame[timestamp_column], 
                                      y=ret_vol_frame["rolling_vol"],
                                      mode="lines", name="30-bar vol", line=dict(color=PURPLE, width=1.5)), row=2, col=1)
        roll_fig.add_trace(go.Scatter(x=ret_vol_frame[timestamp_column], 
                                      y=ret_vol_frame["return"].rolling(90, min_periods=10).std() * 100,
                                      mode="lines", name="90-bar vol", line=dict(color=BLUE, width=1.5, dash="dot")), row=2, col=1)
        
        roll_sharpe = ret_vol_frame["return"].rolling(30, min_periods=5).mean() / (ret_vol_frame["return"].rolling(30, min_periods=5).std() + 1e-10)
        roll_fig.add_trace(go.Scatter(x=ret_vol_frame[timestamp_column], y=roll_sharpe,
                                      mode="lines", name="Rolling Sharpe", line=dict(color=CORAL, width=1.5)), row=3, col=1)
        roll_fig.add_hline(y=0, line_dash="dash", line_color="#9AA7B2", row=3, col=1)
        
        roll_fig.update_layout(template=chart_template(), height=650,
                              margin=dict(l=12, r=12, t=40, b=12), hovermode="x unified")
        roll_fig.update_xaxes(showgrid=False)
        roll_fig.update_yaxes(gridcolor=grid_color)
        st.plotly_chart(roll_fig, use_container_width=True, config={"displaylogo": False})

with tab_calendar:
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
                text=[[f"{v:.1f}%" if pd.notna(v) else "" for v in row] for row in calendar.values],
                texttemplate="%{text}",
                textfont=dict(size=9, color="white")
            )
        )
        calendar_chart.update_layout(
            template=chart_template(),
            height=460,
            margin=dict(l=12, r=12, t=20, b=12),
            xaxis_title="Year",
            yaxis_title="",
        )
        st.plotly_chart(
            calendar_chart,
            use_container_width=True,
            config={"displaylogo": False},
        )
        
        st.markdown("#### Monthly Statistics")
        mcols = st.columns(5)
        mcols[0].metric("Avg Monthly Return", f"{monthly_returns.mean():.2f}%")
        mcols[1].metric("Monthly Vol", f"{monthly_returns.std():.2f}%")
        mcols[2].metric("Best Month", f"{monthly_returns.max():.2f}%")
        mcols[3].metric("Worst Month", f"{monthly_returns.min():.2f}%")
        
        pos_months = (monthly_returns > 0).sum()
        mcols[4].metric("Positive Months", f"{pos_months}/{len(monthly_returns)} ({pos_months/len(monthly_returns)*100:.0f}%)")
        
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