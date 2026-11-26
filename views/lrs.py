"""Leveraged Rotation Strategy page (`/lrs`), routed from app.py."""

from __future__ import annotations

import math
import tempfile
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from fomo.metrics import drawdown, format_stats
from fomo.strategies import LEVERAGE, DATA_MODES, RISK_OFF_CHOICES, UNDERLYING, TrendConfig, load_market_data, run_backtest


PROXY_FILL = "rgba(128,128,128,0.18)"


def nice_log_ticks(lo: float, hi: float) -> list[float]:
    """1-2-5 tick values spanning [lo, hi] for a readable log axis."""
    lo, hi = max(lo, 1e-9), max(hi, 1e-9)
    ticks, k = [], math.floor(math.log10(lo))
    while 10**k <= hi * 10:
        ticks += [m * 10**k for m in (1, 2, 5) if lo / 1.5 <= m * 10**k <= hi * 1.5]
        k += 1
    return ticks


def fmt_money(v: float) -> str:
    for div, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "k")):
        if v >= div:
            return f"${v / div:g}{suffix}"
    return f"${v:g}"



def shade_proxies(fig, periods, annotate=True):
    """Gray band over spans where an ETF didn't exist and stand-in prices were used."""
    for i, p in enumerate(periods):
        fig.add_vrect(
            x0=p.start, x1=p.end, fillcolor=PROXY_FILL, line_width=0, layer="below",
            annotation_text=f"{p.asset}: {p.description}" if annotate else None,
            annotation_position="top left", annotation_font_size=11,
            annotation_yshift=-14 * i,
        )
    return fig


def add_trade_markers(fig, in_market, y, risk_on, risk_off, fmt="$%{y:,.0f}"):
    """Green ▲ where the strategy buys the ETF, red ▼ where it sells to the risk-off asset."""
    prev = in_market.shift(fill_value=False)
    for mask, name, symbol, color in (
        (in_market & ~prev, f"Buy {risk_on}", "triangle-up", "#2ea043"),
        (~in_market & prev, f"Sell → {risk_off}", "triangle-down", "#d1242f"),
    ):
        pts = y.reindex(in_market.index)[mask]
        fig.add_trace(go.Scatter(
            x=pts.index, y=pts, mode="markers", name=name,
            marker=dict(symbol=symbol, color=color, size=11, line=dict(color="white", width=1)),
            hovertemplate=f"{name}<br>{fmt}<extra></extra>",
        ))


def series_color(col: str, result) -> str:
    """Stable colors so adding/removing overlays doesn't reshuffle the palette."""
    if col == result.strategy_name:
        return "#1f4fd8"
    if col == result.underlying_benchmark:
        return "#7fb3ff"
    if col == result.leveraged_benchmark:
        return "#f4a6a6"
    return "#e45756"


@st.cache_data(ttl=6 * 3600, show_spinner="Loading prices...")
def cached_data(risk_on, risk_off, data_mode, expense_ratio):
    return load_market_data(risk_on, risk_off, data_mode, LEVERAGE[risk_on], expense_ratio)


@st.cache_data(show_spinner="Running backtest...")
def cached_backtest(cfg_dict: dict):
    cfg = TrendConfig(**cfg_dict)
    data = cached_data(cfg.risk_on, cfg.risk_off, cfg.data_mode, cfg.expense_ratio)
    return run_backtest(cfg, data)


@st.cache_data(show_spinner="Running parameter sweep...")
def cached_sweep(risk_on, risk_off, data_mode, expense_ratio, cost_bps, start, end, smas, buffers, base,
                 lower_buffers=None):
    from fomo.sweep import parameter_sweep

    data = cached_data(risk_on, risk_off, data_mode, expense_ratio)
    return parameter_sweep(
        data.underlying,
        data.prices[data.risk_on_col],
        data.prices[data.risk_off_col],
        sma_lengths=smas,
        buffers=buffers,
        lower_buffers=lower_buffers,
        cost_bps=cost_bps,
        base=base,
        start=start,
        end=end,
    )


# ---------------- Sidebar ----------------
sb = st.sidebar
sb.header("Strategy")
risk_on = sb.selectbox(
    "Risk-on ETF", list(UNDERLYING),
    format_func=lambda t: f"{t} ({LEVERAGE[t]:g}x {UNDERLYING[t]})",
    help="Signal uses the underlying: QQQ for TQQQ/QLD, SPY for UPRO/SSO",
)
risk_off = sb.selectbox("Risk-off asset", RISK_OFF_CHOICES, help="CASH = 13-week T-bills (^IRX)")
sma_length = sb.slider("SMA length (days)", 20, 300, 200, step=10)
sync_buffers = sb.checkbox(
    "Same upper & lower buffer", value=True,
    help="Uncheck to set the buy (upper) and sell (lower) bands separately.",
)
if sync_buffers:
    upper_pct = lower_pct = sb.slider(
        "Buffer band (%)", 0.0, 5.0, 0.0, step=0.25, key="buf_sym",
        help="Buy when close > SMA×(1+b); sell when close < SMA×(1−b); hold in between.",
    )
else:
    start_val = st.session_state.get("buf_sym", 0.0)
    upper_pct = sb.slider(
        "Upper buffer – buy (%)", 0.0, 5.0, start_val, step=0.25, key="buf_up",
        help="Buy when close > SMA×(1+upper).",
    )
    lower_pct = sb.slider(
        "Lower buffer – sell (%)", 0.0, 5.0, start_val, step=0.25, key="buf_lo",
        help="Sell when close < SMA×(1−lower).",
    )
sb.header("Data & costs")
data_mode = sb.radio(
    "Leveraged ETF prices", DATA_MODES,
    format_func={"spliced": "Synthetic before inception + real", "synthetic": "Synthetic only",
                 "real": "Real ETF only (2009/2010+)"}.get,
)
initial_capital = sb.number_input("Initial amount ($)", 1_000.0, 100_000_000.0, 100_000.0, step=10_000.0, format="%.0f")
cost_bps = sb.number_input("Transaction cost (bps per trade)", 0.0, 100.0, 5.0, step=1.0)
expense_pct = sb.number_input("Synthetic expense ratio (%/yr)", 0.0, 3.0, 0.95, step=0.05)

try:
    data = cached_data(risk_on, risk_off, data_mode, expense_pct / 100)
except Exception as exc:  # network / yfinance failures
    st.error(f"Failed to load prices: {exc}")
    st.stop()

min_d, max_d = data.prices.index[0].date(), data.prices.index[-1].date()
PERIODS = {"YTD": "ytd", "Last 5 years": 5, "Last 10 years": 10, "Last 20 years": 20, "Full history": None,
           "Custom": "custom"}
period = sb.selectbox(
    "Period", list(PERIODS),
    index=list(PERIODS).index(st.query_params.get("period", "Last 10 years")
                              if st.query_params.get("period") in PERIODS else "Last 10 years"),
    help="Earlier dates use synthetic prices where an ETF didn't exist yet (shown as a gray band).",
)
if PERIODS[period] == "ytd":
    # Start from the prior year's last close so the first trading day's return counts.
    before = data.prices.index[data.prices.index < pd.Timestamp(max_d.year, 1, 1)]
    start, end = (before[-1].date() if len(before) else min_d), max_d
elif PERIODS[period] == "custom":
    default_start = max(min_d, (pd.Timestamp(max_d) - pd.DateOffset(years=10)).date())
    dates = sb.date_input("Date range", (default_start, max_d), min_value=min_d, max_value=max_d)
    start, end = (dates if isinstance(dates, (tuple, list)) and len(dates) == 2 else (default_start, max_d))
elif PERIODS[period] is None:
    start, end = min_d, max_d
else:
    start, end = max(min_d, (pd.Timestamp(max_d) - pd.DateOffset(years=PERIODS[period])).date()), max_d

cfg = TrendConfig(
    risk_on=risk_on, risk_off=risk_off, sma_length=sma_length,
    buffer=upper_pct / 100, lower_buffer=lower_pct / 100,
    cost_bps=cost_bps, data_mode=data_mode, expense_ratio=expense_pct / 100,
    initial_capital=initial_capital,
    start=str(start), end=str(end),
)
result = cached_backtest(asdict(cfg))

# ---------------- Main ----------------
st.title("Leveraged ETF trend filter")
st.caption(
    f"Hold **{risk_on}** while {cfg.underlying} is above its {sma_length}-day SMA"
    + (f" (buy above SMA+{cfg.buffer:.2%}, sell below SMA−{cfg.sell_buffer:.2%})"
       if cfg.buffer or cfg.sell_buffer else "")
    + f", else **{risk_off}**. Signal uses the prior close. Not investment advice."
)

tab_bt, tab_sweep = st.tabs(["Backtest", "Optimize (parameter sweep)"])

with tab_bt:
    for p in result.proxy_periods:
        st.warning(
            f"**{p.asset} did not exist from {p.start.date()} to {p.end.date()}** — "
            f"{p.description} prices are used there (gray band on the charts). "
            "Synthetic data is an estimate; pick a later start date for real-ETF-only results.",
            icon="⚠️",
        )
    s = result.stats[result.strategy_name]
    lev = result.stats[result.leveraged_benchmark]
    c = st.columns(6)
    c[0].metric("CAGR", f"{s['CAGR']:.1%}", f"{s['CAGR'] - lev['CAGR']:+.1%} vs B&H {risk_on}")
    c[1].metric("Max drawdown", f"{s['Max drawdown']:.1%}", f"{s['Max drawdown'] - lev['Max drawdown']:+.1%}")
    c[2].metric("Sharpe (rf=0)", f"{s['Sharpe (rf=0)']:.2f}")
    c[3].metric("Switches", f"{result.switches}")
    c[4].metric("Time in market", f"{s['Time in market']:.0%}")
    c[5].metric("Final value", f"${s['Final value']:,.0f}", f"from ${initial_capital:,.0f}", delta_color="off")

    t1, t2 = st.columns(2)
    log_y = t1.toggle(
        "Log scale", value=True,
        help="Log: equal heights = equal % change (best for long periods). Linear: equal heights = equal $.",
    )
    show_sma = t2.toggle(
        f"Show {sma_length}-day SMA", value=True,
        help=f"{cfg.underlying}'s SMA scaled onto the B&H {cfg.underlying} line. The strategy holds "
             f"{risk_on} (green shading) while that line is above the SMA.",
    )
    eq = result.equity
    fig = go.Figure()
    if show_sma:
        fig.add_trace(go.Scatter(
            x=result.in_market.index, y=result.in_market.astype(int), name=f"Holding {risk_on}",
            yaxis="y2", fill="tozeroy", line=dict(width=0, shape="hv"),
            fillcolor="rgba(46,160,67,0.15)", hoverinfo="skip",
        ))
    for col in eq.columns:
        fig.add_trace(go.Scatter(x=eq.index, y=eq[col], name=col, line=dict(
            color=series_color(col, result), width=2.5 if col == result.strategy_name else 1.2)))
    if show_sma:
        # Same scale factor that maps the underlying's price onto its B&H equity curve.
        bh = eq[result.underlying_benchmark]
        scale = (bh / result.underlying.reindex(bh.index)).iloc[0]
        sma_eq = result.sma.reindex(bh.index) * scale
        fig.add_trace(go.Scatter(x=sma_eq.index, y=sma_eq, name=f"SMA {sma_length} ({cfg.underlying})",
                                 line=dict(color="black", dash="dash", width=1.3)))
        fig.update_layout(yaxis2=dict(overlaying="y", range=[0, 1], visible=False))
    fig.update_layout(
        title=f"Growth of ${initial_capital:,.0f} ({'log' if log_y else 'linear'} scale)", height=480,
        legend=dict(orientation="h", y=-0.15), hovermode="x unified",
    )
    if log_y:
        ticks = nice_log_ticks(float(eq.min().min()), float(eq.max().max()))
        fig.update_layout(yaxis=dict(type="log", tickvals=ticks, ticktext=[fmt_money(t) for t in ticks]))
    else:
        fig.update_layout(yaxis=dict(tickprefix="$", tickformat=",.0f"))
    fig.for_each_trace(lambda t: t.update(hovertemplate="$%{y:,.0f}") if t.yaxis != "y2" else None)
    add_trade_markers(fig, result.in_market, eq[result.strategy_name], risk_on, risk_off)
    shade_proxies(fig, result.proxy_periods)
    st.plotly_chart(fig, width="stretch")

    dd = drawdown(result.equity)
    fig_dd = go.Figure()
    for col in dd.columns:
        fig_dd.add_trace(go.Scatter(x=dd.index, y=dd[col], name=col, line=dict(color=series_color(col, result)),
                                    fill="tozeroy" if col == result.strategy_name else None))
    fig_dd.update_layout(title="Drawdown", yaxis_tickformat=".0%", height=320,
                         legend=dict(orientation="h", y=-0.2), hovermode="x unified")
    shade_proxies(fig_dd, result.proxy_periods, annotate=False)
    st.plotly_chart(fig_dd, width="stretch")

    fig_sig = go.Figure()
    fig_sig.add_trace(go.Scatter(
        x=result.in_market.index, y=result.in_market.astype(int), name=f"In {risk_on}",
        yaxis="y2", fill="tozeroy", line=dict(width=0, shape="hv"),
        fillcolor="rgba(46,160,67,0.15)", hoverinfo="skip",
    ))
    fig_sig.add_trace(go.Scatter(x=result.underlying.index, y=result.underlying, name=cfg.underlying))
    fig_sig.add_trace(go.Scatter(x=result.sma.index, y=result.sma, name=f"SMA {sma_length}"))
    add_trade_markers(fig_sig, result.in_market, result.underlying, risk_on, risk_off, fmt="%{y:,.2f}")
    fig_sig.update_layout(
        title=f"{cfg.underlying} vs SMA (green shading = holding {risk_on}; ▲ buy, ▼ sell)", yaxis_type="log" if log_y else "linear",
        height=420,
        yaxis2=dict(overlaying="y", range=[0, 1], visible=False), hovermode="x unified",
        legend=dict(orientation="h", y=-0.15),
    )
    shade_proxies(fig_sig, result.proxy_periods, annotate=False)
    if log_y:
        ticks = nice_log_ticks(float(result.underlying.min()), float(result.underlying.max()))
        fig_sig.update_layout(yaxis=dict(tickvals=ticks, ticktext=[f"{t:g}" for t in ticks]))
    st.plotly_chart(fig_sig, width="stretch")

    st.subheader("Statistics")
    st.dataframe(format_stats(result.stats), width="stretch")

    if result.validation:
        v = result.validation
        with st.expander("Synthetic vs real ETF validation"):
            st.write(
                f"Overlap {v['start']} → {v['end']}: daily return correlation "
                f"**{v['daily_return_corr']:.4f}**, tracking error **{v['tracking_error_ann']:.2%}/yr**, "
                f"CAGR synthetic {v['cagr_synthetic']:.2%} vs real {v['cagr_real']:.2%} "
                f"(**{v['cagr_difference']:+.2%}/yr**)."
            )

    if st.button("Generate quantstats tearsheet"):
        from fomo.report import write_quantstats_report

        with st.spinner("Building report..."), tempfile.TemporaryDirectory() as tmp:
            path = write_quantstats_report(result, Path(tmp) / "tearsheet.html")
            html = path.read_bytes()
        st.download_button("Download tearsheet.html", html, file_name="tearsheet.html", mime="text/html")

with tab_sweep:
    st.write(
        "Grid search over SMA length and buffer band on the selected date range, data and costs. "
        "**The best in-sample cell is almost certainly overfit** — look for broad regions that work, "
        "and check them on a different period (e.g. optimize pre-2015, test after)."
    )
    if result.proxy_periods:
        st.caption("⚠️ Includes periods with stand-in prices: " + "; ".join(
            f"{p.asset} = {p.description} until {p.end.date()}" for p in result.proxy_periods))
    modes = ["SMA length × buffer (symmetric)", f"Upper × lower buffer (at SMA {sma_length})"]
    sweep_mode = st.radio("Sweep", modes, index=0 if sync_buffers else 1, horizontal=True)
    asym = sweep_mode == modes[1]
    c1, c2, c3 = st.columns(3)
    if asym:
        smas = (sma_length,)
        up_max = c1.slider("Max upper buffer (%)", 0.5, 10.0, 3.0, step=0.5)
        lo_max = c2.slider("Max lower buffer (%)", 0.5, 10.0, 3.0, step=0.5)
        step_pct = c3.select_slider("Buffer step (%)", [0.25, 0.5, 1.0], value=0.5)
    else:
        sma_lo, sma_hi = c1.slider("SMA range", 20, 300, (100, 250), step=10)
        sma_step = c2.number_input("SMA step", 5, 100, 25, step=5)
        up_max = lo_max = c3.slider("Max buffer (%)", 0.5, 10.0, 5.0, step=0.5)
        step_pct = 0.25
        smas = tuple(range(sma_lo, sma_hi + 1, int(sma_step)))

    def _grid(mx):
        return tuple(round(i * step_pct / 100, 4) for i in range(int(round(mx / step_pct)) + 1))

    buffers = _grid(up_max)
    lower_buffers = _grid(lo_max) if asym else None
    metric = st.selectbox("Heatmap metric", ["CAGR", "Final value", "Max drawdown", "Sharpe (rf=0)", "Calmar", "Switches"])

    if st.button("Run sweep", type="primary"):
        st.session_state["sweep_args"] = (risk_on, risk_off, data_mode, expense_pct / 100, cost_bps,
                                          str(start), str(end), smas, buffers, initial_capital, lower_buffers)
    if "sweep_args" in st.session_state:
        args = st.session_state["sweep_args"]
        sweep = cached_sweep(*args)
        ran_asym = args[-1] is not None
        if ran_asym:
            grid = sweep.pivot(index="lower_buffer", columns="upper_buffer", values=metric)
            y_labels, y_title = [f"{i:.2%}" for i in grid.index], "Lower (sell) buffer"
            x_title, title = "Upper (buy) buffer", f"{metric}: upper × lower buffer at SMA {args[7][0]}"
        else:
            grid = sweep.pivot(index="sma_length", columns="upper_buffer", values=metric)
            y_labels, y_title = [str(i) for i in grid.index], "SMA length"
            x_title, title = "Buffer (upper = lower)", f"{metric} by SMA length × buffer"
        pct = metric in ("CAGR", "Max drawdown")
        fig_hm = px.imshow(
            grid.values, x=[f"{b:.2%}" for b in grid.columns], y=y_labels,
            color_continuous_scale="RdYlGn_r" if metric == "Switches" else "RdYlGn",
            aspect="auto", text_auto=".1%" if pct else {"Switches": ".0f", "Final value": "$,.0f"}.get(metric, ".2f"),
            labels=dict(x=x_title, y=y_title, color=metric),
        )
        fig_hm.update_layout(title=title, height=520)
        st.plotly_chart(fig_hm, width="stretch")
        show = sweep.copy()
        for col in ("upper_buffer", "lower_buffer", "CAGR", "Max drawdown", "Time in market"):
            show[col] = show[col].map("{:.2%}".format)
        show["Final value"] = show["Final value"].map("${:,.0f}".format)
        st.dataframe(show, width="stretch", hide_index=True)
        st.download_button("Download sweep CSV", sweep.to_csv(index=False), file_name="sweep.csv")

st.caption(
    "Caveats: leveraged ETFs suffer volatility decay; synthetic history ignores swap spreads and "
    "fund-specific frictions; taxes and slippage are not modeled. For research only, not investment advice."
)
