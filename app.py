"""Streamlit dashboard: `streamlit run app.py`."""

from __future__ import annotations

import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from fomo.metrics import drawdown, format_stats
from fomo.strategies import DATA_MODES, RISK_OFF_CHOICES, UNDERLYING, TrendConfig, load_market_data, run_backtest

st.set_page_config(page_title="Leveraged trend backtest", layout="wide")


@st.cache_data(ttl=6 * 3600, show_spinner="Loading prices...")
def cached_data(risk_on, risk_off, data_mode, expense_ratio):
    return load_market_data(risk_on, risk_off, data_mode, 3.0, expense_ratio)


@st.cache_data(show_spinner="Running backtest...")
def cached_backtest(cfg_dict: dict):
    cfg = TrendConfig(**cfg_dict)
    data = cached_data(cfg.risk_on, cfg.risk_off, cfg.data_mode, cfg.expense_ratio)
    return run_backtest(cfg, data)


@st.cache_data(show_spinner="Running parameter sweep...")
def cached_sweep(risk_on, risk_off, data_mode, expense_ratio, cost_bps, start, end, smas, buffers):
    from fomo.sweep import parameter_sweep

    data = cached_data(risk_on, risk_off, data_mode, expense_ratio)
    return parameter_sweep(
        data.underlying,
        data.prices[data.risk_on_col],
        data.prices[data.risk_off_col],
        sma_lengths=smas,
        buffers=buffers,
        cost_bps=cost_bps,
        start=start,
        end=end,
    )


# ---------------- Sidebar ----------------
sb = st.sidebar
sb.header("Strategy")
risk_on = sb.selectbox("Risk-on ETF", list(UNDERLYING), help="Signal uses QQQ for TQQQ, SPY for UPRO")
risk_off = sb.selectbox("Risk-off asset", RISK_OFF_CHOICES, help="CASH = 13-week T-bills (^IRX)")
sma_length = sb.slider("SMA length (days)", 20, 300, 200, step=10)
buffer_pct = sb.slider(
    "Buffer band (%)", 0.0, 5.0, 0.0, step=0.25,
    help="Buy when close > SMA×(1+b); sell when close < SMA×(1−b); hold in between.",
)
sb.header("Data & costs")
data_mode = sb.radio(
    "Leveraged ETF prices", DATA_MODES,
    format_func={"spliced": "Synthetic before inception + real", "synthetic": "Synthetic only",
                 "real": "Real ETF only (2009/2010+)"}.get,
)
cost_bps = sb.number_input("Transaction cost (bps per trade)", 0.0, 100.0, 5.0, step=1.0)
expense_pct = sb.number_input("Synthetic expense ratio (%/yr)", 0.0, 3.0, 0.95, step=0.05)

try:
    data = cached_data(risk_on, risk_off, data_mode, expense_pct / 100)
except Exception as exc:  # network / yfinance failures
    st.error(f"Failed to load prices: {exc}")
    st.stop()

min_d, max_d = data.prices.index[0].date(), data.prices.index[-1].date()
dates = sb.date_input("Date range", (min_d, max_d), min_value=min_d, max_value=max_d)
start, end = (dates if isinstance(dates, (tuple, list)) and len(dates) == 2 else (min_d, max_d))

cfg = TrendConfig(
    risk_on=risk_on, risk_off=risk_off, sma_length=sma_length, buffer=buffer_pct / 100,
    cost_bps=cost_bps, data_mode=data_mode, expense_ratio=expense_pct / 100,
    start=str(start), end=str(end),
)
result = cached_backtest(asdict(cfg))

# ---------------- Main ----------------
st.title("Leveraged ETF trend filter")
st.caption(
    f"Hold **{risk_on}** while {cfg.underlying} is above its {sma_length}-day SMA"
    + (f" (±{cfg.buffer:.2%} band)" if cfg.buffer else "")
    + f", else **{risk_off}**. Signal uses the prior close. Not investment advice."
)

tab_bt, tab_sweep = st.tabs(["Backtest", "Optimize (SMA × buffer sweep)"])

with tab_bt:
    s = result.stats[result.strategy_name]
    lev = result.stats[result.leveraged_benchmark]
    c = st.columns(5)
    c[0].metric("CAGR", f"{s['CAGR']:.1%}", f"{s['CAGR'] - lev['CAGR']:+.1%} vs B&H {risk_on}")
    c[1].metric("Max drawdown", f"{s['Max drawdown']:.1%}", f"{s['Max drawdown'] - lev['Max drawdown']:+.1%}")
    c[2].metric("Sharpe (rf=0)", f"{s['Sharpe (rf=0)']:.2f}")
    c[3].metric("Switches", f"{result.switches}")
    c[4].metric("Time in market", f"{s['Time in market']:.0%}")

    eq = result.equity / result.equity.iloc[0]
    fig = go.Figure()
    for col in eq.columns:
        fig.add_trace(go.Scatter(x=eq.index, y=eq[col], name=col,
                                 line=dict(width=2.5 if col == result.strategy_name else 1.2)))
    fig.update_layout(title="Growth of $1 (log scale)", yaxis_type="log", height=480,
                      legend=dict(orientation="h", y=-0.15), hovermode="x unified")
    st.plotly_chart(fig, width="stretch")

    dd = drawdown(result.equity)
    fig_dd = go.Figure()
    for col in dd.columns:
        fig_dd.add_trace(go.Scatter(x=dd.index, y=dd[col], name=col,
                                    fill="tozeroy" if col == result.strategy_name else None))
    fig_dd.update_layout(title="Drawdown", yaxis_tickformat=".0%", height=320,
                         legend=dict(orientation="h", y=-0.2), hovermode="x unified")
    st.plotly_chart(fig_dd, width="stretch")

    fig_sig = go.Figure()
    fig_sig.add_trace(go.Scatter(
        x=result.in_market.index, y=result.in_market.astype(int), name=f"In {risk_on}",
        yaxis="y2", fill="tozeroy", line=dict(width=0, shape="hv"),
        fillcolor="rgba(46,160,67,0.15)", hoverinfo="skip",
    ))
    fig_sig.add_trace(go.Scatter(x=result.underlying.index, y=result.underlying, name=cfg.underlying))
    fig_sig.add_trace(go.Scatter(x=result.sma.index, y=result.sma, name=f"SMA {sma_length}"))
    if cfg.buffer:
        fig_sig.add_trace(go.Scatter(x=result.sma.index, y=result.sma * (1 + cfg.buffer),
                                     name="Buy line", line=dict(dash="dot", width=1)))
        fig_sig.add_trace(go.Scatter(x=result.sma.index, y=result.sma * (1 - cfg.buffer),
                                     name="Sell line", line=dict(dash="dot", width=1)))
    fig_sig.update_layout(
        title=f"{cfg.underlying} vs SMA (green = holding {risk_on})", yaxis_type="log", height=420,
        yaxis2=dict(overlaying="y", range=[0, 1], visible=False), hovermode="x unified",
        legend=dict(orientation="h", y=-0.15),
    )
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
    c1, c2, c3 = st.columns(3)
    sma_lo, sma_hi = c1.slider("SMA range", 20, 300, (100, 250), step=10)
    sma_step = c2.number_input("SMA step", 5, 100, 25, step=5)
    buf_max = c3.slider("Max buffer (%)", 0.5, 10.0, 5.0, step=0.5)
    buffers = tuple(round(b / 100, 4) for b in pd.Series(range(0, int(buf_max * 4) + 1)) * 0.25)
    smas = tuple(range(sma_lo, sma_hi + 1, int(sma_step)))
    metric = st.selectbox("Heatmap metric", ["CAGR", "Max drawdown", "Sharpe (rf=0)", "Calmar", "Switches"])

    if st.button("Run sweep", type="primary"):
        st.session_state["sweep_args"] = (risk_on, risk_off, data_mode, expense_pct / 100, cost_bps,
                                          str(start), str(end), smas, buffers)
    if "sweep_args" in st.session_state:
        sweep = cached_sweep(*st.session_state["sweep_args"])
        grid = sweep.pivot(index="sma_length", columns="buffer", values=metric)
        pct = metric in ("CAGR", "Max drawdown")
        fig_hm = px.imshow(
            grid.values, x=[f"{b:.2%}" for b in grid.columns], y=[str(i) for i in grid.index],
            color_continuous_scale="RdYlGn_r" if metric == "Switches" else "RdYlGn",
            aspect="auto", text_auto=".1%" if pct else (".0f" if metric == "Switches" else ".2f"),
            labels=dict(x="Buffer", y="SMA length", color=metric),
        )
        fig_hm.update_layout(title=f"{metric} by SMA length × buffer", height=520)
        st.plotly_chart(fig_hm, width="stretch")
        show = sweep.copy()
        for col in ("buffer", "CAGR", "Max drawdown", "Time in market"):
            show[col] = show[col].map("{:.2%}".format)
        st.dataframe(show.sort_values("sma_length"), width="stretch", hide_index=True)
        st.download_button("Download sweep CSV", sweep.to_csv(index=False), file_name="sweep.csv")

st.caption(
    "Caveats: leveraged ETFs suffer volatility decay; synthetic history ignores swap spreads and "
    "fund-specific frictions; taxes and slippage are not modeled. For research only, not investment advice."
)
