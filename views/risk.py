"""Market Risk page (`/risk`): VIX, MOVE and CNN Fear & Greed."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from fomo.data import get_price
from fomo.sentiment import fear_greed_rating, get_fear_greed

HOVER_DATE = "%Y/%m/%d"
PERIODS = {"YTD": "ytd", "1Y": 1, "5Y": 5, "10Y": 10, "Max": None}

# (upper bound, label, fill) — each band runs from the previous bound up to this one.
INDICATORS = {
    "VIX": dict(
        title="VIX — S&P 500 implied volatility (30-day)", color="#a371f7", fmt=".2f",
        help="CBOE Volatility Index (^VIX). Rough guide: < 20 calm, 20–30 elevated, > 30 high fear.",
        zones=[(20, "Calm", "rgba(46,160,67,0.10)"), (30, "Elevated", "rgba(210,153,34,0.12)"),
               (None, "High fear", "rgba(209,36,47,0.12)")],
    ),
    "MOVE": dict(
        title="MOVE — Treasury implied volatility (1-month)", color="#2f81f7", fmt=".1f",
        help="ICE BofA MOVE Index (^MOVE), the bond market's VIX. Rough guide: < 120 normal, "
             "120–150 stress, > 150 crisis.",
        zones=[(120, "Normal", "rgba(46,160,67,0.10)"), (150, "Stress", "rgba(210,153,34,0.12)"),
               (None, "Crisis", "rgba(209,36,47,0.12)")],
    ),
    "Fear & Greed": dict(
        title="CNN Fear & Greed index", color="#e3b341", fmt=".0f", yrange=[0, 100],
        help="0–100 composite of 7 market signals, from CNN's unofficial chart-data feed (history from 2021).",
        zones=[(25, "Extreme fear", "rgba(209,36,47,0.16)"), (45, "Fear", "rgba(209,36,47,0.07)"),
               (55, "Neutral", "rgba(128,128,128,0.06)"), (75, "Greed", "rgba(46,160,67,0.07)"),
               (100, "Extreme greed", "rgba(46,160,67,0.16)")],
    ),
}


@st.cache_data(ttl=6 * 3600, show_spinner="Loading market data...")
def load(name: str) -> pd.Series:
    if name == "Fear & Greed":
        return get_fear_greed()
    return get_price(f"^{name}")


def period_start(period: str, last: pd.Timestamp) -> pd.Timestamp | None:
    years = PERIODS[period]
    if years == "ytd":
        return pd.Timestamp(last.year, 1, 1)
    return None if years is None else last - pd.DateOffset(years=years)


PANEL_HEIGHT, PANEL_GAP = 300, 0.07


def risk_figure(views: dict[str, pd.Series]) -> go.Figure:
    """Stacked panels on one shared date axis so a hover shows every gauge for the same day."""
    n = len(views)
    h = (1 - PANEL_GAP * (n - 1)) / n
    fig = go.Figure()
    axes = {}
    for i, (name, s) in enumerate(views.items()):
        spec = INDICATORS[name]
        suffix = "" if i == 0 else str(i + 1)
        yref, top = f"y{suffix}", 1 - i * (h + PANEL_GAP)
        lo, hi = float(s.min()), float(s.max())
        pad = (hi - lo) * 0.08 or 1
        y0, y1 = spec.get("yrange", [max(0, lo - pad), hi + pad])
        axes[f"yaxis{suffix}"] = dict(domain=[max(0.0, round(top - h, 6)), min(1.0, round(top, 6))],
                                      range=[y0, y1])

        prev = 0.0
        for bound, label, fill in spec["zones"]:
            upper = bound if bound is not None else float("inf")
            # Clip each band to the visible range so its label stays on screen.
            b0, b1 = max(prev, y0), min(upper, y1)
            if b1 > b0:
                fig.add_shape(type="rect", xref="x domain", x0=0, x1=1, yref=yref, y0=b0, y1=b1,
                              fillcolor=fill, line_width=0, layer="below")
                fig.add_annotation(xref="x domain", x=0, yref=yref, y=b1, text=label, showarrow=False,
                                   xanchor="left", yanchor="top", font=dict(size=10), opacity=0.7)
            prev = upper
        fig.add_annotation(xref="paper", x=0, yref="paper", y=top, text=f"<b>{spec['title']}</b>",
                           showarrow=False, xanchor="left", yanchor="bottom", font=dict(size=14))

        hover = f"{name}: %{{y:{spec['fmt']}}}"
        custom = None
        if name == "Fear & Greed":
            custom = [fear_greed_rating(v) for v in s]
            hover += " · %{customdata}"
        fig.add_trace(go.Scatter(
            x=s.index, y=s, name=name, xaxis="x", yaxis=yref, customdata=custom,
            line=dict(color=spec["color"], width=1.6), hovertemplate=hover + "<extra></extra>",
        ))

    fig.update_layout(
        **axes, height=PANEL_HEIGHT * n, showlegend=False, margin=dict(t=40, b=30),
        hovermode="x unified", hoversubplots="axis",
        xaxis=dict(anchor=f"y{n if n > 1 else ''}", hoverformat=HOVER_DATE, showspikes=True,
                   spikemode="across", spikesnap="cursor", spikethickness=1, spikedash="dot",
                   spikecolor="gray"),
    )
    return fig


st.title("Market risk")
st.caption("Equity volatility, bond volatility and sentiment at a glance. Data: Yahoo Finance and CNN. "
           "Not investment advice.")

qp = st.query_params.get("period")
period = st.segmented_control("Period", list(PERIODS), default=qp if qp in PERIODS else "1Y",
                              key="risk_period") or "1Y"

series, errors = {}, {}
for name in INDICATORS:
    try:
        series[name] = load(name).dropna()
    except Exception as exc:  # network / blocked feed: show what we can
        errors[name] = exc

cols = st.columns(len(INDICATORS))
for col, (name, spec) in zip(cols, INDICATORS.items()):
    if name not in series:
        col.metric(name, "—", help=spec["help"])
        continue
    s = series[name]
    last, prev = s.iloc[-1], s.iloc[-2]
    label = f"{name} · {s.index[-1]:%Y/%m/%d}"
    value = f"{last:{spec['fmt']}}" + (f" · {fear_greed_rating(last)}" if name == "Fear & Greed" else "")
    # Higher VIX/MOVE = more risk (red); higher Fear & Greed = more greed (green).
    col.metric(label, value, f"{last - prev:+{spec['fmt']}}", help=spec["help"],
               delta_color="normal" if name == "Fear & Greed" else "inverse")

views = {}
for name in INDICATORS:
    if name in errors:
        st.warning(f"{name} unavailable: {errors[name]}")
        continue
    s = series[name]
    start = period_start(period, s.index[-1])
    views[name] = s.loc[start:] if start is not None else s
    if start is not None and s.index[0] > start:
        st.caption(f"{name} history starts {s.index[0]:%Y/%m/%d}.")
if views:
    st.plotly_chart(risk_figure(views), width="stretch", key="risk_chart")
