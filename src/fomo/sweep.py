"""Fast vectorized parameter sweep over SMA length and buffer.

Uses the same signal and cost model as the bt backtest (trade at the close on the
day after the signal, cost charged on the traded notional), but in plain pandas so
hundreds of combinations run in seconds.
"""

from __future__ import annotations

import pandas as pd

from .metrics import equity_stats
from .signals import trend_signal


def strategy_equity(
    in_market: pd.Series,
    risk_on: pd.Series,
    risk_off: pd.Series,
    cost_bps: float = 5.0,
    base: float = 100.0,
) -> pd.Series:
    """Equity of holding ``risk_on`` when ``in_market`` else ``risk_off``.

    A position decided for day t is entered at day t's close, so it earns the
    return from t to t+1.
    """
    w = in_market.astype(float)
    r_on = risk_on.pct_change()
    r_off = risk_off.pct_change()
    prev = w.shift(1)
    gross = (prev * r_on + (1.0 - prev) * r_off).fillna(0.0)
    # Switching sells one asset and buys the other: 2x notional traded.
    turnover = 2.0 * w.diff().abs()
    turnover.iloc[0] = 1.0  # initial purchase
    eq = base * ((1.0 + gross) * (1.0 - turnover * cost_bps / 1e4)).cumprod()
    return eq.rename("equity")


def parameter_sweep(
    underlying: pd.Series,
    risk_on: pd.Series,
    risk_off: pd.Series,
    sma_lengths=range(100, 251, 25),
    buffers=(0.0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05),
    cost_bps: float = 5.0,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """One row per (sma_length, buffer) with CAGR, max drawdown, Sharpe, switches."""
    idx = risk_on.loc[start:end].index.intersection(risk_off.dropna().index)
    on, off = risk_on.reindex(idx), risk_off.reindex(idx)
    rows = []
    for n in sma_lengths:
        for b in buffers:
            sig = trend_signal(underlying, int(n), float(b)).reindex(idx).fillna(False).astype(bool)
            eq = strategy_equity(sig, on, off, cost_bps)
            s = equity_stats(eq)
            rows.append(
                {
                    "sma_length": int(n),
                    "buffer": float(b),
                    "CAGR": s["CAGR"],
                    "Max drawdown": s["Max drawdown"],
                    "Sharpe (rf=0)": s["Sharpe (rf=0)"],
                    "Calmar": s["Calmar"],
                    "Switches": int(sig.ne(sig.shift()).iloc[1:].sum()),
                    "Time in market": float(sig.mean()),
                }
            )
    return pd.DataFrame(rows)


def pivot(sweep: pd.DataFrame, metric: str) -> pd.DataFrame:
    """SMA length (rows) x buffer (columns) grid of ``metric``."""
    return sweep.pivot(index="sma_length", columns="buffer", values=metric)


