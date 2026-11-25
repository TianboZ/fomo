"""Performance statistics computed from equity curves."""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252

PERCENT_ROWS = ["Total return", "CAGR", "Volatility (ann.)", "Max drawdown", "Time in market"]


def drawdown(equity):
    return equity / equity.cummax() - 1.0


def equity_stats(eq: pd.Series) -> dict:
    eq = eq.dropna()
    rets = eq.pct_change().iloc[1:]
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1
    vol = rets.std() * np.sqrt(TRADING_DAYS)
    downside = np.sqrt((np.minimum(rets, 0.0) ** 2).mean()) * np.sqrt(TRADING_DAYS)
    mean_ann = rets.mean() * TRADING_DAYS
    max_dd = drawdown(eq).min()
    return {
        "Start": eq.index[0].date(),
        "End": eq.index[-1].date(),
        "Total return": eq.iloc[-1] / eq.iloc[0] - 1,
        "CAGR": cagr,
        "Volatility (ann.)": vol,
        "Sharpe (rf=0)": mean_ann / vol if vol > 0 else np.nan,
        "Sortino (rf=0)": mean_ann / downside if downside > 0 else np.nan,
        "Max drawdown": max_dd,
        "Calmar": cagr / abs(max_dd) if max_dd < 0 else np.nan,
    }


def performance_stats(equity: pd.DataFrame) -> pd.DataFrame:
    """One column per strategy, one row per metric."""
    return pd.DataFrame({col: equity_stats(equity[col]) for col in equity.columns})


def format_stats(stats: pd.DataFrame) -> pd.DataFrame:
    """String-formatted copy for display."""
    out = pd.DataFrame(index=stats.index, columns=stats.columns, dtype=object)
    for row in stats.index:
        for col in stats.columns:
            v = stats.loc[row, col]
            if v is None or (isinstance(v, (float, np.floating)) and np.isnan(v)):
                out.loc[row, col] = ""
            elif row in PERCENT_ROWS:
                out.loc[row, col] = f"{v:.1%}"
            elif isinstance(v, (float, np.floating)):
                out.loc[row, col] = f"{v:.2f}"
            else:
                out.loc[row, col] = str(v)
    return out
