"""Synthetic daily-rebalanced leveraged ETF series built from index returns.

Model per trading day::

    r_lev = L * r_index - expense_ratio / 252 - (L - 1) * rf_daily

The ``(L - 1) * rf`` term approximates the cost of borrowing the extra exposure
(swaps / futures financing). Real funds pay a spread above that, so synthetic
series are usually slightly optimistic; see :func:`tracking_stats`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def daily_rf_from_irx(irx: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """Convert ^IRX (13-week T-bill yield, in percent) into a daily rate on ``index``."""
    aligned = irx.reindex(irx.index.union(index)).ffill().reindex(index)
    return (aligned.fillna(0.0) / 100.0 / TRADING_DAYS).rename("rf_daily")


def cash_index(rf_daily: pd.Series, base: float = 100.0) -> pd.Series:
    """Total-return index of T-bill cash."""
    return (base * (1.0 + rf_daily.fillna(0.0)).cumprod()).rename("CASH")


def synthetic_leveraged_returns(
    underlying_returns: pd.Series,
    leverage: float = 3.0,
    expense_ratio: float = 0.0095,
    rf_daily: pd.Series | float = 0.0,
) -> pd.Series:
    if isinstance(rf_daily, pd.Series):
        rf_daily = rf_daily.reindex(underlying_returns.index).fillna(0.0)
    cost = expense_ratio / TRADING_DAYS + (leverage - 1.0) * rf_daily
    # A daily loss beyond -100% would wipe the fund out; floor there.
    return (leverage * underlying_returns - cost).clip(lower=-1.0)


def synthetic_leveraged_prices(
    underlying_prices: pd.Series,
    leverage: float = 3.0,
    expense_ratio: float = 0.0095,
    rf_daily: pd.Series | float = 0.0,
    base: float = 100.0,
) -> pd.Series:
    prices = underlying_prices.dropna()
    rets = prices.pct_change().iloc[1:]
    lev = synthetic_leveraged_returns(rets, leverage, expense_ratio, rf_daily)
    out = pd.concat([pd.Series([1.0], index=prices.index[:1]), (1.0 + lev).cumprod()])
    return (base * out).rename(f"{underlying_prices.name}x{leverage:g}")


def splice(early: pd.Series, late: pd.Series) -> pd.Series:
    """Use ``early`` before ``late`` starts (rescaled to join continuously), then ``late``."""
    early, late = early.dropna(), late.dropna()
    if late.empty:
        return early
    first = late.index[0]
    anchor = early.asof(first) if not early.empty else np.nan
    if pd.isna(anchor):
        return late
    head = early[early.index < first] * (late.iloc[0] / anchor)
    return pd.concat([head, late]).rename(late.name)


def tracking_stats(synthetic: pd.Series, real: pd.Series) -> dict:
    """Compare synthetic vs. real series on their overlapping dates."""
    common = synthetic.dropna().index.intersection(real.dropna().index)
    if len(common) < 30:
        return {}
    s = synthetic.loc[common].pct_change().iloc[1:]
    r = real.loc[common].pct_change().iloc[1:]
    years = len(s) / TRADING_DAYS
    cagr_s = (1 + s).prod() ** (1 / years) - 1
    cagr_r = (1 + r).prod() ** (1 / years) - 1
    return {
        "start": common[0].date(),
        "end": common[-1].date(),
        "daily_return_corr": float(s.corr(r)),
        "tracking_error_ann": float((s - r).std() * np.sqrt(TRADING_DAYS)),
        "cagr_synthetic": float(cagr_s),
        "cagr_real": float(cagr_r),
        "cagr_difference": float(cagr_s - cagr_r),
    }
