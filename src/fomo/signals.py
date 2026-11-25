"""Moving-average trend signal with optional hysteresis buffer."""

from __future__ import annotations

import numpy as np
import pandas as pd


def raw_trend_state(
    prices: pd.Series,
    sma_length: int = 200,
    buffer: float = 0.0,
    lower_buffer: float | None = None,
) -> pd.Series:
    """In-market state known at the *close* of each day.

    Enter when close > SMA * (1 + buffer) (upper / buy band); exit when
    close < SMA * (1 - lower_buffer) (lower / sell band); otherwise keep the
    previous state (hysteresis). ``lower_buffer`` defaults to ``buffer``
    (symmetric band). Out of the market until the SMA has a full window.
    """
    if lower_buffer is None:
        lower_buffer = buffer
    if sma_length < 1:
        raise ValueError("sma_length must be >= 1")
    if not 0.0 <= buffer < 1.0 or not 0.0 <= lower_buffer < 1.0:
        raise ValueError("buffers must be in [0, 1)")

    sma = prices.rolling(sma_length, min_periods=sma_length).mean()
    p = prices.to_numpy(dtype=float)
    upper = (sma * (1 + buffer)).to_numpy(dtype=float)
    lower = (sma * (1 - lower_buffer)).to_numpy(dtype=float)

    state = np.zeros(len(p), dtype=bool)
    in_market = False
    for i in range(len(p)):
        if np.isnan(upper[i]):
            in_market = False
        elif np.isnan(p[i]):
            pass
        elif in_market:
            in_market = not (p[i] < lower[i])
        else:
            in_market = p[i] > upper[i]
        state[i] = in_market
    return pd.Series(state, index=prices.index, name="in_market")


def trend_signal(
    prices: pd.Series,
    sma_length: int = 200,
    buffer: float = 0.0,
    lower_buffer: float | None = None,
    lag: int = 1,
) -> pd.Series:
    """Position to hold on each day, using only closes up to ``lag`` days earlier.

    With the default ``lag=1`` the position held (and traded at the close) on day t
    is decided from the close of day t-1, so there is no look-ahead.
    """
    raw = raw_trend_state(prices, sma_length, buffer, lower_buffer)
    return raw.shift(lag, fill_value=False).astype(bool)
