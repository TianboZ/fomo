import numpy as np
import pandas as pd
import pytest

from fomo.signals import raw_trend_state, trend_signal


def _series(values):
    return pd.Series(values, index=pd.bdate_range("2020-01-01", periods=len(values)), dtype=float)


def test_signal_is_prior_day_state():
    prices = _series(np.linspace(100, 130, 30))
    sig = trend_signal(prices, sma_length=5)
    raw = raw_trend_state(prices, sma_length=5)
    assert not sig.iloc[0]
    assert (sig.iloc[1:].to_numpy() == raw.iloc[:-1].to_numpy()).all()


def test_warmup_is_out_of_market():
    raw = raw_trend_state(_series(np.linspace(100, 130, 30)), sma_length=10)
    assert not raw.iloc[:9].any()
    assert raw.iloc[9:].all()


def test_no_lookahead():
    prices = _series(np.linspace(100, 160, 40))
    k = 25
    crashed = prices.copy()
    crashed.iloc[k:] *= 0.1
    a = trend_signal(prices, sma_length=10)
    b = trend_signal(crashed, sma_length=10)
    # A crash starting on day k cannot change any position up to and including day k.
    pd.testing.assert_series_equal(a.iloc[: k + 1], b.iloc[: k + 1])
    assert b.iloc[k] and not b.iloc[k + 1]


def test_buffer_hysteresis():
    # sma_length=2, buffer=10%:
    # t1: SMA 105, buy line 115.5 -> 110 does not enter
    # t2: SMA 125, buy line 137.5 -> 140 enters
    # t3: SMA 130, sell line 117  -> 120 stays in (would exit without buffer)
    # t4: SMA 110, sell line 99   -> 100 stays in
    # t5: SMA 90,  sell line 81   -> 80 exits
    prices = _series([100, 110, 140, 120, 100, 80])
    buffered = raw_trend_state(prices, sma_length=2, buffer=0.10)
    plain = raw_trend_state(prices, sma_length=2, buffer=0.0)
    assert buffered.tolist() == [False, False, True, True, True, False]
    assert plain.tolist() == [False, True, True, False, False, False]


def test_buffer_one_percent_band():
    prices = _series([100.0] * 5 + [100.5, 101.5, 100.2, 99.5, 98.5])
    sma_len = 5
    raw = raw_trend_state(prices, sma_length=sma_len, buffer=0.01)
    sma = prices.rolling(sma_len).mean()
    expected, state = [], False
    for p, m in zip(prices, sma):
        if np.isnan(m):
            state = False
        elif state:
            state = not p < m * 0.99
        else:
            state = p > m * 1.01
        expected.append(state)
    assert raw.tolist() == expected
    assert raw.iloc[6] and raw.iloc[7] and raw.iloc[8]  # held inside the band


@pytest.mark.parametrize("kwargs", [{"sma_length": 0}, {"buffer": -0.01}, {"buffer": 1.0}])
def test_invalid_params(kwargs):
    with pytest.raises(ValueError):
        raw_trend_state(_series(range(1, 20)), **{"sma_length": 5, **kwargs})


def test_asymmetric_buffers():
    prices = _series([100, 110, 140, 120, 100, 80])
    # Buy at the SMA, sell only 10% below it: rides through the t3/t4 dip.
    loose_exit = raw_trend_state(prices, sma_length=2, buffer=0.0, lower_buffer=0.10)
    assert loose_exit.tolist() == [False, True, True, True, True, False]
    # Buy only 10% above the SMA, sell as soon as it crosses below.
    strict_entry = raw_trend_state(prices, sma_length=2, buffer=0.10, lower_buffer=0.0)
    assert strict_entry.tolist() == [False, False, True, False, False, False]
