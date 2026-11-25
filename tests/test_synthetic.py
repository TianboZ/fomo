import numpy as np
import pandas as pd
from numpy.testing import assert_allclose

from fomo.sweep import strategy_equity
from fomo.synthetic import (
    daily_rf_from_irx,
    splice,
    synthetic_leveraged_prices,
    synthetic_leveraged_returns,
)

IDX = pd.bdate_range("2020-01-01", periods=4)


def test_zero_cost_is_exact_multiple():
    r = pd.Series([0.01, -0.02, 0.005, 0.0], index=IDX)
    assert_allclose(synthetic_leveraged_returns(r, 3.0, 0.0, 0.0), 3 * r)


def test_expense_and_borrow_costs_deducted():
    r = pd.Series([0.01, -0.02, 0.005, 0.0], index=IDX)
    rf = pd.Series(0.04 / 252, index=IDX)
    out = synthetic_leveraged_returns(r, 3.0, 0.0095, rf)
    assert_allclose(out, 3 * r - 0.0095 / 252 - 2 * 0.04 / 252)


def test_loss_floored_at_minus_100pct():
    r = pd.Series([-0.5], index=IDX[:1])
    assert synthetic_leveraged_returns(r, 3.0, 0.0, 0.0).iloc[0] == -1.0


def test_prices_compound_daily():
    under = pd.Series([100.0, 101.0, 99.99], index=IDX[:3], name="QQQ")
    out = synthetic_leveraged_prices(under, 3.0, 0.0, 0.0)
    r1, r2 = 0.01, 99.99 / 101 - 1
    assert_allclose(out.to_numpy(), [100, 100 * (1 + 3 * r1), 100 * (1 + 3 * r1) * (1 + 3 * r2)])


def test_volatility_decay():
    # Underlying round-trips to its start; the daily-rebalanced 3x fund ends lower.
    under = pd.Series([100.0, 110.0, 100.0] * 20, index=pd.bdate_range("2020-01-01", periods=60))
    under = under.iloc[:-2]  # end on 100
    out = synthetic_leveraged_prices(under, 3.0, 0.0, 0.0)
    assert under.iloc[-1] == under.iloc[0]
    assert out.iloc[-1] < out.iloc[0]


def test_daily_rf_from_irx_forward_fills():
    irx = pd.Series([5.04], index=[IDX[0]])
    rf = daily_rf_from_irx(irx, IDX)
    assert_allclose(rf, 0.0504 / 252)


def test_splice_is_continuous():
    early = pd.Series([10.0, 11.0, 12.0, 13.2], index=IDX)
    late = pd.Series([60.0, 66.0], index=IDX[2:])
    out = splice(early, late)
    assert_allclose(out.to_numpy(), [50.0, 55.0, 60.0, 66.0])
    assert_allclose(out.pct_change().iloc[1:], early.pct_change().iloc[1:])


def test_strategy_equity_switching_and_costs():
    on = pd.Series([100.0, 110.0, 121.0, 121.0], index=IDX)
    off = pd.Series([100.0, 100.0, 100.0, 100.0], index=IDX)
    sig = pd.Series([True, True, False, False], index=IDX)
    eq = strategy_equity(sig, on, off, cost_bps=0.0)
    assert_allclose(eq.to_numpy(), [100, 110, 121, 121])
    eq_cost = strategy_equity(sig, on, off, cost_bps=10.0)
    c = 10 / 1e4
    assert_allclose(eq_cost.to_numpy(), np.array([1, 1.1, 1.21 * (1 - 2 * c), 1.21 * (1 - 2 * c)]) * 100 * (1 - c))


def test_sweep_symmetric_and_asymmetric_grids():
    from fomo.sweep import parameter_sweep

    idx = pd.bdate_range("2020-01-01", periods=120)
    under = pd.Series(100 * np.exp(np.cumsum(np.sin(np.arange(120) / 7) * 0.02)), index=idx)
    off = pd.Series(100.0, index=idx)
    sym = parameter_sweep(under, under, off, sma_lengths=[10, 20], buffers=[0.0, 0.02])
    assert len(sym) == 4 and (sym.upper_buffer == sym.lower_buffer).all()
    asym = parameter_sweep(under, under, off, sma_lengths=[10], buffers=[0.0, 0.02], lower_buffers=[0.0, 0.01, 0.03])
    assert len(asym) == 6
    assert set(zip(asym.upper_buffer, asym.lower_buffer)) == {(u, lo) for u in (0.0, 0.02) for lo in (0.0, 0.01, 0.03)}
