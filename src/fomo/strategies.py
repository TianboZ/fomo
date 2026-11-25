"""Market-data assembly and the bt-based trend-filter backtest."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .data import get_price
from .metrics import performance_stats
from .signals import trend_signal
from .synthetic import (
    cash_index,
    daily_rf_from_irx,
    splice,
    synthetic_leveraged_prices,
    tracking_stats,
)

# Leveraged ETF -> (underlying used for the signal and synthetic history, daily leverage)
LEVERAGED_ETFS = {
    "TQQQ": ("QQQ", 3.0),
    "UPRO": ("SPY", 3.0),
    "QLD": ("QQQ", 2.0),
    "SSO": ("SPY", 2.0),
}
UNDERLYING = {etf: u for etf, (u, _) in LEVERAGED_ETFS.items()}
LEVERAGE = {etf: lev for etf, (_, lev) in LEVERAGED_ETFS.items()}
RISK_OFF_CHOICES = ("CASH", "SHY", "TLT")
DATA_MODES = ("spliced", "synthetic", "real")
RF_TICKER = "^IRX"


@dataclass
class TrendConfig:
    risk_on: str = "TQQQ"
    risk_off: str = "CASH"
    sma_length: int = 200
    buffer: float = 0.0  # upper (buy) band
    lower_buffer: float | None = None  # lower (sell) band; None = same as buffer
    cost_bps: float = 5.0
    data_mode: str = "spliced"
    leverage: float | None = None  # defaults to the ETF's leverage (3x TQQQ/UPRO, 2x QLD/SSO)
    expense_ratio: float = 0.0095
    initial_capital: float = 100_000.0
    start: str | None = None
    end: str | None = None

    def __post_init__(self) -> None:
        if self.leverage is None and self.risk_on in LEVERAGE:
            self.leverage = LEVERAGE[self.risk_on]

    def validate(self) -> None:
        if self.risk_on not in UNDERLYING:
            raise ValueError(f"risk_on must be one of {list(UNDERLYING)}")
        if self.risk_off not in RISK_OFF_CHOICES:
            raise ValueError(f"risk_off must be one of {RISK_OFF_CHOICES}")
        if self.data_mode not in DATA_MODES:
            raise ValueError(f"data_mode must be one of {DATA_MODES}")
        if self.sma_length < 1:
            raise ValueError("sma_length must be >= 1")
        if not 0.0 <= self.buffer < 1.0 or not 0.0 <= self.sell_buffer < 1.0:
            raise ValueError("buffers must be in [0, 1)")
        if not self.leverage or self.leverage <= 0:
            raise ValueError("leverage must be > 0")
        if self.cost_bps < 0:
            raise ValueError("cost_bps must be >= 0")
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be > 0")

    @property
    def underlying(self) -> str:
        return UNDERLYING[self.risk_on]

    @property
    def sell_buffer(self) -> float:
        return self.buffer if self.lower_buffer is None else self.lower_buffer

    @property
    def buffer_label(self) -> str:
        if self.buffer == self.sell_buffer:
            return f"±{self.buffer:.2%}" if self.buffer else ""
        return f"+{self.buffer:.2%}/−{self.sell_buffer:.2%}"

    @property
    def label(self) -> str:
        buf = f" {self.buffer_label}" if self.buffer_label else ""
        return f"Trend {self.sma_length}SMA{buf} {self.risk_on}/{self.risk_off}"


@dataclass
class ProxyPeriod:
    """A span where an asset's prices are a stand-in because the real ETF did not exist yet."""

    asset: str
    description: str
    start: pd.Timestamp | None  # None = from the beginning of the data
    end: pd.Timestamp  # exclusive: first date of real data (or last date if fully synthetic)

    def clip(self, start: pd.Timestamp, end: pd.Timestamp) -> "ProxyPeriod | None":
        s = max(self.start or start, start)
        e = min(self.end, end)
        return ProxyPeriod(self.asset, self.description, s, e) if s < e else None


@dataclass
class MarketData:
    prices: pd.DataFrame  # aligned, NaN-free total-return prices
    underlying: pd.Series  # full underlying history (SMA warm-up)
    underlying_col: str
    risk_on_col: str
    risk_off_col: str
    benchmark_cols: list[str]
    validation: dict = field(default_factory=dict)
    proxy_periods: list[ProxyPeriod] = field(default_factory=list)


def load_market_data(
    risk_on: str = "TQQQ",
    risk_off: str = "CASH",
    data_mode: str = "spliced",
    leverage: float | None = None,
    expense_ratio: float = 0.0095,
    refresh: bool = False,
) -> MarketData:
    underlying = UNDERLYING[risk_on]
    if leverage is None:
        leverage = LEVERAGE[risk_on]
    und = get_price(underlying, refresh)
    irx = get_price(RF_TICKER, refresh)
    # Use the prior day's T-bill yield for today's financing/cash accrual.
    rf_daily = daily_rf_from_irx(irx, und.index).shift(1).fillna(0.0)
    cash = cash_index(rf_daily)

    syn = synthetic_leveraged_prices(und, leverage, expense_ratio, rf_daily)
    syn_desc = f"synthetic {leverage:g}x {underlying}"
    validation: dict = {}
    proxies: list[ProxyPeriod] = []
    if data_mode == "synthetic":
        lev, lev_col = syn, f"{risk_on}_syn"
        proxies.append(ProxyPeriod(risk_on, f"{syn_desc} (synthetic-only mode)", None, und.index[-1]))
    else:
        real = get_price(risk_on, refresh)
        validation = tracking_stats(syn, real)
        if data_mode == "real":
            lev, lev_col = real, risk_on
        else:
            lev, lev_col = splice(syn, real), risk_on
            proxies.append(ProxyPeriod(risk_on, syn_desc, None, real.index[0]))

    if risk_off == "CASH":
        off = cash
    else:
        # SHY/TLT start in 2002; before that, fall back to T-bill cash.
        real_off = get_price(risk_off, refresh)
        off = splice(cash, real_off)
        proxies.append(ProxyPeriod(risk_off, "13-week T-bills", None, real_off.index[0]))

    cols = {underlying: und, lev_col: lev, risk_off: off}
    benchmark_cols = [underlying]
    if underlying != "SPY":
        cols["SPY"] = get_price("SPY", refresh)
        benchmark_cols.append("SPY")
    benchmark_cols.append(lev_col)

    prices = pd.concat(cols, axis=1).reindex(und.index).dropna()
    return MarketData(
        prices=prices,
        underlying=und,
        underlying_col=underlying,
        risk_on_col=lev_col,
        risk_off_col=risk_off,
        benchmark_cols=benchmark_cols,
        validation=validation,
        proxy_periods=proxies,
    )


@dataclass
class BacktestResult:
    config: TrendConfig
    equity: pd.DataFrame  # strategy + benchmarks, in dollars (starts at initial_capital)
    strategy_name: str
    benchmark_names: list[str]
    underlying_benchmark: str
    leveraged_benchmark: str
    in_market: pd.Series
    underlying: pd.Series
    sma: pd.Series
    stats: pd.DataFrame
    switches: int
    validation: dict
    proxy_periods: list[ProxyPeriod] = field(default_factory=list)


def _fee(cost_bps: float):
    rate = cost_bps / 1e4
    return lambda quantity, price: abs(quantity) * price * rate


def _buy_and_hold(name: str, ticker: str):
    import bt

    return bt.Strategy(
        name,
        [
            bt.algos.RunOnce(),
            bt.algos.SelectThese([ticker]),
            bt.algos.WeighEqually(),
            bt.algos.Rebalance(),
        ],
    )


def run_backtest(cfg: TrendConfig, data: MarketData | None = None, refresh: bool = False) -> BacktestResult:
    import bt

    cfg.validate()
    if data is None:
        data = load_market_data(
            cfg.risk_on, cfg.risk_off, cfg.data_mode, cfg.leverage, cfg.expense_ratio, refresh
        )

    # Signal and SMA use the full underlying history so the window is warm at `start`.
    full_signal = trend_signal(data.underlying, cfg.sma_length, cfg.buffer, cfg.sell_buffer)
    full_sma = data.underlying.rolling(cfg.sma_length, min_periods=cfg.sma_length).mean()

    prices = data.prices.loc[cfg.start : cfg.end]
    if len(prices) < 2:
        raise ValueError("Not enough data in the selected date range")

    in_mkt = full_signal.reindex(prices.index).fillna(False).astype(bool)
    on, off = data.risk_on_col, data.risk_off_col
    weights = pd.DataFrame({on: in_mkt.astype(float), off: 1.0 - in_mkt.astype(float)})
    changed = in_mkt.ne(in_mkt.shift())
    changed.iloc[0] = True
    # Only rebalance on signal changes (avoids daily drift-rebalancing costs).
    targets = weights.loc[changed]

    fee = _fee(cfg.cost_bps)
    common = dict(
        commissions=fee,
        integer_positions=False,
        progress_bar=False,
        initial_capital=cfg.initial_capital,
    )
    strat_name = cfg.label
    tests = [
        bt.Backtest(
            bt.Strategy(strat_name, [bt.algos.WeighTarget(targets), bt.algos.Rebalance()]),
            prices[[on, off]],
            **common,
        )
    ]
    bench_names = []
    for col in data.benchmark_cols:
        name = f"B&H {col}"
        bench_names.append(name)
        tests.append(bt.Backtest(_buy_and_hold(name, col), prices[[col]], **common))

    res = bt.run(*tests)
    # bt reports prices rebased to 100; convert back to dollars.
    equity = res.prices.copy() / 100.0 * cfg.initial_capital
    equity = equity.loc[equity.index >= prices.index[0], [strat_name, *bench_names]]

    stats = performance_stats(equity)
    switches = int(changed.iloc[1:].sum())
    stats.loc["Switches", strat_name] = switches
    stats.loc["Time in market", strat_name] = float(in_mkt.mean())

    return BacktestResult(
        config=cfg,
        equity=equity,
        strategy_name=strat_name,
        benchmark_names=bench_names,
        underlying_benchmark=f"B&H {data.underlying_col}",
        leveraged_benchmark=f"B&H {data.risk_on_col}",
        in_market=in_mkt,
        underlying=data.underlying.reindex(prices.index),
        sma=full_sma.reindex(prices.index),
        stats=stats,
        switches=switches,
        validation=data.validation,
        proxy_periods=[
            c for p in data.proxy_periods if (c := p.clip(prices.index[0], prices.index[-1])) is not None
        ],
    )
