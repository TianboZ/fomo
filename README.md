# fomo — leveraged ETF trend-filter backtest

Backtests a simple rotation: **hold a 3x ETF (TQQQ or UPRO) while its underlying index
(QQQ / SPY) is above its 200-day SMA, otherwise hold cash or bonds** — with an optional
buffer band, a parameter sweep, a quantstats tearsheet and a Streamlit dashboard.

Stack: Python 3.11+, [`bt`](https://pmorissette.github.io/bt/) (backtest engine),
`yfinance` (data), `quantstats` (tearsheet), `streamlit` + `plotly` (dashboard), `pytest`.

## Setup

```bash
# with uv
uv sync && source .venv/bin/activate

# or plain venv (Python >= 3.11; macOS system python 3.9 is too old)
python3.13 -m venv .venv && .venv/bin/pip install -e . pytest
source .venv/bin/activate
```

Prices are downloaded from Yahoo Finance and cached as CSV in `data/` (refreshed after
~20h, or with `--refresh`). Reports go to `reports/`. Both are git-ignored.

## Strategy rules

* **Signal** on the underlying (QQQ for TQQQ, SPY for UPRO) using total-return closes.
* **Buffer band `b`** (hysteresis): buy only when close > SMA × (1 + b), sell only when
  close < SMA × (1 − b); in between, keep the previous position. `b = 0` is the plain
  200-SMA rule. E.g. `--buffer 0.01` = buy 1% above the SMA, sell 1% below it.
* **No look-ahead:** the position held on day *t* is decided from day *t−1*'s close
  and traded at day *t*'s close.
* **Risk-off:** `CASH` (13-week T-bills, `^IRX`), `SHY` or `TLT` (T-bills before their 2002 inception).
* **Costs:** `--cost-bps` per trade (default 5 bps on traded notional; a switch costs 2×).
* Benchmarks: buy-and-hold underlying, SPY, and the leveraged ETF.

### Synthetic leveraged history

TQQQ/UPRO only exist since 2010/2009, which misses 2000–02 and 2008. Synthetic 3x series are
built from the underlying's daily returns:

    r_3x = 3 · r_index − expense_ratio/252 − 2 · r_tbill_daily

`--data spliced` (default) uses synthetic data before inception and the real ETF after;
`synthetic` and `real` are also available. The CLI prints a synthetic-vs-real check over the
overlap (≈0.999 daily correlation for TQQQ; synthetic runs ~1.5%/yr optimistic because real
funds pay swap spreads above T-bills).

## CLI

```bash
fomo-backtest                                     # TQQQ / cash, 200 SMA, writes HTML tearsheet
fomo-backtest --risk-on TQQQ --buffer 0.01        # ±1% buffer band
fomo-backtest --risk-on UPRO --risk-off SHY --start 2007-01-01 --cost-bps 10
fomo-backtest --data real --no-report             # actual ETF history only
fomo-backtest --sweep --sma-range 100 250 25 --buffers 0 0.005 0.01 0.02 0.03 0.05
```

`--sweep` grid-searches SMA length × buffer with a fast vectorized engine (verified to match
the `bt` result), prints CAGR / max drawdown / Sharpe / switch-count grids and saves a CSV.

## Dashboard

```bash
streamlit run app.py
```

Sidebar: risk-on ETF, risk-off asset, SMA length, buffer %, data mode, costs, date range.
**Backtest** tab: log equity curve vs benchmarks, drawdowns, underlying vs SMA (with buy/sell
lines and in-market shading), stats table, synthetic validation, downloadable quantstats
tearsheet. **Optimize** tab: SMA × buffer heatmap for any metric plus the full table.

## Tests

```bash
pytest
```

Offline unit tests cover no-look-ahead signal timing, buffer hysteresis, synthetic leverage
math (3x returns minus costs, volatility decay), splicing, and the sweep engine's cost model.

## Caveats

* **Overfitting:** with ~25 years of data there are only a handful of bear markets, and the
  sweep grid is noisy (neighbouring cells can differ by several % CAGR). Don't pick the best
  cell — prefer broad stable regions, and validate out of sample (e.g. optimize on data before
  2015 via `--end`, then test after with `--start`).
* **Volatility decay:** daily-rebalanced 3x funds lose value in choppy markets; buy-and-hold
  synthetic TQQQ was nearly wiped out in 2000–02.
* Synthetic data ignores swap spreads, fund closures and intraday risk; taxes, slippage and
  the gap risk of trading at the next close are not modeled.
* Yahoo data can have errors or revisions.
* **Research only — not investment advice.**
