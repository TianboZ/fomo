"""CLI: run the trend-filter backtest, print stats, write a quantstats report.

Examples::

    fomo-backtest --risk-on TQQQ --buffer 0.01          # last 10 years (default)
    fomo-backtest --years 0                             # full history (synthetic before inception)
    fomo-backtest --risk-on UPRO --risk-off SHY --start 2007-01-01
    fomo-backtest --risk-on QLD --buffer 0.01
    fomo-backtest --upper-buffer 0.01 --lower-buffer 0.03   # buy at SMA+1%, sell at SMA-3%
    fomo-backtest --sweep --buffers 0 0.01 0.02 --lower-buffers 0 0.01 0.02 0.03
    fomo-backtest --sweep --sma-range 100 250 25 --buffers 0 0.005 0.01 0.02 0.03
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd

from .metrics import format_stats
from .strategies import DATA_MODES, RISK_OFF_CHOICES, UNDERLYING, TrendConfig, load_market_data, run_backtest

REPORT_DIR = Path(__file__).resolve().parents[2] / "reports"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Leveraged ETF 200-SMA trend-filter backtest")
    p.add_argument(
        "--risk-on",
        choices=list(UNDERLYING),
        default="TQQQ",
        help="TQQQ/UPRO = 3x QQQ/SPY, QLD/SSO = 2x QQQ/SPY",
    )
    p.add_argument("--risk-off", choices=RISK_OFF_CHOICES, default="CASH")
    p.add_argument("--sma", type=int, default=200, help="SMA length in trading days")
    p.add_argument(
        "--buffer",
        type=float,
        default=0.0,
        help="Symmetric hysteresis band as a fraction (0.01 = buy above SMA+1%%, sell below SMA-1%%)",
    )
    p.add_argument("--upper-buffer", type=float, default=None, help="Buy band only (overrides --buffer)")
    p.add_argument("--lower-buffer", type=float, default=None, help="Sell band only (overrides --buffer)")
    p.add_argument("--cost-bps", type=float, default=5.0, help="Transaction cost per trade, bps")
    p.add_argument("--data", choices=DATA_MODES, default="spliced", help="Leveraged ETF price source")
    p.add_argument("--expense-ratio", type=float, default=0.0095)
    p.add_argument("--initial-capital", type=float, default=100_000.0, help="Starting amount in $")
    p.add_argument(
        "--years",
        type=float,
        default=10,
        help="Backtest the last N years (default 10; 0 = full history). Ignored if --start is set",
    )
    p.add_argument("--start", default=None, help="YYYY-MM-DD (overrides --years)")
    p.add_argument("--end", default=None, help="YYYY-MM-DD")
    p.add_argument("--refresh", action="store_true", help="Re-download prices, ignoring cache")
    p.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    p.add_argument("--no-report", action="store_true", help="Skip the quantstats HTML report")
    p.add_argument("--sweep", action="store_true", help="Grid-search SMA length x buffer instead")
    p.add_argument("--sma-range", type=int, nargs=3, default=[100, 250, 25], metavar=("START", "STOP", "STEP"))
    p.add_argument(
        "--buffers",
        type=float,
        nargs="+",
        default=[0.0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05],
        help="Sweep buffers (symmetric), or upper buffers when --lower-buffers is given",
    )
    p.add_argument(
        "--lower-buffers",
        type=float,
        nargs="+",
        default=None,
        help="Sweep upper x lower buffers at the fixed --sma instead of SMA x symmetric buffer",
    )
    return p


def _resolve_start(args) -> str | None:
    if args.start or not args.years:
        return args.start
    end = pd.Timestamp(args.end) if args.end else pd.Timestamp.today().normalize()
    return str((end - pd.DateOffset(months=round(args.years * 12))).date())


def _print_proxy_notes(result) -> None:
    for p in result.proxy_periods:
        total = (result.equity.index[-1] - result.equity.index[0]).days or 1
        share = (p.end - p.start).days / total
        print(
            f"NOTE: {p.asset} did not exist {p.start.date()} -> {p.end.date()} "
            f"({share:.0%} of the period); used {p.description} instead."
        )


def _print_validation(validation: dict, risk_on: str) -> None:
    if not validation:
        return
    v = validation
    print(f"\nSynthetic vs real {risk_on} ({v['start']} -> {v['end']}):")
    print(f"  daily return corr   {v['daily_return_corr']:.4f}")
    print(f"  tracking error/yr   {v['tracking_error_ann']:.2%}")
    print(f"  CAGR synthetic/real {v['cagr_synthetic']:.2%} / {v['cagr_real']:.2%}"
          f"  (diff {v['cagr_difference']:+.2%}/yr)")


def _run_sweep(args, cfg: TrendConfig) -> None:
    from .sweep import parameter_sweep, pivot

    data = load_market_data(cfg.risk_on, cfg.risk_off, cfg.data_mode, cfg.leverage, cfg.expense_ratio, args.refresh)
    lo, hi, step = args.sma_range
    asymmetric = args.lower_buffers is not None
    sweep = parameter_sweep(
        data.underlying,
        data.prices[data.risk_on_col],
        data.prices[data.risk_off_col],
        sma_lengths=[cfg.sma_length] if asymmetric else range(lo, hi + 1, step),
        buffers=args.buffers,
        lower_buffers=args.lower_buffers,
        cost_bps=cfg.cost_bps,
        base=cfg.initial_capital,
        start=cfg.start,
        end=cfg.end,
    )
    pd.set_option("display.width", 200)
    for metric, fmt in [("CAGR", "{:.1%}"), ("Max drawdown", "{:.1%}"), ("Sharpe (rf=0)", "{:.2f}"), ("Switches", "{:.0f}")]:
        if asymmetric:
            print(f"\n{metric} at SMA {cfg.sma_length} (rows = lower/sell buffer, columns = upper/buy buffer)")
            grid = pivot(sweep, metric, rows="lower_buffer", cols="upper_buffer")
            grid.index = [f"{r:.1%}" for r in grid.index]
        else:
            print(f"\n{metric} (rows = SMA length, columns = symmetric buffer)")
            grid = pivot(sweep, metric)
        grid.columns = [f"{c:.1%}" for c in grid.columns]
        print(grid.map(fmt.format).to_string())
    args.report_dir.mkdir(parents=True, exist_ok=True)
    out = args.report_dir / f"sweep_{cfg.risk_on}_{cfg.risk_off}_{date.today():%Y%m%d}.csv"
    sweep.to_csv(out, index=False)
    print(f"\nFull sweep table: {out}")
    print("Caveat: the best cell in-sample is likely overfit; prefer broad, stable regions.")


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = TrendConfig(
        risk_on=args.risk_on,
        risk_off=args.risk_off,
        sma_length=args.sma,
        buffer=args.buffer if args.upper_buffer is None else args.upper_buffer,
        lower_buffer=args.buffer if args.lower_buffer is None else args.lower_buffer,
        cost_bps=args.cost_bps,
        data_mode=args.data,
        expense_ratio=args.expense_ratio,
        initial_capital=args.initial_capital,
        start=_resolve_start(args),
        end=args.end,
    )
    cfg.validate()

    if args.sweep:
        _run_sweep(args, cfg)
        return

    result = run_backtest(cfg, refresh=args.refresh)
    _print_validation(result.validation, cfg.risk_on)
    print()
    _print_proxy_notes(result)
    print(f"\n{result.strategy_name}  (data: {cfg.data_mode}, cost: {cfg.cost_bps:g} bps, start: ${cfg.initial_capital:,.0f})\n")
    print(format_stats(result.stats).to_string())

    if not args.no_report:
        from .report import write_quantstats_report

        name = f"{cfg.risk_on}_{cfg.risk_off}_sma{cfg.sma_length}_buf{cfg.buffer:g}-{cfg.sell_buffer:g}_{date.today():%Y%m%d}.html"
        path = write_quantstats_report(result, args.report_dir / name)
        print(f"\nquantstats report: {path}")


if __name__ == "__main__":
    main()
