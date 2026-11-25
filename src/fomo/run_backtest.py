"""CLI: run the trend-filter backtest, print stats, write a quantstats report.

Examples::

    fomo-backtest --risk-on TQQQ --buffer 0.01
    fomo-backtest --risk-on UPRO --risk-off SHY --start 2007-01-01
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
    p.add_argument("--risk-on", choices=list(UNDERLYING), default="TQQQ")
    p.add_argument("--risk-off", choices=RISK_OFF_CHOICES, default="CASH")
    p.add_argument("--sma", type=int, default=200, help="SMA length in trading days")
    p.add_argument(
        "--buffer",
        type=float,
        default=0.0,
        help="Hysteresis band as a fraction (0.01 = buy above SMA+1%%, sell below SMA-1%%)",
    )
    p.add_argument("--cost-bps", type=float, default=5.0, help="Transaction cost per trade, bps")
    p.add_argument("--data", choices=DATA_MODES, default="spliced", help="Leveraged ETF price source")
    p.add_argument("--expense-ratio", type=float, default=0.0095)
    p.add_argument("--start", default=None, help="YYYY-MM-DD")
    p.add_argument("--end", default=None, help="YYYY-MM-DD")
    p.add_argument("--refresh", action="store_true", help="Re-download prices, ignoring cache")
    p.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    p.add_argument("--no-report", action="store_true", help="Skip the quantstats HTML report")
    p.add_argument("--sweep", action="store_true", help="Grid-search SMA length x buffer instead")
    p.add_argument("--sma-range", type=int, nargs=3, default=[100, 250, 25], metavar=("START", "STOP", "STEP"))
    p.add_argument("--buffers", type=float, nargs="+", default=[0.0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05])
    return p


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
    sweep = parameter_sweep(
        data.underlying,
        data.prices[data.risk_on_col],
        data.prices[data.risk_off_col],
        sma_lengths=range(lo, hi + 1, step),
        buffers=args.buffers,
        cost_bps=cfg.cost_bps,
        start=cfg.start,
        end=cfg.end,
    )
    pd.set_option("display.width", 200)
    for metric, fmt in [("CAGR", "{:.1%}"), ("Max drawdown", "{:.1%}"), ("Sharpe (rf=0)", "{:.2f}"), ("Switches", "{:.0f}")]:
        print(f"\n{metric} (rows = SMA length, columns = buffer)")
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
        buffer=args.buffer,
        cost_bps=args.cost_bps,
        data_mode=args.data,
        expense_ratio=args.expense_ratio,
        start=args.start,
        end=args.end,
    )
    cfg.validate()

    if args.sweep:
        _run_sweep(args, cfg)
        return

    result = run_backtest(cfg, refresh=args.refresh)
    _print_validation(result.validation, cfg.risk_on)
    print(f"\n{result.strategy_name}  (data: {cfg.data_mode}, cost: {cfg.cost_bps:g} bps)\n")
    print(format_stats(result.stats).to_string())

    if not args.no_report:
        from .report import write_quantstats_report

        name = f"{cfg.risk_on}_{cfg.risk_off}_sma{cfg.sma_length}_buf{cfg.buffer:g}_{date.today():%Y%m%d}.html"
        path = write_quantstats_report(result, args.report_dir / name)
        print(f"\nquantstats report: {path}")


if __name__ == "__main__":
    main()
