"""quantstats HTML tearsheet generation."""

from __future__ import annotations

from pathlib import Path

from .strategies import BacktestResult


def write_quantstats_report(
    result: BacktestResult, output: str | Path, benchmark: str | None = None
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import quantstats as qs

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    bench_name = benchmark or result.underlying_benchmark
    eq = result.equity
    rets = eq[result.strategy_name].pct_change().iloc[1:].rename(result.strategy_name)
    bench = eq[bench_name].pct_change().iloc[1:].rename(bench_name)
    qs.reports.html(
        rets,
        benchmark=bench,
        output=str(output),
        download_filename=str(output),
        title=f"{result.strategy_name} vs {bench_name}",
    )
    return output
