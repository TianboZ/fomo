"""Price download (yfinance) with a local CSV cache in ``data/``."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd

DEFAULT_CACHE_DIR = Path(
    os.environ.get("FOMO_DATA_DIR", Path(__file__).resolve().parents[2] / "data")
)
CACHE_MAX_AGE_SECONDS = 20 * 3600
HISTORY_START = "1980-01-01"


def _cache_path(ticker: str, cache_dir: Path) -> Path:
    safe = ticker.replace("^", "_").replace("/", "_")
    return cache_dir / f"{safe}.csv"


def _download(ticker: str) -> pd.Series:
    import yfinance as yf

    df = yf.download(
        ticker,
        start=HISTORY_START,
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if df is None or df.empty:
        raise ValueError(f"No data returned by yfinance for {ticker!r}")
    close = df["Close"]
    if isinstance(close, pd.DataFrame):  # newer yfinance returns (field, ticker) columns
        close = close.iloc[:, 0]
    idx = pd.to_datetime(close.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    close.index = idx.normalize()
    return close.astype(float).dropna().rename(ticker)


def _read_cache(path: Path, ticker: str) -> pd.Series:
    s = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
    return s.astype(float).dropna().rename(ticker)


def get_price(ticker: str, refresh: bool = False, cache_dir: Path | None = None) -> pd.Series:
    """Adjusted (total-return) close for ``ticker``, full history, cached on disk."""
    cache_dir = Path(cache_dir or DEFAULT_CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(ticker, cache_dir)

    fresh = path.exists() and (time.time() - path.stat().st_mtime) < CACHE_MAX_AGE_SECONDS
    if fresh and not refresh:
        return _read_cache(path, ticker)

    try:
        series = _download(ticker)
    except Exception:
        if path.exists():  # offline / rate-limited: fall back to stale cache
            return _read_cache(path, ticker)
        raise
    series.to_csv(path, header=True)
    return series
