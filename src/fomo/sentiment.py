"""CNN Fear & Greed index via CNN's (unofficial) chart-data endpoint, cached like prices."""

from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

import pandas as pd

from .data import CACHE_MAX_AGE_SECONDS, DEFAULT_CACHE_DIR

FEAR_GREED_URL = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata/{start}"
# The endpoint returns a server error for start dates before ~2021.
FEAR_GREED_START = "2021-01-01"
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120 Safari/537.36",
    "Referer": "https://www.cnn.com/",
    "Origin": "https://www.cnn.com",
    "Accept": "application/json, text/plain, */*",  # without these CNN answers 418
    "Accept-Language": "en-US,en;q=0.9",
}


def parse_fear_greed(payload: dict) -> pd.Series:
    """Daily 0-100 score indexed by date (one point per day, last value wins)."""
    points = payload["fear_and_greed_historical"]["data"]
    s = pd.Series(
        [float(p["y"]) for p in points],
        index=pd.to_datetime([p["x"] for p in points], unit="ms").normalize(),
        name="Fear & Greed",
    )
    return s[~s.index.duplicated(keep="last")].sort_index()


def _download_fear_greed() -> pd.Series:
    req = urllib.request.Request(FEAR_GREED_URL.format(start=FEAR_GREED_START), headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=20) as resp:
        return parse_fear_greed(json.load(resp))


def get_fear_greed(refresh: bool = False, cache_dir: Path | None = None) -> pd.Series:
    cache_dir = Path(cache_dir or DEFAULT_CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "FEAR_GREED.csv"

    def read_cache() -> pd.Series:
        return pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0].rename("Fear & Greed")

    fresh = path.exists() and (time.time() - path.stat().st_mtime) < CACHE_MAX_AGE_SECONDS
    if fresh and not refresh:
        return read_cache()
    try:
        series = _download_fear_greed()
    except Exception:
        if path.exists():  # blocked / rate-limited: fall back to stale cache
            return read_cache()
        raise
    series.to_csv(path, header=True)
    return series


def fear_greed_rating(score: float) -> str:
    """CNN's bands: <25 extreme fear, <45 fear, <=55 neutral, <=75 greed, else extreme greed."""
    if score < 25:
        return "Extreme fear"
    if score < 45:
        return "Fear"
    if score <= 55:
        return "Neutral"
    if score <= 75:
        return "Greed"
    return "Extreme greed"
