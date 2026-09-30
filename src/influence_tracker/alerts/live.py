from __future__ import annotations

import logging
import math
from collections.abc import Callable
from datetime import datetime
from typing import Protocol

import pandas as pd

from .. import prices
from ..config import Watchlist
from ..events import MinuteBars

log = logging.getLogger(__name__)

Fetch1m = Callable[[str, datetime, datetime], pd.DataFrame]


class PriceSource(Protocol):
    def bars(self, symbol: str, start: datetime, end: datetime) -> MinuteBars: ...


def frame_to_bars(df: pd.DataFrame) -> MinuteBars:
    """Yahoo 1-minute frame (index = bar start) -> MinuteBars; NaN/non-positive closes dropped, duplicates keep last."""
    if df is None or df.empty or "Close" not in df:
        return MinuteBars([], [])
    by_start: dict[int, float] = {}
    for ts, close in df["Close"].items():
        value = float(close)
        if math.isfinite(value) and value > 0:
            by_start[int(pd.Timestamp(ts).timestamp())] = value
    starts = sorted(by_start)
    return MinuteBars(starts, [by_start[s] for s in starts])


class LivePrices:
    """Live Yahoo 1-minute bars for one watch cycle; failures come back as empty bars (the caller treats a missing
    price as not yet available)."""

    def __init__(self, watchlist: Watchlist, fetch: Fetch1m | None = None) -> None:
        self.watchlist = watchlist
        self.fetch = fetch or prices.fetch_yahoo_1m
        self._cache: dict[tuple[str, datetime, datetime], MinuteBars] = {}

    def bars(self, symbol: str, start: datetime, end: datetime) -> MinuteBars:
        key = (symbol, start, end)
        if key not in self._cache:
            ticker = self.watchlist.ticker(symbol)
            yahoo = ticker.yahoo_symbol if ticker else symbol
            try:
                self._cache[key] = frame_to_bars(self.fetch(yahoo, start, end))
            except Exception as e:  # yfinance raises many unrelated types for network and data problems
                log.warning("live 1m fetch for %s failed: %s: %s", symbol, type(e).__name__, e)
                self._cache[key] = MinuteBars([], [])
        return self._cache[key]
