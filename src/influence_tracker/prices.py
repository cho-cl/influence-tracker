from __future__ import annotations

import logging
import sqlite3
import time
from collections.abc import Callable, Sequence
from datetime import date, datetime, timedelta
from typing import Literal

import pandas as pd
import yfinance as yf
from yfinance.exceptions import YFPricesMissingError, YFRateLimitError

from . import db, market
from .config import Ticker, Watchlist
from .timeutil import NY

log = logging.getLogger(__name__)

# Yahoo rejects 1m requests spanning more than 8 days (checked live 2026-09-24); keep a day of margin.
MAX_REQUEST_SPAN = timedelta(days=7)
RATE_LIMIT_WAIT_S = 60.0
YF_NETWORK_RETRIES = 2

_OHLC = ["Open", "High", "Low", "Close"]
_EPOCH = pd.Timestamp(0, tz="UTC")
_DAY_START_MIN = market.EXTENDED_OPEN.hour * 60 + market.EXTENDED_OPEN.minute
_DAY_END_MIN = market.EXTENDED_CLOSE.hour * 60 + market.EXTENDED_CLOSE.minute

Fetch = Callable[[str, datetime, datetime], pd.DataFrame]
Bar = tuple[int, float, float, float, float, float | None]
Fetched = list[tuple[list[date], dict[date, list[Bar]]]]


def fetch_yahoo_1m(yahoo_symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Yahoo 1-minute bars incl. pre/post market for [start, end), indexed by bar start in New York time."""
    yf.config.network.retries = YF_NETWORK_RETRIES
    # Otherwise yfinance logs failures and returns an empty frame, indistinguishable from a day without trades.
    yf.config.debug.hide_exceptions = False
    return yf.Ticker(yahoo_symbol).history(
        start=start, end=end, interval="1m", prepost=True, auto_adjust=False, actions=False
    )


def snapshot_window(now: datetime, backfill_days: int) -> list[date]:
    """Sessions from backfill_days calendar days before now's New York date through the last completed one."""
    last = market.last_completed_session(now)
    if last is None:
        return []
    return market.sessions_in_range(now.astimezone(NY).date() - timedelta(days=backfill_days), last)


def chunk_sessions(
    window: Sequence[date], missing: Sequence[date], max_span: timedelta = MAX_REQUEST_SPAN
) -> list[list[date]]:
    """Group missing sessions into runs of consecutive window sessions whose 04:00-20:00 span fits one request."""
    wanted = set(missing)
    chunks: list[list[date]] = []
    current: list[date] = []
    for session in window:
        if session not in wanted:
            if current:
                chunks.append(current)
                current = []
            continue
        if current and market.extended_bounds_utc(session)[1] - market.extended_bounds_utc(current[0])[0] > max_span:
            chunks.append(current)
            current = []
        current.append(session)
    if current:
        chunks.append(current)
    return chunks


def bars_by_session(df: pd.DataFrame, sessions: Sequence[date]) -> dict[date, list[Bar]]:
    """Assign each bar to the session whose 04:00-20:00 New York window holds its start; drop the rest."""
    out: dict[date, list[Bar]] = {s: [] for s in sessions}
    if df.empty:
        return out
    absent = [c for c in [*_OHLC, "Volume"] if c not in df.columns]
    if absent:
        raise ValueError(f"price frame is missing columns {absent}")
    if not isinstance(df.index, pd.DatetimeIndex) or df.index.tz is None:
        raise ValueError("price frame needs a tz-aware DatetimeIndex of bar start times")
    df = df.dropna(subset=_OHLC)
    df = df[~df.index.duplicated(keep="first")]
    local = df.index.tz_convert(NY)
    minute = (local.hour * 60 + local.minute).tolist()
    epoch = ((df.index - _EPOCH) // pd.Timedelta(seconds=1)).tolist()
    columns = [df[c].tolist() for c in (*_OHLC, "Volume")]
    for ts, day, m, o, h, lo, c, v in zip(epoch, local.date, minute, *columns, strict=True):
        if day in out and _DAY_START_MIN <= m < _DAY_END_MIN:
            out[day].append((int(ts), float(o), float(h), float(lo), float(c), None if pd.isna(v) else float(v)))
    return out


class _Requester:
    """Strictly sequential requests, paused between each other, with one retry after a rate limit."""

    def __init__(self, fetch: Fetch, sleep: Callable[[float], None], pause_s: float) -> None:
        self._fetch = fetch
        self._sleep = sleep
        self._pause_s = pause_s
        self.requests = 0

    def get(self, yahoo_symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        if self.requests:
            self._sleep(self._pause_s)
        self.requests += 1
        try:
            return self._fetch(yahoo_symbol, start, end)
        except YFRateLimitError:
            log.warning("Yahoo rate limit on %s; retrying once in %.0f s", yahoo_symbol, RATE_LIMIT_WAIT_S)
        self._sleep(RATE_LIMIT_WAIT_S)
        self.requests += 1
        return self._fetch(yahoo_symbol, start, end)


def _fetch_symbol(
    requester: _Requester, ticker: Ticker, chunks: list[list[date]]
) -> tuple[Fetched, Literal["failed", "rate_limited"] | None]:
    """Fetch every chunk, oldest first. A failed chunk is left for the next run without holding back the newer
    ones; a persistent rate limit stops the symbol. Returns what was fetched and the failure, if any."""
    fetched: Fetched = []
    failure: Literal["failed"] | None = None
    for chunk in chunks:
        start = market.extended_bounds_utc(chunk[0])[0]
        end = market.extended_bounds_utc(chunk[-1])[1]
        try:
            df = requester.get(ticker.yahoo_symbol, start, end)
            fetched.append((chunk, bars_by_session(df, chunk)))
        except YFRateLimitError:
            log.error("%s: Yahoo still rate-limiting after a %.0f s wait", ticker.symbol, RATE_LIMIT_WAIT_S)
            return fetched, "rate_limited"
        # yfinance surfaces HTTP, network and parsing problems as many unrelated exception types
        except Exception as exc:
            if isinstance(exc, YFPricesMissingError) and exc.yahoo_reason is None:
                # With exceptions not hidden, this is how yfinance answers a range that holds no bars.
                fetched.append((chunk, {s: [] for s in chunk}))
                continue
            log.warning(
                "%s: 1m fetch for %s..%s failed: %s: %s", ticker.symbol, chunk[0], chunk[-1], type(exc).__name__, exc
            )
            failure = "failed"
    return fetched, failure


def _store(conn: sqlite3.Connection, symbol: str, fetched: Fetched, now: datetime) -> tuple[int, int]:
    bars = sessions = 0
    empty: list[str] = []
    with conn:
        for chunk, by_session in fetched:
            for session in chunk:
                rows = by_session[session]
                if not rows:
                    empty.append(session.isoformat())
                bars += db.insert_bars_1m(conn, symbol, rows)
                db.mark_session(conn, symbol, session.isoformat(), len(rows), now)
                sessions += 1
    if empty:
        log.warning("%s: Yahoo has no 1m bars for %s; they will be retried next run", symbol, ", ".join(empty))
    return bars, sessions


def snapshot_1m(
    conn: sqlite3.Connection,
    watchlist: Watchlist,
    run_id: int,
    now: datetime,
    fetch: Fetch | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> dict:
    """Store 1-minute bars for every configured ticker and every completed session in the backfill window
    that has no bars yet. Sessions still in progress (before 20:00 New York) are never fetched or marked."""
    window = snapshot_window(now, watchlist.prices.backfill_days)
    plan: list[tuple[Ticker, list[list[date]]]] = []
    for ticker in watchlist.tickers:
        have = db.sessions_with_bars(conn, ticker.symbol)
        missing = [s for s in window if s.isoformat() not in have]
        if missing:
            plan.append((ticker, chunk_sessions(window, missing)))
    if window:
        log.info(
            "1m snapshot (run %d): sessions %s..%s, %d of %d symbols need bars",
            run_id,
            window[0],
            window[-1],
            len(plan),
            len(watchlist.tickers),
        )
    else:
        log.info("1m snapshot (run %d): no completed session in the window", run_id)

    requester = _Requester(fetch or fetch_yahoo_1m, sleep, watchlist.prices.request_pause_s)
    bars = sessions = 0
    failed: list[str] = []
    for i, (ticker, chunks) in enumerate(plan):
        fetched, failure = _fetch_symbol(requester, ticker, chunks)
        stored_bars, stored_sessions = _store(conn, ticker.symbol, fetched, now)
        bars += stored_bars
        sessions += stored_sessions
        if failure:
            failed.append(ticker.symbol)
        if failure == "rate_limited":
            # Hammering on would only extend the block; the missing sessions are picked up next run.
            rest = [t.symbol for t, _ in plan[i + 1 :]]
            if rest:
                log.error("1m snapshot stopped early; %d symbols left for the next run", len(rest))
            failed.extend(rest)
            break

    if not failed:
        status = "ok"
    else:
        status = "partial" if sessions else "error"
    counts = {
        "status": status,
        "symbols": len(watchlist.tickers),
        "requests": requester.requests,
        "bars": bars,
        "sessions_marked": sessions,
        "symbols_failed": failed,
    }
    log.info("1m snapshot (run %d): %s", run_id, counts)
    return counts
