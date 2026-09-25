from __future__ import annotations

import logging
import sqlite3
import time
from collections.abc import Callable, Sequence
from datetime import UTC, date, datetime, timedelta
from typing import Any, Literal

import pandas as pd
import yfinance as yf
from yfinance.exceptions import YFPricesMissingError, YFRateLimitError

from . import db, market
from .config import Ticker, Watchlist
from .timeutil import NY, from_iso, to_iso

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

    def __init__(self, fetch: Callable[..., Any], sleep: Callable[[float], None], pause_s: float) -> None:
        self._fetch = fetch
        self._sleep = sleep
        self._pause_s = pause_s
        self.requests = 0

    def get(self, yahoo_symbol: str, *args: Any) -> Any:
        if self.requests:
            self._sleep(self._pause_s)
        self.requests += 1
        try:
            return self._fetch(yahoo_symbol, *args)
        except YFRateLimitError:
            log.warning("Yahoo rate limit on %s; retrying once in %.0f s", yahoo_symbol, RATE_LIMIT_WAIT_S)
        self._sleep(RATE_LIMIT_WAIT_S)
        self.requests += 1
        return self._fetch(yahoo_symbol, *args)


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


# ---------------------------------------------------------------- daily bars

DailyFetch = Callable[[str, date, date], pd.DataFrame]
EarningsFetch = Callable[[str], list[datetime]]
DailyRow = tuple[str, float | None, float | None, float | None, float, float | None, float | None, float]

_DAILY_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close", "Volume", "Stock Splits"]


def fetch_yahoo_daily(yahoo_symbol: str, start: date, end: date) -> pd.DataFrame:
    """Yahoo daily bars for start..end (both inclusive): split-adjusted OHLC, Adj Close (also dividend-adjusted),
    Volume, Dividends and Stock Splits, indexed by New York midnight of each date."""
    yf.config.network.retries = YF_NETWORK_RETRIES
    yf.config.debug.hide_exceptions = False
    return yf.Ticker(yahoo_symbol).history(
        start=start, end=end + timedelta(days=1), interval="1d", auto_adjust=False, actions=True
    )


def _num(value: Any) -> float | None:
    return None if value is None or pd.isna(value) else float(value)


def daily_rows(df: pd.DataFrame, through: date) -> list[DailyRow]:
    """(session_date, open, high, low, close, adj_close, volume, split_ratio) for each XNYS session up to and
    including `through`. Rows without a close, non-session dates and repeated dates are dropped."""
    if df.empty:
        return []
    absent = [c for c in _DAILY_COLUMNS if c not in df.columns]
    if absent:
        raise ValueError(f"daily frame is missing columns {absent}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("daily frame needs a DatetimeIndex")
    rows: list[DailyRow] = []
    seen: set[date] = set()
    skipped: list[str] = []
    columns = [df[col].tolist() for col in _DAILY_COLUMNS]
    # Daily bars are labelled by exchange date at local midnight; converting zones could shift that date.
    for day, o, h, lo, c, adj, v, split in zip(df.index.date, *columns, strict=True):
        if day in seen or day > through:
            continue
        seen.add(day)
        split_ratio = _num(split) or 0.0
        if _num(c) is None or not market.is_session(day):
            skipped.append(day.isoformat())
            if split_ratio:
                log.warning("daily row %s carries a %s split but no session close; the split is lost", day, split)
            continue
        rows.append((day.isoformat(), _num(o), _num(h), _num(lo), float(c), _num(adj), _num(v), split_ratio))
    if skipped:
        log.info("dropped %d daily row(s) without a close or off the XNYS calendar: %s", len(skipped), skipped[:5])
    return rows


def _yahoo_symbol(watchlist: Watchlist, symbol: str) -> str:
    ticker = watchlist.ticker(symbol)
    return ticker.yahoo_symbol if ticker is not None and ticker.symbol == symbol else symbol.replace(".", "-")


def _status(failed: int, total: int) -> str:
    if not failed:
        return "ok"
    return "partial" if failed < total else "error"


def refresh_daily_bars(
    conn: sqlite3.Connection,
    watchlist: Watchlist,
    symbols: Sequence[str],
    start: date,
    run_id: int,
    now: datetime,
    fetch: DailyFetch | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> dict:
    """Download each symbol's daily bars from `start` through today and replace its stored rows whole (never
    merged: a split or dividend re-adjusts the entire history). Only sessions whose extended hours have ended are
    kept, so a partial day is never stored. A symbol that fails keeps its previous rows."""
    symbols = list(dict.fromkeys(symbols))
    through = market.last_completed_session(now)
    today = now.astimezone(NY).date()
    requester = _Requester(fetch or fetch_yahoo_daily, sleep, watchlist.prices.request_pause_s)
    stored = 0
    failed: list[str] = []
    for i, symbol in enumerate(symbols):
        try:
            df = requester.get(_yahoo_symbol(watchlist, symbol), start, today)
            rows = daily_rows(df, through) if through is not None else []
            if not rows:
                raise LookupError(f"no completed daily bars from {start}")
        except YFRateLimitError:
            log.error("daily bars: Yahoo still rate-limiting after a %.0f s wait; stopping", RATE_LIMIT_WAIT_S)
            failed.extend(symbols[i:])
            break
        # yfinance surfaces HTTP, network and parsing problems as many unrelated exception types
        except Exception as exc:
            log.warning("%s: daily bars failed, keeping the stored rows: %s: %s", symbol, type(exc).__name__, exc)
            failed.append(symbol)
            continue
        fetched_at = to_iso(now)
        with conn:
            conn.execute("DELETE FROM bars_1d WHERE symbol = ?", (symbol,))
            conn.executemany(
                """INSERT INTO bars_1d (symbol, session_date, open, high, low, close, adj_close, volume, split_ratio,
                                        fetched_at) VALUES (?,?,?,?,?,?,?,?,?,?)""",
                [(symbol, *row, fetched_at) for row in rows],
            )
        stored += len(rows)

    counts = {
        "status": _status(len(failed), len(symbols)),
        "symbols": len(symbols),
        "requests": requester.requests,
        "rows": stored,
        "symbols_failed": failed,
    }
    log.info("daily bars (run %d) from %s through %s: %s", run_id, start, through, counts)
    return counts


# ---------------------------------------------------------------- earnings dates

# yfinance 1.7.0 scrapes Yahoo's earnings calendar page, which returns 25 rows for any limit up to 25: the next
# scheduled date(s) and about five years of history.
EARNINGS_LIMIT = 25
EARNINGS_MAX_AGE = timedelta(days=7)
EARNINGS_RETRY_AFTER = timedelta(days=1)
# The scheduled run drifts by minutes from day to day; without slack a weekly refetch could slip a whole day.
EARNINGS_SLACK = timedelta(hours=1)


def fetch_yahoo_earnings(yahoo_symbol: str) -> list[datetime]:
    """Past and scheduled earnings announcement times from Yahoo's earnings calendar, as aware UTC datetimes."""
    yf.config.network.retries = YF_NETWORK_RETRIES
    yf.config.debug.hide_exceptions = False
    df = yf.Ticker(yahoo_symbol).get_earnings_dates(limit=EARNINGS_LIMIT)
    if df is None:
        # yfinance only logs (it does not raise) when the page has no earnings table.
        raise LookupError(f"Yahoo has no earnings table for {yahoo_symbol}")
    out: set[datetime] = set()
    for ts in df.index:
        if pd.isna(ts):
            continue
        if not isinstance(ts, pd.Timestamp) or ts.tzinfo is None:
            raise ValueError(f"earnings date {ts!r} for {yahoo_symbol} is not a tz-aware timestamp")
        out.add(ts.to_pydatetime().astimezone(UTC))
    return sorted(out)


def earnings_due(conn: sqlite3.Connection, symbol: str, now: datetime) -> bool:
    """Never fetched, last fetched over 7 days ago, or last fetch failed over a day ago."""
    row = conn.execute("SELECT fetched_at, ok FROM earnings_fetch WHERE symbol = ?", (symbol,)).fetchone()
    if row is None:
        return True
    max_age = EARNINGS_MAX_AGE if row["ok"] else EARNINGS_RETRY_AFTER
    return now - from_iso(row["fetched_at"]) > max_age - EARNINGS_SLACK


def _record_earnings_fetch(conn: sqlite3.Connection, symbol: str, now: datetime, error: str | None) -> None:
    conn.execute(
        """INSERT INTO earnings_fetch (symbol, fetched_at, ok, error) VALUES (?, ?, ?, ?)
           ON CONFLICT(symbol) DO UPDATE SET fetched_at = excluded.fetched_at, ok = excluded.ok,
             error = excluded.error""",
        (symbol, to_iso(now), int(error is None), error),
    )


def refresh_earnings(
    conn: sqlite3.Connection,
    watchlist: Watchlist,
    symbols: Sequence[str],
    now: datetime,
    fetch: EarningsFetch | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> dict:
    """Refetch earnings dates for the symbols that are due (see earnings_due). A success replaces the symbol's
    stored dates; a failure is recorded in earnings_fetch and keeps the old dates."""
    symbols = list(dict.fromkeys(symbols))
    due = [s for s in symbols if earnings_due(conn, s, now)]
    requester = _Requester(fetch or fetch_yahoo_earnings, sleep, watchlist.prices.request_pause_s)
    failed: list[str] = []
    for i, symbol in enumerate(due):
        try:
            stamps = sorted({to_iso(t) for t in requester.get(_yahoo_symbol(watchlist, symbol))})
        except YFRateLimitError:
            log.error("earnings: Yahoo still rate-limiting after a %.0f s wait; stopping", RATE_LIMIT_WAIT_S)
            with conn:
                _record_earnings_fetch(conn, symbol, now, "YFRateLimitError: still rate-limited after one retry")
            failed.extend(due[i:])
            break
        # yfinance surfaces HTTP, network and parsing problems as many unrelated exception types
        except Exception as exc:
            log.warning("%s: earnings dates failed, keeping the stored ones: %s: %s", symbol, type(exc).__name__, exc)
            with conn:
                _record_earnings_fetch(conn, symbol, now, f"{type(exc).__name__}: {exc}")
            failed.append(symbol)
            continue
        if not stamps:
            log.warning("%s: Yahoo lists no earnings dates", symbol)
        with conn:
            conn.execute("DELETE FROM earnings WHERE symbol = ?", (symbol,))
            conn.executemany("INSERT INTO earnings (symbol, earnings_at) VALUES (?, ?)", [(symbol, s) for s in stamps])
            _record_earnings_fetch(conn, symbol, now, None)

    counts = {
        "status": _status(len(failed), len(due)),
        "symbols": len(symbols),
        "due": len(due),
        "requests": requester.requests,
        "failed": failed,
    }
    log.info("earnings dates: %s", counts)
    return counts
