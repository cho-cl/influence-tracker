from __future__ import annotations

import bisect
import logging
import math
import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Literal

from . import db, market, prices
from .config import Watchlist
from .timeutil import NY, from_iso, to_iso

log = logging.getLogger(__name__)

MARKET = "SPY"
DAILY_LOOKBACK_SESSIONS = 150
# An event completes once daily bars reach d0+5 (the end of the M3 post-event window).
DAILY_LOOKAHEAD_SESSIONS = 5
SPLIT_WINDOW_SESSIONS = 5
BAR = timedelta(seconds=60)
PRE_WINDOW = timedelta(minutes=60)
POST_WINDOWS = {"p5": 5, "p15": 15, "p30": 30, "p60": 60}

IntradayOutcome = Literal["pending", "ok", "unavailable"]


# ---------------------------------------------------------------- prices from 1-minute bars


@dataclass(frozen=True)
class PricePoint:
    ts: int  # start (epoch s) of the bar whose close this is
    price: float


class MinuteBars:
    """One symbol's 1-minute bars as (bar start epoch seconds, close), sorted by start."""

    def __init__(self, starts: list[int], closes: list[float]) -> None:
        if any(b <= a for a, b in zip(starts, starts[1:], strict=False)):
            raise ValueError("bar starts must be strictly increasing")
        self.starts = starts
        self.closes = closes

    @classmethod
    def load(cls, conn: sqlite3.Connection, symbol: str, start: datetime, end: datetime) -> MinuteBars:
        """Bars whose start lies in [start, end)."""
        rows = conn.execute(
            """SELECT ts, close FROM bars_1m WHERE symbol = ? AND ts >= ? AND ts < ? AND close > 0
               ORDER BY ts""",
            (symbol, int(start.timestamp()), int(end.timestamp())),
        ).fetchall()
        return cls([r[0] for r in rows], [r[1] for r in rows])

    def _point(self, i: int) -> PricePoint | None:
        return PricePoint(self.starts[i], self.closes[i]) if i >= 0 else None

    def at(self, t: datetime) -> PricePoint | None:
        """The price as of instant t: the close of the last bar that had finished by t (start <= t - 60 s).
        Minutes without trades have no bar, so this is the last bar at or before, never the bar at exactly."""
        cutoff = math.floor(t.timestamp()) - int(BAR.total_seconds())
        return self._point(bisect.bisect_right(self.starts, cutoff) - 1)

    def regular_close(self, session: date) -> PricePoint | None:
        """Close of the session's last regular-hours bar: the last bar starting in [open, close)."""
        open_, close = market.session_bounds_utc(session)
        i = bisect.bisect_left(self.starts, int(close.timestamp())) - 1
        if i < 0 or self.starts[i] < int(open_.timestamp()):
            return None
        return self._point(i)


@dataclass(frozen=True)
class Anchor:
    """Where a window starts or ends: the price as of an instant, or a session's regular close."""

    at: datetime | None = None
    close_of: date | None = None

    def resolve(self, bars: MinuteBars) -> PricePoint | None:
        if self.close_of is not None:
            return bars.regular_close(self.close_of)
        assert self.at is not None
        return bars.at(self.at)


@dataclass(frozen=True)
class WindowSpec:
    win: str
    start: Anchor
    end: Anchor
    truncated: bool


@dataclass(frozen=True)
class WindowRow:
    win: str
    start_ts: int
    end_ts: int
    start_price: float
    end_price: float
    ret: float
    spy_start_price: float | None
    spy_end_price: float | None
    spy_ret: float | None
    truncated: bool


def window_specs(t0: datetime, d0: date) -> list[WindowSpec]:
    """Day-0 legs for every event; the -60..0 and +5..+60 minute windows for regular-session posts, clipped at
    d0's open and (early) close."""
    open_, close = market.session_bounds_utc(d0)
    specs = [
        WindowSpec("pre_leg", Anchor(close_of=market.previous_session(d0)), Anchor(at=t0), False),
        WindowSpec("post_leg", Anchor(at=t0), Anchor(close_of=d0), False),
    ]
    if open_ <= t0 < close:
        pre_start = t0 - PRE_WINDOW
        specs.append(WindowSpec("pre60", Anchor(at=max(pre_start, open_)), Anchor(at=t0), pre_start < open_))
        for win, minutes in POST_WINDOWS.items():
            end = t0 + timedelta(minutes=minutes)
            specs.append(WindowSpec(win, Anchor(at=t0), Anchor(at=min(end, close)), end > close))
    return specs


def spans_open(t0: datetime, d0: date) -> bool:
    """A regular-session post in the first minute: its reference is the last pre-market bar, so its windows
    include the opening auction."""
    open_, close = market.session_bounds_utc(d0)
    return open_ <= t0 < min(open_ + BAR, close)


def compute_windows(
    t0: datetime, d0: date, bars: MinuteBars, spy: MinuteBars
) -> tuple[PricePoint | None, list[WindowRow], list[str]]:
    """(reference price, windows, names of windows skipped because a ticker price was unavailable). SPY is priced
    at the same instants as the ticker; a missing SPY price leaves the spy_* fields empty."""
    ref = bars.at(t0)
    rows: list[WindowRow] = []
    skipped: list[str] = []
    for spec in window_specs(t0, d0):
        start, end = spec.start.resolve(bars), spec.end.resolve(bars)
        if start is None or end is None:
            skipped.append(spec.win)
            continue
        m_start, m_end = spec.start.resolve(spy), spec.end.resolve(spy)
        both = m_start is not None and m_end is not None
        rows.append(
            WindowRow(
                win=spec.win,
                start_ts=start.ts,
                end_ts=end.ts,
                start_price=start.price,
                end_price=end.price,
                ret=end.price / start.price - 1,
                spy_start_price=m_start.price if both else None,
                spy_end_price=m_end.price if both else None,
                spy_ret=m_end.price / m_start.price - 1 if both else None,
                truncated=spec.truncated,
            )
        )
    return ref, rows, skipped


# ---------------------------------------------------------------- event rows


def sync_events(conn: sqlite3.Connection, watchlist: Watchlist, now: datetime) -> dict:
    """Make the events table mirror the mentions of configured non-benchmark tickers: one row per
    (platform, native_id, ticker). Events whose mention is gone (or whose ticker left the universe) are deleted with
    their windows."""
    tickers = {t.symbol for t in watchlist.event_tickers}
    created = skipped = 0
    with conn:
        stale = [
            (r["id"],)
            for r in conn.execute(
                """SELECT e.id, e.ticker FROM events e
                   WHERE NOT EXISTS (SELECT 1 FROM mentions m WHERE m.platform = e.platform
                                       AND m.native_id = e.native_id AND m.ticker = e.ticker)"""
            ).fetchall()
        ]
        stale += [(r["id"],) for r in conn.execute("SELECT id, ticker FROM events") if r["ticker"] not in tickers]
        stale = sorted(set(stale))
        conn.executemany("DELETE FROM event_windows WHERE event_id = ?", stale)
        conn.executemany("DELETE FROM events WHERE id = ?", stale)

        new = conn.execute(
            """SELECT DISTINCT m.platform, m.native_id, m.ticker, p.created_at_utc
               FROM mentions m
               JOIN posts p ON p.platform = m.platform AND p.native_id = m.native_id
               LEFT JOIN events e ON e.platform = m.platform AND e.native_id = m.native_id AND e.ticker = m.ticker
               WHERE e.id IS NULL
               ORDER BY p.created_at_utc, m.platform, m.native_id, m.ticker"""
        ).fetchall()
        for r in new:
            if r["ticker"] not in tickers:
                continue
            try:
                t0 = from_iso(r["created_at_utc"])
                d0 = market.event_session(t0)
                phase = market.session_phase(t0)
                first_minute = spans_open(t0, d0)
            except ValueError as exc:
                log.error("%s/%s %s: cannot place t0 %s on the calendar: %s", r[0], r[1], r[2], r[3], exc)
                skipped += 1
                continue
            conn.execute(
                """INSERT INTO events (platform, native_id, ticker, t0, d0, session_phase, status, spans_open,
                                       created_at) VALUES (?, ?, ?, ?, ?, ?, 'pending', ?, ?)""",
                (r["platform"], r["native_id"], r["ticker"], to_iso(t0), d0.isoformat(), phase, int(first_minute),
                 to_iso(now)),
            )  # fmt: skip
            created += 1
    total = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    if stale or created:
        log.info("events: %d created, %d deleted, %d total", created, len(stale), total)
    return {
        "status": "partial" if skipped else "ok",
        "events": total,
        "created": created,
        "deleted": len(stale),
        "skipped": skipped,
    }


def recompute_clustered(conn: sqlite3.Connection) -> int:
    """Within each (ticker, d0) the earliest post (then lowest id) is primary; every later one is clustered.
    Recomputed for all events, because a late-collected earlier post changes which one is first."""
    rows = conn.execute("SELECT id, ticker, d0, clustered FROM events ORDER BY ticker, d0, t0, id").fetchall()
    changes: list[tuple[int, int]] = []
    previous: tuple[str, str] | None = None
    for r in rows:
        key = (r["ticker"], r["d0"])
        value = int(key == previous)
        previous = key
        if r["clustered"] != value:
            changes.append((value, r["id"]))
    with conn:
        conn.executemany("UPDATE events SET clustered = ? WHERE id = ?", changes)
    return len(changes)


# ---------------------------------------------------------------- completion


def earnings_session(at: datetime) -> date:
    """The session an earnings announcement is first traded in: the event_session rule, except that a 3 PM
    standard-time listing counts as after the close."""
    local = at.astimezone(NY)
    # Yahoo can work out a scheduled winter 4 PM report at the summer offset and list it as 3 PM EST.
    if local.hour == 15 and not local.dst():
        at += timedelta(hours=1)
    return market.event_session(at)


class _Completer:
    """Completes pending events, caching per-symbol lookups for one enrich run."""

    def __init__(self, conn: sqlite3.Connection, watchlist: Watchlist, now: datetime) -> None:
        self.conn = conn
        self.now = now
        self.window = set(prices.snapshot_window(now, watchlist.prices.backfill_days))
        self.last_completed = market.last_completed_session(now)
        self._minute_sessions: dict[str, set[str]] = {}
        self._earnings: dict[str, list[date]] = {}
        self._bars: dict[tuple[str, date], MinuteBars] = {}

    def fetchable(self, session: date) -> bool:
        """The 1m snapshot can still store this session: inside its window, or not finished yet."""
        return session in self.window or self.last_completed is None or session > self.last_completed

    def has_minute_session(self, symbol: str, session: date) -> bool:
        if symbol not in self._minute_sessions:
            self._minute_sessions[symbol] = db.sessions_with_bars(self.conn, symbol)
        return session.isoformat() in self._minute_sessions[symbol]

    def has_daily(self, symbol: str, session: date) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM bars_1d WHERE symbol = ? AND session_date = ?", (symbol, session.isoformat())
        ).fetchone()
        return row is not None

    def bars(self, symbol: str, d0: date) -> MinuteBars:
        """The symbol's bars from 04:00 New York on the session before d0 to 20:00 on d0."""
        if (symbol, d0) not in self._bars:
            start = market.extended_bounds_utc(market.previous_session(d0))[0]
            end = market.extended_bounds_utc(d0)[1]
            self._bars[(symbol, d0)] = MinuteBars.load(self.conn, symbol, start, end)
        return self._bars[(symbol, d0)]

    def earnings_sessions(self, symbol: str) -> list[date]:
        """Sessions each stored earnings announcement is first traded in (see earnings_session)."""
        if symbol not in self._earnings:
            sessions: list[date] = []
            for (stamp,) in self.conn.execute("SELECT earnings_at FROM earnings WHERE symbol = ?", (symbol,)):
                try:
                    sessions.append(earnings_session(from_iso(stamp)))
                except ValueError:
                    log.debug("%s: earnings %s is outside the calendar", symbol, stamp)
            self._earnings[symbol] = sorted(sessions)
        return self._earnings[symbol]

    def earnings_flag(self, symbol: str, d0: date) -> int | None:
        """1 if an earnings announcement is first traded within one session of d0; 0 if the latest fetch
        succeeded, none matches and the stored dates reach back before the window; otherwise unknown (None)."""
        lo, hi = market.previous_session(d0), market.session_offset(d0, 1)
        sessions = self.earnings_sessions(symbol)
        if any(lo <= s <= hi for s in sessions):
            return 1
        fetch = self.conn.execute("SELECT ok FROM earnings_fetch WHERE symbol = ?", (symbol,)).fetchone()
        if fetch is None or not fetch["ok"]:
            return None
        # A list that starts after the window says nothing about the quarters before it.
        if not sessions or sessions[0] > lo:
            return None
        return 0

    def split_flag(self, symbol: str, d0: date) -> int:
        lo = market.session_offset(d0, -SPLIT_WINDOW_SESSIONS)
        hi = market.session_offset(d0, SPLIT_WINDOW_SESSIONS)
        row = self.conn.execute(
            """SELECT 1 FROM bars_1d WHERE symbol = ? AND session_date BETWEEN ? AND ? AND split_ratio != 0
               LIMIT 1""",
            (symbol, lo.isoformat(), hi.isoformat()),
        ).fetchone()
        return int(row is not None)

    def intraday(self, symbol: str, d0: date) -> tuple[IntradayOutcome, list[str]]:
        prev = market.previous_session(d0)
        missing = [
            (sym, s) for sym in (symbol, MARKET) for s in (prev, d0) if not self.has_minute_session(sym, s)
        ]  # fmt: skip
        labels = [f"{sym} {s}" for sym, s in missing]
        if any(not self.fetchable(s) for _, s in missing):
            return "unavailable", labels
        return ("pending" if missing else "ok"), labels

    def try_complete(self, event: sqlite3.Row) -> IntradayOutcome:
        """Returns 'pending' if the event must wait, else the intraday state it was completed with."""
        ticker = event["ticker"]
        t0 = from_iso(event["t0"])
        d0 = date.fromisoformat(event["d0"])
        target = market.session_offset(d0, DAILY_LOOKAHEAD_SESSIONS)
        if not (self.has_daily(ticker, target) and self.has_daily(MARKET, target)):
            return "pending"
        state, missing = self.intraday(ticker, d0)
        if state == "pending":
            return "pending"

        ref: PricePoint | None = None
        rows: list[WindowRow] = []
        if state == "ok":
            ref, rows, skipped = compute_windows(t0, d0, self.bars(ticker, d0), self.bars(MARKET, d0))
            if ref is None:
                log.warning("event %d (%s, t0 %s): no %s bar before t0", event["id"], ticker, event["t0"], ticker)
            if skipped:
                log.warning(
                    "event %d (%s, t0 %s): skipped windows %s (no %s price)",
                    event["id"], ticker, event["t0"], skipped, ticker,
                )  # fmt: skip
            spy_missing = [r.win for r in rows if r.spy_ret is None]
            if spy_missing:
                log.warning("event %d (%s): no %s price for windows %s", event["id"], ticker, MARKET, spy_missing)
        else:
            log.info(
                "event %d (%s, d0 %s): 1m bars %s are no longer fetchable; intraday unavailable",
                event["id"], ticker, d0, ", ".join(missing),
            )  # fmt: skip

        earnings = self.earnings_flag(ticker, d0)
        split = self.split_flag(ticker, d0)
        with self.conn:
            self.conn.execute(
                """UPDATE events SET status = 'complete', intraday_state = ?, ref_ts = ?, ref_price = ?,
                                     earnings_flag = ?, split_flag = ?, completed_at = ?
                   WHERE id = ?""",
                (
                    state,
                    ref.ts if ref else None,
                    ref.price if ref else None,
                    earnings,
                    split,
                    to_iso(self.now),
                    event["id"],
                ),
            )
            self.conn.execute("DELETE FROM event_windows WHERE event_id = ?", (event["id"],))
            self.conn.executemany(
                """INSERT INTO event_windows (event_id, win, start_ts, end_ts, start_price, end_price, ret,
                                              spy_start_price, spy_end_price, spy_ret, truncated)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (event["id"], r.win, r.start_ts, r.end_ts, r.start_price, r.end_price, r.ret,
                     r.spy_start_price, r.spy_end_price, r.spy_ret, int(r.truncated))
                    for r in rows
                ],
            )  # fmt: skip
        return state


def complete_pending(conn: sqlite3.Connection, watchlist: Watchlist, now: datetime) -> dict:
    """Try to complete every pending event; see enrich_events."""
    completer = _Completer(conn, watchlist, now)
    completed = unavailable = 0
    failed: list[int] = []
    for event in conn.execute("SELECT * FROM events WHERE status = 'pending' ORDER BY d0, t0, id").fetchall():
        try:
            outcome = completer.try_complete(event)
        # One malformed event must not hold back the others; it stays pending and is retried next run.
        except Exception:
            log.exception("event %d (%s, t0 %s): completion failed", event["id"], event["ticker"], event["t0"])
            failed.append(event["id"])
            continue
        if outcome != "pending":
            completed += 1
            unavailable += int(outcome == "unavailable")
    return {"completed": completed, "intraday_unavailable": unavailable, "failed_events": failed}


def _no_refresh() -> dict:
    return {"status": "ok", "symbols": 0, "requests": 0, "rows": 0, "symbols_failed": []}


def enrich_events(
    conn: sqlite3.Connection,
    watchlist: Watchlist,
    run_id: int,
    now: datetime,
    fetch_daily: prices.DailyFetch | None = None,
    fetch_earnings: prices.EarningsFetch | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> dict:
    """Sync events with mentions, recompute clustering, refresh daily bars (every event ticker plus SPY, from 150
    sessions before the earliest d0) and earnings dates (tickers with pending events), then complete every pending
    event whose data is ready."""
    synced = sync_events(conn, watchlist, now)
    recompute_clustered(conn)

    tickers = [r[0] for r in conn.execute("SELECT DISTINCT ticker FROM events ORDER BY ticker")]
    if tickers:
        first_d0 = date.fromisoformat(conn.execute("SELECT MIN(d0) FROM events").fetchone()[0])
        start = market.session_offset(first_d0, -DAILY_LOOKBACK_SESSIONS)
        symbols = [t for t in tickers if t != MARKET] + [MARKET]
        daily = prices.refresh_daily_bars(conn, watchlist, symbols, start, run_id, now, fetch=fetch_daily, sleep=sleep)
    else:
        daily = _no_refresh()

    pending_tickers = [r[0] for r in conn.execute("SELECT DISTINCT ticker FROM events WHERE status = 'pending'")]
    if daily["requests"] and any(prices.earnings_due(conn, t, now) for t in pending_tickers):
        sleep(watchlist.prices.request_pause_s)
    earnings = prices.refresh_earnings(conn, watchlist, sorted(pending_tickers), now, fetch=fetch_earnings, sleep=sleep)

    done = complete_pending(conn, watchlist, now)
    pending = conn.execute("SELECT COUNT(*) FROM events WHERE status = 'pending'").fetchone()[0]

    if daily["status"] == "error":
        status = "error"
    elif daily["status"] != "ok" or earnings["status"] != "ok" or synced["status"] != "ok" or done["failed_events"]:
        status = "partial"
    else:
        status = "ok"
    counts = {
        "status": status,
        "events": synced["events"],
        "created": synced["created"],
        "deleted": synced["deleted"],
        "pending": pending,
        "completed": done["completed"],
        "intraday_unavailable": done["intraday_unavailable"],
        "failed_events": done["failed_events"],
        "daily": daily,
        "earnings": earnings,
    }
    log.info("enrich (run %d): %s", run_id, counts)
    return counts
