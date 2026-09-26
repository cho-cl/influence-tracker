from __future__ import annotations

import logging
import sqlite3
from collections.abc import Iterable
from datetime import UTC, date, datetime, time, timedelta

import pandas as pd
import pytest

from influence_tracker import db, events, market, prices
from influence_tracker.config import Watchlist
from influence_tracker.models import Mention, Post
from influence_tracker.timeutil import NY, to_iso

T = "NVDA"
SPY = "SPY"
BASE = int(datetime(2026, 1, 1, tzinfo=UTC).timestamp())
# Default "now" for events on Wed 2026-11-04: d0+5 is Wed 11-11, completed by 20:00 New York on 11-12.
NOW = datetime(2026, 11, 12, 21, 0, tzinfo=NY)
# An earnings date far from every test event, so fetched lists reach back past each event's window.
OLD_EARNINGS = datetime(2025, 1, 15, 21, 5, tzinfo=UTC)


def ny(y: int, m: int, d: int, hh: int, mm: int = 0, ss: int = 0) -> datetime:
    return datetime(y, m, d, hh, mm, ss, tzinfo=NY)


def ep(dt: datetime) -> int:
    return int(dt.timestamp())


def px(symbol: str, ts: int) -> float:
    """Synthetic close for the bar starting at ts: strictly increasing, so a price identifies its bar."""
    if symbol == SPY:
        return 500.0 + (ts - BASE) / 60 * 0.002
    return 100.0 + (ts - BASE) / 60 * 0.01


class Sleeps(list):
    def __call__(self, seconds: float) -> None:
        self.append(seconds)


def add_minutes(
    conn: sqlite3.Connection,
    symbol: str,
    session: date,
    skip: Iterable[datetime] = (),
    first: time = time(4, 0),
    last: time = time(19, 59),
) -> int:
    """Store 1m bars for every minute first..last New York on `session` except `skip`, and mark the session."""
    skipped = {ep(t) for t in skip}
    start = datetime.combine(session, first, tzinfo=NY)
    end = datetime.combine(session, last, tzinfo=NY)
    rows = []
    t = start
    while t <= end:
        ts = ep(t)
        if ts not in skipped:
            p = px(symbol, ts)
            rows.append((ts, p, p, p, p, 1000.0))
        t += timedelta(minutes=1)
    with conn:
        db.insert_bars_1m(conn, symbol, rows)
        db.mark_session(conn, symbol, session.isoformat(), len(rows), NOW)
    return len(rows)


def add_both(conn: sqlite3.Connection, *sessions: date) -> None:
    for s in sessions:
        add_minutes(conn, T, s)
        add_minutes(conn, SPY, s)


def add_post(
    conn: sqlite3.Connection, native_id: str, created: datetime, tickers: Iterable[str] = (T,), platform: str = "x"
) -> None:
    post = Post(
        platform=platform,
        native_id=native_id,
        author="someone",
        created_at_utc=created.astimezone(UTC),
        text=f"post {native_id}",
        url=f"https://example.com/{native_id}",
    )
    with conn:
        db.upsert_post(conn, post, NOW)
        mentions = [Mention(t, "cashtag", f"${t}") for t in tickers]
        db.replace_mentions(conn, platform, native_id, mentions, [], NOW)


def daily_frame(sessions: list[date], splits: dict[date, float] | None = None) -> pd.DataFrame:
    """Shaped like yfinance 1.7.0 history(interval='1d', auto_adjust=False, actions=True)."""
    splits = splits or {}
    index = pd.DatetimeIndex(pd.to_datetime([s.isoformat() for s in sessions])).tz_localize("America/New_York")
    close = [200.0 + i for i in range(len(sessions))]
    df = pd.DataFrame(
        {
            "Open": close,
            "High": [c + 1 for c in close],
            "Low": [c - 1 for c in close],
            "Close": close,
            "Adj Close": [c * 0.99 for c in close],
            "Volume": [1_000_000 + i for i in range(len(sessions))],
            "Dividends": [0.0] * len(sessions),
            "Stock Splits": [splits.get(s, 0.0) for s in sessions],
        },
        index=index,
    )
    df.index.name = "Date"
    return df


class FakeDaily:
    """Every XNYS session from start to end, including end's still-running session, like Yahoo."""

    def __init__(self, splits: dict[str, dict[date, float]] | None = None) -> None:
        self.splits = splits or {}
        self.fail: dict[str, BaseException] = {}
        self.calls: list[tuple[str, date, date]] = []

    def __call__(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        self.calls.append((symbol, start, end))
        if symbol in self.fail:
            raise self.fail[symbol]
        return daily_frame(market.sessions_in_range(start, end), self.splits.get(symbol))


class FakeEarnings:
    def __init__(self, dates: dict[str, list[datetime] | BaseException] | None = None) -> None:
        self.dates = dates or {}
        self.calls: list[str] = []

    def __call__(self, symbol: str) -> list[datetime]:
        self.calls.append(symbol)
        value = self.dates.get(symbol, [OLD_EARNINGS])
        if isinstance(value, BaseException):
            raise value
        return list(value)


class Env:
    def __init__(self, conn: sqlite3.Connection, watchlist: Watchlist) -> None:
        self.conn = conn
        self.wl = watchlist
        self.daily = FakeDaily()
        self.earnings = FakeEarnings()
        self.sleeps = Sleeps()

    def enrich(self, now: datetime = NOW) -> dict:
        return events.enrich_events(
            self.conn, self.wl, 1, now, fetch_daily=self.daily, fetch_earnings=self.earnings, sleep=self.sleeps
        )


@pytest.fixture
def env(conn, watchlist) -> Env:
    return Env(conn, watchlist)


def event(conn: sqlite3.Connection, native_id: str, ticker: str = T) -> sqlite3.Row:
    row = conn.execute("SELECT * FROM events WHERE native_id = ? AND ticker = ?", (native_id, ticker)).fetchone()
    assert row is not None, (native_id, ticker)
    return row


def windows(conn: sqlite3.Connection, event_id: int) -> dict[str, sqlite3.Row]:
    rows = conn.execute("SELECT * FROM event_windows WHERE event_id = ?", (event_id,)).fetchall()
    return {r["win"]: r for r in rows}


def assert_window(
    w: sqlite3.Row,
    start: datetime,
    end: datetime,
    truncated: bool = False,
    spy: tuple[datetime, datetime] | None = None,
) -> None:
    """Check a stored window against the bar starts its prices must come from; SPY defaults to the same bars."""
    assert (w["start_ts"], w["end_ts"]) == (ep(start), ep(end)), (
        datetime.fromtimestamp(w["start_ts"], NY),
        datetime.fromtimestamp(w["end_ts"], NY),
    )
    assert w["start_price"] == px(T, ep(start)) and w["end_price"] == px(T, ep(end))
    assert w["ret"] == pytest.approx(px(T, ep(end)) / px(T, ep(start)) - 1, rel=1e-12, abs=1e-15)
    spy_start, spy_end = spy or (start, end)
    assert w["spy_start_price"] == px(SPY, ep(spy_start)) and w["spy_end_price"] == px(SPY, ep(spy_end))
    assert w["spy_ret"] == pytest.approx(px(SPY, ep(spy_end)) / px(SPY, ep(spy_start)) - 1, rel=1e-12, abs=1e-15)
    assert w["truncated"] == int(truncated)


# ---------------------------------------------------------------- price_at and the regular close


def bars_at(*minutes: datetime) -> events.MinuteBars:
    starts = sorted(ep(t) for t in minutes)
    return events.MinuteBars(starts, [px(T, s) for s in starts])


def test_price_at_excludes_the_bar_containing_t():
    d = date(2026, 11, 4)
    bars = bars_at(*(datetime.combine(d, time(9, m), tzinfo=NY) for m in range(55, 60)), ny(2026, 11, 4, 10, 0))
    # The 10:00 bar covers 10:00:00-10:00:59, so it has not finished at 10:00:30 or even 10:00:59.
    for t0 in (ny(2026, 11, 4, 10, 0, 0), ny(2026, 11, 4, 10, 0, 30), ny(2026, 11, 4, 10, 0, 59)):
        assert bars.at(t0) == events.PricePoint(ep(ny(2026, 11, 4, 9, 59)), px(T, ep(ny(2026, 11, 4, 9, 59))))
    assert bars.at(ny(2026, 11, 4, 10, 1, 0)).ts == ep(ny(2026, 11, 4, 10, 0))
    assert bars.at(ny(2026, 11, 4, 9, 59, 59)).ts == ep(ny(2026, 11, 4, 9, 58))
    assert bars.at(ny(2026, 11, 4, 9, 55, 59)) is None
    # sub-second times round toward the past, never forward
    assert bars.at(ny(2026, 11, 4, 10, 0, 59) + timedelta(microseconds=999_999)).ts == ep(ny(2026, 11, 4, 9, 59))
    assert bars.at(ny(2026, 11, 4, 10, 0, 30).astimezone(UTC)).ts == ep(ny(2026, 11, 4, 9, 59))


def test_price_at_falls_back_over_minutes_without_trades():
    bars = bars_at(ny(2026, 11, 4, 9, 50), ny(2026, 11, 4, 10, 7))
    assert bars.at(ny(2026, 11, 4, 10, 5)).ts == ep(ny(2026, 11, 4, 9, 50))
    assert bars.at(ny(2026, 11, 4, 10, 8)).ts == ep(ny(2026, 11, 4, 10, 7))
    assert bars.at(ny(2026, 11, 5, 3, 0)).ts == ep(ny(2026, 11, 4, 10, 7))


def test_regular_close_is_the_last_bar_starting_before_the_close():
    d = date(2026, 11, 4)
    bars = bars_at(ny(2026, 11, 4, 15, 58), ny(2026, 11, 4, 16, 0), ny(2026, 11, 4, 16, 30))
    assert bars.regular_close(d).ts == ep(ny(2026, 11, 4, 15, 58))
    # only pre-market and after-hours bars: no regular close
    assert bars_at(ny(2026, 11, 4, 9, 29), ny(2026, 11, 4, 16, 0)).regular_close(d) is None
    # early close: 12:59 is the last regular bar, the 13:00 bar is after-hours
    half = date(2026, 11, 27)
    bars = bars_at(ny(2026, 11, 27, 12, 58), ny(2026, 11, 27, 12, 59), ny(2026, 11, 27, 13, 0))
    assert bars.regular_close(half).ts == ep(ny(2026, 11, 27, 12, 59))


def test_minute_bars_reject_unsorted_starts():
    with pytest.raises(ValueError):
        events.MinuteBars([2, 1], [1.0, 1.0])


def test_window_specs_for_non_regular_posts_are_legs_only():
    for t0 in (ny(2026, 11, 4, 8, 0), ny(2026, 11, 3, 17, 0), ny(2026, 11, 7, 12, 0)):
        d0 = market.event_session(t0)
        assert [s.win for s in events.window_specs(t0, d0)] == ["pre_leg", "post_leg"]
    specs = events.window_specs(ny(2026, 11, 4, 11, 0), date(2026, 11, 4))
    assert [s.win for s in specs] == ["pre_leg", "post_leg", "pre60", "p5", "p15", "p30", "p60"]


# ---------------------------------------------------------------- a regular-session post, exactly


def test_regular_post_reference_and_windows(env, conn):
    add_both(conn, date(2026, 11, 3), date(2026, 11, 4))
    t0 = ny(2026, 11, 4, 10, 0, 30)
    add_post(conn, "p1", t0)

    counts = env.enrich()
    assert counts["status"] == "ok"
    assert counts["created"] == 1 and counts["completed"] == 1 and counts["pending"] == 0

    e = event(conn, "p1")
    assert e["t0"] == "2026-11-04T15:00:30Z"
    assert e["d0"] == "2026-11-04" and e["session_phase"] == "regular"
    assert e["status"] == "complete" and e["intraday_state"] == "ok"
    assert e["completed_at"] == to_iso(NOW) and e["created_at"] == to_iso(NOW)
    # 10:00:30 is inside the 10:00 bar, which finishes after the post: the reference is the 09:59 bar.
    assert e["ref_ts"] == ep(ny(2026, 11, 4, 9, 59)) and e["ref_price"] == px(T, e["ref_ts"])
    assert (e["spans_open"], e["clustered"], e["split_flag"], e["earnings_flag"]) == (0, 0, 0, 0)

    w = windows(conn, e["id"])
    assert set(w) == {"pre_leg", "post_leg", "pre60", "p5", "p15", "p30", "p60"}
    ref = ny(2026, 11, 4, 9, 59)
    assert_window(w["pre_leg"], ny(2026, 11, 3, 15, 59), ref)
    assert_window(w["post_leg"], ref, ny(2026, 11, 4, 15, 59))
    # t0 - 60 min is 09:00:30, before the open: clipped at the open, whose price is the 09:29 pre-market bar
    assert_window(w["pre60"], ny(2026, 11, 4, 9, 29), ref, truncated=True)
    assert_window(w["p5"], ref, ny(2026, 11, 4, 10, 4))
    assert_window(w["p15"], ref, ny(2026, 11, 4, 10, 14))
    assert_window(w["p30"], ref, ny(2026, 11, 4, 10, 29))
    assert_window(w["p60"], ref, ny(2026, 11, 4, 10, 59))


def test_pre60_is_untruncated_exactly_60_minutes_after_the_open(env, conn):
    add_both(conn, date(2026, 11, 3), date(2026, 11, 4))
    add_post(conn, "p1", ny(2026, 11, 4, 10, 30))
    env.enrich()
    w = windows(conn, event(conn, "p1")["id"])
    assert_window(w["pre60"], ny(2026, 11, 4, 9, 29), ny(2026, 11, 4, 10, 29), truncated=False)
    assert_window(w["p60"], ny(2026, 11, 4, 10, 29), ny(2026, 11, 4, 11, 29))


def test_window_ending_exactly_at_the_close_is_not_truncated(env, conn):
    add_both(conn, date(2026, 11, 3), date(2026, 11, 4))
    add_post(conn, "p1", ny(2026, 11, 4, 15, 45))
    env.enrich()
    w = windows(conn, event(conn, "p1")["id"])
    ref, last = ny(2026, 11, 4, 15, 44), ny(2026, 11, 4, 15, 59)
    assert_window(w["p15"], ref, last, truncated=False)
    assert_window(w["p30"], ref, last, truncated=True)
    assert_window(w["post_leg"], ref, last)


def test_missing_minutes_fall_back_to_the_last_earlier_bar(env, conn):
    d0, prev = date(2026, 11, 4), date(2026, 11, 3)
    gaps_d0 = [ny(2026, 11, 4, 9, m) for m in (57, 58, 59)] + [ny(2026, 11, 4, 10, m) for m in range(10, 15)]
    gaps_prev = [ny(2026, 11, 3, 15, m) for m in range(50, 60)]
    add_minutes(conn, T, prev, skip=gaps_prev)
    add_minutes(conn, T, d0, skip=gaps_d0)
    add_minutes(conn, SPY, prev)
    add_minutes(conn, SPY, d0)
    add_post(conn, "p1", ny(2026, 11, 4, 10, 0, 30))
    env.enrich()

    e = event(conn, "p1")
    assert e["ref_ts"] == ep(ny(2026, 11, 4, 9, 56))
    w = windows(conn, e["id"])
    ref, spy_ref = ny(2026, 11, 4, 9, 56), ny(2026, 11, 4, 9, 59)
    # the 16:00+ after-hours bars exist on Nov 3, but the regular close is the last bar before 16:00
    assert_window(w["pre_leg"], ny(2026, 11, 3, 15, 49), ref, spy=(ny(2026, 11, 3, 15, 59), spy_ref))
    assert_window(w["p15"], ref, ny(2026, 11, 4, 10, 9), spy=(spy_ref, ny(2026, 11, 4, 10, 14)))
    assert_window(w["p5"], ref, ny(2026, 11, 4, 10, 4), spy=(spy_ref, ny(2026, 11, 4, 10, 4)))


@pytest.mark.parametrize(
    ("t0", "spans", "ref"),
    [
        (ny(2026, 11, 4, 9, 30, 0), 1, ny(2026, 11, 4, 9, 29)),
        (ny(2026, 11, 4, 9, 30, 20), 1, ny(2026, 11, 4, 9, 29)),
        (ny(2026, 11, 4, 9, 30, 59), 1, ny(2026, 11, 4, 9, 29)),
        (ny(2026, 11, 4, 9, 31, 0), 0, ny(2026, 11, 4, 9, 30)),
        (ny(2026, 11, 4, 9, 29, 59), 0, ny(2026, 11, 4, 9, 28)),
    ],
)
def test_spans_open(env, conn, t0: datetime, spans: int, ref: datetime):
    add_both(conn, date(2026, 11, 3), date(2026, 11, 4))
    add_post(conn, "p1", t0)
    env.enrich()
    e = event(conn, "p1")
    assert e["spans_open"] == spans
    assert e["ref_ts"] == ep(ref)
    w = windows(conn, e["id"])
    if e["session_phase"] == "regular":
        # the pre-window collapses onto the open: from the 09:29 bar to the reference
        assert_window(w["pre60"], ny(2026, 11, 4, 9, 29), ref, truncated=True)
        # the +5 window ends with the last bar finished by t0 + 5 min
        p5_end = (t0 + timedelta(minutes=4)).replace(second=0)
        assert_window(w["p5"], ref, p5_end)
    else:
        assert set(w) == {"pre_leg", "post_leg"}


# ---------------------------------------------------------------- calendar edge cases end to end


def test_early_close_truncates_windows_and_after_hours_starts_at_13(env, conn):
    add_both(conn, date(2026, 11, 25), date(2026, 11, 27), date(2026, 11, 30))
    add_post(conn, "noon", ny(2026, 11, 27, 12, 40))
    add_post(conn, "one", ny(2026, 11, 27, 13, 0))
    now = datetime(2026, 12, 8, 21, 0, tzinfo=NY)
    counts = env.enrich(now)
    assert counts["completed"] == 2

    noon = event(conn, "noon")
    assert (noon["d0"], noon["session_phase"]) == ("2026-11-27", "regular")
    w = windows(conn, noon["id"])
    ref = ny(2026, 11, 27, 12, 39)
    assert_window(w["p5"], ref, ny(2026, 11, 27, 12, 44))
    assert_window(w["p15"], ref, ny(2026, 11, 27, 12, 54))
    assert_window(w["p30"], ref, ny(2026, 11, 27, 12, 59), truncated=True)
    assert_window(w["p60"], ref, ny(2026, 11, 27, 12, 59), truncated=True)
    assert_window(w["pre60"], ny(2026, 11, 27, 11, 39), ref)
    assert_window(w["post_leg"], ref, ny(2026, 11, 27, 12, 59))
    assert_window(w["pre_leg"], ny(2026, 11, 25, 15, 59), ref)

    one = event(conn, "one")
    assert (one["d0"], one["session_phase"]) == ("2026-11-30", "after")
    assert one["clustered"] == 0  # a different event session from the 12:40 post
    w = windows(conn, one["id"])
    assert set(w) == {"pre_leg", "post_leg"}
    assert_window(w["pre_leg"], ny(2026, 11, 27, 12, 59), ny(2026, 11, 27, 12, 59))
    assert_window(w["post_leg"], ny(2026, 11, 27, 12, 59), ny(2026, 11, 30, 15, 59))


def test_post_across_the_dst_switch(env, conn):
    add_both(conn, date(2026, 10, 30), date(2026, 11, 2))
    add_post(conn, "dst", ny(2026, 10, 30, 16, 30))
    env.enrich(datetime(2026, 11, 10, 21, 0, tzinfo=NY))

    e = event(conn, "dst")
    assert e["t0"] == "2026-10-30T20:30:00Z"
    assert (e["d0"], e["session_phase"], e["status"]) == ("2026-11-02", "after", "complete")
    # EDT (UTC-4) on the Friday, EST (UTC-5) on the Monday
    assert e["ref_ts"] == ep(datetime(2026, 10, 30, 20, 29, tzinfo=UTC))
    w = windows(conn, e["id"])
    assert w["pre_leg"]["start_ts"] == ep(datetime(2026, 10, 30, 19, 59, tzinfo=UTC))
    assert w["post_leg"]["end_ts"] == ep(datetime(2026, 11, 2, 20, 59, tzinfo=UTC))
    assert_window(w["pre_leg"], ny(2026, 10, 30, 15, 59), ny(2026, 10, 30, 16, 29))
    assert_window(w["post_leg"], ny(2026, 10, 30, 16, 29), ny(2026, 11, 2, 15, 59))


def test_weekend_post_belongs_to_monday(env, conn):
    add_both(conn, date(2026, 11, 6), date(2026, 11, 9))
    add_post(conn, "sat", ny(2026, 11, 7, 12, 0))
    env.enrich(datetime(2026, 11, 17, 21, 0, tzinfo=NY))
    e = event(conn, "sat")
    assert (e["d0"], e["session_phase"], e["status"]) == ("2026-11-09", "closed", "complete")
    # Friday's last after-hours bar is the last price before the post
    assert e["ref_ts"] == ep(ny(2026, 11, 6, 19, 59))
    w = windows(conn, e["id"])
    assert_window(w["pre_leg"], ny(2026, 11, 6, 15, 59), ny(2026, 11, 6, 19, 59))
    assert_window(w["post_leg"], ny(2026, 11, 6, 19, 59), ny(2026, 11, 9, 15, 59))


def test_thanksgiving_post_belongs_to_the_half_day(env, conn):
    add_both(conn, date(2026, 11, 25), date(2026, 11, 27))
    add_post(conn, "turkey", ny(2026, 11, 26, 11, 0))
    env.enrich(datetime(2026, 12, 8, 21, 0, tzinfo=NY))
    e = event(conn, "turkey")
    assert (e["d0"], e["session_phase"]) == ("2026-11-27", "closed")
    assert e["ref_ts"] == ep(ny(2026, 11, 25, 19, 59))
    w = windows(conn, e["id"])
    assert_window(w["post_leg"], ny(2026, 11, 25, 19, 59), ny(2026, 11, 27, 12, 59))


def test_pre_open_post_uses_same_day_and_premarket_reference(env, conn):
    add_both(conn, date(2026, 11, 3), date(2026, 11, 4))
    add_post(conn, "early", ny(2026, 11, 4, 8, 0))
    env.enrich()
    e = event(conn, "early")
    assert (e["d0"], e["session_phase"]) == ("2026-11-04", "pre")
    assert e["ref_ts"] == ep(ny(2026, 11, 4, 7, 59))
    w = windows(conn, e["id"])
    assert set(w) == {"pre_leg", "post_leg"}
    assert_window(w["pre_leg"], ny(2026, 11, 3, 15, 59), ny(2026, 11, 4, 7, 59))


# ---------------------------------------------------------------- pending -> complete


def test_daily_bars_arriving_later_complete_the_event(env, conn):
    add_both(conn, date(2026, 11, 3), date(2026, 11, 4))
    add_post(conn, "p1", ny(2026, 11, 4, 10, 0, 30))

    # d0+5 is Wed Nov 11; its daily bar is only stored once that session's extended hours have ended
    for now in (datetime(2026, 11, 9, 21, 0, tzinfo=NY), datetime(2026, 11, 11, 19, 59, tzinfo=NY)):
        counts = env.enrich(now)
        assert counts["status"] == "ok" and counts["pending"] == 1 and counts["completed"] == 0
        e = event(conn, "p1")
        assert e["status"] == "pending"
        assert e["intraday_state"] is None and e["ref_ts"] is None and e["completed_at"] is None
        assert e["earnings_flag"] is None and e["split_flag"] is None
        assert windows(conn, e["id"]) == {}
    stored = conn.execute("SELECT MAX(session_date) FROM bars_1d WHERE symbol = ?", (T,)).fetchone()[0]
    assert stored == "2026-11-10"

    counts = env.enrich(datetime(2026, 11, 11, 20, 0, tzinfo=NY))
    assert counts["completed"] == 1 and counts["pending"] == 0
    e = event(conn, "p1")
    assert e["status"] == "complete" and e["intraday_state"] == "ok"
    assert e["completed_at"] == to_iso(datetime(2026, 11, 11, 20, 0, tzinfo=NY))
    assert len(windows(conn, e["id"])) == 7

    # complete events are left alone by later runs
    counts = env.enrich(datetime(2026, 11, 20, 21, 0, tzinfo=NY))
    assert counts["completed"] == 0
    assert event(conn, "p1")["completed_at"] == to_iso(datetime(2026, 11, 11, 20, 0, tzinfo=NY))


def test_minute_bars_arriving_later_complete_the_event(env, conn):
    add_both(conn, date(2026, 11, 3))
    add_minutes(conn, SPY, date(2026, 11, 4))
    add_post(conn, "p1", ny(2026, 11, 4, 10, 0, 30))

    counts = env.enrich()
    assert counts["pending"] == 1 and counts["completed"] == 0
    assert event(conn, "p1")["intraday_state"] is None

    # a session the snapshot stored with no bars still counts as missing
    with conn:
        db.mark_session(conn, T, "2026-11-04", 0, NOW)
    assert env.enrich()["pending"] == 1

    add_minutes(conn, T, date(2026, 11, 4))
    counts = env.enrich(NOW + timedelta(minutes=5))
    assert counts["completed"] == 1 and counts["intraday_unavailable"] == 0
    e = event(conn, "p1")
    assert e["intraday_state"] == "ok" and e["ref_ts"] == ep(ny(2026, 11, 4, 9, 59))


def test_intraday_unavailable_when_d0_left_the_snapshot_window(env, conn):
    add_post(conn, "old", ny(2026, 11, 4, 10, 0, 30))
    now = datetime(2026, 12, 20, 21, 0, tzinfo=NY)  # the 29-day window starts at Nov 21
    counts = env.enrich(now)
    assert counts["completed"] == 1 and counts["intraday_unavailable"] == 1 and counts["pending"] == 0
    e = event(conn, "old")
    assert e["status"] == "complete" and e["intraday_state"] == "unavailable"
    assert e["ref_ts"] is None and e["ref_price"] is None
    assert windows(conn, e["id"]) == {}
    assert e["earnings_flag"] == 0 and e["split_flag"] == 0


def test_old_event_with_stored_minute_bars_is_still_computed(env, conn):
    add_both(conn, date(2026, 11, 3), date(2026, 11, 4))
    add_post(conn, "old", ny(2026, 11, 4, 10, 0, 30))
    env.enrich(datetime(2026, 12, 20, 21, 0, tzinfo=NY))
    assert event(conn, "old")["intraday_state"] == "ok"


def test_unavailable_when_one_needed_session_can_never_arrive(env, conn):
    # With an 8-day window at Nov 12 the snapshot still covers d0 (Nov 4) but no longer the session before it.
    env.wl = env.wl.model_copy(update={"prices": env.wl.prices.model_copy(update={"backfill_days": 8})})
    assert date(2026, 11, 4) in prices.snapshot_window(NOW, 8)
    assert date(2026, 11, 3) not in prices.snapshot_window(NOW, 8)
    add_minutes(conn, SPY, date(2026, 11, 3))
    add_minutes(conn, SPY, date(2026, 11, 4))
    add_minutes(conn, T, date(2026, 11, 3))
    add_post(conn, "a", ny(2026, 11, 4, 10, 0, 30))
    assert env.enrich()["pending"] == 1  # only d0 is missing, and it can still be fetched

    with conn:
        conn.execute("DELETE FROM bars_1m_sessions WHERE symbol = ? AND session_date = '2026-11-03'", (T,))
    counts = env.enrich()
    assert counts["intraday_unavailable"] == 1
    assert event(conn, "a")["intraday_state"] == "unavailable"


def test_missing_spy_daily_bar_keeps_the_event_pending(env, conn):
    add_both(conn, date(2026, 11, 3), date(2026, 11, 4))
    add_post(conn, "p1", ny(2026, 11, 4, 10, 0, 30))
    env.daily.fail[SPY] = ConnectionError("reset")
    counts = env.enrich()
    assert counts["status"] == "partial"
    assert counts["daily"]["symbols_failed"] == [SPY]
    assert counts["pending"] == 1 and event(conn, "p1")["status"] == "pending"


def test_all_daily_downloads_failing_is_an_error(env, conn):
    add_post(conn, "p1", ny(2026, 11, 4, 10, 0, 30))
    env.daily.fail = {T: ConnectionError("offline"), SPY: ConnectionError("offline")}
    counts = env.enrich()
    assert counts["status"] == "error" and counts["pending"] == 1


# ---------------------------------------------------------------- flags


def test_flags_are_set_at_completion_not_creation(env, conn):
    add_both(conn, date(2026, 11, 3), date(2026, 11, 4))
    add_post(conn, "p1", ny(2026, 11, 4, 10, 0, 30))
    env.enrich(datetime(2026, 11, 6, 21, 0, tzinfo=NY))
    e = event(conn, "p1")
    assert e["status"] == "pending" and e["earnings_flag"] is None and e["split_flag"] is None
    assert env.earnings.calls == [T]

    # An announcement after Tuesday's close is first traded on d0; it is stored after the event was created.
    with conn:
        conn.execute("INSERT INTO earnings (symbol, earnings_at) VALUES (?, ?)", (T, to_iso(ny(2026, 11, 3, 16, 5))))
    env.enrich()
    e = event(conn, "p1")
    assert env.earnings.calls == [T]  # fetched 6 days ago: not due again
    assert e["status"] == "complete" and e["earnings_flag"] == 1 and e["split_flag"] == 0


def test_earnings_refetched_after_creation_are_seen_at_completion(env, conn):
    add_post(conn, "p1", ny(2026, 11, 4, 10, 0, 30))
    env.enrich(datetime(2026, 11, 6, 21, 0, tzinfo=NY))
    env.earnings.dates[T] = [OLD_EARNINGS, ny(2026, 11, 5, 7, 0)]
    env.enrich(datetime(2026, 12, 20, 21, 0, tzinfo=NY))
    assert env.earnings.calls == [T, T]
    assert event(conn, "p1")["earnings_flag"] == 1


@pytest.mark.parametrize(
    ("announced", "flag"),
    [
        (ny(2026, 11, 3, 16, 5), 1),  # after d0-1's close -> traded on d0
        (ny(2026, 11, 4, 7, 0), 1),  # d0 before the open
        (ny(2026, 11, 4, 12, 0), 1),  # d0 during the session
        (ny(2026, 11, 2, 16, 30), 1),  # after d0-2's close -> traded on d0-1
        (ny(2026, 11, 3, 8, 0), 1),  # d0-1 before the open
        (ny(2026, 11, 5, 8, 0), 1),  # d0+1 before the open
        (ny(2026, 11, 1, 20, 0), 0),  # Sunday evening -> Monday Nov 2 = d0-2
        (ny(2026, 11, 2, 12, 0), 0),  # d0-2 during the session
        (ny(2026, 11, 5, 16, 5), 0),  # after d0+1's close -> traded on d0+2
        (ny(2026, 11, 6, 8, 0), 0),  # d0+2
    ],
)
def test_earnings_flag_window(env, conn, announced: datetime, flag: int):
    env.earnings.dates[T] = [OLD_EARNINGS, announced]
    add_post(conn, "p1", ny(2026, 11, 4, 10, 0, 30))
    env.enrich(datetime(2026, 12, 20, 21, 0, tzinfo=NY))
    assert event(conn, "p1")["earnings_flag"] == flag


@pytest.mark.parametrize(
    ("announced", "session"),
    [
        (datetime(2026, 11, 17, 20, 0, tzinfo=UTC), date(2026, 11, 18)),  # Yahoo's "3 PM EST" for a 4 PM report
        (datetime(2026, 11, 17, 21, 0, tzinfo=UTC), date(2026, 11, 18)),  # 4 PM EST
        (datetime(2025, 11, 19, 21, 0, tzinfo=UTC), date(2025, 11, 20)),  # a past report, listed correctly
        (ny(2026, 11, 18, 15, 59), date(2026, 11, 19)),
        (ny(2026, 11, 18, 14, 59), date(2026, 11, 18)),  # still an in-session announcement
        (ny(2026, 11, 18, 7, 0), date(2026, 11, 18)),  # an 8 AM report at the summer offset stays pre-open
        (ny(2026, 11, 20, 15, 0), date(2026, 11, 23)),  # Friday -> Monday
        (ny(2026, 11, 27, 15, 0), date(2026, 11, 30)),  # already after the 13:00 early close
        (ny(2026, 10, 20, 15, 0), date(2026, 10, 20)),  # 3 PM EDT is not the offset slip: in session
        (ny(2027, 3, 12, 15, 0), date(2027, 3, 15)),  # the last standard-time Friday before the March switch
        (ny(2027, 3, 17, 17, 0), date(2027, 3, 18)),  # a 4 PM EST-offset listing shown as 5 PM EDT
    ],
)
def test_earnings_session(announced: datetime, session: date):
    assert events.earnings_session(announced) == session


def test_earnings_flag_reads_yahoos_3pm_winter_listing_as_after_the_close(env, conn):
    # Fetched in September, Yahoo lists NVDA's Nov 17 after-close report as "3 PM EST": first traded Nov 18.
    env.earnings.dates[T] = [OLD_EARNINGS, datetime(2026, 11, 17, 20, 0, tzinfo=UTC)]
    add_post(conn, "mon", ny(2026, 11, 16, 10, 0, 30))
    add_post(conn, "wed", ny(2026, 11, 18, 10, 0, 30))
    add_post(conn, "thu", ny(2026, 11, 19, 10, 0, 30))
    env.enrich(datetime(2026, 12, 20, 21, 0, tzinfo=NY))
    flags = {n: (event(conn, n)["status"], event(conn, n)["earnings_flag"]) for n in ("mon", "wed", "thu")}
    assert flags == {"mon": ("complete", 0), "wed": ("complete", 1), "thu": ("complete", 1)}


def test_earnings_flag_is_unknown_without_a_successful_fetch(env, conn):
    env.earnings.dates[T] = ConnectionError("offline")
    add_post(conn, "p1", ny(2026, 11, 4, 10, 0, 30))
    counts = env.enrich(datetime(2026, 12, 20, 21, 0, tzinfo=NY))
    assert counts["status"] == "partial" and counts["earnings"]["failed"] == [T]
    fetch = conn.execute("SELECT ok, error FROM earnings_fetch WHERE symbol = ?", (T,)).fetchone()
    assert fetch["ok"] == 0 and "offline" in fetch["error"]
    assert event(conn, "p1")["earnings_flag"] is None


def test_earnings_flag_is_unknown_when_the_dates_start_after_the_event(env, conn):
    env.earnings.dates[T] = [ny(2027, 2, 25, 16, 5)]
    add_post(conn, "p1", ny(2026, 11, 4, 10, 0, 30))
    env.enrich(datetime(2026, 12, 20, 21, 0, tzinfo=NY))
    assert event(conn, "p1")["earnings_flag"] is None


def test_earnings_flag_is_unknown_when_the_latest_fetch_failed(env, conn):
    # Older dates without a match are still stored, but the refetch that would confirm them failed.
    add_post(conn, "p1", ny(2026, 11, 4, 10, 0, 30))
    env.enrich(datetime(2026, 11, 6, 21, 0, tzinfo=NY))
    env.earnings.dates[T] = TimeoutError("read timed out")
    env.enrich(datetime(2026, 12, 20, 21, 0, tzinfo=NY))
    assert conn.execute("SELECT COUNT(*) FROM earnings WHERE symbol = ?", (T,)).fetchone()[0] == 1
    assert event(conn, "p1")["earnings_flag"] is None


def test_minute_sessions_are_fetchable_inside_the_window_or_after_the_last_completed_session(conn, watchlist):
    completer = events._Completer(conn, watchlist, NOW)  # window: Oct 14 .. Nov 12
    assert completer.fetchable(date(2026, 10, 14)) and completer.fetchable(date(2026, 11, 12))
    assert not completer.fetchable(date(2026, 10, 13))
    assert completer.fetchable(date(2026, 11, 13))  # not finished yet at NOW


def test_earnings_are_fetched_only_for_tickers_with_pending_events(env, conn):
    add_post(conn, "p1", ny(2026, 11, 4, 10, 0, 30), tickers=[T, "TSLA"])
    env.enrich(datetime(2026, 12, 20, 21, 0, tzinfo=NY))
    assert sorted(env.earnings.calls) == ["NVDA", "TSLA"]
    add_post(conn, "p2", ny(2026, 12, 21, 10, 0), tickers=["TSLA"])
    env.earnings.calls.clear()
    env.enrich(datetime(2026, 12, 29, 21, 0, tzinfo=NY))
    assert env.earnings.calls == ["TSLA"]


@pytest.mark.parametrize(
    ("split_day", "flag"),
    [
        (date(2026, 11, 11), 1),  # d0+5
        (date(2026, 10, 28), 1),  # d0-5
        (date(2026, 11, 4), 1),  # d0
        (date(2026, 11, 12), 0),  # d0+6
        (date(2026, 10, 27), 0),  # d0-6
    ],
)
def test_split_flag_window(env, conn, split_day: date, flag: int):
    env.daily.splits = {T: {split_day: 4.0}}
    add_post(conn, "p1", ny(2026, 11, 4, 10, 0, 30))
    env.enrich(datetime(2026, 12, 20, 21, 0, tzinfo=NY))
    e = event(conn, "p1")
    assert e["split_flag"] == flag
    stored = conn.execute(
        "SELECT split_ratio FROM bars_1d WHERE symbol = ? AND session_date = ?", (T, split_day.isoformat())
    )
    assert stored.fetchone()["split_ratio"] == 4.0


def test_spy_split_does_not_flag_the_event(env, conn):
    env.daily.splits = {SPY: {date(2026, 11, 4): 2.0}}
    add_post(conn, "p1", ny(2026, 11, 4, 10, 0, 30))
    env.enrich(datetime(2026, 12, 20, 21, 0, tzinfo=NY))
    assert event(conn, "p1")["split_flag"] == 0


# ---------------------------------------------------------------- event rows: sync and clustering


def test_sync_creates_one_event_per_post_and_event_ticker(conn, watchlist):
    add_post(conn, "p1", ny(2026, 11, 4, 10, 0), tickers=[T, SPY, "QQQ", "TSLA"])
    with conn:  # a second match of the same ticker in the same post, and a symbol the config no longer has
        conn.execute("INSERT INTO mentions VALUES ('x', 'p1', 'NVDA', 'name', 'Nvidia')")
        conn.execute("INSERT INTO mentions VALUES ('x', 'p1', 'ZZZZ', 'cashtag', '$ZZZZ')")
    counts = events.sync_events(conn, watchlist, NOW)
    assert counts == {"status": "ok", "events": 2, "created": 2, "deleted": 0, "skipped": 0}
    tickers = [r["ticker"] for r in conn.execute("SELECT ticker FROM events ORDER BY ticker")]
    assert tickers == [T, "TSLA"]
    assert events.sync_events(conn, watchlist, NOW + timedelta(hours=1))["created"] == 0
    assert event(conn, "p1")["created_at"] == to_iso(NOW)


def test_events_are_deleted_with_their_windows_when_the_mention_disappears(env, conn):
    add_both(conn, date(2026, 11, 3), date(2026, 11, 4))
    add_post(conn, "p1", ny(2026, 11, 4, 10, 0, 30), tickers=[T, "TSLA"])
    env.enrich()
    nvda = event(conn, "p1")
    assert len(windows(conn, nvda["id"])) == 7

    # the post was re-matched and no longer mentions NVDA
    with conn:
        db.replace_mentions(conn, "x", "p1", [Mention("TSLA", "cashtag", "$TSLA")], [], NOW)
    counts = env.enrich()
    assert counts["deleted"] == 1 and counts["events"] == 1
    assert conn.execute("SELECT COUNT(*) FROM events WHERE ticker = ?", (T,)).fetchone()[0] == 0
    assert windows(conn, nvda["id"]) == {}
    assert event(conn, "p1", "TSLA")["status"] in ("pending", "complete")


def test_events_are_deleted_when_their_ticker_leaves_the_universe(conn, watchlist):
    add_post(conn, "p1", ny(2026, 11, 4, 10, 0), tickers=[T, "TSLA"])
    events.sync_events(conn, watchlist, NOW)
    fewer = watchlist.model_copy(update={"tickers": [t for t in watchlist.tickers if t.symbol != T]})
    counts = events.sync_events(conn, fewer, NOW)
    assert counts["deleted"] == 1 and counts["events"] == 1


def clustered(conn: sqlite3.Connection) -> dict[str, int]:
    rows = conn.execute("SELECT native_id, ticker, clustered FROM events").fetchall()
    return {f"{r['native_id']}:{r['ticker']}": r["clustered"] for r in rows}


def test_clustering_is_recomputed_when_an_earlier_post_arrives_late(conn, watchlist):
    add_post(conn, "later", ny(2026, 11, 4, 11, 0), tickers=[T, "TSLA"])
    add_post(conn, "other_day", ny(2026, 11, 4, 16, 0))  # after the close: next session
    events.sync_events(conn, watchlist, NOW)
    events.recompute_clustered(conn)
    assert clustered(conn) == {"later:NVDA": 0, "later:TSLA": 0, "other_day:NVDA": 0}

    # collected on a later run, but posted before the open of the same event session
    add_post(conn, "earlier", ny(2026, 11, 4, 7, 30), platform="truthsocial")
    events.sync_events(conn, watchlist, NOW + timedelta(days=1))
    assert events.recompute_clustered(conn) == 1
    assert clustered(conn) == {"later:NVDA": 1, "later:TSLA": 0, "other_day:NVDA": 0, "earlier:NVDA": 0}
    assert events.recompute_clustered(conn) == 0


def test_clustering_ties_on_t0_go_to_the_lower_id(conn, watchlist):
    add_post(conn, "b", ny(2026, 11, 4, 11, 0))
    add_post(conn, "a", ny(2026, 11, 4, 11, 0), platform="reddit")
    events.sync_events(conn, watchlist, NOW)
    events.recompute_clustered(conn)
    rows = conn.execute("SELECT id, clustered FROM events ORDER BY id").fetchall()
    assert [r["clustered"] for r in rows] == [0, 1]


def test_enrich_recomputes_clustering_for_complete_events(env, conn):
    add_both(conn, date(2026, 11, 3), date(2026, 11, 4))
    add_post(conn, "later", ny(2026, 11, 4, 11, 0))
    env.enrich()
    assert event(conn, "later")["status"] == "complete" and event(conn, "later")["clustered"] == 0
    add_post(conn, "earlier", ny(2026, 11, 4, 10, 0))
    env.enrich(NOW + timedelta(hours=1))
    assert event(conn, "later")["clustered"] == 1 and event(conn, "earlier")["clustered"] == 0


# ---------------------------------------------------------------- enrich plumbing


def test_enrich_downloads_event_tickers_and_spy_from_the_lookback_before_the_first_d0(env, conn):
    add_post(conn, "p1", ny(2026, 11, 4, 10, 0, 30), tickers=[T, "BRK.B"])
    add_post(conn, "p2", ny(2026, 11, 9, 10, 0, 30), tickers=["TSLA"])
    counts = env.enrich()
    start = market.session_offset(date(2026, 11, 4), -300)
    today = date(2026, 11, 12)
    assert env.daily.calls == [("BRK-B", start, today), (T, start, today), ("TSLA", start, today), (SPY, start, today)]
    assert counts["daily"]["requests"] == 4 and counts["daily"]["status"] == "ok"
    assert counts["earnings"]["requests"] == 3
    # every request after the first is paced, including the switch from daily bars to earnings dates
    assert env.sleeps == [env.wl.prices.request_pause_s] * 6
    sessions = conn.execute("SELECT COUNT(*) FROM bars_1d WHERE symbol = 'BRK.B'").fetchone()[0]
    assert sessions == len(market.sessions_in_range(start, date(2026, 11, 12)))


def test_enrich_without_events_makes_no_requests(env, conn):
    counts = env.enrich()
    assert counts["status"] == "ok" and counts["events"] == 0
    assert env.daily.calls == [] and env.earnings.calls == [] and env.sleeps == []
    assert counts["daily"]["requests"] == 0 and counts["earnings"]["requests"] == 0


def test_an_event_reset_to_pending_is_recomputed_with_fresh_windows(env, conn):
    add_both(conn, date(2026, 11, 3), date(2026, 11, 4))
    add_post(conn, "p1", ny(2026, 11, 4, 10, 0, 30))
    env.enrich()
    e = event(conn, "p1")
    with conn:
        conn.execute("UPDATE events SET status = 'pending' WHERE id = ?", (e["id"],))
        conn.execute("UPDATE event_windows SET ret = 99 WHERE event_id = ?", (e["id"],))
    assert env.enrich(NOW + timedelta(hours=1))["completed"] == 1
    w = windows(conn, e["id"])
    assert len(w) == 7 and all(r["ret"] != 99 for r in w.values())


def test_a_failing_event_does_not_block_the_others(env, conn, caplog):
    add_both(conn, date(2026, 11, 3), date(2026, 11, 4))
    add_post(conn, "good", ny(2026, 11, 4, 10, 0, 30))
    add_post(conn, "bad", ny(2026, 11, 4, 11, 0, 30), tickers=["TSLA"])
    events.sync_events(conn, env.wl, NOW)
    with conn:  # a corrupt row: its d0 is a Saturday
        conn.execute("UPDATE events SET d0 = '2026-11-07' WHERE native_id = 'bad'")
    with caplog.at_level(logging.ERROR, logger=events.__name__):
        counts = env.enrich()
    assert counts["status"] == "partial"
    assert counts["failed_events"] == [event(conn, "bad", "TSLA")["id"]]
    assert event(conn, "good")["status"] == "complete"
    assert event(conn, "bad", "TSLA")["status"] == "pending"
    assert "completion failed" in caplog.text


def test_window_without_a_ticker_price_is_skipped(env, conn, caplog):
    # NVDA traded only after-hours on Nov 3, so it has no regular close to start the pre-leg from
    add_minutes(conn, T, date(2026, 11, 3), first=time(16, 0))
    add_minutes(conn, T, date(2026, 11, 4))
    add_minutes(conn, SPY, date(2026, 11, 3))
    add_minutes(conn, SPY, date(2026, 11, 4))
    add_post(conn, "p1", ny(2026, 11, 4, 10, 0, 30))
    with caplog.at_level(logging.WARNING, logger=events.__name__):
        env.enrich()
    e = event(conn, "p1")
    assert e["status"] == "complete" and e["intraday_state"] == "ok"
    w = windows(conn, e["id"])
    assert set(w) == {"post_leg", "pre60", "p5", "p15", "p30", "p60"}
    assert "skipped windows ['pre_leg']" in caplog.text


def test_missing_spy_price_leaves_the_spy_columns_empty(env, conn, caplog):
    add_minutes(conn, T, date(2026, 11, 3))
    add_minutes(conn, T, date(2026, 11, 4))
    add_minutes(conn, SPY, date(2026, 11, 3), first=time(16, 0))
    add_minutes(conn, SPY, date(2026, 11, 4), first=time(10, 0))
    add_post(conn, "p1", ny(2026, 11, 4, 10, 0, 30))
    with caplog.at_level(logging.WARNING, logger=events.__name__):
        env.enrich()
    w = windows(conn, event(conn, "p1")["id"])
    assert w["pre_leg"]["start_price"] == px(T, ep(ny(2026, 11, 3, 15, 59)))
    assert (w["pre_leg"]["spy_start_price"], w["pre_leg"]["spy_end_price"], w["pre_leg"]["spy_ret"]) == (None,) * 3
    assert "no SPY price for windows ['pre_leg']" in caplog.text
    # With no SPY bar on Nov 4 before 10:00, SPY's price as of 10:00:30 is Nov 3's last after-hours bar:
    # the last trade before the instant, loaded from 04:00 on the session before d0.
    assert w["p5"]["spy_start_price"] == px(SPY, ep(ny(2026, 11, 3, 19, 59)))
    assert w["p5"]["spy_end_price"] == px(SPY, ep(ny(2026, 11, 4, 10, 4)))
