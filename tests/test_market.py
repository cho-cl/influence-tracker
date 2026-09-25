from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pandas as pd
import pytest

from influence_tracker import market
from influence_tracker.timeutil import NY


def ny(y: int, m: int, d: int, hh: int, mm: int = 0) -> datetime:
    return datetime(y, m, d, hh, mm, tzinfo=NY)


def test_xnys_is_cached():
    assert market.xnys() is market.xnys()
    assert market.xnys().name == "XNYS"


def test_extended_bounds_before_dst_change():
    # Friday 2026-10-30, still EDT (UTC-4)
    start, end = market.extended_bounds_utc(date(2026, 10, 30))
    assert start == datetime(2026, 10, 30, 8, 0, tzinfo=UTC)
    assert end == datetime(2026, 10, 31, 0, 0, tzinfo=UTC)
    assert start.tzinfo is UTC and end.tzinfo is UTC


def test_extended_bounds_after_dst_change():
    # Monday 2026-11-02, EST (UTC-5) after the Nov 1 switch
    start, end = market.extended_bounds_utc(date(2026, 11, 2))
    assert start == datetime(2026, 11, 2, 9, 0, tzinfo=UTC)
    assert end == datetime(2026, 11, 3, 1, 0, tzinfo=UTC)


def test_sessions_skip_thanksgiving_but_keep_early_close():
    sessions = market.sessions_in_range(date(2026, 11, 25), date(2026, 11, 30))
    assert sessions == [date(2026, 11, 25), date(2026, 11, 27), date(2026, 11, 30)]
    assert all(type(s) is date for s in sessions)


def test_sessions_skip_weekends_and_are_inclusive():
    sessions = market.sessions_in_range(date(2026, 9, 18), date(2026, 9, 21))
    assert sessions == [date(2026, 9, 18), date(2026, 9, 21)]


def test_sessions_empty_for_weekend_or_reversed_range():
    assert market.sessions_in_range(date(2026, 9, 19), date(2026, 9, 20)) == []
    assert market.sessions_in_range(date(2026, 9, 24), date(2026, 9, 23)) == []


def test_sessions_across_dst_change():
    sessions = market.sessions_in_range(date(2026, 10, 29), date(2026, 11, 3))
    assert sessions == [date(2026, 10, 29), date(2026, 10, 30), date(2026, 11, 2), date(2026, 11, 3)]


@pytest.mark.parametrize(
    ("now", "expected"),
    [
        (ny(2026, 9, 24, 19, 59), date(2026, 9, 23)),
        (ny(2026, 9, 24, 20, 0), date(2026, 9, 24)),
        (ny(2026, 9, 24, 9, 30), date(2026, 9, 23)),
        (ny(2026, 9, 24, 23, 59), date(2026, 9, 24)),
        # weekend and Monday morning fall back to Friday
        (ny(2026, 9, 26, 12, 0), date(2026, 9, 25)),
        (ny(2026, 9, 28, 8, 0), date(2026, 9, 25)),
        # Thanksgiving is not a session; the Nov 27 early close still waits for 20:00
        (ny(2026, 11, 26, 21, 0), date(2026, 11, 25)),
        (ny(2026, 11, 27, 17, 0), date(2026, 11, 25)),
        (ny(2026, 11, 27, 20, 0), date(2026, 11, 27)),
        # first Monday after the DST switch: 20:00 EST is 01:00 UTC the next day
        (ny(2026, 11, 2, 19, 59), date(2026, 10, 30)),
        (ny(2026, 11, 2, 20, 0), date(2026, 11, 2)),
    ],
)
def test_last_completed_session(now: datetime, expected: date):
    assert market.last_completed_session(now) == expected
    assert market.last_completed_session(now.astimezone(UTC)) == expected


def test_last_completed_session_boundary_in_utc():
    # 2026-11-03 00:59 UTC is 19:59 EST on Nov 2; the UTC date is already the 3rd.
    assert market.last_completed_session(datetime(2026, 11, 3, 0, 59, tzinfo=UTC)) == date(2026, 10, 30)
    assert market.last_completed_session(datetime(2026, 11, 3, 1, 0, tzinfo=UTC)) == date(2026, 11, 2)


def test_last_completed_session_before_calendar_start():
    first = market.xnys().first_session.date()
    assert market.last_completed_session(datetime(first.year, first.month, first.day, 12, tzinfo=UTC)) is None


# ---------------------------------------------------------------- regular-session bounds


@pytest.mark.parametrize(
    ("session", "open_utc", "close_utc"),
    [
        (date(2026, 9, 24), datetime(2026, 9, 24, 13, 30, tzinfo=UTC), datetime(2026, 9, 24, 20, 0, tzinfo=UTC)),
        # last EDT session before the Nov 1 switch, and the first EST one after it
        (date(2026, 10, 30), datetime(2026, 10, 30, 13, 30, tzinfo=UTC), datetime(2026, 10, 30, 20, 0, tzinfo=UTC)),
        (date(2026, 11, 2), datetime(2026, 11, 2, 14, 30, tzinfo=UTC), datetime(2026, 11, 2, 21, 0, tzinfo=UTC)),
        # the day after Thanksgiving closes at 13:00 New York
        (date(2026, 11, 27), datetime(2026, 11, 27, 14, 30, tzinfo=UTC), datetime(2026, 11, 27, 18, 0, tzinfo=UTC)),
    ],
)
def test_session_bounds_utc(session: date, open_utc: datetime, close_utc: datetime):
    open_, close = market.session_bounds_utc(session)
    assert (open_, close) == (open_utc, close_utc)
    assert open_.tzinfo is UTC and close.tzinfo is UTC
    assert type(open_) is datetime


@pytest.mark.parametrize("day", [date(2026, 11, 26), date(2026, 9, 26), date(2026, 9, 7)])
def test_session_bounds_reject_non_sessions(day: date):
    with pytest.raises(ValueError, match="not an XNYS session"):
        market.session_bounds_utc(day)


def test_calendar_functions_reject_datetimes_and_out_of_range_dates():
    with pytest.raises(TypeError):
        market.session_offset(datetime(2026, 9, 24, 12, tzinfo=UTC), 1)
    with pytest.raises(ValueError, match="outside the XNYS calendar"):
        market.event_session(datetime(2099, 1, 5, 15, tzinfo=UTC))
    with pytest.raises(ValueError, match="naive"):
        market.event_session(datetime(2026, 9, 24, 12))
    with pytest.raises(ValueError, match="naive"):
        market.session_phase(datetime(2026, 9, 24, 12))


# ---------------------------------------------------------------- event session (d0) and phase


def nys(y: int, m: int, d: int, hh: int, mm: int = 0, ss: int = 0) -> datetime:
    return datetime(y, m, d, hh, mm, ss, tzinfo=NY)


@pytest.mark.parametrize(
    ("t0", "d0", "phase"),
    [
        # weekend -> Monday
        (nys(2026, 9, 26, 12), date(2026, 9, 28), "closed"),
        (nys(2026, 9, 27, 23, 30), date(2026, 9, 28), "closed"),
        # Friday after-hours -> Monday; the phase belongs to t0's own date
        (nys(2026, 9, 25, 17), date(2026, 9, 28), "after"),
        # pre-open -> same day
        (nys(2026, 9, 24, 8), date(2026, 9, 24), "pre"),
        (nys(2026, 9, 24, 4), date(2026, 9, 24), "pre"),
        (nys(2026, 9, 24, 3, 59, 59), date(2026, 9, 24), "closed"),
        (nys(2026, 9, 24, 9, 29, 59), date(2026, 9, 24), "pre"),
        (nys(2026, 9, 24, 9, 30), date(2026, 9, 24), "regular"),
        # the close itself belongs to the next session; one second earlier to today
        (nys(2026, 9, 24, 15, 59, 59), date(2026, 9, 24), "regular"),
        (nys(2026, 9, 24, 16), date(2026, 9, 25), "after"),
        (nys(2026, 9, 24, 19, 59, 59), date(2026, 9, 25), "after"),
        (nys(2026, 9, 24, 20), date(2026, 9, 25), "closed"),
        (nys(2026, 9, 24, 23, 59), date(2026, 9, 25), "closed"),
        # Labor Day (Mon Sep 7 2026) is a holiday: even 10:00 is 'closed' and belongs to Tuesday
        (nys(2026, 9, 7, 10), date(2026, 9, 8), "closed"),
        (nys(2026, 9, 4, 18, 2), date(2026, 9, 8), "after"),
        # Thanksgiving -> the Nov 27 half day
        (nys(2026, 11, 26, 11), date(2026, 11, 27), "closed"),
        (nys(2026, 11, 25, 16, 30), date(2026, 11, 27), "after"),
        # early close: after-hours starts at 13:00
        (nys(2026, 11, 27, 12, 40), date(2026, 11, 27), "regular"),
        (nys(2026, 11, 27, 12, 59, 59), date(2026, 11, 27), "regular"),
        (nys(2026, 11, 27, 13), date(2026, 11, 30), "after"),
        (nys(2026, 11, 27, 19, 59), date(2026, 11, 30), "after"),
        # DST: Friday Oct 30 (EDT) after close -> Monday Nov 2 (EST)
        (nys(2026, 10, 30, 16, 30), date(2026, 11, 2), "after"),
        (nys(2026, 11, 1, 1, 30), date(2026, 11, 2), "closed"),
        (nys(2026, 11, 2, 9, 29), date(2026, 11, 2), "pre"),
        (nys(2026, 11, 2, 9, 30), date(2026, 11, 2), "regular"),
    ],
)
def test_event_session_and_phase(t0: datetime, d0: date, phase: str):
    for t in (t0, t0.astimezone(UTC)):
        assert market.event_session(t) == d0
        assert market.session_phase(t) == phase
    assert type(market.event_session(t0)) is date


def test_dst_switch_bounds_in_utc():
    # 20:30 UTC on Fri Oct 30 is 16:30 EDT (after the close); the same UTC clock time on Mon Nov 2 is 15:30 EST.
    assert market.event_session(datetime(2026, 10, 30, 20, 30, tzinfo=UTC)) == date(2026, 11, 2)
    assert market.session_phase(datetime(2026, 11, 2, 20, 30, tzinfo=UTC)) == "regular"
    assert market.event_session(datetime(2026, 11, 2, 20, 30, tzinfo=UTC)) == date(2026, 11, 2)
    # 14:00 UTC is 10:00 EDT on Oct 30 but 09:00 EST on Nov 2
    assert market.session_phase(datetime(2026, 10, 30, 14, 0, tzinfo=UTC)) == "regular"
    assert market.session_phase(datetime(2026, 11, 2, 14, 0, tzinfo=UTC)) == "pre"
    # after-hours ends at 20:00 local: 00:00 UTC in EDT, 01:00 UTC in EST
    assert market.session_phase(datetime(2026, 10, 30, 23, 59, tzinfo=UTC)) == "after"
    assert market.session_phase(datetime(2026, 10, 31, 0, 0, tzinfo=UTC)) == "closed"
    assert market.session_phase(datetime(2026, 11, 3, 0, 59, tzinfo=UTC)) == "after"
    assert market.session_phase(datetime(2026, 11, 3, 1, 0, tzinfo=UTC)) == "closed"


@pytest.mark.parametrize(
    ("start", "end"),
    [
        (datetime(2026, 10, 29, 0, 0, tzinfo=UTC), datetime(2026, 11, 4, 0, 0, tzinfo=UTC)),  # DST switch
        (datetime(2026, 11, 25, 0, 0, tzinfo=UTC), datetime(2026, 12, 1, 0, 0, tzinfo=UTC)),  # holiday, half day
    ],
)
def test_event_session_matches_exchange_calendars_minute_rule(start: datetime, end: datetime):
    """PLAN.md defines d0 as minute_to_session(t0 floored to the minute, direction='next'); check every minute,
    each at :00 and :59 seconds."""
    cal = market.xnys()
    t = start
    while t < end:
        expected = cal.minute_to_session(pd.Timestamp(t), direction="next").date()
        assert market.event_session(t) == expected, t
        assert market.event_session(t + timedelta(seconds=59)) == expected, t
        t += timedelta(minutes=1)


# ---------------------------------------------------------------- session arithmetic


@pytest.mark.parametrize(
    ("session", "k", "expected"),
    [
        (date(2026, 9, 24), 0, date(2026, 9, 24)),
        (date(2026, 9, 24), 1, date(2026, 9, 25)),
        (date(2026, 9, 25), 1, date(2026, 9, 28)),
        (date(2026, 9, 28), -1, date(2026, 9, 25)),
        (date(2026, 11, 25), 1, date(2026, 11, 27)),
        (date(2026, 11, 27), -1, date(2026, 11, 25)),
        (date(2026, 11, 20), 5, date(2026, 11, 30)),
        (date(2026, 10, 30), 1, date(2026, 11, 2)),
        (date(2026, 9, 8), -1, date(2026, 9, 4)),
        (date(2026, 9, 8), -5, date(2026, 8, 31)),
    ],
)
def test_session_offset(session: date, k: int, expected: date):
    assert market.session_offset(session, k) == expected
    assert type(market.session_offset(session, k)) is date


def test_session_offset_round_trips_and_previous_session():
    d = date(2026, 11, 27)
    assert market.session_offset(market.session_offset(d, -150), 150) == d
    assert market.previous_session(d) == date(2026, 11, 25)
    assert market.previous_session(date(2026, 11, 2)) == date(2026, 10, 30)
    assert len(market.sessions_in_range(market.session_offset(d, -5), market.session_offset(d, 5))) == 11


def test_session_offset_rejects_non_sessions():
    with pytest.raises(ValueError):
        market.session_offset(date(2026, 11, 26), 1)
    with pytest.raises(ValueError):
        market.previous_session(date(2026, 9, 26))
