from __future__ import annotations

from datetime import UTC, date, datetime

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
