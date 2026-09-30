from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest

from influence_tracker.alerts import timing
from influence_tracker.config import AlertsConfig
from influence_tracker.timeutil import NY

CFG = AlertsConfig()


def ny(y, m, d, hh, mm, ss=0) -> datetime:
    return datetime(y, m, d, hh, mm, ss, tzinfo=NY)


@pytest.mark.parametrize(
    ("t", "active"),
    [
        (ny(2026, 9, 29, 3, 59), False),
        (ny(2026, 9, 29, 4, 0), True),
        (ny(2026, 9, 29, 19, 59), True),
        (ny(2026, 9, 29, 20, 0), False),
        (ny(2026, 10, 3, 12, 0), False),  # Saturday
        (ny(2026, 11, 26, 12, 0), False),  # Thanksgiving
        (ny(2026, 11, 27, 17, 0), True),  # early-close day: still inside 04:00-20:00
    ],
)
def test_is_active(t, active):
    assert timing.is_active(t) is active


def test_next_cycle_aligns_to_the_active_interval():
    assert timing.next_cycle(ny(2026, 9, 29, 10, 31, 20), CFG) == ny(2026, 9, 29, 10, 35)


def test_next_cycle_idle_interval_at_night():
    assert timing.next_cycle(ny(2026, 9, 29, 21, 5), CFG) == ny(2026, 9, 29, 21, 30)


def test_next_cycle_wakes_exactly_at_the_active_window_start():
    assert timing.next_cycle(ny(2026, 11, 2, 3, 40), CFG) == ny(2026, 11, 2, 4, 0)


def test_next_cycle_long_idle_interval_is_clamped_to_the_window_start():
    # The unclamped 240-minute tick would be 07:00, three hours into the active window.
    cfg = AlertsConfig(poll_idle_minutes=240)
    assert timing.next_cycle(ny(2026, 11, 2, 3, 40), cfg) == ny(2026, 11, 2, 4, 0)


def test_follow_window_regular_session():
    w = timing.follow_window(ny(2026, 9, 29, 10, 31), date(2026, 9, 29), 60)
    assert (w.start, w.end, w.truncated) == (ny(2026, 9, 29, 10, 31), ny(2026, 9, 29, 11, 31), False)


def test_follow_window_truncated_at_close():
    w = timing.follow_window(ny(2026, 9, 29, 15, 30), date(2026, 9, 29), 60)
    assert (w.end, w.truncated) == (ny(2026, 9, 29, 16, 0), True)


def test_follow_window_premarket_post_runs_to_first_trading_hour():
    w = timing.follow_window(ny(2026, 9, 29, 8, 0), date(2026, 9, 29), 60)
    assert (w.start, w.end) == (ny(2026, 9, 29, 8, 0), ny(2026, 9, 29, 10, 30))


def test_follow_window_weekend_post():
    w = timing.follow_window(ny(2026, 10, 4, 14, 5), date(2026, 10, 5), 60)
    assert w.end == ny(2026, 10, 5, 10, 30)


def test_follow_window_early_close():
    w = timing.follow_window(ny(2026, 11, 27, 12, 40), date(2026, 11, 27), 60)
    assert (w.end, w.truncated) == (ny(2026, 11, 27, 13, 0), True)


def test_is_late():
    t = ny(2026, 9, 29, 10, 0)
    assert timing.is_late(t, t + timedelta(minutes=30), CFG) is False
    assert timing.is_late(t, t + timedelta(minutes=31), CFG) is True
