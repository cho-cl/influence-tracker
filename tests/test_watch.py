from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from influence_tracker import db
from influence_tracker.alerts import watch
from influence_tracker.alerts.keepawake import ES_CONTINUOUS, ES_SYSTEM_REQUIRED, KeepAwake
from influence_tracker.timeutil import NY

ACTIVE = datetime(2026, 9, 29, 10, 31, tzinfo=NY)
IDLE = datetime(2026, 10, 3, 12, 0, tzinfo=NY)  # Saturday


class Clock:
    def __init__(self, t: datetime) -> None:
        self.t = t.astimezone(UTC)

    def __call__(self) -> datetime:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += timedelta(seconds=seconds)


class SuspendingClock(Clock):
    """Models Windows 8+ relative timers (time.sleep): a sleep that spans the suspend stops counting until resume."""

    def __init__(self, t: datetime, suspend: datetime, resume: datetime) -> None:
        super().__init__(t)
        self.suspend, self.resume = suspend.astimezone(UTC), resume.astimezone(UTC)

    def sleep(self, seconds: float) -> None:
        end = self.t + timedelta(seconds=seconds)
        if self.t < self.suspend <= end:
            end += self.resume - self.suspend
        self.t = end


def steps(clock, calls, *, x=False, fail_collect=False):
    def rec(name):
        def f(*args):
            calls.append((name, clock()))
            if fail_collect and name == "ts":
                raise RuntimeError("boom")
            return {"status": "ok"}

        return f

    states = []
    return watch.Steps(
        clock=clock,
        sleep=clock.sleep,
        collect_truthsocial=rec("ts"),
        collect_x=rec("x") if x else None,
        classify=rec("classify"),
        sync_events=rec("sync"),
        alerts=rec("alerts"),
        keep_awake=KeepAwake(set_state=lambda flags: states.append(flags) or 1),
    ), states


def test_cycle_order_heartbeat_and_cadence(conn, watchlist):
    clock, calls = Clock(ACTIVE), []
    s, states = steps(clock, calls)
    watch.run_watch(conn, watchlist, s, run_id=1, max_cycles=2)
    assert [n for n, _ in calls] == ["ts", "classify", "sync", "alerts"] * 2
    assert calls[4][1] == datetime(2026, 9, 29, 10, 35, tzinfo=NY)  # aligned 5-minute tick
    assert db.get_watermark(conn, "watch", "heartbeat") is not None
    assert watch.heartbeat_fresh(conn, clock()) is True
    assert watch.heartbeat_fresh(conn, clock() + timedelta(minutes=16)) is False
    assert states[0] == ES_CONTINUOUS | ES_SYSTEM_REQUIRED and states[-1] == ES_CONTINUOUS


def test_leaving_the_active_window_lets_the_pc_sleep(conn, watchlist):
    clock, calls = Clock(datetime(2026, 9, 29, 19, 55, tzinfo=NY)), []
    s, states = steps(clock, calls)
    during = []
    s.alerts = lambda now: during.append(list(states)) or {"status": "ok"}
    watch.run_watch(conn, watchlist, s, run_id=1, max_cycles=2)  # 19:55 active, 20:00 idle
    awake = ES_CONTINUOUS | ES_SYSTEM_REQUIRED
    assert during == [[awake], [awake, ES_CONTINUOUS]]  # released at 20:00, not only when the loop ends
    assert states == [awake, ES_CONTINUOUS]


def test_idle_does_not_hold_the_pc_awake(conn, watchlist):
    clock, calls = Clock(IDLE), []
    s, states = steps(clock, calls)
    watch.run_watch(conn, watchlist, s, run_id=1, once=True)
    assert states == []  # never set awake, so nothing to release


def test_a_failing_step_does_not_stop_the_cycle(conn, watchlist):
    clock, calls = Clock(ACTIVE), []
    s, _ = steps(clock, calls, fail_collect=True)
    assert watch.run_watch(conn, watchlist, s, run_id=1, once=True) == 0
    assert [n for n, _ in calls] == ["ts", "classify", "sync", "alerts"]


def test_x_is_polled_at_its_own_interval(conn, watchlist):
    clock, calls = Clock(ACTIVE), []
    s, _ = steps(clock, calls, x=True)
    watch.run_watch(conn, watchlist, s, run_id=1, max_cycles=4)  # 10:31, 10:35, 10:40, 10:45
    x_times = [t for n, t in calls if n == "x"]
    assert [t.astimezone(NY).strftime("%H:%M") for t in x_times] == ["10:31"]  # next X poll is due at 10:46


@pytest.mark.parametrize(
    ("start", "suspend", "resume"),
    [
        # Review Focus night: idle 20:30 cycle, PC sleeps mid-wait, woken at 07:00 on a session day
        (
            datetime(2026, 9, 29, 20, 30, tzinfo=NY),
            datetime(2026, 9, 29, 20, 45, 20, tzinfo=NY),
            datetime(2026, 9, 30, 7, 0, tzinfo=NY),
        ),
        # laptop lid closed during the active window
        (ACTIVE, datetime(2026, 9, 29, 10, 32, 30, tzinfo=NY), datetime(2026, 9, 29, 11, 0, tzinfo=NY)),
    ],
)
def test_a_suspend_mid_wait_does_not_delay_the_first_cycle_after_resume(conn, watchlist, start, suspend, resume):
    clock, calls = SuspendingClock(start, suspend, resume), []
    s, _ = steps(clock, calls)
    watch.run_watch(conn, watchlist, s, run_id=1, max_cycles=3)
    starts = [t for n, t in calls if n == "ts"]
    assert resume <= starts[1] <= resume + timedelta(seconds=watch.SLEEP_SLICE)  # not the rest of the old wait
    assert starts[2] == resume + timedelta(minutes=5)  # back on the aligned 5-minute ticks
