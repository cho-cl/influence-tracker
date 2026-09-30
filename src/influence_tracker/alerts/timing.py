from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

from .. import market
from ..config import AlertsConfig
from ..timeutil import NY

FOLLOW_DELAY = timedelta(minutes=3)  # give Yahoo time to publish the window's last bar
STALE_AFTER = timedelta(minutes=30)
LOOKAHEAD_DAYS = 10


def is_active(now: datetime) -> bool:
    """Inside 04:00-20:00 New York on an XNYS session day."""
    day = now.astimezone(NY).date()
    if not market.is_session(day):
        return False
    start, end = market.extended_bounds_utc(day)
    return start <= now < end


def _next_active_start(now: datetime) -> datetime | None:
    today = now.astimezone(NY).date()
    for session in market.sessions_in_range(today, today + timedelta(days=LOOKAHEAD_DAYS)):
        start = market.extended_bounds_utc(session)[0]
        if start > now:
            return start
    return None


def next_cycle(now: datetime, cfg: AlertsConfig) -> datetime:
    """The next wall-clock-aligned tick for the current window; an idle wait never overshoots the next window start."""
    minutes = cfg.poll_active_minutes if is_active(now) else cfg.poll_idle_minutes
    step = minutes * 60
    tick = datetime.fromtimestamp((math.floor(now.timestamp() / step) + 1) * step, UTC)
    if not is_active(now):
        start = _next_active_start(now)
        if start is not None and start < tick:
            return start
    return tick


@dataclass(frozen=True)
class FollowWindow:
    start: datetime
    end: datetime
    truncated: bool


def follow_window(t0: datetime, d0: date, minutes: int) -> FollowWindow:
    """From the post to `minutes` into trading: the next hour for a regular-session post; for an off-hours post
    through the first hour of d0 (opening gap included). Clipped at d0's (early) close."""
    open_, close = market.session_bounds_utc(d0)
    end = max(t0, open_) + timedelta(minutes=minutes)
    return FollowWindow(t0, min(end, close), end > close)


def is_late(created_at: datetime, found_at: datetime, cfg: AlertsConfig) -> bool:
    return found_at - created_at > timedelta(minutes=cfg.late_after_minutes)
