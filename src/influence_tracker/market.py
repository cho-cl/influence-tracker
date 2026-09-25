from __future__ import annotations

from datetime import UTC, date, datetime, time
from functools import lru_cache
from typing import Literal

import exchange_calendars as xcals
import pandas as pd
from exchange_calendars import ExchangeCalendar

from .timeutil import NY

# Yahoo's extended-hours day: pre-market from 04:00, after-hours until 20:00 New York time.
EXTENDED_OPEN = time(4, 0)
EXTENDED_CLOSE = time(20, 0)

Phase = Literal["pre", "regular", "after", "closed"]


@lru_cache(maxsize=1)
def xnys() -> ExchangeCalendar:
    return xcals.get_calendar("XNYS")


def _aware(t: datetime) -> datetime:
    if t.tzinfo is None:
        raise ValueError("naive datetime; attach a timezone first")
    return t


def is_session(day: date) -> bool:
    if isinstance(day, datetime):
        raise TypeError("expected a date, not a datetime")
    cal = xnys()
    if not cal.first_session.date() <= day <= cal.last_session.date():
        raise ValueError(f"{day} is outside the XNYS calendar ({cal.first_session.date()}..{cal.last_session.date()})")
    return bool(cal.is_session(pd.Timestamp(day)))


@lru_cache(maxsize=8192)
def session_bounds_utc(session: date) -> tuple[datetime, datetime]:
    """Regular-session open and close in UTC; the close honours early closes (13:00 New York)."""
    if not is_session(session):
        raise ValueError(f"{session} is not an XNYS session")
    label = pd.Timestamp(session)
    cal = xnys()
    return (
        cal.session_open(label).to_pydatetime().astimezone(UTC),
        cal.session_close(label).to_pydatetime().astimezone(UTC),
    )


def event_session(t0: datetime) -> date:
    """d0: the first session whose regular close is after t0. A post at exactly the close belongs to the next one."""
    _aware(t0)
    day = t0.astimezone(NY).date()
    # Every earlier session closed before this New York midnight, so only `day` itself can still be open.
    if is_session(day):
        return day if t0 < session_bounds_utc(day)[1] else session_offset(day, 1)
    return xnys().date_to_session(pd.Timestamp(day), direction="next").date()


def session_phase(t0: datetime) -> Phase:
    """pre = [04:00, open), regular = [open, close), after = [close, 20:00) New York time on a session date;
    closed = overnight, weekends and holidays. On early-close days after-hours starts at the early close."""
    _aware(t0)
    day = t0.astimezone(NY).date()
    if not is_session(day):
        return "closed"
    ext_open, ext_close = extended_bounds_utc(day)
    open_, close = session_bounds_utc(day)
    if ext_open <= t0 < open_:
        return "pre"
    if open_ <= t0 < close:
        return "regular"
    if close <= t0 < ext_close:
        return "after"
    return "closed"


def session_offset(session: date, k: int) -> date:
    """The session k sessions after `session` (k < 0: before it); k = 0 returns `session`."""
    if not is_session(session):
        raise ValueError(f"{session} is not an XNYS session")
    return xnys().session_offset(pd.Timestamp(session), k).date()


def previous_session(session: date) -> date:
    return session_offset(session, -1)


def sessions_in_range(start: date, end: date) -> list[date]:
    """NYSE sessions from start to end, both inclusive."""
    if start > end:
        return []
    return [ts.date() for ts in xnys().sessions_in_range(pd.Timestamp(start), pd.Timestamp(end))]


def extended_bounds_utc(session: date) -> tuple[datetime, datetime]:
    """04:00 and 20:00 New York time on the session's date, as UTC."""
    return (
        datetime.combine(session, EXTENDED_OPEN, tzinfo=NY).astimezone(UTC),
        datetime.combine(session, EXTENDED_CLOSE, tzinfo=NY).astimezone(UTC),
    )


def last_completed_session(now: datetime) -> date | None:
    """Latest session whose extended hours (20:00 New York) had ended by `now`, even on early-close days."""
    cal = xnys()
    today = now.astimezone(NY).date()
    if today < cal.first_session.date():
        return None
    session = cal.date_to_session(pd.Timestamp(min(today, cal.last_session.date())), direction="previous")
    if extended_bounds_utc(session.date())[1] > now:
        if session == cal.first_session:
            return None
        session = cal.previous_session(session)
    return session.date()
