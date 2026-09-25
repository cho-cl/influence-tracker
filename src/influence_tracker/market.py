from __future__ import annotations

from datetime import UTC, date, datetime, time
from functools import lru_cache

import exchange_calendars as xcals
import pandas as pd
from exchange_calendars import ExchangeCalendar

from .timeutil import NY

# Yahoo's extended-hours day: pre-market from 04:00, after-hours until 20:00 New York time.
EXTENDED_OPEN = time(4, 0)
EXTENDED_CLOSE = time(20, 0)


@lru_cache(maxsize=1)
def xnys() -> ExchangeCalendar:
    return xcals.get_calendar("XNYS")


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
