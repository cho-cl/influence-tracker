from __future__ import annotations

from datetime import UTC, datetime
from zoneinfo import ZoneInfo

NY = ZoneInfo("America/New_York")
_ISO = "%Y-%m-%dT%H:%M:%SZ"


def utc_now() -> datetime:
    return datetime.now(UTC)


def to_iso(dt: datetime) -> str:
    """Canonical UTC timestamp string; lexicographic order equals time order."""
    if dt.tzinfo is None:
        raise ValueError("naive datetime; attach a timezone first")
    return dt.astimezone(UTC).strftime(_ISO)


def from_iso(s: str) -> datetime:
    return datetime.strptime(s, _ISO).replace(tzinfo=UTC)


def parse_api_time(s: str) -> datetime:
    """Parse API timestamps like '2026-09-24T13:18:56.376Z' or '+00:00' offsets into aware UTC."""
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(UTC)
