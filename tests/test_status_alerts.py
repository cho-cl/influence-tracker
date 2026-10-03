from __future__ import annotations

from datetime import UTC, datetime, timedelta

from rich.console import Console

from influence_tracker import db
from influence_tracker.status import show_status

NOW = datetime(2026, 9, 29, 15, 0, tzinfo=UTC)  # Tue 11:00 ET, active window


def render(conn, watchlist) -> str:
    console = Console(record=True, width=160)
    show_status(conn, watchlist, NOW, console=console)
    return console.export_text()


def test_status_without_watch(conn, watchlist):
    out = render(conn, watchlist)
    assert "Alerts" in out and "watch has never run" in out


def test_status_with_alerts(conn, watchlist):
    with conn:
        db.set_watermark(conn, "watch", "heartbeat", "2026-09-29T14:40:00Z", NOW)
        db.add_alert(conn, "heads_up", "truthsocial", "1", NOW - timedelta(days=1), NOW, status="sent")
        db.add_alert(conn, "follow_60m", "truthsocial", "1", NOW + timedelta(minutes=5), NOW)
        db.add_alert(
            conn,
            "digest",
            "-",
            "2026-09-28T11:00:00Z",
            NOW - timedelta(days=1),
            NOW,
            status="failed",
            error="ntfy down",
        )
    out = render(conn, watchlist)
    assert "last check-in" in out and "20m ago" in out and "STALE" in out
    assert "heads_up" in out and "pending follow-ups: 1" in out and "failed sends: 1" in out
