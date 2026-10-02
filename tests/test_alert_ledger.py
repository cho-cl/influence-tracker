from __future__ import annotations

from datetime import UTC, datetime, timedelta

from influence_tracker import db

T0 = datetime(2026, 9, 29, 14, 0, tzinfo=UTC)


def test_add_alert_is_unique_per_kind_and_post(conn):
    with conn:
        assert db.add_alert(conn, "heads_up", "truthsocial", "1", T0, T0) is True
        assert db.add_alert(conn, "heads_up", "truthsocial", "1", T0, T0) is False
        assert db.add_alert(conn, "follow_60m", "truthsocial", "1", T0, T0) is True
    assert db.alert_status(conn, "heads_up", "truthsocial", "1") == "pending"
    assert db.alert_status(conn, "follow_d1", "truthsocial", "1") is None


def test_due_alerts_selects_pending_and_recent_failures(conn):
    with conn:
        db.add_alert(conn, "heads_up", "truthsocial", "due", T0, T0)
        db.add_alert(conn, "heads_up", "truthsocial", "future", T0 + timedelta(hours=1), T0)
        db.add_alert(conn, "heads_up", "truthsocial", "sent", T0, T0, status="sent")
        db.add_alert(conn, "heads_up", "truthsocial", "failed-recent", T0, T0, status="failed")
        db.add_alert(conn, "heads_up", "truthsocial", "failed-old", T0 - timedelta(hours=30), T0, status="failed")
    rows = db.due_alerts(conn, T0 + timedelta(minutes=1), retry_for=timedelta(hours=24))
    assert [r["native_id"] for r in rows] == ["due", "failed-recent"]  # failed-old is past the 24 h retry period


def test_mark_alert_sets_sent_at_and_counts_attempts(conn):
    with conn:
        db.add_alert(conn, "heads_up", "truthsocial", "1", T0, T0)
    row = db.due_alerts(conn, T0, retry_for=timedelta(hours=24))[0]
    with conn:
        db.mark_alert(conn, row["id"], "failed", T0, error="down")
        db.mark_alert(conn, row["id"], "sent", T0 + timedelta(minutes=5), title="t", message="m")
    got = conn.execute("SELECT * FROM alerts WHERE id = ?", (row["id"],)).fetchone()
    assert got["status"] == "sent" and got["attempts"] == 2
    assert got["sent_at"] == "2026-09-29T14:05:00Z"
    assert (got["title"], got["message"], got["error"]) == ("t", "m", "down")


def test_failed_retry_period_counts_from_the_first_failure(conn):
    # A day-after follow-up is often first tried days after it fell due; it still gets its 24 h of retries.
    due = T0 - timedelta(days=2)
    with conn:
        db.add_alert(conn, "follow_d1", "truthsocial", "1", due, due)
    [row] = db.due_alerts(conn, T0, retry_for=timedelta(hours=24))
    with conn:
        db.mark_alert(conn, row["id"], "failed", T0, error="down")
        db.mark_alert(conn, row["id"], "failed", T0 + timedelta(hours=20), error="down")  # does not extend it

    def due_ids(now):
        return [r["native_id"] for r in db.due_alerts(conn, now, retry_for=timedelta(hours=24))]

    assert due_ids(T0 + timedelta(minutes=5)) == ["1"]
    assert due_ids(T0 + timedelta(hours=23)) == ["1"]
    assert due_ids(T0 + timedelta(hours=25)) == []
