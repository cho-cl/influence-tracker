from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest

from influence_tracker import db, events
from influence_tracker.alerts import followups
from influence_tracker.alerts.engine import AlertEngine
from influence_tracker.alerts.notify import NotifyError
from influence_tracker.alerts.timing import FollowWindow
from influence_tracker.analysis.metrics import EventModel
from influence_tracker.events import MinuteBars
from influence_tracker.models import Mention, Post

T0 = datetime(2026, 9, 29, 14, 31, tzinfo=UTC)  # Tue 10:31 ET
D0 = date(2026, 9, 29)
END = T0 + timedelta(minutes=60)


def minute_bars(start: datetime, end: datetime, first: float, last: float) -> MinuteBars:
    s, e = int(start.timestamp()), int(end.timestamp())
    starts = list(range(s, e, 60))
    step = (last - first) / max(1, len(starts) - 1)
    return MinuteBars(starts, [first + i * step for i in range(len(starts))])


class Prices:
    def __init__(self, series: dict[str, MinuteBars]) -> None:
        self.series = series

    def bars(self, symbol, start, end):
        return self.series.get(symbol, MinuteBars([], []))


def model(beta=1.5):
    return EventModel(alpha=0.0, beta=beta, sigma=0.02, n_est=120, ar={k: 0.0 for k in range(-5, 6)})


def test_follow_60m_rows_market_adjusted():
    window = FollowWindow(T0, END, False)
    live = Prices(
        {
            "NVDA": minute_bars(T0 - timedelta(hours=1), END + timedelta(minutes=2), 100.0, 102.0),
            "SPY": minute_bars(T0 - timedelta(hours=1), END + timedelta(minutes=2), 500.0, 501.0),
        }
    )
    rows = followups.follow_60m_rows(("NVDA",), window, live, lambda t: model(1.5))
    [r] = rows
    nvda, spy = live.bars("NVDA", 0, 0), live.bars("SPY", 0, 0)
    ret = nvda.at(END).price / nvda.at(T0).price - 1
    spy_ret = spy.at(END).price / spy.at(T0).price - 1
    assert r.ret == ret and r.spy_ret == spy_ret
    assert r.abnormal == ret - 1.5 * spy_ret


def test_follow_60m_waits_when_end_bar_is_stale():
    window = FollowWindow(T0, END, False)
    live = Prices(
        {
            "NVDA": minute_bars(T0 - timedelta(hours=1), END - timedelta(minutes=20), 100.0, 101.0),
            "SPY": minute_bars(T0 - timedelta(hours=1), END + timedelta(minutes=2), 500.0, 501.0),
        }
    )
    assert followups.follow_60m_rows(("NVDA",), window, live, lambda t: model()) is None


def test_follow_60m_without_model_or_spy():
    window = FollowWindow(T0, END, False)
    live = Prices({"NVDA": minute_bars(T0 - timedelta(hours=1), END + timedelta(minutes=2), 100.0, 102.0)})
    [r] = followups.follow_60m_rows(("NVDA",), window, live, lambda t: None)
    assert r.spy_ret is None and r.abnormal is None


def test_window_text():
    assert followups.window_text(FollowWindow(T0, END, False), D0) == "From the post (Tue Sep 29 10:31 ET) to 11:31 ET."
    trunc = FollowWindow(datetime(2026, 11, 27, 17, 40, tzinfo=UTC), datetime(2026, 11, 27, 18, 0, tzinfo=UTC), True)
    assert followups.window_text(trunc, date(2026, 11, 27)).endswith("to the close (13:00 ET, early close).")


def _seed_event(conn, native_id, ticker, created, *, platform="truthsocial", d0=D0):
    p = Post(
        platform=platform,
        native_id=native_id,
        author="realDonaldTrump",
        created_at_utc=created,
        text="x",
        url="https://example.com",
    )
    with conn:
        db.upsert_post(conn, p, created)
        db.replace_mentions(conn, platform, native_id, [Mention(ticker, "cashtag", "$" + ticker)], [], created)
        conn.execute(
            """INSERT INTO events (platform, native_id, ticker, t0, d0, session_phase, status, created_at)
               VALUES (?, ?, ?, ?, ?, 'regular', 'pending', ?)""",
            (
                platform,
                native_id,
                ticker,
                created.strftime("%Y-%m-%dT%H:%M:%SZ"),
                d0.isoformat(),
                created.strftime("%Y-%m-%dT%H:%M:%SZ"),
            ),
        )


def test_follow_d1_rows_ready_and_not_ready(conn):
    _seed_event(conn, "p1", "NVDA", T0)
    ready = EventModel(0.0, 1.0, 0.02, 120, {**{k: 0.0 for k in range(-5, 6)}, 0: 0.03, 1: 0.01})
    [r] = followups.follow_d1_rows(conn, "truthsocial", "p1", ("NVDA",), D0, lambda t: ready)
    assert abs(r.car - 0.04) < 1e-12 and abs(r.z - 0.04 / (0.02 * 2**0.5)) < 1e-12
    missing = EventModel(0.0, 1.0, 0.02, 120, {**{k: 0.0 for k in range(-5, 6)}, 1: float("nan")})
    assert followups.follow_d1_rows(conn, "truthsocial", "p1", ("NVDA",), D0, lambda t: missing) is None


SAME_SESSION = "another post about NVDA in the same session"


def test_confounders(conn):
    _seed_event(conn, "p1", "NVDA", T0)
    _seed_event(conn, "p2", "NVDA", T0 + timedelta(hours=2))
    events.recompute_clustered(conn)
    with conn:
        conn.execute("INSERT INTO earnings (symbol, earnings_at) VALUES ('NVDA', '2026-09-29T20:05:00Z')")
        conn.execute("INSERT INTO earnings_fetch (symbol, fetched_at, ok) VALUES ('NVDA', '2026-09-28T00:00:00Z', 1)")
    notes = followups.confounders(conn, "truthsocial", "p2", "NVDA", D0)
    assert "earnings day — the move may be the report, not the post" in notes
    assert SAME_SESSION in notes


def test_same_session_note_goes_on_the_first_post_too(conn):
    # Both posts get the same CAR[0,+1], which covers the whole session, so the first is as confounded as the second.
    _seed_event(conn, "p1", "NVDA", T0)
    _seed_event(conn, "p2", "NVDA", T0 + timedelta(hours=2))
    events.recompute_clustered(conn)  # the study keeps p1 as the primary post (clustered = 0)
    assert followups.confounders(conn, "truthsocial", "p1", "NVDA", D0) == (SAME_SESSION,)
    assert followups.confounders(conn, "truthsocial", "p2", "NVDA", D0) == (SAME_SESSION,)


def test_same_session_note_counts_only_tracked_posts_about_the_ticker_in_that_session(conn):
    _seed_event(conn, "p1", "NVDA", T0)
    _seed_event(conn, "r1", "NVDA", T0 - timedelta(hours=1), platform="reddit")  # crowd chatter, never alerted
    _seed_event(conn, "p2", "AMD", T0 + timedelta(hours=1))
    _seed_event(conn, "p3", "NVDA", T0 + timedelta(days=1), d0=date(2026, 9, 30))
    events.recompute_clustered(conn)  # the earlier Reddit post makes p1 clustered in the study
    assert followups.confounders(conn, "truthsocial", "p1", "NVDA", D0) == ()
    _seed_event(conn, "x1", "NVDA", T0 + timedelta(hours=3), platform="x")
    assert followups.confounders(conn, "truthsocial", "p1", "NVDA", D0) == (SAME_SESSION,)


def test_confounders_keep_stored_earnings_after_a_failed_fetch_and_skip_dates_outside_the_calendar(conn):
    # The study's earnings_flag reads the stored dates whatever the latest fetch did, and skips dates the
    # calendar does not cover; a date outside it must not stop the send loop.
    _seed_event(conn, "p1", "NVDA", T0)
    with conn:
        conn.execute("INSERT INTO earnings (symbol, earnings_at) VALUES ('NVDA', '2001-01-30T21:00:00Z')")
        conn.execute("INSERT INTO earnings_fetch (symbol, fetched_at, ok) VALUES ('NVDA', '2026-09-28T00:00:00Z', 0)")
    assert followups.confounders(conn, "truthsocial", "p1", "NVDA", D0) == ()
    with conn:
        conn.execute("INSERT INTO earnings (symbol, earnings_at) VALUES ('NVDA', '2026-09-28T20:05:00Z')")
    assert followups.confounders(conn, "truthsocial", "p1", "NVDA", D0) == (followups.EARNINGS,)


def test_engine_sends_60m_followup_then_d1_and_skips_stale(conn, watchlist):
    rec = []

    class Rec:
        def send(self, m):
            rec.append(m)

    live = Prices(
        {
            "NVDA": minute_bars(T0 - timedelta(days=1), END + timedelta(minutes=5), 100.0, 101.0),
            "SPY": minute_bars(T0 - timedelta(days=1), END + timedelta(minutes=5), 500.0, 500.5),
        }
    )
    d1_model = EventModel(0.0, 1.0, 0.02, 120, {**{k: 0.0 for k in range(-5, 6)}, 0: 0.01, 1: 0.0})
    eng = AlertEngine(
        conn,
        watchlist,
        Rec(),
        live_factory=lambda: live,
        history_factory=lambda now: lambda a: None,
        model=lambda t, d: d1_model,
    )
    eng.run(T0 - timedelta(minutes=10))
    _seed_event(conn, "p1", "NVDA", T0)
    eng.run(T0 + timedelta(minutes=4))
    eng.run(END + timedelta(minutes=4))
    eng.run(datetime(2026, 10, 1, 0, 30, tzinfo=UTC))
    titles = [m.title for m in rec]
    assert titles == [
        "NVDA · realDonaldTrump · stance unavailable",
        "NVDA · 60 min after realDonaldTrump's post",
        "NVDA · day-after check on realDonaldTrump's post",
    ]
    # a 60-minute follow-up that cannot be sent within 30 minutes of its due time is skipped, not sent late
    _seed_event(conn, "p2", "NVDA", END + timedelta(minutes=10))
    eng.run(END + timedelta(minutes=12))
    eng.run(END + timedelta(minutes=10) + timedelta(hours=3))
    rows = {r["kind"]: r["status"] for r in conn.execute("SELECT kind, status FROM alerts WHERE native_id = 'p2'")}
    assert rows["follow_60m"] == "skipped"


def test_engine_leaves_followups_pending_while_data_is_missing_then_gives_up(conn, watchlist):
    rec = []

    class Rec:
        def send(self, m):
            rec.append(m)

    no_d1_bars = EventModel(0.0, 1.0, 0.02, 120, {**{k: 0.0 for k in range(-5, 6)}, 1: float("nan")})
    eng = AlertEngine(
        conn,
        watchlist,
        Rec(),
        live_factory=lambda: Prices({}),
        history_factory=lambda now: lambda a: None,
        model=lambda t, d: no_d1_bars,
    )

    def row(kind):
        return conn.execute("SELECT * FROM alerts WHERE kind = ? AND native_id = 'p1'", (kind,)).fetchone()

    eng.run(T0 - timedelta(minutes=10))
    _seed_event(conn, "p1", "NVDA", T0)
    eng.run(T0 + timedelta(minutes=4))
    assert eng.run(END + timedelta(minutes=4))["waiting"] == 1
    assert (row("follow_60m")["status"], row("follow_60m")["attempts"]) == ("pending", 0)
    d1_due = datetime(2026, 10, 1, 0, 0, tzinfo=UTC)
    assert eng.run(d1_due + timedelta(days=6))["waiting"] == 1
    assert (row("follow_d1")["status"], row("follow_d1")["attempts"]) == ("pending", 0)
    eng.run(d1_due + followups.D1_GIVE_UP + timedelta(minutes=5))
    assert row("follow_60m")["status"] == "skipped" and row("follow_d1")["status"] == "skipped"
    assert row("follow_d1")["error"] == "gave up: daily bars never arrived"
    assert [m.title for m in rec] == ["NVDA · realDonaldTrump · stance unavailable"]


@pytest.mark.parametrize(
    ("first_try", "retry"),
    [
        (timedelta(days=2, hours=11), timedelta(minutes=5)),  # PC off over the weekend; Wi-Fi not up yet on wake
        (followups.D1_GIVE_UP - timedelta(hours=1), timedelta(hours=2)),  # a failed send is retried past the give-up
    ],
)
def test_late_day_after_followup_gets_its_retries_after_a_failed_send(conn, watchlist, first_try, retry):
    sent, failures = [], [0]

    class FlakyNotifier:
        def send(self, m):
            if failures[0]:
                failures[0] -= 1
                raise NotifyError("network unreachable")
            sent.append(m.title)

    d1_model = EventModel(0.0, 1.0, 0.02, 120, {**{k: 0.0 for k in range(-5, 6)}, 0: 0.01, 1: 0.0})
    eng = AlertEngine(
        conn,
        watchlist,
        FlakyNotifier(),
        live_factory=lambda: Prices({}),
        history_factory=lambda now: lambda a: None,
        model=lambda t, d: d1_model,
    )
    eng.run(T0 - timedelta(minutes=10))
    _seed_event(conn, "p1", "NVDA", T0)
    eng.run(T0 + timedelta(minutes=4))
    d1_due = datetime(2026, 10, 1, 0, 0, tzinfo=UTC)
    failures[0] = 1
    assert eng.run(d1_due + first_try)["failed"] == 1
    assert eng.run(d1_due + first_try + retry)["sent"] == 1
    row = conn.execute("SELECT status, attempts FROM alerts WHERE kind = 'follow_d1' AND native_id = 'p1'").fetchone()
    assert (row["status"], row["attempts"]) == ("sent", 2)
    assert sent[-1] == "NVDA · day-after check on realDonaldTrump's post"
