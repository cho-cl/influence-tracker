from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pandas as pd

from influence_tracker import db
from influence_tracker.alerts import engine
from influence_tracker.alerts.engine import AlertEngine, StudyHistory
from influence_tracker.alerts.messages import History
from influence_tracker.alerts.notify import Message, NotifyError
from influence_tracker.analysis.study import GROUP_COLUMNS
from influence_tracker.events import MinuteBars
from influence_tracker.models import Mention, Post

T_POST = datetime(2026, 9, 29, 14, 31, tzinfo=UTC)  # Tue 10:31 ET


class Recorder:
    def __init__(self, failures: int = 0) -> None:
        self.sent: list[Message] = []
        self.failures = failures

    def send(self, message: Message) -> None:
        if self.failures:
            self.failures -= 1
            raise NotifyError("ntfy down")
        self.sent.append(message)


class NoPrices:
    def bars(self, symbol, start, end):
        return MinuteBars([], [])


def seed(
    conn,
    native_id,
    created,
    tickers,
    *,
    platform="truthsocial",
    author="realDonaldTrump",
    text="Great news for NVIDIA",
    stance=("bullish", 0.91),
):
    p = Post(
        platform=platform,
        native_id=native_id,
        author=author,
        created_at_utc=created,
        text=text,
        url=f"https://example.com/{native_id}",
    )
    with conn:
        db.upsert_post(conn, p, created)
        db.replace_mentions(conn, platform, native_id, [Mention(t, "cashtag", f"${t}") for t in tickers], [], created)
        if stance:
            conn.execute(
                "UPDATE posts SET stance = ?, stance_conf = ?, stance_model = 'm' WHERE platform = ? AND native_id = ?",
                (*stance, platform, native_id),
            )


def make(conn, watchlist, notifier=None, history=None):
    return AlertEngine(
        conn,
        watchlist,
        notifier or Recorder(),
        live_factory=NoPrices,
        history_factory=lambda now: lambda author: history,
        model=lambda ticker, d0: None,
    )


def statuses(conn, kind):
    return {r["native_id"]: r["status"] for r in conn.execute("SELECT * FROM alerts WHERE kind = ?", (kind,))}


def test_first_run_marks_existing_posts_and_sends_nothing(conn, watchlist):
    seed(conn, "old1", T_POST - timedelta(days=100), ["NVDA"])
    seed(conn, "old2", T_POST - timedelta(days=5), ["AAPL", "INTC"])
    rec = Recorder()
    eng = make(conn, watchlist, rec)
    counts = eng.run(T_POST)
    assert rec.sent == []
    assert statuses(conn, "heads_up") == {"old1": "skipped", "old2": "skipped"}
    assert statuses(conn, "follow_60m") == {} and statuses(conn, "follow_d1") == {}
    assert counts["initialized"] == 2
    assert eng.run(T_POST + timedelta(minutes=5))["initialized"] == 0


def test_post_stored_before_alerts_began_stays_quiet_when_it_later_mentions_a_ticker(conn, watchlist):
    seed(conn, "old", T_POST - timedelta(days=3), [])
    eng = make(conn, watchlist, rec := Recorder())
    eng.run(T_POST - timedelta(minutes=10))
    with conn:  # the ticker list changed and every stored post was re-matched
        db.replace_mentions(conn, "truthsocial", "old", [Mention("NVDA", "cashtag", "$NVDA")], [], T_POST)
    eng.run(T_POST + timedelta(minutes=4))
    assert rec.sent == []
    assert statuses(conn, "heads_up") == {"old": "skipped"}
    assert statuses(conn, "follow_d1") == {} and statuses(conn, "digest") == {}


def test_fresh_post_gets_one_heads_up_and_queued_followups(conn, watchlist):
    eng = make(conn, watchlist, rec := Recorder(), history=History(71, -0.006, 0.5))
    eng.run(T_POST - timedelta(minutes=10))  # initialise on an empty history
    seed(conn, "p1", T_POST, ["NVDA"])
    eng.run(T_POST + timedelta(minutes=4))
    eng.run(T_POST + timedelta(minutes=9))
    assert [m.title for m in rec.sent] == ["NVDA · realDonaldTrump · bullish (0.91)"]
    assert "n=71, Holm p=0.50" in rec.sent[0].body
    assert statuses(conn, "heads_up") == {"p1": "sent"}
    due = {r["kind"]: r["due_at"] for r in conn.execute("SELECT kind, due_at FROM alerts WHERE native_id = 'p1'")}
    assert due["follow_60m"] == "2026-09-29T15:34:00Z"  # 11:31 ET + 3 min
    assert due["follow_d1"] == "2026-10-01T00:00:00Z"  # Wed Sep 30 20:00 ET


def test_ignores_reddit_and_benchmark_only_posts(conn, watchlist):
    eng = make(conn, watchlist, rec := Recorder())
    eng.run(T_POST - timedelta(minutes=10))
    seed(conn, "r1", T_POST, ["NVDA"], platform="reddit", author="someone")
    seed(conn, "b1", T_POST, ["SPY"])
    eng.run(T_POST + timedelta(minutes=4))
    assert rec.sent == []
    assert statuses(conn, "heads_up") == {}


def test_late_posts_go_to_one_digest_with_day_after_followups_only(conn, watchlist):
    eng = make(conn, watchlist, rec := Recorder())
    eng.run(datetime(2026, 9, 29, 1, 0, tzinfo=UTC))  # initialise the evening before
    night = datetime(2026, 9, 30, 2, 0, tzinfo=UTC)  # Tue 22:00 ET
    for i, minutes in enumerate((0, 45, 90)):
        seed(conn, f"n{i}", night + timedelta(minutes=minutes), ["NVDA", "INTC"])
    eng.run(datetime(2026, 9, 30, 11, 0, tzinfo=UTC))  # Wed 07:00 ET
    assert [m.title for m in rec.sent] == ["While you were away: 3 stock posts"]
    assert set(statuses(conn, "heads_up").values()) == {"skipped"}
    assert statuses(conn, "follow_60m") == {}
    assert set(statuses(conn, "follow_d1")) == {"n0", "n1", "n2"}
    assert list(statuses(conn, "digest").values()) == ["sent"]


def test_failed_send_is_retried_then_abandoned_after_24h(conn, watchlist):
    rec = Recorder(failures=1000)
    eng = make(conn, watchlist, rec)
    eng.run(T_POST - timedelta(minutes=10))
    seed(conn, "p1", T_POST, ["NVDA"])
    for k in range(1, 4):
        eng.run(T_POST + timedelta(minutes=5 * k))
    row = conn.execute("SELECT * FROM alerts WHERE kind = 'heads_up'").fetchone()
    assert row["status"] == "failed" and row["attempts"] == 3 and "ntfy down" in row["error"]
    eng.run(T_POST + timedelta(hours=25))
    assert conn.execute("SELECT attempts FROM alerts WHERE kind = 'heads_up'").fetchone()[0] == 3
    rec.failures = 0
    eng.run(T_POST + timedelta(hours=26))
    assert rec.sent == []


def test_recovered_ntfy_sends_exactly_once(conn, watchlist):
    rec = Recorder(failures=1)
    eng = make(conn, watchlist, rec)
    eng.run(T_POST - timedelta(minutes=10))
    seed(conn, "p1", T_POST, ["NVDA"])
    eng.run(T_POST + timedelta(minutes=4))
    eng.run(T_POST + timedelta(minutes=9))
    eng.run(T_POST + timedelta(minutes=14))
    assert len(rec.sent) == 1


def test_heads_up_price_comes_from_live_bars(conn, watchlist):
    class Prices:
        def bars(self, symbol, start, end):
            s = int(T_POST.timestamp())
            return MinuteBars([s - 120, s - 60], [180.0, 182.41])

    eng = AlertEngine(
        conn,
        watchlist,
        rec := Recorder(),
        live_factory=Prices,
        history_factory=lambda now: lambda a: None,
        model=lambda t, d: None,
    )
    eng.run(T_POST - timedelta(minutes=10))
    seed(conn, "p1", T_POST, ["NVDA"])
    eng.run(T_POST + timedelta(minutes=4))
    assert "NVDA $182.41" in rec.sent[0].body  # the bar starting 60 s before the post has finished by the post


def _group(group, subset, window, n_posts, mean, p_holm, family="author"):
    return {
        "family": family,
        "group": group,
        "subset": subset,
        "window": window,
        "n_posts": n_posts,
        "mean_signed_car": mean,
        "p_holm": p_holm,
    }


def test_study_history_reads_the_author_main_event_row_once(conn, watchlist, monkeypatch):
    groups = pd.DataFrame.from_records(
        [
            _group("realDonaldTrump", "confident", "event", 40, 0.02, 0.9),
            _group("realDonaldTrump", "main", "pre", 71, 0.03, 0.8),
            _group("realDonaldTrump", "main", "event", 71, -0.006, 0.17),
            _group("realDonaldTrump", "main", "event", 117, 0.01, 0.6, family="platform"),
            _group("WhiteHouse", "main", "event", 12, 0.01, float("nan")),
        ],
        columns=list(GROUP_COLUMNS),
    )
    calls = []

    def fake_study(c, w, now):
        calls.append(now)
        return SimpleNamespace(groups=groups)

    monkeypatch.setattr(engine, "compute_study", fake_study)
    history = StudyHistory(conn, watchlist, T_POST)
    assert history("realDonaldTrump") == History(71, -0.006, 0.17)
    assert history("WhiteHouse") == History(12, 0.01, None)
    assert history("PressSec") is None
    assert calls == [T_POST]


def test_study_history_failure_gives_none_without_recomputing(conn, watchlist, monkeypatch):
    calls = []

    def broken_study(c, w, now):
        calls.append(now)
        raise RuntimeError("no daily bars")

    monkeypatch.setattr(engine, "compute_study", broken_study)
    history = StudyHistory(conn, watchlist, T_POST)
    assert history("realDonaldTrump") is None
    assert history("WhiteHouse") is None
    assert calls == [T_POST]
