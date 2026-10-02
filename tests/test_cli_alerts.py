from __future__ import annotations

import dataclasses
import logging
import re
import shutil
from datetime import UTC, datetime, timedelta

import pytest
import yaml

from influence_tracker import cli, db, events, logsetup, sentiment
from influence_tracker.alerts import watch as watch_mod
from influence_tracker.alerts.engine import AlertEngine, StudyHistory
from influence_tracker.alerts.keepawake import KeepAwake
from influence_tracker.alerts.live import LivePrices
from influence_tracker.alerts.notify import DryRunNotifier, NotifyError, NtfyNotifier
from influence_tracker.collectors import truthsocial as truthsocial_mod
from influence_tracker.collectors import x as x_mod
from influence_tracker.config import REPO_ROOT, load_settings
from influence_tracker.timeutil import to_iso

NOW = datetime(2026, 9, 29, 15, 0, tzinfo=UTC)  # Tuesday 11:00 New York


@pytest.fixture
def root(tmp_path, monkeypatch, no_ntfy_env):
    (tmp_path / "config").mkdir()
    shutil.copy(REPO_ROOT / "config" / "watchlist.yaml", tmp_path / "config" / "watchlist.yaml")
    monkeypatch.setenv(cli.ROOT_ENV_VAR, str(tmp_path))
    monkeypatch.delenv("X_BEARER_TOKEN", raising=False)
    monkeypatch.setattr(cli, "utc_now", lambda: NOW)
    return tmp_path


@pytest.fixture(autouse=True)
def _restore_logging():
    root_logger = logging.getLogger()
    level = root_logger.level
    yield
    logsetup.remove_handlers()
    root_logger.setLevel(level)


def _runs(root) -> list[tuple[str, str, str | None]]:
    conn = db.connect(root / "data" / "tracker.db")
    try:
        rows = conn.execute("SELECT stage, status, error, finished_at FROM runs ORDER BY id").fetchall()
    finally:
        conn.close()
    assert all(r["finished_at"] for r in rows)
    return [(r["stage"], r["status"], r["error"]) for r in rows]


# ---------------------------------------------------------------- alerts setup | test


def test_alerts_setup_writes_a_topic_once(root, capsys):
    assert cli.main(["alerts", "setup"]) == 0
    env = (root / ".env").read_text(encoding="utf-8")
    [line] = [x for x in env.splitlines() if x.startswith("NTFY_TOPIC=")]
    topic = line.split("=", 1)[1]
    # A 32-character topic, in the characters ntfy allows (letters, digits, "_" and "-").
    assert re.fullmatch(r"influence-[A-Za-z0-9_-]{22}", topic)
    assert cli.main(["alerts", "setup"]) == 0
    assert (root / ".env").read_text(encoding="utf-8") == env  # never overwritten
    out = capsys.readouterr().out
    assert "subscribe" in out.lower() and topic in out


def test_alerts_setup_keeps_the_rest_of_the_env_file(root):
    (root / ".env").write_text("# X API bearer token\nNTFY_TOPIC=\n", encoding="utf-8")
    assert cli.main(["alerts", "setup"]) == 0
    lines = (root / ".env").read_text(encoding="utf-8").splitlines()
    assert lines[0] == "# X API bearer token"
    assert [x for x in lines if x.startswith("NTFY_TOPIC=")] == [lines[-1]]
    assert lines[-1].startswith("NTFY_TOPIC=influence-")


def test_alerts_test_needs_a_topic(root, capsys):
    assert cli.main(["alerts", "test"]) == 2
    assert "influence alerts setup" in capsys.readouterr().err


def test_alerts_test_sends_one_message(root, monkeypatch):
    monkeypatch.setenv("NTFY_TOPIC", "influence-test123")
    sent = []
    monkeypatch.setattr(cli, "_make_notifier", lambda settings, dry_run: type("N", (), {"send": sent.append})())
    assert cli.main(["alerts", "test"]) == 0
    assert len(sent) == 1 and "test" in sent[0].title.lower()


def test_alerts_test_reports_a_failed_send(root, monkeypatch, capsys):
    monkeypatch.setenv("NTFY_TOPIC", "influence-test123")

    class Down:
        def send(self, message):
            raise NotifyError("HTTP 503: unavailable")

    monkeypatch.setattr(cli, "_make_notifier", lambda settings, dry_run: Down())
    assert cli.main(["alerts", "test"]) == 1
    assert "HTTP 503" in capsys.readouterr().err


def test_make_notifier_publishes_to_the_configured_topic(tmp_path, no_ntfy_env):
    settings = dataclasses.replace(
        load_settings(tmp_path), ntfy_topic="influence-abc", ntfy_server="https://ntfy.example"
    )
    notifier = cli._make_notifier(settings, dry_run=False)
    try:
        assert isinstance(notifier, NtfyNotifier)
        assert (notifier.url, notifier.topic) == ("https://ntfy.example/", "influence-abc")
    finally:
        notifier.client.close()
    assert isinstance(cli._make_notifier(settings, dry_run=True), DryRunNotifier)


# ---------------------------------------------------------------- watch


def test_watch_once_dry_run_wires_every_step(root, monkeypatch):
    called, ran = [], []
    monkeypatch.setattr(cli, "_watch_steps", lambda *a, **k: called.append(k) or "steps")
    monkeypatch.setattr(
        watch_mod,
        "run_watch",
        lambda conn, wl, steps, run_id, once, max_cycles=None: ran.append((steps, once)) or 0,
    )
    assert cli.main(["watch", "--once", "--dry-run"]) == 0
    assert called and called[0]["dry_run"] is True
    assert ran == [("steps", True)]
    assert _runs(root) == [("watch", "ok", None)]


def test_watch_without_a_topic_stops_before_starting(root, monkeypatch, capsys):
    built = []
    monkeypatch.setattr(cli, "_watch_steps", lambda *a, **k: built.append(k))
    assert cli.main(["watch", "--once"]) == 2
    assert "NTFY_TOPIC is not set: run `influence alerts setup` first" in capsys.readouterr().err
    assert built == [] and _runs(root) == []


def test_watch_refuses_when_alerts_are_disabled(root, capsys):
    path = root / "config" / "watchlist.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw["alerts"]["enabled"] = False
    path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    assert cli.main(["watch", "--dry-run"]) == 2
    assert "alerts.enabled is false" in capsys.readouterr().err
    assert _runs(root) == []


@pytest.mark.parametrize(
    ("raised", "code", "row"),
    [
        (KeyboardInterrupt(), 0, ("watch", "ok", None)),
        (RuntimeError("boom"), 1, ("watch", "error", "RuntimeError: boom")),
    ],
    ids=["ctrl-c", "crash"],
)
def test_watch_run_row_records_how_it_ended(root, monkeypatch, raised, code, row):
    def run_watch(*args, **kwargs):
        raise raised

    monkeypatch.setattr(cli, "_watch_steps", lambda *a, **k: "steps")
    monkeypatch.setattr(watch_mod, "run_watch", run_watch)
    assert cli.main(["watch", "--dry-run"]) == code
    assert _runs(root) == [row]


def test_watch_steps_wire_the_real_stages(tmp_path, conn, watchlist, monkeypatch, no_ntfy_env):
    calls, loads = [], []
    sink = object()
    prepared = iter([(None, RuntimeError("database is locked")), (sink, None)])
    monkeypatch.setattr(cli.Pipeline, "_prepare_sink", lambda self: next(prepared))
    monkeypatch.setattr(truthsocial_mod, "collect_truthsocial", lambda *a: calls.append(("ts", a)) or {})
    monkeypatch.setattr(x_mod, "collect_x", lambda *a: calls.append(("x", a)) or {})
    monkeypatch.setattr(events, "sync_events", lambda *a: calls.append(("events", a)) or {})

    def load_classifier(model_id, batch_size):
        loads.append((model_id, batch_size))
        return lambda texts: [("neutral", 0.9)] * len(texts)

    def classify_posts(conn_, wl, run_id, now, classifier):
        calls.append(("classify", (conn_, wl, run_id, now)))
        return {"labels": classifier(["a"]) + classifier(["b"])}

    monkeypatch.setattr(sentiment, "load_classifier", load_classifier)
    monkeypatch.setattr(sentiment, "classify_posts", classify_posts)
    settings = dataclasses.replace(load_settings(tmp_path), x_bearer_token="tok")

    steps = cli._watch_steps(settings, watchlist, conn, cli.utc_now, dry_run=True)

    with pytest.raises(RuntimeError, match="database is locked"):
        steps.collect_truthsocial(7, NOW)
    steps.collect_truthsocial(7, NOW)  # a failed collect setup is retried on the next cycle
    steps.collect_x(7, NOW)
    assert steps.classify(7, NOW) == {"labels": [("neutral", 0.9), ("neutral", 0.9)]}
    steps.sync_events(NOW)
    assert calls == [
        ("ts", (conn, watchlist, sink, 7, NOW)),
        ("x", (conn, watchlist, "tok", sink, 7, NOW)),
        ("classify", (conn, watchlist, 7, NOW)),
        ("events", (conn, watchlist, NOW)),
    ]
    assert loads == [(watchlist.sentiment.model_id, watchlist.sentiment.batch_size)]  # loaded once, kept in memory
    engine = steps.alerts.__self__
    assert isinstance(engine, AlertEngine) and isinstance(engine.notifier, DryRunNotifier)
    assert isinstance(engine.live_factory(), LivePrices) and isinstance(engine.history_factory(NOW), StudyHistory)
    assert steps.clock is cli.utc_now and isinstance(steps.keep_awake, KeepAwake)

    no_token = dataclasses.replace(settings, x_bearer_token=None)
    assert cli._watch_steps(no_token, watchlist, conn, cli.utc_now, dry_run=True).collect_x is None


# ---------------------------------------------------------------- the nightly run steps aside


@pytest.mark.parametrize(
    ("heartbeat_age", "expected"),
    [
        (
            timedelta(minutes=14),
            {"truthsocial": ("skipped", "live watch is running"), "x": ("skipped", "live watch is running")},
        ),
        (
            timedelta(minutes=15),
            {"truthsocial": ("ok", None), "x": ("skipped", "X_BEARER_TOKEN not set in .env")},
        ),
    ],
    ids=["watch-alive", "watch-stale"],
)
def test_collect_steps_aside_while_watch_is_alive(root, monkeypatch, heartbeat_age, expected):
    conn = db.connect(root / "data" / "tracker.db")
    with conn:
        db.set_watermark(conn, "watch", "heartbeat", to_iso(NOW - heartbeat_age), NOW)
    conn.close()
    ran = []
    monkeypatch.setattr(
        cli.Pipeline,
        "_collector",
        lambda self, n, sink, err: lambda run_id, now: ran.append(n) or {"status": "ok"},
    )
    assert cli.main(["collect"]) == 0
    rows = {stage.removeprefix("collect:"): (status, error) for stage, status, error in _runs(root)}
    assert {name: rows[name] for name in ("truthsocial", "x")} == expected
    assert "reddit" in ran and ("truthsocial" in ran) == (expected["truthsocial"][0] == "ok")
