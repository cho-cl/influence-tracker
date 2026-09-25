from __future__ import annotations

import io
import logging
import os
import re
import shutil
import subprocess
import sys
import types
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest
from rich.console import Console

from influence_tracker import cli, db, ingest, logsetup, status
from influence_tracker.config import REPO_ROOT
from influence_tracker.models import MatchResult, Mention, Post

SCRIPTS = REPO_ROOT / "scripts"
_DEFAULT = object()


class FakeMatcher:
    """Cashtag-only matcher over the configured universe, standing in for mentions.MentionMatcher."""

    def __init__(self, watchlist):
        self.symbols = {t.symbol for t in watchlist.tickers}

    def match(self, text, platform, cashtag_hints=()):
        tags = [w[1:] for w in text.split() if w.startswith("$") and w[1:].isalpha()]
        return MatchResult(
            mentions=[Mention(s, "cashtag", f"${s}") for s in tags if s in self.symbols],
            unknown_cashtags=[s for s in tags if s not in self.symbols],
        )


class Recorder:
    def __init__(self):
        self.calls: list[tuple[str, tuple]] = []
        self.outcomes: dict[str, object] = {}
        self.budget: dict = {}

    def fake(self, name):
        def collector(*args):
            self.calls.append((name, args))
            outcome = self.outcomes.get(name, _DEFAULT)
            if outcome is _DEFAULT:
                return {"status": "ok", "fetched": 2, "stored": 1}
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome

        return collector

    @property
    def order(self) -> list[str]:
        return [name for name, _ in self.calls]

    def args(self, name) -> tuple:
        return next(a for n, a in self.calls if n == name)


def _install(monkeypatch, name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


@pytest.fixture
def fakes(monkeypatch) -> Recorder:
    rec = Recorder()
    _install(
        monkeypatch,
        "influence_tracker.collectors.truthsocial",
        collect_truthsocial=rec.fake("truthsocial"),
        backfill_truthsocial=rec.fake("backfill"),
    )
    _install(monkeypatch, "influence_tracker.collectors.reddit_rss", collect_reddit=rec.fake("reddit"))
    _install(monkeypatch, "influence_tracker.collectors.apewisdom", collect_apewisdom=rec.fake("apewisdom"))
    _install(
        monkeypatch,
        "influence_tracker.collectors.x",
        collect_x=rec.fake("x"),
        x_budget_summary=lambda conn, watchlist, now: rec.budget,
    )
    _install(monkeypatch, "influence_tracker.prices", snapshot_1m=rec.fake("snapshot"))
    _install(monkeypatch, "influence_tracker.mentions", MentionMatcher=FakeMatcher)
    return rec


@pytest.fixture
def root(tmp_path, monkeypatch, fixed_now) -> Path:
    (tmp_path / "config").mkdir()
    shutil.copy(REPO_ROOT / "config" / "watchlist.yaml", tmp_path / "config" / "watchlist.yaml")
    monkeypatch.setenv("INFLUENCE_TRACKER_ROOT", str(tmp_path))
    monkeypatch.delenv("X_BEARER_TOKEN", raising=False)
    monkeypatch.setattr(cli, "utc_now", lambda: fixed_now)
    return tmp_path


@pytest.fixture(autouse=True)
def _restore_logging():
    root_logger = logging.getLogger()
    level = root_logger.level
    yield
    logsetup.remove_handlers()
    root_logger.setLevel(level)


def _runs(root: Path) -> list[tuple[str, str, str | None]]:
    conn = db.connect(root / "data" / "tracker.db")
    try:
        rows = conn.execute("SELECT stage, status, error, finished_at FROM runs ORDER BY id").fetchall()
    finally:
        conn.close()
    assert all(r["finished_at"] for r in rows)
    return [(r["stage"], r["status"], r["error"]) for r in rows]


def _query(root: Path, sql: str, params: tuple = ()) -> list:
    conn = db.connect(root / "data" / "tracker.db")
    try:
        return [tuple(r) for r in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()


# ---------------------------------------------------------------- collect / daily


def test_collect_runs_stages_in_order_and_skips_x_without_token(root, fakes, fixed_now, watchlist):
    assert cli.main(["collect"]) == 0

    assert fakes.order == ["truthsocial", "reddit", "apewisdom"]
    assert _runs(root) == [
        ("collect:truthsocial", "ok", None),
        ("collect:reddit", "ok", None),
        ("collect:apewisdom", "ok", None),
        ("collect:x", "skipped", "X_BEARER_TOKEN not set in .env"),
    ]
    conn_arg, watchlist_arg, sink_arg, run_id, now = fakes.args("truthsocial")
    assert isinstance(sink_arg, ingest.PostSink)
    assert isinstance(run_id, int)
    assert now == fixed_now
    assert fakes.args("apewisdom")[2:] == (3, fixed_now)
    # INFLUENCE_TRACKER_ROOT decides where the database and log live.
    assert (root / "data" / "tracker.db").exists()
    assert (root / "logs" / "influence.log").exists()
    assert _query(root, "SELECT platform, COUNT(*) FROM accounts GROUP BY platform ORDER BY platform") == [
        ("truthsocial", len(watchlist.truthsocial.accounts)),
        ("x", len(watchlist.x.accounts)),
    ]


def test_x_runs_last_with_its_token_and_the_token_is_never_logged(root, fakes, monkeypatch):
    monkeypatch.setenv("X_BEARER_TOKEN", "secret-bearer-123")

    assert cli.main(["collect"]) == 0

    assert fakes.order == ["truthsocial", "reddit", "apewisdom", "x"]
    _, _, token, sink, _, _ = fakes.args("x")
    assert token == "secret-bearer-123"
    assert isinstance(sink, ingest.PostSink)
    assert _runs(root)[-1] == ("collect:x", "ok", None)
    assert "secret-bearer-123" not in (root / "logs" / "influence.log").read_text(encoding="utf-8")


def test_a_raising_stage_does_not_stop_later_stages(root, fakes):
    fakes.outcomes["reddit"] = RuntimeError("feed exploded")
    fakes.outcomes["truthsocial"] = {"status": "partial", "rate_limited": 1, "error": "429 twice; giving up"}

    assert cli.main(["daily"]) == 1

    assert fakes.order == ["truthsocial", "reddit", "apewisdom", "snapshot"]
    assert _runs(root) == [
        ("collect:truthsocial", "partial", "429 twice; giving up"),
        ("collect:reddit", "error", "RuntimeError: feed exploded"),
        ("collect:apewisdom", "ok", None),
        ("collect:x", "skipped", "X_BEARER_TOKEN not set in .env"),
        ("snapshot", "ok", None),
    ]
    log_text = (root / "logs" / "influence.log").read_text(encoding="utf-8")
    assert "Traceback" in log_text and "feed exploded" in log_text


def test_uncommitted_writes_of_a_failed_stage_are_rolled_back(root, fakes, monkeypatch, fixed_now):
    def half_done(conn, watchlist, sink, run_id, now):
        db.set_watermark(conn, "reddit", "stocks", "t3_abc", now)
        raise ConnectionError("dropped mid-page")

    monkeypatch.setattr(sys.modules["influence_tracker.collectors.reddit_rss"], "collect_reddit", half_done)

    assert cli.main(["collect", "--only", "reddit"]) == 1

    assert _query(root, "SELECT * FROM watermarks WHERE source = 'reddit'") == []
    assert _runs(root) == [("collect:reddit", "error", "ConnectionError: dropped mid-page")]


@pytest.mark.parametrize(
    ("outcome", "expected_status", "expected_error", "expected_rc"),
    [
        ({"status": "ok", "stored": 3}, "ok", None, 0),
        (
            {"status": "partial", "errors": ["page 2 timed out", "page 3 timed out"]},
            "partial",
            "page 2 timed out; page 3 timed out",
            0,
        ),
        ({"status": "error", "error": "HTTP 503"}, "error", "HTTP 503", 1),
        ({"stored": 3}, "error", "stage returned invalid status None", 1),
        ("not a dict", "error", "stage returned str, expected a counts dict", 1),
    ],
)
def test_stage_status_comes_from_the_counts_dict(root, fakes, outcome, expected_status, expected_error, expected_rc):
    fakes.outcomes["apewisdom"] = outcome

    assert cli.main(["collect", "--only", "apewisdom"]) == expected_rc

    assert _runs(root) == [("collect:apewisdom", expected_status, expected_error)]


def test_counts_that_are_not_json_are_still_recorded(root, fakes, fixed_now):
    fakes.outcomes["reddit"] = {"status": "ok", "newest": fixed_now, "subs": {"stocks": 4}}

    assert cli.main(["collect", "--only", "reddit"]) == 0

    [(counts_json,)] = _query(root, "SELECT counts_json FROM runs")
    assert '"newest": "2026-09-24 22:30:00+00:00"' in counts_json
    assert '"subs": {"stocks": 4}' in counts_json


def test_only_filter_runs_the_named_collectors_in_canonical_order(root, fakes, monkeypatch):
    monkeypatch.setenv("X_BEARER_TOKEN", "tok")

    assert cli.main(["collect", "--only", "x", "reddit"]) == 0

    assert fakes.order == ["reddit", "x"]
    assert [stage for stage, _, _ in _runs(root)] == ["collect:reddit", "collect:x"]


def test_only_x_without_token_is_skipped_not_failed(root, fakes):
    assert cli.main(["collect", "--only", "x"]) == 0
    assert fakes.order == []
    assert _runs(root) == [("collect:x", "skipped", "X_BEARER_TOKEN not set in .env")]


@pytest.mark.parametrize("argv", [["collect", "--only", "facebook"], ["collect", "--only"], [], ["nonsense"]])
def test_bad_arguments_exit_2_without_running_anything(root, fakes, argv):
    assert cli.main(argv) == 2
    assert fakes.order == []


def test_help_exits_0(capsys):
    assert cli.main(["--help"]) == 0
    assert "backfill" in capsys.readouterr().out


def test_daily_runs_snapshot_after_collect(root, fakes, fixed_now):
    assert cli.main(["daily"]) == 0

    assert fakes.order == ["truthsocial", "reddit", "apewisdom", "snapshot"]
    assert [stage for stage, _, _ in _runs(root)] == [
        "collect:truthsocial",
        "collect:reddit",
        "collect:apewisdom",
        "collect:x",
        "snapshot",
    ]
    _, _, run_id, now = fakes.args("snapshot")
    assert run_id == 5
    assert now == fixed_now


def test_snapshot_alone(root, fakes):
    assert cli.main(["snapshot"]) == 0
    assert fakes.order == ["snapshot"]
    assert _runs(root) == [("snapshot", "ok", None)]


def test_missing_collector_module_fails_only_its_stage(root, fakes, monkeypatch):
    monkeypatch.setitem(sys.modules, "influence_tracker.collectors.reddit_rss", None)

    assert cli.main(["daily"]) == 1

    assert fakes.order == ["truthsocial", "apewisdom", "snapshot"]
    stage, state, error = _runs(root)[1]
    assert (stage, state) == ("collect:reddit", "error")
    assert error.startswith("ModuleNotFoundError")


def test_collect_setup_failure_fails_post_stages_but_not_apewisdom_or_snapshot(root, fakes, monkeypatch):
    def broken_matcher(watchlist):
        raise ValueError("bad regex")

    monkeypatch.setattr(sys.modules["influence_tracker.mentions"], "MentionMatcher", broken_matcher)

    assert cli.main(["daily"]) == 1

    assert fakes.order == ["apewisdom", "snapshot"]
    runs = _runs(root)
    assert [(s, st) for s, st, _ in runs] == [
        ("collect:truthsocial", "error"),
        ("collect:reddit", "error"),
        ("collect:apewisdom", "ok"),
        ("collect:x", "skipped"),
        ("snapshot", "ok"),
    ]
    assert "collect setup failed" in runs[0][2] and "bad regex" in runs[0][2]


def test_posts_stored_through_the_sink_are_matched_and_rematched_when_the_universe_changes(
    root, fakes, monkeypatch, fixed_now
):
    def store_one(conn, watchlist, sink, run_id, now):
        post = Post("truthsocial", "1001", "realDonaldTrump", now - timedelta(hours=1), "$TSLA and $ZZZZ 🚀", "u")
        result = sink.store([post])
        return {"status": "ok", "stored": result.stored, "new": result.new}

    monkeypatch.setattr(sys.modules["influence_tracker.collectors.truthsocial"], "collect_truthsocial", store_one)

    assert cli.main(["collect", "--only", "truthsocial"]) == 0
    assert _query(root, "SELECT ticker FROM mentions") == [("TSLA",)]
    assert _query(root, "SELECT symbol FROM unknown_cashtags") == [("ZZZZ",)]

    config = root / "config" / "watchlist.yaml"
    config.write_text(config.read_text(encoding="utf-8") + "  - { symbol: ZZZZ }\n", encoding="utf-8")

    assert cli.main(["collect", "--only", "apewisdom"]) == 0
    assert _query(root, "SELECT ticker FROM mentions ORDER BY ticker") == [("TSLA",), ("ZZZZ",)]
    assert _query(root, "SELECT symbol FROM unknown_cashtags") == []
    assert "re-matched mentions for 1 stored post" in (root / "logs" / "influence.log").read_text(encoding="utf-8")


# ---------------------------------------------------------------- backfill


def test_backfill_truthsocial_passes_the_since_date(root, fakes, fixed_now):
    assert cli.main(["backfill", "truthsocial", "--since", "2025-01-20"]) == 0

    assert fakes.order == ["backfill"]
    _, _, sink, run_id, since, now = fakes.args("backfill")
    assert isinstance(sink, ingest.PostSink)
    assert since == date(2025, 1, 20)
    assert now == fixed_now
    assert _runs(root) == [("backfill:truthsocial", "ok", None)]


def test_backfill_failure_exits_1(root, fakes):
    fakes.outcomes["backfill"] = TimeoutError("read timed out")
    assert cli.main(["backfill", "truthsocial", "--since", "2025-01-20"]) == 1
    assert _runs(root) == [("backfill:truthsocial", "error", "TimeoutError: read timed out")]


@pytest.mark.parametrize(
    "argv",
    [
        ["backfill", "truthsocial"],
        ["backfill", "truthsocial", "--since", "2025-13-01"],
        ["backfill", "truthsocial", "--since", "20250120"],
        ["backfill", "truthsocial", "--since", "yesterday"],
        ["backfill", "reddit", "--since", "2025-01-20"],
        ["backfill", "--since", "2025-01-20"],
        ["backfill", "truthsocial", "--since", "2026-09-25"],
    ],
)
def test_backfill_rejects_bad_arguments(root, fakes, argv):
    assert cli.main(argv) == 2
    assert fakes.order == []


# ---------------------------------------------------------------- config errors


def test_bad_yaml_exits_2_with_a_friendly_message(root, fakes, capsys):
    (root / "config" / "watchlist.yaml").write_text("x: [unclosed\n", encoding="utf-8")

    assert cli.main(["daily"]) == 2

    err = capsys.readouterr().err
    assert "Config problem in" in err and "watchlist.yaml" in err
    assert "line" in err
    assert fakes.order == []


@pytest.mark.parametrize(
    ("old", "new", "expected"),
    [
        ("- { symbol: SPY, benchmark: true }", "", r"\(top level\): Value error, SPY must be in tickers"),
        ("category: exec }", "category: ceo }", r"x\.accounts\.\d+\.category: Input should be"),
    ],
)
def test_invalid_watchlist_exits_2_and_names_the_problem(root, fakes, capsys, old, new, expected):
    config = root / "config" / "watchlist.yaml"
    config.write_text(config.read_text(encoding="utf-8").replace(old, new, 1), encoding="utf-8")

    assert cli.main(["collect"]) == 2

    assert re.search(expected, capsys.readouterr().err)
    assert fakes.order == []


def test_missing_watchlist_exits_2(root, capsys):
    (root / "config" / "watchlist.yaml").unlink()
    assert cli.main(["status"]) == 2
    assert "Config problem in" in capsys.readouterr().err


# ---------------------------------------------------------------- status


def test_status_works_on_an_empty_database(root, capsys, watchlist):
    assert cli.main(["status"]) == 0

    out = capsys.readouterr().out
    empty = ("never", "no X usage yet", "no mentions yet", "no unknown cashtags yet", "no snapshots yet")
    for phrase in (*empty, f"0 of {len(watchlist.tickers)} symbols"):
        assert phrase in out
    assert _query(root, "SELECT COUNT(*) FROM runs") == [(0,)]


def _render(conn, watchlist, now) -> str:
    console = Console(file=io.StringIO(), width=240, color_system=None)
    status.show_status(conn, watchlist, now, console)
    return console.file.getvalue()


def _post(platform, native_id, text, when, source=None) -> Post:
    return Post(platform, native_id, "someone", when, text, f"https://example.com/{native_id}", source=source)


def test_status_reports_the_database_contents(conn, watchlist, fixed_now, fakes):
    now = fixed_now
    ingest.sync_accounts(conn, watchlist)
    with conn:
        db.set_account_resolution(conn, "truthsocial", "realDonaldTrump", "107780257626128497", None, now)
        db.touch_account_fetch(conn, "truthsocial", "realDonaldTrump", now - timedelta(hours=2))
        db.set_account_resolution(conn, "x", "howardlutnick", None, "user not found", now)
    sink = ingest.PostSink(conn, FakeMatcher(watchlist), now=lambda: now)
    sink.store(
        [
            _post("truthsocial", "t1", "Buy $INTC now", now - timedelta(days=1)),
            _post("truthsocial", "t2", "$INTC $ZZZZ great", now - timedelta(hours=3)),
            _post("reddit", "r1", "$GME $ZZZZ $QQQQ", now - timedelta(hours=5), source="wallstreetbets"),
            _post("x", "x1", "no tickers here", now - timedelta(hours=6)),
        ]
    )

    ts_run = db.start_run(conn, "collect:truthsocial", now - timedelta(hours=1))
    db.finish_run(conn, ts_run, "partial", {"status": "partial", "rate_limited": 2, "stored": 2}, "429", now)
    bf_run = db.start_run(conn, "backfill:truthsocial", now - timedelta(days=2))
    db.finish_run(conn, bf_run, "ok", {"status": "ok", "rate_limited": 1}, None, now)
    old_run = db.start_run(conn, "collect:truthsocial", now - timedelta(days=10))
    db.finish_run(conn, old_run, "ok", {"status": "ok", "rate_limited": 50}, None, now)
    db.start_run(conn, "snapshot", now - timedelta(minutes=5))

    with conn:
        db.record_x_usage(conn, None, now, "jimcramer", "post_read", 40, 0.2)
        db.mark_session(conn, "SPY", "2026-09-23", 960, now)
        db.mark_session(conn, "SPY", "2026-09-24", 960, now)
        db.mark_session(conn, "TSLA", "2026-09-23", 950, now)
        db.mark_session(conn, "BRK.B", "2026-09-24", 900, now)
        db.mark_session(conn, "AAPL", "2026-09-24", 0, now)
    fakes.budget = {
        "cycle_start": datetime(2026, 9, 1, tzinfo=UTC),
        "budget_usd": 10.0,
        "spent_usd": 0.2,
        "remaining_usd": 9.8,
        "per_handle": [
            {"handle": "jimcramer", "post_reads": 40, "user_reads": 0, "cost_usd": 0.2},
            {"handle": None, "post_reads": 0, "user_reads": 17, "cost_usd": 0.17},
        ],
        "daily_allowance_posts": 66,
    }

    out = _render(conn, watchlist, now)

    # last runs
    assert "partial" in out and "rate_limited=2" in out
    assert "running?" in out
    # posts and mentions
    assert "posts with >=1 mention: 3 of 4" in out
    top = out[out.index("Top mentioned tickers") :]
    assert top.index("INTC") < top.index("GME")
    # accounts
    assert "FAILED: user not found" in out
    assert "r/wallstreetbets" in out
    assert "1 account failed to resolve: x/howardlutnick" in out
    # X spend: the cycle starts at 00:00 UTC on the billing day, so no shift into New York's previous evening
    assert "billing cycle from 2026-09-01:" in out
    assert "$0.20 of $10.00" in out and "$9.80 left" in out and "66 posts/day" in out
    assert "jimcramer" in out and "(lookups)" in out
    # Truth Social rate limits: only the last 7 days count
    assert "Truth Social rate-limit hits (last 7 days): 3 in 2 of 2 runs" in out
    # bars: sessions are keyed by watchlist symbol (BRK.B); AAPL has only an empty session
    assert f"3 of {len(watchlist.tickers)} symbols" in out
    assert "2026-09-23 to 2026-09-24" in out
    no_bars = out[out.index("no bars:") : out.index("Unknown cashtags")]
    assert "AAPL" in no_bars and "BRK.B" not in no_bars and "SPY" not in no_bars
    assert "behind the latest session: TSLA" in out
    # unknown cashtags, most posts first
    unknown = out[out.index("Unknown cashtags") :]
    assert unknown.index("ZZZZ") < unknown.index("QQQQ")


def test_status_says_no_x_usage_when_the_x_module_is_missing(conn, watchlist, fixed_now, monkeypatch):
    monkeypatch.setitem(sys.modules, "influence_tracker.collectors.x", None)
    with conn:
        db.record_x_usage(conn, None, fixed_now, "elonmusk", "post_read", 1, 0.005)
    assert "no X usage yet" in _render(conn, watchlist, fixed_now)


def test_status_survives_a_failing_budget_summary(conn, watchlist, fixed_now, monkeypatch):
    def boom(conn, watchlist, now):
        raise KeyError("cycle")

    _install(monkeypatch, "influence_tracker.collectors.x", x_budget_summary=boom)
    with conn:
        db.record_x_usage(conn, None, fixed_now, "elonmusk", "post_read", 1, 0.005)
    out = _render(conn, watchlist, fixed_now)
    assert "X budget summary failed" in out
    assert "Unknown cashtags" in out


def test_status_prints_bracketed_error_text_literally(conn, watchlist, fixed_now):
    ingest.sync_accounts(conn, watchlist)
    with conn:
        db.set_account_resolution(conn, "x", "elonmusk", None, "HTTP 403 [/forbidden]", fixed_now)
    run_id = db.start_run(conn, "collect:reddit", fixed_now)
    db.finish_run(conn, run_id, "error", {"status": "error", "note": "[bold]raw"}, "[/x] [Errno 11001]", fixed_now)

    out = _render(conn, watchlist, fixed_now)

    assert "[/x] [Errno 11001]" in out
    assert "note=[bold]raw" in out
    assert "FAILED: HTTP 403 [/forbidden]" in out


def test_status_hides_unknown_cashtags_that_are_now_in_the_universe(conn, watchlist, fixed_now):
    with conn:
        db.upsert_post(conn, _post("reddit", "r1", "x", fixed_now), fixed_now)
        db.replace_mentions(conn, "reddit", "r1", [], ["TSLA", "ZZZZ"], fixed_now)
    unknown = _render(conn, watchlist, fixed_now).split("Unknown cashtags", 1)[1]
    assert "ZZZZ" in unknown and "TSLA" not in unknown


def test_format_counts():
    counts = {
        "status": "ok",
        "error": "x",
        "stored": 3,
        "subs": {"stocks": 2, "options": 1},
        "ids": [1, 2],
        "r": 0.12345,
    }
    assert status.format_counts(counts) == "stored=3 subs.stocks=2 subs.options=1 ids=[2] r=0.1235"
    assert status.format_counts(counts, max_items=2) == "stored=3 subs.stocks=2 +3 more"
    assert status.format_counts(None) == ""


# ---------------------------------------------------------------- logging


def test_setup_logging_is_idempotent_and_writes_utf8(tmp_path):
    logsetup.setup_logging(tmp_path / "logs")
    logsetup.setup_logging(tmp_path / "logs")
    ours = [h for h in logging.getLogger().handlers if getattr(h, logsetup.HANDLER_MARK, False)]
    assert len(ours) == 2

    logging.getLogger("influence_tracker.test").info("rocket 🚀 posted")
    for handler in ours:
        handler.flush()
    assert "rocket 🚀 posted" in (tmp_path / "logs" / "influence.log").read_text(encoding="utf-8")


def test_redirected_consoles_do_not_wrap_at_80_columns(monkeypatch):
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    monkeypatch.setattr(sys, "stderr", io.StringIO())
    assert logsetup.make_console().width == 160
    assert logsetup.make_console(stderr=True).width == 160


def test_force_utf8_stdio_makes_a_cp1252_stream_accept_emoji(monkeypatch):
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="cp1252")
    monkeypatch.setattr(sys, "stdout", stream)
    monkeypatch.setattr(sys, "stderr", io.StringIO())

    logsetup.force_utf8_stdio()
    print("to the moon 🚀")
    stream.flush()

    assert "to the moon 🚀".encode() in raw.getvalue()


# ---------------------------------------------------------------- Windows scripts


@pytest.mark.skipif(sys.platform != "win32", reason="Windows batch script")
def test_run_daily_cmd_logs_output_and_propagates_the_exit_code(tmp_path):
    installed = REPO_ROOT / ".venv" / "Scripts" / "influence.exe"
    if not installed.exists():
        pytest.skip("the influence console script is not installed in .venv")
    repo = tmp_path / "my repo (copy) & co"
    (repo / "scripts").mkdir(parents=True)
    (repo / "config").mkdir()
    runner = repo / "scripts" / "run_daily.cmd"
    shutil.copy(SCRIPTS / "run_daily.cmd", runner)
    (repo / ".venv" / "Scripts").mkdir(parents=True)
    # The launcher embeds an absolute interpreter path, so a copy still runs this project's CLI.
    shutil.copy(installed, repo / ".venv" / "Scripts" / "influence.exe")
    # A broken config makes `influence daily` exit 2 before any collector touches the network.
    (repo / "config" / "watchlist.yaml").write_text("x: [unclosed\n", encoding="utf-8")

    # The exact command line register_task.ps1 gives the task, started from elsewhere (Task Scheduler: System32).
    proc = subprocess.run(
        f'cmd.exe /c ""{runner}""',
        cwd=Path.home(),
        env={**os.environ, "INFLUENCE_TRACKER_ROOT": str(repo)},
        stdin=subprocess.DEVNULL,
        capture_output=True,
        timeout=120,
    )

    assert proc.returncode == 2, proc.stdout + proc.stderr
    log_text = (repo / "logs" / "scheduled.log").read_text(encoding="utf-8")
    _, rest = log_text.split("influence daily =====", 1)
    exe_output, footer, _ = rest.rsplit("=====", 2)
    assert "Config problem in" in exe_output and "watchlist.yaml" in exe_output
    assert "exit 2" in footer
    assert (repo / "logs" / "influence.log").exists()
    assert '\'/c ""{0}""\' -f $runner' in (SCRIPTS / "register_task.ps1").read_text(encoding="utf-8")


@pytest.mark.skipif(shutil.which("powershell") is None, reason="needs Windows PowerShell")
def test_register_task_script_parses():
    script = SCRIPTS / "register_task.ps1"
    command = (
        "$errs = $null; $null = [System.Management.Automation.Language.Parser]::ParseFile("
        f"'{script}', [ref]$null, [ref]$errs); if ($errs) {{ $errs | ForEach-Object {{ $_.ToString() }}; exit 1 }}"
    )
    proc = subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert script.read_bytes().isascii()


# Never registered: every register_task.ps1 run below fails on -At, or returns early, before registration.
_UNUSED_TASK = "InfluenceTracker test (never registered)"


def _powershell(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", *args],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=120,
    )


@pytest.mark.skipif(shutil.which("powershell") is None, reason="needs Windows PowerShell")
def test_register_task_parses_at_and_builds_a_trigger_that_follows_daylight_saving(tmp_path):
    script = str(SCRIPTS / "register_task.ps1").replace("'", "''")
    probe = tmp_path / "probe.ps1"
    # A stub Register-ScheduledTask does not shadow the real one once ScheduledTasks autoloads, so the safety net
    # is the bad -At: if dot-sourcing ever ran the script body, it would throw before reaching registration.
    probe.write_text(
        "$ErrorActionPreference = 'Stop'\n"
        f". '{script}' -At 'not-a-time' -TaskName '{_UNUSED_TASK}'\n"
        "foreach ($at in '20:30', '9:05', '00:00', '23:59') {\n"
        "    $trigger = New-LocalDailyTrigger (ConvertTo-TaskTime $at)\n"
        '    Write-Output "$at -> $($trigger.StartBoundary) every $($trigger.DaysInterval)"\n'
        "}\n"
        "foreach ($bad in '25:00', '8:5', '8pm') {\n"
        '    try { ConvertTo-TaskTime $bad; Write-Output "$bad -> parsed" }\n'
        '    catch { Write-Output "$bad -> $($_.Exception.Message)" }\n'
        "}\n",
        encoding="utf-8",
    )

    proc = _powershell("-File", str(probe))

    assert proc.returncode == 0, proc.stdout + proc.stderr
    lines = proc.stdout.splitlines()
    # No "Z" or "+hh:mm": Task Scheduler then keeps the local wall-clock time through DST changes.
    for at, hhmm in (("20:30", "20:30"), ("9:05", "09:05"), ("00:00", "00:00"), ("23:59", "23:59")):
        assert any(re.fullmatch(rf"{at} -> \d{{4}}-\d\d-\d\dT{hhmm}:00 every 1", line) for line in lines), lines
    for bad in ("25:00", "8:5", "8pm"):
        assert f"{bad} -> -At must be a 24-hour local time like 20:30, got '{bad}'." in lines


@pytest.mark.skipif(shutil.which("powershell") is None, reason="needs Windows PowerShell")
def test_register_task_run_directly_rejects_a_bad_time_before_registering():
    proc = _powershell("-File", str(SCRIPTS / "register_task.ps1"), "-At", "25:99", "-TaskName", _UNUSED_TASK)

    assert proc.returncode != 0
    output = "".join((proc.stdout + proc.stderr).split())
    assert "-Atmustbea24-hourlocaltimelike20:30,got'25:99'." in output
