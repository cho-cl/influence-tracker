from __future__ import annotations

import csv
import io
import logging
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import types
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest
from rich.console import Console

from influence_tracker import cli, db, ingest, logsetup, status
from influence_tracker.analysis import report
from influence_tracker.analysis.study import (
    ATTENTION_COLUMNS,
    EVENT_COLUMNS,
    GROUP_COLUMNS,
    INTRADAY_PATH_COLUMNS,
    MAGNITUDE_COLUMNS,
    PATH_COLUMNS,
    POST_COLUMNS,
    Study,
    StudyOptions,
)
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
    _install(monkeypatch, "influence_tracker.sentiment", classify_posts=rec.fake("classify"))
    _install(monkeypatch, "influence_tracker.events", enrich_events=rec.fake("enrich"))
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

    assert fakes.order == ["truthsocial", "reddit", "apewisdom", "snapshot", "classify", "enrich"]
    assert _runs(root) == [
        ("collect:truthsocial", "partial", "429 twice; giving up"),
        ("collect:reddit", "error", "RuntimeError: feed exploded"),
        ("collect:apewisdom", "ok", None),
        ("collect:x", "skipped", "X_BEARER_TOKEN not set in .env"),
        ("snapshot", "ok", None),
        ("classify", "ok", None),
        ("enrich", "ok", None),
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


def test_daily_runs_collect_snapshot_classify_enrich_in_order(root, fakes, fixed_now, watchlist):
    assert cli.main(["daily"]) == 0

    assert fakes.order == ["truthsocial", "reddit", "apewisdom", "snapshot", "classify", "enrich"]
    assert [stage for stage, _, _ in _runs(root)] == [
        "collect:truthsocial",
        "collect:reddit",
        "collect:apewisdom",
        "collect:x",
        "snapshot",
        "classify",
        "enrich",
    ]
    for stage, expected_run_id in (("snapshot", 5), ("classify", 6), ("enrich", 7)):
        conn_arg, watchlist_arg, run_id, now = fakes.args(stage)
        assert isinstance(conn_arg, sqlite3.Connection)
        assert watchlist_arg == watchlist
        assert (run_id, now) == (expected_run_id, fixed_now)


@pytest.mark.parametrize("stage", ["snapshot", "classify", "enrich"])
def test_a_stage_alone(root, fakes, stage):
    assert cli.main([stage]) == 0
    assert fakes.order == [stage]
    assert _runs(root) == [(stage, "ok", None)]


def test_enrich_still_runs_when_classify_raises(root, fakes):
    fakes.outcomes["classify"] = MemoryError("model does not fit")
    fakes.outcomes["enrich"] = {"status": "partial", "completed": 3, "failed_events": [7]}

    assert cli.main(["daily"]) == 1

    assert fakes.order[-2:] == ["classify", "enrich"]
    assert _runs(root)[-2:] == [
        ("classify", "error", "MemoryError: model does not fit"),
        ("enrich", "partial", None),
    ]


def test_a_partial_enrich_does_not_fail_the_run(root, fakes):
    fakes.outcomes["enrich"] = {"status": "partial", "daily": {"status": "partial", "symbols_failed": ["NVDA"]}}

    assert cli.main(["enrich"]) == 0

    [(counts_json,)] = _query(root, "SELECT counts_json FROM runs")
    assert '"symbols_failed": ["NVDA"]' in counts_json


def test_missing_sentiment_module_fails_only_classify(root, fakes, monkeypatch):
    monkeypatch.setitem(sys.modules, "influence_tracker.sentiment", None)

    assert cli.main(["daily"]) == 1

    assert fakes.order[-2:] == ["snapshot", "enrich"]
    runs = _runs(root)
    assert [(s, st) for s, st, _ in runs[-3:]] == [("snapshot", "ok"), ("classify", "error"), ("enrich", "ok")]
    assert runs[-2][2].startswith("ModuleNotFoundError")


def test_missing_collector_module_fails_only_its_stage(root, fakes, monkeypatch):
    monkeypatch.setitem(sys.modules, "influence_tracker.collectors.reddit_rss", None)

    assert cli.main(["daily"]) == 1

    assert fakes.order == ["truthsocial", "apewisdom", "snapshot", "classify", "enrich"]
    stage, state, error = _runs(root)[1]
    assert (stage, state) == ("collect:reddit", "error")
    assert error.startswith("ModuleNotFoundError")


def test_collect_setup_failure_fails_post_stages_but_not_apewisdom_or_snapshot(root, fakes, monkeypatch):
    def broken_matcher(watchlist):
        raise ValueError("bad regex")

    monkeypatch.setattr(sys.modules["influence_tracker.mentions"], "MentionMatcher", broken_matcher)

    assert cli.main(["daily"]) == 1

    assert fakes.order == ["apewisdom", "snapshot", "classify", "enrich"]
    runs = _runs(root)
    assert [(s, st) for s, st, _ in runs] == [
        ("collect:truthsocial", "error"),
        ("collect:reddit", "error"),
        ("collect:apewisdom", "ok"),
        ("collect:x", "skipped"),
        ("snapshot", "ok"),
        ("classify", "ok"),
        ("enrich", "ok"),
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


# ---------------------------------------------------------------- events


def _insert_event(conn: sqlite3.Connection) -> int:
    """One complete TSLA event posted Thu 2026-09-24 10:31:07 New York time, with its +15 minute window."""
    ref = int(datetime(2026, 9, 24, 14, 30, tzinfo=UTC).timestamp())
    with conn:
        conn.execute(
            """INSERT INTO posts (platform, native_id, author, created_at_utc, text, url, stance, stance_conf,
                                  stance_model, collected_at)
               VALUES ('x', '1001', 'elonmusk', '2026-09-24T14:31:07Z', 'Buying $TSLA 🚀 [/x]',
                       'https://x.com/elonmusk/status/1001', 'bullish', 0.87, 'm', '2026-09-24T22:30:00Z')"""
        )
        cur = conn.execute(
            """INSERT INTO events (platform, native_id, ticker, t0, d0, session_phase, status, intraday_state,
                                   ref_ts, ref_price, earnings_flag, split_flag, created_at, completed_at)
               VALUES ('x', '1001', 'TSLA', '2026-09-24T14:31:07Z', '2026-09-24', 'regular', 'complete', 'ok',
                       ?, 182.41, 0, 0, '2026-09-24T22:30:00Z', '2026-10-01T22:30:00Z')""",
            (ref,),
        )
        event_id = int(cur.lastrowid)
        conn.execute(
            """INSERT INTO event_windows (event_id, win, start_ts, end_ts, start_price, end_price, ret,
                                          spy_start_price, spy_end_price, spy_ret, truncated)
               VALUES (?, 'p15', ?, ?, 182.41, 183.0, 0.003234, 661.0, 660.67, -0.0005, 0)""",
            (event_id, ref, ref + 15 * 60),
        )
    return event_id


def test_events_on_an_empty_database(root, capsys):
    assert cli.main(["events"]) == 0
    assert "No events yet" in capsys.readouterr().out


def test_events_lists_shows_one_and_exports(root, capsys, tmp_path):
    conn = db.connect(root / "data" / "tracker.db")
    try:
        event_id = _insert_event(conn)
    finally:
        conn.close()

    assert cli.main(["events", "--ticker", "tsla", "--status", "complete", "--platform", "x"]) == 0
    out = capsys.readouterr().out
    assert "1 of 1 (ticker=TSLA, platform=x, status=complete)" in out
    assert "Sep 24 10:31:07 ET" in out and "182.41 @10:30" in out and "+0.32%" in out and "-0.05%" in out
    assert "Buying $TSLA 🚀 [/x]" in out
    assert "hidden to fit" not in out, "redirected output is not squeezed to a terminal width"

    assert cli.main(["events", "--ticker", "NVDA"]) == 0
    assert "No events match ticker=NVDA." in capsys.readouterr().out

    assert cli.main(["events", "--id", str(event_id)]) == 0
    out = " ".join(capsys.readouterr().out.split())
    assert "read the close of the 10:30 ET bar; it should be 182.41" in out
    assert "https://x.com/elonmusk/status/1001" in out

    assert cli.main(["events", "--id", "999"]) == 1
    assert "no event with id 999" in capsys.readouterr().err

    path = tmp_path / "out" / "events.csv"
    assert cli.main(["events", "--csv", str(path)]) == 0
    assert f"wrote 1 event to {path.resolve()}" in capsys.readouterr().out
    with open(path, encoding="utf-8-sig", newline="") as f:
        [row] = list(csv.DictReader(f))
    assert (row["ticker"], row["p15_ret"], row["text"]) == ("TSLA", "0.003234", "Buying $TSLA 🚀 [/x]")


def test_events_csv_that_cannot_be_written_exits_1(root, capsys, tmp_path):
    assert cli.main(["events", "--csv", str(tmp_path)]) == 1
    assert f"cannot write {tmp_path}" in capsys.readouterr().err


@pytest.mark.parametrize(
    "argv",
    [
        ["events", "--status", "open"],
        ["events", "--platform", "facebook"],
        ["events", "--limit", "0"],
        ["events", "--limit", "ten"],
        ["events", "--id", "x"],
        ["events", "--id", "1", "--csv", "events.csv"],
    ],
)
def test_events_rejects_bad_arguments(root, argv):
    assert cli.main(argv) == 2


# ---------------------------------------------------------------- analyze


def _empty_study(options: StudyOptions, now: datetime) -> Study:
    def empty(columns: tuple[str, ...]) -> pd.DataFrame:
        return pd.DataFrame(columns=list(columns))

    return Study(
        options=options,
        generated_at=now,
        events=empty(EVENT_COLUMNS),
        posts=empty(POST_COLUMNS),
        groups=empty(GROUP_COLUMNS),
        magnitude=empty(MAGNITUDE_COLUMNS),
        car_path=empty(PATH_COLUMNS),
        intraday_path=empty(INTRADAY_PATH_COLUMNS),
        attention=empty(ATTENTION_COLUMNS),
        counts={"events_complete": 0, "events_included": 0, "posts_included": 0},
        notes=["No complete events yet."],
    )


class Analysis:
    def __init__(self):
        self.calls: list[tuple] = []
        self.reports: list[tuple] = []
        self.error: BaseException | None = None
        self.report_error: BaseException | None = None

    def compute_study(self, conn, watchlist, now, options):
        self.calls.append((conn, watchlist, now, options))
        if self.error is not None:
            raise self.error
        return _empty_study(options, now)

    def write_report(self, study, conn, out_dir):
        self.reports.append((study, conn, out_dir))
        if self.report_error is not None:
            raise self.report_error
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "report.html"
        path.write_text("<!DOCTYPE html>", encoding="utf-8")
        return path


@pytest.fixture
def analysis(monkeypatch) -> Analysis:
    """compute_study (metrics.py) and write_report replaced by recorders."""
    rec = Analysis()
    _install(monkeypatch, "influence_tracker.analysis.metrics", compute_study=rec.compute_study)
    monkeypatch.setattr(report, "write_report", rec.write_report)
    return rec


def test_analyze_builds_the_options_writes_the_report_and_records_a_run(root, analysis, capsys, fixed_now, watchlist):
    argv = ["analyze", "--since", "2026-08-01", "--until", "2026-09-18", "--include-earnings", "--include-clustered"]

    assert cli.main(argv) == 0

    [(conn, watchlist_arg, now, options)] = analysis.calls
    assert isinstance(conn, sqlite3.Connection) and watchlist_arg == watchlist and now == fixed_now
    assert options == StudyOptions(
        since=date(2026, 8, 1), until=date(2026, 9, 18), include_earnings=True, include_clustered=True
    )
    [(study, report_conn, out_dir)] = analysis.reports
    assert report_conn is conn and study.options == options
    # 22:30 UTC on Sep 24 is still Sep 24 in New York.
    assert out_dir == root / "reports" / "2026-09-24"
    assert _runs(root) == [("analyze", "ok", None)]
    [(counts_json,)] = _query(root, "SELECT counts_json FROM runs")
    assert '"posts_included": 0' in counts_json and "report.html" in counts_json

    out = capsys.readouterr().out.splitlines()
    assert out[0] == f"Report: {(out_dir / 'report.html').resolve()}"
    assert out[1:] == [
        "  Posts analyzed: 0 (0 events included, 0 excluded)",
        "  Mean signed CAR[0,+1]: insufficient data (n = 0 signed posts; the test needs 10)",
        "  |z| > 1.96 in the event window: insufficient data (no events with a z-score)",
    ]


def test_analyze_defaults_leave_every_exclusion_on(root, analysis):
    assert cli.main(["analyze"]) == 0
    assert analysis.calls[0][3] == StudyOptions()


def test_analyze_out_folder_and_open(root, analysis, monkeypatch, tmp_path, capsys):
    opened: list[Path] = []
    monkeypatch.setattr(cli.os, "startfile", opened.append, raising=False)
    target = tmp_path / "my reports" / "run 1"

    assert cli.main(["analyze", "--out", str(target), "--include-splits", "--open"]) == 0

    assert analysis.reports[0][2] == target
    assert analysis.calls[0][3].include_splits is True
    assert opened == [target / "report.html"]
    assert f"Report: {(target / 'report.html').resolve()}" in capsys.readouterr().out


def test_analyze_does_not_open_a_report_that_failed(root, analysis, monkeypatch, capsys):
    opened: list[Path] = []
    monkeypatch.setattr(cli.os, "startfile", opened.append, raising=False)
    analysis.error = ValueError("events table is empty")

    assert cli.main(["analyze", "--open"]) == 1

    assert opened == [] and analysis.reports == []
    assert _runs(root) == [("analyze", "error", "ValueError: events table is empty")]
    err = capsys.readouterr().err
    assert "analyze failed: ValueError: events table is empty; details are in the log" in err
    assert "Traceback" in (root / "logs" / "influence.log").read_text(encoding="utf-8")


def test_analyze_hints_at_excel_when_a_file_is_locked(root, analysis, capsys):
    analysis.report_error = PermissionError(13, "Permission denied", "events.csv")

    assert cli.main(["analyze"]) == 1

    assert _runs(root)[0][:2] == ("analyze", "error")
    assert "(is a CSV open in Excel?)" in capsys.readouterr().err


def test_analyze_fails_cleanly_without_the_metrics_module(root, monkeypatch):
    monkeypatch.setitem(sys.modules, "influence_tracker.analysis.metrics", None)

    assert cli.main(["analyze"]) == 1

    [(stage, state, error)] = _runs(root)
    assert (stage, state) == ("analyze", "error") and error.startswith("ModuleNotFoundError")


def test_analyze_with_the_real_report_writer(root, monkeypatch, capsys, tmp_path):
    rec = Analysis()
    _install(monkeypatch, "influence_tracker.analysis.metrics", compute_study=rec.compute_study)
    out = tmp_path / "out"

    assert cli.main(["analyze", "--out", str(out)]) == 0

    page = (out / "report.html").read_text(encoding="utf-8")
    assert "Do social-media posts move stocks?" in page and "No complete events yet." in page
    assert (out / "events.csv").is_file() and (out / "summary.csv").is_file()
    assert "Posts analyzed: 0" in capsys.readouterr().out


@pytest.mark.parametrize(
    "argv",
    [
        ["analyze", "--since", "2026-13-01"],
        ["analyze", "--until", "yesterday"],
        ["analyze", "--since", "2026-09-10", "--until", "2026-09-01"],
        ["analyze", "--out"],
        ["analyze", "--include-everything"],
    ],
)
def test_analyze_rejects_bad_arguments(root, analysis, argv):
    assert cli.main(argv) == 2
    assert analysis.calls == []
    assert _query(root, "SELECT COUNT(*) FROM runs") == [(0,)]


def test_analyze_is_listed_in_help(capsys):
    assert cli.main(["--help"]) == 0
    assert "analyze" in capsys.readouterr().out
    assert cli.main(["analyze", "--help"]) == 0
    out = capsys.readouterr().out
    for flag in ("--since", "--until", "--include-earnings", "--include-splits", "--include-clustered", "--out"):
        assert flag in out


def test_importing_the_cli_does_not_load_heavy_libraries():
    code = (
        "import sys, influence_tracker.cli; "
        "print(sorted(m for m in ('matplotlib', 'scipy', 'torch', 'transformers', 'yfinance') if m in sys.modules))"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "[]"


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
    m2 = (
        "Stance: no posts mention a watchlist stock yet",
        "Events: none yet",
        "Daily bars: none yet",
        "Earnings dates: no lookups yet",
    )
    for phrase in (*empty, *m2, f"0 of {len(watchlist.tickers)} symbols"):
        assert phrase in out
    assert re.search(r"^classify\W+never", out, re.MULTILINE) and re.search(r"^enrich\W+never", out, re.MULTILINE)
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


def _tagged_post(conn, native_id, tickers, stance=None, model=None) -> None:
    post = _post("truthsocial", native_id, "text", datetime(2026, 9, 24, 14, tzinfo=UTC))
    db.upsert_post(conn, post, post.created_at_utc)
    db.replace_mentions(
        conn, "truthsocial", native_id, [Mention(t, "name", t) for t in tickers], [], post.created_at_utc
    )
    if stance:
        conn.execute(
            "UPDATE posts SET stance = ?, stance_conf = 0.9, stance_model = ? WHERE native_id = ?",
            (stance, model, native_id),
        )


def _status_event(conn, native_id, d0, status, intraday_state=None) -> None:
    conn.execute(
        """INSERT INTO events (platform, native_id, ticker, t0, d0, session_phase, status, intraday_state, created_at)
           VALUES ('truthsocial', ?, 'INTC', ?, ?, 'regular', ?, ?, '2026-09-24T22:30:00Z')""",
        (native_id, f"{d0}T15:00:00Z", d0, status, intraday_state),
    )


def test_status_reports_stances_events_daily_bars_and_earnings(conn, watchlist, fixed_now):
    model = watchlist.sentiment.model_id
    with conn:
        _tagged_post(conn, "p1", ["INTC"], "bullish", model)
        _tagged_post(conn, "p2", ["TSLA", "SPY"], "bearish", "an/older-model")
        _tagged_post(conn, "p3", ["NVDA"])
        # Benchmark-only posts never become events, so classify skips them and they are not counted.
        _tagged_post(conn, "p4", ["SPY", "QQQ"])

        _status_event(conn, "p1", "2026-09-14", "pending")
        _status_event(conn, "p2", "2026-09-24", "pending")
        _status_event(conn, "p3", "2026-09-10", "complete", "ok")
        _status_event(conn, "p4", "2026-09-11", "complete", "unavailable")

        conn.executemany(
            "INSERT INTO bars_1d (symbol, session_date, close, adj_close, fetched_at) VALUES (?, ?, 1, 1, ?)",
            [("SPY", "2026-09-23", "x"), ("SPY", "2026-09-24", "x"), ("INTC", "2026-09-24", "x")],
        )
        conn.executemany(
            "INSERT INTO earnings_fetch (symbol, fetched_at, ok, error) VALUES (?, ?, ?, ?)",
            [
                ("INTC", "2026-09-24T22:00:00Z", 1, None),
                ("NVDA", "2026-09-24T22:00:00Z", 0, "YFException: [/x] 404"),
            ],
        )
        conn.executemany(
            "INSERT INTO earnings (symbol, earnings_at) VALUES ('INTC', ?)",
            [("2026-07-23T20:05:00Z",), ("2026-10-22T20:05:00Z",)],
        )
    old = db.start_run(conn, "enrich", fixed_now - timedelta(days=1))
    db.finish_run(conn, old, "partial", {"status": "partial", "daily": {"symbols_failed": ["AAPL"]}}, None, fixed_now)
    last = db.start_run(conn, "enrich", fixed_now - timedelta(minutes=30))
    daily = {"status": "partial", "symbols_failed": ["NVDA", "TSLA"]}
    db.finish_run(conn, last, "partial", {"status": "partial", "daily": daily}, None, fixed_now)

    out = _render(conn, watchlist, fixed_now)

    assert "Stance of the 3 posts that mention a watchlist stock: 1 bullish, 1 bearish, 0 neutral, 1 unlabelled" in out
    assert f"1 labelled by a model other than {model}; the next classify run relabels them" in out
    assert "Events: 4 (2 complete, 2 pending), intraday data unavailable for 1" in out
    # Sep 15-18 and 21-24 have closed by Thursday 18:30 New York time.
    assert (
        "oldest pending event: d0 2026-09-14 (Mon), 8 sessions old; events complete once daily bars reach d0+5, "
        "so check the enrich runs"
    ) in out
    assert "Daily bars: 2 symbols, latest session 2026-09-24" in out
    assert "failed in the last enrich run, 2026-09-24 18:00 ET (30m ago): NVDA, TSLA" in out
    assert "AAPL" not in out[out.index("Daily bars") : out.index("Earnings dates")]
    assert "Earnings dates: 1 symbol fetched ok, 1 failed (2 dates stored)" in out
    assert "YFException: [/x] 404" in out


@pytest.mark.parametrize(
    ("status_", "counts", "expected"),
    [
        ("ok", {"status": "ok", "daily": {"symbols_failed": []}}, "no symbol failed in the last enrich run"),
        ("error", {}, "the last enrich run, 2026-09-24 18:30 ET (just now), has no daily-bar result (status: error)"),
    ],
)
def test_status_daily_bar_failures_of_the_last_enrich_run(conn, watchlist, fixed_now, status_, counts, expected):
    run_id = db.start_run(conn, "enrich", fixed_now)
    db.finish_run(conn, run_id, status_, counts, None, fixed_now)
    assert expected in _render(conn, watchlist, fixed_now)


@pytest.mark.parametrize(
    ("d0", "now", "expected"),
    [
        ("2026-09-24", datetime(2026, 9, 24, 22, 30, tzinfo=UTC), 0),
        ("2026-09-23", datetime(2026, 9, 24, 22, 30, tzinfo=UTC), 1),
        # 15:00 New York: Thursday's session has not closed yet.
        ("2026-09-23", datetime(2026, 9, 24, 19, 0, tzinfo=UTC), 0),
        # A weekend post's d0 is still ahead.
        ("2026-09-28", datetime(2026, 9, 26, 16, 0, tzinfo=UTC), 0),
        # Thanksgiving is skipped; the Nov 27 early close (13:00) counts once passed.
        ("2026-11-25", datetime(2026, 11, 27, 18, 5, tzinfo=UTC), 1),
        ("2026-11-25", datetime(2026, 11, 30, 21, 0, tzinfo=UTC), 2),
    ],
)
def test_pending_age_counts_closed_sessions_after_d0(d0, now, expected):
    assert status._sessions_old(date.fromisoformat(d0), now) == expected


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
