from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import yaml
from pydantic import ValidationError

from . import db, eventsview, ingest
from .analysis.study import Study, StudyOptions
from .config import Settings, Watchlist, load_settings, load_watchlist
from .logsetup import force_utf8_stdio, setup_logging
from .status import PLATFORMS, format_counts, show_status
from .timeutil import NY, utc_now

log = logging.getLogger(__name__)

ROOT_ENV_VAR = "INFLUENCE_TRACKER_ROOT"
COLLECTORS = ("truthsocial", "reddit", "apewisdom", "x")
COLLECTOR_STATUSES = ("ok", "partial", "error")
X_TOKEN_MISSING = "X_BEARER_TOKEN not set in .env"

Clock = Callable[[], datetime]
StageFn = Callable[[int, datetime], object]


@dataclass(frozen=True)
class StageResult:
    stage: str
    status: str
    counts: dict
    error: str | None


@dataclass
class Analysis:
    """What a successful analyze stage produced, for the terminal summary."""

    study: Study | None = None
    report: Path | None = None


class Pipeline:
    """Runs isolated stages: each gets a runs row, and a failure never stops the stages after it."""

    def __init__(self, settings: Settings, watchlist: Watchlist, conn: sqlite3.Connection, clock: Clock):
        self.settings = settings
        self.watchlist = watchlist
        self.conn = conn
        self.clock = clock
        self.results: list[StageResult] = []

    @property
    def exit_code(self) -> int:
        return 1 if any(r.status == "error" for r in self.results) else 0

    # ------------------------------------------------------------ commands

    def collect(self, only: Sequence[str] | None = None) -> None:
        sink, setup_error = self._prepare_sink()
        for name in COLLECTORS:
            if only and name not in only:
                continue
            if name == "x" and not self.settings.x_bearer_token:
                self._skip("collect:x", X_TOKEN_MISSING)
                continue
            self._stage(f"collect:{name}", self._collector(name, sink, setup_error))

    def snapshot(self) -> None:
        def run(run_id: int, now: datetime) -> object:
            from .prices import snapshot_1m

            return snapshot_1m(self.conn, self.watchlist, run_id, now)

        self._stage("snapshot", run)

    def classify(self) -> None:
        def run(run_id: int, now: datetime) -> object:
            from .sentiment import classify_posts

            return classify_posts(self.conn, self.watchlist, run_id, now)

        self._stage("classify", run)

    def enrich(self) -> None:
        def run(run_id: int, now: datetime) -> object:
            from .events import enrich_events

            return enrich_events(self.conn, self.watchlist, run_id, now)

        self._stage("enrich", run)

    def backfill_truthsocial(self, since: date) -> None:
        sink, setup_error = self._prepare_sink()

        def run(run_id: int, now: datetime) -> object:
            from .collectors.truthsocial import backfill_truthsocial

            return backfill_truthsocial(self.conn, self.watchlist, _require(sink, setup_error), run_id, since, now)

        self._stage("backfill:truthsocial", run)

    def analyze(self, options: StudyOptions, out_dir: Path) -> Analysis:
        result = Analysis()

        def run(run_id: int, now: datetime) -> object:
            from .analysis.metrics import compute_study
            from .analysis.report import write_report

            study = compute_study(self.conn, self.watchlist, now, options)
            report = write_report(study, self.conn, out_dir)
            result.study, result.report = study, report
            return {**study.counts, "status": "ok", "report": str(report)}

        self._stage("analyze", run)
        return result

    # ------------------------------------------------------------ internals

    def _prepare_sink(self) -> tuple[ingest.PostSink | None, Exception | None]:
        try:
            ingest.sync_accounts(self.conn, self.watchlist)
            from .mentions import MentionMatcher

            matcher = MentionMatcher(self.watchlist)
            fingerprint = ingest.universe_fingerprint(self.watchlist)
            rematched = ingest.rematch_if_universe_changed(self.conn, matcher, fingerprint, now=self.clock)
            if rematched is not None:
                log.info("ticker universe changed: re-matched mentions for %d stored post(s)", rematched)
            return ingest.PostSink(self.conn, matcher, now=self.clock), None
        except Exception as e:
            log.exception("collect setup failed (account sync / mention matcher); post collectors will not run")
            _rollback(self.conn)
            return None, e

    def _collector(self, name: str, sink: ingest.PostSink | None, setup_error: Exception | None) -> StageFn:
        conn, watchlist = self.conn, self.watchlist

        def run(run_id: int, now: datetime) -> object:
            if name == "apewisdom":
                from .collectors.apewisdom import collect_apewisdom

                return collect_apewisdom(conn, watchlist, run_id, now)
            if name == "truthsocial":
                from .collectors.truthsocial import collect_truthsocial

                return collect_truthsocial(conn, watchlist, _require(sink, setup_error), run_id, now)
            if name == "reddit":
                from .collectors.reddit_rss import collect_reddit

                return collect_reddit(conn, watchlist, _require(sink, setup_error), run_id, now)
            from .collectors.x import collect_x

            token = self.settings.x_bearer_token
            return collect_x(conn, watchlist, token, _require(sink, setup_error), run_id, now)

        return run

    def _stage(self, stage: str, fn: StageFn) -> None:
        started = self.clock()
        run_id = db.start_run(self.conn, stage, started)
        status, counts, error = "error", {}, None
        try:
            status, counts, error = _interpret(fn(run_id, started))
        except Exception as e:
            log.exception("%s raised", stage)
            _rollback(self.conn)
            error = f"{type(e).__name__}: {e}"
        db.finish_run(self.conn, run_id, status, counts, error, self.clock())
        self._record(StageResult(stage, status, counts, error))

    def _skip(self, stage: str, reason: str) -> None:
        now = self.clock()
        run_id = db.start_run(self.conn, stage, now)
        db.finish_run(self.conn, run_id, "skipped", {}, reason, now)
        self._record(StageResult(stage, "skipped", {}, reason))

    def _record(self, result: StageResult) -> None:
        self.results.append(result)
        line = f"{result.stage}: {result.status}"
        counts = format_counts(result.counts, max_items=12)
        if counts:
            line += f" | {counts}"
        if result.error:
            line += f" | {' '.join(result.error.split())[:300]}"
        level = {"error": logging.ERROR, "partial": logging.WARNING, "skipped": logging.WARNING}.get(
            result.status, logging.INFO
        )
        log.log(level, "%s", line)


def _require(sink: ingest.PostSink | None, setup_error: Exception | None) -> ingest.PostSink:
    if sink is None:
        raise RuntimeError(
            f"collect setup failed, so no posts can be stored: {type(setup_error).__name__}: {setup_error}"
        )
    return sink


def _rollback(conn: sqlite3.Connection) -> None:
    # A stage that died mid-transaction must not have its half-written page committed by finish_run.
    if conn.in_transaction:
        conn.rollback()


def _interpret(result: object) -> tuple[str, dict, str | None]:
    """Collector return value -> (status, counts, error)."""
    if not isinstance(result, dict):
        return "error", {}, f"stage returned {type(result).__name__}, expected a counts dict"
    # Counts are stored as JSON; a stray datetime or Path in them must not break the bookkeeping.
    counts = json.loads(json.dumps(result, ensure_ascii=False, default=str))
    status = counts.get("status")
    if status not in COLLECTOR_STATUSES:
        return "error", counts, f"stage returned invalid status {status!r}"
    error = counts.get("error") or counts.get("errors") or None
    if isinstance(error, list):
        error = "; ".join(str(e) for e in error)
    elif error is not None and not isinstance(error, str):
        error = json.dumps(error, ensure_ascii=False)
    return status, counts, error


# ---------------------------------------------------------------- argument parsing


def _iso_date(text: str) -> date:
    try:
        return datetime.strptime(text, "%Y-%m-%d").date()
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a date like 2025-01-20, got {text!r}") from None


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a whole number, got {text!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be 1 or more, got {value}")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="influence",
        description=(
            "Collect stock posts from X, Truth Social and Reddit, snapshot 1-minute prices, label each post's "
            "stance, turn every ticker mention into an event with its reference price and returns, and measure "
            "how the posts moved the stocks."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    collect = sub.add_parser("collect", help="fetch new posts: Truth Social, Reddit RSS, ApeWisdom, X")
    collect.add_argument(
        "--only",
        nargs="+",
        choices=COLLECTORS,
        metavar="SOURCE",
        help=f"run only these collectors ({', '.join(COLLECTORS)})",
    )
    sub.add_parser("snapshot", help="save 1-minute bars for every watchlist ticker")
    sub.add_parser("classify", help="label posts that mention a watchlist stock bullish, bearish or neutral")
    sub.add_parser("enrich", help="turn ticker mentions into events: daily bars, earnings, reference prices, returns")
    sub.add_parser("daily", help="collect, snapshot, classify, then enrich (what the scheduled task runs)")
    sub.add_parser("status", help="last runs, post counts, stances, events, account health, X spend, price coverage")
    events = sub.add_parser(
        "events",
        help="list events to check by hand against price charts",
        description="List events, newest post first, with their reference price and returns in New York time.",
    )
    events.add_argument("--ticker", metavar="SYMBOL", help="only events on this ticker")
    events.add_argument("--platform", choices=PLATFORMS, help="only events from this platform")
    events.add_argument("--status", choices=("pending", "complete"), help="only pending or only complete events")
    events.add_argument(
        "--limit",
        type=_positive_int,
        metavar="N",
        help=f"the newest N events (default {eventsview.DEFAULT_LIMIT}; --csv exports every match unless given)",
    )
    detail = events.add_mutually_exclusive_group()
    detail.add_argument("--id", type=int, dest="event_id", metavar="N", help="show one event in full")
    detail.add_argument("--csv", type=Path, metavar="PATH", help="export the matching events to a CSV file for Excel")
    analyze = sub.add_parser(
        "analyze",
        help="event study: abnormal returns vs SPY, placebo days, charts and report.html",
        description=(
            "Measure how posts moved stocks over every complete event: market-model abnormal returns, z-scores, "
            "placebo days and group tests. Writes report.html, CSVs and charts to a folder."
        ),
    )
    analyze.add_argument(
        "--since", type=_iso_date, metavar="YYYY-MM-DD", help="only events whose post day (d0) is on or after this"
    )
    analyze.add_argument(
        "--until", type=_iso_date, metavar="YYYY-MM-DD", help="only events whose post day (d0) is on or before this"
    )
    analyze.add_argument(
        "--include-earnings", action="store_true", help="keep events within 1 session of an earnings announcement"
    )
    analyze.add_argument("--include-splits", action="store_true", help="keep events near a stock split")
    analyze.add_argument(
        "--include-clustered", action="store_true", help="keep repeat posts about a ticker in the same session"
    )
    analyze.add_argument(
        "--out", type=Path, metavar="DIR", help="output folder (default reports/<today's New York date>/)"
    )
    analyze.add_argument("--open", action="store_true", dest="open_report", help="open report.html when done")
    backfill = sub.add_parser("backfill", help="one-time, slow history backfill (resumable)")
    backfill.add_argument("source", choices=["truthsocial"])
    backfill.add_argument("--since", required=True, type=_iso_date, metavar="YYYY-MM-DD")
    return parser


# ---------------------------------------------------------------- entry point


def _config_error_message(path: Path, error: Exception) -> str:
    if isinstance(error, ValidationError):
        lines = [f"{'.'.join(str(part) for part in e['loc']) or '(top level)'}: {e['msg']}" for e in error.errors()]
    elif isinstance(error, FileNotFoundError):
        lines = ["file not found"]
    else:
        lines = str(error).splitlines()
    detail = "\n".join(f"  {line}" for line in lines)
    return f"Config problem in {path}:\n{detail}\nFix the file and run the command again."


def _root_from_env() -> Path | None:
    value = os.environ.get(ROOT_ENV_VAR, "").strip()
    return Path(value).resolve() if value else None


def main(argv: list[str] | None = None) -> int:
    force_utf8_stdio()
    try:
        args = build_parser().parse_args(argv)
    except SystemExit as e:  # argparse exits on --help and on usage errors; main() returns the code instead
        return e.code if isinstance(e.code, int) else 0

    settings = load_settings(_root_from_env())
    setup_logging(settings.logs_dir)
    try:
        watchlist = load_watchlist(settings.config_path)
    except (OSError, UnicodeDecodeError, yaml.YAMLError, ValidationError) as e:
        log.error("cannot load %s (%s)", settings.config_path, type(e).__name__)
        print(_config_error_message(settings.config_path, e), file=sys.stderr)
        return 2
    try:
        conn = db.connect(settings.db_path)
    except (sqlite3.Error, RuntimeError) as e:
        log.error("cannot open database %s: %s", settings.db_path, e)
        return 1

    clock = utc_now
    try:
        return _dispatch(args, settings, watchlist, conn, clock)
    except Exception:
        log.exception("influence %s crashed", args.command)
        return 1
    finally:
        conn.close()


def _dispatch(
    args: argparse.Namespace, settings: Settings, watchlist: Watchlist, conn: sqlite3.Connection, clock: Clock
) -> int:
    if args.command == "status":
        show_status(conn, watchlist, clock())
        return 0
    if args.command == "events":
        return _events(args, watchlist, conn)

    if args.command == "backfill" and args.since > clock().astimezone(NY).date():
        print(f"--since {args.since} is in the future", file=sys.stderr)
        return 2
    if args.command == "analyze" and args.since and args.until and args.since > args.until:
        print(f"--since {args.since} is after --until {args.until}", file=sys.stderr)
        return 2

    log.info("influence %s (root %s)", args.command, settings.root)
    pipeline = Pipeline(settings, watchlist, conn, clock)
    analysis: Analysis | None = None
    if args.command == "collect":
        pipeline.collect(args.only)
    elif args.command == "snapshot":
        pipeline.snapshot()
    elif args.command == "classify":
        pipeline.classify()
    elif args.command == "enrich":
        pipeline.enrich()
    elif args.command == "daily":
        pipeline.collect()
        pipeline.snapshot()
        pipeline.classify()
        pipeline.enrich()
    elif args.command == "backfill":
        pipeline.backfill_truthsocial(args.since)
    elif args.command == "analyze":
        options = StudyOptions(
            since=args.since,
            until=args.until,
            include_earnings=args.include_earnings,
            include_splits=args.include_splits,
            include_clustered=args.include_clustered,
        )
        out_dir = args.out or settings.reports_dir / clock().astimezone(NY).date().isoformat()
        analysis = pipeline.analyze(options, out_dir)
    else:
        raise ValueError(f"unhandled command {args.command!r}")

    tally: dict[str, int] = {}
    for result in pipeline.results:
        tally[result.status] = tally.get(result.status, 0) + 1
    summary = ", ".join(f"{n} {status}" for status, n in tally.items())
    log.info("influence %s finished: %s -> exit %d", args.command, summary, pipeline.exit_code)
    if analysis is not None:
        _print_analysis(analysis, pipeline.results[-1], args.open_report)
    return pipeline.exit_code


def _print_analysis(analysis: Analysis, result: StageResult, open_report: bool) -> None:
    if analysis.report is None or analysis.study is None:
        hint = " (is a CSV open in Excel?)" if (result.error or "").startswith("PermissionError") else ""
        print(f"analyze failed: {result.error}{hint}; details are in the log", file=sys.stderr)
        return
    from .analysis.report import headlines

    print(f"Report: {analysis.report.resolve()}")
    try:
        lines = headlines(analysis.study)
    # The report is already written and the run recorded; a summary bug must not turn that into a crash.
    except Exception:
        log.exception("could not summarize the study for the terminal")
        lines = ["(summary unavailable; see the report and the log)"]
    for line in lines:
        print(f"  {line}")
    if open_report:
        _open_file(analysis.report)


def _open_file(path: Path) -> None:
    startfile = getattr(os, "startfile", None)
    try:
        if startfile is not None:
            startfile(path)
        else:
            import webbrowser

            webbrowser.open(path.resolve().as_uri())
    except OSError as e:
        print(f"cannot open {path}: {e}", file=sys.stderr)


def _events(args: argparse.Namespace, watchlist: Watchlist, conn: sqlite3.Connection) -> int:
    if args.event_id is not None:
        if eventsview.show_event(conn, args.event_id, eventsview.events_console()):
            return 0
        print(f"no event with id {args.event_id}", file=sys.stderr)
        return 1

    ticker = args.ticker
    if ticker:
        known = watchlist.ticker(ticker)
        ticker = known.symbol if known else ticker.upper()
    flt = eventsview.EventFilter(ticker, args.platform, args.status)
    if args.csv is not None:
        try:
            n = eventsview.export_csv(conn, args.csv, flt, args.limit)
        except OSError as e:
            hint = " (is it open in Excel?)" if isinstance(e, PermissionError) else ""
            print(f"cannot write {args.csv}: {e}{hint}", file=sys.stderr)
            return 1
        print(f"wrote {n} event{'' if n == 1 else 's'} to {args.csv.resolve()}")
        return 0
    eventsview.show_events(conn, flt, args.limit or eventsview.DEFAULT_LIMIT, eventsview.events_console())
    return 0


if __name__ == "__main__":
    sys.exit(main())
