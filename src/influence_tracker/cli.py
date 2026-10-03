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
from typing import TYPE_CHECKING

import yaml
from pydantic import ValidationError

from . import db, eventsview, ingest
from .analysis.study import Study, StudyOptions
from .config import Settings, Watchlist, load_settings, load_watchlist
from .logsetup import force_utf8_stdio, setup_logging
from .status import PLATFORMS, format_counts, show_status
from .timeutil import NY, utc_now

if TYPE_CHECKING:
    from .alerts.notify import Notifier
    from .alerts.watch import Steps

log = logging.getLogger(__name__)

ROOT_ENV_VAR = "INFLUENCE_TRACKER_ROOT"
COLLECTORS = ("truthsocial", "reddit", "apewisdom", "x")
COLLECTOR_STATUSES = ("ok", "partial", "error")
X_TOKEN_MISSING = "X_BEARER_TOKEN not set in .env"
WATCH_ALIVE = "live watch is running"
WATCH_LOCK = "watch.lock"
WATCH_LOG_FILE = "influence-watch.log"
WATCH_RUNNING = "another influence watch is already running"
TOPIC_PREFIX = "influence-"
NO_TOPIC = "NTFY_TOPIC is not set: run `influence alerts setup` first"
CONFIG_ERRORS = (OSError, UnicodeDecodeError, yaml.YAMLError, ValidationError)

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
            # Only what the live watch collects: one started without an X token must not leave X to nobody.
            if name in _watched_platforms(self.conn, self.clock()):
                self._skip(f"collect:{name}", WATCH_ALIVE)
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
        self._record(_run_stage(self.conn, self.clock, stage, fn))

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


def _run_stage(
    conn: sqlite3.Connection, clock: Clock, stage: str, fn: StageFn, started: datetime | None = None
) -> StageResult:
    """Runs one stage under its own runs row; a failure is logged and recorded, never raised."""
    started = started or clock()
    run_id = db.start_run(conn, stage, started)
    status, counts, error = "error", {}, None
    try:
        status, counts, error = _interpret(fn(run_id, started))
    except Exception as e:
        log.exception("%s raised", stage)
        _rollback(conn)
        error = f"{type(e).__name__}: {e}"
    db.finish_run(conn, run_id, status, counts, error, clock())
    return StageResult(stage, status, counts, error)


def _watched_platforms(conn: sqlite3.Connection, now: datetime) -> frozenset[str]:
    from .alerts.watch import live_platforms

    return live_platforms(conn, now)


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
    watch = sub.add_parser("watch", help="live alerts: check for new stock posts every few minutes and notify phones")
    watch.add_argument("--once", action="store_true", help="run a single cycle and exit")
    watch.add_argument("--dry-run", action="store_true", help="print alerts instead of sending them")
    watch.add_argument(
        "--quiet", action="store_true", help="only warnings and errors on the console (the log file keeps everything)"
    )
    alerts = sub.add_parser("alerts", help="phone alert setup: generate an ntfy topic or send a test notification")
    alerts.add_argument("action", choices=["setup", "test"])
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
    if args.command == "watch":
        # The watch runs for weeks beside the nightly run: its own rotating file, because two processes holding one
        # RotatingFileHandler file on Windows make the rollover fail and lose records.
        setup_logging(
            settings.logs_dir, file_name=WATCH_LOG_FILE, console_level=logging.WARNING if args.quiet else None
        )
    else:
        setup_logging(settings.logs_dir)
    try:
        watchlist = load_watchlist(settings.config_path)
    except CONFIG_ERRORS as e:
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
    if args.command == "alerts":
        return _alerts(args, settings)

    if args.command == "backfill" and args.since > clock().astimezone(NY).date():
        print(f"--since {args.since} is in the future", file=sys.stderr)
        return 2
    if args.command == "analyze" and args.since and args.until and args.since > args.until:
        print(f"--since {args.since} is after --until {args.until}", file=sys.stderr)
        return 2

    log.info("influence %s (root %s)", args.command, settings.root)
    if args.command == "watch":
        return _watch(args, settings, watchlist, conn, clock)
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


# ---------------------------------------------------------------- live alerts


def _make_notifier(settings: Settings, dry_run: bool) -> Notifier:
    from .alerts.notify import DryRunNotifier, NtfyNotifier

    if dry_run:
        return DryRunNotifier()
    if not settings.ntfy_topic:  # callers check first and exit 2; this only keeps a None topic from being used
        raise RuntimeError(NO_TOPIC)
    return NtfyNotifier(settings.ntfy_server, settings.ntfy_topic)


def _alerts(args: argparse.Namespace, settings: Settings) -> int:
    if args.action == "setup":
        return _alerts_setup(settings)
    if not settings.ntfy_topic:
        print(NO_TOPIC, file=sys.stderr)
        return 2
    from .alerts.notify import Message, NotifyError

    message = Message(
        "influence-tracker test alert", "If you can read this, phone alerts work.", 3, ("white_check_mark",)
    )
    try:
        _make_notifier(settings, dry_run=False).send(message)
    except NotifyError as e:
        print(f"Test alert failed: {e}", file=sys.stderr)
        return 1
    print("Test alert sent.")
    return 0


def _alerts_setup(settings: Settings) -> int:
    env = settings.root / ".env"
    if settings.ntfy_topic:
        topic = settings.ntfy_topic
        print(f"Already set up: NTFY_TOPIC={topic} in {env}")
    else:
        import secrets

        # 16 random bytes are 22 URL-safe characters: a 32-character topic in the characters ntfy allows.
        topic = TOPIC_PREFIX + secrets.token_urlsafe(16)
        existing = env.read_text(encoding="utf-8") if env.exists() else ""
        lines = [ln for ln in existing.splitlines() if not ln.startswith("NTFY_TOPIC=")]
        lines.append(f"NTFY_TOPIC={topic}")
        env.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Created NTFY_TOPIC in {env}")
    print(
        "\nOn each phone: install the free 'ntfy' app (App Store / Google Play), tap +, and subscribe to topic\n"
        f"    {topic}\n"
        f"on server {settings.ntfy_server}. The topic works like a password: share it only with your team.\n"
        "Then run: influence alerts test"
    )
    return 0


def _watch_refusal(settings: Settings, watchlist: Watchlist, dry_run: bool) -> str | None:
    """Why the watch must not run with this config, or None."""
    if not watchlist.alerts.enabled:
        return f"alerts.enabled is false in {settings.config_path}"
    if not dry_run and not settings.ntfy_topic:
        return NO_TOPIC
    return None


def _watch_steps(
    settings: Settings,
    watchlist: Watchlist,
    conn: sqlite3.Connection,
    clock: Clock,
    *,
    dry_run: bool,
    kept: dict | None = None,
) -> Steps:
    import time

    from . import events, sentiment
    from .alerts.engine import AlertEngine, StudyHistory
    from .alerts.keepawake import KeepAwake
    from .alerts.live import LivePrices
    from .alerts.watch import Steps
    from .collectors.truthsocial import collect_truthsocial
    from .collectors.x import collect_x

    # `kept` outlives a config reload (the loaded model, the keep-awake state). Everything else is built from this
    # watchlist, including the post sink's mention matcher.
    kept = {} if kept is None else kept
    pipeline = Pipeline(settings, watchlist, conn, clock)
    loaded: dict = {}

    def sink() -> ingest.PostSink:
        # Prepared on first use and retried every cycle until it works: one failure at startup must not leave a
        # long-running watch unable to store posts.
        if "sink" not in loaded:
            loaded["sink"] = _require(*pipeline._prepare_sink())
        return loaded["sink"]

    model = (watchlist.sentiment.model_id, watchlist.sentiment.batch_size)

    def classifier(texts: Sequence[str]) -> list[tuple[str, float]]:
        # Loaded once and kept in memory; a failed load raises, and the next cycle tries again.
        if kept.get("model") != model:
            kept["clf"] = sentiment.load_classifier(*model)
            kept["model"] = model
        return kept["clf"](texts)

    def recorded(stage: str, collect: StageFn) -> Callable[[int, datetime], dict]:
        # Each collect gets its own runs row, as in the nightly run: `status` reads collect outcomes and Truth Social
        # rate-limit hits from those rows, and while the watch runs the nightly only records "skipped".
        def run(_watch_run_id: int, now: datetime) -> dict:
            result = _run_stage(conn, clock, stage, collect, started=now)
            return {**result.counts, "status": result.status, "error": result.error}

        return run

    def collect_ts_step(run_id: int, now: datetime) -> dict:
        return collect_truthsocial(conn, watchlist, sink(), run_id, now)

    def collect_x_step(run_id: int, now: datetime) -> dict:
        return collect_x(conn, watchlist, settings.x_bearer_token, sink(), run_id, now)

    if "keep_awake" not in kept:
        kept["keep_awake"] = KeepAwake()
    engine = AlertEngine(
        conn,
        watchlist,
        _make_notifier(settings, dry_run),
        live_factory=lambda: LivePrices(watchlist),
        history_factory=lambda now: StudyHistory(conn, watchlist, now),
    )
    return Steps(
        clock=clock,
        sleep=time.sleep,
        collect_truthsocial=recorded("collect:truthsocial", collect_ts_step),
        collect_x=recorded("collect:x", collect_x_step) if settings.x_bearer_token else None,
        classify=lambda run_id, now: sentiment.classify_posts(conn, watchlist, run_id, now, classifier=classifier),
        sync_events=lambda now: events.sync_events(conn, watchlist, now),
        alerts=engine.run,
        keep_awake=kept["keep_awake"],
    )


class _WatchSetup:
    """The watch's settings, watchlist and steps, rebuilt between cycles after config/watchlist.yaml or .env changes.

    The watch runs for weeks. With a stale ticker matcher, posts it stores after a ticker is added would never be
    re-matched: the nightly run re-matches stored posts and records the new universe first. Holdings, the cadence
    and an X token added later must reach it too."""

    def __init__(
        self, settings: Settings, watchlist: Watchlist, conn: sqlite3.Connection, clock: Clock, *, dry_run: bool
    ) -> None:
        self.settings, self.watchlist = settings, watchlist
        self.conn, self.clock, self.dry_run = conn, clock, dry_run
        self.env = _env_file_values(settings.root)
        self.stamp = _config_stamp(settings)
        self.kept: dict = {}
        self.steps = _watch_steps(settings, watchlist, conn, clock, dry_run=dry_run, kept=self.kept)

    def reload(self) -> tuple[Watchlist, Steps] | None:
        """The new (watchlist, steps) when either file changed, else None. Raises StopWatch when the new config
        switches alerts off or drops the ntfy topic."""
        stamp = _config_stamp(self.settings)
        if stamp == self.stamp:
            return None
        self.stamp = stamp  # a broken edit is reported once; the next save is tried again
        self.env = _reapply_env(self.settings.root, self.env)
        settings = load_settings(self.settings.root)
        try:
            watchlist = load_watchlist(settings.config_path)
        except CONFIG_ERRORS as e:
            log.error(
                "cannot reload %s (%s: %s); the watch keeps its current config until the file is fixed",
                settings.config_path,
                type(e).__name__,
                e,
            )
            return None
        refusal = _watch_refusal(settings, watchlist, self.dry_run)
        if refusal is not None:
            from .alerts.watch import StopWatch

            raise StopWatch(refusal)
        self.steps = _watch_steps(settings, watchlist, self.conn, self.clock, dry_run=self.dry_run, kept=self.kept)
        self.settings, self.watchlist = settings, watchlist
        log.info("influence watch: config changed, reloaded %s and .env", settings.config_path)
        return watchlist, self.steps


def _config_stamp(settings: Settings) -> tuple[tuple[int, int] | None, ...]:
    return tuple(_file_stamp(path) for path in (settings.config_path, settings.root / ".env"))


def _file_stamp(path: Path) -> tuple[int, int] | None:
    try:
        st = path.stat()
    except OSError:
        return None
    return st.st_mtime_ns, st.st_size


def _env_file_values(root: Path) -> dict[str, str]:
    from dotenv import dotenv_values

    path = root / ".env"
    if not path.is_file():
        return {}
    return {k: v for k, v in dotenv_values(path, encoding="utf-8").items() if v is not None}


def _reapply_env(root: Path, before: dict[str, str]) -> dict[str, str]:
    """Brings this process's environment in line with an edited .env and returns the file's new values.
    load_dotenv never overrides a variable that is already set, so a key the file set (or now sets) follows the
    file, and a variable set outside .env still wins, as at startup."""
    after = _env_file_values(root)
    for key in before.keys() | after.keys():
        if os.environ.get(key) != before.get(key):
            continue
        if key in after:
            os.environ[key] = after[key]
        else:
            os.environ.pop(key, None)
    return after


def _watch(
    args: argparse.Namespace, settings: Settings, watchlist: Watchlist, conn: sqlite3.Connection, clock: Clock
) -> int:
    refusal = _watch_refusal(settings, watchlist, args.dry_run)
    if refusal is not None:
        print(refusal, file=sys.stderr)
        return 2
    from .alerts.watch import single_instance

    # One watch per database. A second one, even a --dry-run, would claim alerts from the shared ledger (a dry
    # run marks them sent) and double the Truth Social polling. The scheduled task's IgnoreNew only stops itself.
    with single_instance(settings.data_dir / WATCH_LOCK) as mine:
        if not mine:
            print(WATCH_RUNNING, file=sys.stderr)
            log.warning("influence watch: %s", WATCH_RUNNING)
            return 2
        return _watch_run(args, settings, watchlist, conn, clock)


def _watch_run(
    args: argparse.Namespace, settings: Settings, watchlist: Watchlist, conn: sqlite3.Connection, clock: Clock
) -> int:
    from .alerts import watch

    run_id = db.start_run(conn, "watch", clock())
    status, error = "ok", None
    try:
        setup = _WatchSetup(settings, watchlist, conn, clock, dry_run=args.dry_run)
        return watch.run_watch(conn, setup.watchlist, setup.steps, run_id, once=args.once, reload=setup.reload)
    except watch.StopWatch as e:
        _rollback(conn)
        error = str(e)
        print(error, file=sys.stderr)
        log.warning("influence watch stopped: %s", error)
        return 2
    except KeyboardInterrupt:
        _rollback(conn)
        log.info("influence watch stopped (Ctrl+C)")
        return 0
    except Exception as e:
        _rollback(conn)
        status, error = "error", f"{type(e).__name__}: {e}"
        raise
    finally:
        db.finish_run(conn, run_id, status, {}, error, clock())


if __name__ == "__main__":
    sys.exit(main())
