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

from . import db, ingest
from .config import Settings, Watchlist, load_settings, load_watchlist
from .logsetup import force_utf8_stdio, setup_logging
from .status import format_counts, show_status
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

    def backfill_truthsocial(self, since: date) -> None:
        sink, setup_error = self._prepare_sink()

        def run(run_id: int, now: datetime) -> object:
            from .collectors.truthsocial import backfill_truthsocial

            return backfill_truthsocial(self.conn, self.watchlist, _require(sink, setup_error), run_id, since, now)

        self._stage("backfill:truthsocial", run)

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="influence",
        description="Collect stock posts from X, Truth Social and Reddit, and snapshot 1-minute prices.",
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
    sub.add_parser("daily", help="collect everything, then snapshot (what the scheduled task runs)")
    sub.add_parser("status", help="last runs, post counts, account health, X spend, price coverage")
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

    if args.command == "backfill" and args.since > clock().astimezone(NY).date():
        print(f"--since {args.since} is in the future", file=sys.stderr)
        return 2

    log.info("influence %s (root %s)", args.command, settings.root)
    pipeline = Pipeline(settings, watchlist, conn, clock)
    if args.command == "collect":
        pipeline.collect(args.only)
    elif args.command == "snapshot":
        pipeline.snapshot()
    elif args.command == "daily":
        pipeline.collect()
        pipeline.snapshot()
    elif args.command == "backfill":
        pipeline.backfill_truthsocial(args.since)
    else:
        raise ValueError(f"unhandled command {args.command!r}")

    tally: dict[str, int] = {}
    for result in pipeline.results:
        tally[result.status] = tally.get(result.status, 0) + 1
    summary = ", ".join(f"{n} {status}" for status, n in tally.items())
    log.info("influence %s finished: %s -> exit %d", args.command, summary, pipeline.exit_code)
    return pipeline.exit_code


if __name__ == "__main__":
    sys.exit(main())
