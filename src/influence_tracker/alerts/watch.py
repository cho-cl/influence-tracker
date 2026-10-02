from __future__ import annotations

import logging
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta

from .. import db
from ..config import Watchlist
from ..timeutil import from_iso, to_iso
from . import timing
from .keepawake import KeepAwake

log = logging.getLogger(__name__)

HEARTBEAT_SOURCE, HEARTBEAT_KEY = "watch", "heartbeat"
HEARTBEAT_FRESH = timedelta(minutes=15)


def heartbeat_fresh(conn: sqlite3.Connection, now: datetime) -> bool:
    value = db.get_watermark(conn, HEARTBEAT_SOURCE, HEARTBEAT_KEY)
    return value is not None and now - from_iso(value) < HEARTBEAT_FRESH


@dataclass
class Steps:
    clock: Callable[[], datetime]
    sleep: Callable[[float], None]
    collect_truthsocial: Callable[[int, datetime], dict]
    collect_x: Callable[[int, datetime], dict] | None
    classify: Callable[[int, datetime], dict]
    sync_events: Callable[[datetime], dict]
    alerts: Callable[[datetime], dict]
    keep_awake: KeepAwake


def _safe(name: str, fn: Callable[..., object], *args: object) -> None:
    try:
        result = fn(*args)
        log.info("watch %s: %s", name, result)
    except Exception:  # one broken step must not stop the watch; it retries next cycle
        log.exception("watch %s raised", name)


def run_watch(
    conn: sqlite3.Connection,
    watchlist: Watchlist,
    steps: Steps,
    run_id: int,
    *,
    once: bool = False,
    max_cycles: int | None = None,
) -> int:
    cfg = watchlist.alerts
    last_x: datetime | None = None
    cycles = 0
    try:
        while True:
            now = steps.clock()
            with conn:
                db.set_watermark(conn, HEARTBEAT_SOURCE, HEARTBEAT_KEY, to_iso(now), now)
            steps.keep_awake.update(timing.is_active(now))
            _safe("collect:truthsocial", steps.collect_truthsocial, run_id, now)
            if steps.collect_x is not None and (
                last_x is None or now - last_x >= timedelta(minutes=cfg.poll_x_minutes)
            ):
                _safe("collect:x", steps.collect_x, run_id, now)
                last_x = now
            _safe("classify", steps.classify, run_id, now)
            _safe("sync_events", steps.sync_events, now)
            _safe("alerts", steps.alerts, now)
            cycles += 1
            if once or (max_cycles is not None and cycles >= max_cycles):
                return 0
            wake = timing.next_cycle(steps.clock(), cfg)
            steps.sleep(max(1.0, (wake - steps.clock()).total_seconds()))
    finally:
        steps.keep_awake.release()
