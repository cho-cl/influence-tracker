from __future__ import annotations

import logging
import os
import sqlite3
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from .. import db
from ..config import Watchlist
from ..timeutil import from_iso, to_iso
from . import timing
from .keepawake import KeepAwake

log = logging.getLogger(__name__)

HEARTBEAT_SOURCE, HEARTBEAT_KEY = "watch", "heartbeat"
# Which platforms the live watch collects, written with each heartbeat: the nightly run leaves only those to it.
PLATFORMS_KEY = "platforms"
HEARTBEAT_FRESH = timedelta(minutes=15)
# Windows 8+ relative timers (time.sleep) stop counting while the PC is suspended, so a single long sleep would
# run on past a resume; waiting in slices and re-reading the wall clock bounds that lag to one slice.
SLEEP_SLICE = 60.0


def heartbeat_fresh(conn: sqlite3.Connection, now: datetime) -> bool:
    value = db.get_watermark(conn, HEARTBEAT_SOURCE, HEARTBEAT_KEY)
    return value is not None and now - from_iso(value) < HEARTBEAT_FRESH


def live_platforms(conn: sqlite3.Connection, now: datetime) -> frozenset[str]:
    """The platforms a live watch is collecting; none while its heartbeat is stale."""
    if not heartbeat_fresh(conn, now):
        return frozenset()
    value = db.get_watermark(conn, HEARTBEAT_SOURCE, PLATFORMS_KEY)
    return frozenset(value.split(",")) if value else frozenset({"truthsocial"})


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

    @property
    def platforms(self) -> str:
        return "truthsocial,x" if self.collect_x is not None else "truthsocial"


class StopWatch(Exception):
    """Raised by a config reload when the watch must not go on (alerts switched off, no ntfy topic)."""


Reload = Callable[[], tuple[Watchlist, Steps] | None]


def _beat(conn: sqlite3.Connection, platforms: str, now: datetime) -> str:
    with conn:
        db.set_watermark(conn, HEARTBEAT_SOURCE, HEARTBEAT_KEY, to_iso(now), now)
        db.set_watermark(conn, HEARTBEAT_SOURCE, PLATFORMS_KEY, platforms, now)
    return to_iso(now)


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
    reload: Reload | None = None,
) -> int:
    """Loops until stopped. `reload` runs before every cycle after the first and returns a new (watchlist, steps)
    when the config changed; it may raise StopWatch to end the loop."""
    last_x: datetime | None = None
    cycles = 0
    try:
        while True:
            if reload is not None and cycles:
                watchlist, steps = _reloaded(reload) or (watchlist, steps)
            cfg = watchlist.alerts
            now = steps.clock()
            _safe("heartbeat", _beat, conn, steps.platforms, now)
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
            while (left := (wake - steps.clock()).total_seconds()) > 0:
                steps.sleep(min(SLEEP_SLICE, max(1.0, left)))
    finally:
        steps.keep_awake.release()


def _reloaded(reload: Reload) -> tuple[Watchlist, Steps] | None:
    try:
        return reload()
    except StopWatch:
        raise
    except Exception:  # a config the watch cannot use must not stop it; it keeps the one it has
        log.exception("watch: reloading the config failed; keeping the current one")
        return None


@contextmanager
def single_instance(path: Path) -> Iterator[bool]:
    """Holds an exclusive OS lock on `path` for the block; yields False when another process holds it. The OS drops
    the lock when its process dies, so a crashed watch never blocks its scheduled restart (a heartbeat check would,
    for up to 15 minutes)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        locked = _try_lock(fd)
        try:
            yield locked
        finally:
            if locked:
                _unlock(fd)
    finally:
        os.close(fd)


def _try_lock(fd: int) -> bool:
    try:
        if sys.platform == "win32":
            import msvcrt

            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return False
    return True


def _unlock(fd: int) -> None:
    if sys.platform == "win32":
        import msvcrt

        os.lseek(fd, 0, os.SEEK_SET)  # msvcrt locks from the file position; unlock the byte that was locked
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_UN)
