from __future__ import annotations

import contextlib
import logging
import sys
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

LOG_FILE_NAME = "influence.log"
MAX_BYTES = 2 * 1024 * 1024
BACKUP_COUNT = 5
HANDLER_MARK = "_influence_tracker_handler"
# Libraries that log every request at INFO; our collectors log their own per-page summaries.
_QUIET_LOGGERS = ("httpx", "httpcore", "urllib3", "peewee", "filelock")
_NON_TTY_WIDTH = 160


def force_utf8_stdio() -> None:
    """Emoji in post text must never crash a cp1252 Windows console or a redirected scheduled-task log."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            with contextlib.suppress(ValueError, OSError):
                reconfigure(encoding="utf-8", errors="replace")


def make_console(stderr: bool = False) -> Console:
    """Rich console for stdout/stderr. Redirected output (the scheduled task, `> file`, a pipe) would otherwise
    wrap at rich's 80-column default."""
    stream = sys.stderr if stderr else sys.stdout
    is_tty = bool(getattr(stream, "isatty", lambda: False)())
    return Console(stderr=stderr, width=None if is_tty else _NON_TTY_WIDTH)


def remove_handlers() -> None:
    """Detach and close the handlers a previous setup_logging() call installed."""
    root = logging.getLogger()
    for handler in list(root.handlers):
        if getattr(handler, HANDLER_MARK, False):
            root.removeHandler(handler)
            handler.close()


def setup_logging(
    logs_dir: Path, level: int = logging.INFO, *, file_name: str = LOG_FILE_NAME, console_level: int | None = None
) -> Path:
    """Root logger -> rich console (stderr) + rotating UTF-8 file. Safe to call more than once."""
    remove_handlers()
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / file_name

    console_handler = RichHandler(
        console=make_console(stderr=True),
        show_path=False,
        markup=False,
        rich_tracebacks=False,
        log_time_format="[%Y-%m-%d %H:%M:%S]",
    )
    if console_level is not None:
        console_handler.setLevel(console_level)

    # delay: a quiet `influence status` never holds the file open, which would block a rollover rename on Windows.
    file_handler = RotatingFileHandler(
        log_path, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT, encoding="utf-8", delay=True
    )
    formatter = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%Y-%m-%dT%H:%M:%SZ")
    formatter.converter = time.gmtime
    file_handler.setFormatter(formatter)

    root = logging.getLogger()
    for handler in (console_handler, file_handler):
        setattr(handler, HANDLER_MARK, True)
        root.addHandler(handler)
    root.setLevel(level)
    for name in _QUIET_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
    return log_path
