"""Truth Social posts from its Mastodon-compatible public API, read without a login, one slow request at a time.

Live probe (2026-09-24): the server ignores `min_id` (and `exclude_reblogs`), always answering with the newest page,
while `max_id` pages backward correctly. So each account is walked newest-to-oldest with `max_id` down to its
watermark; the walk's progress is saved after every page, and the watermark moves only when the walk is complete.
Status ids are Mastodon snowflakes: `id >> 16` is the creation time in unix milliseconds.
"""

from __future__ import annotations

import logging
import re
import sqlite3
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from html.parser import HTMLParser
from typing import Any

import httpx

from .. import db
from ..config import TruthSocialConfig, Watchlist
from ..ingest import PostSink
from ..models import Post
from ..timeutil import parse_api_time

log = logging.getLogger(__name__)

PLATFORM = "truthsocial"
BACKFILL_SOURCE = "truthsocial_backfill"
BASE_URL = "https://truthsocial.com/api/v1"
WEB_URL = "https://truthsocial.com"
# The server returns at most 20 statuses per page whatever `limit` asks for.
PAGE_LIMIT = 20
MAX_PAGES_PER_ACCOUNT = 50
RATE_LIMIT_WAITS_S = (60.0, 120.0)
REQUEST_TIMEOUT_S = 30.0
BACKFILL_LOG_EVERY_PAGES = 25
METRIC_KEYS = ("replies_count", "reblogs_count", "favourites_count")
_CLOUDFLARE_RATE_LIMIT = "error code: 1015"
# Cloudflare's plain-text block pages, e.g. 'error code: 1020' (access denied) or 1006 (IP banned).
_CLOUDFLARE_ERROR = re.compile(r"\s*error code: \d+")
_RUN_STOPPERS = frozenset({"rate_limited", "blocked"})
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
_MS = timedelta(milliseconds=1)


# ---------------------------------------------------------------- snowflake ids


def id_for_time(dt: datetime) -> str:
    """The smallest status id that can be minted at this instant, for use as a time cursor in max_id."""
    if dt.tzinfo is None:
        raise ValueError("naive datetime; attach a timezone first")
    return str(((dt - _EPOCH) // _MS) << 16)


def time_for_id(status_id: str) -> datetime:
    """When a status id was minted, in UTC to the millisecond (live ids sit within a few ms of created_at)."""
    return _EPOCH + (int(status_id) >> 16) * _MS


# ---------------------------------------------------------------- content


class _ContentText(HTMLParser):
    """Visible text of a status's HTML. <p> and <br> become line breaks and link text is kept: Mastodon splits long
    URLs over 'invisible'/'ellipsis' spans that join back into the full URL. Truth Social's quote-post stub,
    <span class="quote-inline"><br/>RT: <quoted status URL></span>, is dropped; the quoted post is not in `content`."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self._skip_depth:
            self._skip_depth += tag == "span"
        elif tag == "span" and "quote-inline" in (dict(attrs).get("class") or "").split():
            self._skip_depth = 1
        elif tag == "br":
            self._parts.append("\n")
        elif tag == "p":
            self._parts.append("\n\n")

    def handle_endtag(self, tag: str) -> None:
        if self._skip_depth:
            self._skip_depth -= tag == "span"
        elif tag == "p":
            self._parts.append("\n\n")

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            self._parts.append(data)

    def text(self) -> str:
        lines = (" ".join(line.split()) for line in "".join(self._parts).split("\n"))
        return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def html_to_text(content: str) -> str:
    parser = _ContentText()
    parser.feed(content)
    parser.close()
    return parser.text()


def status_to_post(status: dict, handle: str, account_id: str) -> Post | None:
    """The author's own words as a Post; None for reblogs, posts with no text (media-only) and other accounts'."""
    if status.get("reblog"):
        return None
    owner = (status.get("account") or {}).get("id")
    if owner is not None and str(owner) != account_id:
        return None
    text = html_to_text(status.get("content") or "")
    if not text:
        return None
    status_id = str(status["id"])
    try:
        created = parse_api_time(status["created_at"])
    except (KeyError, TypeError, ValueError):
        log.warning("truthsocial @%s: status %s has no usable created_at; skipped", handle, status_id)
        return None
    metrics = {k: status[k] for k in METRIC_KEYS if isinstance(status.get(k), int)}
    return Post(
        platform="truthsocial",
        native_id=status_id,
        author=handle,
        author_id=account_id,
        created_at_utc=created,
        text=text,
        url=status.get("url") or f"{WEB_URL}/@{handle}/{status_id}",
        metrics=metrics or None,
    )


# ---------------------------------------------------------------- HTTP


@dataclass(frozen=True)
class _Reply:
    kind: str  # 'ok' | 'refused' (a 4xx JSON answer) | 'rate_limited' | 'blocked' | 'failed'
    data: Any = None
    detail: str = ""


def _error_text(resp: httpx.Response) -> str:
    try:
        data = resp.json()
    except ValueError:
        data = None
    if isinstance(data, dict) and isinstance(data.get("error"), str):
        return data["error"]
    return resp.text[:200].strip()


def _classify(resp: httpx.Response) -> _Reply:
    if resp.status_code == 200:
        try:
            return _Reply("ok", resp.json())
        except ValueError:
            pass
    body = resp.text
    if resp.status_code == 429 or _CLOUDFLARE_RATE_LIMIT in body:
        return _Reply("rate_limited", detail=f"HTTP {resp.status_code}")
    is_html = "html" in resp.headers.get("content-type", "") or body.lstrip().startswith("<")
    if resp.status_code == 403 and is_html:
        return _Reply("blocked", detail="HTTP 403 with an HTML body")
    if resp.status_code == 403 and _CLOUDFLARE_ERROR.match(body):
        return _Reply("blocked", detail=f"HTTP 403 {body.strip()[:40]!r}")
    if resp.status_code == 200:
        return _Reply("failed", detail=f"HTTP 200 but the {len(resp.content)}-byte body is not JSON")
    if 400 <= resp.status_code < 500 and not is_html:
        return _Reply("refused", detail=f"HTTP {resp.status_code}: {_error_text(resp)}")
    return _Reply("failed", detail=f"HTTP {resp.status_code}")


class _Api:
    """Sends one request at a time, each at least `min_interval_s` after the previous one finished, and waits out
    Cloudflare rate limits (60 s, then 120 s) before giving up."""

    def __init__(
        self,
        http: httpx.Client,
        headers: dict[str, str],
        min_interval_s: float,
        sleep: Callable[[float], None],
        clock: Callable[[], float],
        counts: dict,
    ) -> None:
        self._http = http
        self._headers = headers
        self._min_interval_s = min_interval_s
        self._sleep = sleep
        self._clock = clock
        self._counts = counts
        self._last: float | None = None

    def get(self, path: str, params: dict[str, str | int]) -> _Reply:
        reply = self._get_once(path, params)
        for wait in RATE_LIMIT_WAITS_S:
            if reply.kind != "rate_limited":
                break
            log.warning("truthsocial: rate limited (%s); waiting %.0f s before retrying", reply.detail, wait)
            self._sleep(wait)
            reply = self._get_once(path, params)
        return reply

    def _get_once(self, path: str, params: dict[str, str | int]) -> _Reply:
        if self._last is not None:
            wait = self._min_interval_s - (self._clock() - self._last)
            if wait > 0:
                self._sleep(wait)
        self._counts["requests"] += 1
        try:
            resp = self._http.get(BASE_URL + path, params=params, headers=self._headers)
        except httpx.HTTPError as exc:
            return _Reply("failed", detail=f"{type(exc).__name__}: {exc}")
        finally:
            self._last = self._clock()
        return _classify(resp)


# ---------------------------------------------------------------- per-account work


class _Halt(Exception):
    """Ends work on one account. Kinds in _RUN_STOPPERS end the whole run."""

    def __init__(self, kind: str) -> None:
        super().__init__(kind)
        self.kind = kind


def _halt(handle: str, what: str, reply: _Reply) -> _Halt:
    if reply.kind == "blocked":
        log.error(
            "truthsocial @%s: %s was blocked by a Cloudflare challenge (%s); stopping this run, nothing is lost",
            handle,
            what,
            reply.detail,
        )
    elif reply.kind == "rate_limited":
        log.warning("truthsocial @%s: %s still rate limited after backing off; giving up this run", handle, what)
    else:
        log.warning("truthsocial @%s: %s failed (%s); account skipped this run", handle, what, reply.detail)
    return _Halt(reply.kind)


@dataclass
class _Ctx:
    conn: sqlite3.Connection
    sink: PostSink
    api: _Api
    cfg: TruthSocialConfig
    counts: dict
    now: datetime


def _resolve(ctx: _Ctx, handle: str) -> str:
    reply = ctx.api.get("/accounts/lookup", {"acct": handle})
    account_id = reply.data.get("id") if reply.kind == "ok" and isinstance(reply.data, dict) else None
    if account_id is not None:
        with ctx.conn:
            db.set_account_resolution(ctx.conn, PLATFORM, handle, str(account_id), None, ctx.now)
        log.info("truthsocial @%s: resolved to account %s", handle, account_id)
        return str(account_id)
    if reply.kind == "refused":
        with ctx.conn:
            db.set_account_resolution(ctx.conn, PLATFORM, handle, None, reply.detail, ctx.now)
        ctx.counts["unresolved_handles"].append(handle)
        log.warning("truthsocial @%s: handle did not resolve (%s)", handle, reply.detail)
        raise _Halt("failed")
    if reply.kind == "ok":
        reply = _Reply("failed", detail="the lookup answer has no account id")
    raise _halt(handle, "account lookup", reply)


def _parse_page(data: Any) -> list[tuple[int, dict]] | None:
    """(id, status) pairs sorted oldest first, or None unless every element is a status with an integer id."""
    if not isinstance(data, list):
        return None
    page: list[tuple[int, dict]] = []
    for status in data:
        try:
            page.append((int(status["id"]), status))
        except (KeyError, TypeError, ValueError):
            return None
    return sorted(page, key=lambda item: item[0])


def _fetch_page(ctx: _Ctx, handle: str, account_id: str, max_id: int | None) -> list[tuple[int, dict]]:
    params: dict[str, str | int] = {"limit": PAGE_LIMIT, "exclude_reblogs": "true"}
    if max_id is not None:
        params["max_id"] = str(max_id)
    reply = ctx.api.get(f"/accounts/{account_id}/statuses", params)
    if reply.kind != "ok":
        raise _halt(handle, "statuses request", reply)
    page = _parse_page(reply.data)
    if page is None:
        log.warning("truthsocial @%s: statuses answer is not a list of statuses with ids; account skipped", handle)
        raise _Halt("failed")
    ctx.counts["pages"] += 1
    return page


def _store(ctx: _Ctx, handle: str, account_id: str, statuses: Sequence[dict]) -> None:
    posts = [p for p in (status_to_post(s, handle, account_id) for s in statuses) if p is not None]
    result = ctx.sink.store(posts)
    ctx.counts["stored"] += result.stored
    ctx.counts["new"] += result.new
    ctx.counts["with_mentions"] += result.with_mentions


def _load_sweep(conn: sqlite3.Connection, handle: str) -> tuple[int | None, int | None]:
    value = db.get_watermark(conn, PLATFORM, f"sweep:{handle}")
    if not value:
        return None, None
    try:
        top, cursor = (int(part) for part in value.split(":"))
    except ValueError:
        log.warning("truthsocial @%s: unreadable sweep progress %r; starting a fresh sweep", handle, value)
        return None, None
    return top, cursor


def _collect_account(ctx: _Ctx, handle: str, account_id: str) -> str:
    """Walk from the newest status back to the watermark. Watermarks (source 'truthsocial'):
    <handle> = newest status id with everything since first_min_id stored below it;
    first_min_id:<handle> = the first run's start point (also where a backfill begins);
    sweep:<handle> = '<top>:<cursor>' while a walk is unfinished (everything from cursor to top is stored)."""
    conn, now = ctx.conn, ctx.now
    first_key = f"first_min_id:{handle}"
    floor = db.get_watermark(conn, PLATFORM, first_key)
    if floor is None:
        floor = id_for_time(now - timedelta(days=ctx.cfg.first_run_days))
        with conn:
            db.set_watermark(conn, PLATFORM, first_key, floor, now)
    mark = int(db.get_watermark(conn, PLATFORM, handle) or floor)
    top, cursor = _load_sweep(conn, handle)
    resumed = cursor is not None
    for _ in range(MAX_PAGES_PER_ACCOUNT):
        page = _fetch_page(ctx, handle, account_id, cursor)
        ids = [status_id for status_id, _ in page]
        if top is None:
            top = max(ids, default=None)
        _store(ctx, handle, account_id, [s for status_id, s in page if status_id > mark])
        if ids and ids[0] > mark:
            cursor = ids[0]
            with conn:
                db.set_watermark(conn, PLATFORM, f"sweep:{handle}", f"{top}:{cursor}", now)
            continue
        with conn:
            if top is not None and top > mark:
                mark = top
                db.set_watermark(conn, PLATFORM, handle, str(mark), now)
            if cursor is not None:
                db.set_watermark(conn, PLATFORM, f"sweep:{handle}", "", now)
        if not resumed:
            return "done"
        # The finished walk began in an earlier run; walk again for anything posted since it began.
        top = cursor = None
        resumed = False
    log.warning(
        "truthsocial @%s: stopped after %d pages this run; the walk resumes next run", handle, MAX_PAGES_PER_ACCOUNT
    )
    return "capped"


def _backfill_account(ctx: _Ctx, handle: str, account_id: str, floor: int) -> str:
    """Walk backward from where the daily collector's coverage starts down to `floor`. Watermark
    ('truthsocial_backfill', <handle>) = the lowest id reached; everything between it and the start is stored."""
    conn, now = ctx.conn, ctx.now
    saved = db.get_watermark(conn, BACKFILL_SOURCE, handle)
    if saved is not None:
        cursor: int | None = int(saved)
    else:
        stop = db.get_watermark(conn, PLATFORM, f"first_min_id:{handle}") or db.get_watermark(conn, PLATFORM, handle)
        # max_id is exclusive; +1 keeps a status whose id equals the stop point.
        cursor = int(stop) + 1 if stop is not None else None
    pages = 0
    while cursor is None or cursor > floor:
        page = _fetch_page(ctx, handle, account_id, cursor)
        _store(ctx, handle, account_id, [s for status_id, s in page if status_id > floor])
        cursor = max(page[0][0], floor) if page else floor
        with conn:
            db.set_watermark(conn, BACKFILL_SOURCE, handle, str(cursor), now)
        pages += 1
        if pages % BACKFILL_LOG_EVERY_PAGES == 0:
            log.info("truthsocial backfill @%s: back to %s after %d pages", handle, time_for_id(str(cursor)), pages)
    return "done"


# ---------------------------------------------------------------- runs


AccountWork = Callable[[_Ctx, str, str], str]


def _run(
    conn: sqlite3.Connection,
    watchlist: Watchlist,
    sink: PostSink,
    run_id: int,
    now: datetime,
    client: httpx.Client | None,
    sleep: Callable[[float], None],
    clock: Callable[[], float],
    stage: str,
    work: AccountWork,
) -> dict:
    cfg = watchlist.truthsocial
    counts: dict = {
        "status": "ok",
        "requests": 0,
        "stored": 0,
        "new": 0,
        "with_mentions": 0,
        "rate_limited": 0,
        "accounts_ok": 0,
        "accounts_failed": 0,
        "unresolved_handles": [],
        "pages": 0,
    }
    http = client or httpx.Client(timeout=REQUEST_TIMEOUT_S)
    headers = {"User-Agent": cfg.user_agent, "Accept": "application/json"}
    api = _Api(http, headers, cfg.min_request_interval_s, sleep, clock, counts)
    ctx = _Ctx(conn, sink, api, cfg, counts, now)
    capped = halted = False
    try:
        accounts = db.get_accounts(conn, PLATFORM)
        for position, row in enumerate(accounts):
            handle = row["handle"]
            try:
                outcome = work(ctx, handle, row["platform_user_id"] or _resolve(ctx, handle))
            except _Halt as halt:
                counts["accounts_failed"] += 1
                if halt.kind in _RUN_STOPPERS:
                    halted = True
                    counts["rate_limited"] += halt.kind == "rate_limited"
                    left = len(accounts) - position - 1
                    log.warning("%s: run stopped; %d account(s) after @%s wait for the next run", stage, left, handle)
                    break
                continue
            counts["accounts_ok"] += 1
            capped = capped or outcome == "capped"
            with conn:
                db.touch_account_fetch(conn, PLATFORM, handle, now)
    finally:
        if client is None:
            http.close()
    if counts["accounts_failed"] == 0 and not capped:
        counts["status"] = "ok"
    elif halted or capped or counts["accounts_ok"] or counts["stored"]:
        counts["status"] = "partial"
    else:
        counts["status"] = "error"
    log.info("%s (run %d): %s", stage, run_id, counts)
    return counts


def collect_truthsocial(
    conn: sqlite3.Connection,
    watchlist: Watchlist,
    sink: PostSink,
    run_id: int,
    now: datetime,
    client: httpx.Client | None = None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> dict:
    """Daily run: each active account's statuses newer than its watermark (the last `first_run_days` on the first
    run). Never raises on network trouble; see counts['status']."""
    return _run(conn, watchlist, sink, run_id, now, client, sleep, clock, "collect:truthsocial", _collect_account)


def backfill_truthsocial(
    conn: sqlite3.Connection,
    watchlist: Watchlist,
    sink: PostSink,
    run_id: int,
    since: date,
    now: datetime,
    client: httpx.Client | None = None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> dict:
    """One-off history fill from `since` (00:00 UTC) up to where the daily collector's coverage starts. Resumable:
    a rerun continues from the lowest status reached, and a finished backfill costs no requests."""
    floor = int(id_for_time(datetime(since.year, since.month, since.day, tzinfo=UTC)))

    def work(ctx: _Ctx, handle: str, account_id: str) -> str:
        return _backfill_account(ctx, handle, account_id, floor)

    return _run(conn, watchlist, sink, run_id, now, client, sleep, clock, "backfill:truthsocial", work)
