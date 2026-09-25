"""X recent-search collector. X is pay-per-use, so every read is capped by the monthly budget and logged per account."""

from __future__ import annotations

import functools
import html
import json
import logging
import math
import re
import sqlite3
import time
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from itertools import product
from typing import NamedTuple

import httpx

from .. import db
from ..config import Watchlist, XConfig
from ..ingest import PostSink
from ..models import Post
from ..timeutil import from_iso, parse_api_time, to_iso

log = logging.getLogger(__name__)

API_BASE = "https://api.x.com/2"
SEARCH_PATH = "/tweets/search/recent"
USERS_PATH = "/users/by"
USER_AGENT = "influence-tracker/0.1 (personal research project)"
TWEET_FIELDS = "created_at,author_id,entities,public_metrics,lang,note_tweet"
REQUEST_TIMEOUT_S = 30.0

MAX_QUERY_LEN = 512
QUERY_SUFFIX = " -is:retweet"
AMBIGUOUS_CONTEXT = (
    '(stock OR stocks OR shares OR earnings OR investors OR buy OR sell OR bullish OR bearish OR "price target")'
)
CASHTAGS_OPERATOR = "has:cashtags"
CASHTAGS_MODE = "cashtags"
EXPLICIT_MODE = "explicit"
USERS_PER_LOOKUP = 100
MIN_RESULTS = 10
MAX_RESULTS = 100

END_LAG = timedelta(seconds=30)
SEARCH_WINDOW = timedelta(days=7)
# start_time must still be inside the 7-day window when X receives the request.
WINDOW_MARGIN = timedelta(minutes=5)
# Scheduled runs drift by minutes; a run 24h05m after the last caught-up one has earned one day, not two.
DAY_SLACK = timedelta(hours=1)

MAX_RATE_LIMIT_WAIT_S = 15 * 60
RATE_LIMIT_MARGIN_S = 1.0
RETRY_BACKOFF_S = (5.0, 20.0)

WATERMARK_SOURCE = "x"
END_KEY = "search_end_time"
PROGRESS_KEY = "query_progress"
PACE_KEY = "pace_anchor"
MODE_KEY = "query_mode"

INVALID_HANDLE_ERROR = "not a valid X username (1-15 letters, digits or underscores)"
_USERNAME = re.compile(r"[A-Za-z0-9_]{1,15}")
_OPERATOR = re.compile(r"-?[a-z_]+:[A-Za-z0-9_]+")
_STATUS_RANK = {"ok": 0, "partial": 1, "error": 2}


# ---------------------------------------------------------------- queries


def _dedupe(items: Iterable[str]) -> list[str]:
    """Strip, drop blanks and drop case-insensitive repeats, keeping the first spelling."""
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        item = item.strip()
        if item and item.lower() not in seen:
            seen.add(item.lower())
            out.append(item)
    return out


def search_terms(watchlist: Watchlist, mode: str) -> list[str]:
    """Search terms for the query's OR-group: has:cashtags (or explicit $SYMBOLs) plus every unambiguous company name.
    Ambiguous names go in separate queries that also require a finance word (see ambiguous_terms)."""
    names = [n for t in watchlist.tickers for n in (*t.names, *t.cased_names)]
    if mode == CASHTAGS_MODE:
        lead = [CASHTAGS_OPERATOR]
    elif mode == EXPLICIT_MODE:
        lead = [f"${s}" for t in watchlist.tickers for s in (t.symbol, *t.aliases)]
    else:
        raise ValueError(f"unknown X query mode {mode!r}")
    return _dedupe([*lead, *names])


def ambiguous_terms(watchlist: Watchlist) -> list[str]:
    return _dedupe(n for t in watchlist.tickers for n in t.ambiguous_names)


def _format_term(term: str) -> str:
    if '"' in term:
        raise ValueError(f"search term {term!r} contains a double quote, which X queries cannot express")
    if term.startswith("$") or _OPERATOR.fullmatch(term) or term.isalnum():
        return term
    # X tokenizes keywords on spaces and punctuation; only an exact-phrase match keeps "AT&T" or "Coca-Cola" whole.
    return f'"{term}"'


def _group(items: list[str]) -> str:
    return items[0] if len(items) == 1 else "(" + " OR ".join(items) + ")"


def _chunk(items: list[str], limit: int) -> list[str]:
    """Greedily pack items into OR-groups no longer than limit. Every single item must fit on its own."""
    groups: list[list[str]] = []
    current: list[str] = []
    chars = 0
    for item in items:
        if current and chars + len(item) + len(" OR ") * len(current) + 2 > limit:
            groups.append(current)
            current, chars = [], 0
        current.append(item)
        chars += len(item)
    groups.append(current)
    return [_group(g) for g in groups]


def build_queries(
    handles: list[str], terms: list[str], max_len: int = MAX_QUERY_LEN, suffix: str = QUERY_SUFFIX
) -> list[str]:
    """`(from:a OR from:b ...) (term OR term ...)<suffix>` queries, each at most max_len characters.

    Accounts and terms are each split into chunks (every one lands in exactly one chunk) and the result is the full
    product of account chunks x term chunks. The split between the two halves is chosen to minimise query count."""
    accounts = [f"from:{h}" for h in _dedupe(handles)]
    formatted = [_format_term(t) for t in _dedupe(terms)]
    if not accounts or not formatted:
        raise ValueError("an X query needs at least one account and one search term")
    budget = max_len - len(suffix) - 1
    min_accounts = max(map(len, accounts))
    min_terms = max(map(len, formatted))
    if min_accounts + min_terms > budget:
        raise ValueError(f"a single account and term cannot fit in a {max_len}-character X query")
    best: tuple[tuple[int, int], list[str], list[str]] | None = None
    for account_limit in range(min_accounts, budget - min_terms + 1):
        account_chunks = _chunk(accounts, account_limit)
        term_chunks = _chunk(formatted, budget - account_limit)
        key = (len(account_chunks) * len(term_chunks), len(account_chunks))
        if best is None or key < best[0]:
            best = (key, account_chunks, term_chunks)
        if len(account_chunks) == 1:
            break
    assert best is not None
    _, account_chunks, term_chunks = best
    return [f"{a} {t}{suffix}" for a, t in product(account_chunks, term_chunks)]


def all_queries(handles: list[str], watchlist: Watchlist, mode: str) -> list[str]:
    queries = build_queries(handles, search_terms(watchlist, mode))
    if ambiguous := ambiguous_terms(watchlist):
        # X keyword matching ignores case, so "target"/"block"/"meta" alone would bill every ordinary post.
        queries += build_queries(handles, ambiguous, suffix=f" {AMBIGUOUS_CONTEXT}{QUERY_SUFFIX}")
    return queries


# ---------------------------------------------------------------- budget


def billing_cycle_start(now: datetime, cycle_day: int) -> datetime:
    """The most recent 00:00 UTC on day-of-month cycle_day that is at or before now."""
    if now.tzinfo is None:
        raise ValueError("naive datetime; attach a timezone first")
    if not 1 <= cycle_day <= 28:
        raise ValueError(f"billing cycle day must be 1-28, got {cycle_day}")
    now = now.astimezone(UTC)
    year, month = now.year, now.month
    if now.day < cycle_day:
        year, month = (year - 1, 12) if month == 1 else (year, month - 1)
    return datetime(year, month, cycle_day, tzinfo=UTC)


def _whole(x: float) -> int:
    # Dollar arithmetic in floats lands just under whole numbers (0.2 / 0.005 = 39.99999...).
    return math.floor(round(x, 6))


def daily_allowance_posts(cfg: XConfig) -> int:
    return _whole(cfg.monthly_budget_usd / 30 / cfg.cost_per_post_read_usd)


def _pace_days(now: datetime, anchor: datetime) -> int:
    return max(1, math.ceil((now - anchor - DAY_SLACK) / timedelta(days=1)))


def _post_reads_since(conn: sqlite3.Connection, since: datetime) -> int:
    row = conn.execute(
        "SELECT COALESCE(SUM(units), 0) AS n FROM x_usage WHERE kind = 'post_read' AND ts > ?", (to_iso(since),)
    ).fetchone()
    return int(row["n"])


def _read_cap(conn: sqlite3.Connection, cfg: XConfig, now: datetime, anchor: datetime) -> int:
    """Posts this run may read: the daily allowance for each day since the last caught-up run (`anchor`, at most
    max_catchup_days of it), minus what unfinished runs since then already read, within the cycle's remaining budget."""
    allowance = daily_allowance_posts(cfg)
    days = _pace_days(now, anchor)
    unspent = allowance * days - _post_reads_since(conn, anchor)
    spent = db.x_spend_since(conn, billing_cycle_start(now, cfg.billing_cycle_day))
    affordable = _whole((cfg.monthly_budget_usd - spent) / cfg.cost_per_post_read_usd)
    return max(0, min(allowance * min(days, cfg.max_catchup_days), unspent, affordable))


def x_budget_summary(conn: sqlite3.Connection, watchlist: Watchlist, now: datetime) -> dict:
    """Spend in the current billing cycle, overall and per account (most expensive first)."""
    cfg = watchlist.x
    cycle_start = billing_cycle_start(now, cfg.billing_cycle_day)
    spent = db.x_spend_since(conn, cycle_start)
    rows = conn.execute(
        """SELECT handle,
                  SUM(CASE WHEN kind = 'post_read' THEN units ELSE 0 END) AS post_reads,
                  SUM(CASE WHEN kind = 'user_read' THEN units ELSE 0 END) AS user_reads,
                  SUM(est_cost_usd) AS cost
           FROM x_usage WHERE ts >= ? GROUP BY handle ORDER BY cost DESC, handle""",
        (to_iso(cycle_start),),
    ).fetchall()
    return {
        "cycle_start": cycle_start,
        "budget_usd": cfg.monthly_budget_usd,
        "spent_usd": round(spent, 6),
        "remaining_usd": round(cfg.monthly_budget_usd - spent, 6),
        "per_handle": [
            {
                "handle": r["handle"],
                "post_reads": int(r["post_reads"]),
                "user_reads": int(r["user_reads"]),
                "cost_usd": round(float(r["cost"]), 6),
            }
            for r in rows
        ],
        "daily_allowance_posts": daily_allowance_posts(cfg),
    }


# ---------------------------------------------------------------- HTTP


class _Abort(Exception):
    """Ends the run: nothing further can succeed (bad token, no credits, rate or usage cap, X unavailable)."""

    def __init__(self, status: str, message: str) -> None:
        super().__init__(message)
        self.status = status


class _RequestFailed(Exception):
    """A failure that only affects the current query or lookup batch."""

    def __init__(self, message: str, status_code: int | None = None, error_text: str = "") -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_text = error_text


class _CashtagsRejected(Exception):
    """X refused the has:cashtags operator for this app."""


class _WindowPassed(Exception):
    """The rest of a query's unread range has aged out of the 7-day search window."""


def _json(resp: httpx.Response) -> object:
    try:
        return resp.json()
    except ValueError:
        return None


def _describe(resp: httpx.Response) -> str:
    body = _json(resp)
    if isinstance(body, dict):
        errors = body.get("errors")
        if isinstance(errors, list) and errors and isinstance(errors[0], dict):
            message = errors[0].get("message") or errors[0].get("detail")
            if message:
                return str(message)[:300]
        for key in ("detail", "title", "reason"):
            if body.get(key):
                return str(body[key])[:300]
    return " ".join(resp.text.split())[:200] or resp.reason_phrase


def _error_text(resp: httpx.Response) -> str:
    """The human-readable error messages only; X also echoes the offending query in errors[].parameters."""
    body = _json(resp)
    if not isinstance(body, dict):
        return resp.text
    parts = [str(body.get(key) or "") for key in ("title", "detail")]
    for err in body.get("errors") or []:
        if isinstance(err, dict):
            parts.extend(str(err.get(key) or "") for key in ("message", "detail", "title"))
    return "\n".join(p for p in parts if p)


def _is_usage_cap(resp: httpx.Response) -> bool:
    body = _json(resp)
    if not isinstance(body, dict):
        return False
    return str(body.get("type", "")).endswith("/usage-capped") or body.get("title") == "UsageCapExceeded"


def _auth_message(resp: httpx.Response) -> str:
    hint = {
        401: "the bearer token is missing, invalid or revoked",
        402: "the X developer account has no API credits left or has hit its spending cap",
        403: "the app may not use this endpoint, or the X developer account has no API credits",
    }[resp.status_code]
    return f"X refused the request with HTTP {resp.status_code} ({_describe(resp)}): {hint}"


class _XApi:
    def __init__(self, http: httpx.Client, token: str, sleep: Callable[[float], None], now: datetime) -> None:
        self.http = http
        self.headers = {"Authorization": f"Bearer {token}", "User-Agent": USER_AGENT}
        self.sleep = sleep
        self.now = now
        self.slept = 0.0

    def clock(self) -> datetime:
        """The run's start plus every pause so far (request time itself is covered by WINDOW_MARGIN)."""
        return self.now + timedelta(seconds=self.slept)

    def _pause(self, seconds: float) -> None:
        self.sleep(seconds)
        self.slept += seconds

    def _rate_limit_wait(self, resp: httpx.Response) -> float | None:
        try:
            reset = float(resp.headers["x-rate-limit-reset"])
        except (KeyError, ValueError):
            return None
        if not math.isfinite(reset):
            return None
        # X's Date header is the clock the reset epoch belongs to.
        server_now = self.clock()
        if "date" in resp.headers:
            try:
                server_now = parsedate_to_datetime(resp.headers["date"])
            except (TypeError, ValueError):
                pass
            if server_now.tzinfo is None:
                server_now = server_now.replace(tzinfo=UTC)
        return max(0.0, reset - server_now.timestamp()) + RATE_LIMIT_MARGIN_S

    def get(self, path: str, params: Callable[[], dict[str, str | int]]) -> dict:
        """GET with retries. `params` is rebuilt for every attempt, since a retry can come many minutes later."""
        rate_limit_retried = False
        failures = 0
        while True:
            try:
                resp = self.http.get(API_BASE + path, params=params(), headers=self.headers)
            except httpx.RequestError as exc:
                problem = f"{type(exc).__name__}: {exc}"
            else:
                code = resp.status_code
                if code == 200:
                    body = _json(resp)
                    if not isinstance(body, dict):
                        raise _RequestFailed(f"HTTP 200 with an unparseable body ({len(resp.content)} bytes)", 200)
                    return body
                if code in (401, 402, 403):
                    raise _Abort("error", _auth_message(resp))
                if code == 429:
                    if _is_usage_cap(resp):
                        raise _Abort(
                            "partial", f"X usage cap reached (HTTP 429: {_describe(resp)}); no reads until it resets"
                        )
                    wait = self._rate_limit_wait(resp)
                    if rate_limit_retried or wait is None or wait > MAX_RATE_LIMIT_WAIT_S:
                        when = "no reset time given" if wait is None else f"resets in {wait:.0f} s"
                        raise _Abort("partial", f"X rate limit (HTTP 429) on {path}, {when}; stopped this run")
                    log.warning("X rate limit (HTTP 429) on %s; retrying once in %.0f s", path, wait)
                    self._pause(wait)
                    rate_limit_retried = True
                    continue
                if code < 500:
                    raise _RequestFailed(f"HTTP {code}: {_describe(resp)}", code, _error_text(resp))
                problem = f"HTTP {code}: {_describe(resp)}"
            if failures == len(RETRY_BACKOFF_S):
                raise _Abort("partial", f"X API unavailable after {failures + 1} attempts on {path} ({problem})")
            log.warning("X %s failed (%s); retrying in %.0f s", path, problem, RETRY_BACKOFF_S[failures])
            self._pause(RETRY_BACKOFF_S[failures])
            failures += 1


# ---------------------------------------------------------------- posts


def _cashtags(entities: object) -> tuple[str, ...]:
    if not isinstance(entities, dict):
        return ()
    tags = [
        c["tag"].strip().upper()
        for c in entities.get("cashtags") or []
        if isinstance(c, dict) and isinstance(c.get("tag"), str) and c["tag"].strip()
    ]
    return tuple(dict.fromkeys(tags))


def _to_post(item: dict, handle_by_id: dict[str, str]) -> Post | None:
    native_id = str(item.get("id") or "")
    created_raw = item.get("created_at")
    if not native_id or not isinstance(created_raw, str):
        log.warning("X post %s lacks an id or created_at; skipped", native_id or "(no id)")
        return None
    try:
        created = parse_api_time(created_raw)
    except ValueError:
        log.warning("X post %s has an unparseable created_at %r; skipped", native_id, created_raw)
        return None
    author_id = str(item.get("author_id") or "") or None
    handle = handle_by_id.get(author_id) if author_id else None
    note = item.get("note_tweet")
    if isinstance(note, dict) and isinstance(note.get("text"), str) and note["text"]:
        # Long posts come back truncated in `text`; the note carries the full text and its entities.
        text = note["text"]
        entities = note["entities"] if isinstance(note.get("entities"), dict) else item.get("entities")
    else:
        text = str(item.get("text") or "")
        entities = item.get("entities")
    metrics = item.get("public_metrics")
    return Post(
        platform="x",
        native_id=native_id,
        author=handle or author_id or "unknown",
        created_at_utc=created,
        # X sends &, < and > as HTML entities; "AT&amp;T" would never match the name AT&T.
        text=html.unescape(text),
        url=f"https://x.com/{handle or 'i'}/status/{native_id}",
        author_id=author_id,
        metrics=metrics if isinstance(metrics, dict) else None,
        cashtag_hints=_cashtags(entities),
    )


def _page_items(body: dict) -> tuple[list[dict], str | None]:
    data = body.get("data", [])
    meta = body.get("meta", {})
    if not isinstance(data, list) or not isinstance(meta, dict):
        raise ValueError("unexpected response shape (data/meta)")
    errors = body.get("errors")
    if isinstance(errors, list) and errors:
        log.warning("X search page carried %d partial error(s), first: %s", len(errors), errors[0])
    token = meta.get("next_token")
    return [i for i in data if isinstance(i, dict)], (str(token) if token else None)


# ---------------------------------------------------------------- per-query progress


def _optional_time(raw: object) -> datetime | None:
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise ValueError(f"expected a timestamp, got {raw!r}")
    return from_iso(raw)


@dataclass
class _Coverage:
    """How far one query has read. Every post created before `done_to` has been read, except the unread `backlog`
    range [start, end); a None start reaches as far back as the 7-day window allows (a first run's promise)."""

    done_to: datetime | None = None
    backlog: tuple[datetime | None, datetime] | None = None

    @property
    def read_through(self) -> datetime | None:
        """Every post created before this has been read; None if that holds for no point yet."""
        return self.backlog[0] if self.backlog else self.done_to

    def caught_up(self, end: datetime) -> bool:
        return self.backlog is None and self.done_to is not None and self.done_to >= end

    def to_json(self) -> dict:
        backlog = None
        if self.backlog is not None:
            start, stop = self.backlog
            backlog = [to_iso(start) if start else None, to_iso(stop)]
        return {"done_to": to_iso(self.done_to) if self.done_to else None, "backlog": backlog}

    @classmethod
    def from_json(cls, raw: object) -> _Coverage:
        if not isinstance(raw, dict):
            raise ValueError(f"expected an object, got {raw!r}")
        done_to = _optional_time(raw.get("done_to"))
        backlog = raw.get("backlog")
        if backlog is None:
            return cls(done_to)
        if not (isinstance(backlog, list) and len(backlog) == 2 and isinstance(backlog[1], str)):
            raise ValueError(f"malformed backlog {backlog!r}")
        return cls(done_to, (_optional_time(backlog[0]), from_iso(backlog[1])))


class _Paging(NamedTuple):
    """A pagination under way: next_token is only valid with the start_time/end_time it was issued for."""

    token: str
    start_time: str
    end_time: str


def _merge(ranges: Iterable[tuple[datetime, datetime]]) -> list[tuple[datetime, datetime]]:
    merged: list[tuple[datetime, datetime]] = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


# ---------------------------------------------------------------- the run


class _XRun:
    def __init__(
        self,
        conn: sqlite3.Connection,
        watchlist: Watchlist,
        sink: PostSink,
        api: _XApi,
        run_id: int,
        now: datetime,
        counts: dict,
    ) -> None:
        self.conn = conn
        self.watchlist = watchlist
        self.cfg = watchlist.x
        self.sink = sink
        self.api = api
        self.run_id = run_id
        self.now = now
        self.counts = counts
        self.mode = counts["query_mode"]
        self.cap = 0
        self.seen: set[str] = set()
        self.handle_by_id: dict[str, str] = {}
        self.coverage: dict[str, _Coverage] = {}
        self.queried: set[str] = set()
        self.gaps: set[tuple[datetime, datetime]] = set()
        self.sent: dict[str, str | int] = {}
        self.status = "ok"
        self.problems: list[str] = []
        self.cost_usd = 0.0

    def problem(self, status: str, message: str) -> None:
        log.warning("%s", message)
        self.problems.append(message)
        if _STATUS_RANK[status] > _STATUS_RANK[self.status]:
            self.status = status

    def execute(self) -> None:
        self._resolve_handles()
        handles = self._searchable_handles()
        if not handles:
            return
        end = self.now - END_LAG
        watermark = self._load_time(END_KEY)
        anchor = self._load_time(PACE_KEY)
        pace_from = anchor or watermark or self.now - timedelta(days=self.cfg.max_catchup_days)
        self.cap = self.counts["cap"] = _read_cap(self.conn, self.cfg, self.now, pace_from)
        try:
            self._search(handles, watermark, end)
        except _Abort as stop:
            self.problem(stop.status, str(stop))
        self._finish(handles, watermark, end, None if anchor else pace_from)

    def _load_time(self, key: str) -> datetime | None:
        raw = db.get_watermark(self.conn, WATERMARK_SOURCE, key)
        if raw is None:
            return None
        try:
            return from_iso(raw)
        except ValueError:
            log.warning("X watermark %s=%r is unreadable; ignoring it", key, raw)
            return None

    def _floor(self) -> datetime:
        return self.api.clock() - SEARCH_WINDOW + WINDOW_MARGIN

    def _finish(self, handles: list[str], watermark: datetime | None, end: datetime, anchor: datetime | None) -> None:
        """Save progress and pacing, and report posts that aged out of the window before they could be read."""
        if not self.coverage:
            return
        self._save_progress()
        caught_up = all(cov.caught_up(end) for cov in self.coverage.values())
        with self.conn:
            if caught_up:
                db.set_watermark(self.conn, WATERMARK_SOURCE, PACE_KEY, to_iso(self.now), self.now)
                for handle in handles:
                    db.touch_account_fetch(self.conn, "x", handle, self.now)
            elif anchor is not None:
                # Unfinished runs until the next caught-up one all draw on the allowance counted from this anchor.
                db.set_watermark(self.conn, WATERMARK_SOURCE, PACE_KEY, to_iso(anchor), self.now)
        through = self._read_through()
        self.counts["caught_up"] = caught_up
        self.counts["watermark_advanced"] = through is not None and (watermark is None or through > watermark)
        self.counts["read_through"] = to_iso(through) if through else None
        if self.gaps:
            gaps = _merge(self.gaps)
            self.counts["window_gaps"] = [[to_iso(start), to_iso(stop)] for start, stop in gaps]
            spans = ", ".join(f"{to_iso(start)} to {to_iso(stop)}" for start, stop in gaps)
            self.problem(
                "partial",
                f"X posts created {spans} were never read and have aged out of the 7-day search window; "
                "they cannot be collected",
            )

    # ------------------------------------------------------------ handles

    def _resolve_handles(self) -> None:
        pending: list[str] = []
        for row in db.get_accounts(self.conn, "x"):
            if row["platform_user_id"]:
                continue
            if _USERNAME.fullmatch(row["handle"]):
                pending.append(row["handle"])
            else:
                log.warning("X handle %r is not a valid username; fix it in watchlist.yaml", row["handle"])
                with self.conn:
                    db.set_account_resolution(self.conn, "x", row["handle"], None, INVALID_HANDLE_ERROR, self.now)
        for i in range(0, len(pending), USERS_PER_LOOKUP):
            batch = pending[i : i + USERS_PER_LOOKUP]
            try:
                body = self.api.get(USERS_PATH, functools.partial(dict, usernames=",".join(batch)))
            except _RequestFailed as exc:
                self.problem("partial", f"X user lookup failed ({exc}); {len(batch)} handle(s) stay unresolved")
                continue
            self._apply_lookup(batch, body)

    def _apply_lookup(self, batch: list[str], body: dict) -> None:
        wanted = {h.lower(): h for h in batch}
        cost = self.cfg.cost_per_user_read_usd
        found = 0
        with self.conn:
            for user in body.get("data") or []:
                if not isinstance(user, dict) or not user.get("id"):
                    continue
                handle = wanted.pop(str(user.get("username", "")).lower(), None)
                if handle is None:
                    continue
                db.set_account_resolution(self.conn, "x", handle, str(user["id"]), None, self.now)
                db.record_x_usage(self.conn, self.run_id, self.now, handle, "user_read", 1, cost)
                found += 1
            for err in body.get("errors") or []:
                if not isinstance(err, dict):
                    continue
                handle = wanted.pop(str(err.get("value") or err.get("resource_id") or "").lower(), None)
                if handle is not None:
                    detail = str(err.get("detail") or err.get("title") or "lookup failed")
                    db.set_account_resolution(self.conn, "x", handle, None, detail, self.now)
            for handle in wanted.values():
                db.set_account_resolution(
                    self.conn, "x", handle, None, "X returned neither this user nor an error for it", self.now
                )
        self.counts["user_reads"] += found
        self.cost_usd += found * cost
        log.info("X user lookup: %d of %d handle(s) resolved", found, len(batch))

    def _searchable_handles(self) -> list[str]:
        rows = sorted(db.get_accounts(self.conn, "x", active_only=False), key=lambda r: r["active"])
        self.handle_by_id = {r["platform_user_id"]: r["handle"] for r in rows if r["platform_user_id"]}
        resolved = {r["handle"] for r in rows if r["active"] and r["platform_user_id"]}
        active = [a.handle for a in self.cfg.accounts if a.active]
        if not active:
            log.info("no active X accounts configured; nothing to search")
            return []
        handles = [h for h in active if h in resolved]
        if not handles:
            self.problem("error", "no X account resolved to a user id; check the handles in watchlist.yaml")
        return handles

    # ------------------------------------------------------------ progress

    def _load_coverage(self, queries: list[str], watermark: datetime | None) -> dict[str, _Coverage]:
        """Each query's saved progress. A query not seen before (an account or term added, or the mode switched)
        starts at the overall watermark, the point every saved query had read up to."""
        stored = self._stored_progress()
        floor = self._floor()
        coverage: dict[str, _Coverage] = {}
        for query in queries:
            cov = stored[query] if query in stored else _Coverage(watermark)
            self._clamp(cov, floor)
            coverage[query] = cov
        return coverage

    def _stored_progress(self) -> dict[str, _Coverage]:
        raw = db.get_watermark(self.conn, WATERMARK_SOURCE, PROGRESS_KEY)
        if raw is None:
            return {}
        try:
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise ValueError("not a JSON object")
            return {str(query): _Coverage.from_json(cov) for query, cov in data.items()}
        except ValueError as exc:
            log.warning("X query progress is unreadable (%s); every query resumes from the watermark", exc)
            return {}

    def _clamp(self, cov: _Coverage, floor: datetime) -> None:
        """Give up what has aged out of the 7-day window, recording it as a gap wherever a watermark had promised it."""
        if cov.backlog is not None:
            start, stop = cov.backlog
            if start is not None and start < floor:
                self.gaps.add((start, min(stop, floor)))
                start = floor
            cov.backlog = (start, stop) if stop > (start or floor) else None
        if cov.backlog is None and cov.done_to is not None and cov.done_to < floor:
            self.gaps.add((cov.done_to, floor))
            cov.done_to = floor

    def _read_through(self) -> datetime | None:
        marks = [cov.read_through for cov in self.coverage.values()]
        if not marks or any(mark is None for mark in marks):
            return None
        return min(marks)

    def _save_progress(self) -> None:
        progress = json.dumps({query: cov.to_json() for query, cov in self.coverage.items()})
        through = self._read_through()
        with self.conn:
            db.set_watermark(self.conn, WATERMARK_SOURCE, PROGRESS_KEY, progress, self.now)
            if through is not None:
                db.set_watermark(self.conn, WATERMARK_SOURCE, END_KEY, to_iso(through), self.now)

    # ------------------------------------------------------------ search

    def _search(self, handles: list[str], watermark: datetime | None, end: datetime) -> None:
        while True:
            try:
                queries = all_queries(handles, self.watchlist, self.mode)
            except ValueError as exc:
                raise _Abort("error", f"cannot build X search queries: {exc}") from exc
            self.coverage = self._load_coverage(queries, watermark)
            work = self._work(queries, end)
            if not work:
                log.info("X search is caught up to %s; nothing to search", to_iso(end))
                return
            if self.cap < MIN_RESULTS:
                self.counts["budget_stopped"] = True
                self.problem(
                    "partial",
                    f"X read budget too low for one page (cap {self.cap} posts); search skipped and the watermark "
                    "stays put until the daily allowance or the billing cycle allows a page",
                )
                return
            log.info(
                "X search from %s to %s over %d account(s), %d quer(ies) with unread posts, read cap %d, mode %s",
                to_iso(work[0][0]),
                to_iso(end),
                len(handles),
                len({i for _, i, _ in work}),
                self.cap,
                self.mode,
            )
            try:
                self._read(queries, work, end)
                return
            except _CashtagsRejected as exc:
                self.mode = self.counts["query_mode"] = EXPLICIT_MODE
                with self.conn:
                    db.set_watermark(self.conn, WATERMARK_SOURCE, MODE_KEY, EXPLICIT_MODE, self.now)
                log.warning("X rejected has:cashtags (%s); using explicit $TICKER terms from now on", exc)

    def _work(self, queries: list[str], end: datetime) -> list[tuple[datetime, int, bool]]:
        """(start, query index, is new range) for every unread range, oldest first: posts closest to ageing out of
        the window are read before newer ones, and every query's old backlog before any query's new day."""
        floor = self._floor()
        work: list[tuple[datetime, int, bool]] = []
        for i, query in enumerate(queries):
            cov = self.coverage[query]
            if cov.backlog is not None:
                work.append((cov.backlog[0] or floor, i, False))
            if cov.done_to is None or cov.done_to < end:
                work.append((cov.done_to or floor, i, True))
        return sorted(work)

    def _read(self, queries: list[str], work: list[tuple[datetime, int, bool]], end: datetime) -> None:
        for _, i, is_new in work:
            cov = self.coverage[queries[i]]
            if is_new:
                if cov.backlog is not None:
                    continue  # its older range failed this run, and a query holds one unread range at a time
                cov.backlog, cov.done_to = (cov.done_to, end), end
            self._read_backlog(queries[i], cov)
            if self.counts["budget_stopped"]:
                return

    def _read_backlog(self, query: str, cov: _Coverage) -> None:
        """Page through the query's unread range newest first, saving progress after every stored page, until the
        range is read, the read cap is reached or a request fails."""
        self.queried.add(query)
        self.counts["queries"] = len(self.queried)
        paging: _Paging | None = None
        while cov.backlog is not None:
            if len(self.seen) >= self.cap:
                self._stop_for_budget()
                return
            try:
                body = self.api.get(SEARCH_PATH, functools.partial(self._search_params, query, cov, paging))
            except _WindowPassed:
                return
            except _RequestFailed as exc:
                if (
                    exc.status_code == 400
                    and self.mode == CASHTAGS_MODE
                    and CASHTAGS_OPERATOR in query
                    and CASHTAGS_OPERATOR in exc.error_text
                ):
                    raise _CashtagsRejected(str(exc)) from exc
                self.problem("partial", f"X search failed ({exc}) for query {query[:60]!r}...")
                return
            sent = self.sent
            self.counts["pages"] += 1
            try:
                items, next_token = _page_items(body)
            except ValueError as exc:
                self.problem("partial", f"X search page could not be read ({exc}) for query {query[:60]!r}...")
                return
            self._bill(items)
            posts = [p for p in (_to_post(i, self.handle_by_id) for i in items) if p is not None]
            if posts:
                result = self.sink.store(posts)
                self.counts["stored"] += result.stored
                self.counts["new"] += result.new
                self.counts["with_mentions"] += result.with_mentions
            if next_token is None:
                cov.backlog, paging = None, None
            else:
                paging = _Paging(next_token, str(sent["start_time"]), str(sent["end_time"]))
                if posts:
                    # Posts sharing the oldest one's second may still be on the next page, so that second stays unread.
                    oldest = min(p.created_at_utc for p in posts).replace(microsecond=0) + timedelta(seconds=1)
                    start, stop = cov.backlog
                    cov.backlog = (start, min(stop, oldest))
            self._save_progress()

    def _search_params(self, query: str, cov: _Coverage, paging: _Paging | None) -> dict[str, str | int]:
        # Rebuilt per attempt: after a long rate-limit pause the 7-day floor has moved past the old start_time.
        floor = self._floor()
        self._clamp(cov, floor)
        if cov.backlog is None:
            raise _WindowPassed
        start = to_iso(cov.backlog[0] or floor)
        params: dict[str, str | int] = {
            "query": query,
            "start_time": start,
            "end_time": to_iso(cov.backlog[1]),
            "max_results": min(max(self.cap - len(self.seen), MIN_RESULTS), MAX_RESULTS),
            "sort_order": "recency",
            "tweet.fields": TWEET_FIELDS,
        }
        if paging is not None and paging.start_time == start:
            params["end_time"] = paging.end_time
            params["next_token"] = paging.token
        self.sent = params
        return params

    def _stop_for_budget(self) -> None:
        self.counts["budget_stopped"] = True
        through = self._read_through()
        self.problem(
            "partial",
            f"X read cap of {self.cap} posts reached; the watermark stays at {to_iso(through) if through else 'unset'} "
            "and the next run first reads the older posts this one had to leave",
        )

    def _bill(self, items: list[dict]) -> None:
        """Log reads not yet counted this run (X bills a post once per UTC day), attributed to the author."""
        units: Counter[str | None] = Counter()
        for item in items:
            native_id = str(item.get("id") or "")
            if not native_id or native_id in self.seen:
                continue
            self.seen.add(native_id)
            author_id = str(item.get("author_id") or "")
            units[self.handle_by_id.get(author_id, author_id or None)] += 1
        if units:
            cost = self.cfg.cost_per_post_read_usd
            with self.conn:
                for handle, n in units.items():
                    db.record_x_usage(self.conn, self.run_id, self.now, handle, "post_read", n, n * cost)
            self.cost_usd += units.total() * cost
        self.counts["posts_read"] = len(self.seen)


def collect_x(
    conn: sqlite3.Connection,
    watchlist: Watchlist,
    token: str,
    sink: PostSink,
    run_id: int,
    now: datetime,
    client: httpx.Client | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> dict:
    """Search recent posts from the watched X accounts that mention a stock. Never raises on network trouble."""
    counts: dict = {
        "status": "ok",
        "queries": 0,
        "pages": 0,
        "posts_read": 0,
        "user_reads": 0,
        "stored": 0,
        "new": 0,
        "with_mentions": 0,
        "cap": 0,
        "budget_stopped": False,
        "watermark_advanced": False,
        "caught_up": False,
        "read_through": None,
        "window_gaps": [],
        "est_cost_usd": 0.0,
        "query_mode": db.get_watermark(conn, WATERMARK_SOURCE, MODE_KEY) or CASHTAGS_MODE,
        "unresolved_handles": [],
        "error": None,
    }
    if not token:
        counts["status"] = "error"
        counts["error"] = "no X bearer token (set X_BEARER_TOKEN in .env)"
        log.error("%s", counts["error"])
        return counts
    http = client or httpx.Client(timeout=REQUEST_TIMEOUT_S)
    run = _XRun(conn, watchlist, sink, _XApi(http, token, sleep, now), run_id, now, counts)
    try:
        run.execute()
    except _Abort as stop:
        run.problem(stop.status, str(stop))
    finally:
        if client is None:
            http.close()
    counts["unresolved_handles"] = sorted(r["handle"] for r in db.get_accounts(conn, "x") if not r["platform_user_id"])
    if counts["unresolved_handles"]:
        log.warning("X handles that did not resolve: %s", ", ".join(counts["unresolved_handles"]))
    counts["est_cost_usd"] = round(run.cost_usd, 6)
    counts["status"] = run.status
    counts["error"] = "; ".join(run.problems) or None
    log.info("x (run %d): %s", run_id, counts)
    return counts
