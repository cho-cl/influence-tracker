from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from . import db
from .config import Watchlist
from .models import MatchResult, Post
from .timeutil import utc_now


class Matcher(Protocol):
    def match(self, text: str, platform: str, cashtag_hints: Sequence[str] = ()) -> MatchResult: ...


@dataclass
class StoreResult:
    stored: int = 0
    new: int = 0
    with_mentions: int = 0


class PostSink:
    """Stores posts and their ticker mentions atomically. Collectors call store() once per page
    and only advance their watermark after it returns."""

    def __init__(self, conn: sqlite3.Connection, matcher: Matcher, now: Callable[[], datetime] = utc_now):
        self.conn = conn
        self.matcher = matcher
        self.now = now
        self.totals = StoreResult()

    def store(self, posts: Sequence[Post]) -> StoreResult:
        result = StoreResult()
        when = self.now()
        with self.conn:
            for post in posts:
                is_new = db.upsert_post(self.conn, post, when)
                m = self.matcher.match(post.text, post.platform, post.cashtag_hints)
                db.replace_mentions(self.conn, post.platform, post.native_id, m.mentions, m.unknown_cashtags, when)
                result.stored += 1
                result.new += int(is_new)
                result.with_mentions += int(bool(m.mentions))
        self.totals.stored += result.stored
        self.totals.new += result.new
        self.totals.with_mentions += result.with_mentions
        return result


def universe_fingerprint(watchlist: Watchlist) -> str:
    payload = {
        "tickers": [t.model_dump() for t in watchlist.tickers],
        "stoplist": watchlist.bare_ticker_stoplist,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def rematch_all(conn: sqlite3.Connection, matcher: Matcher, now: Callable[[], datetime] = utc_now) -> int:
    """Recompute mentions for every stored post (after the ticker universe changes)."""
    rows = conn.execute("SELECT platform, native_id, text, cashtag_hints FROM posts").fetchall()
    when = now()
    with conn:
        for r in rows:
            hints = tuple(r["cashtag_hints"].split(",")) if r["cashtag_hints"] else ()
            m = matcher.match(r["text"], r["platform"], hints)
            db.replace_mentions(conn, r["platform"], r["native_id"], m.mentions, m.unknown_cashtags, when)
    return len(rows)


def rematch_if_universe_changed(
    conn: sqlite3.Connection, matcher: Matcher, fingerprint: str, now: Callable[[], datetime] = utc_now
) -> int | None:
    """Returns the number of posts re-matched, or None if the universe is unchanged."""
    if db.get_watermark(conn, "mentions", "universe") == fingerprint:
        return None
    n = rematch_all(conn, matcher, now)
    with conn:
        db.set_watermark(conn, "mentions", "universe", fingerprint, now())
    return n
