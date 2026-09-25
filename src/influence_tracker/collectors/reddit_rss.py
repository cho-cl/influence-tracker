"""Top-of-day posts per subreddit from Reddit's public RSS feeds (the .json API refuses unauthenticated clients)."""

from __future__ import annotations

import logging
import math
import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from html.parser import HTMLParser

import feedparser
import httpx

from ..config import RedditConfig, Watchlist
from ..ingest import PostSink
from ..models import Post
from ..timeutil import parse_api_time

log = logging.getLogger(__name__)

RSS_URL = "https://www.reddit.com/r/{sub}/top/.rss"
REQUEST_TIMEOUT_S = 30.0
# Unauthenticated RSS allows about one request per minute; used when Reddit sends no reset header.
DEFAULT_RESET_S = 60.0
RESET_MARGIN_S = 2.0
# A malformed reset header must not stall the daily run.
MAX_WAIT_S = 600.0

# Daily/weekly discussion threads sit at the top of the feed every day but are boilerplate, not a signal.
BOT_AUTHOR = "AutoModerator"

# Reddit's stand-in body for posts old Reddit cannot render (polls, rich embeds); not the author's words.
_OLD_REDDIT_NOTICE = "This post contains content not supported on old Reddit."


class _SelfTextExtractor(HTMLParser):
    """Collects the text inside Reddit's <div class="md"> (the post body). Everything outside it is the feed's
    thumbnail table, 'submitted by' line and [link]/[comments] anchors."""

    _BREAKS = frozenset(
        {"p", "div", "br", "hr", "li", "tr", "pre", "blockquote", "table", "h1", "h2", "h3", "h4", "h5", "h6"}
    )
    _CELLS = frozenset({"td", "th"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._depth = 0
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self._depth == 0:
            if tag == "div" and "md" in (dict(attrs).get("class") or "").split():
                self._depth = 1
            return
        if tag == "div":
            self._depth += 1
        if tag in self._BREAKS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if self._depth == 0:
            return
        if tag == "div":
            self._depth -= 1
        if tag in self._BREAKS:
            self._parts.append("\n")
        elif tag in self._CELLS:
            self._parts.append(" ")

    def handle_data(self, data: str) -> None:
        if self._depth:
            self._parts.append(data)

    def text(self) -> str:
        lines = (" ".join(line.split()) for line in "".join(self._parts).splitlines())
        return "\n".join(line for line in lines if line and not line.startswith(_OLD_REDDIT_NOTICE))


def extract_selftext(content_html: str) -> str:
    """Plain-text self-text of a Reddit RSS entry's content HTML, or '' for link/image posts without a body."""
    parser = _SelfTextExtractor()
    parser.feed(content_html)
    parser.close()
    return parser.text()


def _entry_to_post(entry: feedparser.FeedParserDict, rank: int, subreddit: str) -> Post | None:
    native_id = (entry.get("id") or "").strip()
    url = (entry.get("link") or "").strip()
    published = entry.get("published")
    if not native_id or not url or not published:
        log.warning("r/%s entry #%d lacks an id, link or published time; skipped", subreddit, rank)
        return None
    try:
        created = parse_api_time(published)
    except ValueError:
        log.warning("r/%s entry %s has an unparseable published time %r; skipped", subreddit, native_id, published)
        return None
    author = (entry.get("author") or "").strip().removeprefix("/u/").strip() or "[deleted]"
    if author == BOT_AUTHOR:
        return None
    title = " ".join((entry.get("title") or "").split())
    content = entry.get("content") or []
    body_html = content[0].get("value", "") if content else entry.get("summary", "")
    selftext = extract_selftext(body_html)
    return Post(
        platform="reddit",
        native_id=native_id,
        author=author,
        created_at_utc=created,
        text=f"{title}\n\n{selftext}" if selftext else title,
        url=url,
        source=subreddit,
        feed_rank=rank,
    )


def entries_to_posts(feed: feedparser.FeedParserDict, subreddit: str) -> list[Post]:
    posts = (_entry_to_post(entry, i + 1, subreddit) for i, entry in enumerate(feed.entries))
    return [p for p in posts if p is not None]


def parse_feed(data: bytes, subreddit: str) -> list[Post]:
    return entries_to_posts(feedparser.parse(data), subreddit)


def _header_seconds(headers: httpx.Headers | None, name: str) -> float | None:
    if headers is None or name not in headers:
        return None
    try:
        value = float(headers[name])
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def _clamp_wait(seconds: float) -> float:
    return min(max(seconds, 0.0), MAX_WAIT_S)


def _pace_wait(headers: httpx.Headers | None) -> float:
    """Seconds to wait before the next request, given the previous response's rate-limit headers."""
    remaining = _header_seconds(headers, "x-ratelimit-remaining")
    if remaining is not None and remaining >= 1:
        return 0.0
    reset = _header_seconds(headers, "x-ratelimit-reset")
    return _clamp_wait((DEFAULT_RESET_S if reset is None else reset) + RESET_MARGIN_S)


def _retry_wait(headers: httpx.Headers | None) -> float:
    return _clamp_wait(_header_seconds(headers, "x-ratelimit-reset") or DEFAULT_RESET_S)


@dataclass
class _Fetch:
    feed: feedparser.FeedParserDict | None  # set only when the response was a usable feed
    headers: httpx.Headers | None  # None when no response arrived
    rate_limited: bool = False


def _fetch_once(http: httpx.Client, sub: str, cfg: RedditConfig) -> _Fetch:
    try:
        resp = http.get(
            RSS_URL.format(sub=sub),
            params={"t": "day", "limit": cfg.rss_limit},
            headers={"User-Agent": cfg.user_agent},
            follow_redirects=True,
        )
    except httpx.HTTPError as exc:
        log.warning("r/%s: request failed: %s", sub, exc)
        return _Fetch(None, None)
    if resp.status_code == 429:
        log.warning("r/%s: HTTP 429", sub)
        return _Fetch(None, resp.headers, rate_limited=True)
    if resp.status_code != 200:
        log.warning("r/%s: HTTP %d", sub, resp.status_code)
        return _Fetch(None, resp.headers)
    feed = feedparser.parse(resp.content)
    if not feed.get("version"):
        # Reddit answers a too-early request with 200 and an empty (or non-feed) body instead of a 429.
        log.warning("r/%s: HTTP 200 but no feed in the %d-byte body", sub, len(resp.content))
        return _Fetch(None, resp.headers, rate_limited=True)
    return _Fetch(feed, resp.headers)


def _status(failed: int, total: int, got_data: bool) -> str:
    if failed == 0:
        return "ok"
    return "error" if failed == total and not got_data else "partial"


def collect_reddit(
    conn: sqlite3.Connection,
    watchlist: Watchlist,
    sink: PostSink,
    run_id: int,
    now: datetime,
    client: httpx.Client | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> dict:
    """Fetch each configured subreddit's top-of-day feed and store it as one page. Never raises on network trouble."""
    cfg = watchlist.reddit
    counts = {
        "status": "ok",
        "requests": 0,
        "stored": 0,
        "new": 0,
        "with_mentions": 0,
        "rate_limited": 0,
        "subreddits_failed": 0,
    }
    http = client or httpx.Client(timeout=REQUEST_TIMEOUT_S)
    last_headers: httpx.Headers | None = None
    try:
        for sub in cfg.subreddits:
            if counts["requests"]:
                wait = _pace_wait(last_headers)
                if wait > 0:
                    log.info("r/%s: waiting %.0f s for Reddit's rate limit to reset", sub, wait)
                    sleep(wait)
            counts["requests"] += 1
            fetch = _fetch_once(http, sub, cfg)
            if fetch.rate_limited:
                wait = _retry_wait(fetch.headers)
                log.info("r/%s: rate limited; retrying once in %.0f s", sub, wait)
                sleep(wait)
                counts["requests"] += 1
                fetch = _fetch_once(http, sub, cfg)
                if fetch.rate_limited:
                    counts["rate_limited"] += 1
                    log.warning("r/%s: still rate limited after one retry; skipped this run", sub)
            last_headers = fetch.headers
            if fetch.feed is None:
                counts["subreddits_failed"] += 1
                continue
            posts = entries_to_posts(fetch.feed, sub)
            if posts:
                result = sink.store(posts)
                counts["stored"] += result.stored
                counts["new"] += result.new
                counts["with_mentions"] += result.with_mentions
                log.info(
                    "r/%s: %d posts (%d new, %d with mentions)", sub, result.stored, result.new, result.with_mentions
                )
            else:
                log.info("r/%s: feed had no entries", sub)
    finally:
        if client is None:
            http.close()
    counts["status"] = _status(counts["subreddits_failed"], len(cfg.subreddits), counts["stored"] > 0)
    log.info("reddit (run %d at %s): %s", run_id, now.isoformat(), counts)
    return counts
