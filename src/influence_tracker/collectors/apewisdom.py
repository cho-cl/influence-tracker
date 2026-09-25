"""Daily per-ticker Reddit mention counts from ApeWisdom. They can't be backfilled, so a missed day stays a gap."""

from __future__ import annotations

import html
import logging
import math
import sqlite3
import time
from collections.abc import Callable
from datetime import datetime

import httpx

from .. import db
from ..config import Watchlist
from ..timeutil import NY, to_iso

log = logging.getLogger(__name__)

API_URL = "https://apewisdom.io/api/v1.0/filter/{filter}/page/{page}"
USER_AGENT = "influence-tracker/0.1 (personal research project)"
REQUEST_TIMEOUT_S = 30.0
REQUEST_PAUSE_S = 1.0
_INT_FIELDS = ("rank", "mentions", "upvotes", "rank_24h_ago", "mentions_24h_ago")


def to_int(value: object) -> int | None:
    """ApeWisdom numbers arrive as ints, numeric strings or null."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if math.isfinite(value) else None
    if isinstance(value, str):
        text = value.strip().replace(",", "")
        try:
            return int(text)
        except ValueError:
            pass
        try:
            number = float(text)
        except ValueError:
            return None
        return int(number) if math.isfinite(number) else None
    return None


def snapshot_date_for(now: datetime) -> str:
    return now.astimezone(NY).date().isoformat()


def parse_page(
    payload: object, filter_name: str, snapshot_date: str, fetched_at: datetime
) -> tuple[list[dict], int | None]:
    """Rows for db.upsert_reddit_ticker_daily, plus the response's total page count (None if absent)."""
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        raise ValueError("ApeWisdom response has no results list")
    fetched_iso = to_iso(fetched_at)
    rows: list[dict] = []
    for item in payload["results"]:
        if not isinstance(item, dict):
            continue
        ticker = str(item.get("ticker") or "").strip().upper()
        if not ticker:
            continue
        name = item.get("name")
        row = {
            "snapshot_date": snapshot_date,
            "filter": filter_name,
            "ticker": ticker,
            "name": (html.unescape(str(name)).strip() or None) if name is not None else None,
        }
        row.update({field: to_int(item.get(field)) for field in _INT_FIELDS})
        row["fetched_at"] = fetched_iso
        rows.append(row)
    return rows, to_int(payload.get("pages"))


def _collect_filter(
    conn: sqlite3.Connection,
    http: httpx.Client,
    filter_name: str,
    max_pages: int,
    snapshot_date: str,
    now: datetime,
    sleep: Callable[[float], None],
    counts: dict,
) -> bool:
    """Page through one filter, storing each page as it arrives. False if any request or parse failed."""
    seen: set[str] = set()
    page, last_page, fetched = 1, max_pages, 0
    while page <= last_page:
        if counts["requests"]:
            sleep(REQUEST_PAUSE_S)
        counts["requests"] += 1
        try:
            resp = http.get(
                API_URL.format(filter=filter_name, page=page),
                headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
                follow_redirects=True,
            )
            resp.raise_for_status()
            rows, pages = parse_page(resp.json(), filter_name, snapshot_date, now)
        except (httpx.HTTPError, ValueError) as exc:
            log.warning("ApeWisdom %s page %d failed: %s", filter_name, page, exc)
            return False
        fetched += 1
        if pages is not None:
            last_page = min(max_pages, pages)
        fresh = []
        for row in rows:
            # Rankings can shift between page requests; keep a ticker's first (higher) placement.
            if row["ticker"] not in seen:
                seen.add(row["ticker"])
                fresh.append(row)
        with conn:
            counts["rows"] += db.upsert_reddit_ticker_daily(conn, fresh)
        if not rows:
            break
        page += 1
    log.info("ApeWisdom %s: %d tickers from %d page(s)", filter_name, len(seen), fetched)
    return True


def collect_apewisdom(
    conn: sqlite3.Connection,
    watchlist: Watchlist,
    run_id: int,
    now: datetime,
    client: httpx.Client | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> dict:
    """Snapshot each configured ApeWisdom filter for today's New York date. Never raises on network trouble."""
    cfg = watchlist.apewisdom
    counts = {"status": "ok", "requests": 0, "rows": 0, "filters_failed": 0}
    snapshot_date = snapshot_date_for(now)
    http = client or httpx.Client(timeout=REQUEST_TIMEOUT_S)
    try:
        for filter_name in cfg.filters:
            if not _collect_filter(conn, http, filter_name, cfg.pages, snapshot_date, now, sleep, counts):
                counts["filters_failed"] += 1
    finally:
        if client is None:
            http.close()
    failed = counts["filters_failed"]
    if failed:
        counts["status"] = "error" if failed == len(cfg.filters) and counts["rows"] == 0 else "partial"
    log.info("apewisdom (run %d, snapshot %s): %s", run_id, snapshot_date, counts)
    return counts
