from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Sequence
from datetime import datetime
from pathlib import Path

from .models import Mention, Post
from .timeutil import to_iso

# Bump when an existing table's shape changes. New tables only need CREATE TABLE IF NOT EXISTS.
SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL);

CREATE TABLE IF NOT EXISTS accounts (
    platform TEXT NOT NULL,
    handle TEXT NOT NULL,
    category TEXT NOT NULL,
    active INTEGER NOT NULL DEFAULT 1,
    platform_user_id TEXT,
    resolve_error TEXT,
    resolved_at TEXT,
    last_fetch_at TEXT,
    PRIMARY KEY (platform, handle)
);

CREATE TABLE IF NOT EXISTS posts (
    platform TEXT NOT NULL,
    native_id TEXT NOT NULL,
    author TEXT NOT NULL,
    author_id TEXT,
    created_at_utc TEXT NOT NULL,
    text TEXT NOT NULL,
    url TEXT NOT NULL,
    source TEXT,
    feed_rank INTEGER,
    metrics_json TEXT,
    cashtag_hints TEXT,
    stance TEXT,
    stance_conf REAL,
    stance_model TEXT,
    collected_at TEXT NOT NULL,
    PRIMARY KEY (platform, native_id)
);
CREATE INDEX IF NOT EXISTS posts_created ON posts (created_at_utc);

CREATE TABLE IF NOT EXISTS mentions (
    platform TEXT NOT NULL,
    native_id TEXT NOT NULL,
    ticker TEXT NOT NULL,
    match_type TEXT NOT NULL,
    matched_text TEXT NOT NULL,
    PRIMARY KEY (platform, native_id, ticker, match_type)
);
CREATE INDEX IF NOT EXISTS mentions_ticker ON mentions (ticker);

CREATE TABLE IF NOT EXISTS unknown_cashtags (
    platform TEXT NOT NULL,
    native_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    seen_at TEXT NOT NULL,
    PRIMARY KEY (platform, native_id, symbol)
);

CREATE TABLE IF NOT EXISTS bars_1m (
    symbol TEXT NOT NULL,
    ts INTEGER NOT NULL,          -- bar START, unix epoch seconds (UTC)
    open REAL, high REAL, low REAL, close REAL, volume REAL,
    PRIMARY KEY (symbol, ts)
) WITHOUT ROWID;

-- One row per (symbol, session) whose extended-hours session had fully ended when fetched.
CREATE TABLE IF NOT EXISTS bars_1m_sessions (
    symbol TEXT NOT NULL,
    session_date TEXT NOT NULL,   -- YYYY-MM-DD (exchange date)
    n_bars INTEGER NOT NULL,
    fetched_at TEXT NOT NULL,
    PRIMARY KEY (symbol, session_date)
);

CREATE TABLE IF NOT EXISTS reddit_ticker_daily (
    snapshot_date TEXT NOT NULL,  -- YYYY-MM-DD, New York date of the fetch
    filter TEXT NOT NULL,
    ticker TEXT NOT NULL,
    name TEXT,
    rank INTEGER,
    mentions INTEGER,
    upvotes INTEGER,
    rank_24h_ago INTEGER,
    mentions_24h_ago INTEGER,
    fetched_at TEXT NOT NULL,
    PRIMARY KEY (snapshot_date, filter, ticker)
);

CREATE TABLE IF NOT EXISTS x_usage (
    id INTEGER PRIMARY KEY,
    run_id INTEGER,
    ts TEXT NOT NULL,
    handle TEXT,                  -- account the reads are attributed to; NULL for mixed/lookup
    kind TEXT NOT NULL,           -- 'post_read' | 'user_read'
    units INTEGER NOT NULL,
    est_cost_usd REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS x_usage_ts ON x_usage (ts);

CREATE TABLE IF NOT EXISTS watermarks (
    source TEXT NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (source, key)
);

CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY,
    stage TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    status TEXT,                  -- 'ok' | 'partial' | 'error' | 'skipped'
    counts_json TEXT,
    error TEXT
);
CREATE INDEX IF NOT EXISTS runs_stage ON runs (stage, started_at);
"""


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA synchronous=NORMAL")
    init_schema(conn)
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    with conn:
        conn.executescript(_SCHEMA)
        row = conn.execute("SELECT version FROM schema_version").fetchone()
        if row is None:
            conn.execute("INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,))
        elif row["version"] > SCHEMA_VERSION:
            raise RuntimeError(
                f"database schema v{row['version']} is newer than this code (v{SCHEMA_VERSION}); update the code"
            )
        elif row["version"] < SCHEMA_VERSION:
            raise RuntimeError(
                f"database schema v{row['version']} is older than this code (v{SCHEMA_VERSION}); "
                "a migration is needed"
            )


# ---------------------------------------------------------------- accounts


def upsert_account(conn: sqlite3.Connection, platform: str, handle: str, category: str, active: bool) -> None:
    conn.execute(
        """INSERT INTO accounts (platform, handle, category, active) VALUES (?, ?, ?, ?)
           ON CONFLICT(platform, handle) DO UPDATE SET category = excluded.category, active = excluded.active""",
        (platform, handle, category, int(active)),
    )


def deactivate_missing_accounts(conn: sqlite3.Connection, platform: str, handles: Iterable[str]) -> None:
    keep = {h.lower() for h in handles}
    for row in conn.execute("SELECT handle FROM accounts WHERE platform = ?", (platform,)).fetchall():
        if row["handle"].lower() not in keep:
            conn.execute(
                "UPDATE accounts SET active = 0 WHERE platform = ? AND handle = ?", (platform, row["handle"])
            )


def set_account_resolution(
    conn: sqlite3.Connection,
    platform: str,
    handle: str,
    user_id: str | None,
    error: str | None,
    when: datetime,
) -> None:
    conn.execute(
        """UPDATE accounts SET platform_user_id = ?, resolve_error = ?, resolved_at = ?
           WHERE platform = ? AND handle = ?""",
        (user_id, error, to_iso(when), platform, handle),
    )


def touch_account_fetch(conn: sqlite3.Connection, platform: str, handle: str, when: datetime) -> None:
    conn.execute(
        "UPDATE accounts SET last_fetch_at = ? WHERE platform = ? AND handle = ?",
        (to_iso(when), platform, handle),
    )


def get_accounts(conn: sqlite3.Connection, platform: str, active_only: bool = True) -> list[sqlite3.Row]:
    sql = "SELECT * FROM accounts WHERE platform = ?"
    if active_only:
        sql += " AND active = 1"
    return conn.execute(sql + " ORDER BY handle", (platform,)).fetchall()


# ---------------------------------------------------------------- posts & mentions


def upsert_post(conn: sqlite3.Connection, post: Post, collected_at: datetime) -> bool:
    """Insert or refresh a post. Returns True if it was new. Clears stance if the text changed."""
    exists = conn.execute(
        "SELECT 1 FROM posts WHERE platform = ? AND native_id = ?", (post.platform, post.native_id)
    ).fetchone()
    conn.execute(
        """INSERT INTO posts (platform, native_id, author, author_id, created_at_utc, text, url, source,
                              feed_rank, metrics_json, cashtag_hints, collected_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(platform, native_id) DO UPDATE SET
             stance = CASE WHEN posts.text = excluded.text THEN posts.stance ELSE NULL END,
             stance_conf = CASE WHEN posts.text = excluded.text THEN posts.stance_conf ELSE NULL END,
             stance_model = CASE WHEN posts.text = excluded.text THEN posts.stance_model ELSE NULL END,
             text = excluded.text,
             url = excluded.url,
             author_id = COALESCE(excluded.author_id, posts.author_id),
             metrics_json = COALESCE(excluded.metrics_json, posts.metrics_json),
             cashtag_hints = COALESCE(excluded.cashtag_hints, posts.cashtag_hints),
             feed_rank = CASE
               WHEN posts.feed_rank IS NULL THEN excluded.feed_rank
               WHEN excluded.feed_rank IS NULL THEN posts.feed_rank
               ELSE MIN(posts.feed_rank, excluded.feed_rank) END""",
        (
            post.platform,
            post.native_id,
            post.author,
            post.author_id,
            to_iso(post.created_at_utc),
            post.text,
            post.url,
            post.source,
            post.feed_rank,
            json.dumps(post.metrics, ensure_ascii=False) if post.metrics is not None else None,
            ",".join(post.cashtag_hints) if post.cashtag_hints else None,
            to_iso(collected_at),
        ),
    )
    return exists is None


def replace_mentions(
    conn: sqlite3.Connection,
    platform: str,
    native_id: str,
    mentions: Sequence[Mention],
    unknown_cashtags: Sequence[str],
    when: datetime,
) -> None:
    conn.execute("DELETE FROM mentions WHERE platform = ? AND native_id = ?", (platform, native_id))
    conn.executemany(
        "INSERT OR IGNORE INTO mentions (platform, native_id, ticker, match_type, matched_text) VALUES (?,?,?,?,?)",
        [(platform, native_id, m.ticker, m.match_type, m.matched_text) for m in mentions],
    )
    conn.execute("DELETE FROM unknown_cashtags WHERE platform = ? AND native_id = ?", (platform, native_id))
    conn.executemany(
        "INSERT OR IGNORE INTO unknown_cashtags (platform, native_id, symbol, seen_at) VALUES (?,?,?,?)",
        [(platform, native_id, s, to_iso(when)) for s in unknown_cashtags],
    )


# ---------------------------------------------------------------- watermarks


def get_watermark(conn: sqlite3.Connection, source: str, key: str) -> str | None:
    row = conn.execute("SELECT value FROM watermarks WHERE source = ? AND key = ?", (source, key)).fetchone()
    return row["value"] if row else None


def set_watermark(conn: sqlite3.Connection, source: str, key: str, value: str, when: datetime) -> None:
    conn.execute(
        """INSERT INTO watermarks (source, key, value, updated_at) VALUES (?, ?, ?, ?)
           ON CONFLICT(source, key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at""",
        (source, key, value, to_iso(when)),
    )


# ---------------------------------------------------------------- runs


def start_run(conn: sqlite3.Connection, stage: str, when: datetime) -> int:
    with conn:
        cur = conn.execute("INSERT INTO runs (stage, started_at) VALUES (?, ?)", (stage, to_iso(when)))
    return int(cur.lastrowid)


def finish_run(
    conn: sqlite3.Connection,
    run_id: int,
    status: str,
    counts: dict | None,
    error: str | None,
    when: datetime,
) -> None:
    with conn:
        conn.execute(
            "UPDATE runs SET finished_at = ?, status = ?, counts_json = ?, error = ? WHERE id = ?",
            (to_iso(when), status, json.dumps(counts or {}, ensure_ascii=False), error, run_id),
        )


# ---------------------------------------------------------------- X usage


def record_x_usage(
    conn: sqlite3.Connection,
    run_id: int | None,
    when: datetime,
    handle: str | None,
    kind: str,
    units: int,
    est_cost_usd: float,
) -> None:
    conn.execute(
        "INSERT INTO x_usage (run_id, ts, handle, kind, units, est_cost_usd) VALUES (?, ?, ?, ?, ?, ?)",
        (run_id, to_iso(when), handle, kind, units, est_cost_usd),
    )


def x_spend_since(conn: sqlite3.Connection, since: datetime) -> float:
    row = conn.execute(
        "SELECT COALESCE(SUM(est_cost_usd), 0) AS s FROM x_usage WHERE ts >= ?", (to_iso(since),)
    ).fetchone()
    return float(row["s"])


# ---------------------------------------------------------------- 1-minute bars


def insert_bars_1m(
    conn: sqlite3.Connection,
    symbol: str,
    rows: Iterable[tuple[int, float, float, float, float, float]],
) -> int:
    """rows: (bar_start_epoch_s, open, high, low, close, volume). Replaces duplicates."""
    rows = list(rows)
    conn.executemany(
        "INSERT OR REPLACE INTO bars_1m (symbol, ts, open, high, low, close, volume) VALUES (?,?,?,?,?,?,?)",
        [(symbol, *r) for r in rows],
    )
    return len(rows)


def mark_session(conn: sqlite3.Connection, symbol: str, session_date: str, n_bars: int, when: datetime) -> None:
    conn.execute(
        """INSERT INTO bars_1m_sessions (symbol, session_date, n_bars, fetched_at) VALUES (?, ?, ?, ?)
           ON CONFLICT(symbol, session_date) DO UPDATE SET
             n_bars = excluded.n_bars, fetched_at = excluded.fetched_at""",
        (symbol, session_date, n_bars, to_iso(when)),
    )


def sessions_with_bars(conn: sqlite3.Connection, symbol: str) -> set[str]:
    rows = conn.execute(
        "SELECT session_date FROM bars_1m_sessions WHERE symbol = ? AND n_bars > 0", (symbol,)
    ).fetchall()
    return {r["session_date"] for r in rows}


# ---------------------------------------------------------------- Reddit attention (ApeWisdom)


def upsert_reddit_ticker_daily(conn: sqlite3.Connection, rows: Iterable[dict]) -> int:
    rows = list(rows)
    conn.executemany(
        """INSERT INTO reddit_ticker_daily (snapshot_date, filter, ticker, name, rank, mentions, upvotes,
                                            rank_24h_ago, mentions_24h_ago, fetched_at)
           VALUES (:snapshot_date, :filter, :ticker, :name, :rank, :mentions, :upvotes,
                   :rank_24h_ago, :mentions_24h_ago, :fetched_at)
           ON CONFLICT(snapshot_date, filter, ticker) DO UPDATE SET
             name = excluded.name, rank = excluded.rank, mentions = excluded.mentions,
             upvotes = excluded.upvotes, rank_24h_ago = excluded.rank_24h_ago,
             mentions_24h_ago = excluded.mentions_24h_ago, fetched_at = excluded.fetched_at""",
        rows,
    )
    return len(rows)
