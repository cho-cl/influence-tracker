from __future__ import annotations

import json
import logging
import sqlite3
from datetime import UTC, date, datetime, timedelta

import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.text import Text

from . import db, market
from .config import Watchlist
from .logsetup import make_console
from .timeutil import NY, from_iso, to_iso

log = logging.getLogger(__name__)

PLATFORMS = ("x", "truthsocial", "reddit")
KNOWN_STAGES = (
    "collect:truthsocial",
    "collect:reddit",
    "collect:apewisdom",
    "collect:x",
    "snapshot",
    "classify",
    "enrich",
)
STANCES = ("bullish", "bearish", "neutral")
# Events complete once daily bars reach d0+5; still pending two evening runs after that means something is stuck.
PENDING_STALE_SESSIONS = 7
TS_RATE_LIMIT_STAGES = ("collect:truthsocial", "backfill:truthsocial")
TOP_N = 15
_STATUS_STYLE = {"ok": "green", "partial": "yellow", "error": "bold red", "skipped": "dim", "running?": "yellow"}
_NOT_COUNTS = ("status", "error", "errors")


def format_counts(counts: dict | None, max_items: int | None = None) -> str:
    """Compact 'key=value' rendering of a collector's counts dict (status/error live in their own columns)."""
    items: list[str] = []
    for key, value in (counts or {}).items():
        if key in _NOT_COUNTS:
            continue
        if isinstance(value, dict):
            items.extend(f"{key}.{k}={_scalar(v)}" for k, v in value.items())
        else:
            items.append(f"{key}={_scalar(value)}")
    if max_items is not None and len(items) > max_items:
        items = [*items[:max_items], f"+{len(items) - max_items} more"]
    return " ".join(items)


def _scalar(value: object) -> str:
    if isinstance(value, list | tuple):
        return f"[{len(value)}]"
    if isinstance(value, dict):
        return f"{{{len(value)}}}"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def show_status(conn: sqlite3.Connection, watchlist: Watchlist, now: datetime, console: Console | None = None) -> None:
    console = console or make_console()
    console.rule(f"influence status  {_local(now)}")
    _runs(console, conn, now)
    _posts(console, conn, now)
    _stances(console, conn, watchlist)
    _events(console, conn, now)
    _alerts(console, conn, now)
    _top_tickers(console, conn)
    _accounts(console, conn, watchlist, now)
    _x_spend(console, conn, watchlist, now)
    _truthsocial_rate_limits(console, conn, now)
    _bars(console, conn, watchlist)
    _daily_bars(console, conn, now)
    _earnings(console, conn, now)
    _unknown_cashtags(console, conn, watchlist)


# ---------------------------------------------------------------- formatting helpers


def _heading(console: Console, title: str) -> None:
    console.print()
    console.print(Text(title, style="bold cyan"))


def _table(*columns: str) -> Table:
    table = Table(show_edge=False, pad_edge=False, header_style="bold")
    for column in columns:
        table.add_column(column, overflow="fold")
    return table


def _row(table: Table, *cells: str | Text) -> None:
    # Plain str cells would be parsed as rich markup: "[/x]" in an error message crashes the render.
    table.add_row(*(Text(c) if isinstance(c, str) else c for c in cells))


def _say(console: Console, text: str, style: str = "") -> None:
    console.print(Text(text, style=style))


def _local(ts: datetime) -> str:
    return ts.astimezone(NY).strftime("%Y-%m-%d %H:%M ET")


def _ago(then: datetime, now: datetime) -> str:
    seconds = (now - then).total_seconds()
    if seconds < 90:
        return "just now"
    if seconds < 90 * 60:
        return f"{round(seconds / 60)}m ago"
    if seconds < 36 * 3600:
        return f"{round(seconds / 3600)}h ago"
    return f"{round(seconds / 86400)}d ago"


def _when(iso: str | None, now: datetime) -> str:
    if not iso:
        return "never"
    ts = from_iso(iso)
    return f"{_local(ts)} ({_ago(ts, now)})"


def _took(start: datetime, end: datetime) -> str:
    seconds = (end - start).total_seconds()
    if seconds < 120:
        return f"{seconds:.0f}s"
    if seconds < 7200:
        return f"{seconds / 60:.0f}m"
    return f"{seconds / 3600:.1f}h"


def _clip(text: str | None, limit: int) -> str:
    if not text:
        return ""
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _plural(n: int, noun: str) -> str:
    return f"{n} {noun}{'' if n == 1 else 's'}"


def _money(value: object) -> str:
    return f"${float(value):,.2f}" if isinstance(value, int | float) else "?"


def _utc_date_text(value: object) -> str:
    # X billing cycles start at 00:00 UTC; in New York time that is still the previous day.
    if isinstance(value, datetime):
        return value.astimezone(UTC).date().isoformat() if value.tzinfo else value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return str(value) if value is not None else "?"


# ---------------------------------------------------------------- sections


def _runs(console: Console, conn: sqlite3.Connection, now: datetime) -> None:
    _heading(console, "Last run per stage")
    rows = conn.execute(
        """SELECT * FROM runs r
           WHERE r.id = (SELECT r2.id FROM runs r2 WHERE r2.stage = r.stage
                         ORDER BY r2.started_at DESC, r2.id DESC LIMIT 1)"""
    ).fetchall()
    latest = {r["stage"]: r for r in rows}
    table = _table("stage", "started", "status", "took", "counts", "error")
    for stage in [*KNOWN_STAGES, *sorted(s for s in latest if s not in KNOWN_STAGES)]:
        r = latest.get(stage)
        if r is None:
            _row(table, stage, "never", "", "", "", "")
            continue
        started = from_iso(r["started_at"])
        # NULL status: the stage is still running, or its process died before finishing.
        state = r["status"] or "running?"
        took = _took(started, from_iso(r["finished_at"])) if r["finished_at"] else ""
        counts = json.loads(r["counts_json"]) if r["counts_json"] else {}
        _row(
            table,
            stage,
            _when(r["started_at"], now),
            Text(state, style=_STATUS_STYLE.get(state, "")),
            took,
            format_counts(counts, max_items=6),
            _clip(r["error"], 80),
        )
    console.print(table)


def _posts(console: Console, conn: sqlite3.Connection, now: datetime) -> None:
    _heading(console, "Posts")
    rows = conn.execute(
        """SELECT p.platform, COUNT(*) AS posts,
                  SUM(EXISTS (SELECT 1 FROM mentions m
                              WHERE m.platform = p.platform AND m.native_id = p.native_id)) AS tagged,
                  MAX(p.created_at_utc) AS newest, MAX(p.collected_at) AS last_stored
           FROM posts p GROUP BY p.platform"""
    ).fetchall()
    by_platform = {r["platform"]: r for r in rows}
    table = _table("platform", "posts", "with >=1 mention", "newest post", "last stored")
    total = tagged = 0
    for platform in [*PLATFORMS, *sorted(p for p in by_platform if p not in PLATFORMS)]:
        r = by_platform.get(platform)
        n, t = (r["posts"], r["tagged"]) if r else (0, 0)
        total += n
        tagged += t
        _row(
            table,
            platform,
            str(n),
            str(t),
            _when(r["newest"] if r else None, now),
            _when(r["last_stored"] if r else None, now),
        )
    console.print(table)
    _say(console, f"posts with >=1 mention: {tagged} of {total}")

    ape = conn.execute(
        "SELECT COUNT(*) AS n, COUNT(DISTINCT snapshot_date) AS days, MAX(snapshot_date) AS latest "
        "FROM reddit_ticker_daily"
    ).fetchone()
    if ape["n"]:
        _say(console, f"ApeWisdom: {ape['n']} ticker rows over {ape['days']} days, latest snapshot {ape['latest']}")
    else:
        _say(console, "ApeWisdom: no snapshots yet")


def _stances(console: Console, conn: sqlite3.Connection, watchlist: Watchlist) -> None:
    # classify labels exactly these posts: the ones that mention a non-benchmark ticker.
    symbols = [t.symbol for t in watchlist.event_tickers]
    rows = (
        conn.execute(
            f"""SELECT p.stance, p.stance_model, COUNT(*) AS n FROM posts p
                WHERE EXISTS (SELECT 1 FROM mentions m WHERE m.platform = p.platform AND m.native_id = p.native_id
                              AND m.ticker IN ({",".join("?" * len(symbols))}))
                GROUP BY p.stance, p.stance_model""",
            symbols,
        ).fetchall()
        if symbols
        else []
    )
    total = sum(r["n"] for r in rows)
    if not total:
        _heading(console, "Stance: no posts mention a watchlist stock yet")
        return
    by_stance: dict[str, int] = {}
    other_model = 0
    for r in rows:
        key = r["stance"] or "unlabelled"
        by_stance[key] = by_stance.get(key, 0) + r["n"]
        if r["stance"] and r["stance_model"] != watchlist.sentiment.model_id:
            other_model += r["n"]
    known = (*STANCES, "unlabelled")
    parts = [f"{by_stance.get(s, 0)} {s}" for s in known]
    parts += [f"{n} {s}" for s, n in sorted(by_stance.items()) if s not in known]
    _heading(console, f"Stance of the {_plural(total, 'post')} that mention a watchlist stock: {', '.join(parts)}")
    if other_model:
        _say(
            console,
            f"{other_model} labelled by a model other than {watchlist.sentiment.model_id}; the next classify run "
            "relabels them",
            "yellow",
        )


def _sessions_old(d0: date, now: datetime) -> int:
    """Sessions after d0 whose regular close is at or before now."""
    later = market.sessions_in_range(d0 + timedelta(days=1), now.astimezone(NY).date())
    if later and market.xnys().session_close(pd.Timestamp(later[-1])) > now:
        later.pop()
    return len(later)


def _events(console: Console, conn: sqlite3.Connection, now: datetime) -> None:
    r = conn.execute(
        """SELECT COUNT(*) AS total, COALESCE(SUM(status = 'pending'), 0) AS pending,
                  COALESCE(SUM(status = 'complete'), 0) AS complete,
                  COALESCE(SUM(intraday_state = 'unavailable'), 0) AS unavailable,
                  MIN(CASE WHEN status = 'pending' THEN d0 END) AS oldest_pending
           FROM events"""
    ).fetchone()
    if not r["total"]:
        _heading(console, "Events: none yet ('influence enrich' builds them from tagged posts)")
        return
    _heading(
        console,
        f"Events: {r['total']} ({r['complete']} complete, {r['pending']} pending), "
        f"intraday data unavailable for {r['unavailable']}",
    )
    if r["oldest_pending"]:
        d0 = date.fromisoformat(r["oldest_pending"])
        age = _sessions_old(d0, now)
        line = f"oldest pending event: d0 {d0} ({d0:%a}), {_plural(age, 'session')} old"
        if age >= PENDING_STALE_SESSIONS:
            _say(console, f"{line}; events complete once daily bars reach d0+5, so check the enrich runs", "yellow")
        else:
            _say(console, line)
    _say(console, "'influence events' lists them for checking against price charts.", "dim")


def _alerts(console: Console, conn: sqlite3.Connection, now: datetime) -> None:
    from .alerts.timing import is_active
    from .alerts.watch import HEARTBEAT_FRESH

    _heading(console, "Alerts")
    beat = db.get_watermark(conn, "watch", "heartbeat")
    if beat is None:
        _say(console, "watch has never run (start it with `influence watch` or register the watch task)", "dim")
    else:
        stale = now - from_iso(beat) >= HEARTBEAT_FRESH
        note = " STALE — the watch is not running" if stale and is_active(now) else ""
        _say(console, f"watch last check-in: {_when(beat, now)}{note}", "bold red" if note else "")
    since = to_iso(now - timedelta(days=7))
    table = _table("kind", "sent", "failed", "skipped", "pending")
    rows = conn.execute(
        """SELECT kind, status, COUNT(*) AS n FROM alerts WHERE created_at >= ? OR due_at >= ?
           GROUP BY kind, status""",
        (since, since),
    ).fetchall()
    by_kind: dict[str, dict[str, int]] = {}
    for r in rows:
        by_kind.setdefault(r["kind"], {})[r["status"]] = r["n"]
    for kind in ("heads_up", "follow_60m", "follow_d1", "digest"):
        c = by_kind.get(kind, {})
        _row(table, kind, *(str(c.get(s, 0)) for s in ("sent", "failed", "skipped", "pending")))
    console.print(table)
    pending = conn.execute(
        "SELECT COUNT(*) FROM alerts WHERE status = 'pending' AND kind IN ('follow_60m', 'follow_d1')"
    ).fetchone()[0]
    failed = conn.execute("SELECT COUNT(*) FROM alerts WHERE status = 'failed'").fetchone()[0]
    _say(console, f"pending follow-ups: {pending} · failed sends: {failed} (last 7 days above)")


def _top_tickers(console: Console, conn: sqlite3.Connection) -> None:
    rows = conn.execute(
        """SELECT ticker, COUNT(*) AS posts,
                  SUM(platform = 'x') AS x, SUM(platform = 'truthsocial') AS truthsocial,
                  SUM(platform = 'reddit') AS reddit
           FROM (SELECT DISTINCT platform, native_id, ticker FROM mentions)
           GROUP BY ticker ORDER BY posts DESC, ticker LIMIT ?""",
        (TOP_N,),
    ).fetchall()
    if not rows:
        _heading(console, "Top mentioned tickers: no mentions yet")
        return
    _heading(console, f"Top mentioned tickers (by posts, top {TOP_N})")
    table = _table("ticker", "posts", "X", "Truth Social", "Reddit")
    for r in rows:
        _row(table, r["ticker"], str(r["posts"]), str(r["x"]), str(r["truthsocial"]), str(r["reddit"]))
    console.print(table)


def _accounts(console: Console, conn: sqlite3.Connection, watchlist: Watchlist, now: datetime) -> None:
    _heading(console, "Accounts")
    table = _table("platform", "handle", "category", "resolved / posts", "last fetch")
    failed: list[str] = []
    for platform, accounts in (("x", watchlist.x.accounts), ("truthsocial", watchlist.truthsocial.accounts)):
        rows = db.get_accounts(conn, platform, active_only=False)
        exact = {r["handle"]: r for r in rows}
        # Fallback for a handle whose capitalisation changed in the config since the last collect.
        folded = {r["handle"].lower(): r for r in rows}
        for account in accounts:
            r = exact.get(account.handle) or folded.get(account.handle.lower())
            if not account.active:
                resolved = Text("inactive", style="dim")
            elif r is not None and r["resolve_error"]:
                resolved = Text(f"FAILED: {r['resolve_error']}", style="bold red")
                failed.append(f"{platform}/{account.handle}")
            elif r is not None and r["platform_user_id"]:
                resolved = Text("yes", style="green")
            else:
                resolved = Text("not yet", style="yellow")
            _row(
                table,
                platform,
                account.handle,
                account.category,
                resolved,
                _when(r["last_fetch_at"] if r else None, now),
            )
    subs = {
        r["sub"]: r
        for r in conn.execute(
            "SELECT LOWER(source) AS sub, COUNT(*) AS n, MAX(collected_at) AS last FROM posts "
            "WHERE platform = 'reddit' GROUP BY LOWER(source)"
        )
    }
    for sub in watchlist.reddit.subreddits:
        r = subs.get(sub.lower())
        n = r["n"] if r else 0
        _row(
            table,
            "reddit",
            f"r/{sub}",
            "subreddit",
            f"{n} post{'' if n == 1 else 's'}",
            _when(r["last"] if r else None, now),
        )
    console.print(table)
    if failed:
        noun = "account" if len(failed) == 1 else "accounts"
        _say(console, f"{len(failed)} {noun} failed to resolve: {', '.join(failed)}", "bold red")


def _x_spend(console: Console, conn: sqlite3.Connection, watchlist: Watchlist, now: datetime) -> None:
    _heading(console, "X spend")
    if conn.execute("SELECT COUNT(*) FROM x_usage").fetchone()[0] == 0:
        _say(console, "no X usage yet")
        return
    try:
        from .collectors.x import x_budget_summary
    except ImportError:
        _say(console, "no X usage yet")
        return
    try:
        summary = x_budget_summary(conn, watchlist, now)
    except Exception as e:
        log.exception("x_budget_summary failed")
        _say(console, f"X budget summary failed: {type(e).__name__}: {e}", "red")
        return
    _say(
        console,
        f"billing cycle from {_utc_date_text(summary.get('cycle_start'))}: "
        f"{_money(summary.get('spent_usd'))} of {_money(summary.get('budget_usd'))} spent, "
        f"{_money(summary.get('remaining_usd'))} left; "
        f"daily allowance {summary.get('daily_allowance_posts', '?')} posts/day",
    )
    per_handle = sorted(summary.get("per_handle") or [], key=lambda h: -float(h.get("cost_usd") or 0))
    if per_handle:
        table = _table("account", "post reads", "user reads", "cost")
        for h in per_handle:
            _row(
                table,
                h.get("handle") or "(lookups)",
                str(h.get("post_reads", 0)),
                str(h.get("user_reads", 0)),
                _money(h.get("cost_usd")),
            )
        console.print(table)


def _truthsocial_rate_limits(console: Console, conn: sqlite3.Connection, now: datetime) -> None:
    rows = conn.execute(
        f"SELECT counts_json FROM runs WHERE stage IN ({','.join('?' * len(TS_RATE_LIMIT_STAGES))}) "
        "AND started_at >= ?",
        (*TS_RATE_LIMIT_STAGES, to_iso(now - timedelta(days=7))),
    ).fetchall()
    hits = runs_hit = 0
    for r in rows:
        value = (json.loads(r["counts_json"]) if r["counts_json"] else {}).get("rate_limited", 0)
        n = int(value) if isinstance(value, int | float) else 0
        hits += n
        runs_hit += n > 0
    _heading(console, f"Truth Social rate-limit hits (last 7 days): {hits} in {runs_hit} of {len(rows)} runs")


def _bars(console: Console, conn: sqlite3.Connection, watchlist: Watchlist) -> None:
    rows = conn.execute(
        """SELECT symbol, MIN(session_date) AS first, MAX(session_date) AS last, SUM(n_bars) AS bars
           FROM bars_1m_sessions WHERE n_bars > 0 GROUP BY symbol"""
    ).fetchall()
    stored = {r["symbol"]: r for r in rows}
    have: list[tuple[str, sqlite3.Row]] = []
    missing: list[str] = []
    for ticker in watchlist.tickers:
        # prices.snapshot_1m keys sessions by the watchlist symbol (BRK.B), not the Yahoo one (BRK-B).
        r = stored.get(ticker.symbol)
        if r is None:
            missing.append(ticker.symbol)
        else:
            have.append((ticker.symbol, r))
    headline = f"1-minute bars: {len(have)} of {len(watchlist.tickers)} symbols have bars"
    if have:
        first = min(r["first"] for _, r in have)
        latest = max(r["last"] for _, r in have)
        total = sum(r["bars"] for _, r in have)
        headline += f", sessions {first} to {latest}, {total:,} bars"
    _heading(console, headline)
    if have:
        behind = [symbol for symbol, r in have if r["last"] < latest]
        if behind:
            _say(console, f"behind the latest session: {', '.join(behind)}", "yellow")
    if missing:
        _say(console, f"no bars: {', '.join(missing)}", "yellow")


def _daily_bars(console: Console, conn: sqlite3.Connection, now: datetime) -> None:
    r = conn.execute("SELECT COUNT(DISTINCT symbol) AS symbols, MAX(session_date) AS latest FROM bars_1d").fetchone()
    if r["symbols"]:
        _heading(console, f"Daily bars: {_plural(r['symbols'], 'symbol')}, latest session {r['latest']}")
    else:
        _heading(console, "Daily bars: none yet")
    run = conn.execute("SELECT * FROM runs WHERE stage = 'enrich' ORDER BY started_at DESC, id DESC LIMIT 1").fetchone()
    if run is None:
        return
    when = _when(run["started_at"], now)
    daily = (json.loads(run["counts_json"]) if run["counts_json"] else {}).get("daily")
    failed = daily.get("symbols_failed") if isinstance(daily, dict) else None
    if not isinstance(failed, list):
        state = run["status"] or "still running?"
        _say(console, f"the last enrich run, {when}, has no daily-bar result (status: {state})", "yellow")
    elif failed:
        _say(console, f"failed in the last enrich run, {when}: {', '.join(str(s) for s in failed)}", "yellow")
    else:
        _say(console, f"no symbol failed in the last enrich run, {when}")


def _earnings(console: Console, conn: sqlite3.Connection, now: datetime) -> None:
    rows = conn.execute("SELECT symbol, fetched_at, ok, error FROM earnings_fetch ORDER BY symbol").fetchall()
    if not rows:
        _heading(console, "Earnings dates: no lookups yet")
        return
    failed = [r for r in rows if not r["ok"]]
    dates = conn.execute("SELECT COUNT(*) FROM earnings").fetchone()[0]
    _heading(
        console,
        f"Earnings dates: {_plural(len(rows) - len(failed), 'symbol')} fetched ok, {len(failed)} failed "
        f"({_plural(dates, 'date')} stored)",
    )
    if failed:
        table = _table("symbol", "last try", "error")
        for r in failed:
            _row(table, r["symbol"], _when(r["fetched_at"], now), _clip(r["error"], 100))
        console.print(table)
        _say(console, "Until a lookup works, events on these tickers can complete with the earnings flag unknown (?).")


def _unknown_cashtags(console: Console, conn: sqlite3.Connection, watchlist: Watchlist) -> None:
    rows = conn.execute(
        """SELECT symbol, COUNT(*) AS posts, GROUP_CONCAT(DISTINCT platform) AS platforms, MAX(seen_at) AS last_seen
           FROM unknown_cashtags GROUP BY symbol ORDER BY posts DESC, symbol"""
    ).fetchall()
    # Symbols added to the watchlist since are re-matched on the next collect; don't suggest them again.
    rows = [r for r in rows if watchlist.ticker(r["symbol"]) is None][:TOP_N]
    if not rows:
        _heading(console, "Unknown cashtags: no unknown cashtags yet")
        return
    _heading(console, f"Unknown cashtags (not in the watchlist; top {TOP_N} by posts)")
    table = _table("cashtag", "posts", "platforms", "last seen")
    for r in rows:
        _row(table, f"${r['symbol']}", str(r["posts"]), r["platforms"], r["last_seen"][:10])
    console.print(table)
    _say(console, "Add the ones you care about to tickers in config/watchlist.yaml.")
