# Live Alerts (Phase 2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a long-running `influence watch` command on Bryce's PC. It checks Truth Social (and X, once there is a
token) every few minutes and pushes ntfy phone alerts for stock posts: a heads-up when a post lands, a 60-minute
follow-up, a day-after follow-up, and a catch-up digest after downtime.

**Architecture:** A new `influence_tracker.alerts` package, built from small, separately tested units:

- `notify` — the ntfy client;
- `timing` — cadence and follow-up windows;
- `live` — live 1-minute prices;
- `messages` — pure text formatting;
- `followups` — the follow-up numbers;
- `engine` — the ledger and deciding what to send;
- `watch` — the loop, heartbeat and keep-awake.

It reuses the existing collectors, `PostSink`, sentiment classifier, `events.sync_events`/`MinuteBars`, and the M3
market model (through a new public `metrics.event_model`). A new SQLite `alerts` table is the no-duplicates ledger.

**Tech Stack:** Python 3.11 (uv), httpx, yfinance, exchange_calendars, pandas, numpy, SQLite, pytest, ruff; Windows
Task Scheduler (PowerShell); the ntfy.sh public server.

**Spec:** `docs/superpowers/specs/2026-09-28-live-alerts-design.md`. Read it before starting any task.

**Where to work:** a git worktree on branch `alerts`, e.g. `git worktree add -b alerts ../influence-tracker-alerts`,
then `uv sync` there. The nightly scheduled task keeps running the main checkout's `master`. Merge only after Task 11.

## Global Constraints

- Python 3.11 only; run tools with `uv run --no-sync pytest …` and `uv run --no-sync ruff check …` / `ruff format --check …`. No new dependencies.
- Every file starts with `from __future__ import annotations`; type hints; ruff-clean at line length 120; comments only for a non-obvious WHY.
- Datetimes are tz-aware UTC internally, persisted with `timeutil.to_iso`. New York time is used only for session logic and display.
- Tests are offline and deterministic: inject clocks, notifiers, price sources and fetchers. Tests that touch the network are marked `@pytest.mark.live`.
- **ntfy delivery:** POST JSON to the server root (`{server}/`) with `topic`, `title`, `message`, `priority`, `tags`, `click`. The server defaults to `https://ntfy.sh`, and the topic comes from `.env` `NTFY_TOPIC`.
  - This refines the spec's header wording: HTTP header values are ASCII-only in httpx, and titles carry "⭐" and "·". ntfy documents JSON publishing.
  - Message body ≤ 4,000 bytes (under ntfy's 4,096-byte attachment threshold). One retry after 5 s.
- Cadence and thresholds:

  | Setting | Value |
  |---|---|
  | Active window | XNYS session days 04:00–20:00 New York |
  | `poll_active_minutes` | 5 |
  | `poll_idle_minutes` | 30 |
  | `poll_x_minutes` | 15 |
  | `late_after_minutes` | 30 |
  | `followup_minutes` | 60 |
  | Follow-up delay | 3 min |
  | 60-minute follow-up goes stale | 30 min after due |
  | Failed sends retried for | 24 h |
  | Day-after follow-up given up | 7 days after due |
  | Watch heartbeat counts as "fresh" | < 15 min |
  | Price bar older than this is "not yet available" | 15 min |

- Alerts come only from platforms `truthsocial` and `x`, and only for posts mentioning ≥ 1 non-benchmark configured ticker. Reddit is never alerted.
- Wording: a day-after follow-up says "within the normal range" when |z| < 1.96 and "unusually large (|z| = X.XX)" when |z| ≥ 1.96.
- Holdings: marked with "⭐", listed first, priority 4 on the heads-up (3 on follow-ups); otherwise priority 3 (2 on follow-ups). Digest priority 2.
- The ledger `alerts` table has `UNIQUE (kind, platform, native_id)`. A row is `pending` before sending and `sent` only after ntfy returns 2xx.

## Review Focus

- **First run on the real database** (about 15,000 stored posts, 200+ with stock mentions): `watch` must send **zero** notifications for them. Tested in Task 6 (`test_first_run_marks_existing_posts_and_sends_nothing`).
- **PC asleep overnight, 3 posts made at 22:00–23:30, `watch` resumes at 07:00:** exactly one digest, no heads-ups, no 60-minute follow-ups, but day-after follow-ups still queued. Tested in Task 6 (`test_late_posts_go_to_one_digest_with_day_after_followups_only`).
- **ntfy down for a day:** no duplicate sends while failing; retried each cycle for 24 h; then left `failed` and not retried. Tested in Task 6 (`test_failed_send_is_retried_then_abandoned_after_24h`).
- **A post naming 6 tickers with a 3,000-character emoji-laden text:** the title shows 2 tickers plus "+N", the body stays under 4,000 bytes, and holdings go first. Tested in Task 5 (`test_heads_up_folds_many_tickers_and_caps_long_text`).
- **Yahoo has not yet published the bar at the window end** (the newest bar is 20 minutes older than the end): the follow-up must stay pending, never report a stale price as the 60-minute result. Tested in Task 7 (`test_follow_60m_waits_when_end_bar_is_stale`).

---

### Task 1: Alert config, settings and the ledger table

**Files:**
- Modify: `src/influence_tracker/config.py` (add `AlertsConfig`, `Watchlist.alerts`, holdings validation, `Settings.ntfy_topic` / `ntfy_server`)
- Modify: `config/watchlist.yaml` (add the `alerts:` section)
- Modify: `.env.example` (add `NTFY_TOPIC=`)
- Modify: `src/influence_tracker/db.py` (the `alerts` table in `_SCHEMA` + helpers)
- Test: `tests/test_alert_config.py`, `tests/test_alert_ledger.py`

**Interfaces:**
- Produces:
  - `config.AlertsConfig` (fields: `enabled: bool`, `holdings: list[str]`, `poll_active_minutes: int`, `poll_idle_minutes: int`, `poll_x_minutes: int`, `late_after_minutes: int`, `followup_minutes: int`).
  - `Watchlist.alerts: AlertsConfig` (holdings are canonical symbols).
  - `Settings.ntfy_topic: str | None`, `Settings.ntfy_server: str` (default `"https://ntfy.sh"`).
  - `db.add_alert(conn, kind, platform, native_id, due_at: datetime, when: datetime, *, status="pending", title=None, message=None, error=None) -> bool` (True if inserted).
  - `db.alert_status(conn, kind, platform, native_id) -> str | None`.
  - `db.due_alerts(conn, now: datetime, retry_for: timedelta) -> list[sqlite3.Row]`.
  - `db.mark_alert(conn, alert_id: int, status: str, when: datetime, *, title=None, message=None, error=None) -> None`.

- [ ] **Step 1: Write the failing config tests**

```python
# tests/test_alert_config.py
from __future__ import annotations

import pytest
import yaml
from pydantic import ValidationError

from influence_tracker.config import REPO_ROOT, Watchlist, load_settings


def _raw() -> dict:
    with open(REPO_ROOT / "config" / "watchlist.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_alerts_defaults_when_section_missing():
    raw = _raw()
    raw.pop("alerts", None)
    wl = Watchlist.model_validate(raw)
    assert wl.alerts.enabled is True
    assert wl.alerts.holdings == []
    assert (wl.alerts.poll_active_minutes, wl.alerts.poll_idle_minutes, wl.alerts.poll_x_minutes) == (5, 30, 15)
    assert (wl.alerts.late_after_minutes, wl.alerts.followup_minutes) == (30, 60)


def test_holdings_are_normalised_to_canonical_symbols():
    raw = _raw()
    raw["alerts"] = {"holdings": [" nvda ", "GOOG", "brk.a"]}
    wl = Watchlist.model_validate(raw)
    assert wl.alerts.holdings == ["NVDA", "GOOGL", "BRK.B"]


def test_unknown_holding_is_a_config_error():
    raw = _raw()
    raw["alerts"] = {"holdings": ["ZZZZ"]}
    with pytest.raises(ValidationError, match="ZZZZ"):
        Watchlist.model_validate(raw)


def test_settings_read_ntfy_from_env(tmp_path, monkeypatch):
    monkeypatch.delenv("NTFY_TOPIC", raising=False)
    monkeypatch.delenv("NTFY_SERVER", raising=False)
    (tmp_path / ".env").write_text("NTFY_TOPIC=influence-abc123\n", encoding="utf-8")
    s = load_settings(tmp_path)
    assert s.ntfy_topic == "influence-abc123"
    assert s.ntfy_server == "https://ntfy.sh"


def test_settings_without_topic(tmp_path, monkeypatch):
    monkeypatch.delenv("NTFY_TOPIC", raising=False)
    assert load_settings(tmp_path).ntfy_topic is None
```

- [ ] **Step 2: Write the failing ledger tests**

```python
# tests/test_alert_ledger.py
from __future__ import annotations

from datetime import UTC, datetime, timedelta

from influence_tracker import db

T0 = datetime(2026, 9, 29, 14, 0, tzinfo=UTC)


def test_add_alert_is_unique_per_kind_and_post(conn):
    with conn:
        assert db.add_alert(conn, "heads_up", "truthsocial", "1", T0, T0) is True
        assert db.add_alert(conn, "heads_up", "truthsocial", "1", T0, T0) is False
        assert db.add_alert(conn, "follow_60m", "truthsocial", "1", T0, T0) is True
    assert db.alert_status(conn, "heads_up", "truthsocial", "1") == "pending"
    assert db.alert_status(conn, "follow_d1", "truthsocial", "1") is None


def test_due_alerts_selects_pending_and_recent_failures(conn):
    with conn:
        db.add_alert(conn, "heads_up", "truthsocial", "due", T0, T0)
        db.add_alert(conn, "heads_up", "truthsocial", "future", T0 + timedelta(hours=1), T0)
        db.add_alert(conn, "heads_up", "truthsocial", "sent", T0, T0, status="sent")
        db.add_alert(conn, "heads_up", "truthsocial", "failed-recent", T0, T0, status="failed")
        db.add_alert(conn, "heads_up", "truthsocial", "failed-old", T0 - timedelta(hours=30), T0, status="failed")
    rows = db.due_alerts(conn, T0 + timedelta(minutes=1), retry_for=timedelta(hours=24))
    assert [r["native_id"] for r in rows] == ["due", "failed-recent"]  # failed-old is past the 24 h retry period


def test_mark_alert_sets_sent_at_and_counts_attempts(conn):
    with conn:
        db.add_alert(conn, "heads_up", "truthsocial", "1", T0, T0)
    row = db.due_alerts(conn, T0, retry_for=timedelta(hours=24))[0]
    with conn:
        db.mark_alert(conn, row["id"], "failed", T0, error="down")
        db.mark_alert(conn, row["id"], "sent", T0 + timedelta(minutes=5), title="t", message="m")
    got = conn.execute("SELECT * FROM alerts WHERE id = ?", (row["id"],)).fetchone()
    assert got["status"] == "sent" and got["attempts"] == 2
    assert got["sent_at"] == "2026-09-29T14:05:00Z"
    assert (got["title"], got["message"], got["error"]) == ("t", "m", "down")
```

- [ ] **Step 3: Run the tests to confirm they fail**

Run: `uv run --no-sync pytest tests/test_alert_config.py tests/test_alert_ledger.py -q`
Expected: FAIL: `AttributeError: 'Watchlist' object has no attribute 'alerts'` / `module 'influence_tracker.db' has no attribute 'add_alert'`.

- [ ] **Step 4: Implement the config**

In `src/influence_tracker/config.py`, add after `SentimentConfig`:

```python
class AlertsConfig(BaseModel):
    enabled: bool = True
    # Tickers the team holds: marked with a star, listed first, higher notification priority.
    holdings: list[str] = Field(default_factory=list)
    poll_active_minutes: int = Field(default=5, ge=1, le=60)
    poll_idle_minutes: int = Field(default=30, ge=1, le=240)
    poll_x_minutes: int = Field(default=15, ge=5, le=240)
    late_after_minutes: int = Field(default=30, ge=5, le=240)
    followup_minutes: int = Field(default=60, ge=5, le=390)
```

Add the field to `Watchlist` (after `sentiment`):

```python
    alerts: AlertsConfig = Field(default_factory=AlertsConfig)
```

At the end of `Watchlist._check`, before `return self`, add:

```python
        holdings: list[str] = []
        for raw in self.alerts.holdings:
            ticker = self.ticker(raw.strip())
            if ticker is None:
                raise ValueError(f"alerts.holdings: {raw.strip()!r} is not a configured ticker")
            if ticker.symbol not in holdings:
                holdings.append(ticker.symbol)
        self.alerts.holdings = holdings
```

Add two fields with defaults at the END of the `Settings` dataclass:

```python
    ntfy_topic: str | None = None
    ntfy_server: str = "https://ntfy.sh"
```

In `load_settings`, read them and pass them:

```python
    topic = os.environ.get("NTFY_TOPIC", "").strip() or None
    server = os.environ.get("NTFY_SERVER", "").strip() or "https://ntfy.sh"
    ...
        x_bearer_token=token,
        ntfy_topic=topic,
        ntfy_server=server,
```

Append to `config/watchlist.yaml`:

```yaml

alerts:
  enabled: true
  # Tickers your team holds, e.g. [NVDA, AAPL]: shown with a star, listed first, higher priority.
  holdings: []
  poll_active_minutes: 5     # 04:00-20:00 New York on trading days
  poll_idle_minutes: 30      # nights, weekends, holidays
  poll_x_minutes: 15         # only when X_BEARER_TOKEN is set
  late_after_minutes: 30     # posts found later than this go into one "while you were away" digest
  followup_minutes: 60
```

Append to `.env.example`:

```
# ntfy topic for phone alerts (run `influence alerts setup` to generate one). Treat it like a password.
NTFY_TOPIC=
```

- [ ] **Step 5: Implement the ledger**

Append to `_SCHEMA` in `src/influence_tracker/db.py` (before the closing `"""`):

```sql
-- Phone alerts sent or queued. UNIQUE is the no-duplicates guarantee: a row is 'pending' before sending and
-- 'sent' only after ntfy accepted it. Digest rows use platform '-' and the cycle's UTC ISO time as native_id.
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY,
    kind TEXT NOT NULL,           -- heads_up | follow_60m | follow_d1 | digest
    platform TEXT NOT NULL,
    native_id TEXT NOT NULL,
    due_at TEXT NOT NULL,         -- UTC ISO
    status TEXT NOT NULL,         -- pending | sent | failed | skipped
    attempts INTEGER NOT NULL DEFAULT 0,
    title TEXT,
    message TEXT,
    error TEXT,
    created_at TEXT NOT NULL,
    sent_at TEXT,
    UNIQUE (kind, platform, native_id)
);
CREATE INDEX IF NOT EXISTS alerts_due ON alerts (status, due_at);
```

Append helpers at the end of `db.py` (add `from datetime import timedelta` to the imports):

```python
# ---------------------------------------------------------------- alert ledger


def add_alert(
    conn: sqlite3.Connection,
    kind: str,
    platform: str,
    native_id: str,
    due_at: datetime,
    when: datetime,
    *,
    status: str = "pending",
    title: str | None = None,
    message: str | None = None,
    error: str | None = None,
) -> bool:
    cur = conn.execute(
        """INSERT OR IGNORE INTO alerts (kind, platform, native_id, due_at, status, title, message, error, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (kind, platform, native_id, to_iso(due_at), status, title, message, error, to_iso(when)),
    )
    return cur.rowcount == 1


def alert_status(conn: sqlite3.Connection, kind: str, platform: str, native_id: str) -> str | None:
    row = conn.execute(
        "SELECT status FROM alerts WHERE kind = ? AND platform = ? AND native_id = ?", (kind, platform, native_id)
    ).fetchone()
    return row["status"] if row else None


def due_alerts(conn: sqlite3.Connection, now: datetime, retry_for: timedelta) -> list[sqlite3.Row]:
    """Pending rows that are due, plus failed rows still inside the retry period, oldest due first."""
    return conn.execute(
        """SELECT * FROM alerts
           WHERE due_at <= ? AND (status = 'pending' OR (status = 'failed' AND due_at > ?))
           ORDER BY due_at, id""",
        (to_iso(now), to_iso(now - retry_for)),
    ).fetchall()


def mark_alert(
    conn: sqlite3.Connection,
    alert_id: int,
    status: str,
    when: datetime,
    *,
    title: str | None = None,
    message: str | None = None,
    error: str | None = None,
) -> None:
    conn.execute(
        """UPDATE alerts SET status = ?,
             attempts = attempts + CASE WHEN ? IN ('sent', 'failed') THEN 1 ELSE 0 END,
             title = COALESCE(?, title), message = COALESCE(?, message), error = COALESCE(?, error),
             sent_at = CASE WHEN ? = 'sent' THEN ? ELSE sent_at END
           WHERE id = ?""",
        (status, status, title, message, error, status, to_iso(when), alert_id),
    )
```

- [ ] **Step 6: Run the tests and the full suite**

Run: `uv run --no-sync pytest tests/test_alert_config.py tests/test_alert_ledger.py -q` → PASS.
Run: `uv run --no-sync pytest -q` → all pass. The schema is additive and `Settings` fields have defaults.

- [ ] **Step 7: Commit**

```bash
git add src/influence_tracker/config.py src/influence_tracker/db.py config/watchlist.yaml .env.example tests/test_alert_config.py tests/test_alert_ledger.py
git commit -m "Add alert config, ntfy settings and the alerts ledger table"
```

---

### Task 2: ntfy notifier

**Files:**
- Create: `src/influence_tracker/alerts/__init__.py` (empty)
- Create: `src/influence_tracker/alerts/notify.py`
- Test: `tests/test_notify.py`

**Interfaces:**
- Produces:
  - `notify.Message` (frozen dataclass: `title: str`, `body: str`, `priority: int = 3`, `tags: tuple[str, ...] = ()`, `click: str | None = None`).
  - `notify.Notifier` Protocol (`send(message) -> None`).
  - `notify.NotifyError(Exception)`.
  - `notify.MAX_BODY_BYTES = 4000`.
  - `notify.fit_body(text: str, limit: int = MAX_BODY_BYTES) -> str`.
  - `notify.NtfyNotifier(server: str, topic: str, client: httpx.Client | None = None, sleep=time.sleep)`.
  - `notify.DryRunNotifier()` (logs and keeps `.sent: list[Message]`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_notify.py
from __future__ import annotations

import json

import httpx
import pytest

from influence_tracker.alerts.notify import (
    MAX_BODY_BYTES,
    DryRunNotifier,
    Message,
    NotifyError,
    NtfyNotifier,
    fit_body,
)

MSG = Message(title="⭐ NVDA · realDonaldTrump · bullish (0.91)", body="Buy 🚀", priority=4, tags=("star",), click="https://x")


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_posts_json_to_server_root():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"id": "abc"})

    NtfyNotifier("https://ntfy.sh/", "topic-1", client=_client(handler), sleep=lambda s: None).send(MSG)
    [req] = seen
    assert str(req.url) == "https://ntfy.sh/"
    body = json.loads(req.content.decode("utf-8"))
    assert body == {
        "topic": "topic-1",
        "title": MSG.title,
        "message": "Buy 🚀",
        "priority": 4,
        "tags": ["star"],
        "click": "https://x",
    }


def test_retries_once_then_succeeds():
    codes = iter([500, 200])
    sleeps = []
    n = NtfyNotifier("https://ntfy.sh", "t", client=_client(lambda r: httpx.Response(next(codes))), sleep=sleeps.append)
    n.send(MSG)
    assert sleeps == [5.0]


def test_two_failures_raise():
    n = NtfyNotifier("https://ntfy.sh", "t", client=_client(lambda r: httpx.Response(503, text="down")), sleep=lambda s: None)
    with pytest.raises(NotifyError, match="503"):
        n.send(MSG)


def test_transport_errors_raise_notify_error():
    def boom(request):
        raise httpx.ConnectError("no route")

    n = NtfyNotifier("https://ntfy.sh", "t", client=_client(boom), sleep=lambda s: None)
    with pytest.raises(NotifyError, match="ConnectError"):
        n.send(MSG)


def test_fit_body_caps_utf8_bytes():
    text = "📈" * 3000
    out = fit_body(text)
    assert len(out.encode("utf-8")) <= MAX_BODY_BYTES
    assert out.endswith("…")
    assert fit_body("short") == "short"


def test_dry_run_records():
    n = DryRunNotifier()
    n.send(MSG)
    assert n.sent == [MSG]
```

- [ ] **Step 2: Run to confirm failure**

Run: `uv run --no-sync pytest tests/test_notify.py -q` → FAIL: `ModuleNotFoundError: influence_tracker.alerts`.

- [ ] **Step 3: Implement**

```python
# src/influence_tracker/alerts/notify.py
from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import httpx

log = logging.getLogger(__name__)

# ntfy turns bodies over 4,096 bytes into attachments; stay clearly below.
MAX_BODY_BYTES = 4000
RETRY_WAIT_S = 5.0
TIMEOUT_S = 10.0


@dataclass(frozen=True)
class Message:
    title: str
    body: str
    priority: int = 3
    tags: tuple[str, ...] = ()
    click: str | None = None


class Notifier(Protocol):
    def send(self, message: Message) -> None: ...


class NotifyError(Exception):
    pass


def fit_body(text: str, limit: int = MAX_BODY_BYTES) -> str:
    if len(text.encode("utf-8")) <= limit:
        return text
    ellipsis = "…"
    budget = limit - len(ellipsis.encode("utf-8"))
    cut = text.encode("utf-8")[:budget].decode("utf-8", errors="ignore")
    return cut.rstrip() + ellipsis


class NtfyNotifier:
    """Publishes through ntfy's JSON API. Header publishing would break on titles with emoji (headers are ASCII)."""

    def __init__(
        self,
        server: str,
        topic: str,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.url = server.rstrip("/") + "/"
        self.topic = topic
        self.client = client or httpx.Client(timeout=TIMEOUT_S)
        self.sleep = sleep

    def send(self, message: Message) -> None:
        payload: dict = {
            "topic": self.topic,
            "title": message.title,
            "message": fit_body(message.body),
            "priority": message.priority,
        }
        if message.tags:
            payload["tags"] = list(message.tags)
        if message.click:
            payload["click"] = message.click
        error = ""
        for attempt in (1, 2):
            try:
                resp = self.client.post(self.url, json=payload)
                if resp.status_code < 300:
                    return
                error = f"HTTP {resp.status_code}: {resp.text[:200]}"
            except httpx.HTTPError as e:
                error = f"{type(e).__name__}: {e}"
            if attempt == 1:
                self.sleep(RETRY_WAIT_S)
        raise NotifyError(error)


class DryRunNotifier:
    def __init__(self) -> None:
        self.sent: list[Message] = []

    def send(self, message: Message) -> None:
        self.sent.append(message)
        log.info("[dry run] would notify: %s | %s", message.title, message.body.replace("\n", " / "))
```

- [ ] **Step 4: Run the tests**

Run: `uv run --no-sync pytest tests/test_notify.py -q` → PASS. Then run ruff check and ruff format --check on both files.

- [ ] **Step 5: Commit**

```bash
git add src/influence_tracker/alerts/__init__.py src/influence_tracker/alerts/notify.py tests/test_notify.py
git commit -m "Add the ntfy notifier (JSON publish, one retry, 4 KB body cap)"
```

---

### Task 3: Timing rules (cadence and follow-up windows)

**Files:**
- Create: `src/influence_tracker/alerts/timing.py`
- Test: `tests/test_alert_timing.py`

**Interfaces:**
- Consumes: `market.is_session`, `market.extended_bounds_utc`, `market.session_bounds_utc`, `market.sessions_in_range`, `config.AlertsConfig`.
- Produces:
  - `timing.is_active(now: datetime) -> bool`.
  - `timing.next_cycle(now: datetime, cfg: AlertsConfig) -> datetime`.
  - `timing.FollowWindow(start: datetime, end: datetime, truncated: bool)`.
  - `timing.follow_window(t0: datetime, d0: date, minutes: int) -> FollowWindow`.
  - `timing.FOLLOW_DELAY = timedelta(minutes=3)`, `timing.STALE_AFTER = timedelta(minutes=30)`.
  - `timing.is_late(created_at: datetime, found_at: datetime, cfg: AlertsConfig) -> bool`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_alert_timing.py
from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest

from influence_tracker.alerts import timing
from influence_tracker.config import AlertsConfig
from influence_tracker.timeutil import NY

CFG = AlertsConfig()


def ny(y, m, d, hh, mm, ss=0) -> datetime:
    return datetime(y, m, d, hh, mm, ss, tzinfo=NY)


@pytest.mark.parametrize(
    ("t", "active"),
    [
        (ny(2026, 9, 29, 3, 59), False),
        (ny(2026, 9, 29, 4, 0), True),
        (ny(2026, 9, 29, 19, 59), True),
        (ny(2026, 9, 29, 20, 0), False),
        (ny(2026, 10, 3, 12, 0), False),  # Saturday
        (ny(2026, 11, 26, 12, 0), False),  # Thanksgiving
        (ny(2026, 11, 27, 17, 0), True),  # early-close day: still inside 04:00-20:00
    ],
)
def test_is_active(t, active):
    assert timing.is_active(t) is active


def test_next_cycle_aligns_to_the_active_interval():
    assert timing.next_cycle(ny(2026, 9, 29, 10, 31, 20), CFG) == ny(2026, 9, 29, 10, 35)


def test_next_cycle_idle_interval_at_night():
    assert timing.next_cycle(ny(2026, 9, 29, 21, 5), CFG) == ny(2026, 9, 29, 21, 30)


def test_next_cycle_wakes_exactly_at_the_active_window_start():
    assert timing.next_cycle(ny(2026, 11, 2, 3, 40), CFG) == ny(2026, 11, 2, 4, 0)


def test_follow_window_regular_session():
    w = timing.follow_window(ny(2026, 9, 29, 10, 31), date(2026, 9, 29), 60)
    assert (w.start, w.end, w.truncated) == (ny(2026, 9, 29, 10, 31), ny(2026, 9, 29, 11, 31), False)


def test_follow_window_truncated_at_close():
    w = timing.follow_window(ny(2026, 9, 29, 15, 30), date(2026, 9, 29), 60)
    assert (w.end, w.truncated) == (ny(2026, 9, 29, 16, 0), True)


def test_follow_window_premarket_post_runs_to_first_trading_hour():
    w = timing.follow_window(ny(2026, 9, 29, 8, 0), date(2026, 9, 29), 60)
    assert (w.start, w.end) == (ny(2026, 9, 29, 8, 0), ny(2026, 9, 29, 10, 30))


def test_follow_window_weekend_post():
    w = timing.follow_window(ny(2026, 10, 4, 14, 5), date(2026, 10, 5), 60)
    assert w.end == ny(2026, 10, 5, 10, 30)


def test_follow_window_early_close():
    w = timing.follow_window(ny(2026, 11, 27, 12, 40), date(2026, 11, 27), 60)
    assert (w.end, w.truncated) == (ny(2026, 11, 27, 13, 0), True)


def test_is_late():
    t = ny(2026, 9, 29, 10, 0)
    assert timing.is_late(t, t + timedelta(minutes=30), CFG) is False
    assert timing.is_late(t, t + timedelta(minutes=31), CFG) is True
```

- [ ] **Step 2: Run to confirm failure**

Run: `uv run --no-sync pytest tests/test_alert_timing.py -q` → FAIL (module missing).

- [ ] **Step 3: Implement**

```python
# src/influence_tracker/alerts/timing.py
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

from .. import market
from ..config import AlertsConfig
from ..timeutil import NY

FOLLOW_DELAY = timedelta(minutes=3)  # give Yahoo time to publish the window's last bar
STALE_AFTER = timedelta(minutes=30)
LOOKAHEAD_DAYS = 10


def is_active(now: datetime) -> bool:
    """Inside 04:00-20:00 New York on an XNYS session day."""
    day = now.astimezone(NY).date()
    if not market.is_session(day):
        return False
    start, end = market.extended_bounds_utc(day)
    return start <= now < end


def _next_active_start(now: datetime) -> datetime | None:
    today = now.astimezone(NY).date()
    for session in market.sessions_in_range(today, today + timedelta(days=LOOKAHEAD_DAYS)):
        start = market.extended_bounds_utc(session)[0]
        if start > now:
            return start
    return None


def next_cycle(now: datetime, cfg: AlertsConfig) -> datetime:
    """The next wall-clock-aligned tick for the current window; an idle wait never overshoots the next window start."""
    minutes = cfg.poll_active_minutes if is_active(now) else cfg.poll_idle_minutes
    step = minutes * 60
    tick = datetime.fromtimestamp((math.floor(now.timestamp() / step) + 1) * step, UTC)
    if not is_active(now):
        start = _next_active_start(now)
        if start is not None and start < tick:
            return start
    return tick


@dataclass(frozen=True)
class FollowWindow:
    start: datetime
    end: datetime
    truncated: bool


def follow_window(t0: datetime, d0: date, minutes: int) -> FollowWindow:
    """From the post to `minutes` into trading: the next hour for a regular-session post; for an off-hours post
    through the first hour of d0 (opening gap included). Clipped at d0's (early) close."""
    open_, close = market.session_bounds_utc(d0)
    end = max(t0, open_) + timedelta(minutes=minutes)
    return FollowWindow(t0, min(end, close), end > close)


def is_late(created_at: datetime, found_at: datetime, cfg: AlertsConfig) -> bool:
    return found_at - created_at > timedelta(minutes=cfg.late_after_minutes)
```

- [ ] **Step 4: Run the tests**

Run: `uv run --no-sync pytest tests/test_alert_timing.py -q` → PASS (ruff clean).

- [ ] **Step 5: Commit**

```bash
git add src/influence_tracker/alerts/timing.py tests/test_alert_timing.py
git commit -m "Add alert cadence and follow-up window rules"
```

---

### Task 4: Single-event market model and live 1-minute prices

**Files:**
- Modify: `src/influence_tracker/analysis/metrics.py` (public `EventModel`, `event_model`)
- Create: `src/influence_tracker/alerts/live.py`
- Test: `tests/test_event_model.py`, `tests/test_live_prices.py`

**Interfaces:**
- Consumes: `metrics._Daily`, `metrics._fit`, `study.PATH_DAYS`, `events.MinuteBars`, `prices.fetch_yahoo_1m(yahoo_symbol, start, end) -> DataFrame`.
- Produces:
  - `metrics.EventModel` (frozen: `alpha`, `beta`, `sigma`: float; `n_est`: int; `ar: dict[int, float]` keyed by day offset −5..+5; method `car(first: int, last: int) -> tuple[float, float]` returning (CAR, z), NaN when any AR is missing).
  - `metrics.event_model(conn, ticker: str, d0: date) -> EventModel | None`.
  - `live.PriceSource` Protocol (`bars(symbol, start, end) -> MinuteBars`).
  - `live.frame_to_bars(df) -> MinuteBars`.
  - `live.LivePrices(watchlist, fetch=None)`, which implements PriceSource and caches per (symbol, start, end).

- [ ] **Step 1: Write the failing event-model tests**

```python
# tests/test_event_model.py
from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from influence_tracker import market
from influence_tracker.analysis import metrics


def _write_daily(conn, symbol, sessions, closes):
    conn.executemany(
        """INSERT OR REPLACE INTO bars_1d (symbol, session_date, open, high, low, close, adj_close, volume,
                                           split_ratio, fetched_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, 1000000, 0, '2026-01-01T00:00:00Z')""",
        [(symbol, s.isoformat(), c, c, c, c, c) for s, c in zip(sessions, closes, strict=True)],
    )


@pytest.fixture
def synthetic(conn):
    rng = np.random.default_rng(7)
    sessions = market.sessions_in_range(date(2025, 6, 2), date(2026, 6, 30))
    m = rng.normal(0, 0.01, len(sessions))
    e = rng.normal(0, 0.004, len(sessions))
    r = 0.0002 + 1.5 * m + e
    k = 200
    r[k] += 0.05
    with conn:
        _write_daily(conn, "SPY", sessions, 400 * np.cumprod(1 + m))
        _write_daily(conn, "NVDA", sessions, 100 * np.cumprod(1 + r))
    return sessions, k


def test_event_model_recovers_beta_and_the_jump(conn, synthetic):
    sessions, k = synthetic
    model = metrics.event_model(conn, "NVDA", sessions[k])
    assert model is not None
    assert model.n_est == 120
    assert model.beta == pytest.approx(1.5, abs=0.1)
    assert model.ar[0] == pytest.approx(0.05, abs=0.015)
    car, z = model.car(0, 0)
    assert car == pytest.approx(model.ar[0])
    assert z > 5


def test_car_is_nan_when_a_day_is_missing(conn, synthetic):
    sessions, k = synthetic
    with conn:
        conn.execute("DELETE FROM bars_1d WHERE symbol = 'NVDA' AND session_date = ?", (sessions[k + 1].isoformat(),))
    model = metrics.event_model(conn, "NVDA", sessions[k])
    car, z = model.car(0, 1)
    assert np.isnan(car) and np.isnan(z)


def test_no_model_without_history(conn, synthetic):
    sessions, _ = synthetic
    assert metrics.event_model(conn, "NVDA", sessions[30]) is None
    assert metrics.event_model(conn, "AAPL", sessions[200]) is None
```

- [ ] **Step 2: Write the failing live-price tests**

```python
# tests/test_live_prices.py
from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from influence_tracker.alerts.live import LivePrices, frame_to_bars


def _frame(rows):
    idx = pd.DatetimeIndex([pd.Timestamp(t, tz="America/New_York") for t, _ in rows])
    return pd.DataFrame({"Close": [c for _, c in rows]}, index=idx)


def test_frame_to_bars_uses_bar_starts_and_drops_bad_rows():
    df = _frame([("2026-09-29 10:00", 10.0), ("2026-09-29 10:01", float("nan")), ("2026-09-29 10:02", 11.0)])
    bars = frame_to_bars(df)
    assert bars.starts == [int(datetime(2026, 9, 29, 14, 0, tzinfo=UTC).timestamp()),
                           int(datetime(2026, 9, 29, 14, 2, tzinfo=UTC).timestamp())]
    assert bars.closes == [10.0, 11.0]
    assert frame_to_bars(pd.DataFrame()).starts == []


def test_live_prices_maps_symbols_caches_and_survives_errors(watchlist):
    calls = []

    def fetch(symbol, start, end):
        calls.append(symbol)
        if symbol == "BRK-B":
            raise RuntimeError("yahoo down")
        return _frame([("2026-09-29 10:00", 10.0)])

    live = LivePrices(watchlist, fetch=fetch)
    s, e = datetime(2026, 9, 29, 13, 0, tzinfo=UTC), datetime(2026, 9, 29, 15, 0, tzinfo=UTC)
    assert live.bars("NVDA", s, e).closes == [10.0]
    assert live.bars("NVDA", s, e).closes == [10.0]
    assert live.bars("BRK.B", s, e).starts == []
    assert calls == ["NVDA", "BRK-B"]
```

- [ ] **Step 3: Run to confirm failure**

Run: `uv run --no-sync pytest tests/test_event_model.py tests/test_live_prices.py -q` → FAIL (`event_model` / module missing).

- [ ] **Step 4: Implement `event_model`**

Add to `src/influence_tracker/analysis/metrics.py`, directly after `_fit`:

```python
@dataclass(frozen=True)
class EventModel:
    """One event's market model and abnormal returns around d0, for live follow-ups before the event completes."""

    alpha: float
    beta: float
    sigma: float
    n_est: int
    ar: dict[int, float]

    def car(self, first: int, last: int) -> tuple[float, float]:
        values = [self.ar.get(k, _NAN) for k in range(first, last + 1)]
        if not all(math.isfinite(v) for v in values):
            return _NAN, _NAN
        car = float(sum(values))
        return car, car / (self.sigma * math.sqrt(last - first + 1))


def event_model(conn: sqlite3.Connection, ticker: str, d0: date) -> EventModel | None:
    """The same market model compute_study fits (sessions -130..-11 before d0), for a single (ticker, d0)."""
    daily = _Daily(conn, [ticker, MARKET])
    anchor = daily.index(d0)
    if anchor < 0 or ticker not in daily.row:
        return None
    fit = _fit(daily, [daily.symbol_row(ticker)], [anchor])
    if not bool(fit.ok[0]):
        return None
    return EventModel(
        alpha=float(fit.alpha[0]),
        beta=float(fit.beta[0]),
        sigma=float(fit.sigma[0]),
        n_est=int(fit.n[0]),
        ar={k: float(fit.ar[0, i]) for i, k in enumerate(PATH_DAYS)},
    )
```

(`math`, `sqlite3`, `date`, `dataclass`, `MARKET`, `PATH_DAYS` and `_NAN` are already imported or defined in `metrics.py`. Check them and add any that are missing.)

- [ ] **Step 5: Implement live prices**

```python
# src/influence_tracker/alerts/live.py
from __future__ import annotations

import logging
import math
from collections.abc import Callable
from datetime import datetime
from typing import Protocol

import pandas as pd

from .. import prices
from ..config import Watchlist
from ..events import MinuteBars

log = logging.getLogger(__name__)

Fetch1m = Callable[[str, datetime, datetime], pd.DataFrame]


class PriceSource(Protocol):
    def bars(self, symbol: str, start: datetime, end: datetime) -> MinuteBars: ...


def frame_to_bars(df: pd.DataFrame) -> MinuteBars:
    """Yahoo 1-minute frame (index = bar start) -> MinuteBars; NaN/non-positive closes dropped, duplicates keep last."""
    if df is None or df.empty or "Close" not in df:
        return MinuteBars([], [])
    by_start: dict[int, float] = {}
    for ts, close in df["Close"].items():
        value = float(close)
        if math.isfinite(value) and value > 0:
            by_start[int(pd.Timestamp(ts).timestamp())] = value
    starts = sorted(by_start)
    return MinuteBars(starts, [by_start[s] for s in starts])


class LivePrices:
    """Live Yahoo 1-minute bars for one watch cycle; failures come back as empty bars (the caller treats a missing
    price as not yet available)."""

    def __init__(self, watchlist: Watchlist, fetch: Fetch1m | None = None) -> None:
        self.watchlist = watchlist
        self.fetch = fetch or prices.fetch_yahoo_1m
        self._cache: dict[tuple[str, datetime, datetime], MinuteBars] = {}

    def bars(self, symbol: str, start: datetime, end: datetime) -> MinuteBars:
        key = (symbol, start, end)
        if key not in self._cache:
            ticker = self.watchlist.ticker(symbol)
            yahoo = ticker.yahoo_symbol if ticker else symbol
            try:
                self._cache[key] = frame_to_bars(self.fetch(yahoo, start, end))
            except Exception as e:  # yfinance raises many unrelated types for network and data problems
                log.warning("live 1m fetch for %s failed: %s: %s", symbol, type(e).__name__, e)
                self._cache[key] = MinuteBars([], [])
        return self._cache[key]
```

- [ ] **Step 6: Run the tests**

Run: `uv run --no-sync pytest tests/test_event_model.py tests/test_live_prices.py tests/test_metrics.py -q` → PASS (the existing metrics tests stay green).

- [ ] **Step 7: Commit**

```bash
git add src/influence_tracker/analysis/metrics.py src/influence_tracker/alerts/live.py tests/test_event_model.py tests/test_live_prices.py
git commit -m "Add a single-event market model and live 1-minute prices for alerts"
```

---

### Task 5: Message formatting

**Files:**
- Create: `src/influence_tracker/alerts/messages.py`
- Test: `tests/test_alert_messages.py`

**Interfaces:**
- Consumes: `notify.Message`, `notify.fit_body`, `events.PricePoint`, `study.MIN_GROUP_POSTS`, `study.Z_CRITICAL`, `timeutil.NY`.
- Produces:
  - `messages.PostInfo` (frozen: `platform`, `native_id`, `author`, `created_at: datetime`, `text`, `url`, `stance: str | None`, `stance_conf: float | None`, `tickers: tuple[str, ...]`).
  - `messages.History` (frozen: `n_posts: int`, `mean_signed_car: float`, `p_holm: float | None`).
  - `messages.Follow60Row` (frozen: `ticker`, `ret: float`, `spy_ret: float | None`, `abnormal: float | None`).
  - `messages.D1Row` (frozen: `ticker`, `car: float | None`, `z: float | None`, `notes: tuple[str, ...]`).
  - `messages.order_tickers(tickers, holdings) -> list[str]`.
  - `messages.heads_up(post, holdings: set[str], prices: dict[str, PricePoint | None], history: History | None) -> Message`.
  - `messages.follow_60m(post, holdings, rows: list[Follow60Row], window_text: str) -> Message`.
  - `messages.follow_d1(post, holdings, rows: list[D1Row]) -> Message`.
  - `messages.digest(posts: list[PostInfo]) -> Message`.
  - `messages.et(dt) -> str` (e.g. `"Tue Sep 29 10:31 ET"`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_alert_messages.py
from __future__ import annotations

from datetime import UTC, datetime

from influence_tracker.alerts import messages as m
from influence_tracker.alerts.notify import MAX_BODY_BYTES
from influence_tracker.events import PricePoint

T0 = datetime(2026, 9, 29, 14, 31, tzinfo=UTC)  # 10:31 ET


def post(**kw) -> m.PostInfo:
    base = dict(platform="truthsocial", native_id="1", author="realDonaldTrump", created_at=T0,
                text="NVIDIA is doing GREAT", url="https://truthsocial.com/@realDonaldTrump/1",
                stance="bullish", stance_conf=0.91, tickers=("NVDA",))
    base.update(kw)
    return m.PostInfo(**base)


def test_heads_up_basic():
    msg = m.heads_up(post(), set(), {"NVDA": PricePoint(int(T0.timestamp()) - 60, 182.41)}, None)
    assert msg.title == "NVDA · realDonaldTrump · bullish (0.91)"
    assert msg.priority == 3 and msg.tags == ("chart_with_upwards_trend",)
    assert msg.click == "https://truthsocial.com/@realDonaldTrump/1"
    assert "NVIDIA is doing GREAT" in msg.body
    assert "Posted Tue Sep 29 10:31 ET" in msg.body
    assert "NVDA $182.41" in msg.body
    assert "History: fewer than 10 past posts by realDonaldTrump." in msg.body


def test_heads_up_holdings_first_with_star_and_high_priority():
    msg = m.heads_up(post(tickers=("AAPL", "NVDA")), {"NVDA"}, {}, None)
    assert msg.title.startswith("⭐ NVDA, AAPL · ")
    assert msg.priority == 4 and "star" in msg.tags
    assert "Price at post: unavailable" in msg.body


def test_heads_up_folds_many_tickers_and_caps_long_text():
    text = "🚀 Huge news for American companies! " * 100
    msg = m.heads_up(post(text=text, tickers=("AAPL", "INTC", "MSFT", "NVDA", "QCOM", "TSLA")), {"TSLA"}, {}, None)
    assert msg.title.startswith("⭐ TSLA, AAPL +4 · ")
    first_line = msg.body.splitlines()[0]
    assert len(first_line) <= 221 and first_line.endswith("…")
    assert len(msg.body.encode("utf-8")) <= MAX_BODY_BYTES


def test_heads_up_price_as_of_note_for_old_bars():
    stale = PricePoint(int(T0.timestamp()) - 3 * 3600, 99.5)
    msg = m.heads_up(post(), set(), {"NVDA": stale}, None)
    assert "NVDA $99.50 (as of Tue Sep 29 07:31 ET)" in msg.body


def test_heads_up_without_stance():
    msg = m.heads_up(post(stance=None, stance_conf=None), set(), {}, None)
    assert msg.title.endswith("· stance unavailable") and msg.tags == ("grey_question",)


def test_history_line_significant_and_not():
    ok = m.heads_up(post(), set(), {}, m.History(71, -0.00604, 0.50))
    assert ("History: realDonaldTrump's posts moved their stocks -0.60% on average over 2 days "
            "(n=71, Holm p=0.50, not significant).") in ok.body
    sig = m.heads_up(post(), set(), {}, m.History(40, 0.0123, 0.01))
    assert "(n=40, Holm p=0.01, significant at the 5% level)." in sig.body


def test_follow_60m_lines():
    rows = [m.Follow60Row("NVDA", 0.0084, 0.0010, 0.0069), m.Follow60Row("AAPL", -0.002, None, None)]
    msg = m.follow_60m(post(tickers=("AAPL", "NVDA")), set(), rows, "From the post (10:31 ET) to 11:31 ET")
    assert msg.title == "AAPL, NVDA · 60 min after realDonaldTrump's post"
    assert "NVDA +0.84% vs SPY +0.10% → abnormal +0.69%" in msg.body
    assert "AAPL -0.20% (SPY unavailable)" in msg.body
    assert "From the post (10:31 ET) to 11:31 ET" in msg.body
    assert msg.priority == 2


def test_follow_d1_wording():
    rows = [
        m.D1Row("NVDA", 0.012, 1.95, ()),
        m.D1Row("AAPL", -0.041, -2.40, ("earnings day — the move may be the report, not the post",)),
        m.D1Row("MSFT", None, None, ()),
    ]
    msg = m.follow_d1(post(tickers=("AAPL", "MSFT", "NVDA")), {"AAPL"}, rows)
    assert "NVDA CAR[0,+1] +1.20% (z +1.95) — within the normal range" in msg.body
    assert "AAPL CAR[0,+1] -4.10% (z -2.40) — unusually large (|z| = 2.40); earnings day" in msg.body
    assert "MSFT: no market model (too little price history)" in msg.body
    assert msg.priority == 3 and msg.title.startswith("⭐ AAPL, MSFT +1 · day-after check")


def test_digest_caps_at_eight():
    posts = [post(native_id=str(i), tickers=("NVDA", "INTC")) for i in range(11)]
    msg = m.digest(posts)
    assert msg.title == "While you were away: 11 stock posts"
    lines = msg.body.splitlines()
    assert len(lines) == 9 and lines[-1] == "+3 more — run: influence events"
    assert lines[0] == "Tue Sep 29 10:31 ET · realDonaldTrump · INTC, NVDA · bullish"
    assert msg.priority == 2
```

- [ ] **Step 2: Run to confirm failure**

Run: `uv run --no-sync pytest tests/test_alert_messages.py -q` → FAIL (module missing).

- [ ] **Step 3: Implement**

```python
# src/influence_tracker/alerts/messages.py
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime

from ..analysis.study import MIN_GROUP_POSTS, Z_CRITICAL
from ..events import PricePoint
from ..timeutil import NY
from .notify import Message, fit_body

TEXT_CHARS = 220
TITLE_TICKERS = 2
PRICE_TICKERS = 3
DIGEST_POSTS = 8
STALE_PRICE_S = 15 * 60
STANCE_TAGS = {"bullish": "chart_with_upwards_trend", "bearish": "chart_with_downwards_trend", "neutral": "speech_balloon"}
FOOTER = "Abnormal = the move beyond what SPY explains. Within the normal range = the size ordinary days produce."


@dataclass(frozen=True)
class PostInfo:
    platform: str
    native_id: str
    author: str
    created_at: datetime
    text: str
    url: str
    stance: str | None
    stance_conf: float | None
    tickers: tuple[str, ...]


@dataclass(frozen=True)
class History:
    n_posts: int
    mean_signed_car: float
    p_holm: float | None


@dataclass(frozen=True)
class Follow60Row:
    ticker: str
    ret: float
    spy_ret: float | None
    abnormal: float | None


@dataclass(frozen=True)
class D1Row:
    ticker: str
    car: float | None
    z: float | None
    notes: tuple[str, ...]


def et(dt: datetime) -> str:
    local = dt.astimezone(NY)
    return f"{local:%a} {local:%b} {local.day} {local:%H:%M} ET"


def _pct(x: float) -> str:
    return f"{x * 100:+.2f}%"


def order_tickers(tickers: Iterable[str], holdings: set[str]) -> list[str]:
    ts = sorted(set(tickers))
    return [t for t in ts if t in holdings] + [t for t in ts if t not in holdings]


def _ticker_label(tickers: Iterable[str], holdings: set[str]) -> tuple[str, bool]:
    ordered = order_tickers(tickers, holdings)
    held = any(t in holdings for t in ordered)
    shown, rest = ordered[:TITLE_TICKERS], len(ordered) - TITLE_TICKERS
    label = ", ".join(shown) + (f" +{rest}" if rest > 0 else "")
    return ("⭐ " if held else "") + label, held


def _clip(text: str, limit: int = TEXT_CHARS) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[:limit].rstrip() + "…"


def _stance(post: PostInfo) -> str:
    if post.stance is None:
        return "stance unavailable"
    return f"{post.stance} ({post.stance_conf:.2f})" if post.stance_conf is not None else post.stance


def _history_line(author: str, history: History | None) -> str:
    if history is None or history.n_posts < MIN_GROUP_POSTS or history.p_holm is None:
        return f"History: fewer than {MIN_GROUP_POSTS} past posts by {author}."
    verdict = "significant at the 5% level" if history.p_holm < 0.05 else "not significant"
    return (
        f"History: {author}'s posts moved their stocks {_pct(history.mean_signed_car)} on average over 2 days "
        f"(n={history.n_posts}, Holm p={history.p_holm:.2f}, {verdict})."
    )


def heads_up(
    post: PostInfo, holdings: set[str], prices: dict[str, PricePoint | None], history: History | None
) -> Message:
    label, held = _ticker_label(post.tickers, holdings)
    bits = []
    for t in order_tickers(post.tickers, holdings)[:PRICE_TICKERS]:
        point = prices.get(t)
        if point is None:
            continue
        note = ""
        if post.created_at.timestamp() - point.ts > STALE_PRICE_S:
            note = f" (as of {et(datetime.fromtimestamp(point.ts, post.created_at.tzinfo))})"
        bits.append(f"{t} ${point.price:,.2f}{note}")
    lines = [
        _clip(post.text),
        "",
        f"Posted {et(post.created_at)}",
        "Price at post: " + (" · ".join(bits) if bits else "unavailable"),
        _history_line(post.author, history),
    ]
    tags = (STANCE_TAGS.get(post.stance or "", "grey_question"),) + (("star",) if held else ())
    return Message(
        title=f"{label} · {post.author} · {_stance(post)}",
        body=fit_body("\n".join(lines)),
        priority=4 if held else 3,
        tags=tags,
        click=post.url,
    )


def follow_60m(post: PostInfo, holdings: set[str], rows: list[Follow60Row], window_text: str) -> Message:
    label, held = _ticker_label(post.tickers, holdings)
    lines = []
    for r in rows:
        if r.spy_ret is None:
            lines.append(f"{r.ticker} {_pct(r.ret)} (SPY unavailable)")
        elif r.abnormal is None:
            lines.append(f"{r.ticker} {_pct(r.ret)} vs SPY {_pct(r.spy_ret)} (no market model)")
        else:
            lines.append(f"{r.ticker} {_pct(r.ret)} vs SPY {_pct(r.spy_ret)} → abnormal {_pct(r.abnormal)}")
    lines += [window_text, FOOTER]
    return Message(
        title=f"{label} · 60 min after {post.author}'s post",
        body=fit_body("\n".join(lines)),
        priority=3 if held else 2,
        tags=("hourglass",),
        click=post.url,
    )


def follow_d1(post: PostInfo, holdings: set[str], rows: list[D1Row]) -> Message:
    label, held = _ticker_label(post.tickers, holdings)
    lines = []
    for r in rows:
        if r.car is None or r.z is None:
            lines.append(f"{r.ticker}: no market model (too little price history)")
            continue
        verdict = (
            "within the normal range" if abs(r.z) < Z_CRITICAL else f"unusually large (|z| = {abs(r.z):.2f})"
        )
        line = f"{r.ticker} CAR[0,+1] {_pct(r.car)} (z {r.z:+.2f}) — {verdict}"
        if r.notes:
            line += "; " + "; ".join(r.notes)
        lines.append(line)
    lines.append(FOOTER)
    return Message(
        title=f"{label} · day-after check on {post.author}'s post",
        body=fit_body("\n".join(lines)),
        priority=3 if held else 2,
        tags=("bar_chart",),
        click=post.url,
    )


def digest(posts: list[PostInfo]) -> Message:
    total = len(posts)
    lines = [
        f"{et(p.created_at)} · {p.author} · {', '.join(sorted(p.tickers)[:4])} · {p.stance or 'stance n/a'}"
        for p in posts[:DIGEST_POSTS]
    ]
    if total > DIGEST_POSTS:
        lines.append(f"+{total - DIGEST_POSTS} more — run: influence events")
    return Message(
        title=f"While you were away: {total} stock post{'' if total == 1 else 's'}",
        body=fit_body("\n".join(lines)),
        priority=2,
        tags=("mailbox_with_mail",),
    )
```

Add one more test so the date format is pinned: `assert m.et(datetime(2026, 9, 9, 12, 5, tzinfo=UTC)) == "Wed Sep 9 08:05 ET"`.

- [ ] **Step 4: Run the tests**

Run: `uv run --no-sync pytest tests/test_alert_messages.py -q` → PASS (ruff clean).

- [ ] **Step 5: Commit**

```bash
git add src/influence_tracker/alerts/messages.py tests/test_alert_messages.py
git commit -m "Add alert message formatting (heads-up, follow-ups, digest)"
```

---

### Task 6: Alert engine — detection, ledger, heads-up, digest, retries

**Files:**
- Create: `src/influence_tracker/alerts/engine.py`
- Test: `tests/test_alert_engine.py`

**Interfaces:**
- Consumes: everything from Tasks 1–5; `market.event_session`; `db.*`; `timeutil.to_iso/from_iso/parse_api_time`; `analysis.metrics.compute_study`.
- Produces:
  - `engine.ALERT_PLATFORMS = ("truthsocial", "x")`.
  - `engine.StudyHistory(conn, watchlist, now)`, a callable `author -> History | None` that computes lazily once.
  - `engine.AlertEngine(conn, watchlist, notifier, *, live_factory: Callable[[], PriceSource], history_factory: Callable[[datetime], Callable[[str], History | None]], model: Callable[[str, date], EventModel | None] | None = None)` with:
    - `initialize(now) -> int` (existing posts marked; 0 after the first time);
    - `queue(now) -> dict`;
    - `send_due(now) -> dict`;
    - `run(now) -> dict` (= initialize + queue + send_due, counts merged with `status`).
  - `engine.post_info(conn, platform, native_id, tickers) -> PostInfo` (also used by Task 7).
  - `engine.RETRY_FOR = timedelta(hours=24)`.

Queue rules (from the spec):

- A post is **new** when it is on an alert platform, mentions ≥ 1 non-benchmark ticker, and has no `heads_up` row.
- A new post that is **not late** gets:
  - `heads_up` (due now);
  - `follow_60m` (due `follow_window(t0, d0, followup_minutes).end + FOLLOW_DELAY`);
  - `follow_d1` (due at 20:00 New York on `session_offset(d0, 1)`).
- A **late** post gets:
  - `heads_up` with status `skipped` (error `late: sent in digest`);
  - `follow_d1` only.
- All late posts found in one cycle go into **one** `digest` row. Its `platform` is `-`, its `native_id` is the cycle's ISO time, and its title and message are rendered at queue time.

Task 7 fills in follow-up sending. Until then, `send_due` leaves `follow_60m`/`follow_d1` rows untouched.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_alert_engine.py
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from influence_tracker import db
from influence_tracker.alerts.engine import AlertEngine
from influence_tracker.alerts.messages import History
from influence_tracker.alerts.notify import Message, NotifyError
from influence_tracker.events import MinuteBars
from influence_tracker.models import Mention, Post

T_POST = datetime(2026, 9, 29, 14, 31, tzinfo=UTC)  # Tue 10:31 ET


class Recorder:
    def __init__(self, failures: int = 0) -> None:
        self.sent: list[Message] = []
        self.failures = failures

    def send(self, message: Message) -> None:
        if self.failures:
            self.failures -= 1
            raise NotifyError("ntfy down")
        self.sent.append(message)


class NoPrices:
    def bars(self, symbol, start, end):
        return MinuteBars([], [])


def seed(conn, native_id, created, tickers, *, platform="truthsocial", author="realDonaldTrump",
         text="Great news for NVIDIA", stance=("bullish", 0.91)):
    p = Post(platform=platform, native_id=native_id, author=author, created_at_utc=created, text=text,
             url=f"https://example.com/{native_id}")
    with conn:
        db.upsert_post(conn, p, created)
        db.replace_mentions(conn, platform, native_id, [Mention(t, "cashtag", f"${t}") for t in tickers], [], created)
        if stance:
            conn.execute("UPDATE posts SET stance = ?, stance_conf = ?, stance_model = 'm' "
                         "WHERE platform = ? AND native_id = ?", (*stance, platform, native_id))


def make(conn, watchlist, notifier=None, history=None):
    return AlertEngine(
        conn, watchlist, notifier or Recorder(),
        live_factory=NoPrices,
        history_factory=lambda now: (lambda author: history),
        model=lambda ticker, d0: None,
    )


def statuses(conn, kind):
    return {r["native_id"]: r["status"] for r in conn.execute("SELECT * FROM alerts WHERE kind = ?", (kind,))}


def test_first_run_marks_existing_posts_and_sends_nothing(conn, watchlist):
    seed(conn, "old1", T_POST - timedelta(days=100), ["NVDA"])
    seed(conn, "old2", T_POST - timedelta(days=5), ["AAPL", "INTC"])
    rec = Recorder()
    eng = make(conn, watchlist, rec)
    counts = eng.run(T_POST)
    assert rec.sent == []
    assert statuses(conn, "heads_up") == {"old1": "skipped", "old2": "skipped"}
    assert statuses(conn, "follow_60m") == {} and statuses(conn, "follow_d1") == {}
    assert counts["initialized"] == 2
    assert eng.run(T_POST + timedelta(minutes=5))["initialized"] == 0


def test_fresh_post_gets_one_heads_up_and_queued_followups(conn, watchlist):
    eng = make(conn, watchlist, rec := Recorder(), history=History(71, -0.006, 0.5))
    eng.run(T_POST - timedelta(minutes=10))  # initialise on an empty history
    seed(conn, "p1", T_POST, ["NVDA"])
    eng.run(T_POST + timedelta(minutes=4))
    eng.run(T_POST + timedelta(minutes=9))
    assert [m.title for m in rec.sent] == ["NVDA · realDonaldTrump · bullish (0.91)"]
    assert "n=71, Holm p=0.50" in rec.sent[0].body
    assert statuses(conn, "heads_up") == {"p1": "sent"}
    due = {r["kind"]: r["due_at"] for r in conn.execute("SELECT kind, due_at FROM alerts WHERE native_id = 'p1'")}
    assert due["follow_60m"] == "2026-09-29T15:34:00Z"  # 11:31 ET + 3 min
    assert due["follow_d1"] == "2026-10-01T00:00:00Z"  # Wed Sep 30 20:00 ET


def test_ignores_reddit_and_benchmark_only_posts(conn, watchlist):
    eng = make(conn, watchlist, rec := Recorder())
    eng.run(T_POST - timedelta(minutes=10))
    seed(conn, "r1", T_POST, ["NVDA"], platform="reddit", author="someone")
    seed(conn, "b1", T_POST, ["SPY"])
    eng.run(T_POST + timedelta(minutes=4))
    assert rec.sent == []
    assert statuses(conn, "heads_up") == {}


def test_late_posts_go_to_one_digest_with_day_after_followups_only(conn, watchlist):
    eng = make(conn, watchlist, rec := Recorder())
    eng.run(datetime(2026, 9, 29, 1, 0, tzinfo=UTC))  # initialise the evening before
    night = datetime(2026, 9, 30, 2, 0, tzinfo=UTC)  # Tue 22:00 ET
    for i, minutes in enumerate((0, 45, 90)):
        seed(conn, f"n{i}", night + timedelta(minutes=minutes), ["NVDA", "INTC"])
    eng.run(datetime(2026, 9, 30, 11, 0, tzinfo=UTC))  # Wed 07:00 ET
    assert [m.title for m in rec.sent] == ["While you were away: 3 stock posts"]
    assert set(statuses(conn, "heads_up").values()) == {"skipped"}
    assert statuses(conn, "follow_60m") == {}
    assert set(statuses(conn, "follow_d1")) == {"n0", "n1", "n2"}
    assert list(statuses(conn, "digest").values()) == ["sent"]


def test_failed_send_is_retried_then_abandoned_after_24h(conn, watchlist):
    rec = Recorder(failures=1000)
    eng = make(conn, watchlist, rec)
    eng.run(T_POST - timedelta(minutes=10))
    seed(conn, "p1", T_POST, ["NVDA"])
    for k in range(1, 4):
        eng.run(T_POST + timedelta(minutes=5 * k))
    row = conn.execute("SELECT * FROM alerts WHERE kind = 'heads_up'").fetchone()
    assert row["status"] == "failed" and row["attempts"] == 3 and "ntfy down" in row["error"]
    eng.run(T_POST + timedelta(hours=25))
    assert conn.execute("SELECT attempts FROM alerts WHERE kind = 'heads_up'").fetchone()[0] == 3
    rec.failures = 0
    eng.run(T_POST + timedelta(hours=26))
    assert rec.sent == []


def test_recovered_ntfy_sends_exactly_once(conn, watchlist):
    rec = Recorder(failures=1)
    eng = make(conn, watchlist, rec)
    eng.run(T_POST - timedelta(minutes=10))
    seed(conn, "p1", T_POST, ["NVDA"])
    eng.run(T_POST + timedelta(minutes=4))
    eng.run(T_POST + timedelta(minutes=9))
    eng.run(T_POST + timedelta(minutes=14))
    assert len(rec.sent) == 1


def test_heads_up_price_comes_from_live_bars(conn, watchlist):
    class Prices:
        def bars(self, symbol, start, end):
            s = int(T_POST.timestamp())
            return MinuteBars([s - 120, s - 60], [180.0, 182.41])

    eng = AlertEngine(conn, watchlist, rec := Recorder(), live_factory=Prices,
                      history_factory=lambda now: (lambda a: None), model=lambda t, d: None)
    eng.run(T_POST - timedelta(minutes=10))
    seed(conn, "p1", T_POST, ["NVDA"])
    eng.run(T_POST + timedelta(minutes=4))
    assert "NVDA $182.41" in rec.sent[0].body  # the bar starting 60 s before the post has finished by the post
```

- [ ] **Step 2: Run to confirm failure**

Run: `uv run --no-sync pytest tests/test_alert_engine.py -q` → FAIL (module missing).

- [ ] **Step 3: Implement the engine (heads-up and digest only)**

```python
# src/influence_tracker/alerts/engine.py
from __future__ import annotations

import logging
import sqlite3
from collections.abc import Callable
from datetime import UTC, date, datetime, time as dtime, timedelta

from .. import db, market
from ..analysis.metrics import EventModel, event_model
from ..config import Watchlist
from ..timeutil import NY, from_iso, to_iso
from . import messages, timing
from .live import PriceSource
from .messages import History, PostInfo
from .notify import Notifier, NotifyError

log = logging.getLogger(__name__)

ALERT_PLATFORMS = ("truthsocial", "x")
RETRY_FOR = timedelta(hours=24)
PRICE_LOOKBACK = timedelta(days=4)
INIT_SOURCE, INIT_KEY = "alerts", "initialized"
DIGEST_PLATFORM = "-"
EXISTED_NOTE = "existed before alerts were turned on"
LATE_NOTE = "late: sent in digest"


class StudyHistory:
    """author -> History from a fresh compute_study, computed on first use and reused for the cycle."""

    def __init__(self, conn: sqlite3.Connection, watchlist: Watchlist, now: datetime) -> None:
        self.conn, self.watchlist, self.now = conn, watchlist, now
        self._groups = None

    def __call__(self, author: str) -> History | None:
        if self._groups is None:
            from ..analysis.metrics import compute_study

            try:
                self._groups = compute_study(self.conn, self.watchlist, self.now).groups
            except Exception:
                log.exception("alerts: history unavailable (compute_study failed)")
                return None
        g = self._groups
        rows = g[(g["family"] == "author") & (g["group"] == author) & (g["subset"] == "main") & (g["window"] == "event")]
        if rows.empty:
            return None
        r = rows.iloc[0]
        p = r["p_holm"]
        return History(int(r["n_posts"]), float(r["mean_signed_car"]), None if p != p else float(p))


def _event_symbols(watchlist: Watchlist) -> list[str]:
    return [t.symbol for t in watchlist.event_tickers]


def post_tickers(conn: sqlite3.Connection, platform: str, native_id: str, symbols: list[str]) -> tuple[str, ...]:
    marks = ",".join("?" * len(symbols))
    rows = conn.execute(
        f"""SELECT DISTINCT ticker FROM mentions WHERE platform = ? AND native_id = ? AND ticker IN ({marks})
            ORDER BY ticker""",
        (platform, native_id, *symbols),
    ).fetchall()
    return tuple(r[0] for r in rows)


def post_info(conn: sqlite3.Connection, platform: str, native_id: str, tickers: tuple[str, ...]) -> PostInfo:
    r = conn.execute("SELECT * FROM posts WHERE platform = ? AND native_id = ?", (platform, native_id)).fetchone()
    return PostInfo(
        platform=platform,
        native_id=native_id,
        author=r["author"],
        created_at=from_iso(r["created_at_utc"]),
        text=r["text"],
        url=r["url"],
        stance=r["stance"],
        stance_conf=r["stance_conf"],
        tickers=tickers,
    )


def d1_due(d0: date) -> datetime:
    """20:00 New York on the session after d0: bars_1d for d0+1 exist after the nightly run."""
    return datetime.combine(market.session_offset(d0, 1), dtime(20, 0), tzinfo=NY).astimezone(UTC)


class AlertEngine:
    def __init__(
        self,
        conn: sqlite3.Connection,
        watchlist: Watchlist,
        notifier: Notifier,
        *,
        live_factory: Callable[[], PriceSource],
        history_factory: Callable[[datetime], Callable[[str], History | None]],
        model: Callable[[str, date], EventModel | None] | None = None,
    ) -> None:
        self.conn = conn
        self.watchlist = watchlist
        self.cfg = watchlist.alerts
        self.notifier = notifier
        self.live_factory = live_factory
        self.history_factory = history_factory
        self.model = model or (lambda ticker, d0: event_model(conn, ticker, d0))
        self.symbols = _event_symbols(watchlist)
        self.holdings = set(self.cfg.holdings)

    # ------------------------------------------------------------ detection

    def _new_posts(self) -> list[tuple[str, str]]:
        if not self.symbols:
            return []
        marks = ",".join("?" * len(self.symbols))
        plats = ",".join("?" * len(ALERT_PLATFORMS))
        rows = self.conn.execute(
            f"""SELECT p.platform, p.native_id FROM posts p
                WHERE p.platform IN ({plats})
                  AND EXISTS (SELECT 1 FROM mentions m WHERE m.platform = p.platform AND m.native_id = p.native_id
                              AND m.ticker IN ({marks}))
                  AND NOT EXISTS (SELECT 1 FROM alerts a WHERE a.kind = 'heads_up' AND a.platform = p.platform
                                  AND a.native_id = p.native_id)
                ORDER BY p.created_at_utc, p.platform, p.native_id""",
            (*ALERT_PLATFORMS, *self.symbols),
        ).fetchall()
        return [(r[0], r[1]) for r in rows]

    def initialize(self, now: datetime) -> int:
        if db.get_watermark(self.conn, INIT_SOURCE, INIT_KEY) is not None:
            return 0
        posts = self._new_posts()
        with self.conn:
            for platform, native_id in posts:
                created = from_iso(self.conn.execute(
                    "SELECT created_at_utc FROM posts WHERE platform = ? AND native_id = ?", (platform, native_id)
                ).fetchone()[0])
                db.add_alert(self.conn, "heads_up", platform, native_id, created, now, status="skipped",
                             error=EXISTED_NOTE)
            db.set_watermark(self.conn, INIT_SOURCE, INIT_KEY, to_iso(now), now)
        log.info("alerts: initialised; %d existing stock post(s) will not be alerted", len(posts))
        return len(posts)

    def queue(self, now: datetime) -> dict:
        counts = {"new": 0, "late": 0}
        late: list[PostInfo] = []
        with self.conn:
            for platform, native_id in self._new_posts():
                info = post_info(self.conn, platform, native_id, post_tickers(self.conn, platform, native_id, self.symbols))
                d0 = market.event_session(info.created_at)
                db.add_alert(self.conn, "follow_d1", platform, native_id, d1_due(d0), now)
                if timing.is_late(info.created_at, now, self.cfg):
                    db.add_alert(self.conn, "heads_up", platform, native_id, now, now, status="skipped", error=LATE_NOTE)
                    late.append(info)
                    counts["late"] += 1
                    continue
                window = timing.follow_window(info.created_at, d0, self.cfg.followup_minutes)
                db.add_alert(self.conn, "heads_up", platform, native_id, now, now)
                db.add_alert(self.conn, "follow_60m", platform, native_id, window.end + timing.FOLLOW_DELAY, now)
                counts["new"] += 1
            if late:
                msg = messages.digest(late)
                db.add_alert(self.conn, "digest", DIGEST_PLATFORM, to_iso(now), now, now,
                             title=msg.title, message=msg.body)
        return counts

    # ------------------------------------------------------------ sending

    def send_due(self, now: datetime) -> dict:
        counts = {"sent": 0, "failed": 0, "skipped": 0, "waiting": 0}
        live = self.live_factory()
        history = self.history_factory(now)
        for row in db.due_alerts(self.conn, now, RETRY_FOR):
            kind = row["kind"]
            if kind == "digest":
                self._deliver(row, messages.Message(row["title"], row["message"], priority=2,
                                                    tags=("mailbox_with_mail",)), now, counts)
            elif kind == "heads_up":
                self._deliver(row, self._heads_up(row, live, history), now, counts)
        return counts

    def _heads_up(self, row: sqlite3.Row, live: PriceSource, history) -> messages.Message:
        info = post_info(self.conn, row["platform"], row["native_id"],
                         post_tickers(self.conn, row["platform"], row["native_id"], self.symbols))
        prices = {}
        for t in messages.order_tickers(info.tickers, self.holdings)[: messages.PRICE_TICKERS]:
            prices[t] = live.bars(t, info.created_at - PRICE_LOOKBACK, info.created_at + timedelta(minutes=1)).at(
                info.created_at
            )
        return messages.heads_up(info, self.holdings, prices, history(info.author))

    def _deliver(self, row: sqlite3.Row, message: messages.Message, now: datetime, counts: dict) -> None:
        try:
            self.notifier.send(message)
        except NotifyError as e:
            with self.conn:
                db.mark_alert(self.conn, row["id"], "failed", now, title=message.title, message=message.body,
                              error=str(e))
            counts["failed"] += 1
            log.warning("alerts: %s for %s failed: %s", row["kind"], row["native_id"], e)
            return
        with self.conn:
            db.mark_alert(self.conn, row["id"], "sent", now, title=message.title, message=message.body)
        counts["sent"] += 1

    def run(self, now: datetime) -> dict:
        initialized = self.initialize(now)
        counts = {"status": "ok", "initialized": initialized, **self.queue(now), **self.send_due(now)}
        if counts["failed"]:
            counts["status"] = "partial"
        return counts
```


- [ ] **Step 4: Run the tests**

Run: `uv run --no-sync pytest tests/test_alert_engine.py -q` → PASS. Run ruff on the new files.

- [ ] **Step 5: Commit**

```bash
git add src/influence_tracker/alerts/engine.py tests/test_alert_engine.py
git commit -m "Add the alert engine: first-run marking, heads-up, digest, retries"
```

---

### Task 7: Follow-ups (60 minutes and day after)

**Files:**
- Create: `src/influence_tracker/alerts/followups.py`
- Modify: `src/influence_tracker/alerts/engine.py` (`send_due` handles `follow_60m` and `follow_d1`)
- Test: `tests/test_alert_followups.py`

**Interfaces:**
- Consumes: `timing.follow_window`, `timing.STALE_AFTER`, `live.PriceSource`, `metrics.EventModel`, `messages.Follow60Row/D1Row/follow_60m/follow_d1`, `events.earnings_session`, `engine.post_info/post_tickers`.
- Produces:
  - `followups.MAX_TICKERS = 5`, `followups.MAX_BAR_GAP = timedelta(minutes=15)`, `followups.D1_GIVE_UP = timedelta(days=7)`.
  - `followups.follow_60m_rows(tickers, window, live, model_for) -> list[Follow60Row] | None` (None = not ready).
  - `followups.window_text(window, d0) -> str`.
  - `followups.follow_d1_rows(conn, platform, native_id, tickers, d0, model_for) -> list[D1Row] | None` (None = not ready).
  - `followups.confounders(conn, platform, native_id, ticker, d0) -> tuple[str, ...]`.

Sending rules in the engine:

- `follow_60m`:
  - Pending and `now > due_at + STALE_AFTER` → mark `skipped` ("stale: PC off or prices unavailable").
  - Else rows = `follow_60m_rows(...)`; if None, leave pending (no attempt counted); else deliver.
- `follow_d1`:
  - `now > due_at + D1_GIVE_UP` → `skipped`.
  - Else rows = `follow_d1_rows(...)`; if None, leave pending; else deliver.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_alert_followups.py
from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

from influence_tracker import db
from influence_tracker.alerts import followups
from influence_tracker.alerts.engine import AlertEngine
from influence_tracker.alerts.timing import FollowWindow
from influence_tracker.analysis.metrics import EventModel
from influence_tracker.events import MinuteBars
from influence_tracker.models import Mention, Post

T0 = datetime(2026, 9, 29, 14, 31, tzinfo=UTC)  # Tue 10:31 ET
D0 = date(2026, 9, 29)
END = T0 + timedelta(minutes=60)


def minute_bars(start: datetime, end: datetime, first: float, last: float) -> MinuteBars:
    s, e = int(start.timestamp()), int(end.timestamp())
    starts = list(range(s, e, 60))
    step = (last - first) / max(1, len(starts) - 1)
    return MinuteBars(starts, [first + i * step for i in range(len(starts))])


class Prices:
    def __init__(self, series: dict[str, MinuteBars]) -> None:
        self.series = series

    def bars(self, symbol, start, end):
        return self.series.get(symbol, MinuteBars([], []))


def model(beta=1.5):
    return EventModel(alpha=0.0, beta=beta, sigma=0.02, n_est=120, ar={k: 0.0 for k in range(-5, 6)})


def test_follow_60m_rows_market_adjusted():
    window = FollowWindow(T0, END, False)
    live = Prices({
        "NVDA": minute_bars(T0 - timedelta(hours=1), END + timedelta(minutes=2), 100.0, 102.0),
        "SPY": minute_bars(T0 - timedelta(hours=1), END + timedelta(minutes=2), 500.0, 501.0),
    })
    rows = followups.follow_60m_rows(("NVDA",), window, live, lambda t: model(1.5))
    [r] = rows
    nvda, spy = live.bars("NVDA", 0, 0), live.bars("SPY", 0, 0)
    ret = nvda.at(END).price / nvda.at(T0).price - 1
    spy_ret = spy.at(END).price / spy.at(T0).price - 1
    assert r.ret == ret and r.spy_ret == spy_ret
    assert r.abnormal == ret - 1.5 * spy_ret


def test_follow_60m_waits_when_end_bar_is_stale():
    window = FollowWindow(T0, END, False)
    live = Prices({
        "NVDA": minute_bars(T0 - timedelta(hours=1), END - timedelta(minutes=20), 100.0, 101.0),
        "SPY": minute_bars(T0 - timedelta(hours=1), END + timedelta(minutes=2), 500.0, 501.0),
    })
    assert followups.follow_60m_rows(("NVDA",), window, live, lambda t: model()) is None


def test_follow_60m_without_model_or_spy():
    window = FollowWindow(T0, END, False)
    live = Prices({"NVDA": minute_bars(T0 - timedelta(hours=1), END + timedelta(minutes=2), 100.0, 102.0)})
    [r] = followups.follow_60m_rows(("NVDA",), window, live, lambda t: None)
    assert r.spy_ret is None and r.abnormal is None


def test_window_text():
    assert followups.window_text(FollowWindow(T0, END, False), D0) == "From the post (Tue Sep 29 10:31 ET) to 11:31 ET."
    trunc = FollowWindow(datetime(2026, 11, 27, 17, 40, tzinfo=UTC), datetime(2026, 11, 27, 18, 0, tzinfo=UTC), True)
    assert followups.window_text(trunc, date(2026, 11, 27)).endswith("to the close (13:00 ET, early close).")


def _seed_event(conn, native_id, ticker, created, clustered=0):
    p = Post(platform="truthsocial", native_id=native_id, author="realDonaldTrump", created_at_utc=created,
             text="x", url="https://example.com")
    with conn:
        db.upsert_post(conn, p, created)
        db.replace_mentions(conn, "truthsocial", native_id, [Mention(ticker, "cashtag", "$" + ticker)], [], created)
        conn.execute(
            """INSERT INTO events (platform, native_id, ticker, t0, d0, session_phase, status, clustered, created_at)
               VALUES ('truthsocial', ?, ?, ?, ?, 'regular', 'pending', ?, ?)""",
            (native_id, ticker, created.strftime("%Y-%m-%dT%H:%M:%SZ"), D0.isoformat(), clustered,
             created.strftime("%Y-%m-%dT%H:%M:%SZ")),
        )


def test_follow_d1_rows_ready_and_not_ready(conn):
    _seed_event(conn, "p1", "NVDA", T0)
    ready = EventModel(0.0, 1.0, 0.02, 120, {**{k: 0.0 for k in range(-5, 6)}, 0: 0.03, 1: 0.01})
    [r] = followups.follow_d1_rows(conn, "truthsocial", "p1", ("NVDA",), D0, lambda t: ready)
    assert abs(r.car - 0.04) < 1e-12 and abs(r.z - 0.04 / (0.02 * 2**0.5)) < 1e-12
    missing = EventModel(0.0, 1.0, 0.02, 120, {**{k: 0.0 for k in range(-5, 6)}, 1: float("nan")})
    assert followups.follow_d1_rows(conn, "truthsocial", "p1", ("NVDA",), D0, lambda t: missing) is None


def test_confounders(conn):
    _seed_event(conn, "p1", "NVDA", T0, clustered=1)
    with conn:
        conn.execute("INSERT INTO earnings (symbol, earnings_at) VALUES ('NVDA', '2026-09-29T20:05:00Z')")
        conn.execute("INSERT INTO earnings_fetch (symbol, fetched_at, ok) VALUES ('NVDA', '2026-09-28T00:00:00Z', 1)")
    notes = followups.confounders(conn, "truthsocial", "p1", "NVDA", D0)
    assert "earnings day — the move may be the report, not the post" in notes
    assert "another post about NVDA in the same session" in notes


def test_engine_sends_60m_followup_then_d1_and_skips_stale(conn, watchlist):
    rec = []

    class Rec:
        def send(self, m):
            rec.append(m)

    live = Prices({
        "NVDA": minute_bars(T0 - timedelta(days=1), END + timedelta(minutes=5), 100.0, 101.0),
        "SPY": minute_bars(T0 - timedelta(days=1), END + timedelta(minutes=5), 500.0, 500.5),
    })
    d1_model = EventModel(0.0, 1.0, 0.02, 120, {**{k: 0.0 for k in range(-5, 6)}, 0: 0.01, 1: 0.0})
    eng = AlertEngine(conn, watchlist, Rec(), live_factory=lambda: live,
                      history_factory=lambda now: (lambda a: None), model=lambda t, d: d1_model)
    eng.run(T0 - timedelta(minutes=10))
    _seed_event(conn, "p1", "NVDA", T0)
    eng.run(T0 + timedelta(minutes=4))
    eng.run(END + timedelta(minutes=4))
    eng.run(datetime(2026, 10, 1, 0, 30, tzinfo=UTC))
    titles = [m.title for m in rec]
    assert titles == [
        "NVDA · realDonaldTrump · stance unavailable",
        "NVDA · 60 min after realDonaldTrump's post",
        "NVDA · day-after check on realDonaldTrump's post",
    ]
    # a 60-minute follow-up that cannot be sent within 30 minutes of its due time is skipped, not sent late
    _seed_event(conn, "p2", "NVDA", END + timedelta(minutes=10))
    eng.run(END + timedelta(minutes=12))
    eng.run(END + timedelta(minutes=10) + timedelta(hours=3))
    rows = {r["kind"]: r["status"] for r in conn.execute("SELECT kind, status FROM alerts WHERE native_id = 'p2'")}
    assert rows["follow_60m"] == "skipped"
```

- [ ] **Step 2: Run to confirm failure**

Run: `uv run --no-sync pytest tests/test_alert_followups.py -q` → FAIL (module missing).

- [ ] **Step 3: Implement `followups.py`**

```python
# src/influence_tracker/alerts/followups.py
from __future__ import annotations

import math
import sqlite3
from collections.abc import Callable
from datetime import date, datetime, timedelta

from .. import market
from ..analysis.metrics import EventModel
from ..events import MARKET, earnings_session
from ..timeutil import NY, parse_api_time
from .live import PriceSource
from .messages import D1Row, Follow60Row, et
from .timing import FollowWindow

MAX_TICKERS = 5
MAX_BAR_GAP = timedelta(minutes=15)
D1_GIVE_UP = timedelta(days=7)
PRICE_LOOKBACK = timedelta(days=4)
SPLIT_RADIUS = 5
EARNINGS = "earnings day — the move may be the report, not the post"
SPLIT = "split nearby"


def _fresh(point, at: datetime) -> bool:
    """A price 'as of' `at` whose bar started more than MAX_BAR_GAP before it means Yahoo has not caught up yet."""
    return point is not None and at.timestamp() - 60 - point.ts <= MAX_BAR_GAP.total_seconds()


def follow_60m_rows(
    tickers: tuple[str, ...],
    window: FollowWindow,
    live: PriceSource,
    model_for: Callable[[str], EventModel | None],
) -> list[Follow60Row] | None:
    start, end = window.start - PRICE_LOOKBACK, window.end + timedelta(minutes=2)
    spy = live.bars(MARKET, start, end)
    m_start, m_end = spy.at(window.start), spy.at(window.end)
    spy_ok = m_start is not None and _fresh(m_end, window.end)
    rows: list[Follow60Row] = []
    for t in tickers[:MAX_TICKERS]:
        bars = live.bars(t, start, end)
        p_start, p_end = bars.at(window.start), bars.at(window.end)
        if p_start is None or not _fresh(p_end, window.end):
            return None
        ret = p_end.price / p_start.price - 1
        spy_ret = m_end.price / m_start.price - 1 if spy_ok else None
        model = model_for(t)
        abnormal = ret - model.beta * spy_ret if (model is not None and spy_ret is not None) else None
        rows.append(Follow60Row(t, ret, spy_ret, abnormal))
    return rows


def window_text(window: FollowWindow, d0: date) -> str:
    end_et = window.end.astimezone(NY).strftime("%H:%M ET")
    if window.truncated:
        close = market.session_bounds_utc(d0)[1].astimezone(NY)
        early = "" if close.hour == 16 and close.minute == 0 else ", early close"
        return f"From the post ({et(window.start)}) to the close ({close.strftime('%H:%M ET')}{early})."
    return f"From the post ({et(window.start)}) to {end_et}."


def confounders(conn: sqlite3.Connection, platform: str, native_id: str, ticker: str, d0: date) -> tuple[str, ...]:
    notes: list[str] = []
    fetch = conn.execute("SELECT ok FROM earnings_fetch WHERE symbol = ?", (ticker,)).fetchone()
    if fetch is not None and fetch["ok"]:
        lo, hi = market.previous_session(d0), market.session_offset(d0, 1)
        for (at,) in conn.execute("SELECT earnings_at FROM earnings WHERE symbol = ?", (ticker,)):
            if lo <= earnings_session(parse_api_time(at)) <= hi:
                notes.append(EARNINGS)
                break
    lo_s, hi_s = market.session_offset(d0, -SPLIT_RADIUS), market.session_offset(d0, 1)
    split = conn.execute(
        """SELECT 1 FROM bars_1d WHERE symbol = ? AND session_date BETWEEN ? AND ? AND split_ratio <> 0 LIMIT 1""",
        (ticker, lo_s.isoformat(), hi_s.isoformat()),
    ).fetchone()
    if split is not None:
        notes.append(SPLIT)
    ev = conn.execute(
        "SELECT clustered FROM events WHERE platform = ? AND native_id = ? AND ticker = ?", (platform, native_id, ticker)
    ).fetchone()
    if ev is not None and ev["clustered"]:
        notes.append(f"another post about {ticker} in the same session")
    return tuple(notes)


def follow_d1_rows(
    conn: sqlite3.Connection,
    platform: str,
    native_id: str,
    tickers: tuple[str, ...],
    d0: date,
    model_for: Callable[[str], EventModel | None],
) -> list[D1Row] | None:
    rows: list[D1Row] = []
    for t in tickers[:MAX_TICKERS]:
        model = model_for(t)
        if model is None:
            rows.append(D1Row(t, None, None, ()))
            continue
        car, z = model.car(0, 1)
        if not math.isfinite(car):
            return None
        rows.append(D1Row(t, car, z, confounders(conn, platform, native_id, t, d0)))
    return rows
```

- [ ] **Step 4: Wire the follow-ups into `AlertEngine.send_due`**

In `engine.py`, import `followups` and extend the loop in `send_due`:

```python
            elif kind == "follow_60m":
                self._follow_60m(row, live, now, counts)
            elif kind == "follow_d1":
                self._follow_d1(row, now, counts)
```

and add these methods to `AlertEngine`:

```python
    def _post(self, row: sqlite3.Row) -> tuple[PostInfo, date]:
        info = post_info(self.conn, row["platform"], row["native_id"],
                         post_tickers(self.conn, row["platform"], row["native_id"], self.symbols))
        return info, market.event_session(info.created_at)

    def _skip(self, row: sqlite3.Row, now: datetime, reason: str, counts: dict) -> None:
        with self.conn:
            db.mark_alert(self.conn, row["id"], "skipped", now, error=reason)
        counts["skipped"] += 1

    def _follow_60m(self, row: sqlite3.Row, live: PriceSource, now: datetime, counts: dict) -> None:
        if row["status"] == "pending" and now > from_iso(row["due_at"]) + timing.STALE_AFTER:
            self._skip(row, now, "stale: PC off or prices unavailable", counts)
            return
        info, d0 = self._post(row)
        ordered = tuple(messages.order_tickers(info.tickers, self.holdings))
        window = timing.follow_window(info.created_at, d0, self.cfg.followup_minutes)
        rows = followups.follow_60m_rows(ordered, window, live, lambda t: self.model(t, d0))
        if rows is None:
            counts["waiting"] += 1
            return
        msg = messages.follow_60m(info, self.holdings, rows, followups.window_text(window, d0))
        self._deliver(row, msg, now, counts)

    def _follow_d1(self, row: sqlite3.Row, now: datetime, counts: dict) -> None:
        if now > from_iso(row["due_at"]) + followups.D1_GIVE_UP:
            self._skip(row, now, "gave up: daily bars never arrived", counts)
            return
        info, d0 = self._post(row)
        ordered = tuple(messages.order_tickers(info.tickers, self.holdings))
        rows = followups.follow_d1_rows(
            self.conn, row["platform"], row["native_id"], ordered, d0, lambda t: self.model(t, d0)
        )
        if rows is None:
            counts["waiting"] += 1
            return
        self._deliver(row, messages.follow_d1(info, self.holdings, rows), now, counts)
```

`_seed_event` in the test inserts an `events` row but no stance, so the heads-up title reads "stance unavailable". That is intended.

- [ ] **Step 5: Run the tests**

Run: `uv run --no-sync pytest tests/test_alert_followups.py tests/test_alert_engine.py -q` → PASS. Then run the full suite.

- [ ] **Step 6: Commit**

```bash
git add src/influence_tracker/alerts/followups.py src/influence_tracker/alerts/engine.py tests/test_alert_followups.py
git commit -m "Add 60-minute and day-after follow-ups with confounder notes"
```

---

### Task 8: Watch loop and keep-awake

**Files:**
- Create: `src/influence_tracker/alerts/keepawake.py`, `src/influence_tracker/alerts/watch.py`
- Test: `tests/test_watch.py`

**Interfaces:**
- Consumes: `timing.is_active`, `timing.next_cycle`, `AlertEngine.run`, `db.set_watermark/get_watermark`.
- Produces:
  - `keepawake.ES_CONTINUOUS = 0x80000000`, `keepawake.ES_SYSTEM_REQUIRED = 0x00000001`.
  - `keepawake.KeepAwake(set_state: Callable[[int], int] | None = None)` with `.update(active: bool)` and `.release()`.
  - `watch.HEARTBEAT_FRESH = timedelta(minutes=15)`.
  - `watch.heartbeat_fresh(conn, now) -> bool`.
  - `watch.Steps` (dataclass of callables).
  - `watch.run_watch(conn, watchlist, steps, run_id, *, once=False, max_cycles=None) -> int`.

`Steps` fields:

- `clock: Callable[[], datetime]`
- `sleep: Callable[[float], None]`
- `collect_truthsocial: Callable[[int, datetime], dict]`
- `collect_x: Callable[[int, datetime], dict] | None`
- `classify: Callable[[int, datetime], dict]`
- `sync_events: Callable[[datetime], dict]`
- `alerts: Callable[[datetime], dict]`
- `keep_awake: KeepAwake`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_watch.py
from __future__ import annotations

from datetime import UTC, datetime, timedelta

from influence_tracker import db
from influence_tracker.alerts import watch
from influence_tracker.alerts.keepawake import ES_CONTINUOUS, ES_SYSTEM_REQUIRED, KeepAwake
from influence_tracker.timeutil import NY

ACTIVE = datetime(2026, 9, 29, 10, 31, tzinfo=NY)
IDLE = datetime(2026, 10, 3, 12, 0, tzinfo=NY)  # Saturday


class Clock:
    def __init__(self, t: datetime) -> None:
        self.t = t.astimezone(UTC)

    def __call__(self) -> datetime:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += timedelta(seconds=seconds)


def steps(clock, calls, *, x=False, fail_collect=False):
    def rec(name):
        def f(*args):
            calls.append((name, clock()))
            if fail_collect and name == "ts":
                raise RuntimeError("boom")
            return {"status": "ok"}
        return f

    states = []
    return watch.Steps(
        clock=clock, sleep=clock.sleep,
        collect_truthsocial=rec("ts"), collect_x=rec("x") if x else None,
        classify=rec("classify"), sync_events=rec("sync"), alerts=rec("alerts"),
        keep_awake=KeepAwake(set_state=lambda flags: states.append(flags) or 1),
    ), states


def test_cycle_order_heartbeat_and_cadence(conn, watchlist):
    clock, calls = Clock(ACTIVE), []
    s, states = steps(clock, calls)
    watch.run_watch(conn, watchlist, s, run_id=1, max_cycles=2)
    assert [n for n, _ in calls] == ["ts", "classify", "sync", "alerts"] * 2
    assert calls[4][1] == datetime(2026, 9, 29, 10, 35, tzinfo=NY)  # aligned 5-minute tick
    assert db.get_watermark(conn, "watch", "heartbeat") is not None
    assert watch.heartbeat_fresh(conn, clock()) is True
    assert watch.heartbeat_fresh(conn, clock() + timedelta(minutes=16)) is False
    assert states[0] == ES_CONTINUOUS | ES_SYSTEM_REQUIRED and states[-1] == ES_CONTINUOUS


def test_idle_does_not_hold_the_pc_awake(conn, watchlist):
    clock, calls = Clock(IDLE), []
    s, states = steps(clock, calls)
    watch.run_watch(conn, watchlist, s, run_id=1, once=True)
    assert states == []  # never set awake, so nothing to release


def test_a_failing_step_does_not_stop_the_cycle(conn, watchlist):
    clock, calls = Clock(ACTIVE), []
    s, _ = steps(clock, calls, fail_collect=True)
    assert watch.run_watch(conn, watchlist, s, run_id=1, once=True) == 0
    assert [n for n, _ in calls] == ["ts", "classify", "sync", "alerts"]


def test_x_is_polled_at_its_own_interval(conn, watchlist):
    clock, calls = Clock(ACTIVE), []
    s, _ = steps(clock, calls, x=True)
    watch.run_watch(conn, watchlist, s, run_id=1, max_cycles=4)  # 10:31, 10:35, 10:40, 10:45
    x_times = [t for n, t in calls if n == "x"]
    assert [t.astimezone(NY).strftime("%H:%M") for t in x_times] == ["10:31"]  # next X poll is due at 10:46
```

- [ ] **Step 2: Run to confirm failure**

Run: `uv run --no-sync pytest tests/test_watch.py -q` → FAIL (modules missing).

- [ ] **Step 3: Implement keep-awake**

```python
# src/influence_tracker/alerts/keepawake.py
from __future__ import annotations

import sys
from collections.abc import Callable

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


class KeepAwake:
    """Asks Windows not to sleep while `active` (SetThreadExecutionState). A closed laptop lid can still sleep the
    machine, depending on its power settings. A no-op off Windows."""

    def __init__(self, set_state: Callable[[int], int] | None = None) -> None:
        if set_state is None and sys.platform == "win32":
            import ctypes

            set_state = ctypes.windll.kernel32.SetThreadExecutionState
        self._set = set_state
        self._on = False

    def update(self, active: bool) -> None:
        if self._set is None or active == self._on:
            return
        self._set(ES_CONTINUOUS | ES_SYSTEM_REQUIRED if active else ES_CONTINUOUS)
        self._on = active

    def release(self) -> None:
        self.update(False)
```

- [ ] **Step 4: Implement the loop**

```python
# src/influence_tracker/alerts/watch.py
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


def _safe(name: str, fn: Callable[[], object]) -> None:
    try:
        result = fn()
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
            _safe("collect:truthsocial", lambda: steps.collect_truthsocial(run_id, now))
            if steps.collect_x is not None and (last_x is None or now - last_x >= timedelta(minutes=cfg.poll_x_minutes)):
                _safe("collect:x", lambda: steps.collect_x(run_id, now))
                last_x = now
            _safe("classify", lambda: steps.classify(run_id, now))
            _safe("sync_events", lambda: steps.sync_events(now))
            _safe("alerts", lambda: steps.alerts(now))
            cycles += 1
            if once or (max_cycles is not None and cycles >= max_cycles):
                return 0
            wake = timing.next_cycle(steps.clock(), cfg)
            steps.sleep(max(1.0, (wake - steps.clock()).total_seconds()))
    finally:
        steps.keep_awake.release()
```

- [ ] **Step 5: Run the tests**

Run: `uv run --no-sync pytest tests/test_watch.py -q` → PASS (ruff clean).

- [ ] **Step 6: Commit**

```bash
git add src/influence_tracker/alerts/keepawake.py src/influence_tracker/alerts/watch.py tests/test_watch.py
git commit -m "Add the watch loop with heartbeat, cadence and keep-awake"
```

---

### Task 9: CLI — `watch`, `alerts setup|test`, and the nightly run stepping aside

**Files:**
- Modify: `src/influence_tracker/cli.py`
- Test: `tests/test_cli_alerts.py`

**Interfaces:**
- Consumes: `watch.run_watch`, `watch.Steps`, `watch.heartbeat_fresh`, `engine.AlertEngine`, `engine.StudyHistory`, `live.LivePrices`, `notify.NtfyNotifier`, `notify.DryRunNotifier`, `notify.Message`, `keepawake.KeepAwake`, `sentiment.load_classifier`, `sentiment.classify_posts`, `events.sync_events`, the collectors.
- Produces: CLI commands `influence watch [--once] [--dry-run]`, `influence alerts setup`, `influence alerts test`. `Pipeline.collect` skips `truthsocial` and `x` (as `skipped`, reason `live watch is running`) while `heartbeat_fresh` is true.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_cli_alerts.py
from __future__ import annotations

import shutil
from datetime import UTC, datetime

import pytest

from influence_tracker import cli, db
from influence_tracker.alerts import watch as watch_mod
from influence_tracker.config import REPO_ROOT


@pytest.fixture
def root(tmp_path, monkeypatch):
    (tmp_path / "config").mkdir()
    shutil.copy(REPO_ROOT / "config" / "watchlist.yaml", tmp_path / "config" / "watchlist.yaml")
    monkeypatch.setenv(cli.ROOT_ENV_VAR, str(tmp_path))
    monkeypatch.delenv("NTFY_TOPIC", raising=False)
    return tmp_path


def test_alerts_setup_writes_a_topic_once(root, capsys):
    assert cli.main(["alerts", "setup"]) == 0
    env = (root / ".env").read_text(encoding="utf-8")
    [line] = [x for x in env.splitlines() if x.startswith("NTFY_TOPIC=")]
    topic = line.split("=", 1)[1]
    assert topic.startswith("influence-") and 20 <= len(topic) <= 64
    assert cli.main(["alerts", "setup"]) == 0
    assert (root / ".env").read_text(encoding="utf-8") == env  # never overwritten
    assert "subscribe" in capsys.readouterr().out.lower()


def test_alerts_test_needs_a_topic(root, capsys):
    assert cli.main(["alerts", "test"]) == 2
    assert "influence alerts setup" in capsys.readouterr().err


def test_alerts_test_sends_one_message(root, monkeypatch):
    (root / ".env").write_text("NTFY_TOPIC=influence-test123\n", encoding="utf-8")
    sent = []
    monkeypatch.setattr(cli, "_make_notifier", lambda settings, dry_run: type("N", (), {"send": sent.append})())
    assert cli.main(["alerts", "test"]) == 0
    assert len(sent) == 1 and "test" in sent[0].title.lower()


def test_watch_once_dry_run_wires_every_step(root, monkeypatch):
    called = []
    monkeypatch.setattr(cli, "_watch_steps", lambda *a, **k: called.append(k) or "steps")
    monkeypatch.setattr(watch_mod, "run_watch", lambda conn, wl, steps, run_id, once, max_cycles=None: 0)
    assert cli.main(["watch", "--once", "--dry-run"]) == 0
    assert called and called[0]["dry_run"] is True


def test_daily_skips_truthsocial_while_watch_is_alive(root, monkeypatch):
    conn = db.connect(root / "data" / "tracker.db")
    now = datetime.now(UTC)
    with conn:
        db.set_watermark(conn, "watch", "heartbeat", now.strftime("%Y-%m-%dT%H:%M:%SZ"), now)
    conn.close()
    ran = []
    monkeypatch.setattr(
        cli.Pipeline, "_collector",
        lambda self, n, sink, err: (lambda run_id, now: ran.append(n) or {"status": "ok"}),
    )
    cli.main(["collect"])
    conn = db.connect(root / "data" / "tracker.db")
    row = conn.execute("SELECT status, error FROM runs WHERE stage = 'collect:truthsocial'").fetchone()
    assert (row["status"], row["error"]) == ("skipped", "live watch is running")
    assert "truthsocial" not in ran and "reddit" in ran
```

- [ ] **Step 2: Run to confirm failure**

Run: `uv run --no-sync pytest tests/test_cli_alerts.py -q` → FAIL (unknown commands / missing helpers).

- [ ] **Step 3: Implement the CLI changes**

In `build_parser()` add:

```python
    watch = sub.add_parser("watch", help="live alerts: check for new stock posts every few minutes and notify phones")
    watch.add_argument("--once", action="store_true", help="run a single cycle and exit")
    watch.add_argument("--dry-run", action="store_true", help="print alerts instead of sending them")
    alerts = sub.add_parser("alerts", help="phone alert setup: generate an ntfy topic or send a test notification")
    alerts.add_argument("action", choices=["setup", "test"])
```

In `Pipeline.collect`, inside the loop, before the `x` token check:

```python
            if name in ("truthsocial", "x") and _watch_alive(self.conn, self.clock()):
                self._skip(f"collect:{name}", WATCH_ALIVE)
                continue
```

with module-level:

```python
WATCH_ALIVE = "live watch is running"
TOPIC_PREFIX = "influence-"


def _watch_alive(conn: sqlite3.Connection, now: datetime) -> bool:
    from .alerts.watch import heartbeat_fresh

    return heartbeat_fresh(conn, now)


def _make_notifier(settings: Settings, dry_run: bool):
    from .alerts.notify import DryRunNotifier, NtfyNotifier

    if dry_run:
        return DryRunNotifier()
    if not settings.ntfy_topic:
        raise SystemExit("NTFY_TOPIC is not set: run `influence alerts setup` first")
    return NtfyNotifier(settings.ntfy_server, settings.ntfy_topic)
```

Add `_alerts(args, settings)` and `_watch(args, settings, watchlist, conn, clock)`, and dispatch to them from `_dispatch` before the Pipeline branch:

```python
    if args.command == "alerts":
        return _alerts(args, settings)
    if args.command == "watch":
        return _watch(args, settings, watchlist, conn, clock)
```

```python
def _alerts(args: argparse.Namespace, settings: Settings) -> int:
    if args.action == "setup":
        env = settings.root / ".env"
        if settings.ntfy_topic:
            topic = settings.ntfy_topic
            print(f"Already set up: NTFY_TOPIC={topic} in {env}")
        else:
            import secrets

            topic = TOPIC_PREFIX + secrets.token_urlsafe(18)
            existing = env.read_text(encoding="utf-8") if env.exists() else ""
            lines = [ln for ln in existing.splitlines() if not ln.startswith("NTFY_TOPIC=")]
            lines.append(f"NTFY_TOPIC={topic}")
            env.write_text("\n".join(lines) + "\n", encoding="utf-8")
            print(f"Created NTFY_TOPIC in {env}")
        print(
            "\nOn each phone: install the free 'ntfy' app (App Store / Google Play), tap +, and subscribe to topic\n"
            f"    {topic}\n"
            "on server ntfy.sh. The topic works like a password: share it only with your team.\n"
            "Then run: influence alerts test"
        )
        return 0
    if not settings.ntfy_topic:
        print("NTFY_TOPIC is not set: run `influence alerts setup` first", file=sys.stderr)
        return 2
    from .alerts.notify import Message

    _make_notifier(settings, dry_run=False).send(
        Message("influence-tracker test alert", "If you can read this, phone alerts work.", 3, ("white_check_mark",))
    )
    print("Test alert sent.")
    return 0


def _watch_steps(settings: Settings, watchlist: Watchlist, conn: sqlite3.Connection, clock: Clock, *, dry_run: bool):
    import time

    from . import events, sentiment
    from .alerts.engine import AlertEngine, StudyHistory
    from .alerts.keepawake import KeepAwake
    from .alerts.live import LivePrices
    from .alerts.watch import Steps
    from .collectors.truthsocial import collect_truthsocial

    pipeline = Pipeline(settings, watchlist, conn, clock)
    sink, setup_error = pipeline._prepare_sink()
    loaded: dict = {}

    def classifier(texts):
        if "clf" not in loaded:
            loaded["clf"] = sentiment.load_classifier(watchlist.sentiment.model_id, watchlist.sentiment.batch_size)
        return loaded["clf"](texts)

    collect_x = None
    if settings.x_bearer_token:
        from .collectors.x import collect_x as _cx

        def collect_x(run_id, now):
            return _cx(conn, watchlist, settings.x_bearer_token, _require(sink, setup_error), run_id, now)

    engine = AlertEngine(
        conn, watchlist, _make_notifier(settings, dry_run),
        live_factory=lambda: LivePrices(watchlist),
        history_factory=lambda now: StudyHistory(conn, watchlist, now),
    )
    return Steps(
        clock=clock,
        sleep=time.sleep,
        collect_truthsocial=lambda run_id, now: collect_truthsocial(conn, watchlist, _require(sink, setup_error), run_id, now),
        collect_x=collect_x,
        classify=lambda run_id, now: sentiment.classify_posts(conn, watchlist, run_id, now, classifier=classifier),
        sync_events=lambda now: events.sync_events(conn, watchlist, now),
        alerts=engine.run,
        keep_awake=KeepAwake(),
    )


def _watch(args, settings: Settings, watchlist: Watchlist, conn: sqlite3.Connection, clock: Clock) -> int:
    from .alerts import watch

    if not watchlist.alerts.enabled:
        print("alerts.enabled is false in config/watchlist.yaml", file=sys.stderr)
        return 2
    run_id = db.start_run(conn, "watch", clock())
    status, error = "ok", None
    try:
        steps = _watch_steps(settings, watchlist, conn, clock, dry_run=args.dry_run)
        return watch.run_watch(conn, watchlist, steps, run_id, once=args.once)
    except KeyboardInterrupt:
        return 0
    except Exception as e:
        status, error = "error", f"{type(e).__name__}: {e}"
        raise
    finally:
        db.finish_run(conn, run_id, status, {}, error, clock())
```

- [ ] **Step 4: Run the tests**

Run: `uv run --no-sync pytest tests/test_cli_alerts.py tests/test_cli.py -q` → PASS. Then run the full suite.

- [ ] **Step 5: Commit**

```bash
git add src/influence_tracker/cli.py tests/test_cli_alerts.py
git commit -m "Add watch and alerts commands; nightly run steps aside while watch is alive"
```

---

### Task 10: Alerts section in `influence status`

**Files:**
- Modify: `src/influence_tracker/status.py`
- Test: `tests/test_status_alerts.py`

**Interfaces:**
- Consumes: `alerts` table, watermark (`watch`, `heartbeat`), `alerts.timing.is_active`.
- Produces: `status._alerts(console, conn, now)`, called from `show_status` after `_events`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_status_alerts.py
from __future__ import annotations

from datetime import UTC, datetime, timedelta

from rich.console import Console

from influence_tracker import db
from influence_tracker.status import show_status

NOW = datetime(2026, 9, 29, 15, 0, tzinfo=UTC)  # Tue 11:00 ET, active window


def render(conn, watchlist) -> str:
    console = Console(record=True, width=160)
    show_status(conn, watchlist, NOW, console=console)
    return console.export_text()


def test_status_without_watch(conn, watchlist):
    out = render(conn, watchlist)
    assert "Alerts" in out and "watch has never run" in out


def test_status_with_alerts(conn, watchlist):
    with conn:
        db.set_watermark(conn, "watch", "heartbeat", "2026-09-29T14:40:00Z", NOW)
        db.add_alert(conn, "heads_up", "truthsocial", "1", NOW - timedelta(days=1), NOW, status="sent")
        db.add_alert(conn, "follow_60m", "truthsocial", "1", NOW + timedelta(minutes=5), NOW)
        db.add_alert(conn, "digest", "-", "2026-09-28T11:00:00Z", NOW - timedelta(days=1), NOW, status="failed",
                     error="ntfy down")
    out = render(conn, watchlist)
    assert "last check-in" in out and "20m ago" in out and "STALE" in out
    assert "heads_up" in out and "pending follow-ups: 1" in out and "failed sends: 1" in out
```

- [ ] **Step 2: Run to confirm failure**

Run: `uv run --no-sync pytest tests/test_status_alerts.py -q` → FAIL ("Alerts" missing).

- [ ] **Step 3: Implement**

Add to `status.py` (use the existing helpers `_heading`, `_table`, `_row`, `_say`, `_when`, `from_iso`, `Text`):

```python
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
```

Call `_alerts(console, conn, now)` in `show_status` right after `_events(console, conn, now)`. Add `from datetime import timedelta` and `from .timeutil import to_iso` if they are not already imported.

- [ ] **Step 4: Run the tests**

Run: `uv run --no-sync pytest tests/test_status_alerts.py tests/test_cli.py -q` → PASS.

- [ ] **Step 5: Commit**

```bash
git add src/influence_tracker/status.py tests/test_status_alerts.py
git commit -m "Show watch health and alert counts in influence status"
```

---

### Task 11: Watch scheduled task, docs, and the live check

**Files:**
- Create: `scripts/register_watch_task.ps1`, `scripts/run_watch.cmd`
- Modify: `README.md` (a "Live alerts" section), `docs/PLAN.md` ("As built — Phase 2 live alerts")
- Test: `tests/test_watch_task_script.py` (PowerShell parser check, mirroring the existing register_task test in `tests/test_cli.py`)

**Interfaces:**
- Consumes: nothing from `scripts/register_task.ps1`. Do NOT dot-source it: its `param()` block would overwrite this script's `$TaskName`/`$Unregister` with the daily task's defaults, and `-Unregister` could then delete the nightly task.
- Produces: scheduled task "InfluenceTracker Watch".

- [ ] **Step 1: Write the failing test**

```python
# tests/test_watch_task_script.py
from __future__ import annotations

import shutil
import subprocess

import pytest

from influence_tracker.config import REPO_ROOT

SCRIPT = REPO_ROOT / "scripts" / "register_watch_task.ps1"


@pytest.mark.skipif(shutil.which("powershell") is None, reason="needs Windows PowerShell")
def test_watch_task_script_parses():
    cmd = (
        "$errs = $null; $null = [System.Management.Automation.Language.Parser]::ParseFile("
        f"'{SCRIPT}', [ref]$null, [ref]$errs); $errs.Count"
    )
    out = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "0"


def test_watch_runner_calls_watch():
    text = (REPO_ROOT / "scripts" / "run_watch.cmd").read_text(encoding="utf-8")
    assert 'influence.exe" watch' in text and "logs\\watch.log" in text
```

- [ ] **Step 2: Run to confirm failure**

Run: `uv run --no-sync pytest tests/test_watch_task_script.py -q` → FAIL (files missing).

- [ ] **Step 3: Write the scripts**

`scripts/run_watch.cmd`:

```bat
@echo off
rem Runs `influence watch` for Windows Task Scheduler. Output is appended to logs\watch.log.
setlocal
cd /d "%~dp0.."
set PYTHONUTF8=1
if not exist logs mkdir logs
>>logs\watch.log echo ===== %DATE% %TIME% influence watch =====
".venv\Scripts\influence.exe" watch >>logs\watch.log 2>&1
set RC=%ERRORLEVEL%
>>logs\watch.log echo ===== %DATE% %TIME% exit %RC% =====
exit /b %RC%
```

`scripts/register_watch_task.ps1`:

```powershell
<#
.SYNOPSIS
Creates (or updates) the scheduled task that keeps `influence watch` running for live phone alerts, or removes it.

.DESCRIPTION
Starts at logon and every day at 03:50 local time, restarts within a minute if it stops, and never runs two copies.
The watch keeps the PC awake from 04:00 to 20:00 New York time on trading days; a closed laptop lid can still
sleep the machine depending on Windows power settings.

.EXAMPLE
powershell -ExecutionPolicy Bypass -File scripts\register_watch_task.ps1
powershell -ExecutionPolicy Bypass -File scripts\register_watch_task.ps1 -Unregister
#>
[CmdletBinding()]
param(
    [string]$TaskName = "InfluenceTracker Watch",
    [switch]$Unregister
)

$ErrorActionPreference = "Stop"

$repo = Split-Path -Parent $PSScriptRoot
$runner = Join-Path $PSScriptRoot "run_watch.cmd"

if ($Unregister) {
    if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
        Write-Host "Removed scheduled task '$TaskName'."
    } else {
        Write-Host "No scheduled task named '$TaskName' exists; nothing to remove."
    }
    return
}

if (-not (Test-Path $runner)) { throw "Cannot find $runner." }
$user = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$argument = '/c ""{0}""' -f $runner
$action = New-ScheduledTaskAction -Execute "cmd.exe" -Argument $argument -WorkingDirectory $repo
$at = [datetime]::ParseExact("03:50", [string[]]@("HH:mm"), [Globalization.CultureInfo]::InvariantCulture,
    [Globalization.DateTimeStyles]::None)
$daily = New-ScheduledTaskTrigger -Daily -At $at
# New-ScheduledTaskTrigger pins a UTC offset; without one, Task Scheduler follows local daylight saving.
$daily.StartBoundary = $at.ToString("yyyy-MM-dd'T'HH:mm:ss", [Globalization.CultureInfo]::InvariantCulture)
$triggers = @((New-ScheduledTaskTrigger -AtLogOn -User $user), $daily)
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -DontStopOnIdleEnd -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew `
    -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1)
$principal = New-ScheduledTaskPrincipal -UserId $user -LogonType Interactive -RunLevel Limited
$description = "influence-tracker live alerts: checks for stock posts every few minutes and notifies phones. " +
    "Output: $repo\logs\watch.log"

$null = Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $triggers -Settings $settings `
    -Principal $principal -Description $description -Force

Write-Host "Registered scheduled task '$TaskName' (at logon + daily 03:50, restarts on failure)."
Write-Host "Start it now:   Start-ScheduledTask -TaskName `"$TaskName`""
Write-Host "Remove it:      powershell -ExecutionPolicy Bypass -File `"$PSCommandPath`" -Unregister"
```


- [ ] **Step 4: Docs**

Add to `README.md` after the Commands section:

```markdown
## Live alerts (phone push via ntfy)

    uv run influence alerts setup     # makes a private ntfy topic in .env; subscribe to it in the ntfy app
    uv run influence alerts test      # sends a test notification
    uv run influence watch --dry-run  # prints alerts instead of sending them (Ctrl+C to stop)
    powershell -ExecutionPolicy Bypass -File scripts\register_watch_task.ps1   # keep it running

The watch checks Truth Social (and X, with a token) every 5 minutes from 4 AM to 8 PM ET on trading days and every
30 minutes otherwise. You get:

- a heads-up per post;
- a follow-up about 60 minutes after the post;
- a follow-up after the next day's close;
- one "while you were away" summary for posts found late.

Truth Social can only be checked from this PC (it blocks cloud servers), so alerts stop while the PC is off.
```

Append to `docs/PLAN.md` an "As built — Phase 2 live alerts" section listing:

- the JSON-publish refinement;
- the X step-aside;
- the first-run marking;
- the stale-bar guard;
- the test count.

- [ ] **Step 5: Run the whole suite and lint**

Run: `uv run --no-sync pytest -q` → all pass. Run `uv run --no-sync ruff check src tests` and `uv run --no-sync ruff format --check src tests` → clean.

- [ ] **Step 6: Live checks (controller, before merging)**

1. On a COPY of the live database (`INFLUENCE_TRACKER_ROOT` pointing at a temp root with `config/` and `data/`), run `influence watch --once --dry-run`. The log must show `initialized` equal to the number of existing stock posts and **zero** "would notify" lines.
2. In that copy, insert a fake fresh Truth Social post mentioning NVDA with `created_at` = now and run `watch --once --dry-run` again. Exactly one heads-up must be logged, with the expected title and body.
3. With Bryce's topic set, run `influence alerts test` and confirm the notification arrives on a phone.

- [ ] **Step 7: Commit**

```bash
git add scripts/register_watch_task.ps1 scripts/run_watch.cmd README.md docs/PLAN.md tests/test_watch_task_script.py
git commit -m "Add the watch scheduled task, live-alert docs and script checks"
```

---

## Self-review notes (for the executor)

- Before implementing, check every name in the "Consumes" lines against the real code: `market.is_session`, `events.earnings_session`, `prices.fetch_yahoo_1m`, `sentiment.classify_posts(..., classifier=)`, `Pipeline._prepare_sink`, `_require`. If one differs, follow the code and note the deviation in the commit message.
