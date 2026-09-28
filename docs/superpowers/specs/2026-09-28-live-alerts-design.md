# Live alerts (Phase 2) — design

Status: approved in conversation 2026-09-28; this document is for Bryce's review before an implementation plan.

## Purpose

Two uses, both wanted:

1. **Heads-up for the team's trades** — know within minutes when a tracked account posts about a stock,
   especially one the team holds in the Wharton simulator, and have something to cite in trading notes.
2. **Live examples for the study** — each post followed by a measured reaction, building fresh case studies.

Context: the M3 study found no reliable directional effect (117 signed posts, mean signed CAR[0,+1] −0.28%,
p = 0.38) and bigger moves *before* bullish posts than after. The alerts must therefore inform, not hype:
every follow-up says plainly whether a move was within the normal range.

### Success criteria

- A heads-up on the phones within about 6 minutes of a post while the PC is on.
- A 60-minute follow-up and a day-after follow-up with the market-adjusted move.
- No duplicate notifications, even across restarts and crashes.
- Posts found late (PC off) arrive as one summary, not a burst of stale "live" alerts.
- $0/month.

## Decisions and facts this design rests on

- **Delivery: ntfy phone push** (chosen by Bryce). Verified against docs.ntfy.sh/publish (2026-09-28):
  - Publish with `POST https://ntfy.sh/<topic>`, with the message as the body.
  - Headers: `Title`; `Priority` (1–5, where 3 is the default); `Tags` (comma-separated, emoji short codes);
    `Click` (URL opened on tap); `Actions` (`view, <label>, <url>`).
  - The topic works as a password (letters, digits, `_`, `-`; up to 64 characters).
  - A body over 4,096 bytes becomes an attachment, so messages stay short.
- **Runs on Bryce's PC only** (option A, chosen after a cloud test). On 2026-09-28 a throwaway GitHub Actions job,
  since deleted, showed:
  - Truth Social returns **403 from GitHub's servers**: a Cloudflare block page for curl and httpx, and a
    "Just a moment…" challenge even with curl_cffi browser impersonation.
  - Yahoo 1-minute data and ntfy work from the cloud.

  Truth Social is where the most important accounts post, so it must be polled from the home PC.
- **Volume is low:** 201 Truth Social stock posts in 20 months, about 2–3 alerts a week. No filtering by ticker
  is needed; holdings are highlighted instead of filtered.
- **Reddit is excluded from alerts.** Its posts are crowd chatter, and its RSS is a top-of-day list, not a live feed.
  It stays in the nightly study.
- **X is included automatically once `X_BEARER_TOKEN` exists** (polled less often; existing budget pacing
  applies). Until then it is skipped, as it is in the nightly run.

## Architecture

### `influence watch` — one long-running process

A new CLI command that loops until stopped. It is long-running, not one scheduled run every 5 minutes, because:

- loading the sentiment model (torch + FinTwitBERT) costs 10–90 s, so it must stay in memory;
- keeping the PC awake needs a live process;
- pacing, heartbeat and the in-memory classifier live in one place.

**Cycle cadence** (config):

- **Active window** — XNYS session days, 04:00–20:00 New York time. Every `poll_active_minutes` (default 5).
- **Otherwise** (nights, weekends, holidays) — every `poll_idle_minutes` (default 30). Posts made then still move
  the next session.
- **X**, when a token is set — at most every `poll_x_minutes` (default 15).

**One cycle:**

1. Write a heartbeat: watermark (`watch`, `heartbeat`) = now.
2. Collect.
   - Run the existing Truth Social collector: incremental max-id walk down to each account's watermark, with the
     existing ≥5 s spacing, 429/1015 back-off and resume. All posts are stored through the existing `PostSink`,
     so mentions are tagged exactly as in the study.
   - When a token is set and X is due, run the X collector the same way.
3. For every **new** post, meaning no `heads_up` or `digest` row yet in `alerts`, that mentions ≥1 non-benchmark
   watchlist ticker:
   - label its stance with the in-memory classifier (the same code as `classify`);
   - create its events with `events.sync_events`.
4. Queue notifications (see "Alert kinds") in the `alerts` table, then send everything due.
5. Sleep until the next cycle, aligned to the cadence and the wall clock.

**Keep-awake.**

- During the active window, call `SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)`; outside it, call
  `ES_CONTINUOUS` alone so the PC may sleep.
- This is Windows-only via ctypes and a no-op elsewhere.
- Limitation to state in the setup notes: closing a laptop lid can still sleep the machine, depending on Windows
  power settings.

**Coordination with the nightly `daily` run.**

- `daily`'s `collect:truthsocial` stage records `skipped` ("live watch is running") when the watch heartbeat is
  less than 15 minutes old, so the two never hit Truth Social at the same time.
- `daily`'s classify and enrich stages run as today. With both models loaded, memory is about 2 GB of 15.5 GB.

**Scheduling.** `scripts/register_watch_task.ps1` registers a second task, "InfluenceTracker Watch".

- Triggers: at logon, and daily at 03:50.
- Action: `scripts/run_watch.cmd` → `.venv\Scripts\influence.exe watch`, appending output to
  `logs\watch.log`.
- Settings: restart on failure (every 1 minute, many times); `MultipleInstances IgnoreNew`; no execution time
  limit; runs on battery.
- A `-Unregister` switch removes the task, like the existing script.

### Alert kinds

All times shown are New York time. Every message stays under 4 KB.

Follow-ups are queued only for posts that got a `heads_up` (both follow-ups) or appeared in a `digest` (day-after
only). Posts already stored before `watch` first runs get no follow-ups (see the ledger).

**1. `heads_up`** — one per post, not one per ticker, sent in the cycle that finds the post.

- **When:** only if the post is at most `late_after_minutes` (default 30) old when found.
- **Title:** tickers with holdings first, then `· <author> · <stance> (<conf>)`, e.g.
  `⭐ NVDA, INTC +3 · realDonaldTrump · bullish (0.91)`. `⭐` marks any ticker in `alerts.holdings`.
- **Priority:** 4 if a holding is mentioned, else 3.
- **Tags:** bullish → `chart_with_upwards_trend`, bearish → `chart_with_downwards_trend`, neutral →
  `speech_balloon`, no stance → `grey_question`.
- **Click:** the post URL.
- **Body:**
  - The first ~220 characters of the post text.
  - `Posted 10:31 ET` plus the price at the post for up to 3 tickers: `NVDA $182.41`. This uses the M2 no-look-ahead
    rule on a live 1-minute fetch; outside trading hours, "last $X at <time>".
  - One history line from the latest study numbers for that author, e.g. `History: realDonaldTrump's posts moved
    their stocks −0.60% on average over 2 days (n=71, p=0.17, not significant)`. For fewer than 10 past posts:
    `History: fewer than 10 past posts`.
    - Source: the author's row (family `author`, subset `main`, window `event`) from a fresh
      `metrics.compute_study` run. It runs at most once per cycle, and only when a heads-up is being built (about
      5 s).

**2. `follow_60m`** — one per post, reporting each ticker (up to 5; then "+N more, see `influence events`").

- **Window:** from the reference price at the post (`price_at(t0)`) to
  `price_at(min(max(t0, open(d0)) + followup_minutes, close(d0)))`.
  - For a regular-session post, that is the next hour.
  - For a pre-market, after-hours or overnight post, it runs from the post through the first hour of trading on d0,
    including the opening gap.
- **Due:** the window end plus 3 minutes, so the last bars exist.
- **Line per ticker:** `NVDA +0.84% vs SPY +0.10% → abnormal +0.69%` (β from the market model).
  - No z-score: the live data can't give a same-clock σ for the current session.
  - A truncated window says "to the close (13:00 early close)".
- **Skipped** (row status `skipped`) when its due time passed more than 30 minutes before it could be sent, e.g.
  because the PC was off.

**3. `follow_d1`** — one per post.

- **Due:** once `bars_1d` holds session d0+1 for the tickers and SPY. That is normally right after the nightly run.
- **Content:** the M3 measure, reusing the metrics code for single events: `NVDA CAR[0,+1] +1.20% (z 0.91) —
  within the normal range`, or `unusually large (|z| = 2.40)` when |z| ≥ 1.96.
- **Confounders** are named inline:
  - `earnings day — the move may be the report, not the post`;
  - `split nearby`;
  - `another post about NVDA in the same session`.
- **Sent late:** it is still sent when the PC was off, because it is not time-sensitive.

**4. `digest`** — at most one per cycle.

- **When:** posts found more than `late_after_minutes` after they were made. These get no heads-up and no 60-minute
  follow-up; each still gets its day-after follow-up.
- **Title:** `While you were away: 3 stock posts`.
- **Body:** one line per post (up to 8): `Sep 29 14:05 · realDonaldTrump · INTC, NVDA · bullish`, then
  `+N more — influence events`.
- **Priority:** 2 (low).

### Alert ledger (new table, created with `CREATE TABLE IF NOT EXISTS`)

```sql
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY,
    kind TEXT NOT NULL,            -- heads_up | follow_60m | follow_d1 | digest
    platform TEXT NOT NULL,        -- the post's platform; 'digest' rows use '-'
    native_id TEXT NOT NULL,       -- the post's id; digest rows use the cycle's UTC ISO time
    due_at TEXT NOT NULL,          -- UTC ISO
    status TEXT NOT NULL,          -- pending | sent | failed | skipped
    attempts INTEGER NOT NULL DEFAULT 0,
    title TEXT,
    message TEXT,
    error TEXT,
    created_at TEXT NOT NULL,
    sent_at TEXT,
    UNIQUE (kind, platform, native_id)
);
```

- The `UNIQUE` key is the no-duplicates guarantee.
- A row is written as `pending` before sending and marked `sent` only after ntfy returns 2xx.
- A crash between the two sends at most one duplicate, which is accepted; nothing is ever lost.
- Posts that already exist when `watch` first starts are marked with a `skipped` heads-up row, so turning alerts
  on doesn't notify 15,000 old posts.

### Notifier

- `notify.py`: `send(title, message, priority, tags, click) -> None`. It POSTs to
  `{NTFY_SERVER}/{NTFY_TOPIC}` with httpx, a 10 s timeout, and one retry after 5 s.
- A failure marks the row `failed` with the error. `failed` rows are retried on later cycles for up to 24 hours
  (then left `failed` and shown in `status`).
- `--dry-run` prints to the console and log instead of sending.

### Config and secrets

In `config/watchlist.yaml`:

```yaml
alerts:
  enabled: true
  holdings: []              # tickers your team holds, e.g. [NVDA, AAPL]; shown with ⭐ and higher priority
  poll_active_minutes: 5
  poll_idle_minutes: 30
  poll_x_minutes: 15
  late_after_minutes: 30
  followup_minutes: 60
```

In `.env` (gitignored): `NTFY_TOPIC=<long random>`, plus an optional `NTFY_SERVER` (default `https://ntfy.sh`).

### CLI

- `influence watch [--dry-run] [--once]`: the loop. `--once` runs a single cycle, for testing.
- `influence alerts setup`: generates a 32-character random topic and writes it to `.env` if missing. It then prints
  the phone steps: install ntfy, then "+" and subscribe to that topic.
- `influence alerts test`: sends a test notification.
- `influence status` gains an Alerts section:
  - watch heartbeat age (red when more than 15 minutes old during the active window);
  - alerts sent in the last 7 days by kind;
  - pending follow-ups;
  - failed sends.

## Error handling

| Situation | Behaviour |
|---|---|
| Truth Social 429/1015/403 | The existing collector backs off and ends that collect as `partial`; the cycle still sends due follow-ups. |
| ntfy unreachable | Row `failed`, retried next cycles for 24 h; `status` shows it. |
| Model fails to load | Heads-up is sent with stance "unavailable"; the classifier is retried next cycle; the nightly `classify` fills it in. |
| Live price fetch fails | The heads-up omits prices ("price unavailable"); a follow-up stays `pending` and is retried until 30 min past due, then `skipped`. |
| `watch` crashes | The scheduled task restarts it within 1 minute; the ledger prevents duplicates. |
| PC off | Missed posts go into the `digest` on the next cycle; day-after follow-ups are still sent. |

## Testing

Offline, with a fake clock, a fake Truth Social transport, a fake price source and a recording notifier:

- **Cadence:** active vs idle windows across a weekend, Thanksgiving 2026-11-26, the 2026-11-27 early close and the
  Nov 1 DST switch.
- **Heads-up:**
  - formatting (holdings first with ⭐, "+N" folding, 220-character cut, 4 KB cap);
  - priority and tags;
  - the history line in both its "n ≥ 10" and "fewer than 10" forms.
- **Ledger:**
  - no duplicates across repeated cycles;
  - a restart between "pending" and "sent";
  - failed-send retry and the 24 h give-up;
  - pre-existing posts marked skipped on first start.
- **`follow_60m`:**
  - window math for regular, pre-market, after-hours, overnight and weekend posts;
  - an early-close truncation;
  - "skipped when more than 30 minutes past due".
- **`follow_d1`:** waits for d0+1 bars; within-range vs unusually-large wording; confounder notes.
- **Digest:** threshold at `late_after_minutes`; the 8-post cap with "+N more"; late posts get no heads-up.
- **Coordination:** `daily` skips Truth Social with a fresh heartbeat and runs it with a stale one.
- **Keep-awake:** execution-state calls on window entry and exit (ctypes mocked).
- **Notifier:** request shape against httpx MockTransport (URL, headers, body), retry, dry-run.

Live checks before handing over:

- `influence alerts test` arrives on a phone.
- `influence watch --once --dry-run` against the real database prints nothing for old posts.
- A deliberately injected fresh post in a copy of the DB produces the expected heads-up text.

## Rollout

1. `influence alerts setup` generates the topic. Each teammate installs ntfy and subscribes.
2. `influence alerts test` confirms delivery.
3. Run `watch --dry-run` for a trading day and review `logs\watch.log`.
4. Register the watch task (`scripts\register_watch_task.ps1`) to go live.

## Out of scope / later

- A cloud X checker (GitHub Actions), if X gets a token and the PC being off becomes a problem.
- Reddit alerts.
- ApeWisdom spike alerts, which need a week of snapshots first.
- Alert-on-holdings-only filtering, which is unnecessary at 2–3 alerts a week.

## Risks

- **More Truth Social traffic.** Every 5 minutes means about 1,000–1,300 requests a day instead of about 30. That
  raises rate-limit and terms-of-service exposure, and Cloudflare could start challenging the home IP. If that
  happens, the fixes are:
  - a longer `poll_active_minutes`;
  - polling only realDonaldTrump, WhiteHouse and PressSec every 5 minutes, and the rest every 30.
- **The topic is a password.** Anyone with it can read and send alerts. Share it only with the team.
- **Stance labels can be wrong,** as in the Tesla example. The heads-up shows the confidence, and follow-ups
  report unsigned moves.
- **Follow-ups describe, they don't advise.** They must not be read as trade signals; the wording says so when
  moves are within the normal range.
