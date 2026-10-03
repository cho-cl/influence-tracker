# Influence Tracker — Phase 1 Plan (collect → enrich → analyze)

## Context

Bryce is on a team in the **Wharton Global High School Investment Competition 2026–27**. Trading runs
**Sep 28 – Dec 4, 2026**, and judges score the *strength and articulation of the investment strategy*,
not returns. He wants a bot that tracks when influencers, politicians (on X and Truth Social) and Reddit
crowds post about stocks, and measures how each post affected the stock. Eventually he wants a research dashboard, a backtest
dataset and live alerts. This plan covers the **data pipeline and analysis**. Live alerts become a
separate Phase 2 once the data is trustworthy.

Because the judges care about reasoning, the analysis must hold up to a finance-literate reader. That
means:
- abnormal returns measured against the market;
- z-scores against the stock's normal volatility for that time of day;
- the move **before** the post measured separately from the move after it, which is how the tool tells an
  influencer who *moved* a stock from one who *reacted* to a move;
- earnings-day and stock-split confounders flagged;
- a **placebo baseline** of the same stocks on random days with no posts.

### Decisions made with Bryce
- **Sources:** X, **Truth Social** and Reddit, with a curated watchlist he maintains. The starter accounts are
  below, and he'll add more.
- **Matching:** cashtags plus company-name matching.
- **Prices:** 1-minute and daily data.
- **Stack:** Python, run daily by Windows Task Scheduler.
- **X access:** pay-per-use API with a **$10/month cap**.
- **Bullish/bearish labels:** a local finance-sentiment model.

### Verified external facts (checked 2026-09-24)

**X API**
- There has been no free tier for new developers since Feb 6 2026.
- Pay-per-use costs **$0.005 per post read**. Reading the same post again within one UTC day isn't charged twice.
- A spending cap can be set in the developer console.
- Recent search:
  - covers **7 days** (`start_time` must fall inside that window);
  - allows queries up to **512 characters**;
  - allows **450 requests per 15 minutes** (app auth);
  - supports the `from:` and `has:cashtags` operators.
- Since May 2026, keyword search no longer returns retweets.
- Posts longer than 280 characters come back truncated unless `note_tweet` is requested.

**Reddit**
- New OAuth apps need manual approval, and the old `.json` endpoints return **403** (confirmed by probe).
- **RSS still returns 200.** It gives id, author, title, body HTML and timestamp, but **no scores**.
- **ApeWisdom** (free) returns 200, with Reddit mention and upvote counts per ticker.

**Truth Social** (probed from this machine)
- Its Mastodon-compatible API **works without a login** for prominent accounts:
  - `GET /api/v1/accounts/lookup?acct=…` returns the account;
  - `GET /api/v1/accounts/{id}/statuses` returns JSON posts with HTML `content`, `created_at`, `reblog` and `quote_id`.
- Readable without a login: realDonaldTrump, JDVance1, WhiteHouse, PressSec, DonaldJTrumpJr, EricTrump and
  DevinNunes. Since Aug 2025, lesser-known accounts need a login.
- `/@user.rss` returns only the web app's HTML, so there is **no RSS**.
- **Cloudflare rate-limits fast bursts:** about 20 quick requests got a **429 (error 1015)**.
- `VP` is a squatter account with no posts. Lutnick's account last posted in Sep 2025.
- **The terms of service prohibit automated access** ("any automated system, including … spider, robot …
  scraper"). The only approved route is TMTG's "Truth API", which is licensed to institutions only.
  This plan reads public posts only, at a very low request rate, with no login. The realistic risk is a
  temporary IP block, not an account ban. Bryce accepts or rejects that risk when he approves this plan.

**Price data (yfinance)**
- Latest release is 1.7.0; require **1.5.2 or later**.
- 1-minute bars are retrievable for the **last 30 days**, at most 7 days per request.
- `yf.download` is **not thread-safe**, so make calls one at a time.
- `get_earnings_dates()` works.

**Trading calendar (`exchange_calendars`)**
- The XNYS calendar covers 2026–27, including the Nov 27 early close.
- `minute_to_session(t, direction="next")` implements the event-day rule below.
- `session_open` and `session_close` return UTC times.
- On Windows, `zoneinfo` needs the `tzdata` package.

**Sentiment model**
- Use `StephanAkkerman/FinTwitBERT-sentiment` (MIT, about 0.1B params), which was trained on
  **financial tweets**. `id2label` = `{0: NEUTRAL, 1: BULLISH, 2: BEARISH}`.
- Its training data replaced @-mentions with `@USER` and links with `[URL]`, so do the same to input text.

## Project setup

- Location: `C:\Users\Bryce Chomik\influence-tracker\` (new git repo). Package `influence_tracker`, CLI `influence`.
- Follow the `~/aquarl/pyproject.toml` conventions:
  - src layout and hatchling;
  - `requires-python = ">=3.11,<3.12"`;
  - YAML config validated with pydantic;
  - `rich` for logging;
  - pytest and ruff in a dev group.
- Create the environment with `uv venv --python 3.11`. Write `.python-version` **without a BOM** (use the
  Write tool, not PowerShell `Out-File`). Create `README.md` before the first `uv sync`.
- Dependencies: `httpx`, `feedparser`, `yfinance>=1.5.2`, `exchange_calendars`, `tzdata`, `pandas`, `numpy`,
  `scipy`, `matplotlib`, `pydantic`, `pyyaml`, `rich`, `python-dotenv`, `transformers`, `torch`. The
  PyPI torch wheel is CPU-only on Windows, which is what we want.
- `.env` holds `X_BEARER_TOKEN`. It is gitignored and never logged.
- `.gitignore` also covers `data/`, `reports/`, `logs/` and `.venv/`.
- The first commit copies this plan into the repo as `docs/PLAN.md`.
- **UTF-8 everywhere:** every `open()` passes `encoding="utf-8"`, and the scheduled task sets `PYTHONUTF8=1`.
  Emoji in posts would otherwise crash CSV, HTML and log writes on Windows.

```
influence-tracker/
  config/watchlist.yaml        accounts, subreddits, ticker universe (+names), budget, schedule
  src/influence_tracker/
    cli.py                     influence daily | collect | snapshot | classify | enrich | analyze | status
    config.py                  pydantic models for watchlist.yaml + .env
    db.py                      sqlite3; CREATE TABLE IF NOT EXISTS + schema_version check; WAL + busy_timeout
    collectors/x.py            query builder, recent-search client, budget pacing, watermark
    collectors/truthsocial.py  Mastodon-style API: slow, gap-free paging, optional backfill
    collectors/reddit_rss.py   top-of-day RSS per subreddit
    collectors/apewisdom.py    daily per-ticker mention snapshot
    mentions.py                cashtag / name / bare-ticker matcher (configured universe only)
    sentiment.py               FinTwitBERT batch classifier
    market.py                  exchange_calendars XNYS: sessions, d0, session phase
    prices.py                  1m snapshot (universe + SPY), daily fetch w/ splits, earnings dates
    events.py                  event rows, reference prices, window completeness, flags
    analysis/metrics.py        market model, AR/CAR legs, z-scores, placebo
    analysis/report.py         CSVs, charts (PNG), report.html
  scripts/register_task.ps1    creates/updates the scheduled task
  tests/                       pytest, offline fixtures only
```

## Components

### Config (`config/watchlist.yaml`)

**Starter watchlist.** Each account has a `category`: politician, investor, short_seller, exec, influencer or
media. Bryce will add more; adding an account is a YAML edit only. Handles are resolved on the first run, and
`status` lists any that fail, so a typo is caught right away.

| Platform | Account | Category | Why it's on the list |
|---|---|---|---|
| X | `elonmusk` | exec | Tesla, xAI; his posts have moved TSLA repeatedly |
| X | `jimcramer` | media | High-volume stock calls ("inverse Cramer") |
| X | `CathieDWood` | investor | ARK Invest; growth and tech names |
| X | `BillAckman` | investor | Pershing Square; public activist positions |
| X | `chamath` | investor | Social Capital; SPACs and tech |
| X | `michaeljburry` | investor | Big-short calls (NVDA, PLTR) |
| X | `TheRoaringKitty` | influencer | GME and meme-stock catalyst |
| X | `saylor` | exec | Strategy (MSTR) |
| X | `brian_armstrong` | exec | Coinbase (COIN) |
| X | `muddywatersre` | short_seller | Short reports; clean bearish events |
| X | `CitronResearch` | short_seller | Short and long calls |
| X | `realDonaldTrump` | politician | Rarely posts on X, but the filter makes it cheap to watch |
| X | `WhiteHouse` | politician | Policy announcements |
| X | `JDVance` | politician | Vice President |
| X | `SecScottBessent` | politician | Treasury Secretary |
| X | `howardlutnick` | politician | Commerce Secretary; tariffs and company deals |
| X | `DavidSacks` | politician | White House AI and crypto czar |
| Truth Social | `realDonaldTrump` | politician | **The main market mover.** Company posts (Intel, Tesla, defense) and "great time to buy" |
| Truth Social | `JDVance1` | politician | Vice President (verified; the `VP` handle is a squatter) |
| Truth Social | `WhiteHouse` | politician | Official announcements |
| Truth Social | `PressSec` | politician | Press Secretary Karoline Leavitt |
| Truth Social | `DonaldJTrumpJr` | influencer | Business and political commentary |
| Truth Social | `EricTrump` | exec | Trump Organization and crypto ventures |
| Truth Social | `DevinNunes` | exec | CEO of Trump Media (DJT) |
| Reddit | `wallstreetbets` | — | Meme-stock crowd; biggest retail-attention signal |
| Reddit | `stocks` | — | Mainstream retail discussion |
| Reddit | `investing` | — | Longer-horizon retail |
| Reddit | `StockMarket` | — | General market chatter |
| Reddit | `options` | — | Options-flow chatter, a leading retail signal |

ApeWisdom filters: `all-stocks` and `wallstreetbets`. Leave out high-volume X news bots (`unusual_whales`,
`DeItaone`); they would eat the budget.

**Ticker universe.** About 60 symbols, each with `names` and an `ambiguous` flag.
- Beyond the usual retail favorites, include the names politicians post about: **DJT**, defense (LMT, RTX,
  NOC, GD), INTC, BA, WMT, MAT, and names in the news over tariffs.
- Ambiguous names include Apple, Meta, Target, Ford, Intel, Amazon, Visa, Shell, Snap, Block and Robinhood.
- **SPY** is always included, marked `benchmark: true` (as is QQQ). Benchmarks are never analyzed as events.
- Events are limited to this universe. Cashtags outside it are counted, and `status` lists the most frequent
  ones so Bryce can add them. A newly added ticker still gets 30 days of 1-minute history backfilled.

**X budget.** Settings `monthly_budget_usd: 10` and `billing_cycle_day` (the day X's cycle resets).

### X collector
- **User IDs.** Handles are resolved to user IDs once, via `GET /2/users/by`, and cached in `accounts`.
  Posts carry `author_id` as a normal field, so no `expansions` are needed and there are no user-read charges.
- **Queries.** Pattern: `(from:a OR from:b …) (has:cashtags OR Tesla OR "Palantir" …) -is:retweet`.
  - Queries are greedily chunked so **each stays at or under 512 characters**; the collector runs every
    account-chunk × term-chunk combination.
  - X bills per post returned, so we pay only for posts that mention a stock.
  - Verify `has:cashtags` with one live query in M1. If it fails, fall back to explicit `$TICKER` terms.
- **Fields.** `tweet.fields=created_at,author_id,entities,public_metrics,lang,note_tweet`. When `note_tweet`
  is present, use its text and entities, so long posts are matched in full.
- **Watermark (no silent loss).**
  - Each run sets `end_time = now − 30s` when it starts.
  - `start_time = max(watermark, now − 7d + 5min)`.
  - The watermark moves to `end_time` **only when every query finishes paginating**.
  - If the budget stops a run early, the watermark stays put and the next run repeats the window; some posts
    are re-read and billed again, and a warning is logged. Posts are deduplicated by their primary key.
- **Budget pacing.**
  - Daily allowance = `monthly_budget_usd / 30 / 0.005` ≈ 66 posts.
  - Per-run cap = allowance × (days since last full run, at most 7), limited by what's left of the
    **billing cycle** budget.
  - Reads are logged in `x_usage` **per account**. The console spending cap is the final backstop.
  - Expected spend with server-side filtering is about $4–9/month for the starter list, with Cramer and Musk
    as the biggest share; pacing keeps it under $10.
  - If the allowance keeps running out, `status` shows which account is eating it, so Bryce can drop or
    swap that account.

### Truth Social collector (free, no login)
- **Account IDs.** Handles are resolved once via `/api/v1/accounts/lookup` and cached in `accounts`.
- **Gap-free paging.** Request `/api/v1/accounts/{id}/statuses?min_id=<watermark>&limit=20&exclude_reblogs=true`.
  - `min_id` returns the posts **immediately after** the watermark, oldest first.
  - After each page is stored, move that account's watermark to the newest id on the page. Stop when a page
    comes back empty.
  - An interrupted run resumes exactly where it stopped, and missed days heal themselves: the full history is
    free, unlike X's 7-day window.
  - M1 checks `min_id` live. If it isn't supported, fall back to paging backward with `max_id` from the newest
    post until the watermark is reached, and advance the watermark only after the whole sweep finishes.
- **Rate limiting.** Requests go one at a time with **at least 5 seconds between them** and a normal browser
  User-Agent. On a 429 or Cloudflare 1015: wait 60 seconds, then 120, then give up for this run. The
  watermark means nothing is lost.
  - If Cloudflare starts serving challenge pages (403 with HTML), fail soft and log it. Browser impersonation
    through `curl_cffi` is a possible later fix; don't build it now.
- **Content rules.**
  - Strip the HTML in `content` to text.
  - Skip reblogs (like X retweets) and posts with no text. Many Trump posts are images or video; that's a
    reported limitation.
  - For quote posts, only the author's own text counts.
- **First-run window and optional backfill.** The first run covers the last 30 days, which lines up with the
  1-minute price history. **Optional:** `influence backfill truthsocial --since 2025-01-20` (the second-term
  start) pages back through history at the same slow pace. That's roughly 500+ requests, about an hour, run
  once. Those older events get daily metrics only, since intraday data doesn't go back that far.
  - This gives M3 a large, free historical dataset of Trump posts about companies, so the analysis doesn't
    rest on only 10 weeks of live data.

### Reddit collectors
- **RSS.**
  - Fetch `/r/{sub}/top/.rss?t=day&limit=100` with feedparser, sending a descriptive User-Agent.
  - Store `feed_rank` (position in the day's top list) as a stand-in for influence, since RSS has no scores.
  - Strip body HTML to text with the stdlib `html.parser`.
  - Fail soft: if Reddit shuts RSS the way it shut `.json`, log it and keep going.
- **ApeWisdom.**
  - One snapshot per filter per day, first 3 pages, into `reddit_ticker_daily`.
  - These counts can't be backfilled, so a missed day stays a gap.

### Mention detection (`mentions.py`)
- **Cashtags:** regex `\$[A-Za-z]{1,5}(\.[A-Za-z])?`, which requires a letter so `$5` doesn't match. Combined
  with X's `entities.cashtags`, and kept only if the symbol is in the universe.
- **Company names:** matched on word boundaries, case-insensitive. **Ambiguous** names must be capitalized **and**
  appear with a finance-context word in the same post: stock, shares, earnings, calls, puts, buy, sell, short,
  price target, IPO, dividend, `$` or `%`.
- **Bare tickers** (e.g. "TSLA to the moon"): **Reddit only**, only universe symbols of 3+ letters, minus a
  stoplist (CEO, USA, ALL, …). Truth Social and X skip them: Trump's ALL-CAPS style would falsely match tickers
  like NOW, CAT, ARM, COIN and HOOD, and X already has cashtags.
- Truth Social posts rarely use cashtags, so **company-name matching does most of the work there**.
  All-caps names such as "INTEL" count as capitalized for the ambiguous-name rule.
- Output: `mentions(post_id, ticker, match_type, matched_text)`.

### Price snapshot, part of M1 (`prices.py`)
- **Every daily run** fetches 1-minute bars (`prepost=True`) for **every universe ticker and SPY**, one request at
  a time with a short pause between them. That's about 50 requests a day.
- The first run backfills the last 30 days in 7-day chunks.
- Bars are stored in `bars_1m(symbol, ts_utc, o, h, l, c, v)`. **Only sessions whose extended hours have ended
  (after 8 PM ET) are saved**, so a mid-day catch-up run never stores a partial day.
- Snapshotting everything daily means 1-minute data can't expire before it's used, and it supplies the 20
  sessions of history the intraday σ needs.
- Set `yf.config.network.retries` and back off on rate-limit errors. Don't pass a custom session.

### Sentiment (`sentiment.py`)
- Uses `pipeline("text-classification", model=cfg.model_id)`. The first run downloads the model; after that it loads with
  `local_files_only=True`, so the daily run doesn't need Hugging Face to be reachable.
- Input preprocessing: replace @-mentions with `@USER` and URLs with `[URL]`, and truncate to 512 tokens.
- Output: lowercase labels, stored as `stance`, `stance_conf` and `stance_model` **per post**. Known limitation:
  a post that's bullish on one ticker and bearish on another gets one label.

### Events and windows (`market.py`, `events.py`) — the silent-error hotspot
- **Event.** One `(post_id, ticker)` pair. **t0** is the post time, stored in UTC.
- **d0 (event day).** `minute_to_session(t0.floor("min"), direction="next")`. That means:
  - pre-open and regular-hours posts → the same day;
  - after-close, overnight, weekend and holiday posts → the next session.
- **Reference price.** The close of the last 1-minute bar whose **start is at or before t0 − 60 seconds**.
  Yahoo timestamps each bar by its start, so the bar containing t0 finishes after the post.
- **Window ends.** Use the last bar at or before the window end, since minutes with no trades have no bar.
- **Day-0 legs** (answers "did they move it or react to it?"). Both legs come from 1-minute bars for consistency.
  - **Pre-leg:** the previous regular close to the reference price.
  - **Post-leg:** the reference price to the d0 close.
  - Posts made while the market is closed have a pre-leg covering after-hours and pre-market trading.
- **Intraday windows**, regular-session posts only:
  - **−60 to 0 minutes** (the pre-window), then **+5, +15, +30 and +60 minutes**.
  - Windows are truncated at the session close, respecting early closes, and flagged. A post in the first
    minute of the session is flagged `spans_open`.
- **Status.** An event moves from `pending` to `complete` once daily data through d0+5 exists. Every run
  re-checks pending events.
- **Flags, set when an event completes (not when it's created):**
  - `earnings`: an earnings date within ±1 session of d0. If the lookup fails the flag is NULL (unknown);
    unknown events stay in, and the report counts them.
  - `split`: a stock split between d0−5 and d0+5, taken from yfinance `actions`. These events are excluded.
  - `clustered`: the same ticker has more than one post in a session. The first post is primary.

### Analysis (`analysis/metrics.py`, `analysis/report.py`)

**Daily data.** Daily bars and splits are **downloaded fresh on each `enrich` run**: one request per ticker,
covering d0 − 150 sessions to now. They are never merged incrementally. Merging old adjusted prices with new
ones would create fake returns at the join (a 10:1 split would show as −90%).

**Market model.** Fit `r_i = α + β·r_SPY` over sessions −130 to −11 relative to d0. Abnormal return
`AR = r_i − (α + β·r_SPY)`, and σ comes from the estimation residuals.

**Daily metrics**, each with `z = CAR / (σ·√L)`, where L is the window length in sessions:
- pre-drift CAR from day −5 to −1;
- day-0 pre-leg and post-leg ARs (SPY split into the same legs, times β);
- CAR from day +1 to +5, cumulative from the post;
- relative volume: d0 volume ÷ mean volume over days −25 to −6.

**Intraday metrics.** For each window, the abnormal return is `r − β·r_SPY` over the same minutes. The
z-score uses σ of **the same clock-time window over the prior 20 sessions**, because volatility is highest
near the open and close. Truncated windows use the matching truncated window. If fewer than 10 prior
sessions are valid, z is NULL.

**Placebo baseline.** For each event ticker, pick 5 random sessions from the prior 120 with no event on that
ticker within ±5 sessions, and compute the same daily metrics. The report compares event and placebo
results for the share of |z| > 1.96 and for mean |CAR|, rather than assuming 5% is normal.

**Signed impact and statistics.**
- Signs: bullish = +1, bearish = −1, neutral excluded.
- **The unit of observation is the post.** A post's ticker events are averaged before any test.
- Groups: author, category, platform and stance. Each group reports:
  - n;
  - mean and median signed CAR;
  - mean |z|;
  - share with |z| > 1.96, next to the placebo share;
  - t-test p-values, **Holm-corrected** across groups.
- Groups with n < 10 are labeled "insufficient". The whole section is labeled **exploratory**.
- Robustness check: rerun the signed results using only posts with `stance_conf ≥ 0.6`.

**Market-wide posts.** Posts about tariffs and similar topics move the whole market, and the market model
removes a market-wide effect. For politician-category events the report also shows the **raw SPY move**, and
the caveats section explains why.

**Default exclusions.** Earnings-flagged, split-flagged, benchmark and clustered non-primary events. CLI flags
switch each exclusion off.

**Reddit attention.** One simple table: ApeWisdom mention spike (today ÷ trailing 7-day mean) against the next
session's AR, per ticker.

**Output folder `reports/YYYY-MM-DD/`:**
- `events.csv` (one row per event: snippet, URL, stance, confidence, every metric and flag) and `summary.csv`.
- Charts (PNG):
  - **average CAR path from day −5 to +5, bullish vs bearish, with a 95% CI band, alongside the placebo path**;
  - mean signed post-leg CAR per author, with n;
  - average intraday AR path from −60 to +60 minutes;
  - the top 10 individual events, each a 1-minute price chart with a post marker.
- `report.html`: static tables, charts and a plain-language caveats section:
  - correlation, not causation;
  - small n;
  - watchlist selection bias;
  - market-wide posts;
  - sentiment-model errors (Trump's style is outside the model's training data);
  - image, video and screenshot posts are invisible to text matching.
- Load the `dataviz` skill before writing chart code.

### CLI, scheduling, ops

**CLI commands.**
- `influence daily` runs `collect → snapshot → classify → enrich`. Stages are isolated: a failed stage logs an
  error and writes a `runs` row, and later stages still run. The exit code is non-zero if any stage failed, so
  Task Scheduler shows the failure.
- `influence analyze [--since] [--include-earnings] …` runs on demand.
- `influence backfill truthsocial --since DATE` is optional and run once, by hand. It's resumable.
- `influence status` shows:
  - row counts;
  - the last run of each stage;
  - X spend against the billing-cycle budget, broken down by account;
  - the last successful fetch for each account on each platform, plus any handles that failed to resolve;
  - Truth Social rate-limit hits;
  - pending events and how old the oldest is;
  - the most frequent cashtags outside the universe.

**`scripts/register_task.ps1`** sets up a daily task at a configurable local time, default 6:30 PM (after the
US close). The task:
- runs `<repo>\.venv\Scripts\influence.exe daily` directly (no `uv run`, so no PATH problems or re-syncs);
- has its **working directory set to the repo** (the default is System32);
- sets `PYTHONUTF8=1`;
- uses `StartWhenAvailable`, `AllowStartIfOnBatteries` and `DontStopIfGoingOnBatteries`.

**Logging and database.** Logs go to `logs/` (rotating file) and to the `runs` table. SQLite runs in WAL mode
with a busy timeout, so running `analyze` by hand during a scheduled run is safe.

## Data model (SQLite, `data/tracker.db`)
- `accounts(platform, handle, platform_user_id, category, active)`
- `posts(platform [x|truthsocial|reddit], native_id, author, created_at_utc, text, url, feed_rank, metrics_json, stance, stance_conf, stance_model, collected_at)`
- `mentions`
- `unknown_cashtags`
- `bars_1m`, `bars_1d` (replaced on each fetch, with splits)
- `earnings`
- `events(post_id, ticker, t0, d0, session_phase, status, flags…)`
- `event_metrics(event_id, window, raw_ret, mkt_ret, abn_ret, z, rel_volume)`
- `placebo_metrics`
- `reddit_ticker_daily`
- `x_usage`
- `watermarks(source, key, value)`: the X time watermark, plus one Truth Social last-status-id per account
- `runs`
- `schema_version`

Every write is an idempotent upsert, so any stage can be re-run safely.

## Milestones (phase-gated: Bryce checks each gate on real data before the next one starts)

**M1 — Collect and snapshot.** Target: running around Sep 28, the day trading starts. Truth Social, Reddit and
prices can go live immediately; X goes live once Bryce's token exists.
- Build:
  - scaffold, config, db;
  - X collector (queries, watermark, pacing);
  - Truth Social collector (paging, rate limiting, optional `backfill`);
  - RSS and ApeWisdom;
  - mentions;
  - the 1-minute snapshot with 30-day backfill;
  - `collect` / `snapshot` / `status`, and the scheduled task.
- Gate:
  - a real run stores X, Truth Social and Reddit posts with correct ticker tags;
  - every starter handle resolves (fix any that `status` flags);
  - 1-minute bars exist for all tickers and SPY;
  - X console spend matches `x_usage`;
  - the task fires once on its own.

**M2 — Classify and enrich.** Calendar, daily bars and splits, earnings, events, reference prices and legs,
sentiment. There's no deadline, because 1-minute data is already being kept.
- Gate: Bryce hand-checks 3 real events against Yahoo charts, covering t0, d0, reference price, pre-leg,
  post-leg and the +15-minute return.

**M3 — Analyze and report.** Metrics, placebo, summaries, charts, report.html.
- Gate: Bryce opens the report, and one known event's numbers are reproduced by hand.

**Phase 2 (separate plan, later): live alerts**, reusing the collectors and database.

## Prerequisites for Bryce (during M1)
1. Set up X:
   - create an X developer account, then a project and app;
   - buy about $10 of credits;
   - **set the monthly spend limit to $10**;
   - note the billing-cycle reset day;
   - generate an app Bearer Token and put it in `.env`.
2. Adjust the starter watchlist (accounts, subreddits and tickers) as you like. Truth Social accounts must be
   well known, since lesser-known accounts need a login, which this plan doesn't use.
3. Decide whether to run the optional Truth Social backfill (`--since 2025-01-20`) once M1 is live.

## Verification
- `uv run pytest` runs offline against recorded fixtures: an RSS sample (captured during planning), ApeWisdom
  JSON, a Truth Social statuses page (recorded live in M1, including a reblog and a media-only post), an X
  search response including a `note_tweet`, and synthetic bars.
  - `test_truthsocial`:
    - HTML is stripped to text;
    - reblogs and empty posts are skipped;
    - `min_id` paging advances the watermark page by page;
    - a 429 triggers backoff, then gives up without losing the watermark;
    - an interrupted backfill resumes where it stopped.
  - `test_mentions`:
    - `$5` vs `$TSLA`, and `$BRK.B`;
    - "Apple pie" vs "Apple earnings beat";
    - "intel" vs "Intel shares";
    - bare-ticker stoplist;
    - Trump-style all-caps text ("NOW IS THE TIME … CAT … ARM") produces **no** bare-ticker matches outside
      Reddit, while "INTEL" still matches by name;
    - symbols outside the universe are counted, not matched.
  - `test_x_collector`:
    - every query is ≤ 512 characters, and each account and term appears exactly once;
    - a budget stop does not advance the watermark;
    - the `note_tweet` text is used.
  - `test_market_events` (heaviest):
    - weekend post → Monday d0; after-close → next session; pre-open → same day;
    - Thanksgiving (Nov 26), and the Nov 27 early close truncating a +60-minute window;
    - a post across the **Nov 1 2026 DST switch**;
    - the reference bar excludes the bar containing t0; missing-minute bars;
    - `pending → complete`, with flags set at completion.
  - `test_metrics`: synthetic `r = 2·r_mkt + noise` with a +3% jump after t0 should give β ≈ 2, post-leg ≈ +3%,
    pre-leg ≈ 0 and a large z. A no-jump control should give |z| < 2. A time-of-day volatility pattern should
    be handled by the same-clock σ.
  - `test_sentiment`: label mapping and the `@USER`/`[URL]` masking, against a stubbed pipeline. An optional
    `@pytest.mark.slow` test runs the real model.
- Live smoke test for each milestone: `influence daily`, then `influence status`, then inspect rows with
  `sqlite3`. For M2 and M3, also hand-check against Yahoo charts as described in the gates.

## As built — M1 differences from this plan (2026-09-25)

Facts found while building. Where they contradict a section above, this section wins.

- **Truth Social ignores `min_id` and `exclude_reblogs`** (verified live). The collector walks each account
  backward with `max_id` from the newest post down to its watermark. Progress is saved after every page
  (`sweep:<handle>`), and the watermark moves only when a walk finishes, so an interrupted run resumes without
  gaps. Reblogs are filtered client-side. The backfill walks backward from the first-run start to `--since`.
  Pages hold at most 20 posts; a short page does not mean the end of history.
- **X progress is per query.** Each query records what it has read. After a budget stop, the next run reads
  the unread older part first. Posts that age out of the 7-day window before they are read are logged as a
  window gap, and the run is marked partial. Pacing: the daily allowance × days since the last caught-up run
  (at most 7), minus what unfinished runs have already read, within the billing-cycle budget. A 1-hour slack
  stops scheduler drift from counting a day twice.
- **X queries: ambiguous names only appear together with a finance word.** X keyword search ignores case,
  so "Target", "Block", "Meta" or "Truth Social" alone would bill every ordinary post. Those names are
  searched in separate queries that also require one of: stock, stocks, shares, earnings, investors, buy,
  sell, bullish, bearish, "price target". That's 11 queries per run for the starter watchlist. Post text is
  HTML-unescaped. HTTP 402 (credits depleted) is an error.
- **Mentions.**
  - New `cased_names`: names that must be capitalized but need no finance word ("Super Micro", "General
    Dynamics", "Trump Media").
  - On their own, calls, puts, options, long and short are not finance-context words, because each is everyday
    English. Precise forms like "call options" and "short seller" still count.
  - Trader jargon is excluded: "price target", "block trade", "shell company", "Oracle of Omaha".
  - RTX is stoplisted for bare Reddit tickers, because on Reddit it usually means Nvidia GPUs.
- **Reddit.**
  - Unauthenticated RSS allows about one request per minute (`x-ratelimit-*` headers), so 5 subreddits take
    about 5 minutes.
  - AutoModerator's daily and weekly threads are skipped.
- **Yahoo 1-minute data.**
  - Limits: at most 8 days per request (we use 7) and 30 days back.
  - Minutes with no trades have no bar, so a normal day has about 820–945 bars, not 960.
  - Extended-hours volume is mostly 0, so M2/M3 should not build volume metrics on extended-hours bars.
- **Schedule: 8:30 PM local** (this machine is on US Eastern time). Extended hours end at 8 PM ET, so a
  6:30 PM run would only save the day's bars the next day. The trigger follows local time across DST.

## As built — M2 notes (2026-09-25)

- **"Regular close" means the last 1-minute bar before the close**: the 15:59 bar, or 12:59 on early-close
  days. It is not Yahoo's official daily close. The closing auction lands inside the 16:00 bar and can't be
  recovered from 1-minute data, so the two differ by a few basis points. The day-0 legs chain consistently.
  When hand-checking, compare against the 15:59 1-minute bar. M3 must not mix `bars_1d` closes and 1-minute
  legs inside one return.
- **Pending events have no reference price or windows yet.** They are computed at completion, which needs
  daily bars through d0+5.
- **Earnings.**
  - Yahoo's earnings calendar (yfinance scrapes it) returns about 25 dates per symbol.
  - A report maps to the first session whose close is after it, so an after-close report maps to the next
    session.
  - `earnings_flag` is NULL (unknown) when the latest fetch failed, or when the stored dates don't reach back
    before the event.
- **Daily bars** are re-downloaded each enrich for every ticker with an event, plus SPY, from the earliest d0
  minus 150 sessions. That's one request per ticker: 22 today, more as the ticker count grows.
- **Sentiment.**
  - The FinTwitBERT tokenizer is uncased, so all-caps text needs no special handling (measured).
  - The tokenizer reports `model_max_length = 1e30`, so `max_length=512` is passed explicitly; long Reddit
    posts crashed without it.
  - Performance: roughly 0.4–0.6 s per post on CPU, about 1 GB peak memory.
  - The model is a one-time 420 MB download into `~/.cache/huggingface`.
  - Only posts that mention a non-benchmark watchlist ticker are classified.
- **`influence events`** has table, `--id` detail and `--csv` views. The CSV is written as utf-8-sig, guards
  post text against Excel formula injection, and exports native ids as text so Excel doesn't round them.
- **Independent check (2026-09-25).** 8 completed events were recomputed with separate code from a fresh
  Yahoo 1-minute download, across after-hours, weekend, Labor Day and regular-session posts with truncated
  windows. The check covered d0, phase, reference bar and price, legs, and all windows for the ticker and SPY.
  Result: 0 mismatches.

## As built — M3 notes (2026-09-26)

- **Daily windows actually used** (standard event-study windows; this replaces the day-0-legs-first framing
  above):
  - pre = CAR[−5,−1]: did the stock already move before the post?
  - event = CAR[0,+1]: the headline window.
  - post = CAR[+2,+5]: did the move continue or reverse?

  The day-0 legs and the intraday windows remain as a decomposition wherever 1-minute data exists.
- **Market model.** OLS on sessions −130..−11, with at least 60 observations; otherwise the event is marked
  `no_model`. σ is the residual standard deviation (n−2 degrees of freedom), and z = CAR / (σ·√L).
  - Leg z-scores use the full-day σ, which is conservative.
  - Intraday z-scores use σ of the same clock window over the prior 20 sessions (at least 10 needed).
- **Post-level statistics.**
  - A post's CAR is the mean over its included events.
  - Its z treats those events as **one equal-weighted portfolio**: σ_p comes from the averaged
    estimation-window residuals. For a one-stock post this equals the event z. It replaces a plain mean of
    z-scores, which understated multi-stock posts: under no effect the standard deviation was 1/√k.
  - A very volatile stock dominates a mixed post's σ_p.
- **Tests.** One-sample t-tests on stance-signed post CARs, with Holm correction within each family (author,
  category, …), subset and window. Any group with n < 10 is reported as insufficient and not tested.
- **Placebo days.**
  - Up to 5 per included event, drawn from sessions −130..−10.
  - Skipped: days within ±5 sessions of any event on that ticker, within ±1 session of an earnings report,
    or within ±5 of a split (the same screens events get, unless `--include-*`).
  - Each placebo day gets its own market model.
  - A stock-day drawn for two events counts once.
- **Daily-bar lookback raised to 300 sessions** (enrich), so each placebo day has its own estimation window.
- **Disclosed, not corrected.**
  - Different posts on the same stock can have overlapping windows. An opt-in `--exclude-overlap` was
    proposed and deferred.
  - Estimation windows can contain other posts' reactions.
  - Day 0 of a regular-session post includes the move before the post.
- **Verification.**
  - A reviewer re-implemented the whole method independently and compared it cell by cell: 0 mismatches on
    events, placebo days, posts, groups, paths and magnitude.
  - The portfolio z was verified independently, including correlated and identical-stock cases.
  - Headline statistics were recomputed from the exported CSVs with scipy and matched.
- **First full-history result (2026-09-26).** 117 stance-labelled posts; mean signed CAR[0,+1] −0.28%
  (95% CI −0.90% to +0.35%, p = 0.38). The share of |z| > 1.96 in the event window was 5.5%, against 5.6%
  on placebo days. In the pre window it was 11.5% against 5.9%: stocks were already moving before the posts.

## As built — Phase 2 live alerts (2026-10-02)

Design: `docs/superpowers/specs/2026-09-28-live-alerts-design.md`; plan: `docs/superpowers/plans/2026-09-29-live-alerts.md`.
`influence watch` runs on this PC (Truth Social blocks cloud servers) and pushes ntfy alerts: a heads-up per stock
post, a 60-minute follow-up, a day-after follow-up and one "while you were away" digest for posts found late.

- **ntfy JSON publish.** Messages are POSTed as JSON to the server root (`topic`, `title`, `message`, `priority`,
  `tags`, `click`) instead of header fields, because httpx header values must be ASCII and titles carry "⭐" and "·".
  Bodies are capped at 4,000 bytes; one retry after 5 s.
- **Ledger.** The `alerts` table (`UNIQUE (kind, platform, native_id)`) is the no-duplicates record. A row is `pending`
  before sending and `sent` only after a 2xx. Failed sends are retried for 24 h counted from the first failure (not
  from `due_at`, so an alert first sent late after downtime still gets its retries), then left `failed`.
- **First-run marking.** The first run marks every existing stock post `skipped` and sends nothing. A post stored before
  alerts were turned on stays quiet even if a later ticker edit re-tags it.
- **Stale-bar guard.** A follow-up waits while Yahoo's newest bar is more than 15 minutes older than the window end; it
  is `skipped` 30 minutes after due rather than reporting a stale price.
- **Follow-ups.** Day-after notes use the stored earnings dates whatever the latest fetch did, and skip dates outside
  the XNYS calendar. Follow-ups say "+N more" when a post names more tickers than they list.
- **Watch process.** One instance per database (`data/watch.lock`, released by the OS if the process dies). It
  reloads `config/watchlist.yaml` and `.env` when they change, and stops if alerts are switched off or the topic is
  removed. Each watch collect gets its own `runs` row, so `status` shows its outcomes and rate-limit hits. Waits are
  sliced so a resume from sleep is not delayed. The PC is kept awake 04:00–20:00 ET on trading days.
- **The nightly run steps aside** only for the platforms the live watch collects (recorded with each heartbeat; the
  heartbeat counts as live for 15 minutes). A watch without an X token leaves X to the nightly run.
- **CLI.** `influence watch` without `NTFY_TOPIC` (and without `--dry-run`) exits 2 before opening a run row.
  `alerts setup` generates a 32-character topic (`influence-` + 22 random URL-safe characters).
- **Config.** The `alerts:` block sits above `tickers:` in `watchlist.yaml`, so a ticker appended at the end of the
  file still lands in the list.
- **Tests.** 840 passed, 9 deselected (live/slow). Offline throughout: injected clocks, notifiers, price sources and
  fetchers.
- **Scheduled task.** `scripts/register_watch_task.ps1` registers "InfluenceTracker Watch" (at logon + daily 03:50,
  restarts on failure, never two copies), running `scripts/run_watch.cmd` (output in `logs\watch.log`). It does not
  touch the nightly "InfluenceTracker Daily" task.
