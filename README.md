# influence-tracker

Tracks when influencers, politicians and Reddit crowds post about stocks, and measures how each post moved
the stock. Built for the Wharton Global High School Investment Competition.

- **Collect:** posts from a curated watchlist on X (pay-per-use API), Truth Social (public API) and Reddit
  (top-of-day RSS), plus ApeWisdom's daily Reddit mention counts.
- **Tag:** finds tickers in each post by `$CASHTAG`, company name, or bare ticker (Reddit only).
  Ambiguous names such as "Apple" or "Target" only count when the post also talks about stocks or shares.
- **Prices:** keeps 1-minute bars for every watchlist ticker plus SPY (Yahoo only serves the last 30 days),
  and pulls daily bars, splits and earnings dates.
- **Label:** a local finance-sentiment model (FinTwitBERT) marks each stock post bullish, bearish or neutral.
- **Events:** every post × ticker gets its event day, a no-look-ahead reference price, and the stock's and
  SPY's moves before and after the post.
- **Analyze:** a market-model event study (abnormal returns before, around and after each post, stance-signed
  tests per author, a placebo baseline of ordinary days) written to a self-contained `report.html` with
  charts and CSVs.

See `docs/PLAN.md` for the full design, the methodology and what was verified.

## Setup (Windows, Python 3.11 via uv)

```
uv venv --python 3.11
uv sync
copy .env.example .env      # then paste your X bearer token (optional; X is skipped without it)
```

The first `classify` run downloads the sentiment model once (about 420 MB).

## Commands

```
uv run influence daily        # collect -> snapshot -> classify -> enrich (what the scheduled task runs)
uv run influence collect      # new posts from Truth Social, Reddit RSS, ApeWisdom, X
uv run influence snapshot     # save 1-minute bars for every watchlist ticker + SPY
uv run influence classify     # label new stock posts bullish / bearish / neutral
uv run influence enrich       # build events: daily bars, earnings, reference prices, return windows
uv run influence status       # health, counts, X spend, coverage
uv run influence events       # table of events (--id N for one in detail, --csv FILE to export)
uv run influence analyze      # event-study report -> reports/<date>/report.html (+ CSVs, charts); --open to view
uv run influence backfill truthsocial --since 2025-01-20   # one-time, slow, resumable
```

Schedule the nightly run (8:30 PM local; the machine is on US Eastern time):

```
powershell -ExecutionPolicy Bypass -File scripts\register_task.ps1
```

Edit `config/watchlist.yaml` to change accounts, subreddits or tickers.

## Tests

```
uv run pytest            # offline; live network tests are marked and skipped by default
```
