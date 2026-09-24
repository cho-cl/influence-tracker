# influence-tracker

Collects stock-related posts from a curated watchlist on X, Truth Social and Reddit, snapshots
1-minute price data, and (in later milestones) measures how each post moved the stock.

See `docs/PLAN.md` for the full design and milestones.

## Setup

```
uv venv --python 3.11
uv sync
copy .env.example .env   # then paste your X bearer token
```

## Commands

```
uv run influence collect      # pull new posts from X, Truth Social, Reddit RSS, ApeWisdom
uv run influence snapshot     # save 1-minute bars for every watchlist ticker + SPY
uv run influence daily        # collect + snapshot (what the scheduled task runs)
uv run influence status       # health, spend, counts
uv run influence backfill truthsocial --since 2025-01-20   # optional, one-time, slow
```

Edit `config/watchlist.yaml` to change accounts, subreddits or tickers.
