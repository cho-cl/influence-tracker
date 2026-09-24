from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator, model_validator

Category = Literal["politician", "investor", "short_seller", "exec", "influencer", "media"]
Platform = Literal["x", "truthsocial", "reddit"]

REPO_ROOT = Path(__file__).resolve().parents[2]


class Account(BaseModel):
    handle: str
    category: Category
    active: bool = True


class Ticker(BaseModel):
    symbol: str
    names: list[str] = Field(default_factory=list)
    ambiguous_names: list[str] = Field(default_factory=list)
    aliases: list[str] = Field(default_factory=list)
    benchmark: bool = False

    @field_validator("symbol", "aliases", mode="after")
    @classmethod
    def _upper(cls, v):
        return [s.upper() for s in v] if isinstance(v, list) else v.upper()

    @property
    def yahoo_symbol(self) -> str:
        return self.symbol.replace(".", "-")


class XConfig(BaseModel):
    accounts: list[Account]
    monthly_budget_usd: float = 10.0
    cost_per_post_read_usd: float = 0.005
    cost_per_user_read_usd: float = 0.010
    billing_cycle_day: int = Field(default=1, ge=1, le=28)
    max_catchup_days: int = Field(default=7, ge=1, le=7)


class TruthSocialConfig(BaseModel):
    accounts: list[Account]
    min_request_interval_s: float = Field(default=5.0, ge=5.0)
    first_run_days: int = Field(default=30, ge=1)
    user_agent: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36"
    )


class RedditConfig(BaseModel):
    subreddits: list[str]
    rss_limit: int = Field(default=100, ge=1, le=100)
    user_agent: str = "windows:influence-tracker:0.1 (personal research project)"


class ApeWisdomConfig(BaseModel):
    filters: list[str] = Field(default_factory=lambda: ["all-stocks", "wallstreetbets"])
    pages: int = Field(default=3, ge=1)


class PricesConfig(BaseModel):
    backfill_days: int = Field(default=29, ge=1, le=29)
    request_pause_s: float = Field(default=1.0, ge=0.0)


class Watchlist(BaseModel):
    x: XConfig
    truthsocial: TruthSocialConfig
    reddit: RedditConfig
    apewisdom: ApeWisdomConfig = Field(default_factory=ApeWisdomConfig)
    prices: PricesConfig = Field(default_factory=PricesConfig)
    bare_ticker_stoplist: list[str] = Field(default_factory=list)
    tickers: list[Ticker]

    @model_validator(mode="after")
    def _check(self) -> Watchlist:
        seen: dict[str, str] = {}
        for t in self.tickers:
            for sym in [t.symbol, *t.aliases]:
                if sym in seen:
                    raise ValueError(f"ticker symbol/alias {sym!r} appears twice ({seen[sym]} and {t.symbol})")
                seen[sym] = t.symbol
        spy = next((t for t in self.tickers if t.symbol == "SPY"), None)
        if spy is None or not spy.benchmark:
            raise ValueError("SPY must be in tickers with benchmark: true (it is the market proxy)")
        for platform, accounts in (("x", self.x.accounts), ("truthsocial", self.truthsocial.accounts)):
            handles = [a.handle.lower() for a in accounts]
            dupes = {h for h in handles if handles.count(h) > 1}
            if dupes:
                raise ValueError(f"duplicate {platform} handles: {sorted(dupes)}")
        self.bare_ticker_stoplist = [s.upper() for s in self.bare_ticker_stoplist]
        return self

    def ticker(self, symbol: str) -> Ticker | None:
        symbol = symbol.upper()
        for t in self.tickers:
            if t.symbol == symbol or symbol in t.aliases:
                return t
        return None

    @property
    def event_tickers(self) -> list[Ticker]:
        return [t for t in self.tickers if not t.benchmark]


@dataclass(frozen=True)
class Settings:
    root: Path
    config_path: Path
    data_dir: Path
    db_path: Path
    logs_dir: Path
    reports_dir: Path
    x_bearer_token: str | None


def load_watchlist(path: Path) -> Watchlist:
    with open(path, encoding="utf-8") as f:
        return Watchlist.model_validate(yaml.safe_load(f))


def load_settings(root: Path | None = None) -> Settings:
    root = root or REPO_ROOT
    load_dotenv(root / ".env", encoding="utf-8")
    token = os.environ.get("X_BEARER_TOKEN", "").strip() or None
    data_dir = root / "data"
    return Settings(
        root=root,
        config_path=root / "config" / "watchlist.yaml",
        data_dir=data_dir,
        db_path=data_dir / "tracker.db",
        logs_dir=root / "logs",
        reports_dir=root / "reports",
        x_bearer_token=token,
    )
