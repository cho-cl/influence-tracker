from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

PostPlatform = Literal["x", "truthsocial", "reddit"]
MatchType = Literal["cashtag", "name", "bare"]


@dataclass(frozen=True)
class Post:
    platform: PostPlatform
    native_id: str
    author: str
    created_at_utc: datetime
    text: str
    url: str
    author_id: str | None = None
    # Reddit: the subreddit. Other platforms: None.
    source: str | None = None
    # Reddit: 1-based position in the day's top-of-subreddit feed (RSS has no scores).
    feed_rank: int | None = None
    metrics: dict | None = None
    # Symbols the platform itself tagged as cashtags (X entities.cashtags), without the '$'.
    cashtag_hints: tuple[str, ...] = ()


@dataclass(frozen=True)
class Mention:
    ticker: str
    match_type: MatchType
    matched_text: str


@dataclass(frozen=True)
class MatchResult:
    mentions: list[Mention] = field(default_factory=list)
    # Cashtags that look valid but are not in the configured universe, uppercased, without '$'.
    unknown_cashtags: list[str] = field(default_factory=list)
