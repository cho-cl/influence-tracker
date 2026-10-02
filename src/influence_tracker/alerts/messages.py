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
STANCE_TAGS = {
    "bullish": "chart_with_upwards_trend",
    "bearish": "chart_with_downwards_trend",
    "neutral": "speech_balloon",
}
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


def _more(n: int) -> str:
    return f"+{n} more — run: influence events"


def _left_out(post: PostInfo, shown: int) -> list[str]:
    """Follow-up rows stop at a few tickers; say how many of the post's tickers the message leaves out."""
    rest = len(set(post.tickers)) - shown
    return [_more(rest)] if rest > 0 else []


def _stance(post: PostInfo) -> str:
    if post.stance is None:
        return "stance unavailable"
    return f"{post.stance} ({post.stance_conf:.2f})" if post.stance_conf is not None else post.stance


def _history_line(author: str, history: History | None) -> str:
    # mean_signed_car is +CAR after bullish posts and -CAR after bearish ones, and n_posts counts only those posts.
    # A bare signed percentage would read as the stocks' own move, which is the opposite for bearish posts.
    if history is None:
        return f"History: no study numbers available for {author}."
    if history.n_posts < MIN_GROUP_POSTS:
        return f"History: fewer than {MIN_GROUP_POSTS} past bullish/bearish posts by {author} with price data."
    mean = history.mean_signed_car
    way = "in the direction" if mean >= 0 else "against the direction"
    if history.p_holm is None:
        stats = f"n={history.n_posts}, no p-value"
    else:
        verdict = "significant at the 5% level" if history.p_holm < 0.05 else "not significant"
        stats = f"n={history.n_posts}, Holm p={history.p_holm:.2f}, {verdict}"
    return (
        f"History: after {author}'s bullish/bearish posts, the stocks moved {abs(mean) * 100:.2f}% {way} the post "
        f"pointed, beyond what SPY explains, on average over 2 days ({stats})."
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
    lines += [*_left_out(post, len(rows)), window_text, FOOTER]
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
        verdict = "within the normal range" if abs(r.z) < Z_CRITICAL else f"unusually large (|z| = {abs(r.z):.2f})"
        line = f"{r.ticker} CAR[0,+1] {_pct(r.car)} (z {r.z:+.2f}) — {verdict}"
        if r.notes:
            line += "; " + "; ".join(r.notes)
        lines.append(line)
    lines += [*_left_out(post, len(rows)), FOOTER]
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
        lines.append(_more(total - DIGEST_POSTS))
    return Message(
        title=f"While you were away: {total} stock post{'' if total == 1 else 's'}",
        body=fit_body("\n".join(lines)),
        priority=2,
        tags=("mailbox_with_mail",),
    )
