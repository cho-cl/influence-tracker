"""Write one event study to a folder: report.html (self-contained), CSVs for Excel and PNG charts.

The report only reads the Study (see study.py) plus raw price bars for the individual top-event charts. It never
recomputes a statistic, so the HTML, the CSVs and the CLI headline always show the same numbers.
"""

from __future__ import annotations

import base64
import csv
import html
import math
import re
import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, fields
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .. import market
from ..eventsview import _excel_id, _excel_safe
from ..timeutil import NY
from . import charts
from .charts import as_date, as_utc, format_ny, is_number
from .study import (
    DAILY_WINDOWS,
    ESTIMATION_WINDOW,
    EXCLUSION_REASONS,
    INTRADAY_PATH_MINUTES,
    INTRADAY_SIGMA_SESSIONS,
    MIN_ESTIMATION_OBS,
    MIN_GROUP_POSTS,
    MIN_INTRADAY_SIGMA_OBS,
    PATH_DAYS,
    PLACEBO_CLEARANCE,
    PLACEBO_RANGE,
    REDDIT_CATEGORY,
    Z_CRITICAL,
    Study,
)

REPORT_NAME = "report.html"
CHARTS_DIR = "charts"
TOP_DIR = "top_events"
CSV_FILES: dict[str, str] = {
    "events.csv": "events",
    "posts.csv": "posts",
    "summary.csv": "groups",
    "placebo.csv": "magnitude",
}
CHART_FILES: dict[str, str] = {
    "car_path": "car_path.png",
    "author": "author_effect.png",
    "placebo": "event_vs_placebo.png",
    "intraday": "intraday_path.png",
}
MARKET = "SPY"  # the benchmark of every market model
TOP_EVENTS = 10
EXCERPT_CHARS = 280
MAX_POLITICIAN_ROWS = 25
MAX_ATTENTION_ROWS = 15
# An intraday top-event chart needs this many 1-minute bars in its window, and this many after the post.
MIN_WINDOW_BARS = 5
MIN_BARS_AFTER = 2
DASH = "—"

FAMILIES: tuple[tuple[str, str], ...] = (
    ("all", "All signed posts"),
    ("author", "By author"),
    ("category", "By category"),
    ("platform", "By platform"),
    ("stance", "By stance"),
)
WINDOW_LABELS = {
    "pre": "pre, CAR[−5,−1]",
    "event": "event, CAR[0,+1]",
    "post": "post, CAR[+2,+5]",
}
EXCLUSION_TEXT = {
    "earnings": "an earnings announcement within 1 session of the post day",
    "split": "a stock split within 5 sessions of the post day",
    "clustered": "not the first post about this ticker in that session",
    "no_model": f"fewer than {MIN_ESTIMATION_OBS} trading days to fit the market model",
}
STANCE_ORDER = {"bullish": 0, "bearish": 1, "neutral": 2}


# ---------------------------------------------------------------- number formatting (HTML and CLI)


def fmt_pct(value: object, decimals: int = 2) -> str:
    """'+1.23%' / '-0.50%' / '0.00%'; an em dash when missing."""
    if not is_number(value):
        return DASH
    v = float(value)  # type: ignore[arg-type]
    text = f"{abs(v) * 100:.{decimals}f}%"
    if float(text[:-1]) == 0:
        return text
    return ("+" if v > 0 else "-") + text


def fmt_share(value: object) -> str:
    return f"{float(value) * 100:.1f}%" if is_number(value) else DASH  # type: ignore[arg-type]


def fmt_p(value: object) -> str:
    if not is_number(value):
        return DASH
    v = float(value)  # type: ignore[arg-type]
    return "<0.001" if v < 0.001 else f"{v:.3f}"


def fmt_z(value: object) -> str:
    return f"{float(value):+.2f}" if is_number(value) else DASH  # type: ignore[arg-type]


def fmt_num(value: object, decimals: int = 2) -> str:
    return f"{float(value):.{decimals}f}" if is_number(value) else DASH  # type: ignore[arg-type]


def fmt_int(value: object) -> str:
    return f"{int(value):,}" if is_number(value) else DASH  # type: ignore[arg-type]


def fmt_ci(low: object, high: object) -> str:
    if not (is_number(low) and is_number(high)):
        return DASH
    return f"{fmt_pct(low)} to {fmt_pct(high)}"


def fmt_day(value: object) -> str:
    """A session date with its weekday, 'Tue Sep 8, 2026'; an em dash when missing."""
    d = as_date(value)
    return f"{d:%a} {d:%b} {d.day}, {d:%Y}" if d else DASH


def _p_phrase(value: object) -> str:
    text = fmt_p(value)
    return f"p < {text[1:]}" if text.startswith("<") else f"p = {text}"


def _minus(k: int) -> str:
    return str(k).replace("-", "−")


def _plural(n: int, noun: str, plural: str | None = None) -> str:
    return f"{n:,} {noun if n == 1 else (plural or noun + 's')}"


# ---------------------------------------------------------------- summary numbers


@dataclass(frozen=True)
class Headline:
    n_posts: int
    n_complete: int
    n_included: int
    excluded: dict[str, int]
    overall: pd.Series | None  # the family='all' row, main subset, event window
    events_share: tuple[float, int] | None  # (share |z| > crit, n) for real events, event window
    placebo_share: tuple[float, int] | None
    n_intraday: int


def _included(events: pd.DataFrame) -> pd.DataFrame:
    reason = events["excluded_reason"].fillna("").astype(str)
    return events[reason == ""]


def _count(study: Study, key: str, derived: int) -> int:
    value = study.counts.get(key)
    return int(value) if is_number(value) else derived


def _magnitude_cell(magnitude: pd.DataFrame, window: str, sample: str) -> tuple[float, int] | None:
    rows = magnitude[(magnitude["window"] == window) & (magnitude["sample"] == sample)]
    if rows.empty:
        return None
    row = rows.iloc[0]
    if not (is_number(row["share_abs_z_gt_crit"]) and is_number(row["n"]) and row["n"] > 0):
        return None
    return float(row["share_abs_z_gt_crit"]), int(row["n"])


def summarize(study: Study) -> Headline:
    events = study.events
    reason = events["excluded_reason"].fillna("").astype(str)
    included = int((reason == "").sum())
    excluded = {r: int((reason == r).sum()) for r in EXCLUSION_REASONS}
    for r in sorted(set(reason) - {"", *EXCLUSION_REASONS}):
        excluded[r] = int((reason == r).sum())
    overall_rows = study.groups[
        (study.groups["family"] == "all") & (study.groups["subset"] == "main") & (study.groups["window"] == "event")
    ]
    intraday_state = _included(events)["intraday_state"]
    return Headline(
        n_posts=_count(study, "posts_included", len(study.posts)),
        n_complete=_count(study, "events_complete", len(events)),
        n_included=_count(study, "events_included", included),
        excluded=excluded,
        overall=overall_rows.iloc[0] if not overall_rows.empty else None,
        events_share=_magnitude_cell(study.magnitude, "event", "events"),
        placebo_share=_magnitude_cell(study.magnitude, "event", "placebo"),
        n_intraday=_count(study, "events_intraday", int((intraday_state == "ok").sum())),
    )


def _overall_ok(h: Headline) -> bool:
    return h.overall is not None and h.overall["status"] == "ok" and is_number(h.overall["mean_signed_car"])


def _overall_n(h: Headline) -> int:
    return int(h.overall["n_posts"]) if h.overall is not None and is_number(h.overall["n_posts"]) else 0


def _share_phrase(cell: tuple[float, int] | None, noun: str) -> str:
    if cell is None:
        return f"no {noun}"
    share, n = cell
    return f"{fmt_share(share)} of {noun} ({round(share * n)} of {n})"


def headlines(study: Study) -> list[str]:
    """Three plain lines for the terminal: posts analyzed, the signed CAR[0,+1] test, event vs placebo |z|."""
    h = summarize(study)
    excluded = sum(h.excluded.values())
    lines = [
        f"Posts analyzed: {h.n_posts:,} ({_plural(h.n_included, 'event')} included, {excluded:,} excluded)",
    ]
    if _overall_ok(h):
        o = h.overall
        assert o is not None
        lines.append(
            f"Mean signed CAR[0,+1]: {fmt_pct(o['mean_signed_car'])} (95% CI {fmt_ci(o['ci_low'], o['ci_high'])}, "
            f"{_p_phrase(o['p_value'])}, n = {_overall_n(h)} posts)"
        )
    else:
        lines.append(
            f"Mean signed CAR[0,+1]: insufficient data (n = {_overall_n(h)} signed posts; the test needs "
            f"{MIN_GROUP_POSTS})"
        )
    if h.events_share is None:
        lines.append(f"|z| > {Z_CRITICAL:g} in the event window: insufficient data (no events with a z-score)")
    else:
        lines.append(
            f"|z| > {Z_CRITICAL:g} in the event window: {_share_phrase(h.events_share, 'events')} vs "
            f"{_share_phrase(h.placebo_share, 'placebo days')}"
        )
    return lines


# ---------------------------------------------------------------- CSV


def _csv_value(value: object) -> object:
    if value is None or value is pd.NA or value is pd.NaT:
        return ""
    if isinstance(value, bool | np.bool_):
        return bool(value)
    if isinstance(value, int | np.integer):
        return int(value)
    if isinstance(value, float | np.floating):
        return float(value) if math.isfinite(value) else ""
    if isinstance(value, pd.Timestamp):
        return value.isoformat(sep=" ")
    if isinstance(value, datetime):
        return value.isoformat(sep=" ")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, str):
        return _excel_safe(value)
    return _excel_safe(str(value))


def write_csv(frame: pd.DataFrame, path: Path) -> int:
    """utf-8-sig for Excel, text cells guarded against formula injection, native ids kept as exact text."""
    columns = list(frame.columns)
    id_col = columns.index("native_id") if "native_id" in columns else None
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for values in frame.itertuples(index=False, name=None):
            cells = [_csv_value(v) for v in values]
            if id_col is not None and isinstance(values[id_col], str):
                cells[id_col] = _excel_id(values[id_col])
            writer.writerow(cells)
    return len(frame)


# ---------------------------------------------------------------- prices for the top-event charts


def _minute_changes(
    conn: sqlite3.Connection, symbol: str, t0: datetime, d0: date, lo: datetime, hi: datetime
) -> pd.Series | None:
    """% change from the price as of t0, indexed by minutes from t0 at each bar's END (when its close is known).
    The price as of t0 follows events.MinuteBars.at: the close of the last bar starting at or before t0 - 60 s,
    searched back to the previous session's extended open."""
    cutoff = math.floor(t0.timestamp()) - 60
    floor_ts = int(market.extended_bounds_utc(market.previous_session(d0))[0].timestamp())
    ref = conn.execute(
        """SELECT close FROM bars_1m WHERE symbol = ? AND ts <= ? AND ts >= ? AND close > 0
           ORDER BY ts DESC LIMIT 1""",
        (symbol, cutoff, floor_ts),
    ).fetchone()
    if ref is None:
        return None
    rows = conn.execute(
        "SELECT ts, close FROM bars_1m WHERE symbol = ? AND ts >= ? AND ts <= ? AND close > 0 ORDER BY ts",
        (symbol, int(lo.timestamp()) - 60, int(hi.timestamp()) - 60),
    ).fetchall()
    base = float(ref[0])
    minutes = [(ts + 60 - t0.timestamp()) / 60 for ts, _ in rows]
    return pd.Series([close / base - 1 for _, close in rows], index=minutes, dtype=float)


def _intraday_prices(conn: sqlite3.Connection, ticker: str, t0: datetime, d0: date) -> charts.EventPrices | None:
    day = t0.astimezone(NY).date()
    try:
        if not market.is_session(day):
            return None
        ext_open, ext_close = market.extended_bounds_utc(day)
    except ValueError:
        return None
    lo = max(t0 - charts.INTRADAY_BEFORE, ext_open)
    hi = min(t0 + charts.INTRADAY_AFTER, ext_close)
    if hi <= t0:
        return None
    series = _minute_changes(conn, ticker, t0, d0, lo, hi)
    if series is None or len(series) < MIN_WINDOW_BARS or int((series.index > 0).sum()) < MIN_BARS_AFTER:
        return None
    spy = _minute_changes(conn, MARKET, t0, d0, lo, hi)
    open_, close = market.session_bounds_utc(day)
    regular = ((open_ - t0).total_seconds() / 60, (close - t0).total_seconds() / 60)
    return charts.EventPrices("intraday", series, spy if spy is not None else pd.Series(dtype=float), t0, regular)


def _daily_changes(conn: sqlite3.Connection, symbol: str, sessions: list[date], d0: date) -> pd.Series | None:
    offsets = {s.isoformat(): i - sessions.index(d0) for i, s in enumerate(sessions)}
    rows = conn.execute(
        """SELECT session_date, adj_close FROM bars_1d WHERE symbol = ? AND session_date BETWEEN ? AND ?
           AND adj_close > 0""",
        (symbol, sessions[0].isoformat(), sessions[-1].isoformat()),
    ).fetchall()
    closes = {offsets[d]: float(c) for d, c in rows if d in offsets}
    base = closes.get(-charts.DAILY_SPAN)
    if base is None:
        return None
    return pd.Series({k: closes[k] / base - 1 for k in sorted(closes)}, dtype=float)


def _daily_prices(conn: sqlite3.Connection, ticker: str, t0: datetime, d0: date) -> charts.EventPrices | None:
    try:
        sessions = market.sessions_in_range(
            market.session_offset(d0, -charts.DAILY_SPAN), market.session_offset(d0, charts.DAILY_SPAN)
        )
    except ValueError:
        return None
    series = _daily_changes(conn, ticker, sessions, d0)
    if series is None or 0 not in series.index or len(series) < 3:
        return None
    spy = _daily_changes(conn, MARKET, sessions, d0)
    return charts.EventPrices("daily", series, spy if spy is not None else pd.Series(dtype=float), t0)


def load_event_prices(conn: sqlite3.Connection, event: pd.Series) -> charts.EventPrices | None:
    """1-minute prices from 60 minutes before to 120 minutes after the post (clipped to that day's extended
    session) when stored; otherwise daily closes from d0-10 to d0+10; None when neither exists."""
    t0, d0 = as_utc(event["t0"]), as_date(event["d0"])
    if t0 is None or d0 is None:
        return None
    ticker = str(event["ticker"])
    try:
        return _intraday_prices(conn, ticker, t0, d0) or _daily_prices(conn, ticker, t0, d0)
    except ValueError:
        return None


# ---------------------------------------------------------------- HTML building blocks


def esc(value: object) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _safe_url(url: object) -> str | None:
    if isinstance(url, str) and re.match(r"https?://", url.strip(), re.IGNORECASE):
        return url.strip()
    return None


def _excerpt(text: object) -> str:
    if not isinstance(text, str):
        return ""
    text = text.strip()
    return text if len(text) <= EXCERPT_CHARS else text[: EXCERPT_CHARS - 1].rstrip() + "…"


@dataclass(frozen=True)
class Cell:
    html: str
    numeric: bool = False
    span: int = 1


def _num(text: str) -> Cell:
    return Cell(esc(text), numeric=True)


def _txt(text: object) -> Cell:
    return Cell(esc(text))


def _table(headers: Sequence[tuple[str, bool]], bodies: Iterable[str | Sequence[Cell]]) -> str:
    """headers: (label, numeric). bodies: rows of cells, or prebuilt <tr> strings."""
    head = "".join(f'<th scope="col"{" class=r" if numeric else ""}>{esc(label)}</th>' for label, numeric in headers)
    body = "".join(b if isinstance(b, str) else _row(b) for b in bodies)
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def _row(cells: Sequence[Cell], cls: str = "") -> str:
    attr = f' class="{cls}"' if cls else ""
    tds = "".join(
        f"<td{' class=r' if c.numeric else ''}{f' colspan={c.span}' if c.span > 1 else ''}>{c.html}</td>" for c in cells
    )
    return f"<tr{attr}>{tds}</tr>"


def _img(path: Path | None, alt: str) -> str:
    if path is None:
        return ""
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f'<img src="data:image/png;base64,{data}" alt="{esc(alt)}">'


def _figure(path: Path | None, alt: str, caption: str = "") -> str:
    if path is None:
        return ""
    cap = f"<figcaption>{caption}</figcaption>" if caption else ""
    return f"<figure>{_img(path, alt)}{cap}</figure>"


def _dot(stance: object) -> str:
    color = charts.stance_color(stance)
    return f'<span class="dot" style="background:{color}" aria-hidden="true"></span>'


# ---------------------------------------------------------------- sections


def _date_range(study: Study) -> str:
    opts = study.options
    days = [d for d in (as_date(v) for v in _included(study.events)["d0"]) if d is not None]
    parts = []
    if days:
        parts.append(
            f"post days {min(days):%b} {min(days).day}, {min(days):%Y} to {max(days):%b} {max(days).day}, "
            f"{max(days):%Y}"
        )
    else:
        parts.append("no included events")
    if opts.since or opts.until:
        since = opts.since.isoformat() if opts.since else "the start"
        until = opts.until.isoformat() if opts.until else "the latest complete event"
        parts.append(f"filtered to post days from {since} to {until}")
    return "; ".join(parts)


def _generated(study: Study) -> str:
    at = study.generated_at if study.generated_at.tzinfo else study.generated_at.replace(tzinfo=UTC)
    return format_ny(at)


def _section_title(study: Study, h: Headline) -> str:
    excluded = ", ".join(f"{n} {reason.replace('_', ' ')}" for reason, n in h.excluded.items() if n) or "none"
    coverage = (
        f"{_plural(h.n_posts, 'post')} analyzed · {h.n_included:,} of {h.n_complete:,} complete events included "
        f"· excluded: {excluded}"
    )
    return (
        "<header>"
        '<p class="kicker">Event study · exploratory</p>'
        "<h1>How social-media posts moved stocks</h1>"
        f'<p class="meta">Generated {esc(_generated(study))} · {esc(_date_range(study))}</p>'
        f'<p class="meta">{esc(coverage)}</p>'
        "</header>"
    )


def _tile(label: str, value: str, sub: str, small: bool = False) -> str:
    cls = "value small" if small else "value"
    return (
        f'<div class="tile"><div class="label">{esc(label)}</div><div class="{cls}">{esc(value)}</div>'
        f'<div class="sub">{esc(sub)}</div></div>'
    )


def _section_tiles(study: Study, h: Headline) -> str:
    signed = _count(study, "posts_signed", _overall_n(h))
    tiles = [_tile("Posts analyzed", f"{h.n_posts:,}", f"{signed:,} bullish or bearish · {h.n_included:,} events")]
    if _overall_ok(h):
        o = h.overall
        assert o is not None
        tiles.append(
            _tile(
                "Mean signed CAR[0,+1]",
                fmt_pct(o["mean_signed_car"]),
                f"95% CI {fmt_ci(o['ci_low'], o['ci_high'])} · {_p_phrase(o['p_value'])} · n = {_overall_n(h)} posts",
            )
        )
    else:
        tiles.append(
            _tile(
                "Mean signed CAR[0,+1]",
                f"insufficient data (n < {MIN_GROUP_POSTS})",
                f"{_plural(_overall_n(h), 'signed post')} so far; the test needs at least {MIN_GROUP_POSTS}",
                small=True,
            )
        )
    if h.events_share is not None:
        pl = fmt_share(h.placebo_share[0]) if h.placebo_share else DASH
        tiles.append(
            _tile(
                f"Events with |z| > {Z_CRITICAL:g}",
                f"{fmt_share(h.events_share[0])} vs {pl}",
                f"real events vs placebo days, event window · {h.events_share[1]} events, "
                f"{h.placebo_share[1] if h.placebo_share else 0} placebo days",
            )
        )
    else:
        tiles.append(
            _tile(f"Events with |z| > {Z_CRITICAL:g}", "insufficient data", "no event has a z-score yet", small=True)
        )
    tiles.append(
        _tile(
            "Events with 1-minute data",
            f"{h.n_intraday:,} of {h.n_included:,}",
            "included events whose day 0 can be split at the post",
        )
    )
    return f'<section aria-label="Key numbers"><div class="tiles">{"".join(tiles)}</div></section>'


def _section_method(study: Study) -> str:
    est_lo, est_hi = ESTIMATION_WINDOW
    est_len = est_hi - est_lo + 1
    pl_lo, pl_hi = PLACEBO_RANGE
    opts = study.options
    windows = {
        name: f"days {lo:+d} to {hi:+d}".replace("+0", "0").replace("-", "−")
        for name, (lo, hi) in DAILY_WINDOWS.items()
    }
    items = [
        (
            "Event",
            "Each time a post mentions a watchlist stock, that (post, stock) pair is one event. Day 0 is the "
            "first trading session whose close comes after the post, so a post made after the 4 PM close, overnight or "
            "on a weekend counts toward the next session.",
        ),
        (
            "Normal behaviour: the market model",
            f"For each event the stock's daily return is regressed on SPY's: "
            f"r_stock = α + β × r_SPY, fitted on the {est_len} trading days from day {_minus(est_lo)} to day "
            f"{_minus(est_hi)}, so the estimate ends {-est_hi} trading days before the post and the post cannot "
            f"leak into it. β is how strongly the stock usually follows the market; σ is the typical size of a day's "
            f"unexplained move. Events with fewer than {MIN_ESTIMATION_OBS} usable days are left out.",
        ),
        (
            "Abnormal return",
            "Abnormal return = actual return − (α + β × SPY's return that "
            "day): the part of the move the market does not explain. A cumulative abnormal return (CAR) adds these "
            "up over a window.",
        ),
        (
            "Three windows",
            f"pre = {windows['pre']}: did the stock already move before the post (a sign the author "
            f"was reacting to news)? event = {windows['event']}: the headline window. post = {windows['post']}: did "
            "the move continue or reverse afterwards?",
        ),
        (
            "z-scores",
            f"z = CAR / (σ × √L), where L is the number of days in the window. If abnormal "
            f"returns were normally distributed, |z| > {Z_CRITICAL:g} would happen on about 5% of ordinary days.",
        ),
        (
            "Signing by stance",
            "A sentiment model labels each post bullish, bearish or neutral. The signed CAR is "
            "+CAR for bullish posts and −CAR for bearish ones, so a positive number always means the stock moved "
            "the way the post pointed. Neutral posts have no direction and are left out of the signed tests.",
        ),
        (
            "Testing at the post level",
            f"A post that mentions three stocks is one observation: its events are "
            "averaged first. For a post that mentions several stocks, the z-score treats them as one equal-weighted "
            "portfolio (the average CAR measured against the σ of the stocks' average daily abnormal return over the "
            "same estimation days), so stocks that usually move together are not counted as independent evidence. "
            f"Each group (author, category, platform, stance) gets a t-test of whether its mean signed "
            f"CAR differs from zero, with a 95% confidence interval. Groups with fewer than {MIN_GROUP_POSTS} posts "
            "are shown but not tested.",
        ),
        (
            "Holm correction",
            "Testing many groups at once makes a lucky 'significant' result likely. The Holm "
            "correction raises each p-value to account for the number of groups tested in the same family (the "
            "authors, the categories, the platforms or the stances), within one window and one set of posts (all "
            "labelled posts, or only confident labels): that is the Holm p. It does not correct across families, "
            "windows or sets of posts.",
        ),
        (
            "Placebo comparison",
            f"For every included event, {opts.placebo_draws} random trading days of the same stock are drawn from "
            f"{-pl_lo} to {-pl_hi} trading days before its post, each at least {PLACEBO_CLEARANCE} sessions away "
            "from any post about that ticker, and put through exactly the same calculation. The draws also skip "
            "days within 1 session of an earnings report and within 5 sessions of a stock split (unless "
            "--include-earnings or --include-splits is used), the same screens real events get, and a stock-day "
            "drawn for two events counts once. If posts matter, real "
            "events should show large |z| more often than these ordinary days. This measures what ordinary days "
            "actually look like instead of assuming the textbook 5%.",
        ),
        (
            "Exclusions",
            "By default the study leaves out events with an earnings announcement within 1 session of "
            "day 0 (earnings news would swamp the post), events with a stock split within 5 sessions (prices jump "
            "mechanically), and repeat posts about the same ticker in the same session (only the first counts). The "
            "flags --include-earnings, --include-splits and --include-clustered put them back. Events whose earnings "
            "date could not be looked up stay in and are counted in the appendix.",
        ),
        (
            "Day-0 timing",
            "Day 0 runs from the previous close to the day-0 close. For a post made during market "
            "hours, part of day 0 happened before the post, so the day-0 return mixes the move before the post with "
            "the move after it. Where 1-minute prices exist, day 0 is split at the post into a pre-leg (previous "
            "close to the post) and a post-leg (post to the close), and the intraday section measures the minutes "
            f"around it against the same clock-time window over the prior {INTRADAY_SIGMA_SESSIONS} sessions (at "
            f"least {MIN_INTRADAY_SIGMA_OBS} needed).",
        ),
    ]
    body = "".join(f"<dt>{esc(term)}</dt><dd>{esc(text)}</dd>" for term, text in items)
    return f'<section id="method"><h2>How this works</h2><dl class="method">{body}</dl></section>'


def _path_table(frame: pd.DataFrame, x_col: str, y_col: str, xs: Sequence[int], x_label: str) -> str:
    present = [s for s in ("bullish", "bearish", "neutral", "placebo") if (frame["series"] == s).any()]
    if not present:
        return ""
    headers: list[tuple[str, bool]] = [(x_label, True)]
    for series in present:
        rows = frame[frame["series"] == series]
        ns = pd.to_numeric(rows["n"], errors="coerce").dropna().astype(int)
        n_text = "" if ns.empty else (f"n={ns.min()}" if ns.min() == ns.max() else f"n={ns.min()}–{ns.max()}")
        name = "placebo days" if series == "placebo" else f"{series} posts"
        headers += [(f"{name} ({n_text})" if n_text else name, True), ("95% CI", True)]
    body = []
    for x in xs:
        cells = [_num(f"{x:+d}" if x else "0")]
        for series in present:
            match = frame[(frame["series"] == series) & (pd.to_numeric(frame[x_col], errors="coerce") == x)]
            if match.empty:
                cells += [_num(DASH), _num(DASH)]
                continue
            r = match.iloc[0]
            cells += [_num(fmt_pct(r[y_col])), _num(fmt_ci(r["ci_low"], r["ci_high"]))]
        body.append(cells)
    return _table(headers, body)


def _section_car_path(study: Study, figs: dict[str, Path | None]) -> str:
    chart = _figure(
        figs["car_path"],
        "Line chart of the mean cumulative abnormal return from day -5 to day +5 for bullish and bearish posts, "
        "with placebo days in gray",
    )
    note = (
        ""
        if chart
        else '<p class="note">No bullish or bearish post has a complete day −5 to +5 path yet, so '
        "there is no chart. Any rows below come from neutral posts or placebo days.</p>"
    )
    table = _path_table(study.car_path, "day", "mean_car", PATH_DAYS, "Day")
    return (
        '<section id="car-path"><h2>The average path around a post</h2>'
        "<p>Cumulative abnormal return from day −5 through each day, averaged over posts. Unsigned: a bearish "
        "post that worked shows a falling line. Neutral posts are in the table only.</p>"
        f"{chart}{note}{table}</section>"
    )


def _group_rows(sel: pd.DataFrame) -> list[str]:
    rows: list[str] = []
    for family, label in FAMILIES:
        fam = sel[sel["family"] == family]
        if fam.empty:
            continue
        if family == "stance":
            fam = fam.assign(_o=fam["group"].map(lambda g: STANCE_ORDER.get(str(g), 9))).sort_values(["_o", "group"])
        else:
            fam = fam.sort_values(["n_posts", "group"], ascending=[False, True], kind="stable")
        # The overall row needs no section heading of its own.
        if family != "all":
            rows.append(f'<tr class="fam"><th colspan="{len(_GROUP_HEADERS)}" scope="colgroup">{esc(label)}</th></tr>')
        for _, r in fam.iterrows():
            ok = r["status"] == "ok"
            name = label if family == "all" else str(r["group"])
            if family == "category" and name == REDDIT_CATEGORY:
                name = f"{name} (Reddit)"
            cells = [
                _txt(name),
                _num(fmt_int(r["n_posts"])),
                _num(fmt_pct(r["mean_signed_car"])),
                _num(fmt_pct(r["median_signed_car"])),
                _num(fmt_ci(r["ci_low"], r["ci_high"])),
                _num(fmt_num(r["mean_abs_z"])),
                _num(fmt_share(r["share_abs_z_gt_crit"])),
            ]
            if ok:
                cells += [_num(fmt_p(r["p_value"])), _num(fmt_p(r["p_holm"]))]
            else:
                # The reason stands where the p-values would be.
                unsigned = family == "stance" and str(r["group"]) == "neutral"
                note = "neutral posts are not signed" if unsigned else f"n<{MIN_GROUP_POSTS}: not tested"
                cells.append(Cell(esc(note), numeric=True, span=2))
            classes = ("total " if family == "all" else "") + ("" if ok else "muted")
            rows.append(_row(cells, classes.strip()))
    return rows


_GROUP_HEADERS: list[tuple[str, bool]] = [
    ("Group", False),
    ("Posts", True),
    ("Mean signed CAR", True),
    ("Median", True),
    ("95% CI", True),
    ("Mean |z|", True),
    (f"|z| > {Z_CRITICAL:g}", True),
    ("p", True),
    ("Holm p", True),
]


def _groups_table(groups: pd.DataFrame, subset: str, window: str) -> str:
    sel = groups[(groups["subset"] == subset) & (groups["window"] == window)]
    if sel.empty:
        return '<p class="note">No posts in this subset.</p>'
    return _table(_GROUP_HEADERS, _group_rows(sel))


def _section_groups(study: Study, figs: dict[str, Path | None]) -> str:
    groups = study.groups
    rows = charts.author_rows(groups)
    notes = []
    if not rows.folded.empty:
        folded_posts = int(pd.to_numeric(rows.folded["n_posts"], errors="coerce").fillna(0).sum())
        notes.append(
            f"The chart shows the {len(rows.shown)} authors with the most posts; {len(rows.folded)} more "
            f"({_plural(folded_posts, 'post')}) are in the table."
        )
    if not rows.undrawable.empty:
        notes.append(f"{_plural(len(rows.undrawable), 'author')} with no bullish or bearish post not drawn.")
    chart = _figure(
        figs["author"],
        "Horizontal bar chart of the mean signed CAR[0,+1] per author with 95% confidence interval whiskers",
        esc(" ".join(notes)),
    )
    conf = study.options.min_conf_robust
    details = [
        (
            f"Robustness: CAR[0,+1] for posts the sentiment model labelled with confidence ≥ {conf:g}",
            _groups_table(groups, "confident", "event"),
        ),
        ("Before the post: CAR[−5,−1]", _groups_table(groups, "main", "pre")),
        ("After the event window: CAR[+2,+5]", _groups_table(groups, "main", "post")),
        (
            f"Robustness, before the post: CAR[−5,−1], confidence ≥ {conf:g}",
            _groups_table(groups, "confident", "pre"),
        ),
        (
            f"Robustness, after the event window: CAR[+2,+5], confidence ≥ {conf:g}",
            _groups_table(groups, "confident", "post"),
        ),
    ]
    extra = "".join(f"<details><summary>{esc(title)}</summary>{table}</details>" for title, table in details)
    return (
        '<section id="results"><h2>Results by author, category, platform and stance</h2>'
        f'<p><span class="tag">Exploratory</span> Mean signed CAR[0,+1] per group: positive means the stock moved '
        f"the way the posts pointed. Grey rows have fewer than {MIN_GROUP_POSTS} posts and are not tested; read "
        "their numbers as anecdotes. Holm p corrects for the number of groups tested in the same family (authors, "
        "categories, platforms or stances), not across the whole table.</p>"
        f"{chart}"
        f"<h3>Event window: CAR[0,+1], days 0 to +1</h3>{_groups_table(groups, 'main', 'event')}"
        f"{extra}</section>"
    )


def _section_placebo(study: Study, figs: dict[str, Path | None], h: Headline) -> str:
    chart = _figure(
        figs["placebo"],
        "Grouped bar chart of the share of real events and of placebo days with |z| above 1.96 per window",
    )
    mag = study.magnitude
    body = []
    for window in DAILY_WINDOWS:
        for sample, label in (("events", "real events"), ("placebo", "placebo days")):
            match = mag[(mag["window"] == window) & (mag["sample"] == sample)]
            if match.empty:
                continue
            r = match.iloc[0]
            body.append(
                [
                    _txt(WINDOW_LABELS[window]),
                    _txt(label),
                    _num(fmt_int(r["n"])),
                    _num(fmt_pct(r["mean_car"])),
                    _num(fmt_pct(r["mean_abs_car"])),
                    _num(fmt_pct(r["median_abs_car"])),
                    _num(fmt_share(r["share_abs_z_gt_crit"])),
                ]
            )
    headers = [
        ("Window", False),
        ("Sample", False),
        ("n", True),
        ("Mean CAR", True),
        ("Mean |CAR|", True),
        ("Median |CAR|", True),
        (f"|z| > {Z_CRITICAL:g}", True),
    ]
    table = _table(headers, body) if body else '<p class="note">No events or placebo days to compare yet.</p>'
    if h.events_share is not None:
        summary = (
            f"In the event window, {_share_phrase(h.events_share, 'real events')} had |z| > {Z_CRITICAL:g}, against "
            f"{_share_phrase(h.placebo_share, 'placebo days')}. No significance test is attached to this "
            "comparison; with samples this size, treat it as descriptive."
        )
    else:
        summary = "There are no events with a z-score yet, so there is nothing to compare."
    return (
        '<section id="placebo"><h2>Real events vs ordinary days</h2>'
        "<p>Unsigned size of the move, whatever the post said. The placebo days are the same tickers on random "
        "days without a post, so they show how often a stock makes a large move anyway.</p>"
        f"<p>{esc(summary)}</p>{chart}{table}</section>"
    )


def _section_intraday(study: Study, figs: dict[str, Path | None], h: Headline) -> str:
    chart = _figure(
        figs["intraday"],
        "Line chart of the mean abnormal return from 60 minutes before to 60 minutes after regular-session posts",
    )
    table = _path_table(study.intraday_path, "minute", "mean_ar", INTRADAY_PATH_MINUTES, "Minute")
    if chart:
        lead = (
            f"{_plural(h.n_intraday, 'included event')} have 1-minute prices. For posts made during regular trading "
            "hours, this is the abnormal return from 60 minutes before the post to 60 minutes after, relative to the "
            "price at the post (unsigned). Windows cut short by the close are included as measured."
        )
        return f'<section id="intraday"><h2>Minute by minute</h2><p>{esc(lead)}</p>{chart}{table}</section>'
    why = (
        "No included bullish or bearish post has intraday windows yet. They need a post made during regular trading "
        "hours (9:30 AM to 4 PM New York) while Yahoo still had its 1-minute prices (about 30 days), so most history-"
        f"backfilled posts only have daily data. {_plural(h.n_intraday, 'included event')} "
        f"{'has' if h.n_intraday == 1 else 'have'} 1-minute prices for the day-0 legs."
    )
    return f'<section id="intraday"><h2>Minute by minute</h2><p class="note">{esc(why)}</p>{table}</section>'


def _section_market(study: Study) -> str:
    events = _included(study.events)
    pol = events[events["category"] == "politician"]
    lead = (
        "Posts about tariffs, rates or the economy can move the whole market. The market model subtracts "
        "β × SPY's move, so a market-wide reaction shows up in SPY's raw return here, not in the stock's "
        "abnormal return. A large SPY move on day 0 means the abnormal return leaves out the market-wide part of "
        "the post's effect, and also that the stock's own move cannot be credited to the post alone. Day 0 is the "
        "first session whose close comes after the post, so for a post made after the close, on a weekend or on "
        "a holiday it is a later date than the post; the SPY and stock columns are for that session."
    )
    if pol.empty:
        return (
            '<section id="market"><h2>Market-wide moves</h2>'
            f'<p>{esc(lead)}</p><p class="note">No included events come from politician accounts.</p></section>'
        )
    order = pol["spy_ret_d0"].map(lambda v: abs(float(v)) if is_number(v) else -1.0)
    pol = pol.assign(_o=order).sort_values("_o", ascending=False, kind="stable")
    shown = pol.head(MAX_POLITICIAN_ROWS)
    body = []
    for _, r in shown.iterrows():
        t0 = as_utc(r["t0"])
        body.append(
            [
                _txt(format_ny(t0) if t0 else DASH),
                _txt(fmt_day(r["d0"])),
                _txt(r["author"]),
                _txt(r["ticker"]),
                _num(fmt_pct(r["spy_ret_d0"])),
                _num(fmt_pct(r["spy_post_leg"])),
                _num(fmt_num(r["beta"])),
                _num(fmt_pct(r["ar_d0"])),
                _num(fmt_pct(r["car_event"])),
            ]
        )
    headers = [
        ("Posted", False),
        ("Day 0", False),
        ("Author", False),
        ("Ticker", False),
        ("SPY on day 0 (raw)", True),
        ("SPY after the post (raw)", True),
        ("β", True),
        ("Stock AR day 0", True),
        ("Stock CAR[0,+1]", True),
    ]
    more = ""
    if len(pol) > len(shown):
        more = (
            f'<p class="note">Showing the {len(shown)} events with the largest SPY moves of {len(pol)}; '
            "all are in events.csv.</p>"
        )
    return f'<section id="market"><h2>Market-wide moves</h2><p>{esc(lead)}</p>{_table(headers, body)}{more}</section>'


def _section_attention(study: Study) -> str:
    att = study.attention
    days = _count(study, "attention_days", int(att["snapshot_date"].nunique()) if not att.empty else 0)
    lead = (
        "ApeWisdom counts how often each ticker is mentioned across Reddit. A spike is the day's mentions divided by "
        "the trailing 7-day mean; the table sets the biggest spikes next to the stock's abnormal return on the next "
        "session. Attention can follow a price move as easily as lead it."
    )
    if att.empty:
        note = (
            f"Not enough ApeWisdom history yet: {_plural(days, 'daily snapshot')} collected, and a spike needs a "
            "trailing week of snapshots plus a next session with prices. Snapshots cannot be backfilled, so this "
            "fills in as the daily run keeps collecting."
        )
        return (
            f'<section id="attention"><h2>Reddit attention</h2><p>{esc(lead)}</p>'
            f'<p class="note">{esc(note)}</p></section>'
        )
    order = att["spike"].map(lambda v: float(v) if is_number(v) else -1.0)
    top = att.assign(_o=order).sort_values("_o", ascending=False, kind="stable").head(MAX_ATTENTION_ROWS)
    body = [
        [
            _txt(r["ticker"]),
            _txt(as_date(r["snapshot_date"]) or r["snapshot_date"]),
            _num(fmt_int(r["mentions"])),
            _num(fmt_num(r["trailing_mean"], 1)),
            _num(f"{fmt_num(r['spike'], 1)}×" if is_number(r["spike"]) else DASH),
            _txt(as_date(r["next_session"]) or DASH),
            _num(fmt_pct(r["next_ar"])),
            _num(fmt_z(r["next_z"])),
        ]
        for _, r in top.iterrows()
    ]
    headers = [
        ("Ticker", False),
        ("Snapshot", False),
        ("Mentions", True),
        ("7-day mean", True),
        ("Spike", True),
        ("Next session", False),
        ("Next-session AR", True),
        ("z", True),
    ]
    note = f"{_plural(days, 'snapshot day')} collected; {len(att):,} ticker-days with a trailing mean."
    return (
        f'<section id="attention"><h2>Reddit attention</h2><p>{esc(lead)}</p>'
        f'{_table(headers, body)}<p class="note">{esc(note)}</p></section>'
    )


@dataclass(frozen=True)
class TopEvent:
    rank: int
    event: pd.Series
    chart: Path | None
    prices: charts.EventPrices | None


def _day0_text(d0: object, t0: datetime | None) -> str:
    day = as_date(d0)
    if day is None or t0 is None or day == t0.astimezone(NY).date():
        return fmt_day(d0)
    return f"{fmt_day(day)} (first session after the post)"


def _stray_note(prices: charts.EventPrices | None, ticker: str) -> str:
    """Says which single off-hours prints the chart leaves out of its lines (charts.isolated_prints)."""
    if prices is None or prices.kind != "intraday":
        return ""
    parts = []
    for name, series in ((ticker, prices.ticker.dropna()), (MARKET, prices.spy.dropna())):
        stray = series[charts.isolated_prints(series, prices.regular)]
        if not stray.empty:
            largest = stray.iloc[int(np.argmax(np.abs(stray.to_numpy(float))))]
            parts.append(f"{_plural(len(stray), f'{name} print')} (largest {fmt_pct(largest)})")
    if not parts:
        return ""
    return (
        f"Outside regular hours, {' and '.join(parts)} jumped away from the price and came straight back within "
        f"{charts.SPIKE_MAX_BARS} minutes. In thin trading these are most likely bad ticks, so the chart marks them "
        "with hollow circles (arrowheads at the edge when off the scale) and leaves them out of the line and the "
        "scale. The event's numbers use the stored bars unchanged."
    )


def _top_events(events: pd.DataFrame) -> pd.DataFrame:
    inc = _included(events)
    z = pd.to_numeric(inc["z_event"], errors="coerce")
    inc = inc.assign(_abs_z=z.abs())[z.notna() & np.isfinite(z)]
    return inc.sort_values(["_abs_z", "event_id"], ascending=[False, True], kind="stable").head(TOP_EVENTS)


def _section_top(top: list[TopEvent]) -> str:
    intro = (
        f"The {len(top)} included events with the largest |z| in the event window. They are the most extreme "
        "moves by construction, so they overstate the typical effect: use them to check the numbers by hand "
        "against a price chart, not as evidence on their own."
    )
    if not top:
        return (
            '<section id="top"><h2>The biggest moves</h2>'
            '<p class="note">No included event has an event-window z-score yet.</p></section>'
        )
    cards = []
    for item in top:
        e = item.event
        t0 = as_utc(e["t0"])
        url = _safe_url(e["url"])
        link = f'<a href="{esc(url)}" rel="noopener noreferrer">View the post</a>' if url else ""
        chart = (
            _img(item.chart, f"Price of {e['ticker']} around the post, with SPY for comparison")
            if item.chart
            else '<p class="note">No price bars stored around this post.</p>'
        )
        kind = item.prices.kind if item.chart and item.prices else None
        source = "1-minute prices" if kind == "intraday" else "daily closes" if kind == "daily" else ""
        conf = f" ({fmt_num(e['stance_conf'])})" if is_number(e["stance_conf"]) else ""
        facts = [
            ("Day 0", esc(_day0_text(e["d0"], t0))),
            (
                "Stance",
                f"{_dot(e['stance'])}{esc(e['stance'] if isinstance(e['stance'], str) else 'unlabelled')}{esc(conf)}",
            ),
            ("CAR[0,+1]", esc(f"{fmt_pct(e['car_event'])} (z {fmt_z(e['z_event'])})")),
            ("Before, CAR[−5,−1]", esc(f"{fmt_pct(e['car_pre'])} (z {fmt_z(e['z_pre'])})")),
            ("After, CAR[+2,+5]", esc(f"{fmt_pct(e['car_post'])} (z {fmt_z(e['z_post'])})")),
            (
                "Day 0 split at the post",
                esc(f"before {fmt_pct(e['ar_pre_leg'])}, after {fmt_pct(e['ar_post_leg'])} (abnormal)"),
            ),
        ]
        dl = "".join(f"<dt>{term}</dt><dd>{value}</dd>" for term, value in facts)
        title = f"{item.rank}. {e['ticker']} · {e['author']} · {format_ny(t0) if t0 else DASH}"
        stray = _stray_note(item.prices, str(e["ticker"])) if item.chart else ""
        if stray:
            chart += f'<p class="note">{esc(stray)}</p>'
        cards.append(
            '<article class="card">'
            f'<div class="card-chart">{chart}</div>'
            f'<div class="card-text"><h3>{esc(title)}</h3>'
            f'<p class="meta">{esc(e["platform"])} · event {esc(e["event_id"])}'
            f"{' · chart from ' + esc(source) if source else ''}</p>"
            f'<blockquote class="post">{esc(_excerpt(e["text"]))}</blockquote>'
            f'<dl class="facts">{dl}</dl>{link}</div></article>'
        )
    return f'<section id="top"><h2>The biggest moves</h2><p>{esc(intro)}</p>{"".join(cards)}</section>'


def _section_caveats(study: Study, h: Headline) -> str:
    n = _overall_n(h)
    items = [
        (
            "Correlation, not causation",
            "A move after a post does not show the post caused it. The same news can "
            "drive both the post and the price, and an author may post because a stock is already moving.",
        ),
        (
            "Small samples",
            f"The signed test rests on {_plural(n, 'post')}. Groups under {MIN_GROUP_POSTS} posts "
            "are not tested, and even tested groups have wide confidence intervals. Numbers will move as data "
            "accumulates.",
        ),
        (
            "Watchlist selection",
            "The accounts and tickers were chosen because they are known to talk about stocks "
            "or to move them. Results describe this watchlist, not social media in general.",
        ),
        (
            "Market-wide posts",
            "The market model removes whatever SPY did. A post that moved the whole market "
            "(tariffs, rates) shows up in SPY, not in the abnormal return; see the market-wide section.",
        ),
        (
            "Sentiment-model errors",
            "Stance comes from FinTwitBERT, trained on finance tweets. Political writing "
            "(all caps, sarcasm, praise and threats in one post) is outside its training data, and one post gets one "
            "label even when it is bullish on one company and bearish on another. A wrong label flips the sign of that "
            "post's signed CAR; the confident-labels table is the check.",
        ),
        (
            "Images and video",
            "Posts that are only an image, a video or a screenshot have no text to match, so they "
            "are invisible to this study.",
        ),
        (
            "Many comparisons",
            "Several windows, subsets and groupings are reported. Holm corrects only within one "
            "family of groups (authors, categories, platforms or stances) in one window and one set of posts, so "
            "each results table holds several separately corrected families. Looking across families, windows and "
            "sets of posts still makes a chance finding more likely. The whole results section is exploratory.",
        ),
    ]
    body = "".join(f"<li><strong>{esc(term)}.</strong> {esc(text)}</li>" for term, text in items)
    notes = "".join(f"<li>{esc(note)}</li>" for note in study.notes)
    data_notes = f"<h3>Data coverage notes</h3><ul>{notes}</ul>" if notes else ""
    return f'<section id="caveats"><h2>Caveats</h2><ul class="caveats">{body}</ul>{data_notes}</section>'


_FILE_NOTES = {
    REPORT_NAME: "this report (self-contained: charts are embedded)",
    "events.csv": "one row per complete event, included or excluded, with every metric and flag",
    "posts.csv": (
        "one row per analyzed post: CARs averaged over its events, z-scores of those events as one "
        "equal-weighted portfolio, signed CARs, text and link"
    ),
    "summary.csv": "every group test: family, group, subset, window, n, means, CI, p and Holm p",
    "placebo.csv": "real events vs placebo days per window: n, mean and median |CAR|, share of large |z|",
}


def _section_appendix(study: Study, h: Headline, files: list[str]) -> str:
    unknown = _count(study, "earnings_unknown", 0)
    excl_rows = [
        [_txt(reason.replace("_", " ")), _txt(EXCLUSION_TEXT.get(reason, "")), _num(fmt_int(n))]
        for reason, n in h.excluded.items()
    ]
    excl = _table([("Reason", False), ("Meaning", False), ("Events", True)], excl_rows)
    counts_rows = [[_txt(key.replace("_", " ")), _num(fmt_int(value))] for key, value in sorted(study.counts.items())]
    counts = _table([("Count", False), ("Value", True)], counts_rows) if counts_rows else ""
    file_rows = []
    for name in files:
        note = _FILE_NOTES.get(name)
        if note is None:
            note = "one of the 10 biggest moves" if f"/{TOP_DIR}/" in name else "chart shown in this report (PNG)"
        file_rows.append([Cell(f"<code>{esc(name)}</code>"), _txt(note)])
    file_table = _table([("File", False), ("Contents", False)], file_rows)
    settings = [
        [
            Cell(f"<code>{esc(f.name)}</code>"),
            _txt(getattr(study.options, f.name) if getattr(study.options, f.name) is not None else "not set"),
        ]
        for f in fields(study.options)
    ]
    setting_table = _table([("Setting", False), ("Value", False)], settings)
    return (
        '<section id="appendix"><h2>Appendix</h2>'
        f"<h3>Exclusions</h3>{excl}"
        f'<p class="note">{_plural(unknown, "included event")} had no earnings lookup and could not be checked for '
        "earnings.</p>"
        f"<h3>All counts</h3>{counts}"
        f"<h3>Files in this folder</h3>{file_table}"
        f"<h3>Settings</h3>{setting_table}</section>"
    )


_CSS = """
:root { --bg:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781; --rule:#e1e0d9; --axis:#c3c2b7;
  --tile:#f5f4f0; --link:#1d5fae; color-scheme: light; }
html { background: var(--bg); }
body { margin: 0; background: var(--bg); color: var(--ink);
  font: 15px/1.55 "Segoe UI", system-ui, -apple-system, "Helvetica Neue", Arial, sans-serif; }
main { max-width: 1000px; margin: 0 auto; padding: 40px 24px 72px; }
header { margin-bottom: 8px; }
.kicker { margin: 0 0 4px; font-size: 13px; letter-spacing: .04em; text-transform: uppercase; color: var(--ink2); }
h1 { font-size: 32px; line-height: 1.2; font-weight: 600; margin: 0 0 10px; letter-spacing: -.01em; }
h2 { font-size: 22px; line-height: 1.3; font-weight: 600; margin: 52px 0 10px; padding-top: 22px;
  border-top: 1px solid var(--rule); }
h3 { font-size: 16px; font-weight: 600; margin: 26px 0 8px; }
p, li, dd { max-width: 76ch; }
.meta { color: var(--ink2); margin: 2px 0; }
.note { color: var(--ink2); font-size: 14px; }
a { color: var(--link); text-underline-offset: 2px; }
code { font-family: Consolas, "Cascadia Mono", monospace; font-size: 13px; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; margin: 28px 0 8px; }
.tile { background: var(--tile); border: 1px solid var(--rule); border-radius: 8px; padding: 14px 16px; }
.tile .label { font-size: 13px; color: var(--ink2); }
.tile .value { font-size: 28px; line-height: 1.2; font-weight: 600; margin: 6px 0 6px; overflow-wrap: anywhere; }
.tile .value.small { font-size: 18px; }
.tile .sub { font-size: 13px; color: var(--ink2); }
dl.method { margin: 0; }
dl.method dt { font-weight: 600; margin-top: 14px; }
dl.method dd { margin: 2px 0 0; }
figure { margin: 20px 0 12px; }
figure img, .card-chart img { display: block; width: 100%; height: auto; border-radius: 4px; }
figcaption { color: var(--ink2); font-size: 13px; margin-top: 6px; }
.table-wrap { overflow-x: auto; margin: 10px 0 6px; }
table { border-collapse: collapse; width: 100%; font-size: 13.5px; font-variant-numeric: tabular-nums; }
th, td { padding: 6px 8px; border-bottom: 1px solid var(--rule); text-align: left; vertical-align: top; }
thead th { font-size: 12.5px; font-weight: 600; color: var(--ink2); border-bottom: 1px solid var(--axis);
  vertical-align: bottom; }
.r { text-align: right; }
td.r { white-space: nowrap; }
tr.fam th { padding-top: 16px; font-weight: 600; color: var(--ink); border-bottom: 1px solid var(--axis); }
tr.total td { font-weight: 600; }
tr.muted td { color: var(--muted); }
details { margin: 14px 0; border-top: 1px solid var(--rule); padding-top: 10px; }
summary { cursor: pointer; font-weight: 600; }
.tag { display: inline-block; font-size: 12px; font-weight: 600; padding: 1px 8px; border-radius: 999px;
  border: 1px solid var(--axis); color: var(--ink2); margin-right: 4px; }
.dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 6px; vertical-align: 0; }
.card { display: grid; grid-template-columns: minmax(0, 3fr) minmax(0, 2fr); gap: 20px; padding: 22px 0;
  border-top: 1px solid var(--rule); }
.card h3 { margin-top: 0; }
blockquote.post { margin: 10px 0; padding: 8px 12px; border-left: 3px solid var(--axis); color: var(--ink2);
  white-space: pre-wrap; overflow-wrap: anywhere; font-size: 14px; }
dl.facts { display: grid; grid-template-columns: auto 1fr; gap: 3px 12px; margin: 10px 0; font-size: 14px;
  font-variant-numeric: tabular-nums; }
dl.facts dt { color: var(--ink2); }
dl.facts dd { margin: 0; }
ul.caveats li { margin-bottom: 8px; }
@media (max-width: 760px) { .card { grid-template-columns: 1fr; } main { padding: 28px 16px 56px; } }
@media print {
  body { font-size: 10.5pt; }
  main { max-width: none; padding: 0; }
  h2 { break-after: avoid; }
  figure, .card, .tile, tr { break-inside: avoid; }
  /* A printed page cannot scroll: tables shrink and wrap at spaces so every column reaches the paper. */
  .table-wrap { overflow: visible; }
  table { font-size: 8pt; }
  th, td { padding: 3px 4px; }
  td.r { white-space: normal; }
  .card { grid-template-columns: 1fr 1fr; gap: 14px; padding: 12px 0; }
  .card h3 { font-size: 11pt; }
  .card-text, .card .note, blockquote.post, dl.facts { font-size: 8.5pt; }
  a { color: inherit; }
}
@page { margin: 14mm; }
"""

# Collapsed <details> would print empty; open them all for printing.
_PRINT_SCRIPT = (
    "<script>addEventListener('beforeprint',function(){"
    "document.querySelectorAll('details').forEach(function(d){d.open=true;});});</script>"
)


def _render(study: Study, figs: dict[str, Path | None], top: list[TopEvent], files: list[str]) -> str:
    h = summarize(study)
    sections = [
        _section_title(study, h),
        _section_tiles(study, h),
        _section_method(study),
        _section_car_path(study, figs),
        _section_groups(study, figs),
        _section_placebo(study, figs, h),
        _section_intraday(study, figs, h),
        _section_market(study),
        _section_attention(study),
        _section_top(top),
        _section_caveats(study, h),
        _section_appendix(study, h, files),
    ]
    return (
        '<!DOCTYPE html>\n<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<meta name="color-scheme" content="light">'
        "<title>How social-media posts moved stocks</title>"
        f"<style>{_CSS}</style></head><body><main>{''.join(sections)}</main>{_PRINT_SCRIPT}</body></html>\n"
    )


# ---------------------------------------------------------------- entry point


def _clear_old_charts(charts_dir: Path) -> None:
    """A rerun into the same folder must not leave last run's charts behind (they would be listed as current)."""
    for name in CHART_FILES.values():
        (charts_dir / name).unlink(missing_ok=True)
    top_dir = charts_dir / TOP_DIR
    if top_dir.is_dir():
        for png in top_dir.glob("*.png"):
            png.unlink()


def _chart_name(rank: int, event: pd.Series) -> str:
    ticker = re.sub(r"[^A-Za-z0-9.-]", "_", str(event["ticker"]))
    return f"{rank:02d}_{ticker}_{event['event_id']}.png"


def write_report(study: Study, conn: sqlite3.Connection, out_dir: Path) -> Path:
    """Write report.html, events/posts/summary/placebo CSVs and charts/*.png to out_dir; return the HTML path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    charts_dir = out_dir / CHARTS_DIR
    _clear_old_charts(charts_dir)

    for name, attr in CSV_FILES.items():
        write_csv(getattr(study, attr), out_dir / name)

    figs: dict[str, Path | None] = {
        "car_path": charts.car_path_chart(study.car_path, charts_dir / CHART_FILES["car_path"]),
        "author": charts.author_chart(study.groups, charts_dir / CHART_FILES["author"]),
        "placebo": charts.placebo_chart(study.magnitude, charts_dir / CHART_FILES["placebo"]),
        "intraday": charts.intraday_chart(study.intraday_path, charts_dir / CHART_FILES["intraday"]),
    }
    top: list[TopEvent] = []
    for rank, (_, event) in enumerate(_top_events(study.events).iterrows(), start=1):
        prices = load_event_prices(conn, event)
        path = charts.top_event_chart(event, prices, charts_dir / TOP_DIR / _chart_name(rank, event))
        top.append(TopEvent(rank, event, path, prices))

    written = [p for p in figs.values() if p is not None] + [t.chart for t in top if t.chart is not None]
    files = [REPORT_NAME, *CSV_FILES, *(p.relative_to(out_dir).as_posix() for p in written)]
    report = out_dir / REPORT_NAME
    report.write_text(_render(study, figs, top, files), encoding="utf-8")
    return report
