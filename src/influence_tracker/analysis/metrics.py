"""M3 event-study statistics: fills the Study contract (study.py) from the database.

Daily abnormal returns come from a market model on adjusted daily closes (bars_1d). The day-0 legs and intraday
windows come from the M2 1-minute windows in event_windows, so daily closes and 1-minute prices never meet inside
one return."""

from __future__ import annotations

import bisect
import logging
import math
import sqlite3
import warnings
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from .. import db, market
from ..config import Watchlist
from ..events import MARKET, SPLIT_WINDOW_SESSIONS, MinuteBars, earnings_session, window_specs
from ..timeutil import NY, from_iso
from .study import (
    ATTENTION_COLUMNS,
    DAILY_WINDOWS,
    ESTIMATION_WINDOW,
    EVENT_COLUMNS,
    EXCLUSION_REASONS,
    GROUP_COLUMNS,
    INTRADAY_PATH_COLUMNS,
    INTRADAY_PATH_MINUTES,
    INTRADAY_SIGMA_SESSIONS,
    INTRADAY_WINDOWS,
    MAGNITUDE_COLUMNS,
    MIN_ESTIMATION_OBS,
    MIN_GROUP_POSTS,
    MIN_INTRADAY_SIGMA_OBS,
    PATH_COLUMNS,
    PATH_DAYS,
    PLACEBO_CLEARANCE,
    PLACEBO_RANGE,
    POST_COLUMNS,
    REDDIT_CATEGORY,
    Z_CRITICAL,
    Study,
    StudyOptions,
)

log = logging.getLogger(__name__)

REL_VOLUME_WINDOW: tuple[int, int] = (-25, -6)
MIN_VOLUME_OBS = 10
ATTENTION_FILTER = "all-stocks"
ATTENTION_TRAILING_DAYS = 7
MIN_ATTENTION_TRAILING = 5
EARNINGS_RADIUS = 1  # events.earnings_flag: an earnings report first traded within one session of d0
PLACEBO_SEED_STRIDE = 1_000_003
FAMILIES: tuple[str, ...] = ("all", "author", "category", "platform", "stance")
SUBSETS: tuple[str, ...] = ("main", "confident")
SIGNED_STANCES: tuple[str, ...] = ("bullish", "bearish")
PATH_SERIES: tuple[str, ...] = ("bullish", "bearish", "neutral")
LEGS: tuple[str, ...] = ("pre_leg", "post_leg")
UNKNOWN_CATEGORY = "unknown"
_SIGNS = {"bullish": 1.0, "bearish": -1.0, "neutral": 0.0}
_FIT_CHUNK = 4096
DEFAULT_OPTIONS = StudyOptions()  # frozen, so one shared default is safe


def day_label(k: int) -> str:
    """Relative-day suffix used in column names: -5 -> 'dm5', 0 -> 'd0', 3 -> 'dp3'."""
    return "d0" if k == 0 else f"d{'m' if k < 0 else 'p'}{abs(k)}"


AR_COLUMNS: tuple[str, ...] = tuple(f"ar_{day_label(k)}" for k in PATH_DAYS)
CAR_COLUMNS: tuple[str, ...] = tuple(f"{kind}_{w}" for w in DAILY_WINDOWS for kind in ("car", "z"))
PLACEBO_COLUMNS: tuple[str, ...] = (
    "event_id", "ticker", "day", "alpha", "beta", "sigma", "n_est", *AR_COLUMNS, *CAR_COLUMNS,
)  # fmt: skip

_EST_OFFSETS = np.arange(ESTIMATION_WINDOW[0], ESTIMATION_WINDOW[1] + 1)
_PATH_OFFSETS = np.array(PATH_DAYS)
_VOL_OFFSETS = np.arange(REL_VOLUME_WINDOW[0], REL_VOLUME_WINDOW[1] + 1)
_NAN = float("nan")


# ---------------------------------------------------------------- small statistics helpers


def mean_ci(values: Sequence[float] | np.ndarray) -> tuple[float, float, float]:
    """(mean, low, high) of the finite values with a 95% t interval; the interval is NaN below two values."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return _NAN, _NAN, _NAN
    mean = float(v.mean())
    if len(v) < 2:
        return mean, _NAN, _NAN
    half = float(stats.t.ppf(0.975, len(v) - 1)) * float(v.std(ddof=1)) / math.sqrt(len(v))
    return mean, mean - half, mean + half


def holm_adjust(pvalues: Sequence[float]) -> np.ndarray:
    """Holm step-down adjusted p-values, in the input order: monotone in the sorted order and capped at 1."""
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    if m == 0:
        return p
    order = np.argsort(p, kind="stable")
    adjusted = np.minimum(1.0, np.maximum.accumulate((m - np.arange(m)) * p[order]))
    out = np.empty(m)
    out[order] = adjusted
    return out


def _f(value: Any) -> float:
    return _NAN if value is None else float(value)


# ---------------------------------------------------------------- daily returns on the session grid


class _Daily:
    """Adjusted closes, simple returns and volumes on the XNYS session grid, one row per symbol. A return spans
    exactly one session (previous_session) and is NaN when either close is missing."""

    def __init__(self, conn: sqlite3.Connection, symbols: Iterable[str]) -> None:
        self.sessions: list[date] = list(market.xnys().sessions.date)
        self.pos = {d.isoformat(): i for i, d in enumerate(self.sessions)}
        self.row = {s: i for i, s in enumerate(dict.fromkeys(symbols))}
        self.missing = len(self.row)  # an all-NaN row for symbols without bars
        shape = (len(self.row) + 1, len(self.sessions))
        self.close = np.full(shape, np.nan)
        self.volume = np.full(shape, np.nan)
        for symbol, day, adj, vol in conn.execute("SELECT symbol, session_date, adj_close, volume FROM bars_1d"):
            r, c = self.row.get(symbol), self.pos.get(day)
            if r is None or c is None:
                continue
            if adj is not None and adj > 0:
                self.close[r, c] = adj
            if vol is not None:
                self.volume[r, c] = vol
        self.ret = np.full(shape, np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            self.ret[:, 1:] = self.close[:, 1:] / self.close[:, :-1] - 1
        self.market = self.row.get(MARKET, self.missing)

    def index(self, day: str | date | None) -> int:
        if day is None:
            return -1
        return self.pos.get(day if isinstance(day, str) else day.isoformat(), -1)

    def symbol_row(self, symbol: str) -> int:
        return self.row.get(symbol, self.missing)

    def take(self, matrix: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        """matrix[rows[i], cols[i, j]], NaN where a column falls outside the grid."""
        n = matrix.shape[1]
        inside = (cols >= 0) & (cols < n)
        return np.where(inside, matrix[rows[:, None], np.clip(cols, 0, n - 1)], np.nan)


@dataclass
class _Fit:
    """Market-model results for a batch of (symbol, anchor session) pairs; NaN where no model could be fitted."""

    alpha: np.ndarray
    beta: np.ndarray
    sigma: np.ndarray
    n: np.ndarray
    ar: np.ndarray  # (m, len(PATH_DAYS))
    car: dict[str, np.ndarray]
    z: dict[str, np.ndarray]
    resid: np.ndarray  # (m, len(_EST_OFFSETS)) estimation residuals; NaN on unused days and without a model

    @property
    def ok(self) -> np.ndarray:
        return np.isfinite(self.sigma)


def _fit_chunk(daily: _Daily, rows: np.ndarray, anchors: np.ndarray) -> _Fit:
    known = anchors >= 0

    def cols(offsets: np.ndarray) -> np.ndarray:
        return np.where(known[:, None], anchors[:, None] + offsets, -1)

    mkt = np.full(len(anchors), daily.market, dtype=np.int64)
    est = cols(_EST_OFFSETS)
    y = daily.take(daily.ret, rows, est)
    x = daily.take(daily.ret, mkt, est)
    use = np.isfinite(y) & np.isfinite(x)
    n = use.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        xbar = np.where(use, x, 0.0).sum(axis=1) / n
        ybar = np.where(use, y, 0.0).sum(axis=1) / n
        dx = np.where(use, x - xbar[:, None], 0.0)
        dy = np.where(use, y - ybar[:, None], 0.0)
        sxx = (dx * dx).sum(axis=1)
        beta = (dx * dy).sum(axis=1) / sxx
        alpha = ybar - beta * xbar
        resid = dy - beta[:, None] * dx
        sigma = np.sqrt((resid * resid).sum(axis=1) / (n - 2))
    ok = (n >= MIN_ESTIMATION_OBS) & (sxx > 0) & np.isfinite(sigma) & (sigma > 0)
    alpha, beta, sigma = (np.where(ok, v, np.nan) for v in (alpha, beta, sigma))
    resid = np.where(use & ok[:, None], resid, np.nan)

    path = cols(_PATH_OFFSETS)
    ar = daily.take(daily.ret, rows, path) - (alpha[:, None] + beta[:, None] * daily.take(daily.ret, mkt, path))
    car: dict[str, np.ndarray] = {}
    z: dict[str, np.ndarray] = {}
    for name, (lo, hi) in DAILY_WINDOWS.items():
        car[name] = ar[:, lo - PATH_DAYS[0] : hi - PATH_DAYS[0] + 1].sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            z[name] = car[name] / (sigma * math.sqrt(hi - lo + 1))
    return _Fit(alpha, beta, sigma, n.astype(np.int64), ar, car, z, resid)


def _fit(daily: _Daily, rows: Sequence[int] | np.ndarray, anchors: Sequence[int] | np.ndarray) -> _Fit:
    """OLS r_i = alpha + beta * r_SPY over ESTIMATION_WINDOW relative to each anchor, then AR/CAR/z around it."""
    rows = np.asarray(rows, dtype=np.int64)
    anchors = np.asarray(anchors, dtype=np.int64)
    parts = [
        _fit_chunk(daily, rows[i : i + _FIT_CHUNK], anchors[i : i + _FIT_CHUNK])
        for i in range(0, len(rows), _FIT_CHUNK)
    ]
    if not parts:
        parts = [_fit_chunk(daily, rows, anchors)]
    return _Fit(
        alpha=np.concatenate([p.alpha for p in parts]),
        beta=np.concatenate([p.beta for p in parts]),
        sigma=np.concatenate([p.sigma for p in parts]),
        n=np.concatenate([p.n for p in parts]),
        ar=np.concatenate([p.ar for p in parts]),
        car={w: np.concatenate([p.car[w] for p in parts]) for w in DAILY_WINDOWS},
        z={w: np.concatenate([p.z[w] for p in parts]) for w in DAILY_WINDOWS},
        resid=np.concatenate([p.resid for p in parts]),
    )


# ---------------------------------------------------------------- same-clock intraday volatility


class _Intraday:
    """Market-adjusted returns of the same New York clock-time window on prior sessions, priced with the M2
    price_at rule (events.MinuteBars.at) on bars from 04:00 of the session before, as M2 prices day 0."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn
        self._sessions: dict[str, set[date]] = {}
        self._both: dict[str, list[date]] = {}
        self._needed: dict[str, set[date]] = defaultdict(set)
        self._spans: dict[date, tuple[datetime, datetime]] = {}
        self._bars: dict[str, MinuteBars] = {}

    def _minute_sessions(self, symbol: str) -> set[date]:
        if symbol not in self._sessions:
            self._sessions[symbol] = {date.fromisoformat(s) for s in db.sessions_with_bars(self.conn, symbol)}
        return self._sessions[symbol]

    def prior_sessions(self, ticker: str, d0: date) -> list[date]:
        """The INTRADAY_SIGMA_SESSIONS most recent sessions before d0 with 1-minute bars for ticker and SPY."""
        if ticker not in self._both:
            self._both[ticker] = sorted(self._minute_sessions(ticker) & self._minute_sessions(MARKET))
        both = self._both[ticker]
        i = bisect.bisect_left(both, d0)
        return both[max(0, i - INTRADAY_SIGMA_SESSIONS) : i]

    def need(self, ticker: str, priors: Sequence[date]) -> None:
        for symbol in (ticker, MARKET):
            self._needed[symbol].update(priors)

    def _span(self, session: date) -> tuple[datetime, datetime]:
        """[04:00 New York on the previous session, 20:00 on this one): the bars that may price this session."""
        if session not in self._spans:
            self._spans[session] = (
                market.extended_bounds_utc(market.previous_session(session))[0],
                market.extended_bounds_utc(session)[1],
            )
        return self._spans[session]

    def load(self) -> None:
        """Reads only the spans of the sessions registered with need(), so the cost follows the events rather than
        the length of the stored 1-minute history."""
        for symbol, sessions in self._needed.items():
            merged: list[list[datetime]] = []
            for lo, hi in (self._span(s) for s in sorted(sessions)):
                if merged and lo <= merged[-1][1]:
                    merged[-1][1] = max(merged[-1][1], hi)
                else:
                    merged.append([lo, hi])
            starts: list[int] = []
            closes: list[float] = []
            for lo, hi in merged:
                part = MinuteBars.load(self.conn, symbol, lo, hi)
                starts += part.starts
                closes += part.closes
            self._bars[symbol] = MinuteBars(starts, closes)

    def sigma(self, ticker: str, priors: Sequence[date], clock: tuple[time, time], beta: float) -> tuple[float, str]:
        """(sample std of ret - beta * spy_ret over the priors the clock window fits in, why it is NaN): 'short' below
        MIN_INTRADAY_SIGMA_OBS usable sessions, 'flat' when every session had the same return; '' with a value."""
        bars, spy = self._bars.get(ticker), self._bars.get(MARKET)
        if bars is None or spy is None:
            return _NAN, "short"
        values: list[float] = []
        for session in priors:
            open_, close = market.session_bounds_utc(session)
            start = datetime.combine(session, clock[0], tzinfo=NY)
            end = datetime.combine(session, clock[1], tzinfo=NY)
            if start < open_ or end > close:
                continue
            floor = int(self._span(session)[0].timestamp())
            a, b, ma, mb = bars.at(start), bars.at(end), spy.at(start), spy.at(end)
            if a is None or b is None or ma is None or mb is None or min(a.ts, ma.ts) < floor:
                continue
            values.append((b.price / a.price - 1) - beta * (mb.price / ma.price - 1))
        if len(values) < MIN_INTRADAY_SIGMA_OBS:
            return _NAN, "short"
        s = float(np.std(values, ddof=1))
        return (s, "") if s > 0 else (_NAN, "flat")


def _clock_windows(t0: datetime, d0: date) -> dict[str, tuple[time, time]]:
    """New York wall-clock (start, end) of each intraday window after the event's own truncation."""
    out: dict[str, tuple[time, time]] = {}
    for spec in window_specs(t0, d0):
        if spec.win in INTRADAY_WINDOWS and spec.start.at is not None and spec.end.at is not None:
            out[spec.win] = (spec.start.at.astimezone(NY).time(), spec.end.at.astimezone(NY).time())
    return out


# ---------------------------------------------------------------- loading


def _rows(conn: sqlite3.Connection, sql: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
    cur = conn.execute(sql, params)
    names = [d[0] for d in cur.description]
    return [dict(zip(names, tuple(r), strict=True)) for r in cur.fetchall()]


def _load_events(conn: sqlite3.Connection, options: StudyOptions, benchmarks: set[str]) -> list[dict[str, Any]]:
    sql = """SELECT e.id, e.platform, e.native_id, e.ticker, e.t0, e.d0, e.session_phase, e.intraday_state,
                    e.earnings_flag, e.split_flag, e.clustered,
                    p.author, p.source, p.stance, p.stance_conf, p.text, p.url
             FROM events e LEFT JOIN posts p ON p.platform = e.platform AND p.native_id = e.native_id
             WHERE e.status = 'complete'"""
    params: list[Any] = []
    if options.since is not None:
        sql += " AND e.d0 >= ?"
        params.append(options.since.isoformat())
    if options.until is not None:
        sql += " AND e.d0 <= ?"
        params.append(options.until.isoformat())
    rows = _rows(conn, sql + " ORDER BY e.d0, e.t0, e.id", params)
    return [r for r in rows if r["ticker"] not in benchmarks]


@dataclass(frozen=True)
class _Window:
    ret: float
    spy_ret: float


def _load_windows(conn: sqlite3.Connection, ids: set[int]) -> dict[int, dict[str, _Window]]:
    out: dict[int, dict[str, _Window]] = defaultdict(dict)
    for event_id, win, ret, spy_ret in conn.execute(
        """SELECT w.event_id, w.win, w.ret, w.spy_ret FROM event_windows w
           JOIN events e ON e.id = w.event_id WHERE e.status = 'complete'"""
    ):
        if event_id in ids:
            out[event_id][win] = _Window(_f(ret), _f(spy_ret))
    return out


def _categories(conn: sqlite3.Connection) -> dict[tuple[str, str], str]:
    return {
        (platform, handle.lower()): category
        for platform, handle, category in conn.execute("SELECT platform, handle, category FROM accounts")
    }


def _author_and_category(row: dict[str, Any], categories: dict[tuple[str, str], str]) -> tuple[str, str]:
    if row["platform"] == "reddit":
        return f"r/{row['source'] or 'unknown'}", REDDIT_CATEGORY
    author = row["author"] or ""
    return author, categories.get((row["platform"], author.lower())) or UNKNOWN_CATEGORY


def _et(iso: str) -> str:
    return from_iso(iso).astimezone(NY).strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------- event level


def _exclusion(model_ok: bool, split: float, earnings: float, clustered: int, options: StudyOptions) -> str:
    if not model_ok:
        return "no_model"
    if split == 1 and not options.include_splits:
        return "split"
    if earnings == 1 and not options.include_earnings:
        return "earnings"
    if clustered == 1 and not options.include_clustered:
        return "clustered"
    return ""


def _event_frame(
    conn: sqlite3.Connection,
    daily: _Daily,
    events: list[dict[str, Any]],
    fit: _Fit,
    anchors: np.ndarray,
    categories: dict[tuple[str, str], str],
    options: StudyOptions,
) -> tuple[pd.DataFrame, np.ndarray]:
    """study.events (one row per complete event, with NaN model metrics where no market model exists) and, per
    event, how many intraday windows have no z because their same-clock volatility is zero."""
    n = len(events)
    windows = _load_windows(conn, {e["id"] for e in events})
    legs = {f"{kind}_{leg}": np.full(n, np.nan) for leg in LEGS for kind in ("ar", "z")}
    intraday = {f"{kind}_{w}": np.full(n, np.nan) for w in INTRADAY_WINDOWS for kind in ("iar", "iz")}
    spy_post_leg = np.full(n, np.nan)

    helper = _Intraday(conn)
    todo: list[tuple[int, list[date], dict[str, tuple[time, time]]]] = []
    for i, e in enumerate(events):
        wins = windows.get(e["id"], {})
        post = wins.get("post_leg")
        if post is not None:
            spy_post_leg[i] = post.spy_ret
        if e["intraday_state"] != "ok" or not fit.ok[i]:
            continue
        beta, sigma = fit.beta[i], fit.sigma[i]
        for leg in LEGS:
            w = wins.get(leg)
            if w is not None:
                legs[f"ar_{leg}"][i] = w.ret - beta * w.spy_ret
                legs[f"z_{leg}"][i] = legs[f"ar_{leg}"][i] / sigma
        present = [w for w in INTRADAY_WINDOWS if w in wins]
        if not present:
            continue
        for w in present:
            intraday[f"iar_{w}"][i] = wins[w].ret - beta * wins[w].spy_ret
        d0 = date.fromisoformat(e["d0"])
        priors = helper.prior_sessions(e["ticker"], d0)
        if len(priors) < MIN_INTRADAY_SIGMA_OBS:
            continue
        clocks = _clock_windows(from_iso(e["t0"]), d0)
        wanted = {w: clocks[w] for w in present if w in clocks and math.isfinite(intraday[f"iar_{w}"][i])}
        if wanted:
            todo.append((i, priors, wanted))
            helper.need(e["ticker"], priors)
    helper.load()
    flat = np.zeros(n, dtype=np.int64)
    for i, priors, clocks in todo:
        for w, clock in clocks.items():
            s, why = helper.sigma(events[i]["ticker"], priors, clock, float(fit.beta[i]))
            intraday[f"iz_{w}"][i] = intraday[f"iar_{w}"][i] / s
            flat[i] += why == "flat"

    rows = np.array([daily.symbol_row(e["ticker"]) for e in events], dtype=np.int64)
    vol_cols = np.where(anchors[:, None] >= 0, anchors[:, None] + _VOL_OFFSETS, -1)
    base = daily.take(daily.volume, rows, vol_cols)
    n_base = np.isfinite(base).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        base_mean = np.nansum(base, axis=1) / n_base
        d0_volume = daily.take(daily.volume, rows, np.where(anchors >= 0, anchors, -1)[:, None])[:, 0]
        rel_volume = np.where((n_base >= MIN_VOLUME_OBS) & (base_mean > 0), d0_volume / base_mean, np.nan)
    mkt = np.full(n, daily.market, dtype=np.int64)
    spy_ret_d0 = daily.take(daily.ret, mkt, np.where(anchors >= 0, anchors, -1)[:, None])[:, 0]

    cols: dict[str, Any] = {c: [] for c in EVENT_COLUMNS}
    for i, e in enumerate(events):
        author, category = _author_and_category(e, categories)
        earnings, split = _f(e["earnings_flag"]), _f(e["split_flag"])
        clustered = int(e["clustered"] or 0)
        values: dict[str, Any] = {
            "event_id": int(e["id"]),
            "platform": e["platform"],
            "native_id": str(e["native_id"]),
            "author": author,
            "category": category,
            "ticker": e["ticker"],
            "t0": e["t0"],
            "t0_et": _et(e["t0"]),
            "d0": e["d0"],
            "session_phase": e["session_phase"],
            "stance": e["stance"],
            "stance_conf": _f(e["stance_conf"]),
            "sign": _SIGNS.get(e["stance"] or "", _NAN),
            "excluded_reason": _exclusion(bool(fit.ok[i]), split, earnings, clustered, options),
            "earnings_flag": earnings,
            "split_flag": split,
            "clustered": clustered,
            "intraday_state": e["intraday_state"],
            "alpha": float(fit.alpha[i]),
            "beta": float(fit.beta[i]),
            "sigma": float(fit.sigma[i]),
            "n_est": int(fit.n[i]),
            "rel_volume": float(rel_volume[i]),
            "spy_ret_d0": float(spy_ret_d0[i]),
            "spy_post_leg": float(spy_post_leg[i]),
            "text": e["text"],
            "url": e["url"],
        }
        for j, name in enumerate(AR_COLUMNS):
            values[name] = float(fit.ar[i, j])
        for w in DAILY_WINDOWS:
            values[f"car_{w}"] = float(fit.car[w][i])
            values[f"z_{w}"] = float(fit.z[w][i])
        for name, arr in (*legs.items(), *intraday.items()):
            values[name] = float(arr[i])
        for c in EVENT_COLUMNS:
            cols[c].append(values[c])
    return pd.DataFrame(cols, columns=list(EVENT_COLUMNS)), flat


def _neighbours(conn: sqlite3.Connection, daily: _Daily, included: pd.DataFrame) -> tuple[int, int]:
    """(included events with another post on the ticker whose d0 is 1-5 sessions away, included events with another
    post's d0 inside their estimation window). Posts of every status count: they move the stock either way."""
    days: dict[str, list[int]] = defaultdict(list)
    for ticker, d0 in conn.execute("SELECT ticker, d0 FROM events"):
        i = daily.index(d0)
        if i >= 0:
            days[ticker].append(i)
    for positions in days.values():
        positions.sort()

    def count(ticker: str, lo: int, hi: int) -> int:
        positions = days.get(ticker, [])
        return bisect.bisect_right(positions, hi) - bisect.bisect_left(positions, lo)

    overlap = estimation = 0
    for ticker, d0 in zip(included["ticker"], included["d0"], strict=True):
        a = daily.index(d0)
        overlap += count(ticker, a + PATH_DAYS[0], a + PATH_DAYS[-1]) > count(ticker, a, a)
        estimation += count(ticker, a + ESTIMATION_WINDOW[0], a + ESTIMATION_WINDOW[1]) > 0
    return overlap, estimation


# ---------------------------------------------------------------- placebo


def _blocked(
    daily: _Daily, conn: sqlite3.Connection, tickers: set[str], options: StudyOptions
) -> dict[str, np.ndarray]:
    """Per ticker, the sessions a placebo day may not fall on: within PLACEBO_CLEARANCE of any event d0 on it (any
    status) and, under the rules that exclude events and unless those exclusions are switched off, within
    EARNINGS_RADIUS of an earnings report's first session or SPLIT_WINDOW_SESSIONS of a split."""
    out = {t: np.zeros(len(daily.sessions), dtype=bool) for t in tickers}

    def block(ticker: str, day: str | date, radius: int) -> None:
        i = daily.index(day)
        if ticker in out and i >= 0:
            out[ticker][max(0, i - radius) : i + radius + 1] = True

    for ticker, d0 in conn.execute("SELECT ticker, d0 FROM events"):
        block(ticker, d0, PLACEBO_CLEARANCE)
    if not options.include_earnings:
        for symbol, stamp in conn.execute("SELECT symbol, earnings_at FROM earnings"):
            if symbol not in out:
                continue
            try:
                session = earnings_session(from_iso(stamp))
            except ValueError:
                continue  # outside the calendar; events.earnings_sessions skips these too
            block(symbol, session, EARNINGS_RADIUS)
    if not options.include_splits:
        for symbol, day in conn.execute("SELECT symbol, session_date FROM bars_1d WHERE split_ratio != 0"):
            block(symbol, day, SPLIT_WINDOW_SESSIONS)
    return out


def placebo_sessions(anchor: int, blocked: np.ndarray, draws: int, seed: int, event_id: int) -> np.ndarray:
    """Up to `draws` distinct grid positions drawn uniformly from PLACEBO_RANGE around `anchor`, skipping blocked
    sessions. The generator depends only on (seed, event_id), never on iteration order."""
    lo = max(0, anchor + PLACEBO_RANGE[0])
    hi = anchor + PLACEBO_RANGE[1]
    if draws <= 0 or anchor < 0 or hi < lo:
        return np.empty(0, dtype=np.int64)
    candidates = np.flatnonzero(~blocked[lo : hi + 1]) + lo
    if len(candidates) == 0:
        return np.empty(0, dtype=np.int64)
    rng = np.random.default_rng(seed * PLACEBO_SEED_STRIDE + event_id)
    return np.sort(rng.choice(candidates, size=min(draws, len(candidates)), replace=False))


def _placebo_frame(
    conn: sqlite3.Connection, daily: _Daily, included: pd.DataFrame, options: StudyOptions
) -> pd.DataFrame:
    """One row per distinct (ticker, placebo day) with a market model. Every event draws its own days; a day drawn
    for several events on the same ticker is one observation, kept under the lowest event id."""
    blocked = _blocked(daily, conn, set(included["ticker"]), options)
    seen: set[tuple[str, int]] = set()
    ids: list[int] = []
    tickers: list[str] = []
    anchors: list[int] = []
    order = included.sort_values("event_id", kind="stable")
    for event_id, ticker, d0 in zip(order["event_id"], order["ticker"], order["d0"], strict=True):
        days = placebo_sessions(daily.index(d0), blocked[ticker], options.placebo_draws, options.seed, int(event_id))
        for day in days.tolist():
            if (ticker, day) in seen:
                continue
            seen.add((ticker, day))
            ids.append(int(event_id))
            tickers.append(ticker)
            anchors.append(day)
    fit = _fit(daily, [daily.symbol_row(t) for t in tickers], anchors)
    keep = fit.ok
    data: dict[str, Any] = {
        "event_id": np.asarray(ids, dtype=np.int64)[keep],
        "ticker": np.asarray(tickers, dtype=object)[keep],
        "day": np.asarray([daily.sessions[a].isoformat() for a in anchors], dtype=object)[keep],
        "alpha": fit.alpha[keep],
        "beta": fit.beta[keep],
        "sigma": fit.sigma[keep],
        "n_est": fit.n[keep],
    }
    for j, name in enumerate(AR_COLUMNS):
        data[name] = fit.ar[keep, j]
    for w in DAILY_WINDOWS:
        data[f"car_{w}"] = fit.car[w][keep]
        data[f"z_{w}"] = fit.z[w][keep]
    return pd.DataFrame(data, columns=list(PLACEBO_COLUMNS))


# ---------------------------------------------------------------- post level


_POST_MEANS: tuple[str, ...] = (
    "car_pre", "car_event", "car_post", *AR_COLUMNS, "ar_pre_leg", "ar_post_leg",
    *(f"iar_{w}" for w in INTRADAY_WINDOWS),
)  # fmt: skip
_POST_FIRST: tuple[str, ...] = (
    "author", "category", "t0", "t0_et", "stance", "stance_conf", "sign", "text", "url",
)  # fmt: skip


def _portfolio_z(
    post_car: np.ndarray, car: np.ndarray, resid: np.ndarray, starts: np.ndarray, length: int
) -> np.ndarray:
    """Per post, the z-score of the equal-weighted portfolio of its events with a finite car:
    post_car / (sigma_p * sqrt(length)), where sigma_p is the residual std (n - 2 degrees of freedom) of the events'
    mean estimation residual over the days on which all of them have one; NaN below MIN_ESTIMATION_OBS such days.

    The events of post p are rows starts[p] up to starts[p + 1] of car and resid. They share t0 and so d0, which
    makes each resid column the same calendar day for all of them."""
    member = np.isfinite(car)
    has = member[:, None] & np.isfinite(resid)
    members = np.add.reduceat(member.astype(np.int64), starts)
    present = np.add.reduceat(has.astype(np.int64), starts, axis=0)
    total = np.add.reduceat(np.where(has, resid, 0.0), starts, axis=0)
    common = (present == members[:, None]) & (members[:, None] > 0)
    n_common = common.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_resid = np.where(common, total / members[:, None], 0.0)
        sigma = np.sqrt((mean_resid * mean_resid).sum(axis=1) / (n_common - 2))
        z = post_car / (sigma * math.sqrt(length))
    return np.where((n_common >= MIN_ESTIMATION_OBS) & (sigma > 0), z, np.nan)


def _post_frame(included: pd.DataFrame, resid: np.ndarray) -> pd.DataFrame:
    """study.posts from the included events; resid holds their estimation residuals (_Fit.resid), row for row."""
    if included.empty:
        return pd.DataFrame(columns=list(POST_COLUMNS))
    ordered = included.reset_index(drop=True).sort_values(["platform", "native_id", "ticker"], kind="stable")
    g = ordered.groupby(["platform", "native_id"], sort=False)
    frame = g[list(_POST_FIRST)].first()  # identical on every row of a post
    frame["n_events"] = g.size()
    frame["tickers"] = g["ticker"].agg(",".join)
    frame[list(_POST_MEANS)] = g[list(_POST_MEANS)].mean()  # skips NaN; all-NaN gives NaN
    codes = g.ngroup().to_numpy()  # sorted rows keep each post contiguous, in the same order as frame
    starts = np.flatnonzero(np.r_[True, codes[1:] != codes[:-1]])
    resid = resid[ordered.index.to_numpy()]
    for w, (lo, hi) in DAILY_WINDOWS.items():
        frame[f"z_{w}"] = _portfolio_z(
            frame[f"car_{w}"].to_numpy(dtype=float),
            ordered[f"car_{w}"].to_numpy(dtype=float),
            resid,
            starts,
            hi - lo + 1,
        )
    frame = frame.reset_index()
    sign = frame["sign"].to_numpy(dtype=float)
    signed = np.isin(sign, (1.0, -1.0))
    for w in DAILY_WINDOWS:
        frame[f"signed_car_{w}"] = np.where(signed, sign * frame[f"car_{w}"].to_numpy(dtype=float), np.nan)
    frame = frame[list(POST_COLUMNS)]
    return frame.sort_values(["t0", "platform", "native_id"], kind="stable").reset_index(drop=True)


def _group_row(family: str, group: str, subset: str, window: str, posts: pd.DataFrame) -> dict[str, Any]:
    car = posts[f"signed_car_{window}"].to_numpy(dtype=float)
    keep = np.isfinite(car)
    values = car[keep]
    z = np.abs(posts[f"z_{window}"].to_numpy(dtype=float)[keep])
    z = z[np.isfinite(z)]
    n = len(values)
    mean, low, high = mean_ci(values)
    status = "ok" if n >= MIN_GROUP_POSTS else "insufficient"
    t_stat = p_value = _NAN
    if status == "ok":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            res = stats.ttest_1samp(values, 0.0)
        t_stat, p_value = float(res.statistic), float(res.pvalue)
    return {
        "family": family,
        "group": group,
        "subset": subset,
        "window": window,
        "n_posts": n,
        "mean_signed_car": mean,
        "median_signed_car": float(np.median(values)) if n else _NAN,
        "ci_low": low,
        "ci_high": high,
        "mean_abs_z": float(z.mean()) if len(z) else _NAN,
        "share_abs_z_gt_crit": float((z > Z_CRITICAL).mean()) if len(z) else _NAN,
        "t_stat": t_stat,
        "p_value": p_value,
        "p_holm": _NAN,
        "status": status,
    }


def _group_frame(posts: pd.DataFrame, min_conf: float) -> pd.DataFrame:
    signed = posts[posts["sign"].isin([1.0, -1.0])] if not posts.empty else posts
    if signed.empty:
        return pd.DataFrame(columns=list(GROUP_COLUMNS))
    keys = {
        "all": pd.Series("all", index=signed.index),
        "author": signed["author"],
        "category": signed["category"],
        "platform": signed["platform"],
        "stance": signed["stance"],
    }
    confident = signed["stance_conf"].to_numpy(dtype=float) >= min_conf
    rows: list[dict[str, Any]] = []
    for family in FAMILIES:
        key = keys[family].astype(object)
        present = set(key)
        groups = [s for s in SIGNED_STANCES if s in present] if family == "stance" else sorted(present, key=str)
        block: list[dict[str, Any]] = []
        for group in groups:
            in_group = (key == group).to_numpy()
            for subset in SUBSETS:
                mask = in_group & confident if subset == "confident" else in_group
                for window in DAILY_WINDOWS:
                    block.append(_group_row(family, str(group), subset, window, signed[mask]))
        for subset in SUBSETS:
            for window in DAILY_WINDOWS:
                tested = [
                    r for r in block
                    if r["subset"] == subset and r["window"] == window and r["status"] == "ok"
                    and math.isfinite(r["p_value"])
                ]  # fmt: skip
                for r, adj in zip(tested, holm_adjust([r["p_value"] for r in tested]), strict=True):
                    r["p_holm"] = float(adj)
        rows += block
    return pd.DataFrame.from_records(rows, columns=list(GROUP_COLUMNS))


# ---------------------------------------------------------------- aggregate tables


def _magnitude_frame(included: pd.DataFrame, placebo: pd.DataFrame) -> pd.DataFrame:
    if included.empty:
        return pd.DataFrame(columns=list(MAGNITUDE_COLUMNS))
    rows: list[dict[str, Any]] = []
    for window in DAILY_WINDOWS:
        for sample, frame in (("events", included), ("placebo", placebo)):
            car = frame[f"car_{window}"].to_numpy(dtype=float)
            z = frame[f"z_{window}"].to_numpy(dtype=float)
            keep = np.isfinite(car)
            car, z = car[keep], np.abs(z[keep])
            n = len(car)
            rows.append(
                {
                    "window": window,
                    "sample": sample,
                    "n": n,
                    "mean_car": float(car.mean()) if n else _NAN,
                    "mean_abs_car": float(np.abs(car).mean()) if n else _NAN,
                    "median_abs_car": float(np.median(np.abs(car))) if n else _NAN,
                    "share_abs_z_gt_crit": float((z > Z_CRITICAL).mean()) if n else _NAN,
                }
            )
    return pd.DataFrame.from_records(rows, columns=list(MAGNITUDE_COLUMNS))


def _path_rows(series: str, ar: np.ndarray) -> list[dict[str, Any]]:
    """Mean cumulative AR from day -5 through each day; an observation counts on a day only when all its ARs from
    day -5 through that day exist."""
    cum = np.cumsum(ar, axis=1)
    rows = []
    for j, day in enumerate(PATH_DAYS):
        values = cum[:, j][np.isfinite(cum[:, j])]
        mean, low, high = mean_ci(values)
        rows.append({"series": series, "day": day, "mean_car": mean, "ci_low": low, "ci_high": high, "n": len(values)})
    return rows


def _car_path_frame(posts: pd.DataFrame, placebo: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for series in PATH_SERIES:
        sub = posts[posts["stance"] == series] if not posts.empty else posts
        if not sub.empty:
            rows += _path_rows(series, sub[list(AR_COLUMNS)].to_numpy(dtype=float))
    if not placebo.empty:
        rows += _path_rows("placebo", placebo[list(AR_COLUMNS)].to_numpy(dtype=float))
    return pd.DataFrame.from_records(rows, columns=list(PATH_COLUMNS))


def _intraday_path_frame(posts: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    iar_cols = [f"iar_{w}" for w in INTRADAY_WINDOWS]
    for series in SIGNED_STANCES:
        if posts.empty:
            break
        sub = posts[posts["stance"] == series]
        sub = sub[np.isfinite(sub[iar_cols].to_numpy(dtype=float)).any(axis=1)]
        if sub.empty:
            continue
        for minute in INTRADAY_PATH_MINUTES:
            if minute == 0:
                values = np.zeros(len(sub))
            elif minute < 0:
                values = -sub[f"iar_pre{-minute}"].to_numpy(dtype=float)
            else:
                values = sub[f"iar_p{minute}"].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            mean, low, high = mean_ci(values)
            rows.append(
                {"series": series, "minute": minute, "mean_ar": mean, "ci_low": low, "ci_high": high, "n": len(values)}
            )
    return pd.DataFrame.from_records(rows, columns=list(INTRADAY_PATH_COLUMNS))


# ---------------------------------------------------------------- Reddit attention


def next_session_after(t: datetime) -> date | None:
    """The first session whose regular open is after t."""
    try:
        day = t.astimezone(NY).date()
        session = market.xnys().date_to_session(pd.Timestamp(day), direction="next").date()
        if market.session_bounds_utc(session)[0] <= t:
            session = market.session_offset(session, 1)
    except (ValueError, KeyError, IndexError):
        return None
    return session


def _attention_frame(conn: sqlite3.Connection, daily: _Daily, watch: set[str]) -> tuple[pd.DataFrame, int]:
    """ApeWisdom mention spikes vs the next session's abnormal return, and the number of snapshot dates."""
    rows = _rows(
        conn,
        "SELECT snapshot_date, ticker, mentions, fetched_at FROM reddit_ticker_daily WHERE filter = ?",
        (ATTENTION_FILTER,),
    )
    dates = sorted({r["snapshot_date"] for r in rows})
    position = {d: i for i, d in enumerate(dates)}
    by_ticker: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for r in rows:
        if r["ticker"] in watch:
            by_ticker[r["ticker"]][r["snapshot_date"]] = r
    records: list[dict[str, Any]] = []
    for ticker in sorted(by_ticker):
        snaps = by_ticker[ticker]
        for day in sorted(snaps):
            r = snaps[day]
            if r["mentions"] is None:
                continue
            j = position[day]
            trailing = [
                snaps[d]["mentions"] for d in dates[max(0, j - ATTENTION_TRAILING_DAYS) : j]
                if d in snaps and snaps[d]["mentions"] is not None
            ]  # fmt: skip
            if len(trailing) < MIN_ATTENTION_TRAILING:
                continue
            trailing_mean = float(np.mean(trailing))
            try:
                nxt = next_session_after(from_iso(r["fetched_at"]))
            except ValueError:
                nxt = None
            records.append(
                {
                    "ticker": ticker,
                    "snapshot_date": day,
                    "mentions": int(r["mentions"]),
                    "trailing_mean": trailing_mean,
                    "spike": r["mentions"] / trailing_mean if trailing_mean > 0 else _NAN,
                    "next_session": nxt.isoformat() if nxt else None,
                }
            )
    if records:
        fit = _fit(
            daily,
            [daily.symbol_row(r["ticker"]) for r in records],
            [daily.index(r["next_session"]) for r in records],
        )
        day0 = PATH_DAYS.index(0)
        for i, r in enumerate(records):
            r["next_ar"] = float(fit.ar[i, day0])
            r["next_z"] = float(fit.ar[i, day0] / fit.sigma[i]) if fit.ok[i] else _NAN
    frame = pd.DataFrame.from_records(records, columns=list(ATTENTION_COLUMNS))
    return frame, len(dates)


# ---------------------------------------------------------------- notes


def _attention_gaps(attention: pd.DataFrame, daily: _Daily) -> tuple[int, int, int, int]:
    """(tickers in the table, tickers without any daily price, rows of priced tickers, those rows lacking next_ar)."""
    if attention.empty:
        return 0, 0, 0, 0
    tickers = set(attention["ticker"])
    priced = {t for t in tickers if np.isfinite(daily.close[daily.symbol_row(t)]).any()}
    rows = attention[attention["ticker"].isin(priced)]
    return len(tickers), len(tickers - priced), len(rows), int(rows["next_ar"].isna().sum())


@dataclass(frozen=True)
class _Facts:
    """What the notes report beyond Study.counts."""

    placebo_events: int
    windows_with_iar: int
    windows_short: int  # no intraday z: fewer than MIN_INTRADAY_SIGMA_OBS usable prior sessions
    windows_flat: int  # no intraday z: zero same-clock volatility
    posts_neutral: int
    posts_unlabelled: int
    events_overlap: int
    events_estimation_overlap: int
    attention_rows: int
    attention_tickers: int
    attention_unpriced: int
    attention_priced_rows: int
    attention_priced_blank: int


def _pct(part: int, whole: int) -> str:
    return f"{100 * part / whole:.0f}%" if whole else "0%"


def _and(items: Sequence[str]) -> str:
    return items[0] if len(items) == 1 else f"{', '.join(items[:-1])} and {items[-1]}"


def _notes(counts: dict[str, int], options: StudyOptions, facts: _Facts) -> list[str]:
    notes: list[str] = []
    span = f"{options.since or 'the start'} to {options.until or 'now'}"
    complete, included = counts["events_complete"], counts["events_included"]
    if complete == 0:
        notes.append(
            f"No complete events in the range {span}, so there is nothing to analyze yet. An event completes once "
            "daily prices exist through 5 sessions after its event day."
        )
    elif included == 0:
        notes.append(f"All {complete} complete events are excluded (see the exclusion counts), so no statistics exist.")
    if complete:
        reasons = ", ".join(f"{counts[f'excluded_{r}']} {r.replace('_', ' ')}" for r in EXCLUSION_REASONS)
        notes.append(
            f"{included} of {complete} complete events are included. Excluded: {reasons}. 'earnings' = an earnings "
            "report within one session of the event day; 'split' = a stock split within 5 sessions; 'clustered' = "
            "not the first post about that ticker in the session; 'no model' = fewer than "
            f"{MIN_ESTIMATION_OBS} usable days in the market-model estimation window."
        )
        allowed = [
            name for name, flag in (("earnings", options.include_earnings), ("split", options.include_splits),
                                    ("clustered", options.include_clustered)) if flag
        ]  # fmt: skip
        if allowed:
            notes.append(f"By request, {', '.join(allowed)}-flagged events are kept in the statistics.")
    if counts["earnings_unknown"]:
        notes.append(
            f"{counts['earnings_unknown']} included events have unknown earnings status (the earnings lookup failed "
            "or its dates do not reach back far enough); they are kept in, so an earnings move may hide among them."
        )
    if included:
        if facts.events_overlap:
            notes.append(
                f"{facts.events_overlap} of {included} included events have another post about the same stock whose "
                f"day 0 is 1 to {PATH_DAYS[-1]} sessions from theirs. Their daily windows then contain that post's "
                "day-0 move, so one abnormal return can count toward several posts (with opposite signs when the "
                "stances differ), and those posts are not independent observations. Placebo days have no post within "
                f"{PLACEBO_CLEARANCE} sessions, so they carry no such overlap."
            )
        if facts.events_estimation_overlap:
            lo, hi = ESTIMATION_WINDOW
            notes.append(
                f"{facts.events_estimation_overlap} of {included} included events have another post about the same "
                f"stock with its day 0 inside their market-model estimation window (sessions {lo} to {hi}). If that "
                "post moved the stock, the model treats the move as normal volatility, so sigma is overstated and "
                "these events' z-scores lean toward zero (a conservative bias)."
            )
        daily_only = included - counts["events_intraday"]
        if daily_only:
            notes.append(
                f"{daily_only} of {included} included events ({_pct(daily_only, included)}) are daily-only: no "
                "1-minute data covers their event day (Yahoo keeps 1-minute bars for about 30 days, so backfilled "
                "posts are older than that). They have no day-0 split at the post and no intraday windows."
            )
        if facts.windows_short:
            notes.append(
                f"{facts.windows_short} of {facts.windows_with_iar} intraday windows have no z-score: fewer than "
                f"{MIN_INTRADAY_SIGMA_OBS} usable prior sessions with 1-minute data for the ticker and SPY (a session "
                "counts only when the same clock window fits inside its regular hours)."
            )
        if facts.windows_flat:
            notes.append(
                f"{facts.windows_flat} of {facts.windows_with_iar} intraday windows have no z-score because the "
                "same-clock return was identical on every prior session (zero volatility). This happens when the "
                "window is too short for the price to change, e.g. a post in the first minute: its pre-60 window "
                "starts and ends at the last pre-market price."
            )
    signed, posts = counts["posts_signed"], counts["posts_included"]
    if included and signed < MIN_GROUP_POSTS:
        notes.append(
            f"Too few labelled posts for statistical tests yet: {signed} included posts carry a bullish or bearish "
            f"label, and a group needs at least {MIN_GROUP_POSTS} before a t-test is run. Averages are shown for "
            "description only."
        )
    if facts.posts_neutral:
        notes.append(
            f"{facts.posts_neutral} of {posts} included posts are labelled neutral: they count in the magnitude table "
            "and the neutral CAR path, but not in the signed (bullish/bearish) tests or the intraday path."
        )
    if facts.posts_unlabelled:
        notes.append(
            f"{facts.posts_unlabelled} of {posts} included posts have no stance label (the sentiment model has not "
            "labelled them): they count only in the event-level magnitude table, not in the CAR paths or the signed "
            "tests."
        )
    if included:
        rules = [f"no post about the stock within {PLACEBO_CLEARANCE} sessions"]
        if not options.include_earnings:
            rules.append(f"no earnings report within {EARNINGS_RADIUS} session")
        if not options.include_splits:
            rules.append(f"no split within {SPLIT_WINDOW_SESSIONS} sessions")
        text = (
            f"Placebo baseline: {counts['placebo_days']} distinct stock-days from {facts.placebo_events} included "
            f"events (up to {options.placebo_draws} each, with {_and(rules)}; a day drawn for two events on the same "
            "stock counts once)."
        )
        if counts["placebo_days"] < facts.placebo_events * options.placebo_draws:
            text += (
                " Fewer than the maximum: nearby posts, earnings or splits, short price history, or days already "
                "drawn for another event on the stock ruled some out."
            )
        notes.append(text)
    days = counts["attention_days"]
    need = ATTENTION_TRAILING_DAYS + 1
    if facts.attention_rows == 0:
        notes.append(
            f"ApeWisdom attention: {days} snapshot day(s) collected; {need} are needed for a full trailing week (a "
            f"spike needs at least {MIN_ATTENTION_TRAILING} of the previous {ATTENTION_TRAILING_DAYS} days), so the "
            "attention table is empty for now."
        )
    elif days < need:
        notes.append(
            f"ApeWisdom attention: {days} snapshot days collected; spikes use a partial trailing window until "
            f"{need} days exist."
        )
    if facts.attention_unpriced:
        notes.append(
            f"ApeWisdom attention: {facts.attention_unpriced} of {facts.attention_tickers} tickers in the table have "
            "no daily prices (prices are downloaded only for stocks that a collected post mentions), so their "
            "next-session abnormal return is blank; the returns shown come only from stocks that posts also mention."
        )
    if facts.attention_priced_blank:
        notes.append(
            f"ApeWisdom attention: {facts.attention_priced_blank} of {facts.attention_priced_rows} rows for stocks "
            "with daily prices have no next-session abnormal return: that session's price is not in the data yet, or "
            "the stock has too little history for a market model."
        )
    return notes


# ---------------------------------------------------------------- entry points


def compute_study_with_placebo(
    conn: sqlite3.Connection, watchlist: Watchlist, now: datetime, options: StudyOptions | None = None
) -> tuple[Study, pd.DataFrame]:
    """The Study plus one row per distinct placebo stock-day used (PLACEBO_COLUMNS), for audits."""
    options = options or DEFAULT_OPTIONS
    benchmarks = {t.symbol for t in watchlist.tickers if t.benchmark}
    watch = {t.symbol for t in watchlist.event_tickers}
    events = _load_events(conn, options, benchmarks)
    reddit_tickers = {r[0] for r in conn.execute("SELECT DISTINCT ticker FROM reddit_ticker_daily")} & watch
    daily = _Daily(conn, [MARKET, *sorted({e["ticker"] for e in events}), *sorted(reddit_tickers)])

    anchors = np.array([daily.index(e["d0"]) for e in events], dtype=np.int64)
    fit = _fit(daily, [daily.symbol_row(e["ticker"]) for e in events], anchors)
    event_frame, flat = _event_frame(conn, daily, events, fit, anchors, _categories(conn), options)

    kept = (event_frame["excluded_reason"] == "").to_numpy(dtype=bool)
    included = event_frame[kept].reset_index(drop=True)
    placebo = _placebo_frame(conn, daily, included, options)
    posts = _post_frame(included, fit.resid[kept])
    attention, attention_days = _attention_frame(conn, daily, watch)

    windows_with_iar = int(np.isfinite(included[[f"iar_{w}" for w in INTRADAY_WINDOWS]].to_numpy(dtype=float)).sum())
    windows_with_iz = int(np.isfinite(included[[f"iz_{w}" for w in INTRADAY_WINDOWS]].to_numpy(dtype=float)).sum())
    windows_flat = int(flat[kept].sum())
    overlap, estimation_overlap = _neighbours(conn, daily, included)
    tickers, unpriced, priced_rows, priced_blank = _attention_gaps(attention, daily)

    reasons = event_frame["excluded_reason"]
    counts: dict[str, int] = {
        "events_complete": len(event_frame),
        "events_included": len(included),
        **{f"excluded_{r}": int((reasons == r).sum()) for r in EXCLUSION_REASONS},
        "earnings_unknown": int(included["earnings_flag"].isna().sum()),
        "events_intraday": int((included["intraday_state"] == "ok").sum()),
        "posts_included": len(posts),
        "posts_signed": int(posts["sign"].isin([1.0, -1.0]).sum()) if not posts.empty else 0,
        "placebo_days": len(placebo),
        "attention_days": attention_days,
    }
    sign = posts["sign"].to_numpy(dtype=float)
    facts = _Facts(
        placebo_events=len(included),
        windows_with_iar=windows_with_iar,
        windows_short=windows_with_iar - windows_with_iz - windows_flat,
        windows_flat=windows_flat,
        posts_neutral=int((sign == 0).sum()),
        posts_unlabelled=int(np.isnan(sign).sum()),
        events_overlap=overlap,
        events_estimation_overlap=estimation_overlap,
        attention_rows=len(attention),
        attention_tickers=tickers,
        attention_unpriced=unpriced,
        attention_priced_rows=priced_rows,
        attention_priced_blank=priced_blank,
    )
    study = Study(
        options=options,
        generated_at=now,
        events=event_frame,
        posts=posts,
        groups=_group_frame(posts, options.min_conf_robust),
        magnitude=_magnitude_frame(included, placebo),
        car_path=_car_path_frame(posts, placebo),
        intraday_path=_intraday_path_frame(posts),
        attention=attention,
        counts=counts,
        notes=_notes(counts, options, facts),
    )
    log.info("study: %s", counts)
    return study, placebo


def compute_study(
    conn: sqlite3.Connection, watchlist: Watchlist, now: datetime, options: StudyOptions = DEFAULT_OPTIONS
) -> Study:
    """Every M3 statistic for complete events with d0 in [options.since, options.until]; see study.py."""
    return compute_study_with_placebo(conn, watchlist, now, options)[0]
