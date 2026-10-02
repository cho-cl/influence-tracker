from __future__ import annotations

import math
import sqlite3
from collections.abc import Callable
from datetime import date, datetime, timedelta

from .. import market
from ..analysis.metrics import EventModel
from ..events import MARKET, PricePoint, earnings_session
from ..timeutil import NY, from_iso
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


def _fresh(point: PricePoint | None, at: datetime) -> bool:
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
    lo, hi = market.previous_session(d0), market.session_offset(d0, 1)
    for (stamp,) in conn.execute("SELECT earnings_at FROM earnings WHERE symbol = ?", (ticker,)):
        try:
            session = earnings_session(from_iso(stamp))
        except ValueError:
            continue  # outside the calendar; events.earnings_sessions skips these too
        if lo <= session <= hi:
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
        "SELECT clustered FROM events WHERE platform = ? AND native_id = ? AND ticker = ?",
        (platform, native_id, ticker),
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
