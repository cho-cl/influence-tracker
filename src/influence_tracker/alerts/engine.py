from __future__ import annotations

import logging
import sqlite3
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from datetime import time as dtime

import pandas as pd

from .. import db, market
from ..analysis.metrics import EventModel, compute_study, event_model
from ..config import Watchlist
from ..timeutil import NY, from_iso, to_iso
from . import followups, messages, timing
from .followups import ALERT_PLATFORMS
from .live import PriceSource
from .messages import History, PostInfo
from .notify import Notifier, NotifyError

log = logging.getLogger(__name__)

RETRY_FOR = timedelta(hours=24)
PRICE_LOOKBACK = timedelta(days=4)
INIT_SOURCE, INIT_KEY = "alerts", "initialized"
DIGEST_PLATFORM = "-"
EXISTED_NOTE = "existed before alerts were turned on"
LATE_NOTE = "late: sent in digest"

HistoryLookup = Callable[[str], History | None]


class StudyHistory:
    """author -> History from a fresh compute_study, computed on first use and reused for the cycle."""

    def __init__(self, conn: sqlite3.Connection, watchlist: Watchlist, now: datetime) -> None:
        self.conn, self.watchlist, self.now = conn, watchlist, now
        self._groups: pd.DataFrame | None = None
        self._computed = False

    def __call__(self, author: str) -> History | None:
        if not self._computed:
            # Once per cycle even when it fails: a failing study would otherwise cost seconds per heads-up.
            self._computed = True
            try:
                self._groups = compute_study(self.conn, self.watchlist, self.now).groups
            except Exception:
                log.exception("alerts: history unavailable (compute_study failed)")
        g = self._groups
        if g is None:
            return None
        rows = g[
            (g["family"] == "author") & (g["group"] == author) & (g["subset"] == "main") & (g["window"] == "event")
        ]
        if rows.empty:
            return None
        r = rows.iloc[0]
        p = r["p_holm"]
        return History(int(r["n_posts"]), float(r["mean_signed_car"]), None if pd.isna(p) else float(p))


def _event_symbols(watchlist: Watchlist) -> list[str]:
    return [t.symbol for t in watchlist.event_tickers]


def post_tickers(conn: sqlite3.Connection, platform: str, native_id: str, symbols: list[str]) -> tuple[str, ...]:
    marks = ",".join("?" * len(symbols))
    rows = conn.execute(
        f"""SELECT DISTINCT ticker FROM mentions WHERE platform = ? AND native_id = ? AND ticker IN ({marks})
            ORDER BY ticker""",
        (platform, native_id, *symbols),
    ).fetchall()
    return tuple(r[0] for r in rows)


def post_info(conn: sqlite3.Connection, platform: str, native_id: str, tickers: tuple[str, ...]) -> PostInfo:
    r = conn.execute("SELECT * FROM posts WHERE platform = ? AND native_id = ?", (platform, native_id)).fetchone()
    return PostInfo(
        platform=platform,
        native_id=native_id,
        author=r["author"],
        created_at=from_iso(r["created_at_utc"]),
        text=r["text"],
        url=r["url"],
        stance=r["stance"],
        stance_conf=r["stance_conf"],
        tickers=tickers,
    )


def d1_due(d0: date) -> datetime:
    """20:00 New York on the session after d0: bars_1d for d0+1 exist after the nightly run."""
    return datetime.combine(market.session_offset(d0, 1), dtime(20, 0), tzinfo=NY).astimezone(UTC)


class AlertEngine:
    def __init__(
        self,
        conn: sqlite3.Connection,
        watchlist: Watchlist,
        notifier: Notifier,
        *,
        live_factory: Callable[[], PriceSource],
        history_factory: Callable[[datetime], HistoryLookup],
        model: Callable[[str, date], EventModel | None] | None = None,
    ) -> None:
        self.conn = conn
        self.watchlist = watchlist
        self.cfg = watchlist.alerts
        self.notifier = notifier
        self.live_factory = live_factory
        self.history_factory = history_factory
        self.model = model or (lambda ticker, d0: event_model(conn, ticker, d0))
        self.symbols = _event_symbols(watchlist)
        self.holdings = set(self.cfg.holdings)

    # ------------------------------------------------------------ detection

    def _new_posts(self) -> list[sqlite3.Row]:
        if not self.symbols:
            return []
        marks = ",".join("?" * len(self.symbols))
        plats = ",".join("?" * len(ALERT_PLATFORMS))
        return self.conn.execute(
            f"""SELECT p.platform, p.native_id, p.created_at_utc, p.collected_at FROM posts p
                WHERE p.platform IN ({plats})
                  AND EXISTS (SELECT 1 FROM mentions m WHERE m.platform = p.platform AND m.native_id = p.native_id
                              AND m.ticker IN ({marks}))
                  AND NOT EXISTS (SELECT 1 FROM alerts a WHERE a.kind = 'heads_up' AND a.platform = p.platform
                                  AND a.native_id = p.native_id)
                ORDER BY p.created_at_utc, p.platform, p.native_id""",
            (*ALERT_PLATFORMS, *self.symbols),
        ).fetchall()

    def _mark_existed(self, post: sqlite3.Row, now: datetime) -> None:
        db.add_alert(
            self.conn,
            "heads_up",
            post["platform"],
            post["native_id"],
            from_iso(post["created_at_utc"]),
            now,
            status="skipped",
            error=EXISTED_NOTE,
        )

    def initialize(self, now: datetime) -> int:
        if db.get_watermark(self.conn, INIT_SOURCE, INIT_KEY) is not None:
            return 0
        posts = self._new_posts()
        with self.conn:
            for post in posts:
                self._mark_existed(post, now)
            db.set_watermark(self.conn, INIT_SOURCE, INIT_KEY, to_iso(now), now)
        log.info("alerts: initialised; %d existing stock post(s) will not be alerted", len(posts))
        return len(posts)

    def queue(self, now: datetime) -> dict:
        counts = {"new": 0, "late": 0}
        late: list[PostInfo] = []
        initialized_at = db.get_watermark(self.conn, INIT_SOURCE, INIT_KEY)
        with self.conn:
            for post in self._new_posts():
                if initialized_at is not None and post["collected_at"] < initialized_at:
                    # Stored before alerts began and only now mentions a ticker because the ticker list changed and
                    # stored posts were re-matched: as quiet as the posts marked when alerts were turned on.
                    self._mark_existed(post, now)
                    continue
                platform, native_id = post["platform"], post["native_id"]
                info = post_info(
                    self.conn, platform, native_id, post_tickers(self.conn, platform, native_id, self.symbols)
                )
                d0 = market.event_session(info.created_at)
                db.add_alert(self.conn, "follow_d1", platform, native_id, d1_due(d0), now)
                if timing.is_late(info.created_at, now, self.cfg):
                    db.add_alert(
                        self.conn, "heads_up", platform, native_id, now, now, status="skipped", error=LATE_NOTE
                    )
                    late.append(info)
                    counts["late"] += 1
                    continue
                window = timing.follow_window(info.created_at, d0, self.cfg.followup_minutes)
                db.add_alert(self.conn, "heads_up", platform, native_id, now, now)
                db.add_alert(self.conn, "follow_60m", platform, native_id, window.end + timing.FOLLOW_DELAY, now)
                counts["new"] += 1
            if late:
                msg = messages.digest(late)
                db.add_alert(
                    self.conn, "digest", DIGEST_PLATFORM, to_iso(now), now, now, title=msg.title, message=msg.body
                )
        return counts

    # ------------------------------------------------------------ sending

    def send_due(self, now: datetime) -> dict:
        counts = {"sent": 0, "failed": 0, "skipped": 0, "waiting": 0}
        live = self.live_factory()
        history = self.history_factory(now)
        for row in db.due_alerts(self.conn, now, RETRY_FOR):
            kind = row["kind"]
            if kind == "digest":
                digest = messages.Message(row["title"], row["message"], priority=2, tags=("mailbox_with_mail",))
                self._deliver(row, digest, now, counts)
            elif kind == "heads_up":
                self._deliver(row, self._heads_up(row, live, history), now, counts)
            elif kind == "follow_60m":
                self._follow_60m(row, live, now, counts)
            elif kind == "follow_d1":
                self._follow_d1(row, now, counts)
        return counts

    def _heads_up(self, row: sqlite3.Row, live: PriceSource, history: HistoryLookup) -> messages.Message:
        info = post_info(
            self.conn,
            row["platform"],
            row["native_id"],
            post_tickers(self.conn, row["platform"], row["native_id"], self.symbols),
        )
        prices = {}
        for t in messages.order_tickers(info.tickers, self.holdings)[: messages.PRICE_TICKERS]:
            prices[t] = live.bars(t, info.created_at - PRICE_LOOKBACK, info.created_at + timedelta(minutes=1)).at(
                info.created_at
            )
        return messages.heads_up(info, self.holdings, prices, history(info.author))

    def _post(self, row: sqlite3.Row) -> tuple[PostInfo, date]:
        info = post_info(
            self.conn,
            row["platform"],
            row["native_id"],
            post_tickers(self.conn, row["platform"], row["native_id"], self.symbols),
        )
        return info, market.event_session(info.created_at)

    def _skip(self, row: sqlite3.Row, now: datetime, reason: str, counts: dict) -> None:
        with self.conn:
            db.mark_alert(self.conn, row["id"], "skipped", now, error=reason)
        counts["skipped"] += 1

    def _follow_60m(self, row: sqlite3.Row, live: PriceSource, now: datetime, counts: dict) -> None:
        if row["status"] == "pending" and now > from_iso(row["due_at"]) + timing.STALE_AFTER:
            self._skip(row, now, "stale: PC off or prices unavailable", counts)
            return
        info, d0 = self._post(row)
        ordered = tuple(messages.order_tickers(info.tickers, self.holdings))
        window = timing.follow_window(info.created_at, d0, self.cfg.followup_minutes)
        rows = followups.follow_60m_rows(ordered, window, live, lambda t: self.model(t, d0))
        if rows is None:
            counts["waiting"] += 1
            return
        msg = messages.follow_60m(info, self.holdings, rows, followups.window_text(window, d0))
        self._deliver(row, msg, now, counts)

    def _follow_d1(self, row: sqlite3.Row, now: datetime, counts: dict) -> None:
        # Only a row still waiting for bars gives up; a failed send keeps its 24 h of retries from the first failure.
        if row["status"] == "pending" and now > from_iso(row["due_at"]) + followups.D1_GIVE_UP:
            self._skip(row, now, "gave up: daily bars never arrived", counts)
            return
        info, d0 = self._post(row)
        ordered = tuple(messages.order_tickers(info.tickers, self.holdings))
        rows = followups.follow_d1_rows(
            self.conn, row["platform"], row["native_id"], ordered, d0, lambda t: self.model(t, d0)
        )
        if rows is None:
            counts["waiting"] += 1
            return
        self._deliver(row, messages.follow_d1(info, self.holdings, rows), now, counts)

    def _deliver(self, row: sqlite3.Row, message: messages.Message, now: datetime, counts: dict) -> None:
        try:
            self.notifier.send(message)
        except NotifyError as e:
            with self.conn:
                db.mark_alert(
                    self.conn, row["id"], "failed", now, title=message.title, message=message.body, error=str(e)
                )
            counts["failed"] += 1
            log.warning("alerts: %s for %s failed: %s", row["kind"], row["native_id"], e)
            return
        with self.conn:
            db.mark_alert(self.conn, row["id"], "sent", now, title=message.title, message=message.body)
        counts["sent"] += 1

    def run(self, now: datetime) -> dict:
        initialized = self.initialize(now)
        counts = {"status": "ok", "initialized": initialized, **self.queue(now), **self.send_due(now)}
        if counts["failed"]:
            counts["status"] = "partial"
        return counts
