from __future__ import annotations

import math
from collections.abc import Iterable
from datetime import UTC, date, datetime, time, timedelta

import pandas as pd
import pytest
from yfinance.exceptions import YFPricesMissingError, YFRateLimitError

from influence_tracker import market, prices
from influence_tracker.config import Watchlist
from influence_tracker.timeutil import NY, utc_now

# 04:00, 09:30, 15:59 and 19:59 New York: the first/last extended bar and the regular open/last minute.
SPARSE_MINUTES = (4 * 60, 9 * 60 + 30, 15 * 60 + 59, 19 * 60 + 59)
WINDOW_SEP = [  # sessions for fixed_now (2026-09-24 18:30 NY) with backfill_days=29: Aug 26 .. Sep 23
    date(2026, 8, 26), date(2026, 8, 27), date(2026, 8, 28), date(2026, 8, 31),
    date(2026, 9, 1), date(2026, 9, 2), date(2026, 9, 3), date(2026, 9, 4),
    date(2026, 9, 8), date(2026, 9, 9), date(2026, 9, 10), date(2026, 9, 11),
    date(2026, 9, 14), date(2026, 9, 15), date(2026, 9, 16), date(2026, 9, 17),
    date(2026, 9, 18), date(2026, 9, 21), date(2026, 9, 22), date(2026, 9, 23),
]  # fmt: skip
WINDOW_3 = ["2026-09-21", "2026-09-22", "2026-09-23"]  # fixed_now with backfill_days=3


def frame(epochs: list[int]) -> pd.DataFrame:
    """A frame shaped like yfinance's 1m history(): bar-start index in exchange time, int volume."""
    index = pd.to_datetime(epochs, unit="s", utc=True).tz_convert("America/New_York")
    base = [100.0 + i * 0.01 for i in range(len(epochs))]
    df = pd.DataFrame(
        {
            "Open": base,
            "High": [b + 0.05 for b in base],
            "Low": [b - 0.05 for b in base],
            "Close": [b + 0.01 for b in base],
            "Adj Close": [b + 0.01 for b in base],
            "Volume": [1000] * len(epochs),
        },
        index=index,
    )
    df.index.name = "Datetime"
    return df


class FakeYahoo:
    """Synthetic 1m bars at the given New York minutes on every XNYS session inside [start, end).
    Like yfinance 1.7.0 with exceptions not hidden, a range holding no bars raises instead of returning empty."""

    def __init__(
        self,
        minutes: Iterable[int] = SPARSE_MINUTES,
        *,
        no_bars: Iterable[date] = (),
        errors: dict[str, list[BaseException | None]] | None = None,
        always_fail: dict[str, BaseException] | None = None,
    ) -> None:
        self.minutes = tuple(minutes)
        self.no_bars = set(no_bars)
        self.errors = errors or {}
        self.always_fail = always_fail or {}
        self.calls: list[tuple[str, datetime, datetime]] = []

    def __call__(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        self.calls.append((symbol, start, end))
        queued = self.errors.get(symbol)
        if queued:
            exc = queued.pop(0)
            if exc is not None:
                raise exc
        if symbol in self.always_fail:
            raise self.always_fail[symbol]
        epochs = []
        for s in market.sessions_in_range(start.astimezone(NY).date(), end.astimezone(NY).date()):
            if s in self.no_bars:
                continue
            for m in self.minutes:
                t = datetime.combine(s, time(m // 60, m % 60), tzinfo=NY)
                if start <= t < end:
                    epochs.append(int(t.timestamp()))
        if not epochs:
            raise YFPricesMissingError(symbol, f" (1m {start.astimezone(NY)} -> {end.astimezone(NY)})")
        return frame(epochs)

    def symbols(self) -> list[str]:
        return [c[0] for c in self.calls]


class Sleeps(list):
    def __call__(self, seconds: float) -> None:
        self.append(seconds)


def subset(watchlist: Watchlist, symbols: list[str], **price_settings) -> Watchlist:
    tickers = [t for s in symbols for t in watchlist.tickers if t.symbol == s]
    assert [t.symbol for t in tickers] == symbols
    return watchlist.model_copy(
        update={"tickers": tickers, "prices": watchlist.prices.model_copy(update=price_settings)}
    )


def stored_sessions(conn, symbol: str) -> dict[str, int]:
    rows = conn.execute(
        "SELECT session_date, n_bars FROM bars_1m_sessions WHERE symbol = ? ORDER BY session_date", (symbol,)
    ).fetchall()
    return {r["session_date"]: r["n_bars"] for r in rows}


def stored_ts(conn, symbol: str) -> list[int]:
    return [r["ts"] for r in conn.execute("SELECT ts FROM bars_1m WHERE symbol = ? ORDER BY ts", (symbol,))]


def ny_epoch(d: date, hh: int, mm: int) -> int:
    return int(datetime.combine(d, time(hh, mm), tzinfo=NY).timestamp())


def iso(sessions: Iterable[date]) -> list[str]:
    return [s.isoformat() for s in sessions]


def run(conn, wl: Watchlist, now: datetime, fake, sleeps: Sleeps | None = None) -> dict:
    return prices.snapshot_1m(conn, wl, run_id=1, now=now, fetch=fake, sleep=sleeps if sleeps is not None else Sleeps())


# ---------------------------------------------------------------- window


def test_window_ends_at_last_completed_session(conn, watchlist, fixed_now):
    fake = FakeYahoo()
    counts = run(conn, subset(watchlist, ["SPY"]), fixed_now, fake)

    assert counts == {
        "status": "ok",
        "symbols": 1,
        "requests": len(fake.calls),
        "bars": 20 * len(SPARSE_MINUTES),
        "sessions_marked": 20,
        "symbols_failed": [],
    }
    assert list(stored_sessions(conn, "SPY")) == iso(WINDOW_SEP)
    assert fake.calls[0][1] == market.extended_bounds_utc(date(2026, 8, 26))[0]
    assert max(end for _, _, end in fake.calls) == market.extended_bounds_utc(date(2026, 9, 23))[1]
    assert max(stored_ts(conn, "SPY")) < ny_epoch(date(2026, 9, 24), 4, 0)


def test_todays_session_is_not_fetched_until_20_ny(conn, watchlist):
    wl = subset(watchlist, ["SPY"], backfill_days=3)
    today = date(2026, 9, 24)

    early = FakeYahoo()
    run(conn, wl, datetime(2026, 9, 24, 19, 59, tzinfo=NY), early)
    assert all(end <= datetime(2026, 9, 24, 19, 59, tzinfo=NY) for _, _, end in early.calls)
    assert today.isoformat() not in stored_sessions(conn, "SPY")

    at_close = FakeYahoo()
    counts = run(conn, wl, datetime(2026, 9, 24, 20, 0, tzinfo=NY), at_close)
    assert at_close.calls == [("SPY", *market.extended_bounds_utc(today))]
    assert counts["sessions_marked"] == 1
    assert stored_sessions(conn, "SPY")[today.isoformat()] == len(SPARSE_MINUTES)


def test_no_completed_session_in_window_makes_no_requests(conn, watchlist):
    # Monday 08:00 NY with a 1-day backfill: the window is Sunday..Friday's session, which is out of range.
    fake = FakeYahoo()
    counts = run(conn, subset(watchlist, ["SPY"], backfill_days=1), datetime(2026, 9, 28, 8, 0, tzinfo=NY), fake)
    assert fake.calls == []
    assert counts["status"] == "ok" and counts["requests"] == 0


def test_every_configured_ticker_is_requested(conn, watchlist, fixed_now):
    fake = FakeYahoo()
    counts = run(
        conn,
        watchlist.model_copy(update={"prices": watchlist.prices.model_copy(update={"backfill_days": 1})}),
        fixed_now,
        fake,
    )

    assert sorted(fake.symbols()) == sorted(t.yahoo_symbol for t in watchlist.tickers)
    assert {"SPY", "QQQ", "BRK-B"} <= set(fake.symbols())
    assert counts["symbols"] == len(watchlist.tickers)
    assert counts["requests"] == len(watchlist.tickers)
    assert counts["status"] == "ok"


# ---------------------------------------------------------------- chunking


def test_chunks_respect_the_request_span_limit(conn, watchlist, fixed_now):
    fake = FakeYahoo()
    run(conn, subset(watchlist, ["SPY"]), fixed_now, fake)

    assert prices.MAX_REQUEST_SPAN <= timedelta(days=8)  # Yahoo's hard limit
    assert all(end - start <= prices.MAX_REQUEST_SPAN for _, start, end in fake.calls)
    covered = [
        s for _, start, end in fake.calls
        for s in market.sessions_in_range(start.astimezone(NY).date(), end.astimezone(NY).date())
    ]  # fmt: skip
    assert covered == WINDOW_SEP  # every session exactly once, in order
    assert len(fake.calls) == 5


def test_chunk_sessions_splits_on_present_sessions_and_span():
    window = market.sessions_in_range(date(2026, 9, 14), date(2026, 9, 25))
    missing = [s for s in window if s != date(2026, 9, 17)]
    chunks = prices.chunk_sessions(window, missing)
    assert chunks == [
        [date(2026, 9, 14), date(2026, 9, 15), date(2026, 9, 16)],
        [date(2026, 9, 18), date(2026, 9, 21), date(2026, 9, 22), date(2026, 9, 23), date(2026, 9, 24)],
        [date(2026, 9, 25)],
    ]
    for chunk in chunks:
        assert market.extended_bounds_utc(chunk[-1])[1] - market.extended_bounds_utc(chunk[0])[0] <= timedelta(days=7)


def test_chunk_sessions_across_dst_and_small_span():
    window = market.sessions_in_range(date(2026, 10, 27), date(2026, 11, 3))
    # Tue 04:00 EDT -> Mon 20:00 EST is 6 days 17 hours: fits; one more session does not.
    assert prices.chunk_sessions(window, window) == [window[:5], window[5:]]
    assert prices.chunk_sessions(window, window, max_span=timedelta(hours=16)) == [[s] for s in window]
    assert prices.chunk_sessions(window, []) == []


# ---------------------------------------------------------------- bar assignment


def test_bars_assigned_to_sessions_across_dst(conn, watchlist):
    # 21:00 EST on Mon Nov 2 2026, the first session after the switch back from EDT.
    now = datetime(2026, 11, 3, 2, 0, tzinfo=UTC)
    minutes = (3 * 60 + 59, 4 * 60, 19 * 60 + 59, 20 * 60)  # 03:59 and 20:00 are outside the extended day
    fake = FakeYahoo(minutes)
    run(conn, subset(watchlist, ["SPY"], backfill_days=5), now, fake)

    sessions = [date(2026, 10, 28), date(2026, 10, 29), date(2026, 10, 30), date(2026, 11, 2)]
    assert stored_sessions(conn, "SPY") == {s.isoformat(): 2 for s in sessions}
    expected = sorted(ny_epoch(s, hh, mm) for s in sessions for hh, mm in ((4, 0), (19, 59)))
    assert stored_ts(conn, "SPY") == expected
    fri, mon = date(2026, 10, 30), date(2026, 11, 2)
    assert ny_epoch(fri, 4, 0) == int(datetime(2026, 10, 30, 8, 0, tzinfo=UTC).timestamp())
    assert ny_epoch(fri, 19, 59) == int(datetime(2026, 10, 30, 23, 59, tzinfo=UTC).timestamp())
    assert ny_epoch(mon, 4, 0) == int(datetime(2026, 11, 2, 9, 0, tzinfo=UTC).timestamp())
    # 19:59 EST on Nov 2 is already Nov 3 in UTC but still belongs to the Nov 2 session.
    assert ny_epoch(mon, 19, 59) == int(datetime(2026, 11, 3, 0, 59, tzinfo=UTC).timestamp())


def test_bars_by_session_drops_nan_and_duplicates_and_keeps_bar_start():
    s = date(2026, 9, 23)
    epochs = [ny_epoch(s, 9, 30), ny_epoch(s, 9, 31), ny_epoch(s, 9, 31), ny_epoch(s, 9, 32)]
    df = frame(epochs)
    df.iloc[0, df.columns.get_loc("Close")] = math.nan
    df["Volume"] = df["Volume"].astype(float)
    df.iloc[3, df.columns.get_loc("Volume")] = math.nan

    out = prices.bars_by_session(df, [s])
    assert [row[0] for row in out[s]] == [ny_epoch(s, 9, 31), ny_epoch(s, 9, 32)]
    assert out[s][0][1:] == pytest.approx((100.01, 100.06, 99.96, 100.02, 1000.0))
    assert out[s][1][5] is None
    assert all(type(x) is float for x in out[s][0][1:]) and type(out[s][0][0]) is int


def test_bars_by_session_ignores_sessions_outside_the_chunk():
    inside, outside = date(2026, 9, 22), date(2026, 9, 23)
    df = frame([ny_epoch(inside, 10, 0), ny_epoch(outside, 10, 0)])
    out = prices.bars_by_session(df, [inside])
    assert list(out) == [inside]
    assert [row[0] for row in out[inside]] == [ny_epoch(inside, 10, 0)]
    assert prices.bars_by_session(frame([]), [inside]) == {inside: []}


@pytest.mark.parametrize(
    "bad",
    [
        pd.DataFrame({"Open": [1.0], "Close": [1.0]}, index=pd.DatetimeIndex(["2026-09-23 14:00"], tz="UTC")),
        pd.DataFrame(
            {"Open": [1.0], "High": [1.0], "Low": [1.0], "Close": [1.0], "Volume": [1]},
            index=pd.DatetimeIndex(["2026-09-23 14:00"]),
        ),
    ],
    ids=["missing-columns", "naive-index"],
)
def test_bars_by_session_rejects_malformed_frames(bad):
    with pytest.raises(ValueError):
        prices.bars_by_session(bad, [date(2026, 9, 23)])


# ---------------------------------------------------------------- idempotence and retries


def test_rerun_is_idempotent_and_makes_no_requests(conn, watchlist, fixed_now):
    wl = subset(watchlist, ["SPY", "QQQ"])
    run(conn, wl, fixed_now, FakeYahoo())
    before = (stored_ts(conn, "SPY"), stored_ts(conn, "QQQ"))

    again = FakeYahoo()
    counts = run(conn, wl, fixed_now + timedelta(minutes=5), again)
    assert again.calls == []
    assert counts == {
        "status": "ok", "symbols": 2, "requests": 0, "bars": 0, "sessions_marked": 0, "symbols_failed": []
    }  # fmt: skip
    assert (stored_ts(conn, "SPY"), stored_ts(conn, "QQQ")) == before


def test_zero_bar_sessions_are_retried_next_run(conn, watchlist, fixed_now):
    wl = subset(watchlist, ["SPY"])
    gap = date(2026, 9, 22)
    counts = run(conn, wl, fixed_now, FakeYahoo(no_bars=[gap]))
    assert counts["status"] == "ok"
    assert stored_sessions(conn, "SPY")[gap.isoformat()] == 0
    assert counts["sessions_marked"] == 20

    retry = FakeYahoo()
    run(conn, wl, fixed_now + timedelta(hours=1), retry)
    assert retry.calls == [("SPY", *market.extended_bounds_utc(gap))]
    assert stored_sessions(conn, "SPY")[gap.isoformat()] == len(SPARSE_MINUTES)


def test_zero_bar_session_does_not_hold_back_newer_sessions(conn, watchlist, fixed_now):
    # A full-day halt: Yahoo never has bars for Sep 22, so every retry of it comes back as "no price data".
    wl = subset(watchlist, ["SPY"])
    halt = date(2026, 9, 22)
    run(conn, wl, fixed_now, FakeYahoo(no_bars=[halt]))
    assert stored_sessions(conn, "SPY")[halt.isoformat()] == 0

    fake = FakeYahoo(no_bars=[halt])
    counts = run(conn, wl, datetime(2026, 9, 26, 1, 0, tzinfo=UTC), fake)  # Fri Sep 25 21:00 NY
    assert fake.calls == [
        ("SPY", *market.extended_bounds_utc(halt)),
        ("SPY", market.extended_bounds_utc(date(2026, 9, 24))[0], market.extended_bounds_utc(date(2026, 9, 25))[1]),
    ]
    assert counts["status"] == "ok" and counts["symbols_failed"] == []
    assert counts["sessions_marked"] == 3
    sessions = stored_sessions(conn, "SPY")
    assert sessions[halt.isoformat()] == 0
    assert sessions["2026-09-24"] == sessions["2026-09-25"] == len(SPARSE_MINUTES)


def test_sessions_before_a_listing_do_not_hold_back_later_ones(conn, watchlist, fixed_now, caplog):
    # A ticker that first traded on Sep 10 and joined the watchlist on Sep 24.
    listed = date(2026, 9, 10)
    unlisted = [s for s in market.sessions_in_range(date(2026, 8, 1), listed) if s < listed]
    wl = subset(watchlist, ["TSLA"])
    fake = FakeYahoo(no_bars=unlisted)
    with caplog.at_level("WARNING", logger=prices.__name__):
        counts = run(conn, wl, fixed_now, fake)

    assert counts["status"] == "ok" and counts["symbols_failed"] == []
    assert len(fake.calls) == 5
    assert stored_sessions(conn, "TSLA") == {
        s.isoformat(): 0 if s < listed else len(SPARSE_MINUTES) for s in WINDOW_SEP
    }
    assert len(caplog.records) == 1 and "2026-08-26" in caplog.text and "2026-09-09" in caplog.text

    fake = FakeYahoo(no_bars=unlisted)
    counts = run(conn, wl, datetime(2026, 9, 26, 1, 0, tzinfo=UTC), fake)  # Fri Sep 25 21:00 NY
    assert counts["status"] == "ok"
    assert fake.calls[-1] == (
        "TSLA", market.extended_bounds_utc(date(2026, 9, 24))[0], market.extended_bounds_utc(date(2026, 9, 25))[1]
    )  # fmt: skip
    sessions = stored_sessions(conn, "TSLA")
    assert sessions["2026-09-24"] == sessions["2026-09-25"] == len(SPARSE_MINUTES)


def test_failed_chunk_keeps_the_chunks_around_it(conn, watchlist, fixed_now):
    wl = subset(watchlist, ["SPY"])
    fake = FakeYahoo(errors={"SPY": [None, ConnectionError("reset")]})
    counts = run(conn, wl, fixed_now, fake)

    assert counts["status"] == "partial" and counts["symbols_failed"] == ["SPY"]
    assert len(fake.calls) == 5
    _, start, end = fake.calls[1]
    failed_chunk = market.sessions_in_range(start.astimezone(NY).date(), end.astimezone(NY).date())
    assert failed_chunk == WINDOW_SEP[5:9]
    assert list(stored_sessions(conn, "SPY")) == iso(WINDOW_SEP[:5] + WINDOW_SEP[9:])

    retry = FakeYahoo()
    counts = run(conn, wl, fixed_now, retry)
    assert counts["status"] == "ok"
    assert retry.calls == [fake.calls[1]]
    assert list(stored_sessions(conn, "SPY")) == iso(WINDOW_SEP)


def test_brk_b_requested_as_brk_dash_b_stored_as_brk_dot_b(conn, watchlist, fixed_now):
    fake = FakeYahoo()
    run(conn, subset(watchlist, ["BRK.B"], backfill_days=3), fixed_now, fake)
    assert set(fake.symbols()) == {"BRK-B"}
    assert stored_ts(conn, "BRK.B")
    assert list(stored_sessions(conn, "BRK.B")) == WINDOW_3
    assert stored_ts(conn, "BRK-B") == [] and stored_sessions(conn, "BRK-B") == {}


@pytest.mark.parametrize(
    "error",
    [
        YFPricesMissingError("TSLA", " (1m ...)", yahoo_reason="No data found, symbol may be delisted"),
        ConnectionError("connection reset"),
        TimeoutError("read timed out"),
    ],
    ids=["yahoo-explained", "connection", "timeout"],
)
def test_one_symbol_failing_does_not_stop_the_others(conn, watchlist, fixed_now, error):
    fake = FakeYahoo(always_fail={"TSLA": error})
    counts = run(conn, subset(watchlist, ["SPY", "TSLA", "QQQ"]), fixed_now, fake)

    assert counts["status"] == "partial"
    assert counts["symbols_failed"] == ["TSLA"]
    assert fake.symbols().count("TSLA") == fake.symbols().count("SPY") == 5  # each chunk is tried once
    assert list(stored_sessions(conn, "SPY")) == iso(WINDOW_SEP)
    assert list(stored_sessions(conn, "QQQ")) == iso(WINDOW_SEP)
    assert stored_sessions(conn, "TSLA") == {} and stored_ts(conn, "TSLA") == []


def test_malformed_frame_counts_as_a_failed_symbol(conn, watchlist, fixed_now):
    fake = FakeYahoo()

    def fetch(symbol, start, end):
        df = fake(symbol, start, end)
        return df.drop(columns=["Close"]) if symbol == "SPY" else df

    counts = prices.snapshot_1m(
        conn, subset(watchlist, ["SPY", "QQQ"], backfill_days=3), 1, fixed_now, fetch=fetch, sleep=Sleeps()
    )
    assert counts["status"] == "partial" and counts["symbols_failed"] == ["SPY"]
    assert stored_sessions(conn, "SPY") == {}
    assert list(stored_sessions(conn, "QQQ")) == WINDOW_3


def test_all_symbols_failing_is_an_error(conn, watchlist, fixed_now):
    boom = ConnectionError("offline")
    fake = FakeYahoo(always_fail={"SPY": boom, "QQQ": boom})
    counts = run(conn, subset(watchlist, ["SPY", "QQQ"], backfill_days=3), fixed_now, fake)
    assert counts["status"] == "error"
    assert counts["symbols_failed"] == ["SPY", "QQQ"]
    assert counts["requests"] == 2 and counts["bars"] == 0 and counts["sessions_marked"] == 0


def test_requests_are_paced(conn, watchlist, fixed_now):
    sleeps = Sleeps()
    fake = FakeYahoo()
    counts = run(conn, subset(watchlist, ["SPY", "QQQ"], request_pause_s=1.5), fixed_now, fake, sleeps)
    assert counts["requests"] == len(fake.calls) == 10
    assert sleeps == [1.5] * 9  # between requests, not before the first


def test_rate_limit_waits_60s_and_retries_once(conn, watchlist, fixed_now):
    sleeps = Sleeps()
    fake = FakeYahoo(errors={"SPY": [YFRateLimitError()]})
    counts = run(conn, subset(watchlist, ["SPY"], backfill_days=3), fixed_now, fake, sleeps)

    assert counts["status"] == "ok"
    assert counts["requests"] == 2
    assert fake.calls[0] == fake.calls[1]
    assert sleeps == [60.0]
    assert list(stored_sessions(conn, "SPY")) == WINDOW_3


def test_persistent_rate_limit_stops_the_run(conn, watchlist, fixed_now):
    sleeps = Sleeps()
    fake = FakeYahoo(errors={"SPY": [YFRateLimitError(), YFRateLimitError()]})
    counts = run(conn, subset(watchlist, ["QQQ", "SPY", "TSLA"], backfill_days=3), fixed_now, fake, sleeps)

    assert fake.symbols() == ["QQQ", "SPY", "SPY"]  # TSLA is not tried while Yahoo is throttling
    assert counts["status"] == "partial"
    assert counts["symbols_failed"] == ["SPY", "TSLA"]
    assert sleeps == [1.0, 60.0]
    assert list(stored_sessions(conn, "QQQ")) == WINDOW_3
    assert stored_sessions(conn, "SPY") == {} and stored_sessions(conn, "TSLA") == {}


def test_rate_limit_mid_symbol_keeps_its_earlier_chunks(conn, watchlist, fixed_now):
    fake = FakeYahoo(errors={"SPY": [None, YFRateLimitError(), YFRateLimitError()]})
    counts = run(conn, subset(watchlist, ["SPY", "QQQ"]), fixed_now, fake)

    assert fake.symbols() == ["SPY"] * 3
    assert counts["status"] == "partial" and counts["symbols_failed"] == ["SPY", "QQQ"]
    assert list(stored_sessions(conn, "SPY")) == iso(WINDOW_SEP[:5])


def test_run_that_stores_nothing_is_an_error(conn, watchlist, fixed_now):
    wl = subset(watchlist, ["SPY", "QQQ"], backfill_days=3)
    assert run(conn, wl, fixed_now, FakeYahoo(always_fail={"QQQ": ConnectionError("x")}))["status"] == "partial"
    # SPY is already complete, so the second run's only work was QQQ, and it failed.
    counts = run(conn, wl, fixed_now, FakeYahoo(always_fail={"QQQ": ConnectionError("x")}))
    assert counts["status"] == "error" and counts["symbols_failed"] == ["QQQ"]
    assert counts["requests"] == 1


# ---------------------------------------------------------------- live


@pytest.mark.live
def test_live_snapshot_spy_and_brk_b(conn, watchlist):
    """Two real Yahoo requests: the last ~3 calendar days of SPY and BRK-B 1m bars."""
    wl = subset(watchlist, ["SPY", "BRK.B"], backfill_days=3)
    now = utc_now()
    counts = prices.snapshot_1m(conn, wl, run_id=0, now=now)
    assert counts["status"] == "ok", counts
    window = prices.snapshot_window(now, 3)
    assert window
    for symbol in ("SPY", "BRK.B"):
        sessions = stored_sessions(conn, symbol)
        assert list(sessions) == iso(window)
        assert all(n > 300 for n in sessions.values()), sessions
        local = [datetime.fromtimestamp(ts, UTC).astimezone(NY) for ts in stored_ts(conn, symbol)]
        assert all(t.second == 0 and time(4, 0) <= t.time() < time(20, 0) for t in local)
        assert sum(sessions.values()) == len(local)
        if symbol == "SPY":
            for s in window:
                day = [t.time() for t in local if t.date() == s]
                # labelled by bar start: the regular session opens with a 09:30 bar and after-hours ends at 19:59
                assert time(9, 30) in day and min(day) < time(4, 10) and max(day) > time(19, 50)
    assert prices.snapshot_1m(conn, wl, run_id=0, now=now)["requests"] == 0


@pytest.mark.live
def test_live_range_without_bars_is_reported_without_a_yahoo_reason():
    """One real Yahoo request: 00:00-03:59 New York holds no bars, which snapshot_1m must read as 'no bars'."""
    session = prices.snapshot_window(utc_now(), 5)[-1]
    with pytest.raises(YFPricesMissingError) as info:
        prices.fetch_yahoo_1m(
            "SPY", datetime.combine(session, time(0, 0), tzinfo=NY), datetime.combine(session, time(3, 59), tzinfo=NY)
        )
    assert info.value.yahoo_reason is None
