from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from influence_tracker.alerts.live import LivePrices, frame_to_bars


def _frame(rows):
    idx = pd.DatetimeIndex([pd.Timestamp(t, tz="America/New_York") for t, _ in rows])
    return pd.DataFrame({"Close": [c for _, c in rows]}, index=idx)


def test_frame_to_bars_uses_bar_starts_and_drops_bad_rows():
    df = _frame([("2026-09-29 10:00", 10.0), ("2026-09-29 10:01", float("nan")), ("2026-09-29 10:02", 11.0)])
    bars = frame_to_bars(df)
    assert bars.starts == [
        int(datetime(2026, 9, 29, 14, 0, tzinfo=UTC).timestamp()),
        int(datetime(2026, 9, 29, 14, 2, tzinfo=UTC).timestamp()),
    ]
    assert bars.closes == [10.0, 11.0]
    assert frame_to_bars(pd.DataFrame()).starts == []


def test_frame_to_bars_sorts_drops_non_positive_and_keeps_the_last_duplicate():
    df = _frame(
        [("2026-09-29 10:02", 11.0), ("2026-09-29 10:00", 0.0), ("2026-09-29 10:01", 9.0), ("2026-09-29 10:02", 12.0)]
    )
    bars = frame_to_bars(df)
    assert bars.starts == [int(datetime(2026, 9, 29, 14, m, tzinfo=UTC).timestamp()) for m in (1, 2)]
    assert bars.closes == [9.0, 12.0]


def test_live_prices_maps_symbols_caches_and_survives_errors(watchlist):
    calls = []

    def fetch(symbol, start, end):
        calls.append(symbol)
        if symbol == "BRK-B":
            raise RuntimeError("yahoo down")
        return _frame([("2026-09-29 10:00", 10.0)])

    live = LivePrices(watchlist, fetch=fetch)
    s, e = datetime(2026, 9, 29, 13, 0, tzinfo=UTC), datetime(2026, 9, 29, 15, 0, tzinfo=UTC)
    assert live.bars("NVDA", s, e).closes == [10.0]
    assert live.bars("NVDA", s, e).closes == [10.0]
    assert live.bars("BRK.B", s, e).starts == []
    assert calls == ["NVDA", "BRK-B"]
