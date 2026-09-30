from __future__ import annotations

import math
from datetime import date

import numpy as np
import pytest

from influence_tracker import market
from influence_tracker.analysis import metrics


def _write_daily(conn, symbol, sessions, closes):
    conn.executemany(
        """INSERT OR REPLACE INTO bars_1d (symbol, session_date, open, high, low, close, adj_close, volume,
                                           split_ratio, fetched_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, 1000000, 0, '2026-01-01T00:00:00Z')""",
        [(symbol, s.isoformat(), c, c, c, c, c) for s, c in zip(sessions, closes, strict=True)],
    )


@pytest.fixture
def synthetic(conn):
    rng = np.random.default_rng(7)
    sessions = market.sessions_in_range(date(2025, 6, 2), date(2026, 6, 30))
    m = rng.normal(0, 0.01, len(sessions))
    e = rng.normal(0, 0.004, len(sessions))
    r = 0.0002 + 1.5 * m + e
    k = 200
    r[k] += 0.05
    with conn:
        _write_daily(conn, "SPY", sessions, 400 * np.cumprod(1 + m))
        _write_daily(conn, "NVDA", sessions, 100 * np.cumprod(1 + r))
    return sessions, k, m, r


def test_event_model_recovers_beta_and_the_jump(conn, synthetic):
    sessions, k, m, r = synthetic
    model = metrics.event_model(conn, "NVDA", sessions[k])
    assert model is not None
    assert model.n_est == 120
    # Checked against OLS on the estimation sample (sessions -130..-11), not the true 1.5: with this seed that sample's
    # noise leans against the market, so its beta is 1.396, 2.5 standard errors low.
    x, y = m[k - 130 : k - 10], r[k - 130 : k - 10]
    beta, alpha = np.polyfit(x, y, 1)
    resid = y - alpha - beta * x
    sigma = math.sqrt(resid @ resid / (len(x) - 2))
    assert (model.alpha, model.beta, model.sigma) == pytest.approx((alpha, beta, sigma))
    assert model.ar[0] == pytest.approx(r[k] - alpha - beta * m[k])
    assert model.ar[0] == pytest.approx(0.05, abs=0.015)
    car, z = model.car(0, 0)
    assert car == pytest.approx(model.ar[0])
    assert z > 5
    two_days = model.ar[0] + model.ar[1]
    assert model.car(0, 1) == pytest.approx((two_days, two_days / (sigma * math.sqrt(2))))


def test_car_is_nan_when_a_day_is_missing(conn, synthetic):
    sessions, k, *_ = synthetic
    with conn:
        conn.execute("DELETE FROM bars_1d WHERE symbol = 'NVDA' AND session_date = ?", (sessions[k + 1].isoformat(),))
    model = metrics.event_model(conn, "NVDA", sessions[k])
    car, z = model.car(0, 1)
    assert np.isnan(car) and np.isnan(z)


def test_no_model_without_history(conn, synthetic):
    sessions, *_ = synthetic
    assert metrics.event_model(conn, "NVDA", sessions[30]) is None
    assert metrics.event_model(conn, "AAPL", sessions[200]) is None
