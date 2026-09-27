from __future__ import annotations

import math
import statistics
import time as timer
from collections.abc import Callable, Iterable, Sequence
from datetime import UTC, date, datetime, time, timedelta

import numpy as np
import pytest
from scipy import stats

from influence_tracker import db, market
from influence_tracker.analysis import metrics
from influence_tracker.analysis.metrics import (
    AR_COLUMNS,
    PLACEBO_COLUMNS,
    compute_study,
    compute_study_with_placebo,
    holm_adjust,
    mean_ci,
    next_session_after,
)
from influence_tracker.analysis.study import (
    ATTENTION_COLUMNS,
    DAILY_WINDOWS,
    EVENT_COLUMNS,
    GROUP_COLUMNS,
    INTRADAY_PATH_COLUMNS,
    MAGNITUDE_COLUMNS,
    PATH_COLUMNS,
    PATH_DAYS,
    POST_COLUMNS,
    REDDIT_CATEGORY,
    Study,
    StudyOptions,
)
from influence_tracker.events import MinuteBars, compute_windows
from influence_tracker.timeutil import NY, to_iso

GRID: list[date] = market.sessions_in_range(date(2025, 1, 2), date(2026, 12, 31))
POS = {d: i for i, d in enumerate(GRID)}
NOW = datetime(2027, 1, 15, 12, 0, tzinfo=UTC)
D0 = date(2026, 7, 15)  # a Wednesday with ~380 sessions of synthetic history before it
COUNT_KEYS = {
    "events_complete", "events_included", "excluded_earnings", "excluded_split", "excluded_clustered",
    "excluded_no_model", "earnings_unknown", "events_intraday", "posts_included", "posts_signed", "placebo_days",
    "attention_days",
}  # fmt: skip


def rel(d: date, k: int) -> date:
    return GRID[POS[d] + k]


class Synth:
    """Synthetic rows on the real schema: daily bars on the real XNYS grid, posts, events and event windows."""

    def __init__(self, conn, seed: int = 7) -> None:
        self.conn = conn
        self.rng = np.random.default_rng(seed)
        self.market = self.rng.normal(0.0003, 0.01, len(GRID))
        self.daily("SPY", self.market)

    def returns(
        self,
        alpha: float = 0.0005,
        beta: float = 1.2,
        noise: float = 0.01,
        quiet: Iterable[date] = (),
        jumps: dict[date, float] | None = None,
    ) -> np.ndarray:
        """r = alpha + beta * r_SPY + noise; `quiet` zeroes the noise on sessions -5..+5 around each date."""
        e = self.rng.normal(0.0, noise, len(GRID))
        for d in quiet:
            e[POS[d] - 5 : POS[d] + 6] = 0.0
        r = alpha + beta * self.market + e
        for d, jump in (jumps or {}).items():
            r[POS[d]] += jump
        return r

    def daily(
        self,
        symbol: str,
        returns: np.ndarray,
        volume: float | np.ndarray = 1e6,
        drop: Iterable[date] = (),
        first: date | None = None,
    ) -> None:
        prices = 100.0 * np.cumprod(1.0 + np.asarray(returns))
        volumes = np.broadcast_to(np.asarray(volume, dtype=float), (len(GRID),))
        skip = set(drop)
        rows = [
            (symbol, d.isoformat(), p, p, p, p, p, float(v), 0.0, to_iso(NOW))
            for d, p, v in zip(GRID, prices, volumes, strict=True)
            if d not in skip and (first is None or d >= first)
        ]
        with self.conn:
            self.conn.executemany(
                """INSERT OR REPLACE INTO bars_1d (symbol, session_date, open, high, low, close, adj_close, volume,
                                                   split_ratio, fetched_at) VALUES (?,?,?,?,?,?,?,?,?,?)""",
                rows,
            )

    def post(
        self,
        native_id: str,
        *,
        t0: datetime,
        platform: str = "truthsocial",
        author: str = "realDonaldTrump",
        stance: str | None = "bullish",
        conf: float | None = 0.9,
        source: str | None = None,
    ) -> None:
        with self.conn:
            self.conn.execute(
                """INSERT INTO posts (platform, native_id, author, created_at_utc, text, url, source, stance,
                                      stance_conf, collected_at) VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (platform, native_id, author, to_iso(t0), f"post {native_id}", f"https://example.test/{native_id}",
                 source, stance, conf, to_iso(NOW)),
            )  # fmt: skip

    def event(
        self,
        native_id: str,
        ticker: str,
        d0: date,
        *,
        t0: datetime,
        platform: str = "truthsocial",
        status: str = "complete",
        intraday: str | None = "unavailable",
        earnings: int | None = 0,
        split: int = 0,
        clustered: int = 0,
        phase: str = "after",
    ) -> int:
        with self.conn:
            cur = self.conn.execute(
                """INSERT INTO events (platform, native_id, ticker, t0, d0, session_phase, status, intraday_state,
                                       earnings_flag, split_flag, clustered, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (platform, native_id, ticker, to_iso(t0), d0.isoformat(), phase, status,
                 intraday if status == "complete" else None, earnings, split, clustered, to_iso(NOW)),
            )  # fmt: skip
        return int(cur.lastrowid)

    def add(
        self,
        native_id: str,
        tickers: Sequence[str],
        d0: date,
        *,
        stance: str | None = "bullish",
        conf: float | None = 0.9,
        platform: str = "truthsocial",
        author: str = "realDonaldTrump",
        source: str | None = None,
        **flags,
    ) -> list[int]:
        """One post (made after the close the session before d0) with an event per ticker."""
        t0 = datetime.combine(market.previous_session(d0), time(21, 30), tzinfo=UTC)
        self.post(native_id, t0=t0, platform=platform, author=author, stance=stance, conf=conf, source=source)
        return [self.event(native_id, t, d0, t0=t0, platform=platform, **flags) for t in tickers]

    def window(self, event_id: int, win: str, ret: float, spy_ret: float | None) -> None:
        with self.conn:
            self.conn.execute(
                """INSERT INTO event_windows (event_id, win, start_ts, end_ts, start_price, end_price, ret,
                                              spy_start_price, spy_end_price, spy_ret, truncated)
                   VALUES (?,?,0,0,100,?,?,400,?,?,0)""",
                (event_id, win, 100 * (1 + ret), ret, None if spy_ret is None else 400 * (1 + spy_ret), spy_ret),
            )


def ols(y: np.ndarray, x: np.ndarray) -> tuple[float, float, float]:
    """Independent OLS: (alpha, beta, residual std with n - 2 degrees of freedom)."""
    design = np.column_stack([np.ones(len(x)), x])
    (a, b), *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - a - b * x
    return float(a), float(b), math.sqrt(float(resid @ resid) / (len(x) - 2))


def model_at(r: np.ndarray, m: np.ndarray, anchor: date) -> tuple[float, float, float]:
    """The market model a study should fit for an event on `anchor`: sessions -130..-11 inclusive."""
    i = POS[anchor]
    return ols(r[i - 130 : i - 10], m[i - 130 : i - 10])


def portfolio_z(returns: Sequence[np.ndarray], m: np.ndarray, anchor: date, window: str) -> tuple[float, int]:
    """Independent post z of an equal-weighted portfolio of stocks posted about on `anchor`, and the number of
    estimation days all stocks share. NaN in a return marks a missing day: each stock's market model uses its own
    days, the portfolio sigma (n - 2 degrees of freedom) only the shared ones."""
    i = POS[anchor]
    lo, hi = DAILY_WINDOWS[window]
    est = slice(i - 130, i - 10)
    residuals, cars = [], []
    for r in returns:
        have = np.isfinite(r[est])
        a, b, _ = ols(r[est][have], m[est][have])
        residuals.append(r[est] - a - b * m[est])
        cars.append(sum(r[i + k] - a - b * m[i + k] for k in range(lo, hi + 1)))
    stacked = np.array(residuals)
    shared = np.isfinite(stacked).all(axis=0)
    mean_resid = stacked[:, shared].mean(axis=0)
    sigma = math.sqrt(float(mean_resid @ mean_resid) / (len(mean_resid) - 2))
    return statistics.fmean(cars) / (sigma * math.sqrt(hi - lo + 1)), int(shared.sum())


def run(conn, watchlist, **options) -> Study:
    return compute_study(conn, watchlist, NOW, StudyOptions(**options))


def event_row(study: Study, event_id: int):
    return study.events.set_index("event_id").loc[event_id]


# ---------------------------------------------------------------- market model, ARs, CARs


def test_market_model_recovers_alpha_beta_sigma_over_exact_window(conn, watchlist):
    s = Synth(conn)
    r = s.returns(alpha=0.001, beta=1.5, noise=0.005)
    i = POS[D0]
    r[i - 10] = 0.30  # session -10: just after the estimation window
    r[i - 131] = -0.30  # session -131: just before it
    s.daily("AAPL", r)
    (eid,) = s.add("p1", ["AAPL"], D0)
    assert market.session_offset(D0, -130) == GRID[i - 130]  # the test grid is the real XNYS grid

    ev = event_row(run(conn, watchlist), eid)
    alpha, beta, sigma = model_at(r, s.market, D0)
    assert ev["n_est"] == 120
    assert ev["alpha"] == pytest.approx(alpha, rel=1e-9, abs=1e-12)
    assert ev["beta"] == pytest.approx(beta, rel=1e-9)
    assert ev["sigma"] == pytest.approx(sigma, rel=1e-9)
    assert abs(ev["beta"] - 1.5) < 0.2
    assert abs(ev["alpha"] - 0.001) < 0.002
    assert 0.004 < ev["sigma"] < 0.006
    for k, col in zip(PATH_DAYS, AR_COLUMNS, strict=True):
        assert ev[col] == pytest.approx(r[i + k] - (alpha + beta * s.market[i + k]), abs=1e-12)


def test_injected_jump_gives_large_event_z_and_post_leg(conn, watchlist):
    s = Synth(conn)
    r = s.returns(alpha=0.0, beta=2.0, noise=0.005, quiet=[D0], jumps={D0: 0.05})
    s.daily("AAPL", r)
    (eid,) = s.add("p1", ["AAPL"], D0, intraday="ok")
    s.window(eid, "pre_leg", 2 * -0.004, -0.004)
    s.window(eid, "post_leg", 2 * 0.003 + 0.03, 0.003)

    ev = event_row(run(conn, watchlist), eid)
    assert ev["excluded_reason"] == ""
    assert ev["beta"] == pytest.approx(2.0, abs=0.15)
    assert ev["ar_d0"] == pytest.approx(0.05, abs=0.005)
    assert abs(ev["car_pre"]) < 0.01
    assert ev["car_event"] == pytest.approx(ev["ar_d0"] + ev["ar_dp1"], abs=1e-15)
    assert ev["z_event"] == pytest.approx(ev["car_event"] / (ev["sigma"] * math.sqrt(2)))
    assert ev["z_pre"] == pytest.approx(ev["car_pre"] / (ev["sigma"] * math.sqrt(5)))
    assert ev["z_post"] == pytest.approx(ev["car_post"] / (ev["sigma"] * math.sqrt(4)))
    assert ev["z_event"] > 3
    assert ev["ar_post_leg"] == pytest.approx(2 * 0.003 + 0.03 - ev["beta"] * 0.003)
    assert ev["ar_post_leg"] == pytest.approx(0.03, abs=0.002)
    assert abs(ev["ar_pre_leg"]) < 0.002
    assert ev["z_post_leg"] == pytest.approx(ev["ar_post_leg"] / ev["sigma"])
    assert ev["z_post_leg"] > 3
    assert ev["spy_post_leg"] == pytest.approx(0.003)
    assert ev["spy_ret_d0"] == pytest.approx(s.market[POS[D0]])


def test_no_jump_control_is_not_flagged(conn, watchlist):
    s = Synth(conn, seed=11)
    s.daily("AAPL", s.returns(alpha=0.0003, beta=1.1, noise=0.012))
    (eid,) = s.add("p1", ["AAPL"], D0)
    ev = event_row(run(conn, watchlist), eid)
    for w in ("pre", "event", "post"):
        assert abs(ev[f"z_{w}"]) < 2


def test_missing_bar_makes_spanning_returns_nan(conn, watchlist):
    s = Synth(conn)
    s.daily("AAPL", s.returns(), drop=[rel(D0, -3), rel(D0, -50)])
    (eid,) = s.add("p1", ["AAPL"], D0)
    ev = event_row(run(conn, watchlist), eid)
    assert math.isnan(ev["ar_dm3"]) and math.isnan(ev["ar_dm2"])  # both returns touch the missing close
    assert np.isfinite(ev[["ar_dm5", "ar_dm4", "ar_dm1", "ar_d0", "ar_dp5"]].to_numpy(dtype=float)).all()
    assert math.isnan(ev["car_pre"]) and math.isnan(ev["z_pre"])
    assert np.isfinite(ev["car_event"]) and np.isfinite(ev["car_post"])
    assert ev["n_est"] == 118


def test_short_history_is_no_model_but_still_listed(conn, watchlist):
    s = Synth(conn)
    s.daily("AAPL", s.returns(), first=rel(D0, -60))
    (eid,) = s.add("p1", ["AAPL"], D0)
    study = run(conn, watchlist)
    ev = event_row(study, eid)
    assert ev["excluded_reason"] == "no_model"
    assert ev["n_est"] == 49  # returns for sessions -59..-11
    assert math.isnan(ev["alpha"]) and math.isnan(ev["beta"]) and math.isnan(ev["sigma"])
    assert math.isnan(ev["ar_d0"]) and math.isnan(ev["car_event"]) and math.isnan(ev["z_event"])
    assert ev["rel_volume"] == pytest.approx(1.0)
    assert study.counts["excluded_no_model"] == 1 and study.counts["events_included"] == 0
    assert study.posts.empty and study.counts["placebo_days"] == 0


def test_relative_volume(conn, watchlist):
    s = Synth(conn)
    volume = np.full(len(GRID), 1e6)
    volume[POS[D0]] = 3e6
    volume[POS[D0] - 5] = 50e6  # session -5 is outside the [-25, -6] baseline
    s.daily("AAPL", s.returns(), volume=volume)
    thin = [rel(D0, k) for k in range(-25, -14)]  # 11 of 20 baseline sessions missing -> 9 present
    s.daily("MSFT", s.returns(), volume=volume, drop=thin)
    ids = s.add("p1", ["AAPL", "MSFT"], D0)
    study = run(conn, watchlist)
    assert event_row(study, ids[0])["rel_volume"] == pytest.approx(3.0)
    assert math.isnan(event_row(study, ids[1])["rel_volume"])


# ---------------------------------------------------------------- exclusions


def test_exclusion_precedence_and_include_flags(conn, watchlist):
    s = Synth(conn)
    for t in ("A1", "A2", "A3", "A4"):
        s.daily(t, s.returns())
    s.daily("SHORT", s.returns(), first=rel(D0, -40))
    all3 = s.add("p1", ["A1"], D0, split=1, earnings=1, clustered=1)[0]
    earn_clu = s.add("p2", ["A2"], D0, earnings=1, clustered=1)[0]
    clu = s.add("p3", ["A3"], D0, clustered=1)[0]
    clean = s.add("p4", ["A4"], D0)[0]
    nomodel = s.add("p5", ["SHORT"], D0, split=1)[0]

    def reasons(**opts) -> dict[int, str]:
        study = run(conn, watchlist, **opts)
        return dict(zip(study.events["event_id"], study.events["excluded_reason"], strict=True))

    base = reasons()
    assert base == {all3: "split", earn_clu: "earnings", clu: "clustered", clean: "", nomodel: "no_model"}
    assert reasons(include_splits=True)[all3] == "earnings"
    assert reasons(include_splits=True, include_earnings=True)[all3] == "clustered"
    both = reasons(include_splits=True, include_earnings=True, include_clustered=True)
    assert both == {all3: "", earn_clu: "", clu: "", clean: "", nomodel: "no_model"}

    counts = run(conn, watchlist).counts
    assert counts["excluded_split"] == 1 and counts["excluded_earnings"] == 1
    assert counts["excluded_clustered"] == 1 and counts["excluded_no_model"] == 1
    assert counts["events_complete"] == 5 and counts["events_included"] == 1


def test_unknown_earnings_is_included_and_counted(conn, watchlist):
    s = Synth(conn)
    s.daily("AAPL", s.returns())
    s.daily("MSFT", s.returns())
    unknown = s.add("p1", ["AAPL"], D0, earnings=None)[0]
    s.add("p2", ["MSFT"], D0)
    study = run(conn, watchlist)
    ev = event_row(study, unknown)
    assert ev["excluded_reason"] == "" and math.isnan(ev["earnings_flag"])
    assert study.counts["earnings_unknown"] == 1
    assert any("unknown earnings" in n for n in study.notes)


def test_only_complete_events_in_date_range(conn, watchlist):
    s = Synth(conn)
    s.daily("AAPL", s.returns())
    early, late = rel(D0, -30), D0
    a = s.add("p1", ["AAPL"], early)[0]
    b = s.add("p2", ["AAPL"], late)[0]
    s.add("p3", ["AAPL"], rel(D0, 20), status="pending")
    assert set(run(conn, watchlist).events["event_id"]) == {a, b}
    assert set(run(conn, watchlist, since=rel(D0, -1)).events["event_id"]) == {b}
    assert set(run(conn, watchlist, until=rel(D0, -1)).events["event_id"]) == {a}


# ---------------------------------------------------------------- post level


def test_two_ticker_post_is_averaged_and_bearish_sign_flips(conn, watchlist):
    s = Synth(conn)
    s.daily("AAPL", s.returns(quiet=[D0], jumps={D0: 0.04}), drop=[rel(D0, -2)])
    s.daily("MSFT", s.returns(quiet=[D0], jumps={D0: 0.02}))
    s.daily("NVDA", s.returns(quiet=[D0], jumps={D0: -0.03}))
    s.daily("TSLA", s.returns())
    msft, aapl = s.add("two", ["MSFT", "AAPL"], D0)
    bear = s.add("bear", ["NVDA"], D0, stance="bearish")[0]
    s.add("neutral", ["TSLA"], D0, stance="neutral")
    study = run(conn, watchlist)
    posts = study.posts.set_index("native_id")
    ev = study.events.set_index("event_id")

    two = posts.loc["two"]
    assert two["n_events"] == 2 and two["tickers"] == "AAPL,MSFT"
    assert two["car_event"] == pytest.approx((ev.loc[aapl, "car_event"] + ev.loc[msft, "car_event"]) / 2)
    assert two["car_event"] == pytest.approx(0.03, abs=0.006)
    # AAPL's car_pre is NaN (missing bar), so the NaN-aware mean is MSFT's alone, and so is the pre-window z.
    assert math.isnan(ev.loc[aapl, "car_pre"])
    assert two["car_pre"] == pytest.approx(ev.loc[msft, "car_pre"])
    assert two["z_pre"] == pytest.approx(ev.loc[msft, "z_pre"], rel=1e-12)
    assert two["signed_car_event"] == pytest.approx(two["car_event"])

    b = posts.loc["bear"]
    assert b["sign"] == -1 and b["stance"] == "bearish"
    assert b["car_event"] < -0.02
    assert b["signed_car_event"] == pytest.approx(-ev.loc[bear, "car_event"])
    assert b["signed_car_event"] > 0.02

    n = posts.loc["neutral"]
    assert n["sign"] == 0 and math.isnan(n["signed_car_event"])
    assert study.counts["posts_included"] == 3 and study.counts["posts_signed"] == 2
    assert not any("more than one stock" in note for note in study.notes)


def test_post_uses_only_included_events(conn, watchlist):
    s = Synth(conn)
    s.daily("AAPL", s.returns(quiet=[D0], jumps={D0: 0.04}))
    s.daily("MSFT", s.returns(quiet=[D0], jumps={D0: 0.02}))
    aapl, _ = s.add("two", ["AAPL", "MSFT"], D0)
    conn.execute("UPDATE events SET earnings_flag = 1 WHERE ticker = 'MSFT'")
    conn.commit()
    study = run(conn, watchlist)
    post = study.posts.iloc[0]
    assert post["n_events"] == 1 and post["tickers"] == "AAPL"
    assert post["car_event"] == pytest.approx(event_row(study, aapl)["car_event"])
    for w in DAILY_WINDOWS:  # the excluded MSFT event takes no part in the portfolio sigma either
        assert np.isfinite(post[f"z_{w}"])
        assert post[f"z_{w}"] == pytest.approx(event_row(study, aapl)[f"z_{w}"], rel=1e-12)


def test_one_stock_post_z_is_its_event_z(conn, watchlist):
    s = Synth(conn)
    s.daily("AAPL", s.returns(quiet=[D0], jumps={D0: 0.06}), drop=[rel(D0, -3), rel(D0, -70)])
    s.daily("MSFT", s.returns(noise=0.02))
    ids = s.add("a", ["AAPL"], D0) + s.add("m", ["MSFT"], rel(D0, 20), stance="bearish")
    study = run(conn, watchlist)
    posts = study.posts.set_index("native_id")
    for native_id, eid in zip(("a", "m"), ids, strict=True):
        ev = event_row(study, eid)
        for w in DAILY_WINDOWS:
            if w == "pre" and native_id == "a":  # the missing bar leaves both without a pre-window CAR
                assert math.isnan(ev["z_pre"]) and math.isnan(posts.loc["a", "z_pre"])
                continue
            assert np.isfinite(ev[f"z_{w}"])
            assert posts.loc[native_id, f"z_{w}"] == pytest.approx(ev[f"z_{w}"], rel=1e-12)
    assert posts.loc["a", "z_event"] > 3


def test_two_stock_post_z_is_the_equal_weighted_portfolio_z(conn, watchlist):
    s = Synth(conn)
    shock = s.rng.normal(0.0, 0.008, len(GRID))  # common to both stocks, so their residuals are correlated
    ra = s.returns(beta=1.4, noise=0.006, jumps={D0: 0.03}) + shock
    rb = s.returns(beta=0.7, noise=0.01, jumps={D0: 0.01}) + shock
    s.daily("AAPL", ra)
    s.daily("MSFT", rb)
    aapl, msft = s.add("two", ["AAPL", "MSFT"], D0)
    study = run(conn, watchlist)
    post = study.posts.set_index("native_id").loc["two"]
    ev = study.events.set_index("event_id")
    for w in DAILY_WINDOWS:
        expected, shared = portfolio_z([ra, rb], s.market, D0, w)
        assert shared == 120
        assert post[f"z_{w}"] == pytest.approx(expected, rel=1e-9)
    mean_of_z = (ev.loc[aapl, "z_event"] + ev.loc[msft, "z_event"]) / 2
    assert post["z_event"] != pytest.approx(mean_of_z, rel=0.02)  # not the old average of the events' z


def test_two_identical_stocks_give_the_single_stock_z(conn, watchlist):
    """Perfectly correlated residuals: the portfolio is the stock itself, so its z must not grow by sqrt(2) as it
    would if the two stocks were treated as independent evidence."""
    s = Synth(conn)
    r = s.returns(quiet=[D0], jumps={D0: 0.06})
    s.daily("AAPL", r)
    s.daily("MSFT", r)
    aapl, _ = s.add("two", ["AAPL", "MSFT"], D0)
    study = run(conn, watchlist)
    post = study.posts.set_index("native_id").loc["two"]
    single = event_row(study, aapl)
    for w in DAILY_WINDOWS:
        assert post[f"z_{w}"] == pytest.approx(single[f"z_{w}"], rel=1e-12)
        assert post[f"z_{w}"] == pytest.approx(portfolio_z([r, r], s.market, D0, w)[0], rel=1e-9)
    assert single["z_event"] > 3


def test_no_effect_three_stock_posts_have_unit_z_spread(conn, watchlist):
    """Why posts use the portfolio z: with no real effect a post's z should spread like one stock's (sd about 1, so
    about 5% beyond 1.96), where the average of three z-scores shrinks toward sd 1/sqrt(3)."""
    s = Synth(conn, seed=21)
    tickers = [f"C{k:02d}" for k in range(30)]
    for t in tickers:
        s.daily(t, s.returns(beta=float(s.rng.uniform(0.5, 1.5)), noise=float(s.rng.uniform(0.008, 0.02))))
    for j in range(150):
        s.add(f"p{j}", sorted(s.rng.choice(tickers, size=3, replace=False)), rel(D0, -j))
    study = run(conn, watchlist)
    included = study.events[study.events["excluded_reason"] == ""]
    mean_of_z = included.groupby("native_id")["z_event"].mean()
    z = study.posts["z_event"]
    assert len(z) == 150 and z.notna().all()
    assert 0.85 < z.std() < 1.15
    assert mean_of_z.std() < 0.75


@pytest.mark.parametrize(("gap_from", "shared"), [(-40, 60), (-41, 59)])
def test_portfolio_z_needs_sixty_estimation_days_all_stocks_share(conn, watchlist, gap_from, shared):
    """Each stock alone has enough days for its own model (AAPL 90, MSFT 90 or 89), but the portfolio sigma only uses
    the days both have: AAPL's history starts at session -101 and MSFT has no closes from `gap_from` to -11."""
    s = Synth(conn)
    ra = s.returns(jumps={D0: 0.02})
    rb = s.returns(jumps={D0: 0.02})
    s.daily("AAPL", ra, first=rel(D0, -101))
    s.daily("MSFT", rb, drop=[rel(D0, k) for k in range(gap_from, -10)])
    aapl, msft = s.add("two", ["AAPL", "MSFT"], D0)
    study = run(conn, watchlist)
    post = study.posts.set_index("native_id").loc["two"]
    assert event_row(study, aapl)["n_est"] == 90 and event_row(study, msft)["n_est"] == gap_from + 130
    i = POS[D0]
    ra, rb = ra.copy(), rb.copy()
    ra[: i - 100] = np.nan  # a return needs its own and the previous close
    rb[i + gap_from : i - 9] = np.nan
    for w in DAILY_WINDOWS:
        assert np.isfinite(event_row(study, aapl)[f"z_{w}"]) and np.isfinite(event_row(study, msft)[f"z_{w}"])
        assert np.isfinite(post[f"car_{w}"])
        expected, n = portfolio_z([ra, rb], s.market, D0, w)
        assert n == shared
        if shared >= 60:
            assert post[f"z_{w}"] == pytest.approx(expected, rel=1e-9)
        else:
            assert math.isnan(post[f"z_{w}"])


def test_notes_separate_neutral_from_unlabelled_posts(conn, watchlist):
    s = Synth(conn)
    for t in ("AAPL", "MSFT", "NVDA"):
        s.daily(t, s.returns())
    s.add("bull", ["AAPL"], D0)
    s.add("neutral", ["MSFT"], D0, stance="neutral")
    s.add("nolabel", ["NVDA"], D0, stance=None, conf=None)
    study = run(conn, watchlist)
    assert set(study.car_path["series"]) == {"bullish", "neutral", "placebo"}  # no series for unlabelled posts
    assert study.magnitude.set_index(["window", "sample"]).loc[("event", "events"), "n"] == 3
    notes = study.notes
    neutral = [n for n in notes if n.startswith("1 of 3 included posts are labelled neutral")]
    unlabelled = [n for n in notes if n.startswith("1 of 3 included posts have no stance label")]
    assert len(neutral) == 1 and "the neutral CAR path" in neutral[0]
    assert len(unlabelled) == 1 and "only in the event-level magnitude table" in unlabelled[0]
    assert not any("neutral or unlabelled" in n for n in notes)


def test_posts_near_other_posts_on_the_same_stock_are_disclosed(conn, watchlist):
    s = Synth(conn)
    s.daily("AAPL", s.returns())
    s.daily("MSFT", s.returns())
    s.add("a0", ["AAPL"], D0)
    s.add("a3", ["AAPL"], rel(D0, 3), stance="bearish")  # inside a0's -5..+5 path, and a0 inside a3's
    s.add("a-50", ["AAPL"], rel(D0, -50))  # inside the estimation window (-130..-11) of a0 and a3
    s.add("m0", ["MSFT"], D0)
    s.add("m0b", ["MSFT"], D0, clustered=1)  # same session as m0: the clustered rule, not an overlap
    study = run(conn, watchlist)
    assert study.counts["events_included"] == 4
    assert any(n.startswith("2 of 4 included events have another post about the same stock whose day 0 is 1 to 5 "
                            "sessions") for n in study.notes)  # fmt: skip
    assert any(n.startswith("2 of 4 included events have another post about the same stock with its day 0 inside "
                            "their market-model estimation window") for n in study.notes)  # fmt: skip


def test_author_group_and_category(conn, watchlist):
    s = Synth(conn)
    s.daily("AAPL", s.returns())
    db.upsert_account(conn, "truthsocial", "realDonaldTrump", "politician", True)
    db.upsert_account(conn, "x", "CathieDWood", "investor", True)
    conn.commit()
    s.add("ts", ["AAPL"], D0)
    s.add("x1", ["AAPL"], D0, platform="x", author="cathiedwood")  # handle case differs from the account row
    s.add("x2", ["AAPL"], D0, platform="x", author="someone_else")
    s.add("rd", ["AAPL"], D0, platform="reddit", author="u_private_person", source="wallstreetbets")
    posts = run(conn, watchlist).posts.set_index("native_id")
    assert (posts.loc["ts", "author"], posts.loc["ts", "category"]) == ("realDonaldTrump", "politician")
    assert posts.loc["x1", "category"] == "investor"
    assert posts.loc["x2", "category"] == "unknown"
    assert (posts.loc["rd", "author"], posts.loc["rd", "category"]) == ("r/wallstreetbets", REDDIT_CATEGORY)


# ---------------------------------------------------------------- statistics helpers and group tests


def test_holm_matches_hand_computation():
    # sorted: 0.01*4 = 0.04; 0.03*3 = 0.09; 0.04*2 = 0.08 -> monotone 0.09; 0.20*1 = 0.20
    assert holm_adjust([0.01, 0.04, 0.03, 0.20]) == pytest.approx([0.04, 0.09, 0.09, 0.20])
    assert holm_adjust([0.5, 0.6]) == pytest.approx([1.0, 1.0])
    assert holm_adjust([0.3]) == pytest.approx([0.3])
    assert len(holm_adjust([])) == 0


def test_mean_ci_matches_hand_computation():
    mean, low, high = mean_ci([0.01, 0.02, 0.03])
    half = 4.302652729911275 * 0.01 / math.sqrt(3)  # t(0.975, 2) * s / sqrt(n)
    assert mean == pytest.approx(0.02)
    assert (low, high) == pytest.approx((0.02 - half, 0.02 + half))
    assert mean_ci([0.5])[0] == 0.5 and math.isnan(mean_ci([0.5])[1])
    assert all(math.isnan(v) for v in mean_ci([]))


def hand_t(values: Sequence[float]) -> tuple[float, float]:
    n = len(values)
    t = statistics.fmean(values) / (statistics.stdev(values) / math.sqrt(n))
    return t, 2 * float(stats.t.sf(abs(t), n - 1))


def test_group_t_tests_and_holm(conn, watchlist):
    s = Synth(conn)
    db.upsert_account(conn, "truthsocial", "alice", "investor", True)
    db.upsert_account(conn, "truthsocial", "bob", "short_seller", True)
    db.upsert_account(conn, "truthsocial", "carol", "media", True)
    conn.commit()
    k = 0
    for author, stance, n, jump in (
        ("alice", "bullish", 12, 0.02),
        ("bob", "bearish", 10, -0.015),
        ("carol", "bullish", 3, 0.01),
    ):
        for j in range(n):
            ticker = f"S{k:02d}"
            s.daily(ticker, s.returns(noise=0.01, jumps={D0: jump}))
            conf = 0.5 if (author == "alice" and j < 4) else 0.9
            s.add(f"{author}{j}", [ticker], D0, stance=stance, conf=conf, author=author)
            k += 1
    study = run(conn, watchlist)
    g = study.groups
    posts = study.posts
    assert list(g.columns) == list(GROUP_COLUMNS)

    def row(family: str, group: str, subset: str = "main", window: str = "event"):
        sel = g[(g["family"] == family) & (g["group"] == group) & (g["subset"] == subset) & (g["window"] == window)]
        assert len(sel) == 1
        return sel.iloc[0]

    everyone = row("all", "all")
    values = posts["signed_car_event"].tolist()
    t, p = hand_t(values)
    assert everyone["n_posts"] == 25 and everyone["status"] == "ok"
    assert everyone["mean_signed_car"] == pytest.approx(statistics.fmean(values))
    assert everyone["median_signed_car"] == pytest.approx(statistics.median(values))
    assert everyone["t_stat"] == pytest.approx(t) and everyone["p_value"] == pytest.approx(p)
    assert everyone["p_holm"] == pytest.approx(p)  # one tested group in its family
    z = posts["z_event"].abs()
    assert everyone["mean_abs_z"] == pytest.approx(z.mean())
    assert everyone["share_abs_z_gt_crit"] == pytest.approx((z > 1.96).mean())
    half = float(stats.t.ppf(0.975, 24)) * statistics.stdev(values) / 5
    assert (everyone["ci_low"], everyone["ci_high"]) == pytest.approx((everyone["mean_signed_car"] - half,
                                                                      everyone["mean_signed_car"] + half))  # fmt: skip

    alice, bob, carol = row("author", "alice"), row("author", "bob"), row("author", "carol")
    t_a, p_a = hand_t(posts.loc[posts["author"] == "alice", "signed_car_event"].tolist())
    t_b, p_b = hand_t(posts.loc[posts["author"] == "bob", "signed_car_event"].tolist())
    assert (alice["t_stat"], alice["p_value"]) == pytest.approx((t_a, p_a))
    assert (bob["t_stat"], bob["p_value"]) == pytest.approx((t_b, p_b))
    lo, hi = sorted([p_a, p_b])
    holm = {lo: min(1.0, 2 * lo), hi: min(1.0, max(2 * lo, hi))}
    assert alice["p_holm"] == pytest.approx(holm[p_a]) and bob["p_holm"] == pytest.approx(holm[p_b])
    assert bob["mean_signed_car"] > 0.005  # bearish posts on falling stocks count as correct calls
    assert carol["status"] == "insufficient" and carol["n_posts"] == 3
    assert math.isnan(carol["t_stat"]) and math.isnan(carol["p_value"]) and math.isnan(carol["p_holm"])
    assert np.isfinite(carol["mean_signed_car"]) and np.isfinite(carol["ci_low"])

    assert row("stance", "bullish")["n_posts"] == 15 and row("stance", "bearish")["n_posts"] == 10
    assert set(g.loc[g["family"] == "stance", "group"]) == {"bullish", "bearish"}
    assert row("category", "media")["status"] == "insufficient"
    assert row("platform", "truthsocial")["n_posts"] == 25
    confident = row("author", "alice", subset="confident")
    assert confident["n_posts"] == 8 and confident["status"] == "insufficient"
    assert row("all", "all", subset="confident")["n_posts"] == 21


def test_fewer_than_ten_signed_posts_are_insufficient(conn, watchlist):
    s = Synth(conn)
    for j in range(4):
        s.daily(f"S{j}", s.returns())
        s.add(f"p{j}", [f"S{j}"], D0)
    study = run(conn, watchlist)
    assert (study.groups["status"] == "insufficient").all()
    assert study.groups["p_value"].isna().all() and study.groups["p_holm"].isna().all()
    assert study.groups["mean_signed_car"].notna().any()
    assert any("Too few labelled posts" in n for n in study.notes)


# ---------------------------------------------------------------- placebo


def placebo_days(conn, watchlist, **options) -> list[tuple[int, str]]:
    _, placebo = compute_study_with_placebo(conn, watchlist, NOW, StudyOptions(**options))
    assert list(placebo.columns) == list(PLACEBO_COLUMNS)
    return list(zip(placebo["event_id"], placebo["day"], strict=True))


def test_placebo_draws_are_deterministic_and_seeded(conn, watchlist):
    s = Synth(conn)
    s.daily("AAPL", s.returns())
    s.daily("MSFT", s.returns())
    (eid,) = s.add("p1", ["AAPL"], D0)
    first = placebo_days(conn, watchlist)
    assert len(first) == 5 and len(set(first)) == 5
    assert placebo_days(conn, watchlist) == first
    assert placebo_days(conn, watchlist, seed=1) != first
    s.add("p2", ["MSFT"], rel(D0, 3))  # another event must not change this event's draws
    assert [d for d in placebo_days(conn, watchlist) if d[0] == eid] == first
    rng = np.random.default_rng(StudyOptions().seed * 1_000_003 + eid)
    lo, hi = POS[D0] - 130, POS[D0] - 10
    expected = sorted(rng.choice(np.arange(lo, hi + 1), size=5, replace=False))
    assert [day for _, day in first] == [GRID[i].isoformat() for i in expected]


def test_placebo_clearance_around_any_event_on_the_ticker(conn, watchlist):
    s = Synth(conn)
    s.daily("AAPL", s.returns())
    s.daily("MSFT", s.returns())
    (eid,) = s.add("p1", ["AAPL"], D0)
    s.add("near1", ["AAPL"], rel(D0, -40), status="pending")
    s.add("near2", ["AAPL"], rel(D0, -80), earnings=1)  # complete but excluded: still blocks
    s.add("other", ["MSFT"], rel(D0, -60))  # a different ticker never blocks AAPL
    days = {day for owner, day in placebo_days(conn, watchlist, placebo_draws=1000) if owner == eid}
    blocked = {rel(D0, c + k) for c in (-40, -80) for k in range(-5, 6)}
    expected = {rel(D0, k).isoformat() for k in range(-130, -9)} - {d.isoformat() for d in blocked}
    assert days == expected and len(expected) == 121 - 22
    assert rel(D0, -60).isoformat() in days


def test_each_placebo_day_gets_its_own_model(conn, watchlist):
    s = Synth(conn)
    r = s.returns(beta=0.5, noise=0.006)
    late = s.returns(beta=2.5, noise=0.006)
    r[POS[D0] - 100 :] = late[POS[D0] - 100 :]  # beta regime change inside the placebo range
    s.daily("AAPL", r)
    (eid,) = s.add("p1", ["AAPL"], D0)
    study, placebo = compute_study_with_placebo(conn, watchlist, NOW, StudyOptions(placebo_draws=20))
    assert len(placebo) == 20 == study.counts["placebo_days"]
    for _, p in placebo.iterrows():
        alpha, beta, sigma = model_at(r, s.market, date.fromisoformat(p["day"]))
        assert (p["alpha"], p["beta"], p["sigma"]) == pytest.approx((alpha, beta, sigma), rel=1e-9, abs=1e-12)
        i = POS[date.fromisoformat(p["day"])]
        assert p["ar_d0"] == pytest.approx(r[i] - alpha - beta * s.market[i], abs=1e-12)
    assert placebo["beta"].max() - placebo["beta"].min() > 0.5
    mag = study.magnitude.set_index(["window", "sample"])
    assert mag.loc[("event", "placebo"), "n"] == 20
    assert mag.loc[("event", "placebo"), "mean_abs_car"] == pytest.approx(placebo["car_event"].abs().mean())
    assert mag.loc[("event", "events"), "n"] == 1
    z = placebo["z_event"].abs()
    assert mag.loc[("event", "placebo"), "share_abs_z_gt_crit"] == pytest.approx((z > 1.96).mean())
    assert event_row(study, eid)["excluded_reason"] == ""


def test_a_placebo_stock_day_drawn_for_two_events_counts_once(conn, watchlist):
    s = Synth(conn)
    s.daily("AAPL", s.returns())
    first = s.add("p1", ["AAPL"], D0)[0]
    second = s.add("p2", ["AAPL"], rel(D0, 8))[0]
    study, placebo = compute_study_with_placebo(conn, watchlist, NOW, StudyOptions(placebo_draws=1000))
    blocked = {rel(D0, k) for c in (0, 8) for k in range(c - 5, c + 6)}
    own = {eid: {rel(d, k) for k in range(-130, -9)} - blocked for eid, d in ((first, D0), (second, rel(D0, 8)))}
    assert len(own[first] & own[second]) > 100  # the two ranges mostly overlap

    assert not placebo.duplicated(["ticker", "day"]).any()
    owner = {date.fromisoformat(d): e for e, d in zip(placebo["event_id"], placebo["day"], strict=True)}
    assert set(owner) == own[first] | own[second]
    assert {d for d, e in owner.items() if e == first} == own[first]  # the lower event id keeps a shared day
    assert {d for d, e in owner.items() if e == second} == own[second] - own[first]
    n = len(own[first] | own[second])
    assert study.counts["placebo_days"] == n
    assert study.magnitude.set_index(["window", "sample"]).loc[("event", "placebo"), "n"] == n
    assert study.car_path.set_index(["series", "day"]).loc[("placebo", -5), "n"] == n
    assert any(note.startswith(f"Placebo baseline: {n} distinct stock-days from 2 included events") for note in
               study.notes)  # fmt: skip


def test_placebo_days_get_the_events_earnings_and_split_screens(conn, watchlist):
    s = Synth(conn)
    s.daily("AAPL", s.returns())
    s.add("p1", ["AAPL"], D0)
    # 16:05 New York is after the close, so the report is first traded on the next session (events.earnings_session).
    report = datetime.combine(rel(D0, -60), time(16, 5), tzinfo=NY)
    with conn:
        conn.execute("INSERT INTO earnings (symbol, earnings_at) VALUES ('AAPL', ?)", (to_iso(report),))
        conn.execute("INSERT INTO earnings (symbol, earnings_at) VALUES ('MSFT', ?)", (to_iso(report),))
        conn.execute(
            "UPDATE bars_1d SET split_ratio = 4.0 WHERE symbol = 'AAPL' AND session_date = ?",
            (rel(D0, -100).isoformat(),),
        )
    everything = {rel(D0, k) for k in range(-130, -9)}
    earnings = {rel(D0, k) for k in (-60, -59, -58)}  # within one session of the report's session (-59)
    split = {rel(D0, k) for k in range(-105, -94)}  # within 5 sessions of the split

    def drawn(**opts) -> set[date]:
        _, placebo = compute_study_with_placebo(conn, watchlist, NOW, StudyOptions(placebo_draws=1000, **opts))
        return {date.fromisoformat(d) for d in placebo["day"]}

    assert drawn() == everything - earnings - split
    assert drawn(include_earnings=True) == everything - split
    assert drawn(include_splits=True) == everything - earnings
    notes = run(conn, watchlist).notes
    assert any("no earnings report within 1 session and no split within 5 sessions" in n for n in notes)


def test_placebo_draws_that_fail_their_model_are_skipped(conn, watchlist):
    s = Synth(conn)
    s.daily("AAPL", s.returns(), first=rel(D0, -200))  # early placebo days lack 60 estimation days
    s.add("p1", ["AAPL"], D0)
    _, placebo = compute_study_with_placebo(conn, watchlist, NOW, StudyOptions(placebo_draws=121))
    days = [date.fromisoformat(d) for d in placebo["day"]]
    # a placebo day p fits on sessions p-130..p-11; history starts at -200, so p needs p-11 >= -200+60 -> p >= -129
    assert min(days) == rel(D0, -129) and len(days) == 120


# ---------------------------------------------------------------- CAR path


def test_car_path_cumulates_and_counts(conn, watchlist):
    s = Synth(conn)
    s.daily("AAPL", s.returns(jumps={D0: 0.03}))
    s.daily("MSFT", s.returns(jumps={D0: 0.01}), drop=[rel(D0, -2)])
    s.add("a", ["AAPL"], D0)
    s.add("b", ["MSFT"], D0)
    study, placebo = compute_study_with_placebo(conn, watchlist, NOW, StudyOptions())
    path = study.car_path
    assert list(path.columns) == list(PATH_COLUMNS)
    bull = path[path["series"] == "bullish"].set_index("day")
    posts = study.posts.set_index("native_id")
    a = np.cumsum(posts.loc["a", list(AR_COLUMNS)].to_numpy(dtype=float))
    b = np.cumsum(posts.loc["b", list(AR_COLUMNS)].to_numpy(dtype=float))
    for j, day in enumerate(PATH_DAYS):
        if day <= -3:
            assert bull.loc[day, "n"] == 2
            mean, low, high = mean_ci([a[j], b[j]])
            assert bull.loc[day, "mean_car"] == pytest.approx((a[j] + b[j]) / 2)
            assert (bull.loc[day, "ci_low"], bull.loc[day, "ci_high"]) == pytest.approx((low, high))
        else:
            assert bull.loc[day, "n"] == 1  # b has no AR on day -2, so its path stops at day -3
            assert bull.loc[day, "mean_car"] == pytest.approx(a[j])
            assert math.isnan(bull.loc[day, "ci_low"])
    plac = path[path["series"] == "placebo"].set_index("day")
    cum = np.cumsum(placebo[list(AR_COLUMNS)].to_numpy(dtype=float), axis=1)
    assert plac.loc[5, "n"] == int(np.isfinite(cum[:, -1]).sum())
    assert plac.loc[0, "mean_car"] == pytest.approx(np.nanmean(cum[:, PATH_DAYS.index(0)]))
    assert set(path["series"]) == {"bullish", "placebo"}


# ---------------------------------------------------------------- intraday same-clock sigma


def add_minutes(conn, symbol: str, session: date, price: Callable[[int, time], float], pre_minutes: int = 30) -> None:
    """1-minute bars from `pre_minutes` before the open to the (early) close; price(i, New York bar start)."""
    open_, close = market.session_bounds_utc(session)
    t = open_ - timedelta(minutes=pre_minutes)
    rows = []
    i = 0
    while t < close:
        p = price(i, t.astimezone(NY).time())
        rows.append((int(t.timestamp()), p, p, p, p, 100.0))
        t += timedelta(minutes=1)
        i += 1
    db.insert_bars_1m(conn, symbol, rows)
    db.mark_session(conn, symbol, session.isoformat(), len(rows), NOW)


def walk(rng: np.random.Generator, vol: Callable[[time], float], start: float) -> Callable[[int, time], float]:
    state = {"p": start}

    def price(i: int, t: time) -> float:
        state["p"] *= 1 + rng.normal(0.0, vol(t))
        return state["p"]

    return price


def intraday_event(s: Synth, native_id: str, ticker: str, t0: datetime, stance: str = "bullish") -> int:
    """A regular-session event whose windows are computed exactly as M2 does."""
    d0 = market.event_session(t0)
    s.post(native_id, t0=t0, stance=stance)
    eid = s.event(native_id, ticker, d0, t0=t0, intraday="ok", phase="regular")
    start = market.extended_bounds_utc(market.previous_session(d0))[0]
    end = market.extended_bounds_utc(d0)[1]
    bars, spy = MinuteBars.load(s.conn, ticker, start, end), MinuteBars.load(s.conn, "SPY", start, end)
    _, rows, skipped = compute_windows(t0, d0, bars, spy)
    assert not skipped
    with s.conn:
        s.conn.executemany(
            """INSERT INTO event_windows (event_id, win, start_ts, end_ts, start_price, end_price, ret,
                                          spy_start_price, spy_end_price, spy_ret, truncated)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            [(eid, r.win, r.start_ts, r.end_ts, r.start_price, r.end_price, r.ret, r.spy_start_price,
              r.spy_end_price, r.spy_ret, int(r.truncated)) for r in rows],
        )  # fmt: skip
    return eid


def at_ny(d: date, hh: int, mm: int) -> datetime:
    return datetime.combine(d, time(hh, mm), tzinfo=NY).astimezone(UTC)


def test_intraday_sigma_follows_time_of_day_volatility(conn, watchlist):
    s = Synth(conn)
    s.daily("AAPL", s.returns())
    rng = np.random.default_rng(3)

    def open_heavy(t: time) -> float:
        return 0.004 if time(9, 30) <= t < time(10, 0) else 0.0005

    for session in GRID[POS[D0] - 25 : POS[D0] + 1]:
        add_minutes(conn, "AAPL", session, walk(rng, open_heavy, 100.0))
        add_minutes(conn, "SPY", session, walk(rng, lambda t: 0.0003, 400.0))
    early = intraday_event(s, "early", "AAPL", at_ny(D0, 9, 40))
    midday = intraday_event(s, "midday", "AAPL", at_ny(D0, 12, 30))
    study = run(conn, watchlist)
    e, m = event_row(study, early), event_row(study, midday)
    for w in ("pre60", "p5", "p15", "p30", "p60"):
        assert np.isfinite(e[f"iz_{w}"]) and np.isfinite(m[f"iz_{w}"])
    s_open = e["iar_p15"] / e["iz_p15"]
    s_mid = m["iar_p15"] / m["iz_p15"]
    assert 0.009 < s_open < 0.025  # ~ sqrt(15) * 0.4% per minute
    assert 0.001 < s_mid < 0.003  # ~ sqrt(15) * 0.05% per minute
    assert s_open > 4 * s_mid
    assert study.counts["events_intraday"] == 2

    ip = study.intraday_path.set_index("minute")
    assert list(study.intraday_path.columns) == list(INTRADAY_PATH_COLUMNS)
    assert set(study.intraday_path["series"]) == {"bullish"}
    assert ip.loc[0, "n"] == 2 and ip.loc[0, "mean_ar"] == 0.0
    assert ip.loc[-60, "mean_ar"] == pytest.approx(-(e["iar_pre60"] + m["iar_pre60"]) / 2)
    assert ip.loc[15, "mean_ar"] == pytest.approx((e["iar_p15"] + m["iar_p15"]) / 2)


@pytest.mark.parametrize("spy_gap", [False, True])
def test_intraday_sigma_uses_same_new_york_clock_window(conn, watchlist, spy_gap):
    """Deterministic prices: each prior session's 13:30->13:45 ET return is a known x. The d0 window's sigma must be
    the sample std of the x values over the 20 most recent sessions with bars for both symbols (a session without
    SPY bars does not count), skipping the Nov 27 early close (the window does not fit) and crossing the Nov 1 DST
    switch on New York wall-clock time. Older sessions carry x = 0.5, which would blow up sigma if used."""
    s = Synth(conn)
    s.daily("AAPL", s.returns())
    d0 = date(2026, 11, 30)
    priors = GRID[POS[d0] - 25 : POS[d0]]
    gap = date(2026, 11, 25)
    both = [p for p in priors if not (spy_gap and p == gap)]
    recent = both[-20:]
    xs: dict[date, float] = {}
    for k, session in enumerate(reversed(priors)):
        xs[session] = 0.001 * (k % 7 + 1) * (1 if k % 2 == 0 else -1) if session in recent else 0.5
    xs[d0] = 0.02

    def stepped(x: float) -> Callable[[int, time], float]:
        return lambda i, t: 100.0 * (1 + x) if t >= time(13, 44) else 100.0

    for session in [*priors, d0]:
        add_minutes(conn, "AAPL", session, stepped(xs[session]))
        if not (spy_gap and session == gap):
            add_minutes(conn, "SPY", session, lambda i, t: 400.0)
    eid = intraday_event(s, "p", "AAPL", at_ny(d0, 13, 30))
    study = run(conn, watchlist)
    ev = event_row(study, eid)

    assert date(2026, 11, 27) in recent and date(2026, 10, 30) in recent  # early close and pre-DST session
    assert (date(2026, 10, 29) in recent) == spy_gap
    valid = [xs[p] for p in recent if p != date(2026, 11, 27)]
    assert len(valid) == 19
    sigma = statistics.stdev(valid)
    assert ev["iar_p15"] == pytest.approx(0.02)
    assert ev["iz_p15"] == pytest.approx(0.02 / sigma, rel=1e-9)
    assert ev["iz_p30"] == pytest.approx(0.02 / sigma, rel=1e-9)
    assert ev["iz_p60"] == pytest.approx(0.02 / sigma, rel=1e-9)
    assert ev["iar_p5"] == 0.0 and math.isnan(ev["iz_p5"])  # no prior variation in 13:30->13:35: sigma 0 -> no z
    assert ev["iar_pre60"] == 0.0 and math.isnan(ev["iz_pre60"])  # nor in 12:30->13:30
    assert any(n.startswith("2 of 5 intraday windows have no z-score because") for n in study.notes)


def test_intraday_sigma_needs_ten_prior_sessions(conn, watchlist):
    s = Synth(conn)
    s.daily("AAPL", s.returns())
    rng = np.random.default_rng(5)
    for session in GRID[POS[D0] - 8 : POS[D0] + 1]:
        add_minutes(conn, "AAPL", session, walk(rng, lambda t: 0.001, 100.0))
        add_minutes(conn, "SPY", session, walk(rng, lambda t: 0.0003, 400.0))
    eid = intraday_event(s, "p", "AAPL", at_ny(D0, 11, 0))
    study = run(conn, watchlist)
    ev = event_row(study, eid)
    for w in ("pre60", "p5", "p15", "p30", "p60"):
        assert np.isfinite(ev[f"iar_{w}"]) and math.isnan(ev[f"iz_{w}"])
    assert any(n.startswith("5 of 5 intraday windows have no z-score: fewer than 10 usable prior sessions") for n in
               study.notes)  # fmt: skip
    assert not any("same-clock return was identical" in n for n in study.notes)


def test_zero_same_clock_volatility_is_not_blamed_on_short_history(conn, watchlist):
    s = Synth(conn)
    s.daily("AAPL", s.returns())
    rng = np.random.default_rng(9)
    for session in GRID[POS[D0] - 15 : POS[D0] + 1]:
        add_minutes(conn, "AAPL", session, walk(rng, lambda t: 0.001, 100.0))
        add_minutes(conn, "SPY", session, walk(rng, lambda t: 0.0003, 400.0))
    eid = intraday_event(s, "p", "AAPL", at_ny(D0, 9, 30) + timedelta(seconds=20))
    study = run(conn, watchlist)
    ev = event_row(study, eid)
    # The pre-60 window is 09:30:00 -> 09:30:20: on every day both ends price at the 09:29 pre-market bar.
    assert ev["iar_pre60"] == 0.0 and math.isnan(ev["iz_pre60"])
    for w in ("p5", "p15", "p30", "p60"):
        assert np.isfinite(ev[f"iz_{w}"])
    assert any(n.startswith("1 of 5 intraday windows have no z-score because the same-clock return was identical")
               for n in study.notes)  # fmt: skip
    assert not any("fewer than 10 usable prior sessions" in n for n in study.notes)


def test_intraday_loads_only_the_minute_bars_it_needs(conn, watchlist, monkeypatch):
    """Two events 35 sessions apart on one ticker: the sessions between their prior-session blocks are never read,
    so the cost follows the events, not the length of the stored 1-minute history."""
    s = Synth(conn)
    s.daily("AAPL", s.returns())
    rng = np.random.default_rng(13)
    for session in GRID[POS[D0] - 60 : POS[D0] + 1]:
        add_minutes(conn, "AAPL", session, walk(rng, lambda t: 0.001, 100.0))
        add_minutes(conn, "SPY", session, walk(rng, lambda t: 0.0003, 400.0))
    early = intraday_event(s, "early", "AAPL", at_ny(rel(D0, -35), 11, 0))
    late = intraday_event(s, "late", "AAPL", at_ny(D0, 11, 0))
    loaded: list[int] = []
    load = MinuteBars.load.__func__

    def recording(cls, conn_, symbol, start, end):
        bars = load(cls, conn_, symbol, start, end)
        loaded.extend(bars.starts)
        return bars

    monkeypatch.setattr(MinuteBars, "load", classmethod(recording))
    study = run(conn, watchlist)
    monkeypatch.undo()

    read = {datetime.fromtimestamp(ts, UTC).astimezone(NY).date() for ts in loaded}
    # Each event reads its 20 prior sessions plus the session before them (a window starting at the open is priced
    # at the previous bar, which can be the prior evening's).
    assert read == set(GRID[POS[D0] - 56 : POS[D0] - 35]) | set(GRID[POS[D0] - 21 : POS[D0]])
    together = event_row(study, late)
    assert np.isfinite(event_row(study, early)[[f"iz_{w}" for w in ("pre60", "p15")]].to_numpy(dtype=float)).all()
    with conn:
        conn.execute("DELETE FROM event_windows WHERE event_id = ?", (early,))
        conn.execute("DELETE FROM events WHERE id = ?", (early,))
    alone = event_row(run(conn, watchlist), late)
    for w in ("pre60", "p5", "p15", "p30", "p60"):
        assert np.isfinite(together[f"iz_{w}"]) and alone[f"iz_{w}"] == pytest.approx(together[f"iz_{w}"], rel=1e-12)


# ---------------------------------------------------------------- Reddit attention


def test_attention_spike_and_next_session(conn, watchlist):
    s = Synth(conn)
    r = s.returns()
    s.daily("AAPL", r)
    days = [date(2026, 6, 1) + timedelta(days=k) for k in range(9)]  # Mon Jun 1 .. Tue Jun 9
    mentions = [10, 12, 8, 10, 11, 9, 10, 50, 20]
    fetched = {k: f"{d.isoformat()}T13:02:00Z" for k, d in enumerate(days)}  # 09:02 New York, before the open
    fetched[8] = "2026-06-09T14:00:00Z"  # 10:00 New York: after the open, so the next session is Jun 10
    rows = []
    for k, d in enumerate(days):
        base = {"snapshot_date": d.isoformat(), "name": None, "rank": 1, "upvotes": 0, "rank_24h_ago": None,
                "mentions_24h_ago": None, "fetched_at": fetched[k]}  # fmt: skip
        rows.append({**base, "filter": "all-stocks", "ticker": "AAPL", "mentions": mentions[k]})
        rows.append({**base, "filter": "wallstreetbets", "ticker": "AAPL", "mentions": 1000})
        rows.append({**base, "filter": "all-stocks", "ticker": "SPY", "mentions": 500})
        rows.append({**base, "filter": "all-stocks", "ticker": "ZZZZ", "mentions": 5})
        if k >= 6:
            rows.append({**base, "filter": "all-stocks", "ticker": "MSFT", "mentions": 7})
    db.upsert_reddit_ticker_daily(conn, rows)
    conn.commit()

    study = run(conn, watchlist)
    att = study.attention
    assert list(att.columns) == list(ATTENTION_COLUMNS)
    assert list(att["ticker"]) == ["AAPL"] * 4  # MSFT has < 5 trailing days; SPY is a benchmark; ZZZZ unlisted
    by_day = att.set_index("snapshot_date")
    sat = by_day.loc["2026-06-06"]
    assert sat["trailing_mean"] == pytest.approx(np.mean(mentions[0:5]))
    assert sat["next_session"] == "2026-06-08"
    mon = by_day.loc["2026-06-08"]
    assert mon["mentions"] == 50 and mon["trailing_mean"] == pytest.approx(10.0) and mon["spike"] == pytest.approx(5.0)
    assert mon["next_session"] == "2026-06-08"
    tue = by_day.loc["2026-06-09"]
    assert tue["trailing_mean"] == pytest.approx(np.mean(mentions[1:8]))
    assert tue["next_session"] == "2026-06-10"
    alpha, beta, sigma = model_at(r, s.market, date(2026, 6, 8))
    i = POS[date(2026, 6, 8)]
    assert mon["next_ar"] == pytest.approx(r[i] - alpha - beta * s.market[i], abs=1e-12)
    assert mon["next_z"] == pytest.approx(mon["next_ar"] / sigma)
    assert study.counts["attention_days"] == 9
    assert not any("ApeWisdom" in n for n in study.notes)  # a full week of snapshots, and every row has its AR


def test_attention_rows_without_prices_are_explained(conn, watchlist):
    s = Synth(conn)
    s.daily("AAPL", s.returns(), drop=[date(2026, 6, 9)])  # no close on Jun 9: that session has no return
    rows = []
    for k in range(9):  # Mon Jun 1 .. Tue Jun 9, fetched 09:02 New York, so the next session is the same day
        d = date(2026, 6, 1) + timedelta(days=k)
        base = {"snapshot_date": d.isoformat(), "filter": "all-stocks", "name": None, "rank": 1, "upvotes": 0,
                "rank_24h_ago": None, "mentions_24h_ago": None, "fetched_at": f"{d.isoformat()}T13:02:00Z"}  # fmt: skip
        rows.append({**base, "ticker": "AAPL", "mentions": 10 + k})
        rows.append({**base, "ticker": "TSLA", "mentions": 20 + k})  # listed, but no daily bars were downloaded
    db.upsert_reddit_ticker_daily(conn, rows)
    conn.commit()
    study = run(conn, watchlist)
    att = study.attention.set_index(["ticker", "snapshot_date"])
    assert att.loc["TSLA", "next_ar"].isna().all() and len(att.loc["TSLA"]) == 4
    assert math.isnan(att.loc[("AAPL", "2026-06-09"), "next_ar"])
    assert att.loc["AAPL", "next_ar"].notna().sum() == 3
    assert any(n.startswith("ApeWisdom attention: 1 of 2 tickers in the table have no daily prices") for n in
               study.notes)  # fmt: skip
    assert any(n.startswith("ApeWisdom attention: 1 of 4 rows for stocks with daily prices have no next-session")
               for n in study.notes)  # fmt: skip


def test_next_session_after():
    assert next_session_after(datetime(2026, 9, 25, 13, 2, tzinfo=UTC)) == date(2026, 9, 25)  # Fri 09:02 ET
    assert next_session_after(datetime(2026, 9, 25, 13, 30, tzinfo=UTC)) == date(2026, 9, 28)  # exactly the open
    assert next_session_after(datetime(2026, 9, 26, 0, 35, tzinfo=UTC)) == date(2026, 9, 28)  # Fri 20:35 ET
    assert next_session_after(datetime(2026, 11, 26, 15, 0, tzinfo=UTC)) == date(2026, 11, 27)  # Thanksgiving


# ---------------------------------------------------------------- contract, empty database, performance


def assert_contract(study: Study) -> None:
    for frame, columns in (
        (study.events, EVENT_COLUMNS), (study.posts, POST_COLUMNS), (study.groups, GROUP_COLUMNS),
        (study.magnitude, MAGNITUDE_COLUMNS), (study.car_path, PATH_COLUMNS),
        (study.intraday_path, INTRADAY_PATH_COLUMNS), (study.attention, ATTENTION_COLUMNS),
    ):  # fmt: skip
        assert list(frame.columns) == list(columns)
    assert set(study.counts) == COUNT_KEYS


def test_empty_database_gives_a_valid_empty_study(conn, watchlist):
    study = compute_study(conn, watchlist, NOW)
    assert_contract(study)
    for frame in (study.events, study.posts, study.groups, study.magnitude, study.car_path, study.intraday_path,
                  study.attention):  # fmt: skip
        assert frame.empty
    assert all(v == 0 for v in study.counts.values())
    assert any("No complete events" in n for n in study.notes)
    assert study.generated_at == NOW and study.options == StudyOptions()


def test_populated_study_matches_the_contract(conn, watchlist):
    s = Synth(conn)
    s.daily("AAPL", s.returns())
    s.daily("MSFT", s.returns())
    s.add("p1", ["AAPL", "MSFT"], D0)
    s.add("p2", ["AAPL"], rel(D0, 20), stance=None, conf=None)
    study = run(conn, watchlist)
    assert_contract(study)
    ev = study.events
    assert ev["t0_et"].iloc[0] == "2026-07-14 17:30:00" and ev["d0"].iloc[0] == "2026-07-15"
    assert math.isnan(event_row(study, int(ev["event_id"].iloc[-1]))["sign"])
    assert study.counts["posts_included"] == 2 and study.counts["posts_signed"] == 1
    assert study.counts["placebo_days"] == 15
    assert set(study.magnitude["sample"]) == {"events", "placebo"}


def test_five_thousand_events_compute_quickly(conn, watchlist):
    rng = np.random.default_rng(1)
    s = Synth(conn)
    n_tickers, per_ticker = 500, 10
    betas = rng.uniform(0.5, 2.0, n_tickers)
    returns = 0.0002 + betas[:, None] * s.market[None, :] + rng.normal(0, 0.015, (n_tickers, len(GRID)))
    prices = 50.0 * np.cumprod(1 + returns, axis=1)
    stamp = to_iso(NOW)
    rows = [
        (f"T{k:03d}", d.isoformat(), p, p, p, p, p, 1e6, 0.0, stamp)
        for k in range(n_tickers)
        for d, p in zip(GRID, prices[k], strict=True)
    ]
    posts, events = [], []
    event_id = 0
    for k in range(n_tickers):
        for i in sorted(rng.choice(np.arange(300, len(GRID) - 6), size=per_ticker, replace=False)):
            event_id += 1
            d0 = GRID[i]
            t0 = to_iso(datetime.combine(GRID[i - 1], time(21, 30), tzinfo=UTC))
            stance = ("bullish", "bearish", "neutral")[event_id % 3]
            posts.append(("truthsocial", f"n{event_id}", "realDonaldTrump", t0, "t", "u", stance, 0.8, stamp))
            events.append((event_id, "truthsocial", f"n{event_id}", f"T{k:03d}", t0, d0.isoformat(), "after",
                           "complete", "unavailable", 0, 0, 0, stamp))  # fmt: skip
    with conn:
        conn.executemany("INSERT INTO bars_1d VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
        conn.executemany(
            """INSERT INTO posts (platform, native_id, author, created_at_utc, text, url, stance, stance_conf,
                                  collected_at) VALUES (?,?,?,?,?,?,?,?,?)""",
            posts,
        )
        conn.executemany(
            """INSERT INTO events (id, platform, native_id, ticker, t0, d0, session_phase, status, intraday_state,
                                   earnings_flag, split_flag, clustered, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            events,
        )
    started = timer.perf_counter()
    study = run(conn, watchlist)
    elapsed = timer.perf_counter() - started
    assert study.counts["events_included"] == 5000
    assert study.counts["placebo_days"] >= 20000
    assert study.groups["status"].eq("ok").any()
    assert elapsed < 60, f"compute_study took {elapsed:.1f} s for 5,000 events"


def test_placebo_session_helper_edges():
    blocked = np.zeros(400, dtype=bool)
    assert len(metrics.placebo_sessions(-1, blocked, 5, 1, 1)) == 0
    assert len(metrics.placebo_sessions(300, blocked, 0, 1, 1)) == 0
    blocked[:] = True
    assert len(metrics.placebo_sessions(300, blocked, 5, 1, 1)) == 0
