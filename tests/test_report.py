from __future__ import annotations

import base64
import csv
import dataclasses
import math
import re
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_rgb

from influence_tracker import market
from influence_tracker.analysis import charts, report
from influence_tracker.analysis.study import (
    ATTENTION_COLUMNS,
    DAILY_WINDOWS,
    EVENT_COLUMNS,
    GROUP_COLUMNS,
    INTRADAY_PATH_COLUMNS,
    INTRADAY_PATH_MINUTES,
    MAGNITUDE_COLUMNS,
    MIN_GROUP_POSTS,
    PATH_COLUMNS,
    PATH_DAYS,
    POST_COLUMNS,
    REDDIT_CATEGORY,
    Study,
    StudyOptions,
)

PNG = b"\x89PNG\r\n\x1a\n"
GENERATED = datetime(2026, 9, 26, 19, 5, tzinfo=UTC)
SCRIPT_TEXT = "<script>alert('pwned')</script> Buying $NVDA & \"holding\" \U0001f680\U0001f680"
FORMULA_TEXT = '=HYPERLINK("http://evil.example","click") Tesla is finished'
BAD_URL = "javascript:alert(1)"
# The two biggest moves: an intraday NVDA post (1-minute bars stored) and a Sunday TSLA post (daily bars only).
NVDA_T0 = datetime(2026, 9, 14, 17, 33, 39, tzinfo=UTC)
TSLA_T0 = datetime(2026, 9, 6, 15, 0, tzinfo=UTC)
NY_TZ = "America/New_York"

AUTHORS = [
    ("truthsocial", "realDonaldTrump", "politician"),
    ("truthsocial", "WhiteHouse", "politician"),
    ("x", "elonmusk", "exec"),
    ("reddit", "wallstreetbets", REDDIT_CATEGORY),
]
AUTHOR_CYCLE = [0, 0, 2, 0, 1, 3, 0, 2, 0, 3, 0, 0]
STANCE_CYCLE = ["bullish", "bullish", "bearish", "neutral", "bullish", "bearish"]
TICKERS = ["NVDA", "TSLA", "INTC", "GOOGL", "AAPL", "LMT"]
SIGN = {"bullish": 1, "bearish": -1, "neutral": 0}
AR_COLS = ["ar_dm5", "ar_dm4", "ar_dm3", "ar_dm2", "ar_dm1", "ar_d0", "ar_dp1", "ar_dp2", "ar_dp3", "ar_dp4", "ar_dp5"]
IAR = ["iar_pre60", "iar_p5", "iar_p15", "iar_p30", "iar_p60"]


# ---------------------------------------------------------------- a hand-made Study that follows study.py


def _native_id(platform: str, i: int) -> str:
    return {"truthsocial": f"1152{i:014d}", "x": f"19700{i:014d}", "reddit": f"t3_{i:05x}"}[platform]


def _url(platform: str, author: str, native_id: str) -> str:
    if platform == "truthsocial":
        return f"https://truthsocial.com/@{author}/posts/{native_id}"
    if platform == "x":
        return f"https://x.com/{author}/status/{native_id}"
    return f"https://www.reddit.com/r/{author}/comments/{native_id[3:]}/"


def _events(n_posts: int, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    event_id = 100
    for i in range(n_posts):
        platform, author, category = AUTHORS[AUTHOR_CYCLE[i % len(AUTHOR_CYCLE)]]
        stance = STANCE_CYCLE[i % len(STANCE_CYCLE)]
        t0 = datetime(2026, 5, 4, 14, 31, tzinfo=UTC) + timedelta(days=4 * i, hours=(i % 3) * 3)
        tickers = [TICKERS[i % len(TICKERS)]] + ([TICKERS[(i + 2) % len(TICKERS)]] if i % 4 == 1 else [])
        text = f"Great things happening at ${tickers[0]} today #{i}"
        url_override = None
        if i == 0:
            t0, tickers, stance, text = NVDA_T0, ["NVDA"], "bullish", SCRIPT_TEXT
        elif i == 1:
            t0, tickers, stance, text, platform, author, category = (
                TSLA_T0,
                ["TSLA"],
                "bearish",
                FORMULA_TEXT,
                "truthsocial",
                "realDonaldTrump",
                "politician",
            )
        elif i == 2:
            url_override = BAD_URL
        native_id = _native_id(platform, i)
        conf = 0.45 + 0.54 * ((i * 37) % 100) / 100
        for ticker in tickers:
            ar = rng.normal(0, 0.012, 11)
            ar[5] += 0.006 * SIGN[stance]
            sigma = 0.015
            car_pre, car_event, car_post = ar[:5].sum(), ar[5:7].sum(), ar[7:].sum()
            z_event = float(np.clip(car_event / (sigma * math.sqrt(2)), -2.9, 2.9))
            if i == 0:
                z_event = 4.2
            elif i == 1:
                z_event = -3.6
            car_event = z_event * sigma * math.sqrt(2)
            regular = market.session_phase(t0) == "regular"
            intraday_ok = i % 3 == 0 or i == 0
            reason = ""
            if event_id == 103:
                reason = "earnings"
            elif event_id == 107:
                reason = "clustered"
            elif event_id == 111:
                reason = "no_model"
            iar = rng.normal(0.001 * SIGN[stance], 0.003, 5) if (intraday_ok and regular) else [np.nan] * 5
            row = {
                "event_id": event_id,
                "platform": platform,
                "native_id": native_id,
                "author": author,
                "category": category,
                "ticker": ticker,
                "t0": pd.Timestamp(t0),
                "t0_et": pd.Timestamp(t0).tz_convert(NY_TZ),
                "d0": market.event_session(t0),
                "session_phase": market.session_phase(t0),
                "stance": stance,
                "stance_conf": conf,
                "sign": SIGN[stance],
                "excluded_reason": reason,
                "earnings_flag": 1 if reason == "earnings" else (np.nan if event_id == 105 else 0),
                "split_flag": 0,
                "clustered": 1 if reason == "clustered" else 0,
                "intraday_state": "ok" if intraday_ok else "unavailable",
                "alpha": np.nan if reason == "no_model" else 0.0003,
                "beta": np.nan if reason == "no_model" else 1.2,
                "sigma": np.nan if reason == "no_model" else sigma,
                "n_est": 40 if reason == "no_model" else 120,
                **dict(zip(AR_COLS, ar, strict=True)),
                "car_pre": car_pre,
                "z_pre": car_pre / (sigma * math.sqrt(5)),
                "car_event": car_event,
                "z_event": z_event,
                "car_post": car_post,
                "z_post": car_post / (sigma * 2),
                "ar_pre_leg": ar[5] * 0.4 if intraday_ok else np.nan,
                "z_pre_leg": 0.8 if intraday_ok else np.nan,
                "ar_post_leg": ar[5] * 0.6 if intraday_ok else np.nan,
                "z_post_leg": 1.1 if intraday_ok else np.nan,
                **{c: v for c, v in zip(IAR, iar, strict=True)},
                **{
                    c.replace("iar", "iz"): (v / 0.004 if not math.isnan(v) else np.nan)
                    for c, v in zip(IAR, iar, strict=True)
                },
                "rel_volume": 1.3,
                "spy_ret_d0": rng.normal(0, 0.008),
                "spy_post_leg": rng.normal(0, 0.004),
                "text": text,
                "url": url_override or _url(platform, author, native_id),
            }
            rows.append(row)
            event_id += 1
    return pd.DataFrame(rows, columns=list(EVENT_COLUMNS))


def _posts(events: pd.DataFrame) -> pd.DataFrame:
    inc = events[events["excluded_reason"] == ""]
    rows = []
    for _, grp in inc.groupby(["platform", "native_id"], sort=False):
        first = grp.iloc[0]
        sign = first["sign"]
        row = {
            c: first[c]
            for c in (
                "platform",
                "native_id",
                "author",
                "category",
                "t0",
                "t0_et",
                "stance",
                "stance_conf",
                "sign",
                "text",
                "url",
            )
        }
        row["n_events"] = len(grp)
        row["tickers"] = ",".join(grp["ticker"])
        for c in (
            "car_pre",
            "car_event",
            "car_post",
            "z_pre",
            "z_event",
            "z_post",
            *AR_COLS,
            "ar_pre_leg",
            "ar_post_leg",
            *IAR,
        ):
            row[c] = grp[c].mean()
        for w in ("pre", "event", "post"):
            row[f"signed_car_{w}"] = sign * row[f"car_{w}"] if sign else np.nan
        rows.append(row)
    return pd.DataFrame(rows, columns=list(POST_COLUMNS))


def _holm(p: list[float]) -> list[float]:
    order = np.argsort(p)
    out = np.empty(len(p))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        out[i] = running
    return list(out)


def _groups(posts: pd.DataFrame, conf_cut: float) -> pd.DataFrame:
    from scipy import stats

    rows = []
    for subset in ("main", "confident"):
        sub = posts if subset == "main" else posts[posts["stance_conf"] >= conf_cut]
        for window in DAILY_WINDOWS:
            for family in ("all", "author", "category", "platform", "stance"):
                keys = ["all"] if family == "all" else sorted(sub[family].dropna().unique())
                fam_rows = []
                for key in keys:
                    sel = sub if family == "all" else sub[sub[family] == key]
                    signed = sel[f"signed_car_{window}"].dropna()
                    n = len(sel) if family == "stance" and key == "neutral" else len(signed)
                    z = sel[f"z_{window}"].abs()
                    ok = len(signed) >= MIN_GROUP_POSTS
                    mean = signed.mean() if len(signed) else np.nan
                    se = signed.std(ddof=1) / math.sqrt(len(signed)) if len(signed) > 1 else np.nan
                    half = stats.t.ppf(0.975, len(signed) - 1) * se if len(signed) > 1 else np.nan
                    t = mean / se if ok else np.nan
                    p = 2 * stats.t.sf(abs(t), len(signed) - 1) if ok else np.nan
                    fam_rows.append(
                        {
                            "family": family,
                            "group": key,
                            "subset": subset,
                            "window": window,
                            "n_posts": n,
                            "mean_signed_car": mean,
                            "median_signed_car": signed.median() if len(signed) else np.nan,
                            "ci_low": mean - half,
                            "ci_high": mean + half,
                            "mean_abs_z": z.mean(),
                            "share_abs_z_gt_crit": (z > 1.96).mean(),
                            "t_stat": t,
                            "p_value": p,
                            "p_holm": np.nan,
                            "status": "ok" if ok else "insufficient",
                        }
                    )
                tested = [r for r in fam_rows if r["status"] == "ok"]
                for r, adj in zip(tested, _holm([r["p_value"] for r in tested]), strict=True):
                    r["p_holm"] = adj
                rows += fam_rows
    return pd.DataFrame(rows, columns=list(GROUP_COLUMNS))


def _path(series_n: dict[str, int], with_ci: bool = True) -> pd.DataFrame:
    shapes = {"bullish": 0.0028, "bearish": -0.0021, "neutral": 0.0004, "placebo": 0.0002}
    rows = []
    for series, n in series_n.items():
        level = 0.0
        for day in PATH_DAYS:
            level += shapes[series] * (4 if day in (0, 1) else 0.5)
            half = 0.004 * math.sqrt(day + 6) if with_ci and n > 1 else np.nan
            rows.append(
                {
                    "series": series,
                    "day": day,
                    "mean_car": level,
                    "ci_low": level - half,
                    "ci_high": level + half,
                    "n": n,
                }
            )
    return pd.DataFrame(rows, columns=list(PATH_COLUMNS))


def _intraday(series_n: dict[str, int]) -> pd.DataFrame:
    shapes = {
        "bullish": [-0.0012, 0, 0.0021, 0.0034, 0.0041, 0.0046],
        "bearish": [0.0006, 0, -0.001, -0.0022, -0.0017, -0.0031],
    }
    rows = []
    for series, n in series_n.items():
        for minute, mean in zip(INTRADAY_PATH_MINUTES, shapes[series], strict=True):
            half = 0.0015 if minute else 0.0
            rows.append(
                {
                    "series": series,
                    "minute": minute,
                    "mean_ar": mean,
                    "ci_low": mean - half,
                    "ci_high": mean + half,
                    "n": n,
                }
            )
    return pd.DataFrame(rows, columns=list(INTRADAY_PATH_COLUMNS))


def _magnitude(n_events: int, n_placebo: int) -> pd.DataFrame:
    rows = []
    for window, ev_share, pl_share in (("pre", 0.1154, 0.05), ("event", 0.25, 0.0583), ("post", 0.0769, 0.0417)):
        rows.append(
            {
                "window": window,
                "sample": "events",
                "n": n_events,
                "mean_car": 0.0011,
                "mean_abs_car": 0.0213,
                "median_abs_car": 0.0172,
                "share_abs_z_gt_crit": ev_share,
            }
        )
        rows.append(
            {
                "window": window,
                "sample": "placebo",
                "n": n_placebo,
                "mean_car": -0.0002,
                "mean_abs_car": 0.0141,
                "median_abs_car": 0.0113,
                "share_abs_z_gt_crit": pl_share,
            }
        )
    return pd.DataFrame(rows, columns=list(MAGNITUDE_COLUMNS))


def _attention() -> pd.DataFrame:
    rows = [
        ("GME", date(2026, 9, 22), 1840, 410.5, 4.48, date(2026, 9, 23), 0.034, 1.91),
        ("NVDA", date(2026, 9, 22), 2210, 1480.0, 1.49, date(2026, 9, 23), -0.004, -0.31),
        ("PLTR", date(2026, 9, 23), 960, 300.25, 3.2, date(2026, 9, 24), np.nan, np.nan),
    ]
    return pd.DataFrame(rows, columns=list(ATTENTION_COLUMNS))


def _counts(events: pd.DataFrame, posts: pd.DataFrame, placebo_days: int, attention_days: int) -> dict[str, int]:
    inc = events[events["excluded_reason"] == ""]
    counts = {"events_complete": len(events), "events_included": len(inc)}
    for reason in ("earnings", "split", "clustered", "no_model"):
        counts[f"excluded_{reason}"] = int((events["excluded_reason"] == reason).sum())
    counts |= {
        "earnings_unknown": int(inc["earnings_flag"].isna().sum()),
        "events_intraday": int((inc["intraday_state"] == "ok").sum()),
        "posts_included": len(posts),
        "posts_signed": int(posts["sign"].isin([1, -1]).sum()),
        "placebo_days": placebo_days,
        "attention_days": attention_days,
    }
    return counts


def make_study(n_posts: int = 24, *, seed: int = 7) -> Study:
    options = StudyOptions()
    events = _events(n_posts, np.random.default_rng(seed))
    posts = _posts(events)
    groups = _groups(posts, options.min_conf_robust)
    headline = (groups["family"] == "all") & (groups["subset"] == "main") & (groups["window"] == "event")
    if n_posts >= 20:
        groups.loc[headline, ["mean_signed_car", "median_signed_car", "ci_low", "ci_high", "p_value", "p_holm"]] = [
            0.0123,
            0.0101,
            0.0061,
            0.0185,
            0.0004,
            0.0004,
        ]
    big = n_posts >= 20
    return Study(
        options=options,
        generated_at=GENERATED,
        events=events,
        posts=posts,
        groups=groups,
        magnitude=_magnitude(24, 120) if big else _magnitude(3, 15),
        car_path=_path({"bullish": 14, "bearish": 6, "neutral": 4, "placebo": 120})
        if big
        else _path({"bullish": 2, "bearish": 1}, with_ci=False),
        intraday_path=_intraday({"bullish": 5, "bearish": 3})
        if big
        else pd.DataFrame(columns=list(INTRADAY_PATH_COLUMNS)),
        attention=_attention() if big else pd.DataFrame(columns=list(ATTENTION_COLUMNS)),
        counts=_counts(events, posts, 120 if big else 15, 9 if big else 2),
        notes=[
            "ApeWisdom: 9 days of snapshots collected; spikes need a trailing week.",
            "2 events had fewer than 10 prior sessions of 1-minute data, so their intraday z is empty.",
        ],
    )


def empty_study() -> Study:
    def empty(columns: tuple[str, ...]) -> pd.DataFrame:
        return pd.DataFrame(columns=list(columns))

    return Study(
        options=StudyOptions(since=date(2026, 9, 1)),
        generated_at=GENERATED,
        events=empty(EVENT_COLUMNS),
        posts=empty(POST_COLUMNS),
        groups=empty(GROUP_COLUMNS),
        magnitude=empty(MAGNITUDE_COLUMNS),
        car_path=empty(PATH_COLUMNS),
        intraday_path=empty(INTRADAY_PATH_COLUMNS),
        attention=empty(ATTENTION_COLUMNS),
    )


# ---------------------------------------------------------------- price bars for the top-event charts


def _insert_minutes(conn, symbol: str, start: datetime, end: datetime, price) -> None:
    rows = []
    t = start
    while t < end:
        p = price(t)
        rows.append((symbol, int(t.timestamp()), p, p, p, p, 1000.0))
        t += timedelta(minutes=1)
    conn.executemany("INSERT INTO bars_1m (symbol, ts, open, high, low, close, volume) VALUES (?,?,?,?,?,?,?)", rows)


def _insert_daily(conn, symbol: str, d0: date, before: float, after: float) -> None:
    sessions = market.sessions_in_range(market.session_offset(d0, -12), market.session_offset(d0, 12))
    conn.executemany(
        "INSERT INTO bars_1d (symbol, session_date, close, adj_close, volume, fetched_at) VALUES (?,?,?,?,?,?)",
        [(symbol, s.isoformat(), before if s < d0 else after, before if s < d0 else after, 1e6, "x") for s in sessions],
    )


@pytest.fixture
def priced_conn(conn):
    """1-minute NVDA/SPY bars around NVDA_T0 (a +3% jump in the minute of the post), daily TSLA/SPY bars around
    TSLA_T0's d0 (-5% from d0 on)."""
    minute_of_post = NVDA_T0.replace(second=0)
    with conn:
        _insert_minutes(
            conn,
            "NVDA",
            datetime(2026, 9, 14, 13, 30, tzinfo=UTC),
            datetime(2026, 9, 15, 0, 30, tzinfo=UTC),
            lambda t: 180.0 if t < minute_of_post else 185.4,
        )
        _insert_minutes(
            conn,
            "SPY",
            datetime(2026, 9, 14, 13, 30, tzinfo=UTC),
            datetime(2026, 9, 15, 0, 0, tzinfo=UTC),
            lambda t: 660.0 if t < minute_of_post else 661.32,
        )
        d0 = market.event_session(TSLA_T0)
        _insert_daily(conn, "TSLA", d0, 400.0, 380.0)
        _insert_daily(conn, "SPY", d0, 650.0, 653.25)
    return conn


# ---------------------------------------------------------------- helpers


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return list(reader.fieldnames or []), rows


def _data_uris(page: str) -> list[bytes]:
    return [base64.b64decode(m) for m in re.findall(r'src="data:image/png;base64,([A-Za-z0-9+/=]+)"', page)]


def _png_size(data: bytes) -> tuple[int, int]:
    assert data.startswith(PNG)
    return int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")


def _assert_self_contained(page: str, study: Study) -> None:
    post_links = {u for u in study.events["url"] if isinstance(u, str) and u.startswith("https://")}
    for attr, value in re.findall(r'\b(src|href)\s*=\s*["\']([^"\']*)["\']', page, re.IGNORECASE):
        if attr.lower() == "src":
            assert value.startswith("data:image/png;base64,"), value[:80]
        else:
            assert value in post_links, value
    lowered = page.lower()
    for forbidden in ("<link", "@import", "url(", "<iframe", "<object", "<embed", " src=http"):
        assert forbidden not in lowered, forbidden


# ---------------------------------------------------------------- formatting


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.0123, "+1.23%"),
        (-0.005, "-0.50%"),
        (0.0, "0.00%"),
        (-0.00001, "0.00%"),
        (np.nan, "—"),
        (None, "—"),
        (pd.NA, "—"),
        ("x", "—"),
    ],
)
def test_fmt_pct(value, expected):
    assert report.fmt_pct(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"), [(0.0004, "<0.001"), (0.001, "0.001"), (0.0123, "0.012"), (1.0, "1.000"), (np.nan, "—")]
)
def test_fmt_p(value, expected):
    assert report.fmt_p(value) == expected


def test_other_formats():
    assert report.fmt_share(0.0583) == "5.8%"
    assert report.fmt_z(-1.456) == "-1.46"
    assert report.fmt_ci(-0.005, 0.0218) == "-0.50% to +2.18%"
    assert report.fmt_ci(np.nan, 0.01) == "—"
    assert report.fmt_int(np.float64(12.0)) == "12"


# ---------------------------------------------------------------- write_report on a realistic study


def test_write_report_realistic_study(tmp_path, priced_conn):
    study = make_study()
    out = tmp_path / "reports" / "2026-09-26"

    path = report.write_report(study, priced_conn, out)

    assert path == out / "report.html"
    for name in ("report.html", "events.csv", "posts.csv", "summary.csv", "placebo.csv"):
        assert (out / name).is_file(), name
    for name in ("car_path.png", "author_effect.png", "event_vs_placebo.png", "intraday_path.png"):
        assert _png_size((out / "charts" / name).read_bytes()) == (1600, 900), name
    tops = sorted(p.name for p in (out / "charts" / "top_events").glob("*.png"))
    # Only the two events with stored bars get a chart; the other eight say so in the report.
    assert tops == ["01_NVDA_100.png", "02_TSLA_101.png"]
    assert _png_size((out / "charts" / "top_events" / tops[0]).read_bytes()) == (1200, 640)

    page = path.read_text(encoding="utf-8")
    _assert_self_contained(page, study)
    images = _data_uris(page)
    assert len(images) == 6 and all(img.startswith(PNG) for img in images)

    # sections in order
    headings = [
        "How social-media posts moved stocks",
        "How this works",
        "The average path around a post",
        "Results by author",
        "Real events vs ordinary days",
        "Minute by minute",
        "Market-wide moves",
        "Reddit attention",
        "The biggest moves",
        "Caveats",
        "Appendix",
    ]
    positions = [page.index(h) for h in headings]
    assert positions == sorted(positions)

    # key numbers, formatted
    assert "+1.23%" in page and "95% CI +0.61% to +1.85%" in page and "p &lt; 0.001" in page
    assert "25.0% vs 5.8%" in page and "25.0% of real events (6 of 24)" in page
    assert "Sep 26, 2026 15:05 ET" in page
    ev = study.counts
    assert f"{ev['events_included']} of {ev['events_complete']} complete events included" in page
    assert "excluded: 1 earnings, 1 clustered, 1 no model" in page
    assert "<td class=r>&lt;0.001</td>" in page

    # insufficient groups are greyed and labelled; WhiteHouse has too few posts to test
    assert re.search(r'<tr class="muted"><td>WhiteHouse</td>', page)
    assert "n&lt;10: not tested" in page
    assert "neutral posts are not signed" in page

    # post text is escaped, emoji preserved; the javascript: link is not a link
    assert "&lt;script&gt;alert(&#x27;pwned&#x27;)&lt;/script&gt; Buying $NVDA &amp; &quot;holding&quot;" in page
    assert "<script>alert" not in page
    assert "\U0001f680\U0001f680" in page
    assert BAD_URL not in page
    assert "https://truthsocial.com/@realDonaldTrump/posts/115200000000000000" in page

    # the notes and settings reach the report
    for note in study.notes:
        assert note.replace("'", "&#x27;") in page
    assert "<code>min_conf_robust</code>" in page and "20260928" in page
    assert "charts/top_events/01_NVDA_100.png" in page


def test_csvs_round_trip_with_utf8_sig_guards_and_exact_ids(tmp_path, priced_conn):
    study = make_study()
    out = tmp_path / "r"
    report.write_report(study, priced_conn, out)

    assert (out / "events.csv").read_bytes().startswith(b"\xef\xbb\xbf")
    header, rows = _read_csv(out / "events.csv")
    assert header == list(EVENT_COLUMNS)
    assert len(rows) == len(study.events)
    by_id = {r["event_id"]: r for r in rows}
    nvda = by_id["100"]
    assert nvda["text"] == SCRIPT_TEXT
    assert nvda["native_id"] == '="115200000000000000"'
    assert float(nvda["car_event"]) == pytest.approx(study.events.loc[0, "car_event"], rel=1e-12)
    assert nvda["t0"] == "2026-09-14 17:33:39+00:00" and nvda["t0_et"] == "2026-09-14 13:33:39-04:00"
    assert nvda["d0"] == "2026-09-14" and nvda["excluded_reason"] == ""
    assert by_id["101"]["text"] == "'" + FORMULA_TEXT
    assert by_id["111"]["alpha"] == "" and by_id["111"]["excluded_reason"] == "no_model"
    reddit = next(r for r in rows if r["platform"] == "reddit")
    assert reddit["native_id"].startswith("t3_")

    header, posts = _read_csv(out / "posts.csv")
    assert header == list(POST_COLUMNS) and len(posts) == len(study.posts)
    header, groups = _read_csv(out / "summary.csv")
    assert header == list(GROUP_COLUMNS) and len(groups) == len(study.groups)
    headline = next(g for g in groups if (g["family"], g["subset"], g["window"]) == ("all", "main", "event"))
    assert float(headline["mean_signed_car"]) == 0.0123 and headline["status"] == "ok"
    header, placebo = _read_csv(out / "placebo.csv")
    assert header == list(MAGNITUDE_COLUMNS) and len(placebo) == 6


def test_headlines_for_the_cli():
    study = make_study()
    lines = report.headlines(study)
    assert len(lines) == 3
    assert lines[0] == (
        f"Posts analyzed: {len(study.posts)} ({study.counts['events_included']} events included, 3 excluded)"
    )
    n_signed = int(study.posts["sign"].isin([1, -1]).sum())
    assert lines[1] == f"Mean signed CAR[0,+1]: +1.23% (95% CI +0.61% to +1.85%, p < 0.001, n = {n_signed} posts)"
    assert lines[2] == ("|z| > 1.96 in the event window: 25.0% of events (6 of 24) vs 5.8% of placebo days (7 of 120)")


# ---------------------------------------------------------------- tiny and empty studies


def test_write_report_tiny_study_says_insufficient_data(tmp_path, conn):
    study = make_study(3)
    out = tmp_path / "tiny"

    page = report.write_report(study, conn, out).read_text(encoding="utf-8")

    _assert_self_contained(page, study)
    assert "insufficient data (n &lt; 10)" in page
    assert "the test needs at least 10" in page
    # Every group, in every subset and window, is under the test minimum.
    assert len(re.findall(r'<tr class="(?:total )?muted">', page)) == len(study.groups)
    assert "No included bullish or bearish post has intraday windows yet" in page
    assert "Not enough ApeWisdom history yet: 2 daily snapshots collected" in page
    # No bars in this database: each top event says so instead of showing a chart.
    assert "No price bars stored around this post." in page
    assert not (out / "charts" / "top_events").exists() or not list((out / "charts" / "top_events").glob("*.png"))
    assert (out / "charts" / "car_path.png").is_file()
    assert not (out / "charts" / "intraday_path.png").exists()
    lines = report.headlines(study)
    assert lines[1] == "Mean signed CAR[0,+1]: insufficient data (n = 3 signed posts; the test needs 10)"


def test_write_report_empty_study(tmp_path, conn):
    study = empty_study()
    out = tmp_path / "empty"

    page = report.write_report(study, conn, out).read_text(encoding="utf-8")

    _assert_self_contained(page, study)
    assert "<img" not in page
    assert not list(out.rglob("*.png"))
    for name, columns in (
        ("events.csv", EVENT_COLUMNS),
        ("posts.csv", POST_COLUMNS),
        ("summary.csv", GROUP_COLUMNS),
        ("placebo.csv", MAGNITUDE_COLUMNS),
    ):
        header, rows = _read_csv(out / name)
        assert header == list(columns) and rows == []
    assert "insufficient data (n &lt; 10)" in page
    assert "No bullish or bearish post has a complete day" in page
    assert "No included event has an event-window z-score yet." in page
    assert "No included events come from politician accounts." in page
    assert "filtered to post days from 2026-09-01" in page
    assert report.headlines(study) == [
        "Posts analyzed: 0 (0 events included, 0 excluded)",
        "Mean signed CAR[0,+1]: insufficient data (n = 0 signed posts; the test needs 10)",
        "|z| > 1.96 in the event window: insufficient data (no events with a z-score)",
    ]


def test_a_rerun_into_the_same_folder_drops_stale_charts(tmp_path, priced_conn):
    out = tmp_path / "same"
    report.write_report(make_study(), priced_conn, out)
    assert list(out.rglob("*.png"))

    page = report.write_report(empty_study(), priced_conn, out).read_text(encoding="utf-8")

    assert not list(out.rglob("*.png"))
    assert "top_events/" not in page


# ---------------------------------------------------------------- prices behind the top-event charts


def _event_row(ticker: str, t0: datetime) -> pd.Series:
    return pd.Series({"ticker": ticker, "t0": pd.Timestamp(t0), "d0": market.event_session(t0)})


def test_intraday_prices_start_at_the_reference_price(priced_conn):
    prices = report.load_event_prices(priced_conn, _event_row("NVDA", NVDA_T0))

    assert prices is not None and prices.kind == "intraday"
    before = prices.ticker[prices.ticker.index < 0]
    after = prices.ticker[prices.ticker.index > 0]
    # Bars are plotted at their END: the last one finished by t0 closes at the reference price.
    assert before.max() == 0 and before.min() == 0
    assert after.iloc[0] == pytest.approx(185.4 / 180 - 1)
    assert prices.ticker.index.min() >= -60 and prices.ticker.index.max() <= 120
    assert prices.spy[prices.spy.index > 0].iloc[0] == pytest.approx(661.32 / 660 - 1)


def test_intraday_window_is_clipped_to_the_extended_session(priced_conn):
    t0 = datetime(2026, 9, 14, 23, 30, tzinfo=UTC)  # 19:30 New York, half an hour before extended hours end

    prices = report.load_event_prices(priced_conn, _event_row("NVDA", t0))

    assert prices is not None and prices.kind == "intraday"
    assert prices.ticker.index.max() <= 30


def test_weekend_post_falls_back_to_daily_closes(priced_conn):
    prices = report.load_event_prices(priced_conn, _event_row("TSLA", TSLA_T0))

    assert prices is not None and prices.kind == "daily"
    assert list(prices.ticker.index) == list(range(-10, 11))
    assert prices.ticker[-10] == 0 and prices.ticker[0] == pytest.approx(-0.05)
    assert prices.spy[0] == pytest.approx(653.25 / 650 - 1)


def test_no_bars_means_no_prices(conn):
    assert report.load_event_prices(conn, _event_row("NVDA", NVDA_T0)) is None


# ---------------------------------------------------------------- charts with nothing to draw


def test_charts_return_none_with_nothing_to_draw(tmp_path):
    empty = empty_study()
    assert charts.car_path_chart(empty.car_path, tmp_path / "a.png") is None
    assert charts.author_chart(empty.groups, tmp_path / "b.png") is None
    assert charts.placebo_chart(empty.magnitude, tmp_path / "c.png") is None
    assert charts.intraday_chart(empty.intraday_path, tmp_path / "d.png") is None
    assert charts.top_event_chart(pd.Series({"ticker": "NVDA"}), None, tmp_path / "e.png") is None
    # Only neutral posts and placebo days: no bullish/bearish line to draw.
    only_ref = _path({"neutral": 4, "placebo": 50})
    assert charts.car_path_chart(only_ref, tmp_path / "f.png") is None
    # Placebo days alone say nothing about posts.
    placebo_only = _magnitude(0, 50)
    assert charts.placebo_chart(placebo_only[placebo_only["sample"] == "placebo"], tmp_path / "g.png") is None
    assert not list(tmp_path.glob("*.png"))


def _author_groups(n: int) -> pd.DataFrame:
    rows = []
    for i in range(n):
        posts = 30 - i
        mean = 0.004 * ((i % 7) - 3)
        rows.append(
            {
                "family": "author",
                "group": f"author_{i:02d}",
                "subset": "main",
                "window": "event",
                "n_posts": posts,
                "mean_signed_car": mean,
                "median_signed_car": mean,
                "ci_low": mean - 0.006,
                "ci_high": mean + 0.006,
                "mean_abs_z": 1.0,
                "share_abs_z_gt_crit": 0.1,
                "t_stat": 1.0,
                "p_value": 0.3,
                "p_holm": 0.9,
                "status": "ok" if posts >= MIN_GROUP_POSTS else "insufficient",
            }
        )
    return pd.DataFrame(rows, columns=list(GROUP_COLUMNS))


def test_author_chart_caps_groups_and_says_how_many_were_folded(tmp_path, conn):
    groups = _author_groups(20)
    rows = charts.author_rows(groups)
    assert len(rows.shown) == 15 and len(rows.folded) == 5
    assert list(rows.shown["group"])[:2] == ["author_00", "author_01"]

    assert charts.author_chart(groups, tmp_path / "authors.png") is not None

    study = dataclasses.replace(empty_study(), groups=groups)
    page = report.write_report(study, conn, tmp_path / "out").read_text(encoding="utf-8")
    assert "The chart shows the 15 authors with the most posts; 5 more (65 posts) are in the table." in page
    assert page.count("author_1") == 10  # every author, including the folded ones, is in the table


# ---------------------------------------------------------------- review regressions


@pytest.fixture
def drawn(monkeypatch):
    """Every matplotlib Figure a chart function saves, so a test can inspect what was drawn."""
    figures = []
    save = charts._Canvas.save

    def keep(canvas, out):
        figures.append(canvas.fig)
        return save(canvas, out)

    monkeypatch.setattr(charts._Canvas, "save", keep)
    return figures


def test_holm_is_described_per_family_not_per_table(tmp_path, conn):
    page = report.write_report(make_study(), conn, tmp_path / "r").read_text(encoding="utf-8")
    method = page[page.index('<section id="method">') : page.index('<section id="car-path">')]
    caveats = page[page.index('<section id="caveats">') : page.index('<section id="appendix">')]

    # metrics.py adjusts within one (family, subset, window); each results table holds several families.
    assert page.count("same table") == 0 and page.count("within each table") == 0
    assert "tested in the same family (the authors, the categories, the platforms or the stances)" in method
    assert "It does not correct across families, windows or sets of posts." in method
    assert "Holm corrects only within one family of groups" in caveats


def test_market_table_and_cards_show_day_0_next_to_the_post_time(tmp_path, priced_conn):
    page = report.write_report(make_study(), priced_conn, tmp_path / "r").read_text(encoding="utf-8")

    market = page[page.index('<section id="market">') : page.index('<section id="attention">')]
    assert '<th scope="col">Day 0</th>' in market and "SPY on day 0 (raw)" in market
    # The Sunday TSLA post (Sep 6) has Tuesday Sep 8 as day 0: Monday Sep 7 was Labor Day.
    assert "<tr><td>Sep 6, 2026 11:00 ET</td><td>Tue Sep 8, 2026</td><td>realDonaldTrump</td><td>TSLA</td>" in market
    top = page[page.index('<section id="top">') : page.index('<section id="caveats">')]
    assert "<dt>Day 0</dt><dd>Tue Sep 8, 2026 (first session after the post)</dd>" in top
    # A regular-session post's day 0 is its own date.
    assert "<dt>Day 0</dt><dd>Mon Sep 14, 2026</dd>" in top


def test_results_tables_fit_a_printed_page(tmp_path, conn):
    page = report.write_report(make_study(), conn, tmp_path / "r").read_text(encoding="utf-8")

    # Untested groups give their reason where the p-values would be, so the table has no tenth column.
    for table in re.findall(r"<table>.*?</table>", page):
        header = re.search(r"<thead><tr>(.*?)</tr></thead>", table)
        assert header is not None and header.group(1).count("<th") <= 9
    results = page[page.index('<section id="results">') : page.index('<section id="placebo">')]
    assert "<td class=r colspan=2>n&lt;10: not tested</td>" in results
    assert results.count('colspan="10"') == 0 and '<th colspan="9" scope="colgroup">By author</th>' in results
    # A printed page cannot scroll: numeric cells and headers must be able to wrap there.
    print_css = page[page.index("@media print") :]
    assert "td.r { white-space: normal; }" in print_css and "table { font-size: 8pt; }" in print_css
    assert ".r { text-align: right; }\ntd.r { white-space: nowrap; }" in page


def _placebo_frame(cells: dict[tuple[str, str], tuple[float, int]]) -> pd.DataFrame:
    rows = [
        {
            "window": window,
            "sample": sample,
            "n": n,
            "mean_car": 0.0,
            "mean_abs_car": 0.01,
            "median_abs_car": 0.01,
            "share_abs_z_gt_crit": share,
        }
        for (window, sample), (share, n) in cells.items()
    ]
    return pd.DataFrame(rows, columns=list(MAGNITUDE_COLUMNS))


def test_placebo_labels_near_5_percent_are_not_struck_through(tmp_path, drawn):
    frame = _placebo_frame(
        {
            ("pre", "events"): (0.0, 45),
            ("pre", "placebo"): (0.08, 225),
            ("event", "events"): (2 / 45, 45),
            ("event", "placebo"): (0.045, 225),
            ("post", "events"): (0.067, 45),
            ("post", "placebo"): (0.093, 225),
        }
    )
    assert charts.placebo_chart(frame, tmp_path / "p.png") is not None

    fig = drawn[-1]
    ax = fig.axes[0]
    chance = next(line for line in ax.lines if list(line.get_ydata()) == [0.05, 0.05])
    labels = {t.get_text(): t for t in ax.texts if re.fullmatch(r"\d+\.\d%", t.get_text())}
    assert {"4.4%", "4.5%", "0.0%"} <= set(labels)
    renderer = fig.canvas.get_renderer()
    line_y = ax.transData.transform((0, 0.05))[1]
    extents = [t.get_window_extent(renderer) for t in labels.values()]
    crossed = [box for box in extents if box.y0 <= line_y <= box.y1]
    assert crossed  # the case the review found: the chance line runs through these labels
    for text in labels.values():
        # An opaque surface box drawn above the chance line keeps the value readable wherever the line crosses.
        box = text.get_bbox_patch()
        assert box is not None and to_rgb(box.get_facecolor()) == pytest.approx(to_rgb(charts.SURFACE))
        assert text.get_zorder() > chance.get_zorder()


def test_placebo_share_axis_stops_at_100_percent(tmp_path, drawn):
    frame = _placebo_frame(
        {
            ("event", "events"): (1.0, 3),
            ("event", "placebo"): (0.0, 15),
            ("post", "events"): (0.0, 3),
            ("post", "placebo"): (1.0, 15),
        }
    )
    assert charts.placebo_chart(frame, tmp_path / "p.png") is not None

    ax = drawn[-1].axes[0]
    assert ax.get_ylim()[1] > 1.0  # headroom for the "100.0%" label
    assert max(ax.get_yticks()) == pytest.approx(1.0)


# Minutes from t0 (bar end) for an after-hours post at 18:02 New York; the regular session ended 122 minutes earlier.
AFTER_REGULAR = (-512.0, -122.0)


def test_isolated_prints_flags_only_off_hours_snapbacks():
    minutes = np.arange(-60.0, 120.0)
    values = np.zeros(len(minutes))
    values[5] = -0.062  # one bad print, straight back
    values[40:42] = 0.005  # two bars away together, then back
    values[80] = 0.002  # a jump that comes back but stays under the 0.25% floor: ordinary thin-trading noise
    values[100:] = 0.02  # a real move that holds
    values[130] = 0.05  # jumps away and settles at a new level, not back where it came from
    values[131:] = 0.03
    series = pd.Series(values, index=minutes)

    flagged = charts.isolated_prints(series, AFTER_REGULAR)

    assert list(series.index[flagged]) == [-55.0, -20.0, -19.0]
    # Regular-session bars are never flagged, and daily charts have no regular session to test against.
    assert not charts.isolated_prints(series, (-100.0, 200.0)).any()
    assert not charts.isolated_prints(series, None).any()


def test_top_event_chart_scale_ignores_stray_prints(tmp_path, drawn):
    minutes = np.arange(-60.0, 118.0)
    ticker = pd.Series(0.0004 * np.sin(minutes / 9), index=minutes)
    ticker.iloc[3] = -0.0647  # the MSFT 17:05 print from the live data
    ticker.iloc[104] = 0.0251
    prices = charts.EventPrices(
        "intraday", ticker, pd.Series(0.0, index=minutes), datetime(2026, 9, 4, 22, 2, tzinfo=UTC), AFTER_REGULAR
    )
    event = {"ticker": "MSFT", "author": "realDonaldTrump", "stance": "bullish", "car_event": -0.0089, "z_event": -0.28}

    assert charts.top_event_chart(event, prices, tmp_path / "t.png") is not None

    fig = drawn[-1]
    ax = fig.axes[0]
    lo, hi = ax.get_ylim()
    assert -0.002 < lo < 0 < hi < 0.002  # the real path fills the chart
    edge_marks = sorted((line.get_marker(), line.get_ydata()[0]) for line in ax.lines if line.get_marker() in "v^")
    assert edge_marks == [("^", hi), ("v", lo)]
    legend = [t.get_text() for t in fig.legends[0].get_texts()]
    assert "stray one-minute prints, not joined; arrowheads are off the scale" in legend


def test_stray_prints_are_explained_on_the_card(tmp_path, conn):
    t0 = datetime(2026, 9, 14, 22, 2, tzinfo=UTC)  # 18:02 New York, after the close
    start, end = datetime(2026, 9, 14, 20, 0, tzinfo=UTC), datetime(2026, 9, 15, 0, 0, tzinfo=UTC)
    bad = datetime(2026, 9, 14, 21, 5, tzinfo=UTC)
    with conn:
        _insert_minutes(conn, "NVDA", start, end, lambda t: 169.2 if t == bad else 180.0)
        _insert_minutes(conn, "SPY", start, end, lambda t: 660.0)
    study = make_study(3)
    events = study.events.copy()
    first = events["event_id"] == 100
    events.loc[first, "t0"] = pd.Timestamp(t0)
    events.loc[first, "d0"] = market.event_session(t0)
    study = dataclasses.replace(study, events=events)

    page = report.write_report(study, conn, tmp_path / "r").read_text(encoding="utf-8")

    assert (tmp_path / "r" / "charts" / "top_events" / "01_NVDA_100.png").is_file()
    assert "Outside regular hours, 1 NVDA print (largest -6.00%) jumped away from the price and came straight" in page
    assert "The event&#x27;s numbers use the stored bars unchanged." in page
    assert "<dt>Day 0</dt><dd>Tue Sep 15, 2026 (first session after the post)</dd>" in page


def test_unlabelled_posts_do_not_borrow_a_stance_color(tmp_path, drawn):
    for stance in (None, np.nan, "unlabelled", "mixed"):
        assert charts.stance_color(stance) == charts.UNLABELLED
    assert charts.UNLABELLED not in charts.STANCE_COLORS.values()
    assert charts.stance_color("bearish") == charts.BEARISH

    minutes = np.arange(-60.0, 60.0)
    prices = charts.EventPrices(
        "intraday", pd.Series(minutes / 10000, index=minutes), pd.Series(0.0, index=minutes), NVDA_T0, (-200.0, 200.0)
    )
    event = {"ticker": "NVDA", "author": "a", "stance": None, "car_event": 0.02, "z_event": 1.1}
    assert charts.top_event_chart(event, prices, tmp_path / "u.png") is not None

    ticker_line = next(line for line in drawn[-1].axes[0].lines if line.get_linewidth() == charts.LINE_W)
    assert ticker_line.get_color() == charts.UNLABELLED
    # The card's stance dot uses the same color as the chart.
    assert f"background:{charts.UNLABELLED}" in report._dot(None)
