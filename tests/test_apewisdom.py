from __future__ import annotations

import json
from datetime import UTC, datetime

import httpx
import pytest

from influence_tracker.collectors.apewisdom import collect_apewisdom, parse_page, snapshot_date_for, to_int
from influence_tracker.config import ApeWisdomConfig, Watchlist
from influence_tracker.timeutil import utc_now


class Sleeps(list):
    def __call__(self, seconds: float) -> None:
        self.append(seconds)


class FakeApeWisdom:
    """Serves /filter/<f>/page/<n>: a callable per filter builds the response for page n."""

    def __init__(self, pages_by_filter: dict) -> None:
        self.pages_by_filter = pages_by_filter
        self.requests: list[tuple[str, int]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        parts = request.url.path.split("/")  # ['', 'api', 'v1.0', 'filter', f, 'page', n]
        filter_name, page = parts[4], int(parts[6])
        self.requests.append((filter_name, page))
        return self.pages_by_filter[filter_name](page)

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self))


def synthetic_page(prefix: str, page: int, pages: int | str | None, n: int = 3) -> httpx.Response:
    results = [
        {
            "rank": (page - 1) * n + i + 1,
            "ticker": f"{prefix}{page}X{i}",
            "name": f"Name {i}",
            "mentions": 10,
            "upvotes": 20,
            "rank_24h_ago": 5,
            "mentions_24h_ago": 8,
        }
        for i in range(n)
    ]
    return httpx.Response(200, json={"count": 999, "pages": pages, "current_page": page, "results": results})


def with_apewisdom(watchlist: Watchlist, filters: list[str], pages: int) -> Watchlist:
    return watchlist.model_copy(update={"apewisdom": ApeWisdomConfig(filters=filters, pages=pages)})


@pytest.fixture
def fixture_payload(fixtures_dir) -> dict:
    with open(fixtures_dir / "apewisdom_all-stocks_p1.json", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------- parsing


def test_parse_recorded_page(fixture_payload, fixed_now):
    rows, pages = parse_page(fixture_payload, "all-stocks", "2026-09-24", fixed_now)
    assert pages == 10
    assert len(rows) == 100
    by_ticker = {r["ticker"]: r for r in rows}
    assert by_ticker["SPY"] == {
        "snapshot_date": "2026-09-24",
        "filter": "all-stocks",
        "ticker": "SPY",
        "name": "SPDR S&P 500 ETF Trust",
        "rank": 1,
        "mentions": 232,
        "upvotes": 655,
        "rank_24h_ago": 1,
        "mentions_24h_ago": 265,
        "fetched_at": "2026-09-24T22:30:00Z",
    }
    assert by_ticker["DJT"]["name"] == "Trump Media & Technology Group"
    assert by_ticker["WTI"]["name"] == "W&T Offshore"
    assert by_ticker["WHLR"]["mentions_24h_ago"] is None
    assert by_ticker["WHLR"]["rank_24h_ago"] == 0
    assert all("&amp;" not in (r["name"] or "") for r in rows)


def test_parse_string_and_null_numbers(fixed_now):
    payload = {
        "pages": "2",
        "results": [
            {
                "rank": "3",
                "ticker": " gme ",
                "name": "GameStop",
                "mentions": "12",
                "upvotes": None,
                "rank_24h_ago": "",
                "mentions_24h_ago": "7.0",
            },
            {"rank": 4, "ticker": "", "name": "no ticker"},
            {"rank": 5, "ticker": "AMC", "name": None, "mentions": "n/a"},
        ],
    }
    rows, pages = parse_page(payload, "wallstreetbets", "2026-09-24", fixed_now)
    assert pages == 2
    assert [r["ticker"] for r in rows] == ["GME", "AMC"]
    gme, amc = rows
    assert (gme["rank"], gme["mentions"], gme["upvotes"], gme["rank_24h_ago"], gme["mentions_24h_ago"]) == (
        3,
        12,
        None,
        None,
        7,
    )
    assert amc["name"] is None
    assert amc["mentions"] is None
    assert amc["upvotes"] is None


@pytest.mark.parametrize(
    ("value", "expected"),
    [(5, 5), ("5", 5), (" 1,204 ", 1204), ("7.0", 7), (7.9, 7), (None, None), ("", None), ("x", None), (True, None)],
)
def test_to_int(value, expected):
    assert to_int(value) == expected


@pytest.mark.parametrize("payload", [[], {"results": None}, {"pages": 1}, "oops"])
def test_parse_rejects_unexpected_shapes(payload, fixed_now):
    with pytest.raises(ValueError):
        parse_page(payload, "all-stocks", "2026-09-24", fixed_now)


def test_snapshot_date_is_new_york_date():
    assert snapshot_date_for(datetime(2026, 9, 24, 22, 30, tzinfo=UTC)) == "2026-09-24"
    # 02:00 UTC on the 25th is still the evening of the 24th in New York
    assert snapshot_date_for(datetime(2026, 9, 25, 2, 0, tzinfo=UTC)) == "2026-09-24"
    assert snapshot_date_for(datetime(2026, 9, 25, 4, 30, tzinfo=UTC)) == "2026-09-25"


# ---------------------------------------------------------------- collecting


def test_collect_recorded_page_into_db(conn, watchlist, fixed_now, fixture_payload):
    fake = FakeApeWisdom({"all-stocks": lambda page: httpx.Response(200, json=fixture_payload)})
    sleeps = Sleeps()
    counts = collect_apewisdom(
        conn, with_apewisdom(watchlist, ["all-stocks"], 1), 7, fixed_now, client=fake.client(), sleep=sleeps
    )
    assert counts == {"status": "ok", "requests": 1, "rows": 100, "filters_failed": 0}
    assert sleeps == []
    row = conn.execute("SELECT * FROM reddit_ticker_daily WHERE filter = 'all-stocks' AND ticker = 'SPY'").fetchone()
    assert row["snapshot_date"] == "2026-09-24"
    assert row["name"] == "SPDR S&P 500 ETF Trust"
    assert row["mentions"] == 232
    assert row["fetched_at"] == "2026-09-24T22:30:00Z"
    whlr = conn.execute("SELECT mentions_24h_ago FROM reddit_ticker_daily WHERE ticker = 'WHLR'").fetchone()
    assert whlr["mentions_24h_ago"] is None


def test_request_shape(conn, watchlist, fixed_now):
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return synthetic_page("A", 1, 1)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    collect_apewisdom(conn, with_apewisdom(watchlist, ["all-stocks"], 3), 1, fixed_now, client=client, sleep=Sleeps())
    [request] = seen
    assert str(request.url) == "https://apewisdom.io/api/v1.0/filter/all-stocks/page/1"
    assert request.headers["user-agent"]


def test_pages_capped_by_config(conn, watchlist, fixed_now):
    fake = FakeApeWisdom(
        {
            "all-stocks": lambda page: synthetic_page("A", page, 10),
            "wallstreetbets": lambda page: synthetic_page("W", page, 10),
        }
    )
    sleeps = Sleeps()
    counts = collect_apewisdom(
        conn,
        with_apewisdom(watchlist, ["all-stocks", "wallstreetbets"], 3),
        1,
        fixed_now,
        client=fake.client(),
        sleep=sleeps,
    )
    assert fake.requests == [
        ("all-stocks", 1),
        ("all-stocks", 2),
        ("all-stocks", 3),
        ("wallstreetbets", 1),
        ("wallstreetbets", 2),
        ("wallstreetbets", 3),
    ]
    assert sleeps == [1.0] * 5
    assert counts == {"status": "ok", "requests": 6, "rows": 18, "filters_failed": 0}
    assert conn.execute("SELECT COUNT(*) FROM reddit_ticker_daily").fetchone()[0] == 18


def test_pages_capped_by_response(conn, watchlist, fixed_now):
    fake = FakeApeWisdom({"all-stocks": lambda page: synthetic_page("A", page, "2")})
    counts = collect_apewisdom(
        conn, with_apewisdom(watchlist, ["all-stocks"], 3), 1, fixed_now, client=fake.client(), sleep=Sleeps()
    )
    assert fake.requests == [("all-stocks", 1), ("all-stocks", 2)]
    assert counts["rows"] == 6


def test_empty_page_stops_paging_when_pages_is_unknown(conn, watchlist, fixed_now):
    fake = FakeApeWisdom({"all-stocks": lambda page: synthetic_page("A", page, None, n=3 if page == 1 else 0)})
    counts = collect_apewisdom(
        conn, with_apewisdom(watchlist, ["all-stocks"], 3), 1, fixed_now, client=fake.client(), sleep=Sleeps()
    )
    assert fake.requests == [("all-stocks", 1), ("all-stocks", 2)]
    assert counts == {"status": "ok", "requests": 2, "rows": 3, "filters_failed": 0}


def test_duplicate_ticker_across_pages_keeps_first_rank(conn, watchlist, fixed_now):
    def page_fn(page: int) -> httpx.Response:
        results = [{"rank": page, "ticker": "GME", "name": "GameStop", "mentions": 100 - page}]
        return httpx.Response(200, json={"pages": 2, "results": results})

    fake = FakeApeWisdom({"all-stocks": page_fn})
    counts = collect_apewisdom(
        conn, with_apewisdom(watchlist, ["all-stocks"], 3), 1, fixed_now, client=fake.client(), sleep=Sleeps()
    )
    assert counts["rows"] == 1
    row = conn.execute("SELECT rank, mentions FROM reddit_ticker_daily WHERE ticker = 'GME'").fetchone()
    assert (row["rank"], row["mentions"]) == (1, 99)


def test_failed_filter_is_partial_and_others_still_stored(conn, watchlist, fixed_now):
    fake = FakeApeWisdom(
        {
            "all-stocks": lambda page: httpx.Response(503, text="upstream down"),
            "wallstreetbets": lambda page: synthetic_page("W", page, 1),
        }
    )
    sleeps = Sleeps()
    counts = collect_apewisdom(
        conn,
        with_apewisdom(watchlist, ["all-stocks", "wallstreetbets"], 3),
        1,
        fixed_now,
        client=fake.client(),
        sleep=sleeps,
    )
    assert counts == {"status": "partial", "requests": 2, "rows": 3, "filters_failed": 1}
    assert sleeps == [1.0]


def test_bad_json_on_page_two_keeps_page_one(conn, watchlist, fixed_now):
    def page_fn(page: int) -> httpx.Response:
        if page == 1:
            return synthetic_page("A", 1, 3)
        return httpx.Response(200, text="<html>not json</html>")

    fake = FakeApeWisdom({"all-stocks": page_fn})
    counts = collect_apewisdom(
        conn, with_apewisdom(watchlist, ["all-stocks"], 3), 1, fixed_now, client=fake.client(), sleep=Sleeps()
    )
    assert fake.requests == [("all-stocks", 1), ("all-stocks", 2)]
    # the filter failed, but its first page is real data, so the run is partial rather than an error
    assert counts == {"status": "partial", "requests": 2, "rows": 3, "filters_failed": 1}
    assert conn.execute("SELECT COUNT(*) FROM reddit_ticker_daily").fetchone()[0] == 3


def test_transport_error_does_not_raise(conn, watchlist, fixed_now):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    wl = with_apewisdom(watchlist, ["all-stocks", "wallstreetbets"], 3)
    counts = collect_apewisdom(conn, wl, 1, fixed_now, client=client, sleep=Sleeps())
    assert counts == {"status": "error", "requests": 2, "rows": 0, "filters_failed": 2}


def test_same_day_rerun_upserts(conn, watchlist, fixed_now, fixture_payload):
    wl = with_apewisdom(watchlist, ["all-stocks"], 1)
    for _ in range(2):
        fake = FakeApeWisdom({"all-stocks": lambda page: httpx.Response(200, json=fixture_payload)})
        collect_apewisdom(conn, wl, 1, fixed_now, client=fake.client(), sleep=Sleeps())
    assert conn.execute("SELECT COUNT(*) FROM reddit_ticker_daily").fetchone()[0] == 100


@pytest.mark.live
def test_live_first_page(conn, watchlist):
    counts = collect_apewisdom(conn, with_apewisdom(watchlist, ["all-stocks"], 1), 0, utc_now())
    assert counts["status"] == "ok", counts
    assert counts["rows"] > 0
    row = conn.execute("SELECT * FROM reddit_ticker_daily ORDER BY rank LIMIT 1").fetchone()
    assert row["ticker"] and row["rank"] is not None and row["mentions"] is not None
