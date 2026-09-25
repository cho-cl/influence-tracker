from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import httpx
import pytest

from influence_tracker import db
from influence_tracker.collectors import truthsocial as ts
from influence_tracker.collectors.truthsocial import (
    backfill_truthsocial,
    collect_truthsocial,
    html_to_text,
    id_for_time,
    status_to_post,
    time_for_id,
)
from influence_tracker.config import Watchlist
from influence_tracker.ingest import PostSink, sync_accounts
from influence_tracker.models import MatchResult, Mention, Post
from influence_tracker.timeutil import parse_api_time

FIXTURES = Path(__file__).parent / "fixtures"
TRUMP = "realDonaldTrump"
TRUMP_ID = "107780257626128497"
VANCE = "JDVance1"
VANCE_ID = "900000000000000001"  # made up; only the fake server knows it


def _load(name: str):
    with open(FIXTURES / name, encoding="utf-8") as f:
        return json.load(f)


# Real pages recorded live on 2026-09-24, newest-first as served.
NEWEST = _load("truthsocial_statuses_newest.json")
OLDER = _load("truthsocial_statuses_older.json")
JAN_2025 = _load("truthsocial_statuses_2025-01-20.json")
ALL_RECORDED = NEWEST + OLDER + JAN_2025
# Recorded statuses with no text of their own: media-only ('<p></p>') and one quote post of a video.
NO_OWN_TEXT = {
    "117326000312372207",
    "117325992695243662",
    "117325988723052373",
    "117325987378356876",
    "117325694783376982",
    "117323511508375196",
    "117322092055919832",
    "117318365806497574",
    "113860871478605894",
    "113857377306346900",
}


def by_id(status_id: str) -> dict:
    return next(s for s in ALL_RECORDED if s["id"] == status_id)


def storable_ids(statuses: list[dict]) -> set[str]:
    return {s["id"] for s in statuses if s["reblog"] is None and s["id"] not in NO_OWN_TEXT}


def synthetic_status(handle: str, account_id: str, when: datetime, text: str, reblog: dict | None = None) -> dict:
    sid = str(int(id_for_time(when)) + 7)
    return {
        "id": sid,
        "created_at": when.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        "url": f"https://truthsocial.com/@{handle}/{sid}",
        "content": f"<p>{text}</p>",
        "account": {"id": account_id, "username": handle, "acct": handle},
        "reblog": reblog,
        "quote_id": None,
        "quote": None,
        "media_attachments": [],
        "replies_count": 1,
        "reblogs_count": 2,
        "favourites_count": 3,
    }


# ---------------------------------------------------------------- fakes


class FakeTime:
    """Monotonic clock that only moves when the collector sleeps (or a test advances it)."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class FakeTruthSocial:
    """Serves /accounts/lookup and max_id-paged /accounts/{id}/statuses newest-first, ignoring min_id and
    exclude_reblogs the way the live server did. `override(index, request)` may replace any response."""

    def __init__(self, clock: FakeTime, page_size: int = 20) -> None:
        self.clock = clock
        self.page_size = page_size
        self.accounts: dict[str, str] = {}
        self.timelines: dict[str, list[dict]] = {}
        self.requests: list[httpx.Request] = []
        self.request_times: list[float] = []
        self.override: Callable[[int, httpx.Request], httpx.Response | None] | None = None
        self.on_request: Callable[[httpx.Request], None] | None = None

    def add_account(self, handle: str, account_id: str, statuses: list[dict]) -> None:
        self.accounts[handle] = account_id
        self.timelines[account_id] = list(statuses)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        index = len(self.requests)
        self.requests.append(request)
        self.request_times.append(self.clock.now)
        if self.on_request is not None:
            self.on_request(request)
        if self.override is not None:
            replaced = self.override(index, request)
            if replaced is not None:
                return replaced
        path = request.url.path
        if path == "/api/v1/accounts/lookup":
            handle = request.url.params["acct"]
            if handle not in self.accounts:
                return httpx.Response(404, json=_load("truthsocial_lookup_not_found.json"))
            return httpx.Response(200, json={"id": self.accounts[handle], "username": handle, "acct": handle})
        account_id = path.removeprefix("/api/v1/accounts/").removesuffix("/statuses")
        statuses = sorted(self.timelines[account_id], key=lambda s: int(s["id"]), reverse=True)
        if "max_id" in request.url.params:
            statuses = [s for s in statuses if int(s["id"]) < int(request.url.params["max_id"])]
        return httpx.Response(200, json=statuses[: self.page_size])

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self))

    def statuses_requests(self) -> list[httpx.Request]:
        return [r for r in self.requests if r.url.path.endswith("/statuses")]


class StubMatcher:
    def match(self, text: str, platform: str, cashtag_hints=()) -> MatchResult:
        return MatchResult([Mention("BA", "name", "Boeing")] if "BOEING" in text.upper() else [])


class RecordingSink(PostSink):
    def __init__(self, conn, now: datetime) -> None:
        super().__init__(conn, StubMatcher(), now=lambda: now)
        self.calls: list[list[str]] = []

    def store(self, posts):
        self.calls.append([p.native_id for p in posts])
        return super().store(posts)


def rate_limited_429() -> httpx.Response:
    return httpx.Response(429, json={"error": "Too many requests"})


def cloudflare_1015() -> httpx.Response:
    return httpx.Response(403, text="error code: 1015", headers={"content-type": "text/plain; charset=UTF-8"})


def challenge_403() -> httpx.Response:
    body = "<!DOCTYPE html><html><head><title>Just a moment...</title></head><body>challenge</body></html>"
    return httpx.Response(403, text=body, headers={"content-type": "text/html; charset=UTF-8"})


def ts_watchlist(watchlist: Watchlist, *handles: str, **cfg) -> Watchlist:
    accounts = [a for a in watchlist.truthsocial.accounts if a.handle in handles]
    assert len(accounts) == len(handles)
    truthsocial = watchlist.truthsocial.model_copy(update={"accounts": accounts, **cfg})
    return watchlist.model_copy(update={"truthsocial": truthsocial})


@pytest.fixture
def clock() -> FakeTime:
    return FakeTime()


@pytest.fixture
def server(clock: FakeTime) -> FakeTruthSocial:
    return FakeTruthSocial(clock)


@pytest.fixture
def trump_only(watchlist: Watchlist, conn) -> Watchlist:
    wl = ts_watchlist(watchlist, TRUMP)
    sync_accounts(conn, wl)
    return wl


def collect(conn, wl, sink, server, clock, now) -> dict:
    with server.client() as client:
        return collect_truthsocial(conn, wl, sink, 1, now, client=client, sleep=clock.sleep, clock=clock.clock)


def backfill(conn, wl, sink, server, clock, since: date, now) -> dict:
    with server.client() as client:
        return backfill_truthsocial(conn, wl, sink, 1, since, now, client=client, sleep=clock.sleep, clock=clock.clock)


def stored_ids(conn) -> set[str]:
    return {r["native_id"] for r in conn.execute("SELECT native_id FROM posts WHERE platform = 'truthsocial'")}


def watermark(conn, key: str, source: str = "truthsocial") -> str | None:
    return db.get_watermark(conn, source, key)


# ---------------------------------------------------------------- content


def test_html_to_text_keeps_link_text_paragraphs_and_entities():
    # Mastodon splits long URLs over 'invisible'/'ellipsis' spans; together they are the full URL.
    assert html_to_text(by_id("117326044871082645")["content"]) == (
        "Midterm Fact Check: Don’t believe the economic doomers: https://www.foxnews.com/video/6405515361112"
    )
    assert html_to_text(by_id("117312693997244079")["content"]) == (
        "Thanks Johnny. You are a total winner! President DJT\n\n"
        "SPORTS MLB Legend Johnny Damon Refuses To Apologize For His Opinion on President Trump: "
        "https://www.miamiherald.com/sports/article317314631.html"
    )
    assert html_to_text("<p>AT&amp;T &#39;up&#39; &quot;big&quot;<br>next line</p>") == "AT&T 'up' \"big\"\nnext line"


def test_html_to_text_collapses_non_breaking_spaces_and_blank_lines():
    text = html_to_text(by_id("113855616848696050")["content"])
    assert "\xa0" not in text
    assert "before my order.\n\nAmericans deserve" in text
    assert "a joint venture. By doing this" in text
    assert "\n\n\n" not in text
    assert text == text.strip()


def test_quote_post_keeps_only_the_authors_own_words():
    status = by_id("117326047233268228")
    assert status["quote_id"] == "117326044871082645"
    text = html_to_text(status["content"])
    assert text.startswith("This is a survey of people in business — the S&P Flash Survey — of purchasing")
    assert "RT:" not in text
    assert "statuses/117326044871082645" not in text
    assert "Midterm Fact Check" not in text  # the quoted post's own words
    assert "62 month high!\n\nManufacturing — 53 month high!" in text
    assert text.endswith("https://www.pmi.spglobal.com/Public/Home/PressRelease/ed177f50167b4203ac490a961ea706be")


def test_statuses_without_own_text_have_empty_text():
    for sid in NO_OWN_TEXT:
        assert html_to_text(by_id(sid)["content"]) == "", sid


def test_reblogs_and_textless_posts_are_skipped():
    reblogs = [s for s in ALL_RECORDED if s["reblog"] is not None]
    assert reblogs, "fixtures must include reblogs"
    for s in ALL_RECORDED:
        post = status_to_post(s, TRUMP, TRUMP_ID)
        if s["reblog"] is not None or s["id"] in NO_OWN_TEXT:
            assert post is None, s["id"]
        else:
            assert post is not None and post.text, s["id"]


def test_status_to_post_fields():
    s = by_id("117323148147220269")
    post = status_to_post(s, TRUMP, TRUMP_ID)
    assert post == Post(
        platform="truthsocial",
        native_id="117323148147220269",
        author=TRUMP,
        author_id=TRUMP_ID,
        created_at_utc=datetime(2026, 9, 24, 0, 19, 29, 727000, tzinfo=UTC),
        text=html_to_text(s["content"]),
        url="https://truthsocial.com/@realDonaldTrump/117323148147220269",
        metrics={
            "replies_count": s["replies_count"],
            "reblogs_count": s["reblogs_count"],
            "favourites_count": s["favourites_count"],
        },
    )
    assert post.text.startswith("BIG DAY FOR BOEING AND AMERICAN MANUFACTURING!")


def test_status_to_post_url_fallback_and_foreign_account():
    s = by_id("117323148147220269")
    assert status_to_post({**s, "url": None}, TRUMP, TRUMP_ID).url == (
        "https://truthsocial.com/@realDonaldTrump/117323148147220269"
    )
    assert status_to_post({**s, "account": {"id": "1"}}, TRUMP, TRUMP_ID) is None
    assert status_to_post({**s, "created_at": "not a time"}, TRUMP, TRUMP_ID) is None


# ---------------------------------------------------------------- snowflake ids


def test_snowflake_ids_match_created_at_on_real_statuses():
    statuses = ALL_RECORDED + [s["reblog"] for s in ALL_RECORDED if s["reblog"]]
    for s in statuses:
        minted = time_for_id(s["id"])
        assert minted.tzinfo is not None
        # Live ids carry a timestamp a few ms off created_at (-1..+6 ms seen in the probe).
        assert abs(minted - parse_api_time(s["created_at"])) <= timedelta(milliseconds=10), s["id"]
        low = int(id_for_time(minted))
        assert low <= int(s["id"]) < int(id_for_time(minted + timedelta(milliseconds=1)))


def test_id_for_time_is_exact_and_rejects_naive_datetimes():
    # The cursor the live probe synthesized for 2025-01-21 00:00 UTC and got January 20 posts back for.
    assert id_for_time(datetime(2025, 1, 21, tzinfo=UTC)) == "113863399833600000"
    assert time_for_id("113863399833600000") == datetime(2025, 1, 21, tzinfo=UTC)
    with pytest.raises(ValueError):
        id_for_time(datetime(2025, 1, 21))


# ---------------------------------------------------------------- daily paging


def test_first_run_walks_back_to_the_window_stores_each_page_oldest_first(conn, trump_only, server, clock, fixed_now):
    server.add_account(TRUMP, TRUMP_ID, NEWEST + OLDER)
    sink = RecordingSink(conn, fixed_now)

    counts = collect(conn, trump_only, sink, server, clock, fixed_now)

    assert counts["status"] == "ok"
    assert stored_ids(conn) == storable_ids(NEWEST + OLDER)
    assert counts["stored"] == counts["new"] == len(storable_ids(NEWEST + OLDER))
    assert counts["with_mentions"] == 1  # the Boeing post
    for call in sink.calls:
        assert call == sorted(call, key=int)
    # Watermark = newest id seen, even though later sweeps start from the newest page again.
    assert watermark(conn, TRUMP) == max((s["id"] for s in NEWEST + OLDER), key=int) == "117326213046216794"
    assert watermark(conn, f"first_min_id:{TRUMP}") == id_for_time(fixed_now - timedelta(days=30))
    assert not watermark(conn, f"sweep:{TRUMP}")
    # lookup, newest page, older page, then an empty page ends the walk.
    assert counts["requests"] == 4 and counts["pages"] == 3
    assert counts["accounts_ok"] == 1 and counts["accounts_failed"] == 0 and counts["unresolved_handles"] == []
    row = db.get_accounts(conn, "truthsocial")[0]
    assert row["platform_user_id"] == TRUMP_ID and row["resolve_error"] is None
    assert row["last_fetch_at"] == "2026-09-24T22:30:00Z"


def test_requests_carry_browser_headers_and_never_rely_on_min_id(conn, trump_only, server, clock, fixed_now):
    server.add_account(TRUMP, TRUMP_ID, NEWEST)
    collect(conn, trump_only, RecordingSink(conn, fixed_now), server, clock, fixed_now)

    lookup = server.requests[0]
    assert str(lookup.url) == "https://truthsocial.com/api/v1/accounts/lookup?acct=realDonaldTrump"
    for request in server.requests:
        assert request.headers["User-Agent"] == trump_only.truthsocial.user_agent
        assert request.headers["Accept"] == "application/json"
    for request in server.statuses_requests():
        assert request.url.path == f"/api/v1/accounts/{TRUMP_ID}/statuses"
        assert request.url.params["limit"] == "20"
        assert request.url.params["exclude_reblogs"] == "true"
        assert "min_id" not in request.url.params


def test_first_run_ignores_statuses_older_than_the_window(conn, trump_only, server, clock, fixed_now):
    server.add_account(TRUMP, TRUMP_ID, NEWEST + OLDER + JAN_2025)
    counts = collect(conn, trump_only, RecordingSink(conn, fixed_now), server, clock, fixed_now)

    assert counts["status"] == "ok"
    assert stored_ids(conn) == storable_ids(NEWEST + OLDER)
    # The second page (19 older statuses + the newest January 2025 one) reaches below the 30-day window,
    # which ends the walk without asking for an empty page.
    assert counts["pages"] == 2 and counts["requests"] == 3
    assert watermark(conn, TRUMP) == "117326213046216794"


def test_sweep_progress_is_saved_page_by_page(conn, trump_only, clock, fixed_now):
    server = FakeTruthSocial(clock, page_size=5)
    server.add_account(TRUMP, TRUMP_ID, NEWEST + OLDER)
    seen: list[tuple[str | None, str | None]] = []

    def snapshot(request: httpx.Request) -> None:
        if request.url.path.endswith("/statuses"):
            seen.append((request.url.params.get("max_id"), watermark(conn, f"sweep:{TRUMP}")))

    server.on_request = snapshot
    counts = collect(conn, trump_only, RecordingSink(conn, fixed_now), server, clock, fixed_now)

    ordered = sorted((s["id"] for s in NEWEST + OLDER), key=int, reverse=True)
    page_mins = [ordered[i : i + 5][-1] for i in range(0, len(ordered), 5)]
    assert counts["pages"] == len(page_mins) + 1
    assert seen[0] == (None, None)
    for (max_id, saved), page_min in zip(seen[1:], page_mins, strict=True):
        # Each request resumes exactly where the previous page's saved progress points.
        assert max_id == page_min
        assert saved == f"{ordered[0]}:{page_min}"
    assert watermark(conn, TRUMP) == ordered[0]
    assert stored_ids(conn) == storable_ids(NEWEST + OLDER)


def test_page_cap_leaves_a_resumable_sweep(conn, trump_only, clock, fixed_now, monkeypatch):
    server = FakeTruthSocial(clock, page_size=5)
    server.add_account(TRUMP, TRUMP_ID, NEWEST + OLDER)
    ordered = sorted((s["id"] for s in NEWEST + OLDER), key=int, reverse=True)
    monkeypatch.setattr(ts, "MAX_PAGES_PER_ACCOUNT", 3)

    first = collect(conn, trump_only, RecordingSink(conn, fixed_now), server, clock, fixed_now)

    assert first["status"] == "partial"
    assert first["pages"] == 3 and first["accounts_ok"] == 1
    assert watermark(conn, TRUMP) is None
    assert watermark(conn, f"sweep:{TRUMP}") == f"{ordered[0]}:{ordered[14]}"
    assert stored_ids(conn) == storable_ids([by_id(sid) for sid in ordered[:15]])

    # A post arrives before the next run; the resumed sweep finishes, then a fresh one picks the new post up.
    later = fixed_now + timedelta(days=1)
    fresh = synthetic_status(TRUMP, TRUMP_ID, later - timedelta(hours=1), "Buy BOEING")
    server.timelines[TRUMP_ID].append(fresh)
    monkeypatch.setattr(ts, "MAX_PAGES_PER_ACCOUNT", 50)
    server.requests.clear()

    second = collect(conn, trump_only, RecordingSink(conn, later), server, clock, later)

    assert second["status"] == "ok"
    assert server.requests[0].url.params["max_id"] == ordered[14]
    assert "max_id" not in server.requests[-1].url.params
    assert watermark(conn, TRUMP) == fresh["id"]
    assert not watermark(conn, f"sweep:{TRUMP}")
    assert stored_ids(conn) == storable_ids(NEWEST + OLDER) | {fresh["id"]}
    assert watermark(conn, f"first_min_id:{TRUMP}") == id_for_time(fixed_now - timedelta(days=30))


def test_next_run_walks_back_only_to_the_watermark(conn, trump_only, clock, fixed_now):
    server = FakeTruthSocial(clock, page_size=5)
    server.add_account(TRUMP, TRUMP_ID, OLDER)
    collect(conn, trump_only, RecordingSink(conn, fixed_now), server, clock, fixed_now)
    assert watermark(conn, TRUMP) == OLDER[0]["id"]

    server.timelines[TRUMP_ID] = NEWEST + OLDER
    server.requests.clear()
    sink = RecordingSink(conn, fixed_now)
    counts = collect(conn, trump_only, sink, server, clock, fixed_now)

    assert counts["status"] == "ok"
    # Four pages of the 20 new statuses, then one that reaches the watermark; nothing older is re-stored.
    assert counts["pages"] == 5 and counts["new"] == counts["stored"] == len(storable_ids(NEWEST))
    assert sink.calls[-1] == []
    assert server.requests[-1].url.params["max_id"] == NEWEST[-1]["id"]
    assert watermark(conn, TRUMP) == NEWEST[0]["id"]
    assert stored_ids(conn) == storable_ids(NEWEST + OLDER)


def test_watermark_advances_past_pages_with_nothing_to_store(conn, trump_only, server, clock, fixed_now):
    skipped = [s for s in NEWEST if s["reblog"] is not None or s["id"] in NO_OWN_TEXT]
    server.add_account(TRUMP, TRUMP_ID, skipped)
    counts = collect(conn, trump_only, RecordingSink(conn, fixed_now), server, clock, fixed_now)

    assert counts["status"] == "ok" and counts["stored"] == 0
    assert watermark(conn, TRUMP) == max((s["id"] for s in skipped), key=int)


def test_a_run_with_nothing_new_costs_one_request(conn, trump_only, server, clock, fixed_now):
    server.add_account(TRUMP, TRUMP_ID, NEWEST + OLDER)
    collect(conn, trump_only, RecordingSink(conn, fixed_now), server, clock, fixed_now)
    server.requests.clear()
    sink = RecordingSink(conn, fixed_now)

    counts = collect(conn, trump_only, sink, server, clock, fixed_now + timedelta(days=1))

    assert counts["status"] == "ok"
    assert counts["requests"] == 1 and counts["stored"] == 0
    assert sink.calls == [[]]
    assert watermark(conn, TRUMP) == "117326213046216794"


# ---------------------------------------------------------------- pacing and rate limits


def test_min_interval_is_enforced_between_every_request(conn, watchlist, server, clock, fixed_now):
    wl = ts_watchlist(watchlist, TRUMP, VANCE, min_request_interval_s=7.5)
    sync_accounts(conn, wl)
    server.add_account(TRUMP, TRUMP_ID, NEWEST + OLDER)
    server.add_account(VANCE, VANCE_ID, [synthetic_status(VANCE, VANCE_ID, fixed_now - timedelta(days=2), "Hi")])

    def slow_response(index: int, request: httpx.Request) -> None:
        clock.now += 2.0

    class SlowSink(RecordingSink):
        def store(self, posts):
            clock.now += 1.0
            return super().store(posts)

    server.override = slow_response
    counts = collect(conn, wl, SlowSink(conn, fixed_now), server, clock, fixed_now)

    assert counts["status"] == "ok"
    # JDVance1 (lookup, 1 status, empty page), then realDonaldTrump (lookup, 20, 19, empty page).
    assert counts["requests"] == len(server.requests) == 7
    # The interval runs from the end of one response to the start of the next request; time spent
    # storing a page counts toward it, the response's own 2 s do not.
    quiet = [b - (a + 2.0) for a, b in zip(server.request_times, server.request_times[1:], strict=False)]
    assert quiet == [7.5] * 6
    assert clock.sleeps == [7.5, 6.5, 6.5, 7.5, 6.5, 6.5]


def test_429_backs_off_60_then_120_then_gives_up_without_losing_progress(conn, trump_only, clock, fixed_now):
    server = FakeTruthSocial(clock, page_size=5)
    server.add_account(TRUMP, TRUMP_ID, NEWEST + OLDER)
    ordered = sorted((s["id"] for s in NEWEST + OLDER), key=int, reverse=True)
    server.override = lambda i, request: rate_limited_429() if i >= 3 else None

    counts = collect(conn, trump_only, RecordingSink(conn, fixed_now), server, clock, fixed_now)

    assert clock.sleeps == [5.0, 5.0, 5.0, 60.0, 120.0]
    assert counts["requests"] == 6 and len(server.requests) == 6
    assert counts["rate_limited"] == 1
    assert counts["status"] == "partial"
    assert counts["accounts_failed"] == 1 and counts["accounts_ok"] == 0
    # Two pages were stored and their progress kept; the daily watermark waits for the sweep to finish.
    assert stored_ids(conn) == storable_ids([by_id(sid) for sid in ordered[:10]])
    assert watermark(conn, f"sweep:{TRUMP}") == f"{ordered[0]}:{ordered[9]}"
    assert watermark(conn, TRUMP) is None
    assert db.get_accounts(conn, "truthsocial")[0]["last_fetch_at"] is None

    server.override = None
    server.requests.clear()
    again = collect(conn, trump_only, RecordingSink(conn, fixed_now), server, clock, fixed_now)

    assert again["status"] == "ok"
    assert server.requests[0].url.params["max_id"] == ordered[9]
    assert stored_ids(conn) == storable_ids(NEWEST + OLDER)
    assert watermark(conn, TRUMP) == ordered[0]


def test_one_429_then_recovery_finishes_the_run(conn, trump_only, server, clock, fixed_now):
    server.add_account(TRUMP, TRUMP_ID, NEWEST + OLDER)
    server.override = lambda i, request: rate_limited_429() if i == 1 else None

    counts = collect(conn, trump_only, RecordingSink(conn, fixed_now), server, clock, fixed_now)

    assert counts["status"] == "ok" and counts["rate_limited"] == 0
    assert clock.sleeps[:2] == [5.0, 60.0]
    assert stored_ids(conn) == storable_ids(NEWEST + OLDER)


def test_cloudflare_1015_body_is_treated_like_a_429(conn, trump_only, server, clock, fixed_now):
    server.add_account(TRUMP, TRUMP_ID, NEWEST)
    server.override = lambda i, request: cloudflare_1015()

    counts = collect(conn, trump_only, RecordingSink(conn, fixed_now), server, clock, fixed_now)

    assert clock.sleeps == [60.0, 120.0]
    assert counts["requests"] == 3 and counts["rate_limited"] == 1 and counts["status"] == "partial"
    row = db.get_accounts(conn, "truthsocial")[0]
    assert row["platform_user_id"] is None and row["resolve_error"] is None


def test_a_post_quoting_error_1015_is_not_a_rate_limit(conn, trump_only, server, clock, fixed_now):
    joke = synthetic_status(TRUMP, TRUMP_ID, fixed_now - timedelta(hours=3), "They sent me error code: 1015!")
    server.add_account(TRUMP, TRUMP_ID, [joke])
    counts = collect(conn, trump_only, RecordingSink(conn, fixed_now), server, clock, fixed_now)
    assert counts["status"] == "ok" and counts["rate_limited"] == 0
    assert stored_ids(conn) == {joke["id"]}


def test_403_challenge_page_stops_the_whole_run(conn, watchlist, server, clock, fixed_now, caplog):
    wl = ts_watchlist(watchlist, TRUMP, VANCE)
    sync_accounts(conn, wl)
    server.add_account(TRUMP, TRUMP_ID, NEWEST)
    server.add_account(VANCE, VANCE_ID, [])
    server.override = lambda i, request: challenge_403() if i == 1 else None

    with caplog.at_level(logging.WARNING, logger=ts.__name__):
        counts = collect(conn, wl, RecordingSink(conn, fixed_now), server, clock, fixed_now)

    assert len(server.requests) == 2  # JDVance1's lookup, then its statuses page hit the challenge
    assert counts["status"] == "partial"
    assert counts["accounts_failed"] == 1 and counts["accounts_ok"] == 0
    assert counts["rate_limited"] == 0
    assert 60.0 not in clock.sleeps
    assert any("challenge" in r.getMessage() and r.levelno == logging.ERROR for r in caplog.records)
    assert stored_ids(conn) == set()


def test_cloudflare_access_denied_on_lookup_stops_the_run_without_blaming_the_handle(
    conn, watchlist, server, clock, fixed_now
):
    wl = ts_watchlist(watchlist, TRUMP, VANCE)
    sync_accounts(conn, wl)
    server.add_account(TRUMP, TRUMP_ID, NEWEST)
    server.add_account(VANCE, VANCE_ID, [])
    denied = httpx.Response(403, text="error code: 1020", headers={"content-type": "text/plain; charset=UTF-8"})
    server.override = lambda i, request: denied

    counts = collect(conn, wl, RecordingSink(conn, fixed_now), server, clock, fixed_now)

    assert len(server.requests) == 1
    assert counts["status"] == "partial" and counts["rate_limited"] == 0
    assert counts["unresolved_handles"] == []
    for row in db.get_accounts(conn, "truthsocial"):
        assert row["platform_user_id"] is None and row["resolve_error"] is None


# ---------------------------------------------------------------- account resolution and other failures


def test_unknown_handle_records_the_resolution_error_and_others_continue(conn, watchlist, server, clock, fixed_now):
    wl = ts_watchlist(watchlist, TRUMP, VANCE)
    sync_accounts(conn, wl)
    server.add_account(TRUMP, TRUMP_ID, NEWEST)  # JDVance1 is unknown to the server

    counts = collect(conn, wl, RecordingSink(conn, fixed_now), server, clock, fixed_now)

    assert counts["status"] == "partial"
    assert counts["unresolved_handles"] == [VANCE]
    assert counts["accounts_failed"] == 1 and counts["accounts_ok"] == 1
    rows = {r["handle"]: r for r in db.get_accounts(conn, "truthsocial")}
    assert rows[VANCE]["platform_user_id"] is None
    assert rows[VANCE]["resolve_error"] == "HTTP 404: Record not found"
    assert rows[VANCE]["last_fetch_at"] is None
    assert rows[TRUMP]["platform_user_id"] == TRUMP_ID
    assert stored_ids(conn) == storable_ids(NEWEST)


def test_resolved_accounts_are_not_looked_up_again(conn, trump_only, server, clock, fixed_now):
    server.add_account(TRUMP, TRUMP_ID, NEWEST)
    collect(conn, trump_only, RecordingSink(conn, fixed_now), server, clock, fixed_now)
    server.requests.clear()
    collect(conn, trump_only, RecordingSink(conn, fixed_now), server, clock, fixed_now)
    assert [r.url.path for r in server.requests] == [f"/api/v1/accounts/{TRUMP_ID}/statuses"]


@pytest.mark.parametrize(
    "failure",
    [
        lambda request: httpx.Response(500, text="oops"),
        lambda request: httpx.Response(200, text="<html>the web app</html>", headers={"content-type": "text/html"}),
        lambda request: httpx.Response(200, json={"error": "not a list"}),
        lambda request: httpx.Response(200, json=[{"content": "<p>no id</p>"}]),
        lambda request: (_ for _ in ()).throw(httpx.ReadTimeout("timed out", request=request)),
    ],
    ids=["http-500", "html-body", "json-object", "status-without-id", "timeout"],
)
def test_other_failures_skip_only_that_account(conn, watchlist, server, clock, fixed_now, failure):
    wl = ts_watchlist(watchlist, TRUMP, VANCE)
    sync_accounts(conn, wl)
    server.add_account(TRUMP, TRUMP_ID, NEWEST)
    server.add_account(VANCE, VANCE_ID, [])

    def fail_vance(index: int, request: httpx.Request) -> httpx.Response | None:
        return failure(request) if request.url.path == f"/api/v1/accounts/{VANCE_ID}/statuses" else None

    server.override = fail_vance
    counts = collect(conn, wl, RecordingSink(conn, fixed_now), server, clock, fixed_now)

    assert counts["status"] == "partial"
    assert counts["accounts_failed"] == 1 and counts["accounts_ok"] == 1
    assert counts["rate_limited"] == 0
    assert stored_ids(conn) == storable_ids(NEWEST)
    rows = {r["handle"]: r for r in db.get_accounts(conn, "truthsocial")}
    assert rows[VANCE]["last_fetch_at"] is None and rows[TRUMP]["last_fetch_at"] is not None
    assert watermark(conn, VANCE) is None


def test_every_account_failing_is_an_error(conn, trump_only, server, clock, fixed_now):
    server.add_account(TRUMP, TRUMP_ID, NEWEST)
    server.override = lambda i, request: httpx.Response(503, text="unavailable")
    counts = collect(conn, trump_only, RecordingSink(conn, fixed_now), server, clock, fixed_now)
    assert counts["status"] == "error"
    assert counts["accounts_failed"] == 1 and counts["requests"] == 1


# ---------------------------------------------------------------- backfill


def test_backfill_stops_at_first_min_id_and_resumes_after_an_interruption(conn, trump_only, clock, fixed_now):
    server = FakeTruthSocial(clock, page_size=5)
    server.add_account(TRUMP, TRUMP_ID, NEWEST + OLDER + JAN_2025)
    stop = id_for_time(datetime(2026, 9, 22, tzinfo=UTC))
    with conn:
        db.set_watermark(conn, "truthsocial", f"first_min_id:{TRUMP}", stop, fixed_now)
    since = date(2025, 1, 19)
    floor = int(id_for_time(datetime(2025, 1, 19, tzinfo=UTC)))
    expected = {sid for sid in storable_ids(ALL_RECORDED) if floor < int(sid) <= int(stop)}
    assert len(expected) == 7
    older_first = sorted((s["id"] for s in ALL_RECORDED if int(s["id"]) <= int(stop)), key=int, reverse=True)
    # Lookup, two pages, then Cloudflare starts refusing.
    server.override = lambda i, request: rate_limited_429() if i >= 3 else None

    first = backfill(conn, trump_only, RecordingSink(conn, fixed_now), server, clock, since, fixed_now)

    assert first["status"] == "partial" and first["rate_limited"] == 1
    assert server.statuses_requests()[0].url.params["max_id"] == str(int(stop) + 1)
    assert watermark(conn, TRUMP, source="truthsocial_backfill") == older_first[9]
    assert stored_ids(conn) == {sid for sid in expected if int(sid) >= int(older_first[9])}
    assert watermark(conn, TRUMP) is None  # the daily watermark is not the backfill's to move

    server.override = None
    server.requests.clear()
    second = backfill(conn, trump_only, RecordingSink(conn, fixed_now), server, clock, since, fixed_now)

    assert second["status"] == "ok"
    assert server.requests[0].url.params["max_id"] == older_first[9]
    assert stored_ids(conn) == expected
    assert watermark(conn, TRUMP, source="truthsocial_backfill") == str(floor)

    server.requests.clear()
    third = backfill(conn, trump_only, RecordingSink(conn, fixed_now), server, clock, since, fixed_now)
    assert third["status"] == "ok" and third["requests"] == 0 and server.requests == []


def test_backfill_uses_the_daily_watermark_when_there_is_no_first_min_id(conn, trump_only, server, clock, fixed_now):
    server.add_account(TRUMP, TRUMP_ID, NEWEST + OLDER + JAN_2025)
    daily = "117320509296515188"
    with conn:
        db.set_watermark(conn, "truthsocial", TRUMP, daily, fixed_now)

    counts = backfill(conn, trump_only, RecordingSink(conn, fixed_now), server, clock, date(2026, 9, 21), fixed_now)

    assert counts["status"] == "ok"
    assert server.statuses_requests()[0].url.params["max_id"] == str(int(daily) + 1)
    floor = int(id_for_time(datetime(2026, 9, 21, tzinfo=UTC)))
    assert stored_ids(conn) == {sid for sid in storable_ids(ALL_RECORDED) if floor < int(sid) <= int(daily)}


def test_backfill_with_no_daily_state_starts_from_the_newest_page(conn, trump_only, server, clock, fixed_now):
    server.add_account(TRUMP, TRUMP_ID, NEWEST + OLDER)
    counts = backfill(conn, trump_only, RecordingSink(conn, fixed_now), server, clock, date(2026, 9, 23), fixed_now)

    assert counts["status"] == "ok"
    assert "max_id" not in server.statuses_requests()[0].url.params
    floor = int(id_for_time(datetime(2026, 9, 23, tzinfo=UTC)))
    assert stored_ids(conn) == {sid for sid in storable_ids(NEWEST + OLDER) if int(sid) > floor}


# ---------------------------------------------------------------- live


@pytest.mark.live
def test_live_lookup_and_one_page(watchlist):
    """Two real requests, 6 s apart: the account resolves and a statuses page parses into snowflake-ordered posts."""
    headers = {"User-Agent": watchlist.truthsocial.user_agent, "Accept": "application/json"}
    with httpx.Client(headers=headers, timeout=30.0) as client:
        lookup = client.get(f"{ts.BASE_URL}/accounts/lookup", params={"acct": TRUMP})
        assert lookup.status_code == 200, lookup.text[:200]
        account_id = lookup.json()["id"]
        time.sleep(6.0)
        page = client.get(f"{ts.BASE_URL}/accounts/{account_id}/statuses", params={"limit": 20})
        assert page.status_code == 200, page.text[:200]
    statuses = page.json()
    assert statuses
    for s in statuses:
        assert abs(time_for_id(s["id"]) - parse_api_time(s["created_at"])) <= timedelta(seconds=1)
