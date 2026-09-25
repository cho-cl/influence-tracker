from __future__ import annotations

import itertools
import json
import logging
import re
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from pathlib import Path

import httpx
import pytest

from influence_tracker import db
from influence_tracker.collectors.x import (
    AMBIGUOUS_CONTEXT,
    all_queries,
    ambiguous_terms,
    billing_cycle_start,
    build_queries,
    collect_x,
    search_terms,
    x_budget_summary,
)
from influence_tracker.config import Account
from influence_tracker.ingest import PostSink, sync_accounts
from influence_tracker.mentions import MentionMatcher
from influence_tracker.models import MatchResult, Mention
from influence_tracker.timeutil import NY, parse_api_time, to_iso

FIXTURES = Path(__file__).parent / "fixtures"
TOKEN = "test-token-not-real"
SEARCH_PATH = "/2/tweets/search/recent"
USERS_PATH = "/2/users/by"
SUFFIX = " -is:retweet"

SearchFn = Callable[[httpx.Request, str], httpx.Response]


def load(name: str) -> dict:
    with open(FIXTURES / name, encoding="utf-8") as f:
        return json.load(f)


def ok(body: dict) -> httpx.Response:
    return httpx.Response(200, json=body)


EMPTY = load("x_search_recent_empty.json")
PAGE1 = load("x_search_recent_page1.json")
PAGE2 = load("x_search_recent_page2.json")
USERS = load("x_users_by.json")


class CashtagMatcher:
    """Stands in for mentions.py: one cashtag mention per platform-tagged symbol."""

    def match(self, text: str, platform: str, cashtag_hints=()) -> MatchResult:
        return MatchResult(mentions=[Mention(t, "cashtag", f"${t}") for t in dict.fromkeys(cashtag_hints)])


class FakeX:
    """X API v2 stand-in: /2/users/by answers from the users fixture, search delegates to a per-test function."""

    def __init__(
        self,
        search: SearchFn | None = None,
        users: dict | None = None,
        lookup: Callable[[list[str]], httpx.Response] | None = None,
    ):
        users = users or USERS
        self.users = {u["username"].lower(): u for u in users.get("data", [])}
        self.not_found = users["errors"][0]
        self.search = search or (lambda request, query: ok(EMPTY))
        self.lookup = lookup or self._users
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.path == USERS_PATH:
            return self.lookup(request.url.params["usernames"].split(","))
        if request.url.path == SEARCH_PATH:
            return self.search(request, request.url.params["query"])
        return httpx.Response(404, json={"title": "Not Found Error", "detail": "no route"})

    def _users(self, names: list[str]) -> httpx.Response:
        body: dict = {}
        found = [self.users[n.lower()] for n in names if n.lower() in self.users]
        missing = [n for n in names if n.lower() not in self.users]
        if found:
            body["data"] = found
        if missing:
            body["errors"] = [
                {
                    **self.not_found,
                    "value": n,
                    "resource_id": n,
                    "detail": f"Could not find user with usernames: [{n}].",
                }
                for n in missing
            ]
        return ok(body)

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self.handler))

    @property
    def search_requests(self) -> list[httpx.Request]:
        return [r for r in self.requests if r.url.path == SEARCH_PATH]

    @property
    def user_requests(self) -> list[httpx.Request]:
        return [r for r in self.requests if r.url.path == USERS_PATH]


def paged(routes: dict[str, list[dict]]) -> SearchFn:
    """Serve fixture pages per exact query, following each page's meta.next_token; other queries get no results."""

    def search(request: httpx.Request, query: str) -> httpx.Response:
        pages = routes.get(query)
        if pages is None:
            return ok(EMPTY)
        token = request.url.params.get("next_token")
        if token is None:
            return ok(pages[0])
        for i, page in enumerate(pages[:-1]):
            if page["meta"].get("next_token") == token:
                return ok(pages[i + 1])
        raise AssertionError(f"unknown next_token {token!r}")

    return search


def no_sleep(seconds: float) -> None:
    raise AssertionError(f"unexpected sleep({seconds})")


class PostStream:
    """Recent search over a synthetic stream of $TSLA posts by elonmusk, served the way X does: only posts in
    [start_time, end_time), newest first, max_results per page, next_token paging. A start_time older than 7 days by
    the stream's clock is refused with X's 400. Only has:cashtags queries match. Billing is per post per UTC day."""

    def __init__(self, times: list[datetime], clock: Callable[[], datetime]):
        posts = [(str(2104000000000000000 + i), t) for i, t in enumerate(sorted(times))]
        self.posts = posts[::-1]
        self.clock = clock
        self.rejected: list[str] = []
        self.billed: dict[str, set[str]] = {}

    def __call__(self, request: httpx.Request, query: str) -> httpx.Response:
        params = request.url.params
        start, end = parse_api_time(params["start_time"]), parse_api_time(params["end_time"])
        if start < self.clock() - timedelta(days=7):
            self.rejected.append(params["start_time"])
            return httpx.Response(400, json=load("x_error_400_invalid_request.json"))
        if "has:cashtags" not in query:
            return ok(EMPTY)
        matching = [p for p in self.posts if start <= p[1] < end]
        offset = int(params.get("next_token", "0"))
        size = int(params["max_results"])
        page = matching[offset : offset + size]
        self.billed.setdefault(self.clock().date().isoformat(), set()).update(pid for pid, _ in page)
        meta: dict = {"result_count": len(page)}
        if offset + size < len(matching):
            meta["next_token"] = str(offset + size)
        data = [
            {
                "id": pid,
                "author_id": "44196397",
                "created_at": t.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                "text": f"$TSLA post {pid}",
                "entities": {"cashtags": [{"start": 0, "end": 5, "tag": "TSLA"}]},
            }
            for pid, t in page
        ]
        return ok({"data": data, "meta": meta} if data else {"meta": meta})

    def ids_between(self, start: datetime, end: datetime) -> set[str]:
        return {pid for pid, t in self.posts if start <= t < end}

    def total_billed(self) -> int:
        return sum(len(ids) for ids in self.billed.values())


def evenly(start: datetime, end: datetime, per_day: int, offset_s: int = 0) -> list[datetime]:
    step = timedelta(days=1) / per_day
    count = int((end - start) / step)
    return [start + timedelta(seconds=offset_s) + i * step for i in range(count)]


def stored_ids(conn) -> set[str]:
    return {r["native_id"] for r in conn.execute("SELECT native_id FROM posts WHERE platform = 'x'")}


def post_reads(conn) -> int:
    return conn.execute("SELECT COALESCE(SUM(units), 0) FROM x_usage WHERE kind = 'post_read'").fetchone()[0]


def run_collector(
    conn, watchlist, fake: FakeX, now: datetime, sleeps: list[float] | None = None, token=TOKEN, matcher=None
):
    sink = PostSink(conn, matcher or CashtagMatcher(), now=lambda: now)
    run_id = db.start_run(conn, "collect_x", now)
    sleep = sleeps.append if sleeps is not None else no_sleep
    with fake.client() as client:
        return collect_x(conn, watchlist, token, sink, run_id, now, client=client, sleep=sleep)


def preresolve(conn, now: datetime) -> None:
    ids = {u["username"].lower(): u["id"] for u in USERS["data"]}
    with conn:
        for row in db.get_accounts(conn, "x"):
            if row["handle"].lower() in ids:
                db.set_account_resolution(conn, "x", row["handle"], ids[row["handle"].lower()], None, now)


def real_queries(watchlist, mode: str = "cashtags") -> list[str]:
    return all_queries([a.handle for a in watchlist.x.accounts if a.active], watchlist, mode)


def cashtag_query(watchlist) -> str:
    return next(q for q in real_queries(watchlist) if "has:cashtags" in q)


def spend(conn, when: datetime, usd: float) -> None:
    """Spend by earlier runs; tests of the cycle budget date it more than 7 days back, outside the pacing period."""
    with conn:
        db.record_x_usage(conn, None, when, "someone", "post_read", round(usd / 0.005), usd)


def usage_by_handle(conn, kind: str) -> dict[str, int]:
    rows = conn.execute("SELECT handle, SUM(units) AS u FROM x_usage WHERE kind = ? GROUP BY handle", (kind,))
    return {r["handle"]: r["u"] for r in rows}


def post_row(conn, native_id: str):
    return conn.execute("SELECT * FROM posts WHERE platform = 'x' AND native_id = ?", (native_id,)).fetchone()


# ---------------------------------------------------------------- query building

_TOKEN_RE = re.compile(r'"[^"]*"|[^\s"()]+')


def parse_query(q: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    assert q.endswith(SUFFIX)
    body = q[: -len(SUFFIX)]
    if body.startswith("("):
        close = body.index(") ")
        accounts, rest = tuple(body[1:close].split(" OR ")), body[close + 2 :]
    else:
        first, rest = body.split(" ", 1)
        accounts = (first,)
    if rest.startswith("(") and rest.endswith(")"):
        terms = tuple(t for t in _TOKEN_RE.findall(rest[1:-1]) if t != "OR")
    else:
        terms = (rest,)
    return accounts, terms


def unquote(term: str) -> str:
    return term[1:-1] if term.startswith('"') else term


@pytest.mark.parametrize("mode", ["cashtags", "explicit"])
@pytest.mark.parametrize("max_len", [512, 200])
def test_chunking_invariants_on_real_watchlist(watchlist, mode, max_len):
    handles = [a.handle for a in watchlist.x.accounts]
    terms = search_terms(watchlist, mode)
    queries = build_queries(handles, terms, max_len=max_len)

    assert all(len(q) <= max_len for q in queries)
    parsed = [parse_query(q) for q in queries]
    account_chunks = list(dict.fromkeys(a for a, _ in parsed))
    term_chunks = list(dict.fromkeys(t for _, t in parsed))
    # Exactly once: the chunks partition the inputs.
    assert sorted(a for chunk in account_chunks for a in chunk) == sorted(f"from:{h}" for h in handles)
    assert sorted(unquote(t) for chunk in term_chunks for t in chunk) == sorted(terms)
    # Full product of account chunks x term chunks, each pair once.
    assert len(queries) == len(account_chunks) * len(term_chunks)
    assert set(parsed) == set(itertools.product(account_chunks, term_chunks))
    if max_len == 200:
        assert len(account_chunks) > 1 and len(term_chunks) > 1


def test_real_watchlist_terms_include_every_name_once(watchlist):
    terms = search_terms(watchlist, "cashtags")
    assert terms[0] == "has:cashtags"
    names = {n for t in watchlist.tickers for n in (*t.names, *t.cased_names)}
    assert set(terms[1:]) == names
    assert len(terms) == len({t.lower() for t in terms})
    ambiguous = ambiguous_terms(watchlist)
    assert set(ambiguous) == {n for t in watchlist.tickers for n in t.ambiguous_names}
    assert not {t.lower() for t in terms} & {a.lower() for a in ambiguous}

    explicit = search_terms(watchlist, "explicit")
    assert "has:cashtags" not in explicit
    assert {"$TSLA", "$GOOG", "$GOOGL", "$BRK.B", "$BRK.A", "$SPY", "Tesla", "AT&T"} <= set(explicit)


@pytest.mark.parametrize("mode", ["cashtags", "explicit"])
def test_ambiguous_names_are_only_searched_with_a_finance_word(watchlist, mode):
    handles = [a.handle for a in watchlist.x.accounts]
    queries = all_queries(handles, watchlist, mode)
    assert all(len(q) <= 512 for q in queries)
    ambiguous = [q for q in queries if AMBIGUOUS_CONTEXT in q]
    plain = [q for q in queries if AMBIGUOUS_CONTEXT not in q]
    assert plain == build_queries(handles, search_terms(watchlist, mode))
    assert ambiguous and all(q.endswith(f" {AMBIGUOUS_CONTEXT}{SUFFIX}") for q in ambiguous)
    # Every ambiguous name appears in some context query, and never in a plain one.
    for name in ambiguous_terms(watchlist):
        token = name if name.isalnum() else f'"{name}"'
        pattern = re.compile(rf"(?<![\w\"$]){re.escape(token)}(?![\w\"])")
        assert any(pattern.search(q.replace(AMBIGUOUS_CONTEXT, "")) for q in ambiguous), name
        assert not any(pattern.search(q) for q in plain), name
    # Every account is covered by the context queries too.
    for h in handles:
        assert any(f"from:{h} " in q or f"from:{h})" in q for q in ambiguous), h


def test_terms_with_spaces_or_punctuation_are_quoted():
    terms = [
        "has:cashtags",
        "Tesla",
        "AT&T",
        "McDonald's",
        "J.P. Morgan",
        "Coca-Cola",
        "Bank of America",
        "$TSLA",
        "$BRK.B",
    ]
    assert build_queries(["elonmusk", "jimcramer"], terms) == [
        "(from:elonmusk OR from:jimcramer) "
        '(has:cashtags OR Tesla OR "AT&T" OR "McDonald\'s" OR "J.P. Morgan" OR "Coca-Cola" OR "Bank of America" '
        "OR $TSLA OR $BRK.B) -is:retweet"
    ]


def test_real_watchlist_queries_quote_punctuated_names(watchlist):
    joined = "\n".join(real_queries(watchlist))
    for quoted in ('"AT&T"', '"McDonald\'s"', '"J.P. Morgan"', '"Coca-Cola"', '"Bank of America"'):
        assert quoted in joined
    assert re.search(r"[( ]Tesla[ )]", joined)
    assert '"Tesla"' not in joined and '"has:cashtags"' not in joined


def test_single_account_and_term_need_no_parentheses():
    assert build_queries(["elonmusk"], ["Tesla"]) == ["from:elonmusk Tesla -is:retweet"]


def test_duplicates_are_dropped_case_insensitively():
    assert build_queries(["saylor", "Saylor"], ["Tesla", "tesla"]) == ["from:saylor Tesla -is:retweet"]


def test_build_queries_rejects_what_cannot_fit():
    with pytest.raises(ValueError):
        build_queries(["elonmusk"], ["x" * 600])
    with pytest.raises(ValueError):
        build_queries([], ["Tesla"])
    with pytest.raises(ValueError):
        build_queries(["elonmusk"], [])


# ---------------------------------------------------------------- billing cycle


@pytest.mark.parametrize(
    ("now", "cycle_day", "expected"),
    [
        (datetime(2026, 10, 3, 12, 0, tzinfo=UTC), 15, datetime(2026, 9, 15, tzinfo=UTC)),
        (datetime(2026, 10, 1, 0, 0, tzinfo=UTC), 1, datetime(2026, 10, 1, tzinfo=UTC)),
        (datetime(2026, 10, 1, 23, 59, tzinfo=UTC), 1, datetime(2026, 10, 1, tzinfo=UTC)),
        (datetime(2026, 9, 30, 23, 59, 59, tzinfo=UTC), 1, datetime(2026, 9, 1, tzinfo=UTC)),
        (datetime(2026, 10, 15, 0, 0, tzinfo=UTC), 15, datetime(2026, 10, 15, tzinfo=UTC)),
        (datetime(2026, 10, 14, 23, 59, 59, tzinfo=UTC), 15, datetime(2026, 9, 15, tzinfo=UTC)),
        (datetime(2027, 1, 3, 8, 0, tzinfo=UTC), 15, datetime(2026, 12, 15, tzinfo=UTC)),
        # 21:00 New York on Sep 30 is already Oct 1 in UTC.
        (datetime(2026, 9, 30, 21, 0, tzinfo=NY), 1, datetime(2026, 10, 1, tzinfo=UTC)),
    ],
)
def test_billing_cycle_start(now, cycle_day, expected):
    start = billing_cycle_start(now, cycle_day)
    assert start == expected
    assert start.tzinfo is not None and start.utcoffset() == timedelta(0)


def test_billing_cycle_start_rejects_bad_input():
    with pytest.raises(ValueError):
        billing_cycle_start(datetime(2026, 10, 3), 1)
    with pytest.raises(ValueError):
        billing_cycle_start(datetime(2026, 10, 3, tzinfo=UTC), 31)


# ---------------------------------------------------------------- full runs


def test_full_run_stores_posts_and_advances_watermark(conn, watchlist, fixed_now):
    sync_accounts(conn, watchlist)
    queries = real_queries(watchlist)
    fake = FakeX(paged({cashtag_query(watchlist): [PAGE1, PAGE2]}))

    counts = run_collector(conn, watchlist, fake, fixed_now)

    assert counts["status"] == "ok"
    assert counts["query_mode"] == "cashtags"
    assert counts["queries"] == len(queries)
    assert counts["pages"] == len(queries) + 1
    assert (counts["posts_read"], counts["stored"], counts["new"], counts["with_mentions"]) == (5, 5, 5, 4)
    assert counts["watermark_advanced"] is True and counts["budget_stopped"] is False
    assert counts["unresolved_handles"] == []
    assert db.get_watermark(conn, "x", "search_end_time") == "2026-09-24T22:29:30Z"

    first = fake.search_requests[0].url.params
    assert first["start_time"] == "2026-09-17T22:35:00Z"
    assert first["end_time"] == "2026-09-24T22:29:30Z"
    assert first["max_results"] == "100"
    assert first["tweet.fields"] == "created_at,author_id,entities,public_metrics,lang,note_tweet"
    assert "next_token" not in first
    assert fake.search_requests[0].headers["Authorization"] == f"Bearer {TOKEN}"
    follow_up = [r.url.params for r in fake.search_requests if "next_token" in r.url.params]
    assert len(follow_up) == 1
    assert follow_up[0]["next_token"] == PAGE1["meta"]["next_token"]
    assert follow_up[0]["query"] == cashtag_query(watchlist)
    assert {r.url.params["query"] for r in fake.search_requests} == set(queries)

    assert len(fake.user_requests) == 1
    rows = db.get_accounts(conn, "x")
    assert all(r["platform_user_id"] for r in rows)
    assert all(r["last_fetch_at"] == to_iso(fixed_now) for r in rows)


def test_note_tweet_text_and_cashtags_are_preferred(conn, watchlist, fixed_now):
    sync_accounts(conn, watchlist)
    run_collector(conn, watchlist, FakeX(paged({cashtag_query(watchlist): [PAGE1, PAGE2]})), fixed_now)

    long_post, short_post = PAGE1["data"][0], PAGE1["data"][1]
    row = post_row(conn, long_post["id"])
    assert row["text"] == long_post["note_tweet"]["text"]
    assert row["text"] != long_post["text"]
    assert row["cashtag_hints"] == "TSLA,NVDA"
    assert row["author"] == "elonmusk" and row["author_id"] == "44196397"
    assert row["url"] == f"https://x.com/elonmusk/status/{long_post['id']}"
    assert row["created_at_utc"] == "2026-09-24T19:42:07Z"
    assert json.loads(row["metrics_json"]) == long_post["public_metrics"]
    tickers = {r["ticker"] for r in conn.execute("SELECT ticker FROM mentions WHERE native_id = ?", (long_post["id"],))}
    assert tickers == {"TSLA", "NVDA"}

    row = post_row(conn, short_post["id"])
    assert row["text"] == short_post["text"]
    assert row["cashtag_hints"] == "AAPL"
    assert row["author"] == "jimcramer"


def test_html_entities_in_post_text_are_decoded(conn, watchlist, fixed_now):
    """X sends &, < and > escaped; the stored text must carry the real characters, or names like AT&T never match."""
    sync_accounts(conn, watchlist)
    plain = {
        "id": "2102000000000000002",
        "author_id": "44196397",
        "created_at": "2026-09-24T20:00:00.000Z",
        "text": "AT&amp;T and Johnson &amp; Johnson shares look cheap &gt; 5% yield",
    }
    long_post = {
        "id": "2102000000000000003",
        "author_id": "14216123",
        "created_at": "2026-09-24T20:05:00.000Z",
        "text": "Q&amp;A on AT&amp;T",
        "note_tweet": {"text": "Q&amp;A on AT&amp;T &lt;- buy the stock, literal &amp;amp; stays"},
    }
    page = {"data": [plain, long_post], "meta": {"result_count": 2}}
    fake = FakeX(paged({cashtag_query(watchlist): [page]}))
    run_collector(conn, watchlist, fake, fixed_now, matcher=MentionMatcher(watchlist))

    assert post_row(conn, plain["id"])["text"] == "AT&T and Johnson & Johnson shares look cheap > 5% yield"
    assert post_row(conn, long_post["id"])["text"] == "Q&A on AT&T <- buy the stock, literal &amp; stays"
    tickers = {
        (r["native_id"], r["ticker"])
        for r in conn.execute("SELECT native_id, ticker FROM mentions WHERE platform = 'x'")
    }
    assert tickers == {(plain["id"], "T"), (plain["id"], "JNJ"), (long_post["id"], "T")}


def test_unknown_author_falls_back_to_author_id(conn, watchlist, fixed_now):
    sync_accounts(conn, watchlist)
    stranger = {**PAGE2["data"][1], "id": "2102000000000000001", "author_id": "999999"}
    page = {"data": [stranger], "meta": {"result_count": 1}}
    run_collector(conn, watchlist, FakeX(paged({cashtag_query(watchlist): [page]})), fixed_now)

    row = post_row(conn, "2102000000000000001")
    assert row["author"] == "999999"
    assert row["url"] == "https://x.com/i/status/2102000000000000001"


def test_watermark_not_advanced_when_a_query_fails(conn, watchlist, fixed_now):
    sync_accounts(conn, watchlist)
    queries = real_queries(watchlist)
    failing = next(q for q in queries if "has:cashtags" not in q)
    pages = paged({cashtag_query(watchlist): [PAGE1, PAGE2]})

    def search(request, query):
        if query == failing:
            return httpx.Response(400, json=load("x_error_400_invalid_request.json"))
        return pages(request, query)

    fake = FakeX(search)
    counts = run_collector(conn, watchlist, fake, fixed_now)

    assert counts["status"] == "partial"
    assert counts["watermark_advanced"] is False
    assert counts["error"] and "400" in counts["error"]
    assert db.get_watermark(conn, "x", "search_end_time") is None
    # The other queries still ran and their posts were kept.
    assert {r.url.params["query"] for r in fake.search_requests} == set(queries)
    assert counts["stored"] == 5
    assert all(r["last_fetch_at"] is None for r in db.get_accounts(conn, "x"))


def test_unparseable_page_is_partial_not_raised(conn, watchlist, fixed_now):
    sync_accounts(conn, watchlist)
    preresolve(conn, fixed_now)
    target = real_queries(watchlist)[-1]

    def search(request, query):
        if query == target:
            return httpx.Response(200, text="<html>upstream error</html>")
        return ok(EMPTY)

    counts = run_collector(conn, watchlist, FakeX(search), fixed_now)
    assert counts["status"] == "partial"
    assert db.get_watermark(conn, "x", "search_end_time") is None


# ---------------------------------------------------------------- window and pacing


def test_second_run_starts_at_watermark_and_paces_by_days(conn, watchlist, fixed_now):
    sync_accounts(conn, watchlist)
    run_collector(conn, watchlist, FakeX(), fixed_now)

    later = fixed_now + timedelta(days=1)
    fake = FakeX()
    counts = run_collector(conn, watchlist, fake, later)

    params = fake.search_requests[0].url.params
    assert params["start_time"] == "2026-09-24T22:29:30Z"
    assert params["end_time"] == "2026-09-25T22:29:30Z"
    assert counts["cap"] == 66
    assert params["max_results"] == "66"
    assert fake.user_requests == []
    assert db.get_watermark(conn, "x", "search_end_time") == "2026-09-25T22:29:30Z"


def test_old_watermark_is_clamped_to_the_search_window(conn, watchlist, fixed_now):
    sync_accounts(conn, watchlist)
    preresolve(conn, fixed_now)
    with conn:
        db.set_watermark(conn, "x", "search_end_time", to_iso(fixed_now - timedelta(days=10)), fixed_now)
    fake = FakeX()
    counts = run_collector(conn, watchlist, fake, fixed_now)

    assert fake.search_requests[0].url.params["start_time"] == "2026-09-17T22:35:00Z"
    assert counts["cap"] == 66 * 7


def test_empty_window_is_skipped(conn, watchlist, fixed_now):
    sync_accounts(conn, watchlist)
    preresolve(conn, fixed_now)
    mark = to_iso(fixed_now - timedelta(seconds=10))
    with conn:
        db.set_watermark(conn, "x", "search_end_time", mark, fixed_now)
    fake = FakeX()
    counts = run_collector(conn, watchlist, fake, fixed_now)

    assert counts["status"] == "ok"
    assert fake.search_requests == []
    assert counts["posts_read"] == 0
    assert db.get_watermark(conn, "x", "search_end_time") == mark


def test_first_run_cap_is_daily_allowance_times_max_catchup_days(conn, watchlist, fixed_now):
    sync_accounts(conn, watchlist)
    preresolve(conn, fixed_now)
    assert run_collector(conn, watchlist, FakeX(), fixed_now)["cap"] == 66 * 7


def test_cap_counts_whole_days_since_watermark(conn, watchlist, fixed_now):
    sync_accounts(conn, watchlist)
    preresolve(conn, fixed_now)
    end = fixed_now - timedelta(seconds=30)
    with conn:
        db.set_watermark(conn, "x", "search_end_time", to_iso(end - timedelta(days=2, hours=12)), fixed_now)
    assert run_collector(conn, watchlist, FakeX(), fixed_now)["cap"] == 66 * 3


def test_cap_limited_by_remaining_cycle_budget(conn, watchlist, fixed_now):
    sync_accounts(conn, watchlist)
    preresolve(conn, fixed_now)
    spend(conn, fixed_now - timedelta(days=10), 9.80)
    fake = FakeX()
    counts = run_collector(conn, watchlist, fake, fixed_now)

    assert counts["cap"] == 40
    assert fake.search_requests[0].url.params["max_results"] == "40"


def test_cap_ignores_spend_from_before_the_cycle_rollover(conn, watchlist):
    watchlist.x.billing_cycle_day = 15
    now = datetime(2026, 10, 3, 12, 0, tzinfo=UTC)
    sync_accounts(conn, watchlist)
    preresolve(conn, now)
    spend(conn, datetime(2026, 9, 10, 18, 0, tzinfo=UTC), 9.00)
    spend(conn, datetime(2026, 9, 20, 18, 0, tzinfo=UTC), 9.50)

    counts = run_collector(conn, watchlist, FakeX(), now)
    assert counts["cap"] == 100

    summary = x_budget_summary(conn, watchlist, now)
    assert summary["cycle_start"] == datetime(2026, 9, 15, tzinfo=UTC)
    assert summary["spent_usd"] == pytest.approx(9.50)
    assert summary["remaining_usd"] == pytest.approx(0.50)


def test_budget_below_minimum_page_skips_search(conn, watchlist, fixed_now):
    sync_accounts(conn, watchlist)
    preresolve(conn, fixed_now)
    spend(conn, fixed_now - timedelta(days=10), 9.97)
    fake = FakeX()
    counts = run_collector(conn, watchlist, fake, fixed_now)

    assert fake.search_requests == []
    assert counts["cap"] == 6
    assert counts["status"] == "partial" and counts["budget_stopped"] is True
    assert db.get_watermark(conn, "x", "search_end_time") is None


def _synthetic_post(n: int, author_id: str = "44196397") -> dict:
    return {
        "id": str(2103000000000000000 + n),
        "author_id": author_id,
        "created_at": "2026-09-24T15:00:00.000Z",
        "lang": "en",
        "text": f"$TSLA note {n}",
        "entities": {"cashtags": [{"start": 0, "end": 5, "tag": "TSLA"}]},
        "public_metrics": {"retweet_count": 0, "reply_count": 0, "like_count": n, "quote_count": 0},
    }


def test_budget_stop_caps_reads_and_keeps_watermark(conn, watchlist, fixed_now, caplog):
    sync_accounts(conn, watchlist)
    preresolve(conn, fixed_now)
    spend(conn, fixed_now - timedelta(days=10), 9.85)
    serial = itertools.count(1)

    def endless(request, query):
        if "has:cashtags" not in query:
            return ok(EMPTY)
        n = min(int(request.url.params["max_results"]), 20)
        posts = [_synthetic_post(next(serial)) for _ in range(n)]
        return ok({"data": posts, "meta": {"result_count": n, "next_token": f"t{posts[-1]['id']}"}})

    fake = FakeX(endless)
    with caplog.at_level(logging.WARNING, logger="influence_tracker.collectors.x"):
        counts = run_collector(conn, watchlist, fake, fixed_now)

    assert counts["cap"] == 30
    assert [r.url.params["max_results"] for r in fake.search_requests] == ["30", "10"]
    assert counts["posts_read"] == 30
    assert counts["budget_stopped"] is True and counts["status"] == "partial"
    assert counts["watermark_advanced"] is False
    assert db.get_watermark(conn, "x", "search_end_time") is None
    assert usage_by_handle(conn, "post_read")["elonmusk"] == 30
    assert counts["est_cost_usd"] == pytest.approx(0.15)
    assert "watermark" in caplog.text


def run_daily(conn, watchlist, stream: PostStream, clock: dict, start: datetime, days: int) -> list[dict]:
    runs = []
    for day in range(days):
        clock["now"] = start + timedelta(days=day)
        runs.append(run_collector(conn, watchlist, FakeX(stream), clock["now"]))
    return runs


def test_busy_day_backlog_is_worked_off_at_the_daily_allowance(conn, watchlist, fixed_now):
    """50 posts a day plus one 200-post day. A budget stop must not hand the already-spent allowance back: every later
    run gets one day's 66 reads, continues with the unread older posts, and catches up without losing any."""
    sync_accounts(conn, watchlist)
    preresolve(conn, fixed_now)
    busy = fixed_now + timedelta(days=3)
    times = evenly(fixed_now - timedelta(days=8), fixed_now + timedelta(days=20), 50)
    times += evenly(busy, busy + timedelta(days=1), 150, offset_s=7)
    clock = {"now": fixed_now}
    stream = PostStream(times, lambda: clock["now"])

    runs = run_daily(conn, watchlist, stream, clock, fixed_now, 16)

    assert runs[0]["cap"] == 66 * 7
    assert all(r["cap"] <= 66 for r in runs[1:])
    # Reads since the first run never outrun the daily allowance by more than one minimum page (10 posts).
    reads = list(itertools.accumulate(r["posts_read"] for r in runs[1:]))
    assert all(total <= 66 * day + 9 for day, total in enumerate(reads, start=1))
    stopped = [r["budget_stopped"] for r in runs]
    assert stopped[:5] == [False, False, False, False, True]
    assert runs[-1]["caught_up"] is True and runs[-1]["status"] == "ok"
    assert db.get_watermark(conn, "x", "search_end_time") == to_iso(clock["now"] - timedelta(seconds=30))
    first_floor = fixed_now - timedelta(days=7) + timedelta(minutes=5)
    assert stream.ids_between(first_floor, clock["now"]) <= stored_ids(conn)
    # A resumed page re-reads only the second of the oldest post it had reached.
    assert 0 < stream.total_billed() - len(stored_ids(conn)) <= sum(stopped)
    assert post_reads(conn) == stream.total_billed()
    assert stream.rejected == []


def test_sustained_overload_is_paced_to_the_daily_allowance(conn, watchlist, fixed_now):
    """100 posts a day against a 66-post allowance: after the first run's 7-day grant each run reads one day's
    allowance, each still stores new posts instead of re-buying the same newest ones, and the posts it falls too far
    behind to reach are reported rather than silently dropped."""
    sync_accounts(conn, watchlist)
    preresolve(conn, fixed_now)
    clock = {"now": fixed_now}
    times = evenly(fixed_now - timedelta(days=8), fixed_now + timedelta(days=16), 100)
    stream = PostStream(times, lambda: clock["now"])

    runs = run_daily(conn, watchlist, stream, clock, fixed_now, 14)

    assert (runs[0]["cap"], runs[0]["posts_read"]) == (66 * 7, 66 * 7)
    assert [(r["cap"], r["posts_read"]) for r in runs[1:]] == [(66, 66)] * 13
    # Only the resumed page's boundary second is read twice.
    assert all(r["new"] >= r["posts_read"] - 1 for r in runs[1:])
    assert post_reads(conn) == 66 * 7 + 66 * 13
    assert all(r["status"] == "partial" for r in runs)
    assert any(r["window_gaps"] for r in runs)
    assert stream.rejected == []


def test_failed_query_is_retried_alone_while_the_others_move_on(conn, watchlist, fixed_now):
    sync_accounts(conn, watchlist)
    preresolve(conn, fixed_now)
    queries = real_queries(watchlist)
    failing = queries[-1]
    broken = {"on": True}

    def search(request, query):
        if query == failing and broken["on"]:
            return httpx.Response(400, json=load("x_error_400_invalid_request.json"))
        return ok(EMPTY)

    counts = run_collector(conn, watchlist, FakeX(search), fixed_now)
    assert counts["status"] == "partial" and counts["caught_up"] is False

    broken["on"] = False
    later = fixed_now + timedelta(days=1)
    fake = FakeX(search)
    counts = run_collector(conn, watchlist, fake, later)

    starts: dict[str, list[str]] = {}
    for r in fake.search_requests:
        starts.setdefault(r.url.params["query"], []).append(r.url.params["start_time"])
    # Only the failed query goes back over the first run's window; the others read just the new day.
    assert starts.pop(failing) == [to_iso(later - timedelta(days=7) + timedelta(minutes=5)), "2026-09-24T22:29:30Z"]
    assert starts == {q: ["2026-09-24T22:29:30Z"] for q in queries if q != failing}
    assert counts["status"] == "ok" and counts["caught_up"] is True
    assert db.get_watermark(conn, "x", "search_end_time") == "2026-09-25T22:29:30Z"


def test_watermark_older_than_the_window_reports_the_lost_range(conn, watchlist, fixed_now, caplog):
    sync_accounts(conn, watchlist)
    preresolve(conn, fixed_now)
    mark = fixed_now - timedelta(days=9)
    with conn:
        db.set_watermark(conn, "x", "search_end_time", to_iso(mark), mark)
    clock = {"now": fixed_now}
    stream = PostStream(evenly(fixed_now - timedelta(days=10), fixed_now + timedelta(days=2), 5), lambda: clock["now"])

    with caplog.at_level(logging.WARNING, logger="influence_tracker.collectors.x"):
        counts = run_collector(conn, watchlist, FakeX(stream), fixed_now)

    floor = fixed_now - timedelta(days=7) + timedelta(minutes=5)
    assert counts["status"] == "partial"
    assert counts["window_gaps"] == [[to_iso(mark), to_iso(floor)]]
    assert to_iso(mark) in counts["error"] and to_iso(mark) in caplog.text and to_iso(floor) in caplog.text
    assert counts["caught_up"] is True
    assert stream.ids_between(floor, fixed_now) <= stored_ids(conn)

    clock["now"] = fixed_now + timedelta(days=1)
    counts = run_collector(conn, watchlist, FakeX(stream), clock["now"])
    assert counts["status"] == "ok" and counts["window_gaps"] == []


# ---------------------------------------------------------------- billing attribution


def test_duplicate_posts_across_queries_are_counted_once(conn, watchlist, fixed_now):
    sync_accounts(conn, watchlist)
    preresolve(conn, fixed_now)
    queries = real_queries(watchlist)
    overlap = {"data": [PAGE1["data"][0], PAGE2["data"][0]], "meta": {"result_count": 2}}
    fake = FakeX(paged({queries[0]: [PAGE1, PAGE2], queries[1]: [overlap]}))

    counts = run_collector(conn, watchlist, fake, fixed_now)

    assert counts["posts_read"] == 5
    assert (counts["stored"], counts["new"]) == (7, 5)
    assert sum(usage_by_handle(conn, "post_read").values()) == 5
    assert counts["est_cost_usd"] == pytest.approx(5 * 0.005)


def test_x_usage_is_attributed_per_handle(conn, watchlist, fixed_now):
    sync_accounts(conn, watchlist)
    counts = run_collector(conn, watchlist, FakeX(paged({cashtag_query(watchlist): [PAGE1, PAGE2]})), fixed_now)

    assert usage_by_handle(conn, "post_read") == {"elonmusk": 2, "jimcramer": 1, "CathieDWood": 1, "saylor": 1}
    user_reads = usage_by_handle(conn, "user_read")
    assert user_reads == {a.handle: 1 for a in watchlist.x.accounts}
    assert counts["est_cost_usd"] == pytest.approx(5 * 0.005 + 17 * 0.010)

    summary = x_budget_summary(conn, watchlist, fixed_now)
    assert summary["cycle_start"] == datetime(2026, 9, 1, tzinfo=UTC)
    assert summary["budget_usd"] == 10.0
    assert summary["spent_usd"] == pytest.approx(0.195)
    assert summary["remaining_usd"] == pytest.approx(9.805)
    assert summary["daily_allowance_posts"] == 66
    top = summary["per_handle"][0]
    assert top == {"handle": "elonmusk", "post_reads": 2, "user_reads": 1, "cost_usd": pytest.approx(0.02)}
    assert len(summary["per_handle"]) == 17


def test_budget_summary_with_no_usage(conn, watchlist, fixed_now):
    summary = x_budget_summary(conn, watchlist, fixed_now)
    assert summary == {
        "cycle_start": datetime(2026, 9, 1, tzinfo=UTC),
        "budget_usd": 10.0,
        "spent_usd": 0.0,
        "remaining_usd": 10.0,
        "per_handle": [],
        "daily_allowance_posts": 66,
    }


# ---------------------------------------------------------------- handle resolution


def test_users_by_resolution_with_not_found_and_invalid_handles(conn, watchlist, fixed_now):
    accounts = [a for a in watchlist.x.accounts if a.handle != "elonmusk"]
    watchlist.x.accounts = [
        Account(handle="ElonMusk", category="exec"),
        *accounts,
        Account(handle="nosuchuser_x1", category="media"),
        Account(handle="bad-handle!", category="media"),
    ]
    sync_accounts(conn, watchlist)
    fake = FakeX()
    counts = run_collector(conn, watchlist, fake, fixed_now)

    assert counts["status"] == "ok"
    assert len(fake.user_requests) == 1
    sent = fake.user_requests[0].url.params["usernames"].split(",")
    assert "ElonMusk" in sent and "nosuchuser_x1" in sent and "bad-handle!" not in sent
    rows = {r["handle"]: r for r in db.get_accounts(conn, "x")}
    assert rows["ElonMusk"]["platform_user_id"] == "44196397"
    assert rows["ElonMusk"]["resolve_error"] is None
    assert rows["nosuchuser_x1"]["platform_user_id"] is None
    assert rows["nosuchuser_x1"]["resolve_error"] == "Could not find user with usernames: [nosuchuser_x1]."
    assert "valid" in rows["bad-handle!"]["resolve_error"]
    assert counts["unresolved_handles"] == ["bad-handle!", "nosuchuser_x1"]
    assert usage_by_handle(conn, "user_read")["ElonMusk"] == 1
    assert sum(usage_by_handle(conn, "user_read").values()) == 17
    assert not any("bad-handle!" in r.url.params["query"] for r in fake.search_requests)
    # Unresolved accounts are not marked as fetched.
    assert rows["nosuchuser_x1"]["last_fetch_at"] is None
    assert rows["ElonMusk"]["last_fetch_at"] == to_iso(fixed_now)

    fake = FakeX()
    run_collector(conn, watchlist, fake, fixed_now + timedelta(days=1))
    assert [r.url.params["usernames"] for r in fake.user_requests] == ["nosuchuser_x1"]


def test_users_by_is_called_in_batches_of_100(conn, watchlist, fixed_now):
    handles = [f"acct{i:03d}" for i in range(150)]
    watchlist.x.accounts = [Account(handle=h, category="influencer") for h in handles]
    users = {
        "data": [{"id": str(5000 + i), "name": h, "username": h} for i, h in enumerate(handles)],
        "errors": USERS["errors"],
    }
    sync_accounts(conn, watchlist)
    fake = FakeX(users=users)
    counts = run_collector(conn, watchlist, fake, fixed_now)

    assert [len(r.url.params["usernames"].split(",")) for r in fake.user_requests] == [100, 50]
    assert counts["unresolved_handles"] == []
    assert sum(usage_by_handle(conn, "user_read").values()) == 150
    assert all(len(r.url.params["query"]) <= 512 for r in fake.search_requests)


# ---------------------------------------------------------------- HTTP failure handling


def _rate_limited(now: datetime, wait_s: int) -> httpx.Response:
    return httpx.Response(
        429,
        json=load("x_error_429.json"),
        headers={
            "x-rate-limit-limit": "450",
            "x-rate-limit-remaining": "0",
            "x-rate-limit-reset": str(int(now.timestamp()) + wait_s),
            "date": format_datetime(now, usegmt=True),
        },
    )


def test_rate_limit_short_wait_sleeps_and_retries_once(conn, watchlist, fixed_now):
    sync_accounts(conn, watchlist)
    preresolve(conn, fixed_now)
    pages = paged({cashtag_query(watchlist): [PAGE1, PAGE2]})
    calls = itertools.count()

    def search(request, query):
        return _rate_limited(fixed_now, 120) if next(calls) == 0 else pages(request, query)

    sleeps: list[float] = []
    counts = run_collector(conn, watchlist, FakeX(search), fixed_now, sleeps=sleeps)

    assert len(sleeps) == 1 and 120 <= sleeps[0] <= 122
    assert counts["status"] == "ok"
    assert counts["posts_read"] == 5
    assert counts["watermark_advanced"] is True


def test_rate_limit_pause_moves_start_time_with_the_window(conn, watchlist, fixed_now):
    """A 10-minute rate-limit pause mid-pagination on a first run: the retry re-clamps start_time to the 7-day window
    as it stands after the pause, drops the next_token issued for the old window and resumes below the posts read."""
    sync_accounts(conn, watchlist)
    preresolve(conn, fixed_now)
    sleeps: list[float] = []

    def clock() -> datetime:
        return fixed_now + timedelta(seconds=sum(sleeps))

    stream = PostStream(evenly(fixed_now - timedelta(days=8), fixed_now, 50), clock)
    calls = itertools.count()

    def search(request, query):
        return _rate_limited(clock(), 600) if next(calls) == 1 else stream(request, query)

    fake = FakeX(search)
    counts = run_collector(conn, watchlist, fake, fixed_now, sleeps=sleeps)

    assert stream.rejected == []
    assert len(sleeps) == 1 and 600 <= sleeps[0] <= 602
    first, limited, retry = (r.url.params for r in fake.search_requests[:3])
    assert first["query"] == limited["query"] == retry["query"]
    assert "next_token" in limited and "next_token" not in retry
    later_floor = fixed_now + timedelta(seconds=sleeps[0]) - timedelta(days=7) + timedelta(minutes=5)
    assert retry["start_time"] == to_iso(later_floor)
    window = sorted(t for _, t in stream.posts if parse_api_time(first["start_time"]) <= t < fixed_now)
    oldest_read = window[-int(first["max_results"])]
    assert retry["end_time"] == to_iso(oldest_read + timedelta(seconds=1))
    assert stream.ids_between(later_floor, fixed_now) <= stored_ids(conn)
    assert counts["status"] == "ok" and counts["caught_up"] is True


def test_rate_limit_retries_only_once(conn, watchlist, fixed_now):
    sync_accounts(conn, watchlist)
    preresolve(conn, fixed_now)
    fake = FakeX(lambda request, query: _rate_limited(fixed_now, 60))
    sleeps: list[float] = []
    counts = run_collector(conn, watchlist, fake, fixed_now, sleeps=sleeps)

    assert len(sleeps) == 1
    assert len(fake.search_requests) == 2
    assert counts["status"] == "partial"


def test_rate_limit_long_wait_stops_partial(conn, watchlist, fixed_now):
    sync_accounts(conn, watchlist)
    preresolve(conn, fixed_now)
    fake = FakeX(lambda request, query: _rate_limited(fixed_now, 3600))
    sleeps: list[float] = []
    counts = run_collector(conn, watchlist, fake, fixed_now, sleeps=sleeps)

    assert sleeps == []
    assert len(fake.search_requests) == 1
    assert counts["status"] == "partial"
    assert "rate limit" in counts["error"]
    assert db.get_watermark(conn, "x", "search_end_time") is None


def test_rate_limit_without_reset_header_stops_without_waiting(conn, watchlist, fixed_now):
    sync_accounts(conn, watchlist)
    preresolve(conn, fixed_now)
    fake = FakeX(lambda request, query: httpx.Response(429, json=load("x_error_429.json")))
    counts = run_collector(conn, watchlist, fake, fixed_now)

    assert len(fake.search_requests) == 1
    assert counts["status"] == "partial"
    assert "rate limit" in counts["error"]


def test_usage_cap_stops_without_waiting(conn, watchlist, fixed_now):
    sync_accounts(conn, watchlist)
    preresolve(conn, fixed_now)
    capped = httpx.Response(429, json=load("x_error_429_usage_capped.json"))
    fake = FakeX(lambda request, query: capped)
    counts = run_collector(conn, watchlist, fake, fixed_now)

    assert len(fake.search_requests) == 1
    assert counts["status"] == "partial"
    assert "usage cap" in counts["error"]


def test_has_cashtags_rejection_switches_to_explicit_mode(conn, watchlist, fixed_now):
    sync_accounts(conn, watchlist)
    explicit_tsla = next(q for q in real_queries(watchlist, "explicit") if "$TSLA" in q)
    pages = paged({explicit_tsla: [PAGE1, PAGE2]})

    def search(request, query):
        if "has:cashtags" in query:
            return httpx.Response(400, json=load("x_error_400_has_cashtags.json"))
        return pages(request, query)

    fake = FakeX(search)
    counts = run_collector(conn, watchlist, fake, fixed_now)

    assert counts["query_mode"] == "explicit"
    assert db.get_watermark(conn, "x", "query_mode") == "explicit"
    assert sum("has:cashtags" in r.url.params["query"] for r in fake.search_requests) == 1
    assert counts["status"] == "ok"
    assert counts["posts_read"] == 5
    assert counts["watermark_advanced"] is True
    sent = {r.url.params["query"] for r in fake.search_requests}
    assert set(real_queries(watchlist, "explicit")) <= sent

    fake = FakeX(search)
    counts = run_collector(conn, watchlist, fake, fixed_now + timedelta(days=1))
    assert counts["query_mode"] == "explicit"
    assert not any("has:cashtags" in r.url.params["query"] for r in fake.search_requests)
    assert fake.search_requests and "$" in fake.search_requests[0].url.params["query"]


def test_unrelated_400_echoing_the_query_keeps_cashtags_mode(conn, watchlist, fixed_now):
    sync_accounts(conn, watchlist)
    preresolve(conn, fixed_now)
    target = cashtag_query(watchlist)
    rejected = {
        "errors": [{"parameters": {"query": [target]}, "message": "There were errors processing your request: bad"}],
        "title": "Invalid Request",
        "detail": "One or more parameters to your request was invalid.",
        "type": "https://api.x.com/2/problems/invalid-request",
    }

    def search(request, query):
        return httpx.Response(400, json=rejected) if query == target else ok(EMPTY)

    fake = FakeX(search)
    counts = run_collector(conn, watchlist, fake, fixed_now)

    assert counts["query_mode"] == "cashtags"
    assert db.get_watermark(conn, "x", "query_mode") is None
    assert counts["status"] == "partial"
    assert {r.url.params["query"] for r in fake.search_requests} == set(real_queries(watchlist))


@pytest.mark.parametrize("where", ["users", "search"])
@pytest.mark.parametrize("status", [401, 402, 403])
def test_auth_failures_return_error_without_raising(conn, watchlist, fixed_now, where, status):
    sync_accounts(conn, watchlist)
    if where == "search":
        preresolve(conn, fixed_now)
    body = {
        401: load("x_error_401.json"),
        402: load("x_error_402_credits_depleted.json"),
        403: {"title": "Forbidden", "detail": "Forbidden", "status": 403},
    }[status]
    denied = httpx.Response(status, json=body)
    fake = FakeX(lambda request, query: denied, lookup=(lambda names: denied) if where == "users" else None)

    counts = run_collector(conn, watchlist, fake, fixed_now)

    assert counts["status"] == "error"
    assert str(status) in counts["error"]
    assert len(fake.search_requests) == (0 if where == "users" else 1)
    assert db.get_watermark(conn, "x", "search_end_time") is None


def test_missing_token_is_an_error_without_requests(conn, watchlist, fixed_now):
    sync_accounts(conn, watchlist)
    fake = FakeX()
    counts = run_collector(conn, watchlist, fake, fixed_now, token="")
    assert counts["status"] == "error"
    assert fake.requests == []


def test_server_errors_retry_with_backoff_then_partial(conn, watchlist, fixed_now):
    sync_accounts(conn, watchlist)
    preresolve(conn, fixed_now)
    fake = FakeX(lambda request, query: httpx.Response(503, json=load("x_error_503.json")))
    sleeps: list[float] = []
    counts = run_collector(conn, watchlist, fake, fixed_now, sleeps=sleeps)

    assert len(fake.search_requests) == 3
    assert len(sleeps) == 2 and sleeps[0] < sleeps[1]
    assert counts["status"] == "partial"
    assert db.get_watermark(conn, "x", "search_end_time") is None


def test_timeouts_are_retried_and_recover(conn, watchlist, fixed_now):
    sync_accounts(conn, watchlist)
    preresolve(conn, fixed_now)
    calls = itertools.count()

    def search(request, query):
        if next(calls) < 2:
            raise httpx.ReadTimeout("timed out", request=request)
        return ok(EMPTY)

    sleeps: list[float] = []
    counts = run_collector(conn, watchlist, FakeX(search), fixed_now, sleeps=sleeps)

    assert len(sleeps) == 2
    assert counts["status"] == "ok"
    assert counts["watermark_advanced"] is True
