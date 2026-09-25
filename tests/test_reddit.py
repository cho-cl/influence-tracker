from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from influence_tracker import ingest
from influence_tracker.collectors.reddit_rss import collect_reddit, extract_selftext, parse_feed
from influence_tracker.config import Watchlist
from influence_tracker.models import MatchResult, Mention, Post
from influence_tracker.timeutil import utc_now

BOILERPLATE = ("submitted by", "[link]", "[comments]", "SC_OFF", "SC_ON", "&#32;", "&amp;", "<div", "<a ")
TWO_SUBS = ["wallstreetbets", "stocks"]


class FakeMatcher:
    """Tags posts that mention Nvidia, so with_mentions is exercised without the real matcher."""

    def match(self, text: str, platform: str, cashtag_hints: Sequence[str] = ()) -> MatchResult:
        if "nvidia" in text.lower():
            return MatchResult(mentions=[Mention("NVDA", "name", "Nvidia")])
        return MatchResult()


class SpySink(ingest.PostSink):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.pages: list[list[Post]] = []

    def store(self, posts: Sequence[Post]) -> ingest.StoreResult:
        self.pages.append(list(posts))
        return super().store(posts)


class FakeReddit:
    """Serves scripted responses per subreddit, in order, and records every request."""

    def __init__(self, script: dict[str, list[httpx.Response | Exception]]) -> None:
        self.script = {sub: list(items) for sub, items in script.items()}
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        sub = request.url.path.split("/")[2]
        item = self.script[sub].pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self))


class Sleeps(list):
    def __call__(self, seconds: float) -> None:
        self.append(seconds)


def rl_headers(remaining: str | None, reset: str | None) -> dict[str, str]:
    headers = {"content-type": "application/atom+xml; charset=UTF-8"}
    if remaining is not None:
        headers["x-ratelimit-used"] = "1"
        headers["x-ratelimit-remaining"] = remaining
    if reset is not None:
        headers["x-ratelimit-reset"] = reset
    return headers


def resp(content: bytes, status: int = 200, remaining: str | None = "5.0", reset: str | None = "40") -> httpx.Response:
    return httpx.Response(status, content=content, headers=rl_headers(remaining, reset))


def with_subs(watchlist: Watchlist, subs: list[str], **extra) -> Watchlist:
    return watchlist.model_copy(update={"reddit": watchlist.reddit.model_copy(update={"subreddits": subs, **extra})})


def run(conn, watchlist, sink, now, fake: FakeReddit, subs: list[str] = TWO_SUBS, sleep: Sleeps | None = None) -> dict:
    sleep = Sleeps() if sleep is None else sleep
    return collect_reddit(conn, with_subs(watchlist, subs), sink, 1, now, client=fake.client(), sleep=sleep)


@pytest.fixture
def wsb_xml(fixtures_dir) -> bytes:
    return (fixtures_dir / "reddit_top_wallstreetbets.xml").read_bytes()


@pytest.fixture
def stocks_xml(fixtures_dir) -> bytes:
    return (fixtures_dir / "reddit_top_stocks.xml").read_bytes()


@pytest.fixture
def sink(conn, watchlist, fixed_now) -> SpySink:
    ingest.sync_accounts(conn, watchlist)
    return SpySink(conn, FakeMatcher(), now=lambda: fixed_now)


# ---------------------------------------------------------------- parsing the recorded feeds


def test_parse_wallstreetbets_fixture(wsb_xml):
    posts = parse_feed(wsb_xml, "wallstreetbets")
    assert len(posts) == 25
    assert [p.feed_rank for p in posts] == list(range(1, 26))
    assert len({p.native_id for p in posts}) == 25
    first = posts[0]
    assert first.platform == "reddit"
    assert first.native_id == "t3_1wo92yv"
    assert first.author == "Apprehensive-Koala18"
    assert first.url == "https://www.reddit.com/r/wallstreetbets/comments/1wo92yv/bears_saw_it_coming/"
    assert first.source == "wallstreetbets"
    assert first.created_at_utc == datetime(2026, 9, 23, 15, 25, 56, tzinfo=UTC)
    for p in posts:
        assert p.created_at_utc.utcoffset() == timedelta(0)
        assert p.native_id.startswith("t3_")
        assert p.author and not p.author.startswith("/u/")


def test_parse_stocks_fixture(stocks_xml):
    posts = parse_feed(stocks_xml, "stocks")
    # Entry 8 is AutoModerator's daily thread; it is dropped but the others keep their true feed positions.
    assert len(posts) == 11
    assert [p.feed_rank for p in posts] == [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12]
    assert "t3_1wox1e0" not in {p.native_id for p in posts}
    assert posts[0].native_id == "t3_1wohx89"
    assert posts[-1].native_id == "t3_1wot2j1"
    assert all(p.source == "stocks" for p in posts)


def test_feed_boilerplate_never_reaches_text(wsb_xml, stocks_xml):
    posts = parse_feed(wsb_xml, "wallstreetbets") + parse_feed(stocks_xml, "stocks")
    for p in posts:
        for marker in BOILERPLATE:
            assert marker not in p.text, (p.native_id, marker)
        assert f"/u/{p.author}" not in p.text


def test_link_post_is_title_only(wsb_xml):
    link_post = parse_feed(wsb_xml, "wallstreetbets")[1]
    assert link_post.text == "10-year Treasury yield leaps to fresh 19-year high after hot economic readings"


def test_image_post_keeps_its_caption(wsb_xml):
    assert parse_feed(wsb_xml, "wallstreetbets")[0].text == "BEARS SAW IT COMING\n\nBear omniscience strikes again"


def test_text_post_has_unescaped_selftext(stocks_xml):
    post = next(p for p in parse_feed(stocks_xml, "stocks") if p.native_id == "t3_1wp1dob")
    title, body = post.text.split("\n\n", 1)
    assert title == "Apple and Nvidia are taking up more of the S&P 500"
    assert body.startswith("The S&P 500 has 500 companies, but a relatively small group is driving")
    assert "Apple and Nvidia now account for more than 15% of the S&P 500" in body
    assert len(body.split("\n")) == 5  # one line per <p>


def test_selftext_links_keep_their_text(wsb_xml):
    post = next(p for p in parse_feed(wsb_xml, "wallstreetbets") if p.native_id == "t3_1wojlv7")
    assert "internal document detailing more than 100 investment-banking deals" in post.text
    assert "in Asia, Bloomberg reports" in post.text


def test_old_reddit_placeholder_is_not_selftext(wsb_xml):
    post = next(p for p in parse_feed(wsb_xml, "wallstreetbets") if p.native_id == "t3_1woxizn")
    assert post.text == "Daily Discussion Thread for September 24, 2026"


def test_extract_selftext_ignores_everything_outside_md_div():
    content = (
        '<table><tr><td><a href="x"><img alt="t" src="y" /></a></td><td>'
        '<!-- SC_OFF --><div class="md"><p>Buy &amp; hold <strong>$NVDA</strong></p>'
        "<ul><li>one</li><li>two</li></ul><div><p>nested &lt;tag&gt;</p></div></div><!-- SC_ON -->"
        ' &#32; submitted by &#32; <a href="u"> /u/someone </a><br/>'
        '<span><a href="l">[link]</a></span> &#32; <span><a href="c">[comments]</a></span></td></tr></table>'
    )
    assert extract_selftext(content) == "Buy & hold $NVDA\none\ntwo\nnested <tag>"


def test_extract_selftext_without_md_div_is_empty():
    assert extract_selftext('<a href="u"> /u/x </a> <span><a href="l">[link]</a></span>') == ""


def test_entry_without_author_is_deleted():
    xml = (
        b'<?xml version="1.0" encoding="UTF-8"?><feed xmlns="http://www.w3.org/2005/Atom">'
        b'<entry><id>t3_abc</id><link href="https://www.reddit.com/r/stocks/comments/abc/x/" />'
        b"<published>2026-09-24T01:02:03+00:00</published><title>Hello</title></entry></feed>"
    )
    [post] = parse_feed(xml, "stocks")
    assert post.author == "[deleted]"
    assert post.text == "Hello"
    assert post.created_at_utc == datetime(2026, 9, 24, 1, 2, 3, tzinfo=UTC)


def test_entry_without_timestamp_is_skipped_but_ranks_keep_positions():
    xml = (
        b'<?xml version="1.0" encoding="UTF-8"?><feed xmlns="http://www.w3.org/2005/Atom">'
        b'<entry><id>t3_a</id><link href="https://r/a" /><title>no time</title></entry>'
        b'<entry><id>t3_b</id><link href="https://r/b" /><published>2026-09-24T01:02:03+00:00</published>'
        b"<title>ok</title></entry></feed>"
    )
    [post] = parse_feed(xml, "stocks")
    assert post.native_id == "t3_b"
    assert post.feed_rank == 2


# ---------------------------------------------------------------- collecting


def test_collect_stores_each_feed_once(conn, watchlist, sink, fixed_now, wsb_xml, stocks_xml):
    fake = FakeReddit({"wallstreetbets": [resp(wsb_xml)], "stocks": [resp(stocks_xml)]})
    sleeps = Sleeps()

    counts = run(conn, watchlist, sink, fixed_now, fake, sleep=sleeps)

    expected_mentions = sum(
        "nvidia" in p.text.lower() for p in parse_feed(wsb_xml, "wallstreetbets") + parse_feed(stocks_xml, "stocks")
    )
    assert expected_mentions > 0
    assert counts == {
        "status": "ok",
        "requests": 2,
        "stored": 36,
        "new": 36,
        "with_mentions": expected_mentions,
        "rate_limited": 0,
        "subreddits_failed": 0,
    }
    assert sleeps == []
    assert [len(page) for page in sink.pages] == [25, 11]
    row = conn.execute("SELECT * FROM posts WHERE platform = 'reddit' AND native_id = 't3_1wo92yv'").fetchone()
    assert row["author"] == "Apprehensive-Koala18"
    assert row["created_at_utc"] == "2026-09-23T15:25:56Z"
    assert row["source"] == "wallstreetbets"
    assert row["feed_rank"] == 1
    assert row["collected_at"] == "2026-09-24T22:30:00Z"
    n_mentions = conn.execute("SELECT COUNT(*) FROM mentions WHERE platform = 'reddit'").fetchone()[0]
    assert n_mentions == expected_mentions


def test_collect_request_shape(conn, watchlist, sink, fixed_now, stocks_xml):
    fake = FakeReddit({"stocks": [resp(stocks_xml)]})
    run(conn, watchlist, sink, fixed_now, fake, subs=["stocks"])
    [request] = fake.requests
    assert request.url.host == "www.reddit.com"
    assert request.url.path == "/r/stocks/top/.rss"
    assert request.url.params["t"] == "day"
    assert request.url.params["limit"] == str(watchlist.reddit.rss_limit)
    assert request.headers["user-agent"] == watchlist.reddit.user_agent


def test_uses_the_real_watchlist_subreddits_in_order(conn, watchlist, sink, fixed_now, stocks_xml):
    fake = FakeReddit({sub: [resp(stocks_xml)] for sub in watchlist.reddit.subreddits})
    counts = collect_reddit(conn, watchlist, sink, 1, fixed_now, client=fake.client(), sleep=Sleeps())
    assert [r.url.path.split("/")[2] for r in fake.requests] == watchlist.reddit.subreddits
    assert counts["status"] == "ok"


def test_rerun_is_idempotent(conn, watchlist, sink, fixed_now, stocks_xml):
    for _ in range(2):
        counts = run(conn, watchlist, sink, fixed_now, FakeReddit({"stocks": [resp(stocks_xml)]}), subs=["stocks"])
    assert counts["stored"] == 11
    assert counts["new"] == 0
    assert conn.execute("SELECT COUNT(*) FROM posts").fetchone()[0] == 11


def test_exhausted_quota_waits_for_reset_before_next_subreddit(conn, watchlist, sink, fixed_now, wsb_xml, stocks_xml):
    fake = FakeReddit({"wallstreetbets": [resp(wsb_xml, remaining="0.0", reset="52")], "stocks": [resp(stocks_xml)]})
    sleeps = Sleeps()
    counts = run(conn, watchlist, sink, fixed_now, fake, sleep=sleeps)
    assert sleeps == [54.0]
    assert counts["status"] == "ok"
    assert counts["requests"] == 2


def test_missing_ratelimit_headers_default_to_62s(conn, watchlist, sink, fixed_now, wsb_xml, stocks_xml):
    fake = FakeReddit({"wallstreetbets": [resp(wsb_xml, remaining=None, reset=None)], "stocks": [resp(stocks_xml)]})
    sleeps = Sleeps()
    run(conn, watchlist, sink, fixed_now, fake, sleep=sleeps)
    assert sleeps == [62.0]


def test_remaining_quota_means_no_wait(conn, watchlist, sink, fixed_now, wsb_xml, stocks_xml):
    fake = FakeReddit({"wallstreetbets": [resp(wsb_xml, remaining="3.0", reset="52")], "stocks": [resp(stocks_xml)]})
    sleeps = Sleeps()
    run(conn, watchlist, sink, fixed_now, fake, sleep=sleeps)
    assert sleeps == []


def test_absurd_reset_header_is_capped(conn, watchlist, sink, fixed_now, wsb_xml, stocks_xml):
    fake = FakeReddit({"wallstreetbets": [resp(wsb_xml, remaining="0", reset="99999")], "stocks": [resp(stocks_xml)]})
    sleeps = Sleeps()
    run(conn, watchlist, sink, fixed_now, fake, sleep=sleeps)
    assert sleeps == [600.0]


def test_empty_body_retries_once_then_fails_the_subreddit(conn, watchlist, sink, fixed_now, stocks_xml):
    fake = FakeReddit(
        {
            "wallstreetbets": [resp(b"", remaining="0.0", reset="40"), resp(b"", remaining="0.0", reset="35")],
            "stocks": [resp(stocks_xml)],
        }
    )
    sleeps = Sleeps()
    counts = run(conn, watchlist, sink, fixed_now, fake, sleep=sleeps)
    # retry wait = the empty response's reset; then the usual reset + 2 before the next subreddit
    assert sleeps == [40.0, 37.0]
    assert [len(page) for page in sink.pages] == [11]
    assert counts == {
        "status": "partial",
        "requests": 3,
        "stored": 11,
        "new": 11,
        "with_mentions": sink.totals.with_mentions,
        "rate_limited": 1,
        "subreddits_failed": 1,
    }


def test_429_then_success_on_retry(conn, watchlist, sink, fixed_now, stocks_xml):
    fake = FakeReddit({"stocks": [resp(b"Too Many Requests", status=429, reset=None), resp(stocks_xml)]})
    sleeps = Sleeps()
    counts = run(conn, watchlist, sink, fixed_now, fake, subs=["stocks"], sleep=sleeps)
    assert sleeps == [60.0]
    assert counts["status"] == "ok"
    assert counts["requests"] == 2
    assert counts["rate_limited"] == 0
    assert counts["stored"] == 11


def test_html_page_instead_of_feed_counts_as_rate_limited(conn, watchlist, sink, fixed_now, stocks_xml):
    html_page = b"<!doctype html><html><body><h1>whoa there, pardner!</h1></body></html>"
    fake = FakeReddit(
        {"wallstreetbets": [resp(html_page, reset="20"), resp(html_page, reset="20")], "stocks": [resp(stocks_xml)]}
    )
    counts = run(conn, watchlist, sink, fixed_now, fake)
    assert counts["rate_limited"] == 1
    assert counts["subreddits_failed"] == 1
    assert counts["status"] == "partial"


def test_http_error_fails_subreddit_without_retry(conn, watchlist, sink, fixed_now, stocks_xml):
    fake = FakeReddit({"wallstreetbets": [resp(b"Forbidden", status=403)], "stocks": [resp(stocks_xml)]})
    counts = run(conn, watchlist, sink, fixed_now, fake)
    assert counts["requests"] == 2
    assert counts["rate_limited"] == 0
    assert counts["subreddits_failed"] == 1
    assert counts["status"] == "partial"
    assert counts["stored"] == 11


def test_transport_error_does_not_raise(conn, watchlist, sink, fixed_now, stocks_xml):
    fake = FakeReddit({"wallstreetbets": [httpx.ConnectTimeout("timed out")], "stocks": [resp(stocks_xml)]})
    sleeps = Sleeps()
    counts = run(conn, watchlist, sink, fixed_now, fake, sleep=sleeps)
    assert counts["subreddits_failed"] == 1
    assert counts["status"] == "partial"
    assert counts["stored"] == 11
    assert sleeps == [62.0]  # no rate-limit headers to go on, so the conservative default


def test_every_subreddit_failing_is_an_error(conn, watchlist, sink, fixed_now):
    fake = FakeReddit({"wallstreetbets": [resp(b"gone", status=403)], "stocks": [resp(b"gone", status=403)]})
    counts = run(conn, watchlist, sink, fixed_now, fake)
    assert counts["status"] == "error"
    assert counts["subreddits_failed"] == 2
    assert sink.pages == []


@pytest.mark.live
def test_live_single_subreddit(conn, watchlist):
    ingest.sync_accounts(conn, watchlist)
    live_sink = ingest.PostSink(conn, FakeMatcher())
    counts = collect_reddit(conn, with_subs(watchlist, ["stocks"], rss_limit=5), live_sink, 0, utc_now())
    assert counts["status"] == "ok", counts
    assert counts["stored"] > 0
