from __future__ import annotations

import html
import re
import time

import feedparser
import pytest

from influence_tracker.config import Ticker
from influence_tracker.mentions import MentionMatcher
from influence_tracker.models import MatchResult, Mention


@pytest.fixture
def matcher(watchlist) -> MentionMatcher:
    return MentionMatcher(watchlist)


def pairs(result: MatchResult) -> set[tuple[str, str]]:
    return {(m.ticker, m.match_type) for m in result.mentions}


def tickers(result: MatchResult) -> set[str]:
    return {m.ticker for m in result.mentions}


# ---------------------------------------------------------------- cashtags


def test_dollar_amounts_are_not_cashtags(matcher):
    r = matcher.match("Paid $5, then $5.00, then $1B and US$20 for it", "x")
    assert r.mentions == []
    assert r.unknown_cashtags == []


def test_cashtag_matches(matcher):
    r = matcher.match("$5 says $TSLA goes up", "x")
    assert r.mentions == [Mention("TSLA", "cashtag", "$TSLA")]


def test_cashtag_is_case_insensitive(matcher):
    r = matcher.match("loading up on $tsla", "x")
    assert r.mentions == [Mention("TSLA", "cashtag", "$tsla")]


def test_cashtag_trailing_punctuation_and_possessives(matcher):
    r = matcher.match("Sold $TSLA. Bought $AAPL's dip ($NVDA) and $AMD’s too, $MSFT!", "x")
    assert tickers(r) == {"TSLA", "AAPL", "NVDA", "AMD", "MSFT"}
    assert all(m.match_type == "cashtag" for m in r.mentions)


def test_cashtag_class_suffix(matcher):
    assert matcher.match("Buffett's $BRK.B keeps compounding", "x").mentions == [Mention("BRK.B", "cashtag", "$BRK.B")]


@pytest.mark.parametrize(
    ("text", "ticker"),
    [("$GOOG", "GOOGL"), ("$BRK.A", "BRK.B"), ("$FB", "META"), ("$SQ", "XYZ"), ("$brk.a", "BRK.B")],
)
def test_cashtag_aliases_normalize_to_canonical(matcher, text, ticker):
    r = matcher.match(f"watching {text} today", "x")
    assert r.mentions == [Mention(ticker, "cashtag", text)]
    assert r.unknown_cashtags == []


def test_benchmarks_are_in_the_universe(matcher):
    assert pairs(matcher.match("$SPY and $QQQ both red", "x")) == {("SPY", "cashtag"), ("QQQ", "cashtag")}


def test_dollar_glued_to_a_word_or_dollar_is_not_a_cashtag(matcher):
    r = matcher.match("price$TSLA and $$AAPL", "x")
    assert r.mentions == []
    assert r.unknown_cashtags == []


def test_six_letter_cashtag_is_ignored(matcher):
    r = matcher.match("$TSLAQQ is not a thing", "x")
    assert r.mentions == []
    assert r.unknown_cashtags == []


def test_sentence_period_glued_to_cashtag_falls_back_to_base_symbol(matcher):
    assert pairs(matcher.match("I sold $TSLA.I regret nothing", "x")) == {("TSLA", "cashtag")}


def test_unknown_cashtags_are_collected_not_matched(matcher):
    r = matcher.match("$ZZZZ to the moon, $zzzz again, and $ABC", "x")
    assert r.mentions == []
    assert r.unknown_cashtags == ["ABC", "ZZZZ"]


def test_unknown_cashtag_that_spells_a_name_still_matches_the_name(matcher):
    r = matcher.match("$Tesla deliveries beat", "x")
    assert r.mentions == [Mention("TSLA", "name", "Tesla")]
    assert r.unknown_cashtags == ["TESLA"]


def test_known_cashtag_is_not_also_a_name_match(matcher):
    assert pairs(matcher.match("$META and $AMD earnings", "x")) == {("META", "cashtag"), ("AMD", "cashtag")}


def test_cashtag_hints_are_added(matcher):
    r = matcher.match("great quarter", "x", ["TSLA"])
    assert r.mentions == [Mention("TSLA", "cashtag", "$TSLA")]


def test_cashtag_hints_normalize_aliases_and_collect_unknowns(matcher):
    r = matcher.match("big day", "x", ["$goog", "BTC", "WAYTOOLONG", ""])
    assert r.mentions == [Mention("GOOGL", "cashtag", "$goog")]
    assert r.unknown_cashtags == ["BTC"]


def test_cashtag_hint_duplicate_keeps_text_surface_form(matcher):
    r = matcher.match("$tsla ripping", "x", ["TSLA"])
    assert r.mentions == [Mention("TSLA", "cashtag", "$tsla")]


@pytest.mark.parametrize("text", ["$BRK-B", "$BRK/B", "$brk-b", "$BRK"])
def test_class_share_cashtag_spellings(matcher, text):
    r = matcher.match(f"{text} up big", "x")
    assert r.mentions == [Mention("BRK.B", "cashtag", text)]
    assert r.unknown_cashtags == []


def test_class_share_separator_in_hints_and_unknowns(matcher):
    assert matcher.match("big day", "x", ["BRK-B"]).mentions == [Mention("BRK.B", "cashtag", "$BRK-B")]
    assert matcher.match("$ZZZ-B and $ZZZ/B", "x").unknown_cashtags == ["ZZZ.B"]


def test_slash_between_cashtags_is_not_a_class_suffix(matcher):
    r = matcher.match("$SPY/QQQ ratio and $AAPL-based funds", "x")
    assert pairs(r) == {("SPY", "cashtag"), ("AAPL", "cashtag")}
    assert r.unknown_cashtags == []


def test_class_less_cashtag_resolves_only_when_one_ticker_owns_the_base(watchlist):
    custom = watchlist.model_copy(
        update={
            "tickers": [
                Ticker(symbol="SPY", benchmark=True),
                Ticker(symbol="BF.B", aliases=["BF.A"]),
                Ticker(symbol="FOO.A"),
                Ticker(symbol="FOO.B"),
            ]
        }
    )
    m = MentionMatcher(custom)
    assert m.match("$BF is cheap", "x").mentions == [Mention("BF.B", "cashtag", "$BF")]
    r = m.match("$FOO is cheap", "x")
    assert r.mentions == []
    assert r.unknown_cashtags == ["FOO"]


# ---------------------------------------------------------------- names


def test_name_matches_case_insensitively_with_possessive(matcher):
    assert matcher.match("Tesla's deliveries", "x").mentions == [Mention("TSLA", "name", "Tesla")]
    assert matcher.match("i love my nvidia gpu", "x").mentions == [Mention("NVDA", "name", "nvidia")]


def test_name_must_be_bounded_by_non_alphanumerics(matcher):
    assert matcher.match("Teslaville and Googleplex tours", "x").mentions == []
    assert matcher.match("myTesla app, 2Nvidia, SuperMicrosoft", "x").mentions == []
    assert pairs(matcher.match("(Tesla)/#Nvidia/@Microsoft_", "x")) == {
        ("TSLA", "name"),
        ("NVDA", "name"),
        ("MSFT", "name"),
    }


def test_non_ascii_names_match_in_any_case(watchlist):
    custom = watchlist.model_copy(
        update={
            "tickers": [
                Ticker(symbol="SPY", benchmark=True),
                Ticker(symbol="NSRGY", names=["Nestlé"]),
                Ticker(symbol="ESLOY", ambiguous_names=["Élan"]),
                Ticker(symbol="GRB", names=["Großbank"]),
            ]
        }
    )
    m = MentionMatcher(custom)
    assert pairs(m.match("NESTLÉ and nestlé", "x")) == {("NSRGY", "name")}
    assert m.match("Nestle without the accent", "x").mentions == []
    assert pairs(m.match("ÉLAN shares slide", "x")) == {("ESLOY", "name")}
    assert m.match("élan shares slide", "x").mentions == []
    # casefold() would turn the configured "ß" into "ss"; the name must still match itself as written.
    assert m.match("Großbank earnings", "x").mentions == [Mention("GRB", "name", "Großbank")]
    assert m.match("GROßBANK earnings", "x").mentions == [Mention("GRB", "name", "GROßBANK")]


@pytest.mark.parametrize(
    ("text", "ticker", "surface"),
    [
        ("AT&T raises its dividend", "T", "AT&T"),
        ("AT&T's network was down", "T", "AT&T"),
        ("McDonald’s Investor Day", "MCD", "McDonald’s"),
        ("McDonald's Investor Day", "MCD", "McDonald's"),
        ("J.P. Morgan upgrades the sector", "JPM", "J.P. Morgan"),
        ("Coca-Cola beats", "KO", "Coca-Cola"),
        ("Coca Cola beats", "KO", "Coca Cola"),
        ("Coca‑Cola beats", "KO", "Coca‑Cola"),
        ("Coca‐Cola beats", "KO", "Coca‐Cola"),
        ("Johnson & Johnson settles", "JNJ", "Johnson & Johnson"),
        ("Bank of\nAmerica reports", "BAC", "Bank of\nAmerica"),
    ],
)
def test_punctuated_names(matcher, text, ticker, surface):
    assert matcher.match(text, "truthsocial").mentions == [Mention(ticker, "name", surface)]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("AT&amp;T to cut 10,000 jobs", [Mention("T", "name", "AT&T")]),
        ("Johnson &amp; Johnson talc settlement", [Mention("JNJ", "name", "Johnson & Johnson")]),
        ("S&amp;P 500 hits record, Apple leads", [Mention("AAPL", "name", "Apple")]),
    ],
)
def test_html_entities_from_x_are_decoded(matcher, text, expected):
    assert matcher.match(text, "x").mentions == expected


def test_longest_name_wins(matcher):
    r = matcher.match("Goldman Sachs and Lucid Motors", "x")
    assert r.mentions == [Mention("GS", "name", "Goldman Sachs"), Mention("LCID", "name", "Lucid Motors")]


def test_names_inside_urls_are_ignored(matcher):
    r = matcher.match("see https://www.google.com/search?q=tesla+stock and www.netflix.com/title", "x")
    assert r.mentions == []
    assert matcher.match("VISIT WWW.TESLA.COM OR HTTPS://NVIDIA.COM TODAY", "truthsocial").mentions == []
    assert matcher.match("Tesla https://t.co/abc", "x").mentions == [Mention("TSLA", "name", "Tesla")]


# ---------------------------------------------------------------- ambiguous names


def test_apple_pie_is_not_apple_stock(matcher):
    assert matcher.match("Apple pie recipe", "x").mentions == []


def test_apple_with_finance_context(matcher):
    r = matcher.match("Apple earnings beat, shares up 5%", "x")
    assert r.mentions == [Mention("AAPL", "name", "Apple")]


def test_ambiguous_name_must_be_capitalized(matcher):
    assert matcher.match("apple earnings beat, shares up 5%", "x").mentions == []
    assert matcher.match("APPLE STOCK IS FLYING", "truthsocial").mentions == [Mention("AAPL", "name", "APPLE")]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Super Micro wins a huge server order", [Mention("SMCI", "name", "Super Micro")]),
        ("General Dynamics lands a Navy contract", [Mention("GD", "name", "General Dynamics")]),
        ("TRUMP MEDIA IS DOING GREAT", [Mention("DJT", "name", "TRUMP MEDIA")]),
        ("these super micro-cap names are gambling", []),
        ("the general dynamics of this market are weird", []),
        ("constant trump media coverage", []),
    ],
)
def test_cased_names_need_capitals_but_no_context(matcher, text, expected):
    assert matcher.match(text, "reddit").mentions == expected


def test_ambiguous_name_uses_first_qualifying_occurrence(matcher):
    r = matcher.match("an apple a day; Apple shares up", "x")
    assert r.mentions == [Mention("AAPL", "name", "Apple")]


def test_intel_lowercase_is_not_a_mention(matcher):
    assert matcher.match("need more intel on this", "x").mentions == []
    assert matcher.match("need more intel on this stock", "x").mentions == []


def test_intel_with_context(matcher):
    assert matcher.match("Intel shares plunge after guidance cut", "x").mentions == [Mention("INTC", "name", "Intel")]


def test_intel_capitalized_without_context(matcher):
    assert matcher.match("Senate Intel committee meets today", "x").mentions == []


def test_trump_style_all_caps_intel(matcher):
    text = "INTEL SHOULD BE A GREAT AMERICAN COMPANY AGAIN. THE STOCK IS GOING UP!"
    assert matcher.match(text, "truthsocial").mentions == [Mention("INTC", "name", "INTEL")]


@pytest.mark.parametrize(
    "text",
    [
        "Ford stock is cheap",
        "Ford shares slide",
        "Ford shareholders vote today",
        "Ford earnings tomorrow",
        "Ford price target raised",
        "Ford raises its dividend",
        "Ford IPO'd a century ago",
        "Ford market cap passes GM",
        "bearish on Ford",
        "Ford short seller report drops",
        "Short sellers circle Ford",
        "I'm buying Ford",
        "Sold my Ford position",
        "Ford investors are angry",
        "Ford up 3% today",
        "Ford at $12 is a steal",
        "Ford trades at £12 in London",
        "Ford at €11 in Frankfurt",
        "Ford vs $GM",
        "Ford 15c expiring Friday",
        "Ford call options are cheap",
        "Ford CEO Jim Farley says",
        "Ford dragged the S&P500 lower",
        "Ford is the most-shorted name on Wall Street",
        "Ford will invest in a new plant",
        "Ford sold outright to a buyer",
        "Ford stock market debut",
        "Bitcoin stocks and Ford rally",
        "Pelosi buys Ford",
        "Burry sells Ford",
        "Ford leads the tech selloff",
        "Big rally in stocks led by Ford",
        "A basket of stocks including Ford",
        "Ford 12.5p expiring Friday",
        "Ford 15C 10/17",
    ],
)
def test_finance_context_signals(matcher, text):
    assert pairs(matcher.match(text, "x")) >= {("F", "name")}


@pytest.mark.parametrize(
    "text",
    [
        "Harrison Ford's new movie",
        "Ford is a long way from home",
        "Ford made a short film",
        "Ford calls on Congress to act",
        "Ford puts on a show",
        "Ford has many options for dinner",
        "Ford shares a photo from the set",
        "Ford made 15 calls today",
        "Target sold out of the new console",
        "Target is selling out of Halloween candy",
        "Apple Watch is back in stock",
        "Target is out of stock again",
        "Apple stock photo of the new iPhone",
        "Target sells out of the new console",
        "Target is in  stock",
        "Apple Watch is back in\nstock",
        "Amazon rainforest temperatures hit 40C this week",
        "Record heat in Visa, California, 45C!",
        "Harrison Ford's new trailer is out in 4K and 1080p",
        "Shell beach today, 30C and sunny",
    ],
)
def test_weak_words_are_not_finance_context(matcher, text):
    assert pairs(matcher.match(text, "x")) == set()


@pytest.mark.parametrize(
    ("text", "platform", "expected"),
    [
        ("Wedbush Raises Tesla Price Target To $500", "x", {("TSLA", "name")}),
        ("Morgan Stanley Raises Nvidia Price Target to $200", "x", {("NVDA", "name")}),
        ("Goldman Sachs Lifts Microsoft Price Target", "reddit", {("GS", "name"), ("MSFT", "name")}),
        ("My thesis on NVDA. Target price: $200", "reddit", {("NVDA", "bare")}),
        ("NVDA calls. Target: 200. Stop: 180", "reddit", {("NVDA", "bare")}),
        ("SPY 600C PRINTED, TARGET HIT", "reddit", {("SPY", "bare")}),
        ("Block trade: 2M shares of Nvidia crossed at $180", "reddit", {("NVDA", "name")}),
        ("Huge Block Trade on SPY this morning", "reddit", {("SPY", "bare")}),
        ("Shell companies are how they hide the losses, stock will crash", "reddit", set()),
        ("The Oracle of Omaha sold $5B of stock", "x", set()),
        ("What is the Oracle of Omaha buying now?", "reddit", set()),
        ("Just google it before asking", "reddit", set()),
        ('For the basics, google "investopedia delta" and buy the dip', "reddit", set()),
    ],
)
def test_jargon_containing_a_name_is_not_a_mention(matcher, text, platform, expected):
    assert pairs(matcher.match(text, platform)) == expected


@pytest.mark.parametrize(
    ("text", "surface"),
    [
        ("Target price target raised to $150", "Target"),
        ("Target price cuts lift the stock", "Target"),
        ("Target hit by boycott, stock falls", "Target"),
        ("Block trades higher after earnings", "Block"),
        ("Shell earnings beat, buyback raised", "Shell"),
        ("Oracle shares jump on cloud deal", "Oracle"),
        ("Buffett, the Oracle of Omaha, buys more Oracle stock", "Oracle"),
        ("Google search revenue is up", "Google"),
        ("Google this week announced a buyback", "Google"),
        ("Google “AI Mode” rollout lifts shares", "Google"),
        ('Google "Gemini" launch', "Google"),
    ],
)
def test_name_next_to_jargon_still_matches(matcher, text, surface):
    (mention,) = matcher.match(text, "x").mentions
    assert mention.match_type == "name"
    assert mention.matched_text == surface


def test_eps_is_case_sensitive(matcher):
    assert matcher.match("Target's new eps of the show", "x").mentions == []
    assert matcher.match("Target EPS miss", "x").mentions == [Mention("TGT", "name", "Target")]


def test_cashtag_hint_counts_as_finance_context(matcher):
    assert pairs(matcher.match("Apple looks strong", "x", ["AAPL"])) == {("AAPL", "cashtag"), ("AAPL", "name")}


def test_reddit_bare_ticker_counts_as_finance_context(matcher):
    assert pairs(matcher.match("NVDA or Intel for 2027?", "reddit")) == {("NVDA", "bare"), ("INTC", "name")}
    assert pairs(matcher.match("INTC or Intel?", "reddit")) == {("INTC", "bare"), ("INTC", "name")}


@pytest.mark.parametrize(
    ("text", "ticker"),
    [("META is broken in this game", "META"), ("AMC to the moon", "AMC"), ("AMC AMC AMC", "AMC")],
)
def test_bare_token_is_not_finance_context_for_itself(matcher, text, ticker):
    assert matcher.match(text, "reddit").mentions == [Mention(ticker, "bare", ticker)]


def test_invalid_cashtag_hint_is_not_finance_context(matcher):
    assert matcher.match("Apple pie recipe", "x", ["WAYTOOLONG", "", "$"]) == MatchResult()
    assert pairs(matcher.match("Apple pie recipe", "x", ["BTC"])) == {("AAPL", "name")}


# ---------------------------------------------------------------- bare tickers


def test_reddit_bare_ticker(matcher):
    assert matcher.match("TSLA to the moon", "reddit").mentions == [Mention("TSLA", "bare", "TSLA")]


def test_reddit_bare_ticker_stoplist(matcher):
    assert matcher.match("CAT to the moon", "reddit").mentions == []


@pytest.mark.parametrize("token", ["BRK-B", "BRK/B"])
def test_reddit_bare_class_share_separators(matcher, token):
    assert matcher.match(f"{token} up big", "reddit").mentions == [Mention("BRK.B", "bare", token)]


@pytest.mark.parametrize("platform", ["x", "truthsocial"])
def test_bare_tickers_only_on_reddit(matcher, platform):
    assert matcher.match("TSLA to the moon", platform).mentions == []


def test_bare_ticker_must_be_uppercase_and_three_letters(matcher):
    assert matcher.match("tsla to the moon", "reddit").mentions == []
    assert matcher.match("GM and GS rally", "reddit").mentions == []


def test_bare_ticker_word_bounded(matcher):
    assert matcher.match("TSLAQ incoming, mTSLA", "reddit").mentions == []


def test_bare_alias_and_class_share(matcher):
    r = matcher.match("GOOG and BRK.B are my core", "reddit")
    assert r.mentions == [Mention("BRK.B", "bare", "BRK.B"), Mention("GOOGL", "bare", "GOOG")]


def test_bare_skipped_when_cashtag_matched(matcher):
    r = matcher.match("$TSLA TSLA Tesla tesla $tsla", "reddit")
    assert r.mentions == [Mention("TSLA", "cashtag", "$TSLA"), Mention("TSLA", "name", "Tesla")]


def test_trump_all_caps_produces_no_matches(matcher):
    text = (
        "NOW IS THE TIME TO BUY! CAT, ARM, COIN AND HOOD ARE GREAT COMPANIES. "
        "THE SPY WHO LOVED ME. MAKE AMERICA GREAT AGAIN!"
    )
    for platform in ("truthsocial", "x"):
        r = matcher.match(text, platform)
        assert r.mentions == []
        assert r.unknown_cashtags == []
    # Reddit does bare matching, but the stoplist blocks CAT/ARM/COIN/HOOD; SPY is a real bare match.
    assert pairs(matcher.match(text, "reddit")) == {("SPY", "bare")}


# ---------------------------------------------------------------- output shape


def test_one_mention_per_ticker_and_type_in_deterministic_order(matcher):
    text = "Tesla and Nvidia, $NVDA $TSLA, Tesla again, AMD AMD"
    first = matcher.match(text, "reddit")
    assert first.mentions == [
        Mention("AMD", "name", "AMD"),
        Mention("AMD", "bare", "AMD"),
        Mention("NVDA", "cashtag", "$NVDA"),
        Mention("NVDA", "name", "Nvidia"),
        Mention("TSLA", "cashtag", "$TSLA"),
        Mention("TSLA", "name", "Tesla"),
    ]
    assert matcher.match(text, "reddit") == first


def test_empty_text(matcher):
    assert matcher.match("", "x") == MatchResult()


# ---------------------------------------------------------------- real fixtures & performance


def reddit_fixture_entries(fixtures_dir):
    paths = sorted(fixtures_dir.glob("reddit_top_*.xml"))
    assert paths
    entries = []
    for path in paths:
        feed = feedparser.parse(path.read_bytes())
        assert feed.entries
        entries.extend(feed.entries)
    return entries


def test_reddit_fixture_titles(matcher, fixtures_dir):
    by_title = {}
    for e in reddit_fixture_entries(fixtures_dir):
        by_title[e.title] = matcher.match(e.title, "reddit")
        matcher.match(e.get("summary", ""), "reddit")

    def found(fragment: str) -> set[tuple[str, str]]:
        (title,) = [t for t in by_title if fragment in t]
        return pairs(by_title[title])

    assert found("Chevron Could Be the Biggest Winner") == {("CVX", "name")}
    assert found("Apple and Nvidia are taking up more of the S&P 500") == {("AAPL", "name"), ("NVDA", "name")}
    assert found("Oracle shares drop") == {("ORCL", "name")}
    assert found("Google Nears Release") == {("GOOGL", "name")}
    assert found('"Investor Day" did not go over well') == {("MCD", "name")}
    assert found("wrecked by MCD") == {("MCD", "bare")}
    assert found("Morgan Stanley inadvertently leaks") == set()
    assert found("BEARS SAW IT COMING") == set()
    assert found("Should I hold VLO") == set()
    tandem = [r for t, r in by_title.items() if "Tandem Diabetes" in t][0]
    assert tandem.mentions == []
    assert tandem.unknown_cashtags == ["TNDM"]


def test_daily_thread_boilerplate_is_not_a_google_mention(matcher, fixtures_dir):
    # AutoModerator posts this body every day; the verb "google" must not become a daily GOOGL event.
    (entry,) = [e for e in reddit_fixture_entries(fixtures_dir) if e.title.startswith("r/Stocks Daily Discussion")]
    body = re.sub(r"<[^>]+>", " ", entry.summary)
    assert 'google "investopedia delta"' in html.unescape(body)
    assert matcher.match(f"{entry.title}\n\n{body}", "reddit").mentions == []


def test_matching_2000_posts_is_fast(matcher, fixtures_dir):
    entries = reddit_fixture_entries(fixtures_dir)
    samples = [(e.title, "reddit") for e in entries]
    # A few long title+body Reddit posts (the raw HTML bodies stand in for selftext).
    samples += [(f"{e.title}\n\n{e.get('summary', '')}", "reddit") for e in entries[:3]]
    samples += [
        (
            "Apple earnings beat, shares up 5%. $TSLA and $NVDA ripping, Intel lagging. "
            "Watching https://example.com/charts?t=1 for the Goldman Sachs note on J.P. Morgan.",
            "x",
        ),
        (
            "NOW IS THE TIME TO BUY! CAT, ARM, COIN AND HOOD ARE GREAT COMPANIES. INTEL SHOULD BE GREAT AGAIN, "
            "THE STOCK MARKET IS AT RECORD HIGHS, THANK YOU PRESIDENT DJT! " * 3,
            "truthsocial",
        ),
        ("TSLA GOOG BRK.B to the moon, AMD calls printing, CAT is cheap, $ZZZZ lottery ticket", "reddit"),
        ("Just a normal day at the park with nothing about markets in it at all, really nothing.", "x"),
    ]
    posts = [samples[i % len(samples)] for i in range(2000)]
    start = time.perf_counter()
    for text, platform in posts:
        matcher.match(text, platform)
    assert time.perf_counter() - start < 1.0
