from __future__ import annotations

import logging
import sqlite3
import weakref
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from influence_tracker import db, sentiment
from influence_tracker.config import REPO_ROOT, Watchlist
from influence_tracker.models import Mention, Post

FINTWIT = {0: "NEUTRAL", 1: "BULLISH", 2: "BEARISH"}
FINBERT = {0: "positive", 1: "negative", 2: "neutral"}
T0 = datetime(2026, 9, 23, 14, 0, tzinfo=UTC)


# ---------------------------------------------------------------- helpers


def add_post(
    conn: sqlite3.Connection,
    native_id: str,
    text: str,
    tickers: Sequence[str],
    *,
    platform: str = "truthsocial",
    minutes: int = 0,
    stance: tuple[str, float, str] | None = None,
) -> None:
    post = Post(
        platform=platform,
        native_id=native_id,
        author="realDonaldTrump",
        created_at_utc=T0 + timedelta(minutes=minutes),
        text=text,
        url=f"https://example.com/{native_id}",
    )
    with conn:
        db.upsert_post(conn, post, T0)
        db.replace_mentions(conn, platform, native_id, [Mention(t, "name", t) for t in tickers], [], T0)
        if stance is not None:
            conn.execute(
                "UPDATE posts SET stance = ?, stance_conf = ?, stance_model = ? WHERE platform = ? AND native_id = ?",
                (*stance, platform, native_id),
            )


def stances(conn: sqlite3.Connection) -> dict[str, tuple[str | None, float | None, str | None]]:
    rows = conn.execute("SELECT native_id, stance, stance_conf, stance_model FROM posts ORDER BY native_id")
    return {r["native_id"]: (r["stance"], r["stance_conf"], r["stance_model"]) for r in rows}


def with_sentiment(watchlist: Watchlist, **settings) -> Watchlist:
    return watchlist.model_copy(update={"sentiment": watchlist.sentiment.model_copy(update=settings)})


class FakeClassifier:
    """Labels by keyword; records every batch it is given and can fail on a chosen call."""

    def __init__(self, fail_on_call: int | None = None) -> None:
        self.batches: list[list[str]] = []
        self.fail_on_call = fail_on_call

    def __call__(self, texts: Sequence[str]) -> list[tuple[str, float]]:
        self.batches.append(list(texts))
        if self.fail_on_call == len(self.batches):
            raise RuntimeError("classifier blew up")
        out = []
        for t in texts:
            low = t.lower()
            if "buy" in low:
                out.append(("bullish", 0.9))
            elif "sell" in low:
                out.append(("bearish", 0.8))
            else:
                out.append(("neutral", 0.7))
        return out


class FakePipeline:
    """Shaped like a transformers text-classification pipeline: config.id2label, tokenizer.model_max_length,
    and a call that returns one {'label', 'score'} dict per input."""

    def __init__(self, id2label: dict[int, str], model_max_length: int = 512) -> None:
        self.model = SimpleNamespace(config=SimpleNamespace(id2label=id2label))
        self.tokenizer = SimpleNamespace(model_max_length=model_max_length)
        self.calls: list[tuple[list[str], dict]] = []

    # Raw labels (either scheme) a keyword in the text should produce.
    _RAW = {"moon": {"BULLISH", "positive"}, "crash": {"BEARISH", "negative"}}

    def __call__(self, inputs: list[str], **kwargs) -> list[dict]:
        assert isinstance(inputs, list), "a tuple would be run as one single input"
        self.calls.append((inputs, kwargs))
        out = []
        for text in inputs:
            wanted = next((raws for word, raws in self._RAW.items() if word in text), {"NEUTRAL", "neutral"})
            label = next(lbl for lbl in self.model.config.id2label.values() if lbl in wanted)
            out.append({"label": label, "score": 0.75})
        return out


@pytest.fixture
def fake_pipelines(monkeypatch):
    """Replaces the real pipeline factory. Configure .cached / .id2label / .max_len before loading."""

    class Factory:
        def __init__(self) -> None:
            self.cached = True
            self.id2label = FINTWIT
            self.max_len = 512
            self.calls: list[tuple[str, bool]] = []
            self.built: list[FakePipeline] = []

        def __call__(self, model_id: str, local_files_only: bool) -> FakePipeline:
            self.calls.append((model_id, local_files_only))
            if local_files_only and not self.cached:
                raise OSError("We couldn't connect to 'https://huggingface.co' ... couldn't find them in the cache")
            pipe = FakePipeline(self.id2label, self.max_len)
            self.built.append(pipe)
            return pipe

    factory = Factory()
    monkeypatch.setattr(sentiment, "_new_pipeline", factory)
    return factory


# ---------------------------------------------------------------- preprocess


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Great quarter @elonmusk! https://t.co/abc123", "Great quarter @USER! [URL]"),
        ("see www.example.com/x?y=1 now", "see [URL] now"),
        ("(WWW.Example.com/a).", "([URL])."),
        # A stretched interjection is not a link; masking from its inner "www." would eat the next word.
        ("Awww.no more gains for $NVDA", "Awww.no more gains for $NVDA"),
        ("Owww..bullish on $AMD again", "Owww..bullish on $AMD again"),
        ("Read this: https://truthsocial.com/@realDonaldTrump/1234.", "Read this: [URL]."),
        ("(https://example.com/a)", "([URL])"),
        ("@JDVance1 and @Real_Donald_Trump agree", "@USER and @USER agree"),
        ("mail me at someone@example.com", "mail me at someone@example.com"),
        ("Title\n\n  body\twith   gaps  ", "Title body with gaps"),
        ("RT @USER: already masked [URL]", "RT @USER: already masked [URL]"),
        ("", ""),
    ],
)
def test_preprocess_masks_handles_and_links_and_collapses_whitespace(raw, expected):
    assert sentiment.preprocess(raw) == expected


def test_preprocess_is_idempotent():
    once = sentiment.preprocess("Buy $INTC @potus https://x.com/a   now")
    assert sentiment.preprocess(once) == once


# ---------------------------------------------------------------- label mapping


def test_label_mapping_fintwitbert_scheme():
    assert sentiment.stance_labels(FINTWIT) == {"NEUTRAL": "neutral", "BULLISH": "bullish", "BEARISH": "bearish"}


def test_label_mapping_finbert_scheme():
    assert sentiment.stance_labels(FINBERT) == {"positive": "bullish", "negative": "bearish", "neutral": "neutral"}


@pytest.mark.parametrize(
    "id2label",
    [
        {0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"},
        {0: "positive", 1: "negative"},
        {0: "positive", 1: "bullish", 2: "negative"},
        {0: "NEUTRAL", 1: "BULLISH", 2: "BEARISH", 3: "SPAM"},
        {},
    ],
)
def test_label_mapping_rejects_other_schemes(id2label):
    with pytest.raises(ValueError, match="label"):
        sentiment.stance_labels(id2label)


# ---------------------------------------------------------------- load_classifier (fake pipeline)


@pytest.mark.parametrize(("id2label", "scheme"), [(FINTWIT, "fintwit"), (FINBERT, "finbert")])
def test_classifier_maps_both_label_schemes(fake_pipelines, id2label, scheme):
    fake_pipelines.id2label = id2label
    classify = sentiment.load_classifier("some/model", batch_size=8)
    out = classify(["to the moon", "it will crash", "the weather"])
    assert out == [("bullish", 0.75), ("bearish", 0.75), ("neutral", 0.75)], scheme


def test_classifier_rejects_unknown_label_scheme_at_load_time(fake_pipelines):
    fake_pipelines.id2label = {0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"}
    with pytest.raises(ValueError, match="label"):
        sentiment.load_classifier("some/model")


def test_classifier_feeds_preprocessed_text_with_truncation_and_batch_size(fake_pipelines):
    classify = sentiment.load_classifier("some/model", batch_size=4)
    classify(("Buy it @trader https://t.co/x",))
    [(inputs, kwargs)] = fake_pipelines.built[0].calls
    assert inputs == ["Buy it @USER [URL]"]
    assert kwargs == {"batch_size": 4, "truncation": True, "max_length": 512}


@pytest.mark.parametrize(("tokenizer_max", "expected"), [(int(1e30), 512), (512, 512), (128, 128)])
def test_classifier_truncates_to_the_model_max_length(fake_pipelines, tokenizer_max, expected):
    fake_pipelines.max_len = tokenizer_max
    sentiment.load_classifier("some/model")(["x"])
    assert fake_pipelines.built[0].calls[0][1]["max_length"] == expected


def test_classifier_on_empty_input_does_not_call_the_model(fake_pipelines):
    assert sentiment.load_classifier("some/model")([]) == []
    assert fake_pipelines.built[0].calls == []


def test_load_uses_the_local_cache_first(fake_pipelines):
    sentiment.load_classifier("some/model")
    assert fake_pipelines.calls == [("some/model", True)]


def test_load_downloads_only_when_not_cached(fake_pipelines, caplog):
    fake_pipelines.cached = False
    with caplog.at_level(logging.WARNING, logger="influence_tracker.sentiment"):
        classify = sentiment.load_classifier("some/model")
    assert fake_pipelines.calls == [("some/model", True), ("some/model", False)]
    assert "download" in caplog.text and "some/model" in caplog.text
    assert classify(["to the moon"]) == [("bullish", 0.75)]


# ---------------------------------------------------------------- classify_posts


def test_candidates_need_a_non_benchmark_ticker_and_a_missing_or_stale_label(conn, watchlist):
    model = watchlist.sentiment.model_id
    add_post(conn, "no-mention", "Buy everything", [])
    add_post(conn, "spy-only", "Buy the market", ["SPY"])
    add_post(conn, "benchmarks", "Buy SPY and QQQ", ["SPY", "QQQ"])
    add_post(conn, "unconfigured", "Buy ZZZZ", ["ZZZZ"])
    add_post(conn, "tsla", "Buy Tesla", ["TSLA"], minutes=1)
    add_post(conn, "spy-and-intc", "Sell Intel", ["SPY", "INTC"], minutes=2)
    add_post(conn, "labelled", "Buy Apple", ["AAPL"], minutes=3, stance=("bullish", 0.9, model))
    add_post(conn, "old-model", "Sell Apple", ["AAPL"], minutes=4, stance=("bullish", 0.6, "ProsusAI/finbert"))
    add_post(conn, "no-model", "Sell Nvidia", ["NVDA"], minutes=5, stance=("bullish", 0.6, None))

    fake = FakeClassifier()
    counts = sentiment.classify_posts(conn, watchlist, 1, T0, classifier=fake)

    assert counts["status"] == "ok"
    assert counts["candidates"] == 4
    assert counts["classified"] == 4
    seen = sorted(t for batch in fake.batches for t in batch)
    assert seen == ["Buy Tesla", "Sell Apple", "Sell Intel", "Sell Nvidia"]
    got = stances(conn)
    assert got["tsla"] == ("bullish", 0.9, model)
    assert got["spy-and-intc"] == ("bearish", 0.8, model)
    assert got["old-model"] == ("bearish", 0.8, model)
    assert got["no-model"] == ("bearish", 0.8, model)
    assert got["labelled"] == ("bullish", 0.9, model)
    for skipped in ("no-mention", "spy-only", "benchmarks", "unconfigured"):
        assert got[skipped] == (None, None, None)


def test_second_run_finds_nothing_and_a_model_change_relabels(conn, watchlist):
    add_post(conn, "a", "Buy Tesla", ["TSLA"])
    add_post(conn, "b", "Sell Tesla", ["TSLA"], minutes=1)
    assert sentiment.classify_posts(conn, watchlist, 1, T0, classifier=FakeClassifier())["classified"] == 2

    again = FakeClassifier()
    counts = sentiment.classify_posts(conn, watchlist, 2, T0, classifier=again)
    assert (counts["status"], counts["candidates"], counts["classified"]) == ("ok", 0, 0)
    assert again.batches == []

    switched = with_sentiment(watchlist, model_id="ProsusAI/finbert")
    counts = sentiment.classify_posts(conn, switched, 3, T0, classifier=FakeClassifier())
    assert (counts["candidates"], counts["classified"], counts["model_id"]) == (2, 2, "ProsusAI/finbert")
    assert {v[2] for v in stances(conn).values()} == {"ProsusAI/finbert"}


def test_edited_post_text_clears_the_label_and_is_classified_again(conn, watchlist):
    add_post(conn, "a", "Buy Tesla", ["TSLA"])
    sentiment.classify_posts(conn, watchlist, 1, T0, classifier=FakeClassifier())
    add_post(conn, "a", "Sell Tesla now", ["TSLA"])
    counts = sentiment.classify_posts(conn, watchlist, 2, T0, classifier=FakeClassifier())
    assert counts["classified"] == 1
    assert stances(conn)["a"][0] == "bearish"


def test_classifies_in_batches_of_the_configured_size(conn, watchlist):
    for i in range(7):
        add_post(conn, f"p{i}", f"Buy Tesla {i}", ["TSLA"], minutes=i)
    fake = FakeClassifier()
    counts = sentiment.classify_posts(conn, with_sentiment(watchlist, batch_size=3), 1, T0, classifier=fake)
    assert [len(b) for b in fake.batches] == [3, 3, 1]
    assert counts["classified"] == 7


def test_each_batch_is_committed_so_an_interrupted_run_keeps_finished_batches(conn, watchlist, tmp_path, caplog):
    for i in range(5):
        add_post(conn, f"p{i}", f"Buy Tesla {i}", ["TSLA"], minutes=i)
    fake = FakeClassifier(fail_on_call=2)
    with caplog.at_level(logging.ERROR, logger="influence_tracker.sentiment"):
        counts = sentiment.classify_posts(conn, with_sentiment(watchlist, batch_size=2), 1, T0, classifier=fake)

    assert counts["status"] == "partial"
    assert (counts["candidates"], counts["classified"]) == (5, 2)
    assert "classifier blew up" in counts["error"]
    assert not conn.in_transaction
    other = sqlite3.connect(tmp_path / "test.db")
    try:
        done = other.execute("SELECT COUNT(*) FROM posts WHERE stance IS NOT NULL").fetchone()[0]
    finally:
        other.close()
    assert done == 2

    rerun = sentiment.classify_posts(conn, watchlist, 2, T0, classifier=FakeClassifier())
    assert (rerun["status"], rerun["candidates"], rerun["classified"]) == ("ok", 3, 3)


def test_failure_on_the_first_batch_is_an_error(conn, watchlist):
    add_post(conn, "a", "Buy Tesla", ["TSLA"])
    counts = sentiment.classify_posts(conn, watchlist, 1, T0, classifier=FakeClassifier(fail_on_call=1))
    assert (counts["status"], counts["classified"]) == ("error", 0)
    assert stances(conn)["a"] == (None, None, None)


@pytest.mark.parametrize(
    "bad_output",
    [[("bullish", 0.9)], [("bullish", 0.9), ("bullish", 0.9), ("bullish", 0.9)], [("up", 0.9), ("down", 0.9)]],
)
def test_malformed_classifier_output_writes_nothing(conn, watchlist, bad_output):
    add_post(conn, "a", "Buy Tesla", ["TSLA"])
    add_post(conn, "b", "Buy Intel", ["INTC"], minutes=1)
    counts = sentiment.classify_posts(conn, watchlist, 1, T0, classifier=lambda texts: bad_output)
    assert (counts["status"], counts["classified"]) == ("error", 0)
    assert {v for v in stances(conn).values()} == {(None, None, None)}


def test_no_candidates_means_no_model_load(conn, watchlist, monkeypatch):
    def must_not_load(*args, **kwargs):
        raise AssertionError("the model must not be loaded without candidates")

    monkeypatch.setattr(sentiment, "load_classifier", must_not_load)
    add_post(conn, "spy-only", "Buy the market", ["SPY"])
    counts = sentiment.classify_posts(conn, watchlist, 1, T0)
    assert counts == {
        "status": "ok",
        "candidates": 0,
        "classified": 0,
        "model_id": watchlist.sentiment.model_id,
        "load_seconds": 0.0,
        "classify_seconds": 0.0,
    }


def test_loads_the_configured_model_when_there_are_candidates(conn, watchlist, monkeypatch):
    loads: list[tuple[str, int]] = []

    def fake_load(model_id: str, batch_size: int = 16):
        loads.append((model_id, batch_size))
        return FakeClassifier()

    monkeypatch.setattr(sentiment, "load_classifier", fake_load)
    add_post(conn, "a", "Buy Tesla", ["TSLA"])
    counts = sentiment.classify_posts(conn, with_sentiment(watchlist, batch_size=5), 1, T0)
    assert loads == [(watchlist.sentiment.model_id, 5)]
    assert counts["status"] == "ok" and counts["classified"] == 1
    assert counts["load_seconds"] >= 0 and counts["classify_seconds"] >= 0


@pytest.mark.parametrize(
    ("failure", "status"), [(None, "ok"), ("classify", "error"), ("load", "error")], ids=["ok", "classify", "load"]
)
def test_a_model_loaded_by_the_run_is_released_when_it_ends(conn, watchlist, monkeypatch, failure, status):
    refs: list[weakref.ref] = []

    def fake_load(model_id: str, batch_size: int = 16):
        classifier = FakeClassifier(fail_on_call=1 if failure == "classify" else None)
        # Like the real pipeline, which only a cyclic GC pass frees.
        classifier.cycle = classifier
        refs.append(weakref.ref(classifier))
        if failure == "load":
            raise ValueError("model labels ['LABEL_0', 'LABEL_1', 'LABEL_2'] are not a stance label set")
        return classifier

    monkeypatch.setattr(sentiment, "load_classifier", fake_load)
    # pytest's log capture keeps the logged traceback, and through it the classifier, alive.
    monkeypatch.setattr(sentiment.log, "disabled", True)
    add_post(conn, "a", "Buy Tesla", ["TSLA"])
    counts = sentiment.classify_posts(conn, watchlist, 1, T0)
    assert counts["status"] == status
    assert refs[0]() is None


def test_model_load_failure_is_reported_not_raised(conn, watchlist, monkeypatch):
    def offline(model_id: str, batch_size: int = 16):
        raise OSError("We couldn't connect to 'https://huggingface.co'")

    monkeypatch.setattr(sentiment, "load_classifier", offline)
    add_post(conn, "a", "Buy Tesla", ["TSLA"])
    counts = sentiment.classify_posts(conn, watchlist, 1, T0)
    assert (counts["status"], counts["candidates"], counts["classified"]) == ("error", 1, 0)
    assert "huggingface.co" in counts["error"]
    assert stances(conn)["a"] == (None, None, None)


# ---------------------------------------------------------------- live: the real model


@pytest.fixture(scope="module")
def real_classifier():
    from influence_tracker.config import SentimentConfig

    return sentiment.load_classifier(SentimentConfig().model_id)


@pytest.mark.live
def test_live_real_model_labels_clear_examples(real_classifier):
    out = real_classifier(
        [
            "$TSLA breaking out, loading up on shares, this is going to the moon",
            "Dumping all my $NVDA shares. This stock is going to crash hard.",
            "The meeting is scheduled for Tuesday at 3pm.",
        ]
    )
    assert [label for label, _ in out] == ["bullish", "bearish", "neutral"]
    assert all(0.34 < conf <= 1.0 for _, conf in out)


@pytest.mark.live
def test_live_classify_every_candidate_in_the_repo_db(real_classifier, watchlist, tmp_path, capsys):
    """Copies posts and mentions from data/tracker.db (read-only) into a scratch DB, classifies every candidate
    with the real model and prints each one so a person can judge the labels."""
    source = REPO_ROOT / "data" / "tracker.db"
    if not source.exists():
        pytest.skip(f"{source} not found")
    conn = db.connect(tmp_path / "copy.db")
    try:
        _copy_posts(Path(source), conn)
        counts = sentiment.classify_posts(conn, watchlist, 1, datetime.now(UTC), classifier=real_classifier)
        assert counts["status"] == "ok"
        assert counts["classified"] == counts["candidates"] > 0
        rows = conn.execute(
            "SELECT platform, author, stance, stance_conf, text FROM posts WHERE stance IS NOT NULL "
            "ORDER BY platform, created_at_utc"
        ).fetchall()
    finally:
        conn.close()
    with capsys.disabled():
        print(f"\n{counts}")
        for r in rows:
            snippet = " ".join(r["text"].split())[:120]
            print(f"{r['platform']:<11} {r['author'][:18]:<18} {r['stance']:<7} {r['stance_conf']:.2f}  {snippet}")


def _copy_posts(source: Path, conn: sqlite3.Connection) -> None:
    src = sqlite3.connect(f"{source.as_uri()}?mode=ro", uri=True)
    try:
        for table in ("posts", "mentions"):
            names = [r[1] for r in src.execute(f"PRAGMA table_info({table})")]
            cols, marks = ", ".join(names), ", ".join("?" * len(names))
            rows = src.execute(f"SELECT {cols} FROM {table}").fetchall()
            with conn:
                conn.executemany(f"INSERT INTO {table} ({cols}) VALUES ({marks})", rows)
    finally:
        src.close()
    with conn:
        conn.execute("UPDATE posts SET stance = NULL, stance_conf = NULL, stance_model = NULL")
