from __future__ import annotations

import gc
import logging
import os
import re
import sqlite3
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from typing import Any

from .config import Watchlist

log = logging.getLogger(__name__)

Classifier = Callable[[Sequence[str]], list[tuple[str, float]]]

STANCES = ("bullish", "bearish", "neutral")
# Lowercased model label -> stance. FinTwitBERT says bullish/bearish/neutral, FinBERT positive/negative/neutral.
RAW_LABEL_STANCES = {
    "bullish": "bullish",
    "bearish": "bearish",
    "neutral": "neutral",
    "positive": "bullish",
    "negative": "bearish",
}
MAX_TOKENS = 512
MAX_TORCH_THREADS = 4

# A URL's trailing sentence punctuation stays in the text ("see https://a.b/c." -> "see [URL].").
# \b keeps "www." inside a word ("Awww.no") from masking the next word.
_URL = re.compile(r"(?:https?://|\bwww\.)\S+?(?=[.,;:!?)\]}'\"]*(?:\s|$))", re.IGNORECASE)
# Not preceded by a word character, so e-mail addresses are left alone.
_HANDLE = re.compile(r"(?<!\w)@\w+")


def preprocess(text: str) -> str:
    """Mask text the way FinTwitBERT's training tweets were masked: links -> [URL], @handles -> @USER."""
    text = _URL.sub("[URL]", text)
    text = _HANDLE.sub("@USER", text)
    return " ".join(text.split())


def stance_labels(id2label: Mapping[int, str]) -> dict[str, str]:
    """Raw model label -> stance. Raises ValueError unless the labels are exactly one bullish, one bearish and
    one neutral label under either naming scheme."""
    mapping = {raw: RAW_LABEL_STANCES.get(str(raw).lower()) for raw in id2label.values()}
    if len(id2label) != len(STANCES) or sorted(s for s in mapping.values() if s) != sorted(STANCES):
        raise ValueError(
            f"model labels {sorted(id2label.values())} are not a bullish/bearish/neutral "
            "(or positive/negative/neutral) label set"
        )
    return {raw: stance for raw, stance in mapping.items() if stance}


def _new_pipeline(model_id: str, local_files_only: bool) -> Any:
    import torch
    from transformers import pipeline

    return pipeline(
        "text-classification",
        model=model_id,
        device="cpu",
        dtype=torch.float32,
        local_files_only=local_files_only,
    )


def _load_pipeline(model_id: str) -> Any:
    try:
        return _new_pipeline(model_id, local_files_only=True)
    except OSError:
        log.warning(
            "sentiment model %s is not in the Hugging Face cache: downloading it now (one time, about 400 MB)",
            model_id,
        )
    pipe = _new_pipeline(model_id, local_files_only=False)
    log.info("sentiment model %s downloaded; later runs load it from the cache", model_id)
    return pipe


def load_classifier(model_id: str, batch_size: int = 16) -> Classifier:
    """A CPU text-classification model as texts -> [(stance, confidence of that stance)]. Loads from the local
    Hugging Face cache and downloads only if the model is missing there."""
    import torch

    # Leave cores for the rest of the machine; batch-16 BERT inference gains little beyond 4 threads.
    torch.set_num_threads(min(MAX_TORCH_THREADS, os.cpu_count() or 1))
    pipe = _load_pipeline(model_id)
    labels = stance_labels(pipe.model.config.id2label)
    # FinTwitBERT's tokenizer reports no limit (1e30); truncation alone would crash its 512 positions on long posts.
    max_length = min(MAX_TOKENS, int(pipe.tokenizer.model_max_length))

    def classify(texts: Sequence[str]) -> list[tuple[str, float]]:
        if not texts:
            return []
        with torch.inference_mode():
            results = pipe(
                [preprocess(t) for t in texts], batch_size=batch_size, truncation=True, max_length=max_length
            )
        return [(labels[r["label"]], float(r["score"])) for r in results]

    return classify


def _candidates(conn: sqlite3.Connection, watchlist: Watchlist) -> list[sqlite3.Row]:
    symbols = [t.symbol for t in watchlist.event_tickers]
    if not symbols:
        return []
    marks = ",".join("?" * len(symbols))
    return conn.execute(
        f"""SELECT p.platform, p.native_id, p.text FROM posts p
            WHERE (p.stance IS NULL OR COALESCE(p.stance_model, '') <> ?)
              AND EXISTS (SELECT 1 FROM mentions m
                          WHERE m.platform = p.platform AND m.native_id = p.native_id AND m.ticker IN ({marks}))
            ORDER BY p.created_at_utc, p.platform, p.native_id""",
        (watchlist.sentiment.model_id, *symbols),
    ).fetchall()


def _store_batch(
    conn: sqlite3.Connection, batch: Sequence[sqlite3.Row], results: Sequence[tuple[str, float]], model_id: str
) -> int:
    if len(results) != len(batch):
        raise ValueError(f"classifier returned {len(results)} results for {len(batch)} texts")
    bad = sorted({stance for stance, _ in results if stance not in STANCES})
    if bad:
        raise ValueError(f"classifier returned unknown stance(s) {bad}")
    stored = 0
    with conn:
        for row, (stance, conf) in zip(batch, results, strict=True):
            # A post edited since it was read keeps its cleared label; the next run classifies the new text.
            cur = conn.execute(
                """UPDATE posts SET stance = ?, stance_conf = ?, stance_model = ?
                   WHERE platform = ? AND native_id = ? AND text = ?""",
                (stance, float(conf), model_id, row["platform"], row["native_id"], row["text"]),
            )
            stored += cur.rowcount
    return stored


def classify_posts(
    conn: sqlite3.Connection,
    watchlist: Watchlist,
    run_id: int,
    now: datetime,
    classifier: Classifier | None = None,
) -> dict:
    """Label the stance of every post that mentions a non-benchmark ticker and has no label from the configured
    model yet. Each batch is committed on its own, so an interrupted run keeps the batches it finished."""
    cfg = watchlist.sentiment
    rows = _candidates(conn, watchlist)
    counts: dict = {
        "status": "ok",
        "candidates": len(rows),
        "classified": 0,
        "model_id": cfg.model_id,
        "load_seconds": 0.0,
        "classify_seconds": 0.0,
    }
    if not rows:
        log.info("classify (run %d): no posts need a stance from %s", run_id, cfg.model_id)
        return counts
    log.info("classify (run %d): %d post(s) need a stance from %s", run_id, len(rows), cfg.model_id)

    if classifier is not None:
        _classify_batches(conn, rows, classifier, watchlist, run_id, counts)
    else:
        started = time.perf_counter()
        try:
            classifier = load_classifier(cfg.model_id, cfg.batch_size)
            counts["load_seconds"] = round(time.perf_counter() - started, 2)
            _classify_batches(conn, rows, classifier, watchlist, run_id, counts)
        # Hub, network, cache and model-config problems surface as many unrelated exception types.
        except Exception as e:
            log.exception("classify (run %d): cannot load sentiment model %s", run_id, cfg.model_id)
            counts.update(status="error", error=f"model load failed: {type(e).__name__}: {e}")
        finally:
            # The pipeline has reference cycles; without a GC pass its ~0.5 GB would outlive the stage.
            classifier = None
            gc.collect()
    log.info("classify (run %d): %s", run_id, counts)
    return counts


def _classify_batches(
    conn: sqlite3.Connection,
    rows: Sequence[sqlite3.Row],
    classifier: Classifier,
    watchlist: Watchlist,
    run_id: int,
    counts: dict,
) -> None:
    cfg = watchlist.sentiment
    # Similar lengths in a batch mean less padding for the model to chew through.
    rows = sorted(rows, key=lambda r: len(r["text"]))
    started = time.perf_counter()
    try:
        for i in range(0, len(rows), cfg.batch_size):
            batch = rows[i : i + cfg.batch_size]
            results = classifier([r["text"] for r in batch])
            counts["classified"] += _store_batch(conn, batch, results, cfg.model_id)
    # A model failure mid-run keeps the batches already committed; the rest are picked up next run.
    except Exception as e:
        log.exception("classify (run %d): stopped after %d post(s)", run_id, counts["classified"])
        counts.update(status="partial" if counts["classified"] else "error", error=f"{type(e).__name__}: {e}")
    counts["classify_seconds"] = round(time.perf_counter() - started, 2)
