from __future__ import annotations

import json

import httpx
import pytest

from influence_tracker.alerts.notify import (
    MAX_BODY_BYTES,
    DryRunNotifier,
    Message,
    NotifyError,
    NtfyNotifier,
    fit_body,
)

MSG = Message(
    title="⭐ NVDA · realDonaldTrump · bullish (0.91)", body="Buy 🚀", priority=4, tags=("star",), click="https://x"
)


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_posts_json_to_server_root():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"id": "abc"})

    NtfyNotifier("https://ntfy.sh/", "topic-1", client=_client(handler), sleep=lambda s: None).send(MSG)
    [req] = seen
    assert str(req.url) == "https://ntfy.sh/"
    body = json.loads(req.content.decode("utf-8"))
    assert body == {
        "topic": "topic-1",
        "title": MSG.title,
        "message": "Buy 🚀",
        "priority": 4,
        "tags": ["star"],
        "click": "https://x",
    }


def test_retries_once_then_succeeds():
    codes = iter([500, 200])
    sleeps = []
    n = NtfyNotifier("https://ntfy.sh", "t", client=_client(lambda r: httpx.Response(next(codes))), sleep=sleeps.append)
    n.send(MSG)
    assert sleeps == [5.0]


def test_two_failures_raise():
    n = NtfyNotifier(
        "https://ntfy.sh", "t", client=_client(lambda r: httpx.Response(503, text="down")), sleep=lambda s: None
    )
    with pytest.raises(NotifyError, match="503"):
        n.send(MSG)


def test_transport_errors_raise_notify_error():
    def boom(request):
        raise httpx.ConnectError("no route")

    n = NtfyNotifier("https://ntfy.sh", "t", client=_client(boom), sleep=lambda s: None)
    with pytest.raises(NotifyError, match="ConnectError"):
        n.send(MSG)


def test_fit_body_caps_utf8_bytes():
    text = "📈" * 3000
    out = fit_body(text)
    assert len(out.encode("utf-8")) <= MAX_BODY_BYTES
    assert out.endswith("…")
    assert fit_body("short") == "short"


def test_dry_run_records():
    n = DryRunNotifier()
    n.send(MSG)
    assert n.sent == [MSG]
