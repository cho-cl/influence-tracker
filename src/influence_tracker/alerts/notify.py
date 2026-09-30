from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import httpx

log = logging.getLogger(__name__)

# ntfy turns bodies over 4,096 bytes into attachments; stay clearly below.
MAX_BODY_BYTES = 4000
RETRY_WAIT_S = 5.0
TIMEOUT_S = 10.0


@dataclass(frozen=True)
class Message:
    title: str
    body: str
    priority: int = 3
    tags: tuple[str, ...] = ()
    click: str | None = None


class Notifier(Protocol):
    def send(self, message: Message) -> None: ...


class NotifyError(Exception):
    pass


def fit_body(text: str, limit: int = MAX_BODY_BYTES) -> str:
    if len(text.encode("utf-8")) <= limit:
        return text
    ellipsis = "…"
    budget = limit - len(ellipsis.encode("utf-8"))
    cut = text.encode("utf-8")[:budget].decode("utf-8", errors="ignore")
    return cut.rstrip() + ellipsis


class NtfyNotifier:
    """Publishes through ntfy's JSON API. Header publishing would break on titles with emoji (headers are ASCII)."""

    def __init__(
        self,
        server: str,
        topic: str,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.url = server.rstrip("/") + "/"
        self.topic = topic
        self.client = client or httpx.Client(timeout=TIMEOUT_S)
        self.sleep = sleep

    def send(self, message: Message) -> None:
        payload: dict = {
            "topic": self.topic,
            "title": message.title,
            "message": fit_body(message.body),
            "priority": message.priority,
        }
        if message.tags:
            payload["tags"] = list(message.tags)
        if message.click:
            payload["click"] = message.click
        error = ""
        for attempt in (1, 2):
            try:
                resp = self.client.post(self.url, json=payload)
                if resp.status_code < 300:
                    return
                error = f"HTTP {resp.status_code}: {resp.text[:200]}"
            except httpx.HTTPError as e:
                error = f"{type(e).__name__}: {e}"
            if attempt == 1:
                self.sleep(RETRY_WAIT_S)
        raise NotifyError(error)


class DryRunNotifier:
    def __init__(self) -> None:
        self.sent: list[Message] = []

    def send(self, message: Message) -> None:
        self.sent.append(message)
        log.info("[dry run] would notify: %s | %s", message.title, message.body.replace("\n", " / "))
