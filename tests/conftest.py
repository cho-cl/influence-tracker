from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from influence_tracker import db
from influence_tracker.config import REPO_ROOT, Watchlist, load_watchlist

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES


@pytest.fixture
def watchlist() -> Watchlist:
    """The real, committed watchlist config."""
    return load_watchlist(REPO_ROOT / "config" / "watchlist.yaml")


@pytest.fixture
def conn(tmp_path):
    c = db.connect(tmp_path / "test.db")
    yield c
    c.close()


@pytest.fixture
def no_ntfy_env(monkeypatch) -> None:
    """No ntfy settings in the environment for this test, and none left behind by it."""
    for name in ("NTFY_TOPIC", "NTFY_SERVER"):
        # load_dotenv writes straight into os.environ, and delenv on an absent name records nothing to undo;
        # setenv first registers the name, so teardown also removes a value a loaded .env wrote.
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)


@pytest.fixture
def fixed_now() -> datetime:
    # Thursday 2026-09-24 22:30 UTC = 18:30 New York (EDT), after the extended session.
    return datetime(2026, 9, 24, 22, 30, tzinfo=UTC)
