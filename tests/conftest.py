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
def fixed_now() -> datetime:
    # Thursday 2026-09-24 22:30 UTC = 18:30 New York (EDT), after the extended session.
    return datetime(2026, 9, 24, 22, 30, tzinfo=UTC)
