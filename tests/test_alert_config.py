from __future__ import annotations

import os

import pytest
import yaml
from pydantic import ValidationError

from influence_tracker.config import REPO_ROOT, Watchlist, load_settings


def _raw() -> dict:
    with open(REPO_ROOT / "config" / "watchlist.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_alerts_defaults_when_section_missing():
    raw = _raw()
    raw.pop("alerts", None)
    wl = Watchlist.model_validate(raw)
    assert wl.alerts.enabled is True
    assert wl.alerts.holdings == []
    assert (wl.alerts.poll_active_minutes, wl.alerts.poll_idle_minutes, wl.alerts.poll_x_minutes) == (5, 30, 15)
    assert (wl.alerts.late_after_minutes, wl.alerts.followup_minutes) == (30, 60)


def test_holdings_are_normalised_to_canonical_symbols():
    raw = _raw()
    raw["alerts"] = {"holdings": [" nvda ", "GOOG", "brk.a"]}
    wl = Watchlist.model_validate(raw)
    assert wl.alerts.holdings == ["NVDA", "GOOGL", "BRK.B"]


def test_unknown_holding_is_a_config_error():
    raw = _raw()
    raw["alerts"] = {"holdings": ["ZZZZ"]}
    with pytest.raises(ValidationError, match="ZZZZ"):
        Watchlist.model_validate(raw)


def test_settings_read_ntfy_from_env(tmp_path, no_ntfy_env):
    (tmp_path / ".env").write_text("NTFY_TOPIC=influence-abc123\n", encoding="utf-8")
    s = load_settings(tmp_path)
    assert s.ntfy_topic == "influence-abc123"
    assert s.ntfy_server == "https://ntfy.sh"


def test_settings_without_topic(tmp_path, no_ntfy_env):
    assert load_settings(tmp_path).ntfy_topic is None


def test_a_topic_loaded_from_env_file_does_not_outlive_the_test(tmp_path, monkeypatch, no_ntfy_env):
    (tmp_path / ".env").write_text("NTFY_TOPIC=influence-abc123\n", encoding="utf-8")
    load_settings(tmp_path)
    assert os.environ["NTFY_TOPIC"] == "influence-abc123"  # load_dotenv writes straight into os.environ
    monkeypatch.undo()  # what teardown does
    assert os.environ.get("NTFY_TOPIC") != "influence-abc123"
