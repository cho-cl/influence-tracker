from __future__ import annotations

import shutil
import subprocess

import pytest

from influence_tracker.config import REPO_ROOT

SCRIPT = REPO_ROOT / "scripts" / "register_watch_task.ps1"


@pytest.mark.skipif(shutil.which("powershell") is None, reason="needs Windows PowerShell")
def test_watch_task_script_parses():
    cmd = (
        "$errs = $null; $null = [System.Management.Automation.Language.Parser]::ParseFile("
        f"'{SCRIPT}', [ref]$null, [ref]$errs); $errs.Count"
    )
    out = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "0"


def test_watch_runner_calls_watch():
    text = (REPO_ROOT / "scripts" / "run_watch.cmd").read_text(encoding="utf-8")
    assert 'influence.exe" watch' in text and "logs\\watch.log" in text
