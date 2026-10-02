from __future__ import annotations

import sys
from collections.abc import Callable

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


class KeepAwake:
    """Asks Windows not to sleep while `active` (SetThreadExecutionState). A closed laptop lid can still sleep the
    machine, depending on its power settings. A no-op off Windows."""

    def __init__(self, set_state: Callable[[int], int] | None = None) -> None:
        if set_state is None and sys.platform == "win32":
            import ctypes

            set_state = ctypes.windll.kernel32.SetThreadExecutionState
        self._set = set_state
        self._on = False

    def update(self, active: bool) -> None:
        if self._set is None or active == self._on:
            return
        self._set(ES_CONTINUOUS | ES_SYSTEM_REQUIRED if active else ES_CONTINUOUS)
        self._on = active

    def release(self) -> None:
        self.update(False)
