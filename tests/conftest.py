from __future__ import annotations

import pytest

from distill.settings import Settings


@pytest.fixture(autouse=True)
def _isolate_settings_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Settings read the process environment; a developer shell exporting e.g. ROUTER_API_KEY or MAX_INFLIGHT
    must not change test outcomes."""
    for name in Settings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)
