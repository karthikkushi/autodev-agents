import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(autouse=True)
def _no_dashboard(monkeypatch):
    """Agents and the router post logs and AI-call events to the dashboard
    server; a test run must not add fake ones to a live dashboard."""
    import server.bridge as bridge
    monkeypatch.setattr(bridge, "_post", lambda *a, **k: None)
