"""Dashboard server pieces: model limits and the live-preview relay — offline."""
import http.server
import json
import threading
import time

import pytest
from fastapi.testclient import TestClient

import server.server as srv


def test_model_status_shows_limits_and_missing_keys(tmp_path, monkeypatch):
    monkeypatch.setattr(srv, "COOLDOWN_FILE", tmp_path / "cooldowns.json")
    monkeypatch.setattr(srv, "_role_models", lambda: {
        "gemini/gemini-3.8-flash": {"env": "GOOGLE_API_KEY", "roles": ["fast", "coding"]},
        "nvidia_nim/nvidia/nemotron-3-super-120b-a12b": {"env": "NVIDIA_API_KEY", "roles": ["coding"]},
        "groq/openai/gpt-oss-120b": {"env": "GROQ_API_KEY", "roles": ["fast"]}})
    monkeypatch.setenv("GOOGLE_API_KEY", "x" * 20)
    monkeypatch.setenv("NVIDIA_API_KEY", "x" * 20)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    until = time.time() + 3600
    (tmp_path / "cooldowns.json").write_text(json.dumps({"models": {"gemini/gemini-3.8-flash": {
        "until": until, "kind": "daily", "reason": "daily quota used up (20/day) — resets 12:31"}}}))
    monkeypatch.setattr(srv, "_llm_events", [
        {"t": time.time(), "provider": "gemini", "model": "gemini-3.8-flash", "ok": False,
         "error_detail": "429 · PerDay quota"},
        {"t": time.time(), "provider": "nvidia", "model": "nvidia/nemotron-3-super-120b-a12b", "ok": True}])
    rows = {r["model"]: r for r in srv.model_status()["models"]}
    gemini = rows["gemini-3.8-flash"]
    assert gemini["status"] == "daily" and gemini["until"] == until and "20/day" in gemini["reason"]
    assert gemini["failed"] == 1 and gemini["last_error"] == "429 · PerDay quota"
    assert rows["nvidia/nemotron-3-super-120b-a12b"]["status"] == "ready"
    assert rows["nvidia/nemotron-3-super-120b-a12b"]["ok"] == 1
    assert rows["openai/gpt-oss-120b"]["status"] == "no_key"


@pytest.fixture
def site(tmp_path):
    """A stand-in for a built app: echoes the Cookie header it receives."""
    seen = {}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            seen["cookie"] = self.headers.get("cookie", "")
            body = b"<h1>Portfolio</h1>" if self.path == "/" else b"missing"
            self.send_response(200 if self.path == "/" else 404)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass
    httpd = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}", seen
    httpd.shutdown()


def test_preview_relays_the_app_without_the_dashboard_code(site, monkeypatch):
    target, seen = site
    monkeypatch.setattr(srv, "_preview", {"name": "demo", "target": target})
    monkeypatch.setattr(srv.apps, "info", lambda name: {"running": True})
    monkeypatch.setattr(srv, "_preview_authorized", lambda conn: True)
    client = TestClient(srv.preview_app)
    r = client.get("/", cookies={"autodev_token": "secret-code", "session": "abc"})
    assert r.status_code == 200 and "Portfolio" in r.text
    assert "secret-code" not in seen["cookie"] and "session=abc" in seen["cookie"]
    assert client.get("/nope").status_code == 404


def test_preview_is_locked_for_strangers_and_explains_when_nothing_runs(monkeypatch):
    client = TestClient(srv.preview_app)
    monkeypatch.setattr(srv, "_preview_authorized", lambda conn: False)
    assert client.get("/").status_code == 401
    monkeypatch.setattr(srv, "_preview_authorized", lambda conn: True)
    monkeypatch.setattr(srv, "_preview", {"name": "", "target": ""})
    r = client.get("/")
    assert r.status_code == 503 and "Preview" in r.text
