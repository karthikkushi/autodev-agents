"""
Sync bridge — agents call log() / state() from synchronous code.
Uses HTTP POST so it works whether the server is in-process or a separate process.
Silently no-ops if the server isn't running.
"""
import json
import urllib.request
import urllib.error

SERVER_URL = "http://localhost:8081"


def _post(path: str, data: dict):
    try:
        body = json.dumps(data).encode()
        req = urllib.request.Request(
            SERVER_URL + path,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=1)
    except Exception:
        pass  # server not running — silently skip


def log(agent: str, message: str):
    """Broadcast a log line to all WebSocket clients (phone + dashboard)."""
    _post("/internal/log", {"agent": agent, "message": message})


def state(updates: dict):
    """Push pipeline state updates to all WebSocket clients."""
    _post("/internal/state", updates)


def llm_event(event: dict):
    """One LLM call's outcome (provider, ok, latency, tokens) for the
    dashboard's provider charts."""
    _post("/internal/llm", event)
