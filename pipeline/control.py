"""
Pause / resume / offline control, shared by the pipeline worker and the
dashboard server (separate processes — they talk through files in the repo
root, the same way the GitHub approval flags work).

- Pause:   the dashboard creates .pipeline_paused; the pipeline stops before
           its next agent step or LLM call and waits. The flag survives
           restarts, so a paused project stays paused after a reboot.
- Offline: no internet counts as an automatic pause; the pipeline carries on
           by itself once the connection is back.
- Durable: every graph step is checkpointed to logs/checkpoints.sqlite, so
           after a power cut or shutdown the worker resumes from the last
           finished step instead of starting over.
"""
import json
import os
import socket
import threading
import time
from datetime import datetime
from pathlib import Path

from rich.console import Console

console = Console()

REPO_ROOT = Path(__file__).resolve().parent.parent
PAUSE_FLAG = REPO_ROOT / ".pipeline_paused"
CHECKPOINT_DB = REPO_ROOT / "logs" / "checkpoints.sqlite"
PROJECTS_DIR = REPO_ROOT / "projects"
STATUS_FILE = ".autodev_status.json"

_ONLINE_HOSTS = [("generativelanguage.googleapis.com", 443), ("1.1.1.1", 443)]
_ONLINE_CACHE_SECONDS = 10
_online_cache = {"at": 0.0, "value": True}
_status_lock = threading.Lock()


def is_paused() -> bool:
    return PAUSE_FLAG.exists()


def pause(reason: str = "user") -> None:
    PAUSE_FLAG.write_text(json.dumps({"reason": reason, "at": datetime.now().isoformat()}))


def resume() -> None:
    PAUSE_FLAG.unlink(missing_ok=True)


def is_online(force: bool = False) -> bool:
    """True if any provider-ish host accepts a TCP connection. Cached briefly
    because it runs before every LLM call."""
    now = time.time()
    if not force and now - _online_cache["at"] < _ONLINE_CACHE_SECONDS:
        return _online_cache["value"]
    value = False
    for host, port in _ONLINE_HOSTS:
        try:
            socket.create_connection((host, port), timeout=3).close()
            value = True
            break
        except OSError:
            continue
    _online_cache.update(at=now, value=value)
    return value


def wait_if_paused(where: str) -> None:
    """Block while the user has paused the pipeline or the internet is down.
    Called before every agent step and every LLM call."""
    announced = ""
    while True:
        if is_paused():
            reason = "paused"
        elif not is_online(force=bool(announced)):
            reason = "offline"
        else:
            break
        if reason != announced:
            from server.bridge import log as blog, state as bstate
            msg = ("⏸  Paused — waiting for Resume" if reason == "paused"
                   else "📡 Internet lost — paused until the connection is back")
            console.print(f"[yellow]{msg} (before {where})[/yellow]")
            blog("system", f"{msg} (before {where})")
            bstate({"paused": True, "pause_reason": reason})
            announced = reason
        time.sleep(5)
    if announced:
        from server.bridge import log as blog, state as bstate
        console.print(f"[green]▶️  Resumed ({where})[/green]")
        blog("system", f"▶️ Resumed — continuing with {where}")
        bstate({"paused": False, "pause_reason": ""})


def write_status(project_path: str, **fields) -> None:
    """Per-project status file the dashboard's project list reads. Merged, so
    callers only pass what changed. Locked + atomic replace because parallel
    graph branches write it at the same moment."""
    path = Path(project_path) / STATUS_FILE
    if not path.parent.is_dir():
        return  # project folder not created yet (before intake)
    with _status_lock:
        try:
            current = json.loads(path.read_text()) if path.exists() else {}
        except Exception:
            current = {}
        current.update(fields, updated_at=datetime.now().isoformat(timespec="seconds"))
        tmp = path.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(current, indent=2))
            os.replace(tmp, path)
        except OSError:
            pass


SETTINGS_FILE = REPO_ROOT / "logs" / "settings.json"


def settings() -> dict:
    try:
        return json.loads(SETTINGS_FILE.read_text())
    except Exception:
        return {}


def save_setting(key: str, value) -> None:
    s = settings()
    s[key] = value
    SETTINGS_FILE.parent.mkdir(exist_ok=True)
    SETTINGS_FILE.write_text(json.dumps(s, indent=2))


def push_to_phone(title: str, message: str, tags: str = "robot") -> bool:
    """Push notification via ntfy (free, open source) when the user has set a
    topic in the dashboard — reaches the phone even with the dashboard closed.
    Off unless configured: messages pass through ntfy's public server (only
    project names and status; the random topic acts as the password)."""
    topic = settings().get("ntfy_topic", "").strip()
    if not topic:
        return False
    import ssl
    import urllib.request
    try:
        # python.org's macOS Python ships without root certificates, so plain
        # urllib HTTPS fails verification; use certifi's bundle.
        import certifi
        context = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        context = ssl.create_default_context()
    try:
        req = urllib.request.Request(
            f"https://ntfy.sh/{topic}", data=message.encode(), method="POST",
            headers={"Title": title.encode("ascii", "ignore").decode() or "AutoDev", "Tags": tags},
        )
        urllib.request.urlopen(req, timeout=8, context=context)
        return True
    except Exception:
        return False


def notify(title: str, message: str, tags: str = "robot") -> None:
    """Mac notification plus a phone push when configured — a multi-day run
    is mostly unattended."""
    import subprocess
    script = f"display notification {json.dumps(message)} with title {json.dumps(title)}"
    try:
        subprocess.run(["osascript", "-e", script], timeout=5, capture_output=True)
    except Exception:
        pass
    push_to_phone(title, message, tags)


def read_status(project_path: Path) -> dict:
    try:
        return json.loads((project_path / STATUS_FILE).read_text())
    except Exception:
        return {}
