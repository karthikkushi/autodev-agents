"""AutoDev control server — dashboard, WebSocket live feed, and every button.

The pipeline runs as a separate worker process (python3 main.py --worker)
that this server starts, stops, pauses and resumes, so a Stop never takes the
dashboard down with it and a restarted server adopts a worker still running.
"""
import asyncio, hmac, ipaddress, json, os, plistlib, re, secrets, shutil, signal, socket, sqlite3, subprocess, sys, threading, time
from collections import deque
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse
import psutil, uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))
from pipeline import control  # noqa: E402

APPROVAL_FLAG = BASE_DIR / ".github_approved"
REJECT_FLAG = BASE_DIR / ".github_rejected"     # agents/git_github.py ends the run without pushing
PENDING_FLAG = BASE_DIR / ".pending_approval"
DESIGN_INBOX = BASE_DIR / "design_inbox"
DESIGN_INBOX.mkdir(exist_ok=True)
PROJECTS_DIR = BASE_DIR / "projects"
PROJECTS_DIR.mkdir(exist_ok=True)
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
WORKER_LOG = LOG_DIR / "pipeline.log"
WORKER_PID = LOG_DIR / "worker.pid"
LLM_LOG = LOG_DIR / "llm_calls.jsonl"
COOLDOWN_FILE = LOG_DIR / "provider_cooldowns.json"   # written by router/llm_router.py
LAUNCH_AGENT = Path.home() / "Library" / "LaunchAgents" / "com.autodev.api.plist"
TOKEN_FILE = LOG_DIR / "access_token"
SKIP_DIRS = {".git", "__pycache__", ".memory", "node_modules", ".pytest_cache", ".venv", "venv"}
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
PORT = 8081

app = FastAPI(title="AutoDev Pipeline Server")
# No CORS middleware on purpose: the dashboard is served from this same
# origin, and allowing any origin let any web page in the Mac's browser press
# these buttons.


# ── Access control ────────────────────────────────────────────────────────────
# The Mac itself and the user's own Tailscale devices need no code. Anything
# else that can reach 0.0.0.0:8081 (another device on the same Wi-Fi) needs
# the access code. The pipeline runs model-written code, so an open port
# would let anyone run code on this Mac by uploading a design.

def _load_token() -> str:
    try:
        token = TOKEN_FILE.read_text().strip()
    except OSError:
        token = ""
    if not token:
        token = secrets.token_urlsafe(18)
        TOKEN_FILE.write_text(token)
        TOKEN_FILE.chmod(0o600)
    return token


_token = {"value": _load_token()}
LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost"}
LOCAL_ORIGINS = {f"http://localhost:{PORT}", f"http://127.0.0.1:{PORT}"}
PROXY_HEADERS = ("cf-connecting-ip", "cf-ray", "x-forwarded-for")


def _lan_ips() -> list[str]:
    ips = []
    for addrs in psutil.net_if_addrs().values():
        for a in addrs:
            if a.family == socket.AF_INET and not a.address.startswith(("127.", "169.254.")):
                ips.append(a.address)
    return ips


def _is_local(conn) -> bool:
    """Typed on this Mac: loopback client, not relayed through a local proxy,
    and not a cross-site request from another page open in the Mac's browser."""
    if not conn.client or conn.client.host not in LOCAL_HOSTS:
        return False
    if any(h in conn.headers for h in PROXY_HEADERS):
        return False
    origin = conn.headers.get("origin")
    return origin is None or origin in LOCAL_ORIGINS


TAILNET = [ipaddress.ip_network("100.64.0.0/10"), ipaddress.ip_network("fd7a:115c:a1e0::/48")]


def _via_tailscale(conn) -> bool:
    """Both ends on Tailscale addresses. Tailscale only routes packets from the
    user's own signed-in devices, and a LAN device can't fake a 100.x source
    because this Mac's replies to it would go out over Tailscale."""
    try:
        client = ipaddress.ip_address(conn.client.host)
        server = ipaddress.ip_address((conn.scope.get("server") or ("", 0))[0])
    except (ValueError, AttributeError):
        return False
    return any(client in net for net in TAILNET) and any(server in net for net in TAILNET)


def _token_of(conn) -> str:
    return (conn.headers.get("x-autodev-token") or conn.query_params.get("token")
            or conn.cookies.get("autodev_token") or "")


def _authorized(conn) -> bool:
    if _is_local(conn):
        return True
    origin = conn.headers.get("origin")
    if origin and urlparse(origin).netloc != conn.headers.get("host", ""):
        return False  # another site's page in the phone's browser — never allowed
    if _via_tailscale(conn):
        return True
    token = _token_of(conn)
    return bool(token) and hmac.compare_digest(token, _token["value"])


@app.middleware("http")
async def require_access(request: Request, call_next):
    # The dashboard files themselves aren't secret — without the code the page
    # just shows its lock screen, because every /api call is refused.
    if not (request.url.path.startswith("/ui") or _authorized(request)):
        return JSONResponse({"detail": "locked"}, status_code=401)
    response = await call_next(request)
    query_token = request.query_params.get("token")
    if query_token and hmac.compare_digest(query_token, _token["value"]):
        # Scanning the QR opens /ui/?token=… — keep it as a cookie so every
        # later fetch and the WebSocket carry it without the page seeing it.
        secure = request.headers.get("x-forwarded-proto") == "https" or request.url.scheme == "https"
        response.set_cookie("autodev_token", query_token, max_age=180 * 86400,
                            httponly=True, samesite="lax", secure=secure)
    return response


def _tailscale() -> dict:
    """What the phone needs: is Tailscale up on this Mac, and at which address.
    The 100.x address is permanent — it doesn't change between restarts."""
    exe = shutil.which("tailscale") or "/usr/local/bin/tailscale"
    if not Path(exe).exists():
        return {"installed": False, "state": "not_installed"}
    try:
        out = subprocess.run([exe, "status", "--json"], capture_output=True, text=True, timeout=5)
        data = json.loads(out.stdout)
    except Exception:
        return {"installed": True, "state": "daemon_stopped"}
    me = data.get("Self") or {}
    ips = me.get("TailscaleIPs") or []
    return {
        "installed": True,
        "state": data.get("BackendState", ""),  # Running / NeedsLogin / Stopped
        "ip": next((i for i in ips if "." in i), ""),
        "dns": (me.get("DNSName") or "").rstrip("."),
        "auth_url": data.get("AuthURL", ""),
    }


dashboard_dir = BASE_DIR / "dashboard"
if dashboard_dir.exists():
    app.mount("/ui", StaticFiles(directory=str(dashboard_dir), html=True), name="dashboard")


class ConnectionManager:
    def __init__(self): self.active: list[WebSocket] = []
    async def connect(self, ws): await ws.accept(); self.active.append(ws)
    def disconnect(self, ws):
        if ws in self.active: self.active.remove(ws)
    async def broadcast(self, data):
        dead = []
        for ws in self.active:
            try: await ws.send_text(json.dumps(data))
            except: dead.append(ws)
        for ws in dead: self.disconnect(ws)


manager = ConnectionManager()
_pipeline_state = {
    "type": "state", "phase": "idle", "current_agent": "", "project_name": "",
    "cpu": 0.0, "ram": 0.0, "ram_free_gb": 0.0, "cpu_temp": None,
    "github_ready": False, "github_approved": False, "is_complete": False,
    "last_activity": None, "paused": False, "pause_reason": "",
}
_log_history: list[dict] = []                 # last 500 log entries, replayed to new clients
_health_history: deque = deque(maxlen=900)    # 30 min of 2-second samples, for the charts
_llm_events: deque = deque(maxlen=3000)       # recent LLM calls, for the provider charts
_net = {"online": True}


# ── Pipeline worker process ───────────────────────────────────────────────────

class PipelineWorker:
    """Start/stop the `main.py --worker` process. The worker owns
    logs/worker.pid, so a worker started before a server restart is adopted
    rather than duplicated."""

    def __init__(self):
        self.proc: subprocess.Popen | None = None

    def pid(self) -> int | None:
        if self.proc is not None:
            self.proc.poll()  # reap our own exited child so it doesn't linger as a zombie
        try:
            pid = int(WORKER_PID.read_text().strip())
            cmd = psutil.Process(pid).cmdline()
        except (OSError, ValueError, psutil.Error):
            return None
        # After a reboot the old pid can belong to an unrelated process.
        return pid if any("main.py" in c for c in cmd) and "--worker" in cmd else None

    def running(self) -> bool:
        return self.pid() is not None

    def start(self) -> bool:
        if self.running():
            return False
        log = open(WORKER_LOG, "a", buffering=1)
        log.write(f"\n===== worker started {datetime.now():%Y-%m-%d %H:%M:%S} =====\n")
        self.proc = subprocess.Popen(
            [sys.executable, str(BASE_DIR / "main.py"), "--worker"],
            cwd=str(BASE_DIR), stdout=log, stderr=subprocess.STDOUT,
        )
        for _ in range(50):  # the worker writes its pid file right after start-up
            if self.running() or self.proc.poll() is not None:
                break
            time.sleep(0.1)
        return True

    def stop(self) -> bool:
        pid = self.pid()
        if pid is None:
            return False
        os.kill(pid, signal.SIGTERM)
        for _ in range(150):
            if self.pid() is None:
                break
            time.sleep(0.1)
        else:
            os.kill(pid, signal.SIGKILL)
        WORKER_PID.unlink(missing_ok=True)  # SIGTERM skips the worker's atexit cleanup
        return True

    def info(self) -> dict:
        pid = self.pid()
        exit_code = self.proc.returncode if self.proc is not None and pid is None else None
        return {"running": pid is not None, "pid": pid, "last_exit": exit_code}


worker = PipelineWorker()


# ── Built-app runner ("Run app" button) ───────────────────────────────────────

def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class AppRunner:
    def __init__(self):
        self.apps: dict[str, dict] = {}

    def info(self, name: str) -> dict:
        a = self.apps.get(name)
        if a and a["proc"].poll() is None:
            return {"running": True, "url": a["url"]}
        return {"running": False}

    def start(self, name: str) -> dict:
        if self.info(name)["running"]:
            return self.info(name)
        src = _project_dir(name) / "src"
        port = _free_port()
        # Explicit port: Flask's default 5000 is taken by AirPlay Receiver on
        # macOS. app_command also picks the project's own venv when it has one.
        from tools.webapp import app_command
        cmd = app_command(src, port)
        if not cmd:
            raise HTTPException(400, "No web app found — expected src/app.py (Flask/FastAPI) or src/index.html")
        log_path = LOG_DIR / f"app_{name}.log"
        # The generated app may only write inside its project, and never sees
        # the pipeline's API keys (it used to get the whole environment).
        from tools.safety import clean_env, sandbox_prefix
        proc = subprocess.Popen(sandbox_prefix(str(src.parent)) + cmd, cwd=str(src), stdout=open(log_path, "w"),
                                stderr=subprocess.STDOUT,
                                env=clean_env({"PYTHONPATH": str(src), "PORT": str(port)}))
        for _ in range(100):
            if proc.poll() is not None:
                tail = ANSI.sub("", log_path.read_text(errors="replace"))[-600:]
                raise HTTPException(500, f"The app exited on start-up:\n{tail}")
            try:
                socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
                break
            except OSError:
                time.sleep(0.1)
        url = f"http://127.0.0.1:{port}"
        self.apps[name] = {"proc": proc, "url": url}
        return {"running": True, "url": url}

    def stop(self, name: str) -> bool:
        a = self.apps.pop(name, None)
        if a and a["proc"].poll() is None:
            a["proc"].terminate()
            return True
        return False

    def stop_all(self):
        for name in list(self.apps):
            self.stop(name)


apps = AppRunner()


# ── Health + metrics ──────────────────────────────────────────────────────────

def _cpu_temp() -> float | None:
    """Get CPU temperature on macOS using osx-cpu-temp."""
    try:
        out = subprocess.check_output(["osx-cpu-temp"], timeout=3, text=True)
        return float(out.strip().replace("°C", "").replace("C", "").strip())
    except Exception:
        pass
    try:
        temps = psutil.sensors_temperatures()
        if temps:
            for _, entries in temps.items():
                if entries:
                    return round(entries[0].current, 1)
    except Exception:
        pass
    return None


def _pause_reason_from_flag() -> str:
    try:
        return json.loads(control.PAUSE_FLAG.read_text()).get("reason", "user")
    except Exception:
        return "user" if control.is_paused() else ""


def _health():
    mem = psutil.virtual_memory()
    result = {
        "cpu": psutil.cpu_percent(interval=0.2),
        "ram": mem.percent,
        "ram_free_gb": round(mem.available / (1024 ** 3), 2),
        "github_ready": PENDING_FLAG.exists() and not REJECT_FLAG.exists(),
        "github_approved": APPROVAL_FLAG.exists(),
        "worker_running": worker.running(),
        "user_paused": control.is_paused(),
        "online": _net["online"],
    }
    temp = _cpu_temp()
    if temp is not None:
        result["cpu_temp"] = temp
        result["cpu_temp_c"] = temp  # name the phone app reads
    # For the phone app: a multi-day run on battery is worth watching.
    battery = psutil.sensors_battery()
    result["battery_percent"] = round(battery.percent) if battery else None
    result["battery_charging"] = bool(battery and battery.power_plugged)
    result["auto_paused_no_internet"] = not _net["online"]
    return result


def _load_llm_history():
    if not LLM_LOG.exists():
        return
    try:
        for line in LLM_LOG.read_text().splitlines()[-_llm_events.maxlen:]:
            _llm_events.append(json.loads(line))
    except Exception:
        pass


async def _health_broadcast_loop():
    last_net_check = 0.0
    while True:
        if time.time() - last_net_check > 15:
            last_net_check = time.time()
            _net["online"] = await asyncio.to_thread(control.is_online, True)
        h = await asyncio.to_thread(_health)
        _pipeline_state.update(h)
        _health_history.append({"t": time.time(), "cpu": h["cpu"], "ram": h["ram"], "temp": h.get("cpu_temp")})
        await manager.broadcast({**_pipeline_state, "type": "health"})
        await asyncio.sleep(2)


@app.on_event("startup")
async def on_startup():
    _load_llm_history()
    asyncio.create_task(_health_broadcast_loop())
    _start_preview_server()
    if os.environ.get("AUTODEV_AUTOSTART_WORKER") == "1":
        # Unfinished projects resume from their checkpoint; a user pause
        # (.pipeline_paused) survives restarts, so a paused run stays paused.
        await asyncio.to_thread(worker.start)


@app.on_event("shutdown")
def on_shutdown():
    apps.stop_all()
    if worker.proc is not None:
        worker.stop()


# ── WebSocket + existing endpoints ────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    if not _authorized(ws):  # HTTP middleware doesn't run for WebSockets
        # Accept first so the browser receives code 4401 and the dashboard
        # shows its lock screen; a refused handshake only reads as 1006.
        await ws.accept()
        await ws.close(code=4401)
        return
    await manager.connect(ws)
    # Send current state first
    await ws.send_text(json.dumps({**_pipeline_state, "type": "state"}))
    # Replay log history so a late-connecting client sees everything
    for entry in _log_history:
        try:
            await ws.send_text(json.dumps(entry))
        except Exception:
            break
    try:
        while True:
            msg = json.loads(await ws.receive_text())
            if msg.get("type") == "approve_github":
                APPROVAL_FLAG.touch()
                _pipeline_state["github_approved"] = True
                await manager.broadcast({
                    "type": "notification", "agent": "system",
                    "message": "GitHub push approved.",
                    "time": time.strftime("%Y-%m-%dT%H:%M:%S")
                })
            elif msg.get("type") == "reject_github":
                await reject_github()
    except WebSocketDisconnect:
        manager.disconnect(ws)


@app.get("/health")
def health_check():
    return _health()


@app.get("/state")
def get_state():
    return _pipeline_state


@app.post("/approve-github")
async def approve_github():
    REJECT_FLAG.unlink(missing_ok=True)
    APPROVAL_FLAG.touch()
    _pipeline_state["github_approved"] = True
    await manager.broadcast({
        "type": "notification", "agent": "system",
        "message": "GitHub push approved via REST.",
        "time": time.strftime("%Y-%m-%dT%H:%M:%S")
    })
    return {"ok": True}


@app.post("/reject-github")
async def reject_github():
    """The build isn't good enough: the pipeline ends the run without pushing
    (the project stays committed on this Mac and can be rebuilt)."""
    APPROVAL_FLAG.unlink(missing_ok=True)
    REJECT_FLAG.touch()
    _pipeline_state["github_approved"] = False
    _pipeline_state["github_ready"] = False
    await manager.broadcast({
        "type": "notification", "agent": "system",
        "message": "GitHub push rejected — nothing will be pushed.",
        "time": time.strftime("%Y-%m-%dT%H:%M:%S")
    })
    return {"ok": True}


@app.post("/upload-design")
async def upload_design(file: UploadFile = File(...)):
    name = Path(file.filename or "").name
    if not name.endswith(".md"):
        raise HTTPException(400, "Only .md files accepted")
    (DESIGN_INBOX / name).write_bytes(await file.read())
    await manager.broadcast({
        "type": "notification", "agent": "system",
        "message": f"Design uploaded: {name}",
        "time": time.strftime("%Y-%m-%dT%H:%M:%S")
    })
    return {"ok": True, "filename": name}


# ── Control buttons ───────────────────────────────────────────────────────────

def _control_state() -> dict:
    return {
        "worker": worker.info(),
        "user_paused": control.is_paused(),
        "pause_reason": _pause_reason_from_flag(),
        "pipeline_paused": _pipeline_state.get("paused", False),
        "pipeline_pause_reason": _pipeline_state.get("pause_reason", ""),
        "online": _net["online"],
        "autostart": LAUNCH_AGENT.exists(),
    }


@app.get("/api/control")
def get_control():
    return _control_state()


@app.post("/api/control/{action}")
async def do_control(action: str):
    if action == "start":
        await asyncio.to_thread(worker.start)
    elif action == "stop":
        await asyncio.to_thread(worker.stop)
    elif action == "restart":
        await asyncio.to_thread(worker.stop)
        await asyncio.to_thread(worker.start)
    elif action == "pause":
        control.pause("user")
    elif action == "resume":
        control.resume()
    else:
        raise HTTPException(404, f"Unknown action '{action}'")
    labels = {"start": "▶ Worker started", "stop": "■ Worker stopped — progress saved",
              "restart": "↻ Worker restarted", "pause": "⏸ Pause requested — stops before the next step",
              "resume": "▶ Resumed"}
    await push_log("system", labels[action])
    state = _control_state()
    await manager.broadcast({"type": "control", **state})
    return state


@app.post("/control/{action}")
async def phone_app_control(action: str):
    """The AutoDev phone app's Pause / Resume / Terminate buttons (it calls these
    paths). Terminate stops the worker; progress is saved and resumes later."""
    mapping = {"pause": "pause", "resume": "resume", "terminate": "stop"}
    if action not in mapping:
        raise HTTPException(404, f"Unknown action '{action}'")
    return await do_control(mapping[action])


# ── Projects ──────────────────────────────────────────────────────────────────

def _project_name(design_filename: str) -> str:
    return design_filename.replace(".md", "").replace(" ", "_").lower()


def _project_dir(name: str) -> Path:
    if not name or "/" in name or name.startswith("."):
        raise HTTPException(404, "Unknown project")
    d = PROJECTS_DIR / name
    if not d.is_dir():
        raise HTTPException(404, "Unknown project")
    return d


def _list_projects() -> list[dict]:
    names: dict[str, Path | None] = {}
    for d in PROJECTS_DIR.iterdir():
        if d.is_dir() and not d.name.startswith((".", "_")):
            names[d.name] = d
    designs = {_project_name(md.name): md.name for md in DESIGN_INBOX.glob("*.md")}
    for n in designs:
        names.setdefault(n, None)

    running, paused = worker.running(), control.is_paused()
    out = []
    for name, d in sorted(names.items()):
        st = control.read_status(d) if d else {}
        status = st.get("status") or ("queued" if d is None else "unknown")
        if status == "building":
            status = "stopped" if not running else ("paused" if paused else "building")
        out.append({
            "name": name,
            "status": status,
            "step": st.get("step", ""),
            "phase": st.get("phase", ""),
            "task_index": st.get("task_index", 0),
            "task_total": st.get("task_total", 0),
            "updated_at": st.get("updated_at", ""),
            "known_issues": st.get("known_issues"),
            "ui_score": st.get("ui_score"),
            "error": st.get("error", ""),
            "design": designs.get(name, ""),
            "has_folder": d is not None,
            "app": apps.info(name),
        })
    return out


@app.get("/api/projects")
def list_projects():
    return _list_projects()


@app.get("/api/projects/{name}/files")
def project_files(name: str):
    d = _project_dir(name)
    files = []
    for root, dirs, fnames in os.walk(d):
        dirs[:] = sorted(x for x in dirs if x not in SKIP_DIRS)
        for f in sorted(fnames):
            p = Path(root) / f
            files.append({"path": str(p.relative_to(d)), "size": p.stat().st_size})
            if len(files) >= 500:
                return files
    return files


@app.get("/api/projects/{name}/file")
def project_file(name: str, path: str):
    d = _project_dir(name).resolve()
    p = (d / path).resolve()
    if not p.is_relative_to(d) or not p.is_file():
        raise HTTPException(404, "File not found")
    if p.stat().st_size > 300_000:
        raise HTTPException(413, "File too large to preview")
    return {"path": path, "content": p.read_text(errors="replace")}


_IMAGE_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif",
                ".webp": "image/webp", ".svg": "image/svg+xml"}


@app.get("/api/projects/{name}/image")
def project_image(name: str, path: str):
    """Images (visual QA screenshots, assets) for the dashboard's file viewer."""
    d = _project_dir(name).resolve()
    p = (d / path).resolve()
    kind = _IMAGE_TYPES.get(p.suffix.lower())
    if not kind or not p.is_relative_to(d) or not p.is_file():
        raise HTTPException(404, "Image not found")
    return Response(p.read_bytes(), media_type=kind, headers={"Cache-Control": "no-store"})


@app.post("/api/projects/{name}/open")
def open_project_folder(name: str):
    subprocess.Popen(["open", str(_project_dir(name))])
    return {"ok": True}


@app.post("/api/projects/{name}/rebuild")
async def rebuild_project(name: str):
    """Start a project over: forget its checkpoints and move the old folder to
    projects/.trash (recoverable), then let the worker build it again."""
    designs = {_project_name(md.name) for md in DESIGN_INBOX.glob("*.md")}
    if name not in designs:
        raise HTTPException(400, "Its design file is no longer in design_inbox — upload it again first")
    was_running = await asyncio.to_thread(worker.running)
    await asyncio.to_thread(worker.stop)
    apps.stop(name)
    from langgraph.checkpoint.sqlite import SqliteSaver
    if control.CHECKPOINT_DB.exists():
        conn = sqlite3.connect(str(control.CHECKPOINT_DB))
        SqliteSaver(conn).delete_thread(name)
        conn.close()
    d = PROJECTS_DIR / name
    if control.read_status(d).get("phase") == "awaiting_github_approval":
        # Its push question goes with it — the old banner would otherwise
        # stay up for the whole new build.
        for flag in (PENDING_FLAG, APPROVAL_FLAG, REJECT_FLAG):
            flag.unlink(missing_ok=True)
    if d.is_dir():
        trash = PROJECTS_DIR / ".trash"
        trash.mkdir(exist_ok=True)
        shutil.move(str(d), str(trash / f"{name}-{datetime.now():%Y%m%d-%H%M%S}"))
    if was_running:
        await asyncio.to_thread(worker.start)
    await push_log("system", f"↻ Rebuilding {name} from scratch (old folder moved to projects/.trash)")
    return {"ok": True}


@app.post("/api/projects/{name}/run")
async def run_project_app(name: str):
    return await asyncio.to_thread(apps.start, name)


@app.post("/api/projects/{name}/preview")
async def preview_project(name: str):
    """Start the project's app (if needed) and show it on the preview port,
    which the dashboard embeds — reachable from the phone over Tailscale,
    unlike the app itself, which only listens on this Mac."""
    info = await asyncio.to_thread(apps.start, name)
    _preview.update(name=name, target=info["url"])
    return {"name": name, "running": True, "port": PREVIEW_PORT}


@app.get("/api/preview")
def preview_status():
    name = _preview["name"]
    return {"name": name, "running": bool(name) and apps.info(name)["running"], "port": PREVIEW_PORT}


@app.post("/api/projects/{name}/stop-app")
def stop_project_app(name: str):
    return {"ok": apps.stop(name)}


# ── Metrics, raw log, settings ────────────────────────────────────────────────

@app.get("/api/metrics")
def metrics():
    return {"health": list(_health_history), "llm": list(_llm_events)}


def _role_models() -> dict:
    """{model id: {"env": key name, "roles": [...]}} in routing order, read from
    router/roles.py on its own (importing the router package loads LiteLLM)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("autodev_roles", BASE_DIR / "router" / "roles.py")
    roles = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(roles)
    models = {}
    for role, deployments in roles.ROLE_DEPLOYMENTS.items():
        for model, env_key, _rpm in deployments:
            models.setdefault(model, {"env": env_key, "roles": []})["roles"].append(role)
    return models


@app.get("/api/models")
def model_status():
    """Every model the pipeline routes to and whether it can answer right now:
    ready, at its daily limit (with the reset time), resting after a rate
    limit or error, or missing its API key — plus its last 24 h of calls.
    Cooldowns come from the file the worker and the benchmark share."""
    now = time.time()
    try:
        cooldowns = json.loads(COOLDOWN_FILE.read_text()).get("models", {})
    except (OSError, ValueError, AttributeError):
        cooldowns = {}
    day = [e for e in list(_llm_events) if e.get("t", 0) >= now - 86400]
    rows = []
    for model, info in _role_models().items():
        prefix, short = model.split("/", 1)
        provider = {"nvidia_nim": "nvidia", "ollama_chat": "ollama"}.get(prefix, prefix)
        mine = [e for e in day if e.get("provider") == provider and e.get("model") == short]
        cd = cooldowns.get(model) if isinstance(cooldowns.get(model), dict) else {}
        until = float(cd.get("until") or 0)
        resting = until > now
        status = ("no_key" if not os.environ.get(info["env"])
                  else (cd.get("kind") or "backoff") if resting else "ready")
        failed = [e for e in mine if not e.get("ok")]
        rows.append({"model": short, "provider": provider, "roles": info["roles"], "status": status,
                     "until": until if resting else None, "reason": cd.get("reason", "") if resting else "",
                     "ok": len(mine) - len(failed), "failed": len(failed),
                     "last_error": (failed[-1].get("error_detail") or failed[-1].get("error", ""))[:160] if failed else ""})
    return {"models": rows}


@app.get("/api/logs/tail")
def logs_tail(lines: int = 300):
    if not WORKER_LOG.exists():
        return {"text": ""}
    with open(WORKER_LOG, "rb") as f:
        f.seek(0, 2)
        f.seek(max(0, f.tell() - 400_000))
        text = f.read().decode(errors="replace")
    text = ANSI.sub("", text)
    return {"text": "\n".join(text.splitlines()[-max(1, min(lines, 2000)):])}


@app.post("/api/settings/autostart")
async def set_autostart(req: Request):
    """Login item so a power cut or reboot resumes on its own: macOS starts
    this server at login, the server starts the worker, the worker resumes."""
    enabled = bool((await req.json()).get("enabled"))
    if enabled:
        LAUNCH_AGENT.parent.mkdir(parents=True, exist_ok=True)
        LAUNCH_AGENT.write_bytes(plistlib.dumps({
            "Label": "com.autodev.api",
            "ProgramArguments": [sys.executable, str(BASE_DIR / "main.py"), "--no-browser"],
            "WorkingDirectory": str(BASE_DIR),
            "RunAtLoad": True,
            "StandardOutPath": str(LOG_DIR / "server.log"),
            "StandardErrorPath": str(LOG_DIR / "server.log"),
            # launchd's PATH is minimal — the worker needs python3/git/npm
            "EnvironmentVariables": {"PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin")},
        }))
    else:
        LAUNCH_AGENT.unlink(missing_ok=True)
    return {"autostart": LAUNCH_AGENT.exists()}


@app.get("/api/settings/notify")
def get_notify():
    return {"topic": control.settings().get("ntfy_topic", "")}


@app.post("/api/settings/notify")
async def set_notify(req: Request):
    """Phone push notifications through ntfy — topic "" turns them off."""
    topic = str((await req.json()).get("topic", "")).strip()
    if topic and not re.fullmatch(r"[A-Za-z0-9_-]{8,64}", topic):
        raise HTTPException(400, "Topic: 8-64 letters, digits, - or _ (it works like a password)")
    control.save_setting("ntfy_topic", topic)
    return {"topic": topic}


@app.post("/api/settings/notify/test")
def test_notify():
    ok = control.push_to_phone("AutoDev test", "Notifications work — you'll hear from AutoDev here.", "bell")
    if not ok:
        raise HTTPException(400, "Couldn't send — set a topic first, and check the internet connection")
    return {"ok": True}


# ── Phone access ──────────────────────────────────────────────────────────────

def _require_local(request: Request):
    # The access code and its QR are only ever shown on the Mac itself.
    if not _is_local(request):
        raise HTTPException(403, "Only available on the Mac running AutoDev")


@app.get("/api/phone")
async def phone_info(request: Request):
    ts = await asyncio.to_thread(_tailscale)
    info = {"is_local": _is_local(request), "via_tailscale": _via_tailscale(request), "tailscale": ts}
    if ts.get("ip"):
        info["link"] = f"http://{ts['ip']}:{PORT}/ui/"
        info["app_address"] = f"{ts['ip']}:{PORT}"
    if info["is_local"]:
        info["code"] = _token["value"]  # for devices on the Wi-Fi without Tailscale
    return info


@app.post("/api/phone/{action}")
async def phone_action(action: str, request: Request):
    _require_local(request)
    if action == "setup":
        # Opens Terminal for the two steps that need the user themselves: the
        # Mac password (runs Tailscale at every boot) and the Google sign-in.
        script = ("sudo brew services start tailscale && sleep 3 && tailscale up && "
                  "echo && echo 'Tailscale is on. This Mac:' && tailscale ip -4")
        subprocess.Popen(["osascript", "-e", f'tell application "Terminal" to do script {json.dumps(script)}',
                          "-e", 'tell application "Terminal" to activate'])
    elif action == "new-code":
        # Signs out every browser that used the old code (Wi-Fi-without-Tailscale access).
        _token["value"] = secrets.token_urlsafe(18)
        TOKEN_FILE.write_text(_token["value"])
    else:
        raise HTTPException(404, f"Unknown action '{action}'")
    return await phone_info(request)


@app.get("/api/phone/qr.svg")
async def phone_qr():
    ts = await asyncio.to_thread(_tailscale)
    if not ts.get("ip"):
        raise HTTPException(404, "Tailscale isn't connected on this Mac yet")
    link = f"http://{ts['ip']}:{PORT}/ui/"
    import qrcode, qrcode.image.svg
    img = qrcode.make(link, image_factory=qrcode.image.svg.SvgPathImage, box_size=10, border=2)
    return Response(img.to_string(encoding="unicode"), media_type="image/svg+xml",
                    headers={"Cache-Control": "no-store"})


# ── Internal endpoints called by the pipeline worker (via bridge.py) ──────────

async def push_log(agent: str, message: str):
    entry = {"type": "log", "agent": agent, "message": message,
             "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
    _log_history.append(entry)
    if len(_log_history) > 500:
        _log_history.pop(0)
    _pipeline_state["last_activity"] = entry["time"]
    await manager.broadcast(entry)


async def push_state(updates: dict):
    _pipeline_state.update(updates)
    await manager.broadcast({**_pipeline_state, "type": "state"})


@app.post("/internal/log")
async def internal_log(req: Request):
    body = await req.json()
    agent = body.get("agent", "pipeline")
    message = body.get("message", "")
    await push_log(agent, message)
    return {"ok": True}


@app.post("/internal/state")
async def internal_state(req: Request):
    body = await req.json()
    await push_state(body)
    return {"ok": True}


@app.post("/internal/llm")
async def internal_llm(req: Request):
    event = {**(await req.json()), "t": time.time()}
    _llm_events.append(event)
    try:
        with open(LLM_LOG, "a") as f:
            f.write(json.dumps(event) + "\n")
    except OSError:
        pass
    await manager.broadcast({"type": "llm", **event})
    return {"ok": True}


# ── Live preview (port 8082) ──────────────────────────────────────────────────
# The built app listens on 127.0.0.1 only (it's model-written code). This port
# relays one app at a time to the dashboard's preview frame, behind the same
# access rules as the dashboard, so it works from the phone over Tailscale.
# Its own port rather than a /preview/ path: generated apps use absolute
# paths (/static/app.js, fetch('/api/split')) that a sub-path would break.
PREVIEW_PORT = PORT + 1
_preview = {"name": "", "target": ""}
_HOP_HEADERS = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers",
                "transfer-encoding", "upgrade", "host", "content-length", "content-encoding"}
_CACHE_HEADERS = {"cache-control", "expires", "etag", "last-modified", "if-none-match", "if-modified-since"}
preview_app = FastAPI(title="AutoDev live preview")


def _preview_authorized(conn) -> bool:
    # The previewed page's own requests carry its own origin (…:8082).
    if conn.client and conn.client.host in LOCAL_HOSTS and not any(h in conn.headers for h in PROXY_HEADERS):
        origin = conn.headers.get("origin")
        if origin is None or urlparse(origin).netloc == conn.headers.get("host", ""):
            return True
    return _authorized(conn)


@preview_app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"])
async def preview_proxy(path: str, request: Request):
    if not _preview_authorized(request):
        return Response("Locked — open the AutoDev dashboard first.", status_code=401, media_type="text/plain")
    name, target = _preview["name"], _preview["target"]
    if not name or not apps.info(name)["running"]:
        return Response("<p style='font:16px system-ui;padding:24px'>No app running — press Preview in the "
                        "AutoDev dashboard.</p>", status_code=503, media_type="text/html")
    import httpx
    # Never hand the dashboard's access code to model-written code.
    cookie = "; ".join(c for c in request.headers.get("cookie", "").split("; ")
                       if c and not c.startswith("autodev_token="))
    # Every project is served at this same address, so nothing may be cached:
    # after switching projects the browser showed the previous one's page.
    headers = {k: v for k, v in request.headers.items()
               if k.lower() not in _HOP_HEADERS | _CACHE_HEADERS | {"cookie", "x-autodev-token"}}
    if cookie:
        headers["cookie"] = cookie
    url = f"{target}/{path}" + (f"?{request.url.query}" if request.url.query else "")
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.request(request.method, url, headers=headers, content=await request.body())
    except httpx.HTTPError as e:
        return Response(f"The app didn't answer: {e}", status_code=502, media_type="text/plain")
    out = Response(content=resp.content, status_code=resp.status_code)
    for key, value in resp.headers.multi_items():
        if key.lower() in _HOP_HEADERS | _CACHE_HEADERS:
            continue
        if key.lower() == "location" and value.startswith(target):
            value = value[len(target):] or "/"
        out.headers.append(key, value)
    out.headers["cache-control"] = "no-store"
    return out


def _start_preview_server() -> None:
    config = uvicorn.Config(preview_app, host="0.0.0.0", port=PREVIEW_PORT, log_level="warning")
    threading.Thread(target=uvicorn.Server(config).run, daemon=True, name="autodev-preview").start()


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8081, reload=False)
