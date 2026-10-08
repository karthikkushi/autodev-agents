"""
Run a generated web app and screenshot it — used by the visual QA agent
(agents/ui_review.py). Unit tests can't see that a page renders badly; a real
browser render + a vision model can.

Headless Chrome is already on this Mac (no Playwright download). Screenshots
go through Chrome's DevTools protocol, which emulates the viewport exactly;
we stop Chrome ourselves afterwards (pages that hold a connection open never
let it exit).
"""
import base64
import itertools
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

BROWSERS = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def app_command(src: Path, port: int) -> list[str] | None:
    """How to serve src/app.py (with the project's own venv when it has one),
    or a plain static site (src/index.html) with Python's built-in file
    server; None if it's neither. Static sites used to get no Visual QA at
    all — an unstyled page with most of its sections missing went through."""
    app_py = src / "app.py"
    if app_py.exists():
        venv_py = src.parent / ".venv" / "bin" / "python"
        py = str(venv_py) if venv_py.exists() else sys.executable
        text = app_py.read_text(errors="replace")
        if "FastAPI(" in text:
            return [py, "-m", "uvicorn", "app:app", "--port", str(port)]
        if "Flask(" in text or "create_app" in text:
            return [py, "-m", "flask", "--app", "app", "run", "--port", str(port)]
    if (src / "index.html").exists():
        return [sys.executable, "-m", "http.server", str(port), "--bind", "127.0.0.1"]
    return None


def start_app(project_path: str, log_path: str) -> tuple[subprocess.Popen | None, str, str]:
    """(process, url, error). Waits up to 20s for the port to accept."""
    src = Path(project_path) / "src"
    port = _free_port()
    cmd = app_command(src, port)
    if not cmd:
        return None, "", "no src/app.py (Flask/FastAPI) and no src/index.html"
    # Generated app: may only write inside its project, and gets no API keys.
    from tools.safety import clean_env, sandbox_prefix
    proc = subprocess.Popen(sandbox_prefix(project_path) + cmd, cwd=str(src), stdout=open(log_path, "w"),
                            stderr=subprocess.STDOUT,
                            env=clean_env({"PYTHONPATH": str(src), "PORT": str(port)}),
                            start_new_session=True)
    for _ in range(200):
        if proc.poll() is not None:
            return None, "", "the app exited on start-up:\n" + Path(log_path).read_text(errors="replace")[-800:]
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
            return proc, f"http://127.0.0.1:{port}", ""
        except OSError:
            time.sleep(0.1)
    stop(proc)
    return None, "", "the app didn't start listening within 20s"


def stop(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=5)
    except Exception:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            pass


def browser() -> str | None:
    return next((b for b in BROWSERS if Path(b).exists()), None)


# Chrome never makes a window narrower than 500px on macOS, so the old
# --window-size=390 screenshot laid the page out at 500px and cropped it: every
# "phone" render showed a page cut off on the right that no phone would show,
# and the design critic and fixer both worked from it. DevTools device
# emulation sets the real viewport (and, for phones, honours <meta viewport>
# and touch) the way Chrome's device toolbar does.
CLI_MIN_WIDTH = 500
PHONE_MAX_WIDTH = 600


class _DevTools:
    """Just enough of the Chrome DevTools protocol for a screenshot or a page
    check. `events` has every event name seen, `messages` the full events."""

    def __init__(self, ws_url: str, deadline: float):
        from websockets.sync.client import connect
        self.ws = connect(ws_url, max_size=None, open_timeout=10)
        self.deadline = deadline
        self.events: list[str] = []
        self.messages: list[dict] = []
        self._ids = itertools.count(1)

    def _recv(self) -> dict:
        left = self.deadline - time.time()
        if left <= 0:
            raise TimeoutError("the page took too long to render")
        return json.loads(self.ws.recv(timeout=left))

    def _keep(self, msg: dict) -> None:
        if "method" in msg:
            self.events.append(msg["method"])
            self.messages.append(msg)

    def call(self, method: str, **params) -> dict:
        call_id = next(self._ids)
        self.ws.send(json.dumps({"id": call_id, "method": method, "params": params}))
        while True:
            msg = self._recv()
            if msg.get("id") == call_id:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error'].get('message')}")
                return msg.get("result", {})
            self._keep(msg)

    def wait_for(self, event: str) -> None:
        while event not in self.events:
            self._keep(self._recv())

    def close(self) -> None:
        try:
            self.ws.close()
        except Exception:
            pass


def _page_socket(port: int, deadline: float) -> str:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    while time.time() < deadline:
        try:
            with opener.open(f"http://127.0.0.1:{port}/json/list", timeout=2) as r:
                for target in json.load(r):
                    if target.get("type") == "page" and target.get("webSocketDebuggerUrl"):
                        return target["webSocketDebuggerUrl"]
        except (OSError, ValueError):
            pass
        time.sleep(0.2)
    raise TimeoutError("Chrome's DevTools port never opened")


def _screenshot_devtools(exe: str, url: str, out_path: str, width: int, height: int, timeout: int) -> bool:
    profile = tempfile.mkdtemp(prefix="autodev-chrome-")
    port = _free_port()
    proc = subprocess.Popen(
        [exe, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--no-first-run",
         "--no-default-browser-check", f"--user-data-dir={profile}", f"--remote-debugging-port={port}",
         "--window-size=1280,900", "about:blank"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True,
    )
    deadline = time.time() + timeout
    tools = None
    try:
        tools = _DevTools(_page_socket(port, deadline), deadline)
        phone = width <= PHONE_MAX_WIDTH
        tools.call("Emulation.setDeviceMetricsOverride", width=width, height=height,
                   deviceScaleFactor=1, mobile=phone)
        if phone:
            tools.call("Emulation.setTouchEmulationEnabled", enabled=True, maxTouchPoints=5)
        # Always the light theme: headless Chrome follows the Mac's appearance
        # (dark after sunset on "Auto"), and Visual QA compares before/after
        # shots — two themes can't be compared. The page check covers dark mode.
        tools.call("Emulation.setEmulatedMedia", features=[{"name": "prefers-color-scheme", "value": "light"}])
        tools.call("Page.enable")
        tools.events.clear()
        tools.call("Page.navigate", url=url)
        tools.wait_for("Page.loadEventFired")
        # Web fonts, then a moment for first-render scripts.
        tools.call("Runtime.evaluate", awaitPromise=True,
                   expression="document.fonts.ready.then(() => new Promise(r => setTimeout(r, 500)))")
        shot = tools.call("Page.captureScreenshot", format="png")
        Path(out_path).write_bytes(base64.b64decode(shot["data"]))
        return True
    finally:
        if tools:
            tools.close()
        stop(proc)
        shutil.rmtree(profile, ignore_errors=True)


def _screenshot_cli(exe: str, url: str, out_path: str, width: int, height: int, timeout: int) -> bool:
    """Chrome's --screenshot flag. Only true to size from CLI_MIN_WIDTH up."""
    profile = tempfile.mkdtemp(prefix="autodev-chrome-")
    proc = subprocess.Popen(
        [exe, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--no-first-run",
         "--no-default-browser-check", f"--user-data-dir={profile}", f"--window-size={width},{height}",
         "--virtual-time-budget=5000", f"--screenshot={out_path}", url],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True,
    )
    ok, last_size, deadline = False, -1, time.time() + timeout
    while time.time() < deadline:
        if Path(out_path).exists():
            size = Path(out_path).stat().st_size
            if size > 0 and size == last_size:  # finished writing
                ok = True
                break
            last_size = size
        if proc.poll() is not None and not Path(out_path).exists():
            break
        time.sleep(0.5)
    stop(proc)
    shutil.rmtree(profile, ignore_errors=True)
    return ok


def screenshot(url: str, out_path: str, width: int, height: int, timeout: int = 40) -> bool:
    """Render `url` in a width x height viewport into out_path (PNG). Widths up
    to PHONE_MAX_WIDTH are emulated as a phone."""
    exe = browser()
    if not exe:
        return False
    Path(out_path).unlink(missing_ok=True)
    try:
        return _screenshot_devtools(exe, url, out_path, width, height, timeout)
    except Exception as e:
        print(f"[webapp] DevTools screenshot failed ({type(e).__name__}: {e})")
    # The flag-based screenshot can't go narrower than 500px; below that no
    # screenshot beats a wrong one.
    if width >= CLI_MIN_WIDTH:
        return _screenshot_cli(exe, url, out_path, width, height, timeout)
    return False


# Sections that show nothing but their heading once the page has loaded.
_EMPTY_SECTIONS_JS = """[...document.querySelectorAll('section')]
  .filter(s => getComputedStyle(s).display !== 'none')
  .map(s => {
    const c = s.cloneNode(true);
    c.querySelectorAll('h1,h2,h3,h4,h5,h6,script,style,template').forEach(e => e.remove());
    const text = (c.textContent || '').replace(/\\s+/g, ' ').trim();
    const media = s.querySelectorAll('img,svg,video,canvas,iframe,input,textarea,select,button,picture').length;
    const heading = ((s.querySelector('h1,h2,h3,h4') || {}).textContent || '').trim();
    return {id: s.id, heading, text: text.length, media};
  })
  .filter(s => s.text < 15 && s.media === 0)
  .map(s => s.id ? '#' + s.id : (s.heading || 'a section'))"""

# Headings no bigger than body text. The Tailwind Play CDN adds its reset
# (h1-h6 { font-size: inherit }) after the page's stylesheet, so a portfolio's
# heading sizes in custom.css never applied: every heading was 16px.
_FLAT_HEADINGS_JS = r"""(() => {
  const px = el => parseFloat(getComputedStyle(el).fontSize);
  const body = px(document.body);
  const shown = [...document.querySelectorAll('h1, h2')].filter(h => h.getBoundingClientRect().height > 0);
  const flat = shown.filter(h => px(h) <= body * (h.tagName === 'H1' ? 1.25 : 1.1));
  const h2s = shown.filter(h => h.tagName === 'H2'), flatH2s = flat.filter(h => h.tagName === 'H2');
  if (!flat.some(h => h.tagName === 'H1') && flatH2s.length * 2 <= h2s.length) return [];
  return flat.slice(0, 4).map(h => `${h.tagName.toLowerCase()} “${h.textContent.trim().slice(0, 30)}” (${px(h)}px)`);
})()"""

# Text drawn in (nearly) the colour behind it. A portfolio lost its Tailwind
# config, so "bg-accent text-white" buttons became white text on white.
_INVISIBLE_TEXT_JS = r"""(() => {
  const rgb = c => {
    if (!c || !c.startsWith('rgb')) return null;
    const p = c.slice(c.indexOf('(') + 1, -1).split(',').map(parseFloat);
    return {r: p[0], g: p[1], b: p[2], a: p.length > 3 ? p[3] : 1};
  };
  const ch = v => { v /= 255; return v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4; };
  const lum = c => 0.2126 * ch(c.r) + 0.7152 * ch(c.g) + 0.0722 * ch(c.b);
  // The element that paints the background, so a fix changes it rather than
  // one text at a time: dark-mode fixes missed a card's fixed "bg-white" four rounds running.
  const backdrop = el => {
    for (let e = el; e; e = e.parentElement) {
      const s = getComputedStyle(e);
      if (s.backgroundImage !== 'none') return null;  // over an image or gradient: can't tell
      const c = rgb(s.backgroundColor);
      if (c && c.a >= 0.5) return {c, e};
    }
    return {c: {r: 255, g: 255, b: 255, a: 1}, e: null};
  };
  const where = e => !e ? 'the page background' : '<' + e.tagName.toLowerCase() + (e.id ? ` id="${e.id}"` : '')
    + (e.classList.length ? ` class="${[...e.classList].slice(0, 3).join(' ')}${e.classList.length > 3 ? ' …' : ''}"` : '') + '>';
  const found = [];
  for (const el of document.body.querySelectorAll('*')) {
    if (el.closest('svg')) continue;
    const text = [...el.childNodes].filter(n => n.nodeType === 3).map(n => n.textContent).join('').trim();
    if (text.length < 2) continue;
    const s = getComputedStyle(el), box = el.getBoundingClientRect();
    if (s.visibility !== 'visible' || box.width < 2 || box.height < 2) continue;
    const fg = rgb(s.color), bg = backdrop(el);
    if (!fg || !bg || fg.a < 0.5) continue;
    const [hi, lo] = [lum(fg), lum(bg.c)].sort((a, b) => b - a);
    if ((hi + 0.05) / (lo + 0.05) < 1.4) found.push([text.replace(/\s+/g, ' ').slice(0, 40), where(bg.e)]);
  }
  const seen = new Set();
  return found.filter(([t]) => !seen.has(t) && seen.add(t)).slice(0, 6);
})()"""


def inspect_page(url: str, timeout: int = 40) -> dict | None:
    """Load `url` in headless Chrome and report what a user would hit:
    {"errors": [(message, script url, line)], "failed": ["404 /js/x.js"],
    "empty_sections": ["#services"], "blocked": [CSP message],
    "invisible": ['“Contact me” (on the page background)'],
    "flat_headings": ["h1 “Asha” (16px)"]}. Unit tests can't see that a page's cards
    never render — a portfolio passed its tests with four empty sections
    because the script holding their content was never loaded.
    None when no browser is available or the page couldn't be opened."""
    exe = browser()
    if not exe:
        return None
    profile, port = tempfile.mkdtemp(prefix="autodev-chrome-"), _free_port()
    proc = subprocess.Popen(
        [exe, "--headless=new", "--disable-gpu", "--no-first-run", "--no-default-browser-check",
         f"--user-data-dir={profile}", f"--remote-debugging-port={port}", "about:blank"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    deadline, tools = time.time() + timeout, None
    origin = re.match(r"^[a-z]+://[^/]+", url).group(0)
    try:
        tools = _DevTools(_page_socket(port, deadline), deadline)
        tools.call("Emulation.setDeviceMetricsOverride", width=1280, height=900, deviceScaleFactor=1, mobile=False)
        for domain in ("Page", "Runtime", "Log", "Network"):
            tools.call(f"{domain}.enable")
        # Light first, whatever the Mac's appearance is right now: headless
        # Chrome follows it, and macOS can switch to dark at sunset.
        tools.call("Emulation.setEmulatedMedia", features=[{"name": "prefers-color-scheme", "value": "light"}])
        tools.events.clear()
        tools.messages.clear()
        tools.call("Page.navigate", url=url)
        tools.wait_for("Page.loadEventFired")
        tools.call("Runtime.evaluate", awaitPromise=True,
                   expression="document.fonts.ready.then(() => new Promise(r => setTimeout(r, 1000)))")
        empty = tools.call("Runtime.evaluate", returnByValue=True, expression=_EMPTY_SECTIONS_JS)
        empty_sections = (empty.get("result") or {}).get("value") or []
        flat = tools.call("Runtime.evaluate", returnByValue=True, expression=_FLAT_HEADINGS_JS)
        flat_headings = (flat.get("result") or {}).get("value") or []
        def unseen(mode: str) -> list[str]:
            """Unreadable texts, grouped by the element painting the background behind them."""
            found = (tools.call("Runtime.evaluate", returnByValue=True, expression=_INVISIBLE_TEXT_JS)
                     .get("result") or {}).get("value") or []
            groups: dict[str, list[str]] = {}
            for text, behind in found:
                groups.setdefault(behind, []).append(f"“{text}”")
            return [f"{', '.join(texts)} ({mode}on {behind})" for behind, texts in groups.items()]
        invisible = unseen("")
        # Then dark: a portfolio's text turned white in dark mode while its
        # sections kept a fixed white background.
        tools.call("Emulation.setEmulatedMedia", features=[{"name": "prefers-color-scheme", "value": "dark"}])
        tools.call("Runtime.evaluate", awaitPromise=True, expression="new Promise(r => setTimeout(r, 600))")
        invisible += unseen("dark mode, ")
        errors, failed, blocked, requests = [], [], [], {}
        for msg in tools.messages:
            method, p = msg["method"], msg.get("params") or {}
            if method == "Runtime.exceptionThrown":
                d = p.get("exceptionDetails") or {}
                text = ((d.get("exception") or {}).get("description") or d.get("text") or "error").splitlines()[0]
                errors.append((text, d.get("url", ""), d.get("lineNumber", -1) + 1))
            elif method == "Runtime.consoleAPICalled" and p.get("type") == "error":
                frame = ((p.get("stackTrace") or {}).get("callFrames") or [{}])[0]
                text = " ".join(str(a.get("value", a.get("description", ""))) for a in p.get("args", []))
                errors.append((f"console.error: {text}", frame.get("url", ""), frame.get("lineNumber", -1) + 1))
            elif method == "Network.requestWillBeSent":
                requests[p.get("requestId")] = (p.get("request") or {}).get("url", "")
            elif method == "Network.responseReceived":
                r = p.get("response") or {}
                if r.get("status", 0) >= 400 and r.get("url", "").startswith(origin):
                    failed.append(f"{r['status']} {r['url'][len(origin):]}")
            elif method == "Network.loadingFailed" and not p.get("canceled"):
                req_url = requests.get(p.get("requestId"), "")
                if req_url.startswith(origin):
                    failed.append(f"{p.get('errorText', 'failed')} {req_url[len(origin):]}")
            elif method == "Log.entryAdded":
                entry = p.get("entry") or {}
                # The page's own Content-Security-Policy blocking it ("Applying inline
                # style violates…") is only reported here — a security "fix" left a
                # portfolio unstyled this way.
                if entry.get("source") == "security" and entry.get("level") == "error":
                    blocked.append(entry.get("text", "").split(". ")[0])
        # Chrome asks for /favicon.ico on its own; a site without one isn't broken.
        failed = [f for f in dict.fromkeys(failed) if not f.endswith(" /favicon.ico")]
        return {"errors": list(dict.fromkeys(errors)), "failed": failed, "empty_sections": empty_sections,
                "blocked": list(dict.fromkeys(blocked)), "invisible": invisible, "flat_headings": flat_headings}
    except Exception as e:
        print(f"[webapp] page check failed ({type(e).__name__}: {e})")
        return None
    finally:
        if tools:
            tools.close()
        stop(proc)
        shutil.rmtree(profile, ignore_errors=True)
