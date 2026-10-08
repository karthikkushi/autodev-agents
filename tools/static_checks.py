"""
Free, deterministic checks on generated code — no AI quota.

    ruff       syntax errors and names that can't exist (rules E9, F63, F7, F82)
    mypy       imports that can't work ("from models import ExpenseTracker"
               when models.py has no such class) — nothing else, because
               generated code is untyped and other type errors are noise
    bandit     security issues, MEDIUM or worse with MEDIUM or better confidence
    pip-audit  known-vulnerable packages installed in the project's venv

Each check returns findings {tool, file, line, severity, code, message} and
never raises: a tool that is missing, times out or can't reach the internet
gives [] and a dim note. Tools run as `python -m <tool>` on this interpreter
with fixed arguments and no shell; they only read files, so they don't need
the sandbox.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from rich.console import Console

from tools.pyenv import project_python

console = Console()

TIMEOUT = 60
RUFF_FATAL = "E9,F63,F7,F82"
MYPY_KEEP = {"import-not-found", "name-defined", "attr-defined"}
TOOLING_PACKAGES = {"pip", "setuptools", "wheel"}   # venv tooling, not the app's dependencies
MYPY_CACHE = Path(__file__).resolve().parent.parent / "logs" / "mypy_cache"
REPORT_DOC = "05c_static_checks.md"
TOOL_INFO = {"ruff": "syntax errors and undefined names", "mypy": "imports that can't work",
             "web": "stylesheets browsers can't read, missing linked files, scripts nothing loads",
             "page": "the built page in a real browser: JavaScript errors, failed loads, empty sections",
             "bandit": "security issues", "pip-audit": "known-vulnerable dependencies"}
# Module names can be swapped in tests to simulate a missing tool.
_MODULES = {"ruff": "ruff", "mypy": "mypy", "bandit": "bandit", "pip-audit": "pip_audit"}
_MYPY_LINE = re.compile(r"^(?P<file>[^:\n]+):(?P<line>\d+):(?:\d+:)? (?P<sev>error|note): (?P<msg>.*?)"
                        r"(?:  \[(?P<code>[\w-]+)\])?$")
_last_skip: dict[str, str] = {}   # tool -> why its latest run didn't happen, for the report


def _skip(tool: str, why: str) -> None:
    _last_skip[tool] = why
    console.print(f"[dim]  {tool} skipped: {why}[/dim]")


def _run(tool: str, args: list[str], cwd, env: dict | None = None):
    """subprocess result of `python -m <tool> args`, or None when the tool is
    missing or takes longer than TIMEOUT."""
    module = _MODULES[tool]
    try:
        r = subprocess.run([sys.executable, "-m", module, *args], cwd=str(cwd), capture_output=True, text=True,
                           timeout=TIMEOUT, env={**os.environ, **(env or {})})
    except subprocess.TimeoutExpired:
        return _skip(tool, f"no result within {TIMEOUT}s")
    except OSError as e:
        return _skip(tool, str(e))
    if r.returncode != 0 and f"No module named {module}" in r.stderr:
        return _skip(tool, "not installed (pip install -r requirements.txt)")
    _last_skip.pop(tool, None)
    return r


def _rel(path: str, root) -> str:
    try:
        return str(Path(path).resolve().relative_to(Path(root).resolve()))
    except (ValueError, OSError):
        return str(path)


def _finding(tool, file, line, severity, code, message) -> dict:
    return {"tool": tool, "file": file, "line": int(line or 0), "severity": severity, "code": code or "",
            "message": " ".join(str(message).split())[:300]}


def _py_files(root: Path, dirs=("src", "tests")) -> list[str]:
    return [str(p.relative_to(root)) for d in dirs if (root / d).is_dir() for p in sorted((root / d).rglob("*.py"))
            if not {".venv", "__pycache__", "node_modules"} & set(p.parts)]


def ruff_check(paths: list[str], root=None) -> list[dict]:
    """Syntax errors and undefined names in the given .py files (absolute, or
    relative to `root`)."""
    files = [p for p in paths if p.endswith(".py")]
    if not files:
        return []
    try:
        r = _run("ruff", ["check", "--select", RUFF_FATAL, "--output-format", "json", "--isolated", "--no-cache",
                          *files], cwd=root or os.getcwd())
        if r is None:
            return []
        if r.returncode not in (0, 1):
            _skip("ruff", (r.stderr.strip() or "failed")[:200])
            return []
        base = root or os.getcwd()
        return [_finding("ruff", _rel(row["filename"], base), (row.get("location") or {}).get("row"), "HIGH",
                         row.get("code") or "syntax-error", row.get("message", ""))
                for row in json.loads(r.stdout or "[]")]
    except Exception as e:  # never break the agent that asked
        _skip("ruff", f"{type(e).__name__}: {e}")
        return []


def _python_for(project_path: str) -> str:
    py = str(project_python(project_path))
    return py if os.path.isabs(py) else (shutil.which(py) or sys.executable)


def mypy_check(project_path: str, dirs=("src",)) -> list[dict]:
    """Imports that can't work, found with the project's own interpreter
    (its venv's installed packages). Every other mypy error is dropped."""
    root = Path(project_path)
    targets = [d for d in dirs if (root / d).is_dir() and any((root / d).rglob("*.py"))]
    if not targets:
        return []
    try:
        cache = MYPY_CACHE / root.name   # outside the project, so it never reaches git
        r = _run("mypy", [*targets, "--python-executable", _python_for(project_path), "--explicit-package-bases",
                          "--follow-imports=silent", "--check-untyped-defs", "--no-error-summary",
                          "--show-error-codes", "--no-color-output", "--hide-error-context", "--no-pretty",
                          "--cache-dir", str(cache)],
                 cwd=root, env={"MYPYPATH": str(root / "src")})
        if r is None:
            return []
        if "  [syntax]" in r.stdout:
            # mypy stops at the first syntax error; ruff reports those.
            _skip("mypy", "a syntax error stops mypy — fix the ruff findings first")
            return []
        if r.returncode not in (0, 1):
            _skip("mypy", (r.stderr.strip() or r.stdout.strip() or "failed")[-200:])
            return []
        findings = []
        for line in r.stdout.splitlines():
            m = _MYPY_LINE.match(line.strip())
            if not m or m["sev"] != "error":
                continue
            code, msg = m["code"] or "", m["msg"]
            if code in MYPY_KEEP and (code != "attr-defined" or re.match(r'Module "[^"]+" has no attribute', msg)):
                findings.append(_finding("mypy", m["file"], m["line"], "HIGH", code, msg))
        return findings
    except Exception as e:
        _skip("mypy", f"{type(e).__name__}: {e}")
        return []


def bandit_check(project_path: str) -> list[dict]:
    """Security issues in src/ (tests and the venv excluded; assert-in-tests
    B101 skipped), MEDIUM or worse with MEDIUM or better confidence."""
    root = Path(project_path)
    if not (root / "src").is_dir():
        return []
    try:
        r = _run("bandit", ["-r", "src", "-f", "json", "-q", "-s", "B101",
                            "-x", "*/tests/*,*/.venv/*,*/node_modules/*"], cwd=root)
        if r is None:
            return []
        data = json.loads(r.stdout or "{}")
        rank = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
        return [_finding("bandit", _rel(row["filename"], root), row.get("line_number"), row["issue_severity"],
                         row.get("test_id"), row.get("issue_text", ""))
                for row in data.get("results", [])
                if rank.get(row.get("issue_severity"), 0) >= 1 and rank.get(row.get("issue_confidence"), 0) >= 1]
    except Exception as e:
        _skip("bandit", f"{type(e).__name__}: {e}")
        return []


def _requirement_line(root: Path, package: str) -> int:
    want = re.sub(r"[-_.]+", "-", package).lower()
    try:
        for n, line in enumerate((root / "requirements.txt").read_text().splitlines(), 1):
            name = re.split(r"[\s<>=!~;\[]", line.strip(), maxsplit=1)[0]
            if name and re.sub(r"[-_.]+", "-", name).lower() == want:
                return n
    except OSError:
        pass
    return 0


def pip_audit_check(project_path: str) -> list[dict]:
    """Known vulnerabilities in the packages installed in the project's venv
    (not requirements.txt: resolving that would build packages). Needs the
    internet; skipped when offline."""
    root = Path(project_path)
    site = sorted((root / ".venv").glob("lib/python*/site-packages"))
    if not site:
        _skip("pip-audit", "the project has no .venv to audit")
        return []
    try:
        r = _run("pip-audit", ["--path", str(site[0]), "-f", "json", "--progress-spinner", "off",
                               "--timeout", "15"], cwd=root)
        if r is None:
            return []
        try:
            data = json.loads(r.stdout)
        except ValueError:
            _skip("pip-audit", "couldn't reach the vulnerability database (offline?)")
            return []
        findings = []
        for dep in data.get("dependencies", []):
            if dep.get("name", "").lower() in TOOLING_PACKAGES:
                continue
            for vid, v in {v["id"]: v for v in dep.get("vulns") or []}.items():
                fixes = v.get("fix_versions") or []
                aliases = [a for a in v.get("aliases") or [] if a.startswith("CVE-")][:2]
                findings.append(_finding(
                    "pip-audit", "requirements.txt", _requirement_line(root, dep["name"]),
                    # With a fixed version the fix is a one-line pin; without one there is nothing to do yet.
                    "HIGH" if fixes else "MEDIUM", vid,
                    f"{dep['name']} {dep.get('version', '')} has {vid}"
                    + (f" ({', '.join(aliases)})" if aliases else "")
                    + (f" — fixed in {', '.join(fixes)}" if fixes else " — no fixed version yet")))
        return findings
    except Exception as e:
        _skip("pip-audit", f"{type(e).__name__}: {e}")
        return []


_BUILD_ONLY_CSS = re.compile(r"^\s*@(tailwind|apply|config|plugin)\b", re.MULTILINE)
_LOCAL_REF = re.compile(r"""\b(?:href|src)\s*=\s*["']([^"'#?]+)""", re.IGNORECASE)
# Files a model can't write (it only produces text): a missing photo or PDF is
# for the owner to add, not a bug to send the coder round after — three review
# rounds once went on "create profile.jpg".
_BINARY_ASSETS = (".jpg", ".jpeg", ".png", ".gif", ".webp", ".avif", ".ico", ".pdf", ".woff", ".woff2",
                  ".ttf", ".otf", ".mp4", ".webm", ".mp3")
_JS_IMPORT = re.compile(r"""(?:\bfrom\s*|\bimport\s*\(?\s*)["'](\.{1,2}/[^"']+)["']""")
_NEEDS_BUILD = {".astro", ".jsx", ".tsx", ".vue", ".svelte"}
_TAILWIND_CDN = re.compile(r"""<script[^>]+src=["'][^"']*(?:cdn\.tailwindcss\.com|@tailwindcss/browser)""",
                           re.IGNORECASE)
_UTILITY_CLASS = re.compile(
    r"\[|^(?:[a-z]+:)+|^-?(?:bg|text|font|p[trblxy]?|m[trblxy]?|w|h|min-w|max-w|min-h|max-h|gap|space-[xy]|"
    r"rounded|shadow|border|items|justify|self|place|col-span|row-span|grid-cols|inset|z|opacity|leading|"
    r"tracking|translate-[xy]|scale|rotate|duration|ease|overflow|order|basis|grow|shrink)-")


def web_check(project_path: str) -> list[dict]:
    """Two certain bugs in web pages, no tool needed:
    - a stylesheet the browser can't read: @tailwind/@apply only work after a
      build step, and nothing builds the project — a portfolio went out as an
      unstyled page because of this;
    - a page linking a local file (css/js/image) that doesn't exist."""
    root = Path(project_path)
    src = root / "src"
    if not src.is_dir():
        return []
    findings, linked = [], set()
    # Framework source files: nothing in this pipeline compiles them.
    built = [p for p in sorted(src.rglob("*")) if p.suffix in _NEEDS_BUILD and not {"node_modules", ".venv"} & set(p.parts)]
    if built:
        findings.append(_finding("web", _rel(str(built[0]), root), 0, "HIGH", "needs-build",
                                 f"{len(built)} file(s) like {built[0].name} need a framework build that nothing here "
                                 "runs, so browsers never see them. Write the site as plain HTML/CSS/JS instead."))
    for page in sorted(src.rglob("*.html")):
        if {".venv", "node_modules"} & set(page.parts):
            continue
        text = page.read_text(errors="replace")
        for m in _LOCAL_REF.finditer(text):
            ref = m.group(1).strip()
            if not ref or re.match(r"^(?:[a-z][a-z0-9+.-]*:|//|\{\{|\$)", ref, re.IGNORECASE) or "{" in ref:
                continue  # external URL, mailto:, template placeholder
            target = (src / ref.lstrip("/")) if ref.startswith("/") else (page.parent / ref)
            if target.exists():
                linked.add(target.resolve())
            elif ref.lower().endswith(_BINARY_ASSETS):
                findings.append(_finding("web", _rel(str(page), root), text[:m.start()].count("\n") + 1, "MEDIUM",
                                         "missing-asset", f'"{ref}" is linked but doesn\'t exist — add the real file '
                                         "before publishing (or use an inline SVG/CSS placeholder)"))
            else:
                findings.append(_finding("web", _rel(str(page), root), text[:m.start()].count("\n") + 1, "HIGH",
                                         "missing-file", f'"{ref}" is linked but doesn\'t exist'))
    # A plain static site (no app.py): a script no page loads and no script
    # imports never runs — the portfolio's data.js held every card's content.
    if not (src / "app.py").exists():
        scripts = [p for p in sorted(src.rglob("*.js")) if not {".venv", "node_modules", "tests"} & set(p.parts)]
        for js in scripts:
            for ref in _JS_IMPORT.findall(js.read_text(errors="replace")):
                linked.add((js.parent / ref).resolve())
        for js in scripts:
            if js.resolve() not in linked:
                findings.append(_finding("web", _rel(str(js), root), 0, "MEDIUM", "unused-script",
                                         "no page loads this script and no script imports it, so it never runs"))
        # A stylesheet no page links, holding the theme the page uses: a
        # portfolio's colours all pointed at var(--accent) from a styles.css
        # nothing linked, so it rendered black and white.
        sheets = [p for p in sorted(src.rglob("*.css")) if not {".venv", "node_modules"} & set(p.parts)]
        pages_text = "".join(p.read_text(errors="replace") for p in sorted(src.rglob("*.html"))
                             if not {".venv", "node_modules"} & set(p.parts))
        defs = lambda text: set(re.findall(r"(--[\w-]+)\s*:", text))
        known = defs(pages_text).union(*(defs(c.read_text(errors="replace")) for c in sheets if c.resolve() in linked))
        wanted = set(re.findall(r"var\(\s*(--[\w-]+)", pages_text)) - known
        for css in sheets:
            missing = sorted(wanted & defs(css.read_text(errors="replace"))) if css.resolve() not in linked else []
            if missing:
                findings.append(_finding("web", _rel(str(css), root), 0, "HIGH", "unlinked-stylesheet",
                                         f"the page uses {', '.join(missing[:4])} from this file, but no page links "
                                         f'it — add <link rel="stylesheet" href="{css.relative_to(src)}"> to <head>'))
    # Only stylesheets a page actually loads: a leftover Tailwind input file
    # that nothing links doesn't change what the page looks like.
    for css in sorted(src.rglob("*.css")):
        if {".venv", "node_modules"} & set(css.parts) or css.resolve() not in linked:
            continue
        text = css.read_text(errors="replace")
        m = _BUILD_ONLY_CSS.search(text)
        if m:
            findings.append(_finding("web", _rel(str(css), root), text[:m.start()].count("\n") + 1, "HIGH",
                                     "css-needs-build", f"@{m.group(1)} only works after a Tailwind build, and nothing "
                                     "builds this project: browsers ignore it and the page is unstyled. Write plain CSS."))
    # The same missing build step, seen from the HTML: Tailwind utility
    # classes (bg-gray-50, max-w-[1100px]) that no stylesheet defines do nothing.
    defined = set()
    for css in linked:
        if css.suffix == ".css":
            defined |= {c.replace("\\", "") for c in re.findall(r"\.(-?[_a-zA-Z][\w\-\\\[\]\.:/%]*)", css.read_text(errors="replace"))}
    for page in sorted(src.rglob("*.html")):
        if {".venv", "node_modules"} & set(page.parts):
            continue
        text = page.read_text(errors="replace")
        if _TAILWIND_CDN.search(text):
            continue  # Tailwind's Play CDN builds the classes in the browser
        classes = {c for attr in re.findall(r"""\bclass\s*=\s*["']([^"']*)["']""", text) for c in attr.split()}
        undefined = sorted(c for c in classes if _UTILITY_CLASS.search(c) and c not in defined)
        if len(undefined) >= 8:
            findings.append(_finding("web", _rel(str(page), root), 0, "HIGH", "tailwind-classes",
                                     f"{len(undefined)} Tailwind utility classes (e.g. {', '.join(undefined[:5])}) have no "
                                     "CSS rule — nothing builds Tailwind, so they do nothing."))
    return findings


def page_check(project_path: str) -> list[dict]:
    """Start the built web app and open it in headless Chrome: JavaScript
    errors, local files that fail to load, and sections that show nothing but
    their heading. Unit tests of the HTML can't see any of these."""
    from tools import webapp
    if not webapp.browser():
        _skip("page", "no Chrome/Edge installed")
        return []
    root = Path(project_path)
    log = Path(tempfile.gettempdir()) / f"autodev-page-{root.name}.log"
    proc, url, err = webapp.start_app(project_path, str(log))
    if not proc:
        if err.startswith("no src/app.py"):
            # Only called for web projects: no page to open is the worst bug of all
            # (a portfolio planned on Astro ended up as .astro files and no page).
            return [_finding("page", "src/index.html", 0, "HIGH", "no-entry-page",
                             "there is no src/index.html (and no Flask/FastAPI src/app.py), so the site can't be "
                             "opened at all")]
        return [_finding("page", "src/app.py", 0, "HIGH", "app-wont-start", err)]
    try:
        report = webapp.inspect_page(url + "/")
    finally:
        webapp.stop(proc)
    if report is None:
        _skip("page", "the page couldn't be opened")
        return []
    _last_skip.pop("page", None)
    page = next((p for p in ("src/index.html", "src/templates/index.html") if (root / p).exists()), "src/index.html")

    def src_file(script_url: str) -> str:
        path = re.sub(r"^[a-z]+://[^/]+/", "", script_url or "").split("?")[0]
        return f"src/{path}" if path and (root / "src" / path).exists() else page
    findings = [_finding("page", src_file(script), line, "HIGH", "js-error", text) for text, script, line in report["errors"]]
    findings += [_finding("page", page, 0, "MEDIUM" if f.split("?")[0].lower().endswith(_BINARY_ASSETS) else "HIGH",
                          "failed-load", f"the page couldn't load {f}") for f in report["failed"]]
    findings += [_finding("page", page, 0, "HIGH", "empty-section",
                          f"section {s} shows nothing but its heading once the page has loaded")
                 for s in report["empty_sections"]]
    findings += [_finding("page", page, 0, "HIGH", "csp-blocked", f"the page's Content-Security-Policy blocks it: {b}")
                 for b in report.get("blocked", [])]
    if report.get("flat_headings"):
        findings.append(_finding("page", page, 0, "HIGH", "flat-headings",
                                 "headings are no bigger than the body text: " + ", ".join(report["flat_headings"])))
    if report.get("invisible"):  # one finding, so one fix task covers every such text
        findings.append(_finding("page", page, 0, "HIGH", "invisible-text",
                                 "this text is the same colour as what's behind it, so nobody can read it: "
                                 + "; ".join(report["invisible"])))
    return findings


def code_checks(project_path: str) -> dict[str, list]:
    """ruff and mypy on src/ and tests/, and the web check on src/. A name
    mypy calls undefined on a line ruff already reported is dropped (same
    bug, found twice)."""
    root = Path(project_path)
    ruff = ruff_check(_py_files(root), root=root)
    ruff_names = {(f["file"], f["line"]) for f in ruff if f["code"] == "F821"}
    mypy = [f for f in mypy_check(project_path, ("src", "tests"))
            if not (f["code"] == "name-defined" and (f["file"], f["line"]) in ruff_names)]
    return {"ruff": ruff, "mypy": mypy, "web": web_check(project_path)}


def security_checks(project_path: str) -> dict[str, list]:
    return {"bandit": bandit_check(project_path), "pip-audit": pip_audit_check(project_path)}


def as_lines(findings: list[dict]) -> str:
    return "\n".join(f"- {f['file']}:{f['line']} [{f['severity']}] {f['tool']} {f['code']}: {f['message']}"
                     for f in findings)


def write_report(project_path: str, results: dict[str, list]) -> None:
    """Rewrite docs/05c_static_checks.md with each tool's latest run. The
    reviewer (ruff, mypy) and the security agent (bandit, pip-audit) run at
    different times, so the latest results per tool are kept in
    .memory/static_checks.json (ignored by git) and the page shows all four."""
    from tools.file_ops import write_doc
    root = Path(project_path)
    store = root / ".memory" / "static_checks.json"
    try:
        saved = json.loads(store.read_text())
    except (OSError, ValueError):
        saved = {}
    now = time.strftime("%Y-%m-%d %H:%M")
    for tool, findings in results.items():
        saved[tool] = {"at": now, "skipped": _last_skip.get(tool, ""), "findings": findings}
    try:
        store.parent.mkdir(exist_ok=True)
        store.write_text(json.dumps(saved, indent=1))
    except OSError:
        pass
    lines = ["# Static checks", "",
             "Free, deterministic checks — no AI quota. Their findings are facts, not opinions: the reviewer and",
             "the security agent pass them to the coder as high-priority fixes. Rewritten on every run.", ""]
    for tool in TOOL_INFO:
        entry = saved.get(tool)
        if not entry:
            continue
        found = entry.get("findings") or []
        status = f"skipped — {entry['skipped']}" if entry.get("skipped") else f"{len(found)} finding(s)"
        lines += [f"## {tool} — {TOOL_INFO[tool]}", "", f"_{entry.get('at', '')} · {status}_", ""]
        if found:
            lines += ["| File | Line | Severity | Code | Message |", "|---|---|---|---|---|"]
            lines += [f"| {f['file']} | {f['line']} | {f['severity']} | {f['code']} | {f['message'].replace('|', '/')} |"
                      for f in found[:200]]
            lines.append("")
    write_doc(project_path, REPORT_DOC, "\n".join(lines))
