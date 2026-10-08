#!/usr/bin/env python3
"""
AutoDev API — Autonomous AI Development Pipeline (100% cloud API, no local models)
Usage:  python3 main.py                     # dashboard server + pipeline worker (normal use)
        python3 main.py --no-browser        # same, without opening the dashboard (login item)
        python3 main.py --worker            # pipeline worker only (the server starts this)
        python3 main.py <path_to_design.md> # one-off run of a single design
        add --auditor to also run the Shadow Auditor (same APIs, "fast" role)

The dashboard (http://localhost:8081/ui/index.html) starts/stops/pauses the
worker with buttons. Every graph step is checkpointed, so a stopped, crashed
or powered-off run resumes where it left off the next time the worker starts.

Every agent call — including the Shadow Auditor — goes through
router.get_llm(role) to free cloud APIs (Gemini, NVIDIA, OpenRouter free
models). No local inference, nothing to install beyond requirements.txt.
"""
import sys
import os
import json
import time
import sqlite3
import atexit
import threading
import urllib.request
import webbrowser
from pathlib import Path
from datetime import datetime
from rich.console import Console
from rich.panel import Panel

BASE_DIR = Path(__file__).parent
try:
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env")
except ImportError:
    pass

console = Console()
DESIGN_INBOX = BASE_DIR / "design_inbox"
DESIGN_INBOX.mkdir(exist_ok=True)
(BASE_DIR / "projects").mkdir(exist_ok=True)
(BASE_DIR / "logs").mkdir(exist_ok=True)
WORKER_PID = BASE_DIR / "logs" / "worker.pid"
DASHBOARD_URL = "http://localhost:8081/ui/index.html"

sys.path.insert(0, str(BASE_DIR))


# ── Server ────────────────────────────────────────────────────────────────────

def _server_running() -> bool:
    try:
        urllib.request.urlopen("http://localhost:8081/health", timeout=1)
        return True
    except Exception:
        return False


def _start_server_background():
    """Start uvicorn in a daemon thread if not already running (one-off mode)."""
    if _server_running():
        console.print("[dim]Server already running on :8081[/dim]")
        return

    def _run():
        import uvicorn
        config = uvicorn.Config(
            "server.server:app",
            host="0.0.0.0", port=8081,
            loop="asyncio", log_level="warning",
        )
        server = uvicorn.Server(config)
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(server.serve())

    t = threading.Thread(target=_run, daemon=True, name="autodev-api-server")
    t.start()
    for _ in range(50):
        if _server_running():
            console.print("[green]Server started on :8081[/green]")
            return
        time.sleep(0.1)
    console.print("[yellow]Server may not be ready yet — continuing anyway[/yellow]")


def run_control_plane(open_browser: bool):
    """Normal mode: the dashboard server in the foreground. It launches the
    pipeline worker itself (see PipelineWorker in server/server.py), so
    everything after this one command is a button in the dashboard."""
    if _server_running():
        console.print(f"[yellow]AutoDev is already running — dashboard: {DASHBOARD_URL}[/yellow]")
        if open_browser:
            webbrowser.open(DASHBOARD_URL)
        return
    import uvicorn
    os.environ["AUTODEV_AUTOSTART_WORKER"] = "1"
    console.print(Panel.fit(
        f"[bold cyan]🤖 AutoDev control server[/bold cyan]\n"
        f"[white]Dashboard:[/white] {DASHBOARD_URL}\n"
        f"[dim]Start / Stop / Pause / Resume from the dashboard. Ctrl+C here stops everything;\n"
        f"unfinished projects resume from their last step next time.[/dim]",
        border_style="cyan",
    ))
    if open_browser:
        threading.Timer(2.0, webbrowser.open, args=(DASHBOARD_URL,)).start()
    uvicorn.run("server.server:app", host="0.0.0.0", port=8081, log_level="warning")


def _maybe_start_auditor(enabled: bool):
    if not (enabled or os.environ.get("AUTODEV_AUDITOR") == "1"):
        return
    try:
        from auditor import shadow_auditor
        shadow_auditor.start()
        console.print("[magenta]⬡ Shadow Auditor started — auditing in parallel, zero pipeline impact[/magenta]")
    except Exception as e:
        console.print(f"[yellow]Shadow Auditor failed to start (non-fatal): {e}[/yellow]")


def _check_provider_keys():
    # Only the providers router/roles.py actually routes to — unused keys
    # can sit in .env without being reported.
    from router.roles import ROLE_DEPLOYMENTS
    names = {"GOOGLE_API_KEY": "gemini", "NVIDIA_API_KEY": "nvidia", "OPENROUTER_API_KEY": "openrouter",
             "GROQ_API_KEY": "groq", "CEREBRAS_API_KEY": "cerebras", "MISTRAL_API_KEY": "mistral",
             "SAMBANOVA_API_KEY": "sambanova", "CLOUDFLARE_API_KEY": "cloudflare", "ZAI_API_KEY": "zai",
             "OLLAMA_API_KEY": "ollama"}
    used_keys = list(dict.fromkeys(env_key for deps in ROLE_DEPLOYMENTS.values() for _, env_key, _ in deps))
    available = [names.get(k, k) for k in used_keys if os.environ.get(k)]
    missing = [names.get(k, k) for k in used_keys if not os.environ.get(k)]
    if available:
        console.print(f"[green]✅ Providers ready: {', '.join(available)}[/green]")
    if missing:
        console.print(f"[dim]Providers without a key (skipped): {', '.join(missing)}[/dim]")
    if not available:
        console.print(Panel.fit(
            "[bold red]No provider API keys found![/bold red]\n"
            f"Set at least one of {', '.join(used_keys)}\n"
            "in .env before running the pipeline.",
            border_style="red"
        ))


# ── Pipeline runner ───────────────────────────────────────────────────────────

def load_design_file(path: str) -> tuple[str, str]:
    if not os.path.exists(path):
        console.print(f"[red]❌ File not found: {path}[/red]")
        sys.exit(1)
    with open(path) as f:
        content = f.read()
    filename = Path(path).name
    return filename, content


def _open_checkpointer():
    from langgraph.checkpoint.sqlite import SqliteSaver
    from pipeline import control
    control.CHECKPOINT_DB.parent.mkdir(exist_ok=True)
    # check_same_thread=False: LangGraph runs parallel branches on worker threads
    return SqliteSaver(sqlite3.connect(str(control.CHECKPOINT_DB), check_same_thread=False))


def run_pipeline(design_path: str):
    filename, content = load_design_file(design_path)
    project_name = filename.replace(".md", "").replace(" ", "_").lower()

    from server.bridge import log as blog, state as bstate
    from pipeline.state import new_state
    from pipeline.graph import build_graph
    from pipeline import control

    graph = build_graph(checkpointer=_open_checkpointer())
    config = {"configurable": {"thread_id": project_name}}
    snapshot = graph.get_state(config)
    if snapshot.values and not snapshot.next:
        console.print(f"[dim]✓ {project_name} already built — skipping (use Rebuild in the dashboard to start over)[/dim]")
        return
    resuming = bool(snapshot.next)

    console.print(Panel.fit(
        f"[bold cyan]🤖 AutoDev API Pipeline[/bold cyan]\n"
        f"[white]Project:[/white] {project_name}\n"
        f"[white]Design:[/white] {filename}\n"
        + (f"[white]Resuming at:[/white] {', '.join(snapshot.next)}\n" if resuming else "")
        + f"[white]Started:[/white] {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        border_style="cyan"
    ))

    blog("system", f"Pipeline {'resuming' if resuming else 'starting'}: {project_name}"
                   + (f" (at {', '.join(snapshot.next)})" if resuming else ""))
    bstate({
        "phase": "starting",
        "project_name": project_name,
        "current_agent": "system",
        "is_complete": False,
        "github_ready": False,
        "github_approved": False,
    })

    console.print("[bold green]🚀 Resuming pipeline...[/bold green]\n" if resuming
                  else "[bold green]🚀 Starting pipeline...[/bold green]\n")
    blog("system", "Graph built — running agents...")

    project_path = snapshot.values.get("project_path") if resuming else new_state(filename, content)["project_path"]
    try:
        final_state = graph.invoke(None if resuming else new_state(filename, content), config)

        console.print("\n" + "=" * 60)
        known = len(final_state.get("known_issues", []))
        if final_state.get("github_ready"):
            console.print(Panel.fit(
                f"[bold green]✅ PROJECT COMPLETE![/bold green]\n\n"
                f"[white]Location:[/white] {final_state['project_path']}\n"
                f"[white]Files:[/white] {len(final_state.get('files_written', []))}\n"
                f"[white]Known issues:[/white] {known}",
                border_style="green"
            ))
            blog("system", f"✅ Pipeline done! {len(final_state.get('files_written', []))} files written.")
        else:
            console.print(Panel.fit(
                f"[bold yellow]⚠️  Pipeline finished with issues[/bold yellow]\n\n"
                f"[white]Location:[/white] {final_state['project_path']}\n"
                f"[white]Errors:[/white] {len(final_state.get('errors', []))}",
                border_style="yellow"
            ))
            blog("system", f"Pipeline finished. Errors: {len(final_state.get('errors', []))}")
        bstate({"phase": "complete", "is_complete": True})
        control.write_status(final_state["project_path"], status="complete", step="done",
                             known_issues=known, finished_at=datetime.now().isoformat(timespec="seconds"))
        ui = final_state.get("ui_review_score") or 0
        control.notify("AutoDev — project complete", f"{project_name} is built — {known} known issue(s)"
                       + (f", design score {ui:.1f}/10." if ui else "."), "tada")

        log_path = f"{final_state['project_path']}/docs/final_state.json"
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w") as f:
            safe_state = {k: v for k, v in final_state.items()
                          if isinstance(v, (str, int, float, bool, list, dict))}
            json.dump(safe_state, f, indent=2, default=str)

    except KeyboardInterrupt:
        console.print("\n[yellow]Pipeline interrupted — progress is saved, it resumes from the last step[/yellow]")
        blog("system", "Pipeline interrupted — will resume from the last finished step.")
        raise  # stop the whole worker; staying alive in the watch loop would orphan it
    except Exception as e:
        console.print(f"\n[red]Pipeline error: {e}[/red]")
        blog("system", f"Pipeline error: {e}")
        bstate({"phase": "error", "is_complete": True})
        control.write_status(project_path, status="failed", error=str(e)[:300])
        control.notify("AutoDev — pipeline error", f"{project_name}: {str(e)[:120]}", "warning")
        import traceback
        traceback.print_exc()


# ── Worker (watch mode) ───────────────────────────────────────────────────────

def _claim_worker_slot() -> bool:
    """One worker at a time — two workers on the same design would build into
    the same project folder. Returns False if another live worker holds it."""
    if WORKER_PID.exists():
        try:
            pid = int(WORKER_PID.read_text().strip())
            os.kill(pid, 0)
            if pid != os.getpid():
                return False
        except (ValueError, ProcessLookupError, PermissionError):
            pass  # stale pid file from a crash or power cut
    WORKER_PID.write_text(str(os.getpid()))
    atexit.register(lambda: WORKER_PID.unlink(missing_ok=True))
    return True


def watch_inbox():
    """Poll design_inbox/ and run every design. Unfinished projects resume
    from their checkpoint; finished ones are skipped."""
    from server.bridge import log as blog
    console.print(f"[bold cyan]👀 Watching {DESIGN_INBOX} for design files...[/bold cyan]")
    blog("system", "AutoDev worker ready — watching for design uploads")
    processed = set()

    while True:
        for md_file in sorted(DESIGN_INBOX.glob("*.md")):
            if md_file.name not in processed:
                processed.add(md_file.name)
                console.print(f"\n[bold green]📄 Design: {md_file.name}[/bold green]")
                run_pipeline(str(md_file))
        time.sleep(3)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    flags = {a for a in sys.argv[1:] if a.startswith("--")}
    args = [a for a in sys.argv[1:] if not a.startswith("--")]

    if "--worker" in flags:
        if not _claim_worker_slot():
            console.print("[red]Another AutoDev worker is already running — exiting.[/red]")
            sys.exit(1)
        _check_provider_keys()
        _maybe_start_auditor("--auditor" in flags)
        watch_inbox()
    elif args:
        _start_server_background()
        _check_provider_keys()
        _maybe_start_auditor("--auditor" in flags)
        run_pipeline(args[0])
    else:
        run_control_plane(open_browser="--no-browser" not in flags)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[dim]Stopped.[/dim]")
