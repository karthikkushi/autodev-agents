from langchain_core.messages import HumanMessage
from pipeline.state import AgentState, log_agent
from tools.safety import run_safe_command
from pathlib import Path
from tools.file_ops import write_root_file
from tools.file_blocks import FILE_FORMAT, parse_files
from tools.pyenv import ensure_venv, rewrite_pip_command, install_command
from server.bridge import log as blog, state as bstate
from router import router
from rich.console import Console
import re

console = Console()


def run_environment(state: AgentState) -> AgentState:
    console.print("\n[bold blue]⚙️  ENVIRONMENT AGENT starting...[/bold blue]")
    state["current_agent"] = "environment"
    state["phase"] = "environment"
    blog("environment", "Starting — setting up project environment")
    bstate({"current_agent": "environment", "phase": "environment"})

    llm = router.get_llm("fast")
    project_path = state["project_path"]

    # Each project gets its own Python environment (projects/<name>/.venv) so
    # projects can't break each other or fill up the Mac's main Python.
    venv_ok, venv_msg = ensure_venv(project_path)
    console.print(f"[{'green' if venv_ok else 'yellow'}]🐍 Project venv: {venv_msg}[/]")

    # Install libraries found by researcher
    installed = []
    failed = []
    for lib_cmd in state["libraries"]:
        safe_cmd = lib_cmd.strip()
        # npx is left out on purpose: it downloads and *runs* a package, and
        # these commands come straight from model output.
        if not safe_cmd.startswith(("pip ", "pip3 ", "python -m pip", "python3 -m pip", "npm install", "npm i ")):
            continue
        if "pip" in safe_cmd.split()[0] or "-m pip" in safe_cmd:
            venv_cmd = rewrite_pip_command(project_path, safe_cmd) if venv_ok else None
            safe_cmd = venv_cmd or re.sub(r"^pip3?\s", "python3 -m pip ", safe_cmd)
        console.print(f"[yellow]📦 Installing: {lib_cmd.strip()}[/yellow]")
        ok, stdout, stderr = run_safe_command(safe_cmd, cwd=project_path)
        if ok:
            installed.append(safe_cmd)
            console.print(f"[green]  ✅ Done[/green]")
        else:
            failed.append(f"{safe_cmd}: {stderr[:100]}")
            console.print(f"[red]  ❌ Failed: {stderr[:80]}[/red]")

    # Generate project scaffold files
    scaffold_prompt = f"""Generate the initial project scaffold files for this project.

PROJECT TYPE: {state['project_type']}
PROJECT NAME: {state['project_name']}
PLAN: {state['plan'][:800]}
RESEARCH: {state['research_notes'][:600]}

Include: requirements.txt or package.json, README.md, .gitignore, and any other config files needed.
Do NOT include actual source code files — just config/setup files.

{FILE_FORMAT.format(root="the project root (e.g. requirements.txt, README.md, .gitignore)")}"""

    resp = llm.invoke([HumanMessage(content=scaffold_prompt)])
    for filename, content in (parse_files(resp.content) or {}).items():
        if filename.endswith(".py"):
            # Code belongs to the coder under src/. A root app.py would
            # also shadow src/app.py when tests run from the project root.
            console.print(f"[yellow]  ⚠️ Skipped scaffold source file {filename}[/yellow]")
            continue
        if filename.count("/") <= 1:
            write_root_file(project_path, filename, content)

    requirements = Path(project_path) / "requirements.txt"
    if venv_ok and requirements.exists():
        ok, _, err = run_safe_command(install_command(project_path, [], "requirements.txt"), cwd=project_path)
        console.print("[green]  ✅ requirements.txt installed into the venv[/green]" if ok
                      else f"[yellow]  ⚠️ requirements.txt install failed: {err[:120]}[/yellow]")
        if not ok:
            failed.append(f"requirements.txt: {err[:100]}")

    log_agent(state, "environment",
              f"Installed {len(installed)} libs. Failed: {len(failed)}. Scaffold created.")
    if failed:
        state["errors"].extend(failed)

    console.print(f"[green]✅ Environment ready[/green]")
    blog("environment", f"Done — {len(installed)} libs installed, scaffold ready")
    bstate({"current_agent": "environment", "phase": "coding"})
    state["phase"] = "coding"
    return state
