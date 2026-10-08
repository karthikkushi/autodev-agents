"""
Git commit (auto) + GitHub push (only after you press Approve in the
dashboard or phone app; Reject ends the run with nothing pushed).

The push uses the GitHub CLI (`gh`), logged in once on this Mac: its login
lives in the macOS keychain, so no token is ever written to a file, a remote
URL or a command line. Repos are created private, and a secret scan blocks
the push if an API key is anywhere in the files or the git history.
"""
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from pipeline import control
from pipeline.state import AgentState, log_agent
from tools.safety import run_safe_command
from tools import secret_scan
from server.bridge import log as blog, state as bstate
from rich.console import Console
from datetime import datetime

console = Console()
APPROVAL_POLL_INTERVAL = 5   # seconds
APPROVAL_TIMEOUT = 3600      # 1 hour max wait
REPO_VISIBILITY = "private"  # user decision 2026-09-30: nothing is public unless changed here

# Same repo-root flag files server/server.py writes (phone/dashboard approve)
# and reads (github_ready banner). Keeping them per-project meant an approval
# never reached the pipeline.
REPO_ROOT = Path(__file__).resolve().parent.parent
APPROVAL_FLAG = REPO_ROOT / ".github_approved"
REJECT_FLAG = REPO_ROOT / ".github_rejected"
PENDING_FLAG = REPO_ROOT / ".pending_approval"


def run_git_commit(state: AgentState) -> AgentState:
    console.print("\n[bold blue]📦 GIT COMMIT starting...[/bold blue]")
    state["current_agent"] = "git_commit"
    state["phase"] = "git_commit"
    blog("git_commit", "Committing project to git")
    bstate({"current_agent": "git_commit", "phase": "git_commit"})

    project_path = state["project_path"]

    # Init git if not already
    ok, _, _ = run_safe_command("git init", cwd=project_path)
    if not ok:
        console.print("[yellow]Git init skipped[/yellow]")
    _set_commit_author(project_path)

    # .gitignore: the model-written one may miss these, and committing .venv
    # would push thousands of library files (and pipeline internals) to GitHub.
    gitignore_path = f"{project_path}/.gitignore"
    existing = open(gitignore_path).read() if os.path.exists(gitignore_path) else ""
    required = ["__pycache__/", "*.pyc", ".env", "node_modules/", ".memory/", "*.pkl", ".venv/",
                ".autodev_status.json", ".pytest_cache/"]
    missing = [line for line in required if line not in existing.splitlines()]
    if missing:
        with open(gitignore_path, "a") as f:
            f.write(("\n" if existing and not existing.endswith("\n") else "") + "\n".join(missing) + "\n")

    # The message file used to live in the project folder, so `git add .`
    # committed it. It now goes inside .git/, which git never tracks.
    if os.path.exists(f"{project_path}/.git_commit_msg.txt"):
        run_safe_command("git rm -q --cached --ignore-unmatch .git_commit_msg.txt", cwd=project_path)
        os.remove(f"{project_path}/.git_commit_msg.txt")

    # Stage all files
    run_safe_command("git add .", cwd=project_path)

    # Commit — a resumed or rebuilt project already has its first commit.
    has_history, _, _ = run_safe_command("git rev-parse --verify -q HEAD", cwd=project_path)
    title = (f"chore: update build of {state['project_name']}" if has_history
             else f"feat: initial complete build of {state['project_name']}")
    commit_msg = (
        f"{title}\n\n"
        f"Built by AutoDev pipeline\n"
        f"Type: {state['project_type']}\n"
        f"Files: {len(state.get('files_written', []))}\n"
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    )
    msg_file = f"{project_path}/.git/AUTODEV_COMMIT_MSG"
    with open(msg_file, "w") as f:
        f.write(commit_msg)

    ok, stdout, stderr = run_safe_command(f"git commit -F {msg_file}", cwd=project_path)
    if ok:
        console.print("[green]✅ Git commit done[/green]")
        log_agent(state, "git_commit", "Committed successfully")
    else:
        console.print(f"[yellow]Git commit warning: {stderr[:100]}[/yellow]")

    state["phase"] = "awaiting_github_approval"
    state["github_ready"] = True

    # A leftover approval or rejection from an earlier project must not
    # decide this one.
    APPROVAL_FLAG.unlink(missing_ok=True)
    REJECT_FLAG.unlink(missing_ok=True)
    # Broadcast notification (picked up by WebSocket server)
    PENDING_FLAG.write_text("github_push")

    blog("git_commit", "Committed! Check it, then Approve (push) or Reject (keep it on this Mac)")
    control.notify("AutoDev — approval needed",
                   f"{state['project_name']} is built and committed. Approve the GitHub push or Reject it in "
                   f"the dashboard (it waits {APPROVAL_TIMEOUT // 60} min).", "white_check_mark")
    bstate({"current_agent": "git_commit", "phase": "awaiting_github_approval", "github_ready": True})

    console.print("[bold yellow]📱 Waiting for your approval to push to GitHub...[/bold yellow]")
    console.print("[dim]Approve from your phone app or web dashboard[/dim]")
    return state


def wait_for_github_approval(state: AgentState) -> AgentState:
    start = time.time()

    while time.time() - start < APPROVAL_TIMEOUT:
        if REJECT_FLAG.exists():
            # Rejected in the dashboard: the run ends here, nothing is pushed.
            REJECT_FLAG.unlink(missing_ok=True)
            APPROVAL_FLAG.unlink(missing_ok=True)
            PENDING_FLAG.unlink(missing_ok=True)
            state["github_approved"] = False
            console.print("[yellow]✋ GitHub push rejected — the project stays on this Mac only[/yellow]")
            blog("git_commit", "Push rejected — nothing was sent to GitHub. Rebuild the project any time.")
            log_agent(state, "git_commit", "GitHub push rejected by the user")
            return state
        if APPROVAL_FLAG.exists():
            APPROVAL_FLAG.unlink(missing_ok=True)
            PENDING_FLAG.unlink(missing_ok=True)
            state["github_approved"] = True
            console.print("[green]✅ GitHub push approved![/green]")
            return state
        # Also check state directly (set by WebSocket handler)
        if state.get("github_approved"):
            PENDING_FLAG.unlink(missing_ok=True)
            return state
        time.sleep(APPROVAL_POLL_INTERVAL)

    PENDING_FLAG.unlink(missing_ok=True)
    console.print("[yellow]⏰ Approval timeout — skipping GitHub push[/yellow]")
    return state


def run_github_push(state: AgentState) -> AgentState:
    console.print("\n[bold blue]🚀 GITHUB PUSH starting...[/bold blue]")
    state["current_agent"] = "github_push"
    state["phase"] = "github_push"
    blog("github_push", "Pushing to GitHub...")
    bstate({"current_agent": "github_push", "phase": "github_push"})

    project_path = state["project_path"]
    pushed = _push(project_path, state["project_name"], state)
    if not pushed:
        log_agent(state, "github_push", "Not pushed — see the log")

    bstate({"current_agent": "github_push", "phase": "complete", "is_complete": True})
    state["phase"] = "complete"
    state["is_complete"] = True
    return state


def _gh(args: list[str], cwd: str, timeout: int = 180) -> tuple[bool, str]:
    """Run the GitHub CLI with fixed arguments (no shell)."""
    gh = shutil.which("gh") or next((p for p in ("/usr/local/bin/gh", "/opt/homebrew/bin/gh") if os.path.exists(p)), None)
    if not gh:
        return False, "the GitHub CLI (gh) isn't installed"
    try:
        r = subprocess.run([gh, *args], cwd=cwd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as e:
        return False, str(e)
    return r.returncode == 0, (r.stdout.strip() or r.stderr.strip())


def _set_commit_author(project_path: str) -> None:
    """Commit as the GitHub account, with GitHub's private no-reply email, when
    gh is logged in. Git's default was "APPLE <apple@APPLEs-MacBook-Pro.local>":
    the Mac's name on GitHub, linked to no account. Set in this repo only."""
    ok, out = _gh(["api", "user", "--jq", '.login + " " + (.id|tostring)'], project_path, timeout=30)
    parts = out.split() if ok else []
    if len(parts) == 2 and parts[1].isdigit() and re.fullmatch(r"[A-Za-z0-9-]+", parts[0]):
        login, uid = parts
        run_safe_command(f"git config user.name {login}", cwd=project_path)
        run_safe_command(f"git config user.email {uid}+{login}@users.noreply.github.com", cwd=project_path)


def _say(msg: str, style: str = "yellow") -> None:
    console.print(f"[{style}]{msg}[/{style}]")
    blog("github_push", msg)


def _push(project_path: str, project_name: str, state: AgentState) -> bool:
    # 1. Never push a secret — the files and the whole history are checked.
    leaks = secret_scan.scan_repo(project_path)
    if leaks:
        _say(f"🔒 Push blocked — {len(leaks)} secret(s) would have been pushed: " + "; ".join(leaks[:3]), "red")
        control.notify("AutoDev — push blocked",
                       f"{project_name} was NOT pushed: {leaks[0]}. Remove it (and from git history) first.",
                       "no_entry")
        return False
    # 2. Logged in to GitHub on this Mac (one-time `gh auth login`)?
    ok, _ = _gh(["auth", "status", "--hostname", "github.com"], project_path, timeout=30)
    ok_user, user = _gh(["api", "user", "--jq", ".login"], project_path, timeout=30) if ok else (False, "")
    if not (ok and ok_user and user):
        _say("GitHub isn't set up on this Mac — not pushed. Run `gh auth login` once; "
             "the project stays committed locally.")
        return False
    _gh(["auth", "setup-git"], project_path, timeout=30)  # git uses gh's keychain login for github.com

    # 3. Create the repo the first time (private); later pushes go to the same repo.
    repo = f"{user}/{re.sub(r'[^A-Za-z0-9._-]', '-', project_name)}"
    run_safe_command("git branch -M main", cwd=project_path)
    exists, _ = _gh(["repo", "view", repo, "--json", "name"], project_path, timeout=30)
    if not exists:
        ok, out = _gh(["repo", "create", repo, f"--{REPO_VISIBILITY}", "--source", ".", "--remote", "origin",
                       "--push"], project_path)
    else:
        # Plain https remote: the login comes from gh, never from the URL.
        run_safe_command("git remote remove origin", cwd=project_path)
        run_safe_command(f"git remote add origin https://github.com/{repo}.git", cwd=project_path)
        ok, out, err = run_safe_command("git push -u origin main", cwd=project_path)
        out = out or err
    if ok:
        _say(f"🎉 Pushed to GitHub ({REPO_VISIBILITY}): https://github.com/{repo}", "bold green")
        log_agent(state, "github_push", f"Pushed to github.com/{repo} ({REPO_VISIBILITY})")
        control.notify("AutoDev — pushed", f"{project_name} is on GitHub: github.com/{repo}", "rocket")
        return True
    _say(f"Push failed: {out[:200]}", "red")
    return False
