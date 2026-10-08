"""
One Python environment per generated project (projects/<name>/.venv).

Before this, every project installed its libraries into the Mac's main
Python, so projects could break each other with conflicting versions and the
system Python filled up with whatever a model asked for. uv (already on this
Mac) creates the venv and installs ~10-100x faster than pip; plain
`python3 -m venv` + pip is the fallback when uv isn't there.
"""
import os
import re
import shutil
from pathlib import Path

from tools.safety import run_safe_command

VENV = ".venv"


def uv() -> str | None:
    return shutil.which("uv") or next(
        (p for p in [os.path.expanduser("~/.local/bin/uv"), "/usr/local/bin/uv", "/opt/homebrew/bin/uv"]
         if Path(p).exists()), None)


def venv_python(project_path: str) -> Path:
    return Path(project_path) / VENV / "bin" / "python"


def project_python(project_path: str) -> str:
    """The interpreter to run this project's code and tests with: its own venv
    when it has one (projects built before venvs existed use the main python3)."""
    py = venv_python(project_path)
    return str(py) if py.exists() else "python3"


def ensure_venv(project_path: str) -> tuple[bool, str]:
    """Create the project's venv with pytest in it. (ok, message)"""
    py = venv_python(project_path)
    if py.exists():
        return True, "exists"
    u = uv()
    cmd = f"{u} venv {VENV} --python python3 --quiet" if u else f"python3 -m venv {VENV}"
    ok, _, err = run_safe_command(cmd, cwd=project_path)
    if not ok or not py.exists():
        return False, f"venv creation failed: {err[:200]}"
    ok, _, err = run_safe_command(install_command(project_path, ["pytest"]), cwd=project_path)
    return ok, "created" if ok else f"venv created but pytest install failed: {err[:200]}"


def install_command(project_path: str, packages: list[str], requirements: str = "") -> str:
    """A command that installs into the project's venv (not the main Python)."""
    target = " ".join(packages) if packages else f"-r {requirements}"
    u = uv()
    if u:
        return f"{u} pip install --quiet --python {venv_python(project_path)} {target}"
    return f"{venv_python(project_path)} -m pip install --quiet {target}"


def install_requirements(project_path: str) -> None:
    """Install requirements.txt (root and src/) into the project's venv, if it
    has one. The coder can add a dependency after the environment step; the
    reviewer's import check would otherwise report it as missing."""
    if not venv_python(project_path).exists():
        return
    for req in (f"{project_path}/requirements.txt", f"{project_path}/src/requirements.txt"):
        if os.path.exists(req):
            run_safe_command(install_command(project_path, [], req), cwd=project_path)


def rewrite_pip_command(project_path: str, cmd: str) -> str | None:
    """Turn a model's "pip install flask pytest" into an install into the
    project venv. None for anything that isn't a plain pip install."""
    m = re.match(r"^(?:python3?\s+-m\s+)?pip3?\s+install\s+(.+)$", cmd.strip())
    if not m:
        return None
    args = [a for a in m.group(1).split() if a not in ("--user", "-U", "--upgrade")]
    if any(a.startswith(("-e", "--editable", "--index-url", "-i", "--extra-index-url")) for a in args):
        return None  # editable installs / custom indexes aren't something to run unattended
    return install_command(project_path, args)
