import os
from pathlib import Path
from rich.console import Console

console = Console()

# Files that legitimately have no extension. Any other extension-less name in
# model output is almost always a step name returned as a path, with the
# instruction text as its "content" (e.g. src/check_file_paths).
EXTENSIONLESS_OK = {"Dockerfile", "Makefile", "Procfile", "LICENSE", ".gitignore", ".env.example"}


def check_llm_path(path: str) -> bool:
    """Validate a file path taken from model output before writing it.
    Prints a yellow warning and returns False if it must be skipped."""
    # Not stripped: callers write the path exactly as given, so a leading or
    # trailing space must be caught by the space check below.
    p = path or ""
    name = p.rstrip("/").rsplit("/", 1)[-1]
    reason = ""
    if not p.strip():
        reason = "empty path"
    elif os.path.isabs(p):
        reason = "absolute path"
    elif ".." in p:
        reason = "contains '..'"
    elif " " in p:
        reason = "contains spaces"
    elif not os.path.splitext(name)[1] and name not in EXTENSIONLESS_OK:
        reason = "no file extension"
    if reason:
        console.print(f"[yellow]  ⚠️ Skipped model-returned path {p!r}: {reason}[/yellow]")
        return False
    return True


def create_project_structure(project_path: str) -> None:
    dirs = [
        project_path,
        f"{project_path}/docs",
        f"{project_path}/src",
        f"{project_path}/tests",
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    console.print(f"[green]📁 Created project structure at {project_path}[/green]")


def write_doc(project_path: str, filename: str, content: str) -> str:
    path = f"{project_path}/docs/{filename}"
    with open(path, "w") as f:
        f.write(content)
    console.print(f"[blue]📄 Wrote doc: docs/{filename}[/blue]")
    return path


def write_code_file(project_path: str, relative_path: str, content: str) -> str:
    """Returns the written path, or "" if the path failed check_llm_path."""
    if not check_llm_path(relative_path):
        return ""
    # "tests/…" from a model means the project's tests/ folder, where pytest
    # runs — not src/tests/, where tests were silently never collected.
    shown = relative_path if relative_path.startswith("tests/") else f"src/{relative_path}"
    full_path = f"{project_path}/{shown}"
    Path(full_path).parent.mkdir(parents=True, exist_ok=True)
    with open(full_path, "w") as f:
        f.write(content)
    console.print(f"[cyan]💾 Wrote: {shown}[/cyan]")
    return full_path


ROOT_FILES = {"requirements.txt", "pyproject.toml", "setup.cfg", "setup.py", "pytest.ini", "package.json",
              "README.md", ".gitignore", ".env.example", "Procfile", "Dockerfile", "Makefile", "runtime.txt"}


def write_project_file(project_path: str, path: str, content: str) -> str:
    """For model replies whose paths are relative to the project root: src/…
    and tests/… go to their folders, known config files to the root, and a
    bare name like "app.py" means source code (models often drop the src/)."""
    p = path.strip().removeprefix(project_path.rstrip("/") + "/").removeprefix("./")
    # Told "paths relative to src/", models write tests as "../tests/x.py":
    # one leading ../ means the project root. Any other ".." is still refused.
    p = p.removeprefix("../")
    if p.startswith("src/"):
        return write_code_file(project_path, p[4:], content)
    if p.startswith("tests/"):
        return write_code_file(project_path, p, content)
    if p in ROOT_FILES:
        return write_root_file(project_path, p, content)
    return write_code_file(project_path, p, content)


def project_file_target(project_path: str, path: str) -> str:
    """Where write_project_file would write `path` ("" if it would refuse it),
    so a caller can look at the file that is about to be replaced."""
    p = path.strip().removeprefix(project_path.rstrip("/") + "/").removeprefix("./").removeprefix("../")
    if p in ROOT_FILES and not p.startswith(("src/", "tests/")):
        return f"{project_path}/{p}" if check_llm_path(p) else ""
    rel = p[4:] if p.startswith("src/") else p
    if not check_llm_path(rel):
        return ""
    return f"{project_path}/{rel}" if rel.startswith("tests/") else f"{project_path}/src/{rel}"


def python_syntax_error(path: str, content: str) -> str:
    """The SyntaxError a .py file would raise, or "" (always "" for other
    files). Fix writers check this first: a broken "fix" replaced a working
    37-test file, and with it every test the next round could have run."""
    if not path.endswith(".py"):
        return ""
    try:
        compile(content, path, "exec")
    except (SyntaxError, ValueError) as e:
        return f"{type(e).__name__}: {e}"
    return ""


def write_root_file(project_path: str, filename: str, content: str) -> str:
    """Returns the written path, or "" if the path failed check_llm_path."""
    if not check_llm_path(filename):
        return ""
    full_path = f"{project_path}/{filename}"
    Path(full_path).parent.mkdir(parents=True, exist_ok=True)
    with open(full_path, "w") as f:
        f.write(content)
    console.print(f"[cyan]💾 Wrote: {filename}[/cyan]")
    return full_path


def read_file(path: str) -> str:
    with open(path, "r") as f:
        return f.read()


def list_files(project_path: str) -> list:
    result = []
    for root, _, files in os.walk(project_path):
        for file in files:
            rel = os.path.relpath(os.path.join(root, file), project_path)
            result.append(rel)
    return result
