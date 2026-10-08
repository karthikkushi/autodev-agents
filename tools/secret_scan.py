"""
The last check before a project leaves the Mac: no API key, token or private
key in its files or anywhere in its git history (a push sends the history,
so a key that was committed and later deleted still leaks).

Findings say where and what ("src/app.py: the value of GOOGLE_API_KEY"),
never the secret itself.
"""
import os
import re
import subprocess
from pathlib import Path

_PIPELINE_ENV = Path(__file__).resolve().parent.parent / ".env"
_KEY_NAME = re.compile(r"(_API_KEY|_KEY|_TOKEN|_SECRET|_PASSWORD)$|^(GITHUB|GH)_", re.IGNORECASE)
PATTERNS = {
    "a Google API key": r"AIza[\w\-]{30,}",
    "an NVIDIA API key": r"nvapi-[\w\-]{20,}",
    "an OpenRouter key": r"sk-or-v1-\w{20,}",
    "an OpenAI-style key": r"\bsk-[A-Za-z0-9]{32,}",
    "a Groq key": r"\bgsk_\w{20,}",
    "a GitHub token": r"\b(?:gh[pousr]_\w{30,}|github_pat_\w{20,})",
    "an AWS access key": r"\bAKIA[0-9A-Z]{16}\b",
    "a Slack token": r"\bxox[baprs]-[\w-]{10,}",
    "a private key": r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----",
}
_COMPILED = {name: re.compile(p) for name, p in PATTERNS.items()}


def _known_secrets() -> dict:
    """The pipeline's own credentials (name -> value), from the environment
    and the pipeline's .env file."""
    values = {k: v for k, v in os.environ.items() if _KEY_NAME.search(k) and v and len(v) >= 12}
    try:
        from dotenv import dotenv_values
        values.update({k: v for k, v in dotenv_values(_PIPELINE_ENV).items() if v and len(v) >= 12})
    except Exception:
        pass
    return values


def _find(text: str, known: dict) -> list[str]:
    hits = [f"the value of {name}" for name, value in known.items() if value in text]
    return hits + [name for name, rx in _COMPILED.items() if rx.search(text)]


def scan_repo(project_path: str) -> list[str]:
    """Every place a secret would be pushed from this git repo, or []."""
    root, known, findings = Path(project_path), _known_secrets(), []
    tracked = subprocess.run(["git", "ls-files", "-z"], cwd=root, capture_output=True, text=True).stdout
    for rel in filter(None, tracked.split("\0")):
        if re.search(r"(^|/)\.env$", rel):
            findings.append(f"{rel}: an .env file is tracked")
        try:
            text = (root / rel).read_text(errors="replace")
        except OSError:
            continue
        findings += [f"{rel}: {hit}" for hit in _find(text, known)]
    history = subprocess.run(["git", "log", "--all", "-p", "--no-color", "--no-ext-diff"], cwd=root,
                             capture_output=True, text=True).stdout
    findings += [f"git history: {hit}" for hit in _find(history, known)]
    return sorted(set(findings))
