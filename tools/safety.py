import os
import psutil
import re
import subprocess
import shlex
from rich.console import Console

console = Console()

# Only these command prefixes are allowed
ALLOWED_COMMANDS = [
    "python", "python3", "pip", "pip3", "uv",
    "npm", "npx", "node",
    "swift", "swiftc", "xcodebuild",
    "git",
    "mkdir", "touch", "cp", "mv", "ls", "cat", "echo",
    "curl", "wget",
    "brew",
    "cargo", "rustc",
    "go", "java", "javac", "mvn", "gradle",
    "uvicorn", "gunicorn",
    "expo", "react-native",
    "pytest", "jest", "mocha",
]

# These are never allowed regardless
BLOCKED_PATTERNS = [
    "rm -rf", "rm -r /", "sudo rm",
    "chmod 777", "chmod -R 777",
    "dd if=", "mkfs",
    "> /dev/", ">> /dev/",
    "shutdown", "reboot", "halt",
    "killall", "kill -9",
    "launchctl", "csrutil",
    "defaults write com.apple.SoftwareUpdate",
    "systemsetup",
    "diskutil eraseDisk",
    "format",
    "sudo",
]


def is_safe_command(cmd: str) -> tuple[bool, str]:
    cmd_lower = cmd.strip().lower()

    for pattern in BLOCKED_PATTERNS:
        if pattern in cmd_lower:
            return False, f"Blocked: contains '{pattern}'"

    first_word = shlex.split(cmd)[0].split("/")[-1]
    if first_word not in ALLOWED_COMMANDS:
        return False, f"Blocked: '{first_word}' is not in the allowed command list"

    return True, "ok"


def get_health() -> dict:
    """Informational only — used for the dashboard's live CPU/RAM readout.
    Nothing in the pipeline blocks on this; every agent call is a network
    request to a cloud API, not local inference, so there's no local
    compute load to protect the machine from."""
    cpu = psutil.cpu_percent(interval=1)
    ram = psutil.virtual_memory()
    disk = psutil.disk_usage("/")
    return {
        "cpu_percent": cpu,
        "ram_percent": ram.percent,
        "ram_available_gb": round(ram.available / (1024**3), 1),
        "disk_percent": disk.percent,
    }


# Model-written code (tests, the generated app) runs unattended for days, so
# it runs under macOS's sandbox with writes allowed only inside its own
# project folder and temp dirs — a buggy or hostile test can't delete or
# overwrite anything else. Reads stay open: read isolation breaks Python /
# pytest / uv toolchains. Set AUTODEV_SANDBOX=0 to turn it off.
_SANDBOX_PROFILE = """(version 1)
(allow default)
(deny file-write*)
(allow file-write*
    (subpath (param "PROJECT"))
    (subpath "/private/tmp") (subpath "/private/var/folders") (subpath "/dev"))
"""
_SANDBOX_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "sandbox.sb")


def sandbox_prefix(project_dir: str) -> list[str]:
    """Argument list to put in front of a command so it can only write inside
    project_dir. Empty when sandboxing is off or unavailable."""
    if os.environ.get("AUTODEV_SANDBOX") == "0" or not os.path.exists("/usr/bin/sandbox-exec"):
        return []
    os.makedirs(os.path.dirname(_SANDBOX_FILE), exist_ok=True)
    if not os.path.exists(_SANDBOX_FILE) or open(_SANDBOX_FILE).read() != _SANDBOX_PROFILE:
        with open(_SANDBOX_FILE, "w") as f:
            f.write(_SANDBOX_PROFILE)
    return ["/usr/bin/sandbox-exec", "-f", _SANDBOX_FILE, "-D", f"PROJECT={os.path.realpath(project_dir)}"]


# Generated code never needs the pipeline's own credentials. It used to run
# with GOOGLE_API_KEY, NVIDIA_API_KEY... in its environment: one test that
# dumped os.environ into a file, and the key would have been committed and
# pushed to GitHub with the project.
_SECRET_ENV = re.compile(r"(_API_KEY|_KEY|_TOKEN|_SECRET|_PASSWORD|_ACCOUNT_ID)$|^(GITHUB|GH)_", re.IGNORECASE)


def clean_env(extra: dict = None) -> dict:
    """os.environ without anything that looks like a credential, plus `extra`."""
    env = {k: v for k, v in os.environ.items() if not _SECRET_ENV.search(k)}
    env.update(extra or {})
    return env


def run_safe_command(cmd: str, cwd: str = None, env: dict = None,
                     sandbox_dir: str = None) -> tuple[bool, str, str]:
    safe, reason = is_safe_command(cmd)
    if not safe:
        return False, "", reason
    if sandbox_dir:
        prefix = sandbox_prefix(sandbox_dir)
        if prefix:
            cmd = " ".join(shlex.quote(p) for p in prefix) + " /bin/sh -c " + shlex.quote(cmd)

    # Extra vars are layered over the real environment — handing subprocess a
    # bare dict would drop PATH/HOME and break every command. Env vars go here
    # rather than as a "VAR=x cmd" prefix because the allowlist checks the
    # command's first word. Sandboxed runs are generated code: no credentials.
    if sandbox_dir:
        run_env = clean_env(env)
    else:
        run_env = {**os.environ, **env} if env else None
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True,
            timeout=120, cwd=cwd, env=run_env
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out after 120s"
    except Exception as e:
        return False, "", str(e)
