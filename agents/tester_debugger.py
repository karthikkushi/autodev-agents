from langchain_core.messages import HumanMessage
from pipeline.state import AgentState, log_agent
from tools.safety import run_safe_command
from tools.file_ops import write_doc, read_file, write_project_file, python_syntax_error
from tools.file_blocks import FILE_FORMAT, parse_files
from agents.coder import _existing_code_context
from agents.test_writer import run_test_writer
from tools.pyenv import project_python, venv_python, install_command
from tools.web_search import search_web
from tools.memory import AgentMemory
from server.bridge import log as blog, state as bstate
from router import router
from rich.console import Console
import re, os

console = Console()
MAX_DEBUG_RETRIES = 3


def _detect_entry_point(project_path: str, project_type: str) -> tuple[str, str, dict]:
    """Auto-detect how to run the project. Returns (cmd, cwd, extra_env).

    Every path in the command is relative to the cwd it runs from — the
    debugger and decision agents can only write under src/, so a command
    that looks for files anywhere else could never be fixed by them."""
    src = f"{project_path}/src"
    tests_dir = f"{project_path}/tests"
    py = project_python(project_path)  # the project's own venv when it has one
    if os.path.isdir(tests_dir) and any(
        f.startswith("test_") and f.endswith(".py") for f in os.listdir(tests_dir)
    ):
        return f"{py} -m pytest tests/ -q", project_path, {"PYTHONPATH": src}
    # --help so a CLI doesn't fail just because it was given no arguments.
    if os.path.exists(f"{src}/main.py"):
        return f"{py} src/main.py --help", project_path, {}
    if os.path.exists(f"{src}/app.py"):
        return f"{py} src/app.py --help", project_path, {}
    if os.path.isdir(src):
        for name in sorted(os.listdir(src)):
            if os.path.exists(f"{src}/{name}/__main__.py"):
                return f"{py} -m {name} --help", src, {}
    if os.path.exists(f"{src}/index.html"):
        # A plain HTML site: nothing installs npm packages here, so a stray
        # package.json once sent every debug attempt to fixing `npm test`.
        # Without tests, check the page at least parses.
        return (f"{py} -c \"from html.parser import HTMLParser; "
                f"HTMLParser().feed(open('src/index.html').read())\"", project_path, {})
    if os.path.exists(f"{project_path}/package.json"):
        return "npm test", project_path, {}
    # Nothing runnable yet — this fails with a clear "can't open file
    # src/main.py", which the debugger can fix by writing that file.
    return f"{py} src/main.py --help", project_path, {}


def _error_query(error_text: str) -> str:
    """A web-search query from the most informative line of a failure."""
    lines = [l.strip() for l in error_text.splitlines() if l.strip()]
    for pattern in (r"^E\s+(\w+(Error|Exception)\b.*)", r"^(\w+(Error|Exception):.*)", r"^E\s+(.+)",
                    r"^(FAILED|ERROR)\s+(.*)"):
        for line in lines:
            m = re.match(pattern, line)
            if m:
                return m.group(1)[:120]
    return lines[-1][:120] if lines else "error"


def pytest_counts(output: str) -> tuple[int, int]:
    """(passed, failed) from pytest's summary line ("1 failed, 16 passed in
    0.2s"); errors count as failed. (0, 0) when there is no summary."""
    passed = failed = 0
    for n, word in re.findall(r"(\d+) (passed|failed|errors?)\b", (output or "")[-1500:]):
        if word == "passed":
            passed = int(n)
        else:
            failed += int(n)
    return passed, failed


def head_tail(text: str, limit: int) -> str:
    """Both ends of a long output: pytest puts the failure detail near the
    start and the pass/fail summary at the very end."""
    if len(text) <= limit:
        return text
    head = limit * 2 // 5
    return text[:head] + "\n...\n" + text[-(limit - head):]


def run_tester_debugger(state: AgentState) -> AgentState:
    console.print("\n[bold blue]🧪 TESTER/DEBUGGER AGENT starting...[/bold blue]")
    state["current_agent"] = "tester"
    state["phase"] = "testing"
    blog("tester", "Starting — running tests and debugging")
    bstate({"current_agent": "tester", "phase": "testing"})

    llm = router.get_llm("coding")
    memory = AgentMemory(state["project_path"])
    project_path = state["project_path"]

    # test_writer runs only after a passing review. A review loop that ends at
    # the decision agent skips it — a portfolio reached this point with no
    # tests at all — so write the suite here, with the same rules.
    existing_tests = []
    if os.path.isdir(f"{project_path}/tests"):
        existing_tests = [f for f in os.listdir(f"{project_path}/tests") if f.endswith(".py")]

    if not existing_tests:
        run_test_writer(state)

    # The coder may have added libraries to requirements.txt after the
    # environment step — install them into the project venv before running.
    if venv_python(project_path).exists():
        for req in (f"{project_path}/requirements.txt", f"{project_path}/src/requirements.txt"):
            if os.path.exists(req):
                run_safe_command(install_command(project_path, [], req), cwd=project_path)

    # Run the project
    run_cmd, run_cwd, run_env = _detect_entry_point(project_path, state["project_type"])
    console.print(f"[yellow]▶️  Running: {run_cmd}[/yellow]")

    debug_attempts = 0
    last_error = ""

    while debug_attempts <= MAX_DEBUG_RETRIES:
        ok, stdout, stderr = run_safe_command(run_cmd, cwd=run_cwd, env=run_env, sandbox_dir=project_path)
        combined_output = f"STDOUT:\n{stdout}\nSTDERR:\n{stderr}"

        # Exit code only — pytest and Python warnings write to stderr even
        # when everything passes.
        if ok:
            console.print("[green]✅ Project runs successfully![/green]")
            write_doc(project_path, "08_test_results.md",
                      f"# Test Results\n\n**Status:** PASS\n\n```\n{combined_output[:2000]}\n```")
            state["phase"] = "complete"
            state["github_ready"] = True
            log_agent(state, "tester", "All tests passed")
            memory.record_outcome("tester", "passed")
            blog("tester", "All tests passed!")
            bstate({"current_agent": "tester", "phase": "final_review", "github_ready": True})
            return state

        # Has errors - debug. pytest reports failures on stdout while a crash
        # reports on stderr (which may also hold harmless warnings) — keep both.
        error_text = f"{stderr.strip()}\n{stdout.strip()}".strip()
        if error_text == last_error:
            debug_attempts += 1
        last_error = error_text

        console.print(f"[red]❌ Error found (attempt {debug_attempts + 1}/{MAX_DEBUG_RETRIES + 1})[/red]")
        console.print(f"[red]{error_text[:300]}[/red]")
        blog("tester", f"Error (attempt {debug_attempts + 1}): {error_text[:120]}")

        if debug_attempts >= MAX_DEBUG_RETRIES:
            console.print("[red]🚨 Max debug retries reached — escalating to Decision Agent[/red]")
            state["phase"] = "decision"
            # Head and tail, not the first 500 chars: for pytest those were
            # mostly progress dots, and the "1 failed, 16 passed" summary the
            # decision agent needs was cut off.
            state["errors"].append(head_tail(error_text, 3000))
            return state

        # Debug: search for fix — from the most telling error line, not the
        # start of the output (for pytest that's a row of progress dots).
        search_result = search_web(f"python {_error_query(error_text)}", max_results=2)

        # The failing test files, so a wrong test can be fixed too (not just src/).
        test_files = ""
        for rel in sorted(set(re.findall(r"\btests/[\w/]+\.py", error_text)))[:3]:
            try:
                test_files += f"\n### {rel}\n```\n{read_file(f'{project_path}/{rel}')[:4000]}\n```"
            except Exception:
                pass

        # pytest puts the useful summary at the end of its output: keep both ends.
        shown_error = head_tail(error_text, 3500)
        debug_prompt = f"""You are debugging a Python project. Make the failing command pass.

COMMAND: {run_cmd}   (run from the project root; src/ is on PYTHONPATH)
OUTPUT:
{shown_error}

PROJECT CODE (src/):
{_existing_code_context(project_path)}
{'FAILING TEST FILES:' + test_files if test_files else ''}

WEB SEARCH RESULT:
{search_result[:1000]}

Fix the root cause, not just the symptom. If a test itself is wrong (imports something that
doesn't exist, expects behaviour the design never asked for), fix the test instead of the code.
Return only the files you change, each complete.

{FILE_FORMAT.format(root="the project root — src/app.py, tests/test_app.py")}"""

        debug_resp = llm.invoke([HumanMessage(content=debug_prompt)])
        fixes = parse_files(debug_resp.content) or {}
        for filepath, content in list(fixes.items()):
            bad = python_syntax_error(filepath, content)
            if bad:
                # Keep the working file; the next attempt sees the same failure.
                console.print(f"[yellow]  ⚠️ Kept the old {filepath} — the fix doesn't compile ({bad[:80]})[/yellow]")
                del fixes[filepath]
            elif write_project_file(project_path, filepath, content):
                console.print(f"[cyan]  🔧 Fixed: {filepath}[/cyan]")
        if fixes:
            memory.record_outcome("debugger", "fixed", f"Fixed: {error_text[:100]}")
        else:
            console.print("[yellow]Debugger: no complete file in the reply[/yellow]")

        debug_attempts += 1

    state["phase"] = "decision"
    return state
