"""
Test writer agent — runs in parallel with doc_writer_draft, right after
reviewer passes (see pipeline/graph.py). Writes a real test suite (unit +
edge cases) covering the finished code, straight to disk — tester_debugger.py
then just runs whatever's in tests/ instead of also generating its own basic
tests. Returns {} (nothing to persist to graph state): its real output is
the files on disk, and doc_writer_draft is its parallel sibling in this
step, so this must not touch any shared state key.
"""
import os
from langchain_core.messages import HumanMessage
from pipeline.state import AgentState
from tools.file_ops import write_doc, check_llm_path
from tools.file_blocks import FILE_FORMAT, parse_files
from agents.coder import _existing_code_context
from server.bridge import log as blog, state as bstate
from router import router
from rich.console import Console

console = Console()


def run_test_writer(state: AgentState) -> AgentState:
    console.print("\n[bold blue]🧷 TEST WRITER AGENT starting...[/bold blue]")
    blog("test_writer", "Starting — writing test suite")
    bstate({"current_agent": "test_writer", "phase": "test_writing"})

    llm = router.get_llm("coding")
    project_path = state["project_path"]

    # Whole-project view (full files up to a budget, then signatures): tests
    # written against a partial view imported functions that don't exist.
    full_code = _existing_code_context(project_path)
    if not full_code:
        console.print("[yellow]No source files to test[/yellow]")
        blog("test_writer", "No files to test — skipping")
        return {}

    test_prompt = f"""You are a test engineer. Write a real pytest test suite for this project.

PROJECT: {state['project_name']} ({state['project_type']})
PLAN: {state['plan'][:600]}

CODEBASE (everything under src/):
{full_code}

HOW THE TESTS ARE RUN: `python3 -m pytest tests/ -q` from the project root, with src/ on
PYTHONPATH. So import modules directly — `from app import app`, `from calculator import split` —
never `from src.app import ...`. For a Flask app use app.test_client(); for FastAPI use
fastapi.testclient.TestClient. Only test functions and routes that exist in the code above.
No browser automation (Playwright, Selenium, pyppeteer): the tests run offline in a sandbox that
can't download a browser, so every such test errors. For a static site (src/index.html), read the
HTML with Python's html.parser and check what the design asks for: sections and their ids, headings,
meta and Open Graph tags, image alt text, form fields and labels, and that linked local files exist.

Cover: main functionality (happy path), edge cases (empty input, invalid input,
boundary values), and integration between modules where relevant. Keep it focused:
at most 3 test files.

{FILE_FORMAT.format(root="tests/ (e.g. test_app.py)")}"""

    resp = llm.invoke([HumanMessage(content=test_prompt)])
    written = []
    os.makedirs(f"{project_path}/tests", exist_ok=True)
    for filepath, content in (parse_files(resp.content) or {}).items():
        fname = filepath.replace("tests/", "", 1) if filepath.startswith("tests/") else filepath
        if not check_llm_path(fname):
            continue
        full_path = f"{project_path}/tests/{fname}"
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w") as f:
            f.write(content)
        written.append(fname)
        console.print(f"[cyan]  🧷 Wrote: tests/{fname}[/cyan]")
    if not written:
        console.print("[yellow]Test writer: no complete test file in the reply[/yellow]")

    write_doc(project_path, "07b_tests_written.md",
        f"# Test Suite\n\n**Files:** {len(written)}\n\n" +
        "\n".join([f"- `tests/{f}`" for f in written]))

    console.print(f"[green]✅ Test writing done — {len(written)} files[/green]")
    blog("test_writer", f"Done — {len(written)} test files written")
    return {}
