"""
Optimizer agent — runs after coder, in parallel with security (both
read-only, see security.py's docstring for why). Suggested improvements go
into state for agents/code_merge.py to apply — never writes files directly.
"""
from langchain_core.messages import HumanMessage
from pipeline.state import AgentState
from tools.file_blocks import FILE_FORMAT, parse_file_blocks, parse_meta_json
from agents.coder import _existing_code_context, REVIEW_CODE_BUDGET
from server.bridge import log as blog, state as bstate
from router import router
from rich.console import Console

console = Console()


def run_optimizer(state: AgentState) -> AgentState:
    console.print("\n[bold blue]⚡ OPTIMIZER AGENT starting...[/bold blue]")
    blog("optimizer", "Starting — looking for performance/readability improvements")
    bstate({"current_agent": "optimizer", "phase": "optimizing"})

    llm = router.get_llm("coding")

    full_code = _existing_code_context(state["project_path"], REVIEW_CODE_BUDGET)
    if not full_code:
        console.print("[yellow]No source files to optimize[/yellow]")
        blog("optimizer", "No files to optimize — skipping")
        return {"_optimization_notes": [], "_optimized_files": {}}

    optimize_prompt = f"""You are a senior engineer doing a performance + readability pass.
This code is about to go to review — don't change behavior, only improve it.

PROJECT: {state['project_name']} ({state['project_type']})

CODEBASE (src/):
{full_code}

Look for: N+1 patterns, unnecessary loops/allocations, unclear naming, duplicated logic,
missing caching where obviously safe, non-idiomatic code.

Only include a file if you have a real, concrete improvement — don't rewrite files that
are already fine. At most 2 files.

First a JSON object: {{"notes": ["short description of each change"]}}
Then each improved file, complete, as a FILE block. If nothing needs improving: {{"notes": []}} and no blocks.

{FILE_FORMAT.format(root="the project folder, e.g. src/app.py")}"""

    resp = llm.invoke([HumanMessage(content=optimize_prompt)])
    raw = resp.content.strip()
    data = parse_meta_json(raw)
    notes = data.get("notes", []) if isinstance(data.get("notes"), list) else []
    optimized_files = parse_file_blocks(raw) or {
        fp: c for fp, c in (data.get("optimized_files") or {}).items() if fp and isinstance(c, str)
    }

    console.print(f"[green]✅ Optimization pass done — {len(optimized_files)} files suggested[/green]")
    blog("optimizer", f"Done — {len(optimized_files)} files suggested")
    # Partial return — parallel sibling of security (see agents/code_merge.py).
    return {"_optimization_notes": notes, "_optimized_files": optimized_files}
