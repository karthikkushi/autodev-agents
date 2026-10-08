"""
Research phase — split into 3 independent parallel branches
(research_web, research_github, research_libraries) that fan out from
reflect_planner and join at research_merge. Each branch writes to its own
_-prefixed scratch field, so they never touch the same state key or file —
safe to run concurrently. research_merge is the only one that produces the
real research_notes/libraries the rest of the pipeline reads.
"""
from langchain_core.messages import HumanMessage
from pipeline.state import AgentState, log_agent
from tools.file_ops import write_doc
from tools.llm_json import parse_llm_json
from tools.web_search import search_web, search_github
from tools.memory import AgentMemory
from server.bridge import log as blog, state as bstate
from router import router
from rich.console import Console

console = Console()


def _json_list(raw: str) -> list:
    """First JSON array in a model reply, or [] — a malformed reply used to
    raise out of json.loads and kill the whole pipeline."""
    return parse_llm_json(raw, list) or []


def run_research_web(state: AgentState) -> AgentState:
    console.print("\n[bold blue]🔍 RESEARCH (web) starting...[/bold blue]")
    blog("researcher", "Web search — patterns and pitfalls")
    bstate({"current_agent": "researcher", "phase": "research"})

    llm = router.get_llm("fast")
    query_prompt = f"""You are a research agent. Given this project, list exactly what to search for on the web.

PROJECT TYPE: {state['project_type']}
PLAN SUMMARY: {state['plan'][:800]}

List 4 specific search queries covering: code examples/patterns, common pitfalls,
testing approaches, and deployment/packaging specifics. (Library recommendations
are handled separately — don't include those.)

Return as JSON array of strings: ["query1", "query2", ...]
Return ONLY valid JSON."""

    resp = llm.invoke([HumanMessage(content=query_prompt)])
    raw = resp.content.strip()
    queries = _json_list(raw) or [
        f"{state['project_type']} code examples",
        f"{state['project_type']} common mistakes",
    ]

    results = []
    for query in queries[:4]:
        result = search_web(query, max_results=3)
        results.append(f"### Search: {query}\n{result}")

    web_results = "\n\n".join(results)
    console.print("[green]✅ Web research done[/green]")
    blog("researcher", "Web search done")
    # Partial return — runs in parallel with research_github/research_libraries
    # (see pipeline/graph.py), so this must touch only its own key.
    return {"_web_results": web_results}


def run_research_github(state: AgentState) -> AgentState:
    console.print("\n[bold blue]🔍 RESEARCH (github) starting...[/bold blue]")
    blog("researcher", "GitHub search — similar projects")

    gh_result = search_github(f"{state['project_type']} {state['project_name']} example")

    console.print("[green]✅ GitHub research done[/green]")
    blog("researcher", "GitHub search done")
    # Partial return — parallel sibling of research_web/research_libraries.
    return {"_github_results": gh_result}


def run_research_libraries(state: AgentState) -> AgentState:
    console.print("\n[bold blue]🔍 RESEARCH (libraries) starting...[/bold blue]")
    blog("researcher", "Library search — recommended packages")

    llm = router.get_llm("fast")
    # The design's tech stack is the contract: tonight a Flask design that said
    # "no build step" got an npm Tailwind install from a plan-only view.
    lib_prompt = f"""You are a research agent. Recommend the libraries this project needs installed.

PROJECT TYPE: {state['project_type']}
DESIGN DOCUMENT (its tech stack is binding):
{state['design_content'][:2500]}

PLAN SUMMARY: {state['plan'][:800]}

Rules: only what the design's tech stack actually needs — no extra frameworks, no build tools or
npm packages if the design says plain HTML/CSS/JS or "no build step". Standard-library modules
need no install. Include pytest for Python projects.

Return a JSON array of install commands, most important first, e.g. ["pip install flask", "pip install pytest"].
Return ONLY valid JSON array."""

    resp = llm.invoke([HumanMessage(content=lib_prompt)])
    raw = resp.content.strip()
    candidates = [c for c in _json_list(raw) if isinstance(c, str)]

    console.print(f"[green]✅ Library research done — {len(candidates)} candidates[/green]")
    blog("researcher", f"Library search done — {len(candidates)} candidates")
    # Partial return — parallel sibling of research_web/research_github.
    return {"_library_candidates": candidates}


def run_research_merge(state: AgentState) -> AgentState:
    console.print("\n[bold blue]🔍 RESEARCH merge — synthesizing...[/bold blue]")
    state["current_agent"] = "researcher"
    state["phase"] = "research"

    llm = router.get_llm("fast")
    memory = AgentMemory(state["project_path"])
    cached = memory.search(f"{state['project_type']} {state['project_name']} implementation")
    if cached:
        console.print("[yellow]📚 Found relevant memory from past projects[/yellow]")

    combined = (
        f"{state.get('_web_results', '')}\n\n"
        f"### GitHub References\n{state.get('_github_results', '')}"
    )

    synth_prompt = f"""Synthesize this research into actionable guidance for building the project.

PROJECT: {state['project_name']} ({state['project_type']})
PLAN: {state['plan'][:600]}

RESEARCH RESULTS:
{combined[:4000]}

CANDIDATE LIBRARIES: {', '.join(state.get('_library_candidates', []))}

{'PAST MEMORY (from similar projects):' + cached if cached else ''}

Write a concise research summary with:
1. Recommended libraries (with install commands) — refine the candidates above if needed
2. Key patterns to follow
3. Pitfalls to avoid
4. Useful code snippets or references

Be specific and practical."""

    synth = llm.invoke([HumanMessage(content=synth_prompt)])
    state["research_notes"] = synth.content
    state["libraries"] = state.get("_library_candidates", [])

    write_doc(state["project_path"], "06_research.md",
              f"# Research Notes\n\n{state['research_notes']}\n\n## Libraries\n" +
              "\n".join([f"- `{lib}`" for lib in state["libraries"]]))

    memory.store("researcher", state["research_notes"], {"project": state["project_name"]})
    log_agent(state, "researcher", f"Research complete. Found {len(state['libraries'])} libraries.")
    console.print(f"[green]✅ Research merged — {len(state['libraries'])} libraries identified[/green]")
    blog("researcher", f"Done — {len(state['libraries'])} libraries identified")
    bstate({"current_agent": "researcher", "phase": "environment"})
    state["phase"] = "environment"
    return state
