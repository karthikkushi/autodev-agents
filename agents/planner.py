from langchain_core.messages import HumanMessage
from pipeline.state import AgentState, log_agent
from tools.file_ops import write_doc
from tools.llm_json import parse_llm_json
from server.bridge import log as blog, state as bstate
from router import router
from rich.console import Console

console = Console()


def _is_task_list(tasks) -> bool:
    return isinstance(tasks, list) and all(isinstance(t, dict) and t.get("name") for t in tasks)


def _parse_task_list(raw: str):
    """The task array from a model reply, or None. Only a list whose items
    are all dicts with a name counts, so bracketed prose or reasoning text
    before the array ("step [1]") is skipped, not taken for the task list."""
    tasks = parse_llm_json(raw, list, accept=_is_task_list)
    if not tasks or not _is_task_list(tasks):
        return None
    for t in tasks:
        t.setdefault("description", t["name"])
    return tasks


def run_planner(state: AgentState) -> AgentState:
    console.print("\n[bold blue]📋 PLANNER AGENT starting...[/bold blue]")
    state["current_agent"] = "planner"
    state["phase"] = "planning"
    blog("planner", "Starting — creating implementation plan")
    bstate({"current_agent": "planner", "phase": "planning"})

    llm = router.get_llm("reasoning")

    # A backtrack from the decision agent lands here. Without this, the new
    # plan was written from the design alone and the decision's reasoning and
    # proposed plan were silently overwritten — the same approach came back.
    backtrack_note = ""
    if state.get("backtrack_count", 0) > 0 and state.get("decision_log"):
        backtrack_note = (
            "\nA PREVIOUS BUILD OF THIS PROJECT FAILED. The senior engineer's decision:\n"
            f"{state['decision_log']}\n\nTheir proposed plan:\n{state['plan'][:3000]}\n\n"
            "Write the new plan so it avoids what went wrong.\n"
        )

    # Step 1: Full plan
    plan_prompt = f"""You are a senior software architect creating a complete project plan.

PROJECT TYPE: {state['project_type']}
DESIGN DOCUMENT:
{state['design_content']}

ANALYSIS:
{state['thinking']}
{backtrack_note}
WHAT THIS PIPELINE CAN RUN (hard limits — they override any tech-stack choice in the design):
- Nothing runs npm, a bundler or a build step. A website must work as plain files a browser opens:
  src/index.html with its css/ and js/ files, or a Python backend (Flask/FastAPI in src/app.py)
  serving templates and static files.
- So no Next.js, Astro, React/Vite, Vue, Svelte or TypeScript that needs compiling — even when the
  design says a framework is optional. If the design wants Tailwind, it comes from its Play CDN script.

Create a detailed implementation plan with:
1. Project overview
2. Tech stack with versions
3. Folder/file structure (tree format)
4. Implementation phases (each phase = group of related tasks)
5. Each task must have: id, name, description, files_to_create, dependencies

Format as markdown. Be specific and detailed."""

    plan_resp = llm.invoke([HumanMessage(content=plan_prompt)])
    state["plan"] = plan_resp.content
    write_doc(state["project_path"], "02_plan.md", f"# Implementation Plan\n\n{state['plan']}")

    # Step 2: Architecture
    arch_prompt = f"""Based on this plan, write a detailed architecture document.

PLAN:
{state['plan']}

Include:
- System components and their responsibilities
- Data flow between components
- API contracts (if any)
- Database schema (if any)
- Key design decisions and why

Format as markdown."""

    arch_resp = llm.invoke([HumanMessage(content=arch_prompt)])
    state["architecture"] = arch_resp.content
    write_doc(state["project_path"], "03_architecture.md", f"# Architecture\n\n{state['architecture']}")

    # Step 3: Flow diagram (Mermaid)
    flow_prompt = f"""Create a Mermaid flowchart for this project's main workflow.

PROJECT: {state['project_name']} ({state['project_type']})
PLAN SUMMARY: {(state['plan'] or '')[:1000]}

Return ONLY a valid Mermaid flowchart diagram starting with 'flowchart TD' or 'graph TD'.
No explanation, no markdown code fences."""

    flow_resp = llm.invoke([HumanMessage(content=flow_prompt)])
    state["flowchart"] = flow_resp.content
    write_doc(state["project_path"], "04_flowchart.md",
              f"# Flow Diagram\n\n```mermaid\n{state['flowchart']}\n```")

    # Step 4: Task list
    # Coding tasks only: environment, tests, docs and git have their own agents,
    # and a coder task for "set up CI" or "run tests" just writes junk into src/.
    task_prompt = f"""Extract the CODING tasks from this plan, in build order.

PLAN:
{state['plan']}

Rules:
- At most 15 tasks, each sized to create or change 1-3 source files.
- Only work that writes application code (backend, frontend pages, styles, scripts).
- Leave out environment setup, virtualenvs, installing packages, git, CI, linting, running
  tests, writing tests and README/docs — other pipeline agents handle those.
- Each description says exactly which files to create or change and what they must do.
- Web projects: put the stylesheet with the design system (CSS variables, type scale, base
  element and form-control styles, dark mode) BEFORE any page markup task, so every page is built
  on it instead of inventing its own styles.
- Nothing builds the project: no build, bundler or Tailwind-config tasks. If the design wants
  Tailwind, it comes from the Play CDN script, and the design-system task writes its config there.
- Every section and feature the design lists gets a task that builds it with real content.

Return ONLY a JSON array like:
[
  {{"id": 1, "name": "Create main module", "description": "Write src/main.py: the entry point that ...", "type": "code"}},
  ...
]

Return ONLY valid JSON array, no other text."""

    task_resp = llm.invoke([HumanMessage(content=task_prompt)])
    tasks = _parse_task_list(task_resp.content.strip())
    if tasks is None:
        # Long plans often come back cut off. One retry for a compact list
        # beats collapsing the whole project into a single coder task.
        console.print("[yellow]⚠️  Task list not valid JSON — retrying once for a compact list[/yellow]")
        retry_resp = llm.invoke([HumanMessage(content=(
            task_prompt + "\n\nYour previous reply was not a valid JSON array. Reply with ONLY a "
            "compact JSON array of at most 15 tasks, each description one or two sentences."
        ))])
        tasks = _parse_task_list(retry_resp.content.strip())
    if tasks is None:
        console.print("[red]❌ Task list still unparseable — falling back to one task[/red]")
        tasks = [{"id": 1, "name": "Implement project", "description": state['plan'][:500], "type": "code"}]
    state["task_list"] = tasks

    write_doc(state["project_path"], "05_tasks.md",
              f"# Task List\n\n" + "\n".join([f"- [ ] {t['name']}: {t['description']}" for t in state["task_list"]]))

    log_agent(state, "planner", f"Created {len(state['task_list'])} tasks")
    console.print(f"[green]✅ Planner done — {len(state['task_list'])} tasks created[/green]")
    blog("planner", f"Done — {len(state['task_list'])} tasks planned")
    bstate({"current_agent": "planner", "phase": "research"})
    state["phase"] = "research"
    return state
