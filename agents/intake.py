from langchain_core.messages import HumanMessage
from pipeline.state import AgentState, log_agent
from tools.file_ops import create_project_structure, write_doc
from tools.llm_json import parse_llm_json
from server.bridge import log as blog, state as bstate
from router import router
from rich.console import Console

console = Console()


def run_intake(state: AgentState) -> AgentState:
    console.print("\n[bold blue]🧠 INTAKE AGENT starting...[/bold blue]")
    state["current_agent"] = "intake"
    state["phase"] = "intake"
    blog("intake", "Starting — analyzing design document")
    bstate({"current_agent": "intake", "phase": "intake"})

    llm = router.get_llm("fast")

    prompt = f"""You are an expert software architect. Analyze this technical design document carefully.

DESIGN DOCUMENT:
{state['design_content']}

Return a JSON object with these exact fields:
{{
  "project_type": "one of: api | webapp | mac_app | cli | ai_agent | scheduler | library | fullstack",
  "tech_stack": ["list", "of", "technologies"],
  "main_features": ["feature1", "feature2"],
  "complexity": "simple | medium | complex",
  "thinking": "Your deep analysis of what this project needs, challenges, and approach in 300+ words"
}}

Return ONLY valid JSON, no markdown, no explanation."""

    response = llm.invoke([HumanMessage(content=prompt)])
    raw = response.content.strip()

    data = parse_llm_json(raw, dict)
    if data:
        state["project_type"] = str(data.get("project_type") or "webapp")
        state["thinking"] = str(data.get("thinking") or raw)
        log_agent(state, "intake", f"Detected project type: {state['project_type']}")
    else:
        state["project_type"] = "webapp"
        state["thinking"] = raw

    create_project_structure(state["project_path"])
    write_doc(state["project_path"], "01_thinking.md",
              f"# Project Analysis\n\n**Type:** {state['project_type']}\n\n{state['thinking']}")

    console.print(f"[green]✅ Intake done — Type: {state['project_type']}[/green]")
    blog("intake", f"Done — project type: {state['project_type']}")
    bstate({"current_agent": "intake", "phase": "planning"})
    state["phase"] = "planning"
    return state
