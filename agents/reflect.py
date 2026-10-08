"""
Reflection node — runs after intake, planner, researcher.
Agent critiques its own output and scores quality.
Low score triggers a retry of that agent.
"""
from langchain_core.messages import HumanMessage
from pipeline.state import AgentState, log_agent
from server.bridge import log as blog, state as bstate
from tools.llm_json import parse_llm_json
from router import router
from rich.console import Console

console = Console()

REFLECT_PROMPTS = {
    "intake": """Review this project analysis. Is it complete and accurate?
OUTPUT: {output}
Score 0.0-1.0. Return JSON: {{"score": 0.9, "issues": [], "ok": true}}""",

    "planner": """Review this project plan. Is it detailed enough to build from?
Are all tasks clear? Is the architecture solid?
OUTPUT: {output}
Score 0.0-1.0. Return JSON: {{"score": 0.9, "issues": [], "ok": true}}""",

    "researcher": """Review this research. Does it cover all libraries and patterns needed?
Are there gaps?
OUTPUT: {output}
Score 0.0-1.0. Return JSON: {{"score": 0.9, "issues": [], "ok": true}}""",
}


def _reflect(agent: str, output: str) -> tuple[float, bool, list]:
    llm = router.get_llm("fast")
    template = REFLECT_PROMPTS.get(agent, "Review this output. Score 0.0-1.0. Return JSON: {{\"score\": 0.8, \"issues\": [], \"ok\": true}}")
    prompt = template.format(output=output[:2000])
    resp = llm.invoke([HumanMessage(content=prompt)])
    data = parse_llm_json(resp.content, dict)
    if data:
        try:
            score = float(data.get("score", 0.8))
            issues = data.get("issues", [])
            ok = score >= 0.6
            return score, ok, issues
        except Exception:
            pass
    return 0.8, True, []


def reflect_after_intake(state: AgentState) -> AgentState:
    # Runs in parallel with planner (see pipeline/graph.py) — planner is the
    # "primary" sibling and keeps the normal full-state return, so this one
    # must not touch shared state at all (console/phone logging only) or the
    # two branches would conflict when LangGraph merges the parallel step.
    score, ok, issues = _reflect("intake", state["thinking"])
    blog("intake", f"Self-reflection score: {score:.2f}")
    if not ok:
        console.print(f"[yellow]🔄 Intake reflect score {score:.2f} — issues: {issues}[/yellow]")
    else:
        console.print(f"[dim green]✓ Intake reflect {score:.2f}[/dim green]")
    return {}


def reflect_after_planner(state: AgentState) -> AgentState:
    score, ok, issues = _reflect("planner", state["plan"])
    log_agent(state, "reflect:planner", f"Score: {score:.2f} OK: {ok}")
    blog("planner", f"Self-reflection score: {score:.2f}")
    bstate({"current_agent": "planner"})
    if not ok:
        console.print(f"[yellow]🔄 Planner reflect score {score:.2f}[/yellow]")
    else:
        console.print(f"[dim green]✓ Planner reflect {score:.2f}[/dim green]")
    return state


def reflect_after_researcher(state: AgentState) -> AgentState:
    score, ok, issues = _reflect("researcher", state["research_notes"])
    log_agent(state, "reflect:researcher", f"Score: {score:.2f} OK: {ok}")
    blog("researcher", f"Self-reflection score: {score:.2f}")
    bstate({"current_agent": "researcher"})
    state["_research_reflect_score"] = score
    if not ok:
        console.print(f"[yellow]🔄 Research reflect score {score:.2f} — may need more research[/yellow]")
    else:
        console.print(f"[dim green]✓ Research reflect {score:.2f}[/dim green]")
    return state
