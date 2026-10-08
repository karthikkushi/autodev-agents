"""
Critic gate — runs after researcher reflect.
Decides: is the plan + research good enough to start coding?
If not, sends back to researcher for another pass.
Max 2 extra research passes to avoid infinite loops.
"""
from langchain_core.messages import HumanMessage
from pipeline.state import AgentState, log_agent
from server.bridge import log as blog, state as bstate
from tools.llm_json import parse_llm_json
from router import router
from rich.console import Console

console = Console()
MAX_RESEARCH_PASSES = 2


def run_critic_gate(state: AgentState) -> AgentState:
    console.print("\n[bold magenta]🔬 CRITIC GATE checking plan + research...[/bold magenta]")

    research_passes = state.get("_research_passes", 0)
    if research_passes >= MAX_RESEARCH_PASSES:
        console.print(f"[yellow]Critic: max research passes reached — proceeding anyway[/yellow]")
        state["_critic_pass"] = True
        return state

    llm = router.get_llm("reasoning")

    prompt = f"""You are a quality gate. Evaluate if this plan + research is sufficient to start coding.

PROJECT TYPE: {state['project_type']}
PLAN (summary): {state['plan'][:1200]}
RESEARCH NOTES: {state['research_notes'][:1200]}

Criteria:
1. Are all required libraries identified?
2. Is the architecture clear enough to code from?
3. Are there obvious missing pieces?
4. Is the task list actionable?

Return JSON:
{{"pass": true, "confidence": 0.85, "gaps": [], "reason": "..."}}
Return ONLY valid JSON."""

    resp = llm.invoke([HumanMessage(content=prompt)])
    data = parse_llm_json(resp.content, dict)

    passed = True
    confidence = 0.8
    gaps = []

    if data:
        try:
            passed = data.get("pass", True)
            confidence = float(data.get("confidence", 0.8))
            gaps = data.get("gaps", [])
            reason = data.get("reason", "")

            if passed:
                console.print(f"[green]✅ Critic PASS (confidence: {confidence:.2f})[/green]")
            else:
                console.print(f"[yellow]⚠️  Critic FAIL (confidence: {confidence:.2f}) — needs more research[/yellow]")
                console.print(f"[yellow]   Gaps: {gaps}[/yellow]")
        except Exception:
            pass

    state["_critic_pass"] = passed
    state["_critic_confidence"] = confidence
    state["_research_passes"] = research_passes + (0 if passed else 1)
    log_agent(state, "critic", f"Pass: {passed}, confidence: {confidence:.2f}, gaps: {gaps}")
    verdict = "PASS" if passed else "FAIL — needs more research"
    blog("researcher", f"Critic gate: {verdict} (confidence: {confidence:.0%})")
    bstate({"current_agent": "researcher"})
    return state


def route_critic(state: AgentState) -> str:
    if state.get("_critic_pass", True):
        return "environment"
    return "researcher"
