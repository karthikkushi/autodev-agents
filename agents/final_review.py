"""
Final review agent — runs after all tasks pass testing.
Does a complete project audit, generates summary doc,
then hands off to git commit.
"""
from langchain_core.messages import HumanMessage
from pipeline.state import AgentState, log_agent
from tools.file_ops import write_doc, list_files
from server.bridge import log as blog, state as bstate
from router import router
from rich.console import Console
from datetime import datetime

console = Console()


def run_final_review(state: AgentState) -> AgentState:
    console.print("\n[bold green]🏁 FINAL REVIEW AGENT starting...[/bold green]")
    state["current_agent"] = "final_review"
    state["phase"] = "final_review"
    blog("final_review", "Starting — generating final project report")
    bstate({"current_agent": "final_review", "phase": "final_review"})

    llm = router.get_llm("review", model_override="gemini-2.0-flash")

    all_files = list_files(state["project_path"])
    src_files = [f for f in all_files if f.startswith("src/")]
    doc_files = [f for f in all_files if f.startswith("docs/")]

    summary_prompt = f"""Write a complete project summary for this completed project.

PROJECT: {state['project_name']} ({state['project_type']})
FILES CREATED: {len(src_files)} source files, {len(doc_files)} docs
FILE LIST: {', '.join(src_files[:20])}

ORIGINAL DESIGN:
{state['design_content'][:1000]}

ARCHITECTURE:
{state['architecture'][:800]}

Write a concise, professional project summary including:
1. What was built
2. How to run it
3. Key features implemented
4. Tech stack used
5. File structure overview"""

    resp = llm.invoke([HumanMessage(content=summary_prompt)])

    # Build complete final report
    agent_log_lines = []
    for entry in state.get("agent_logs", [])[-30:]:
        agent_log_lines.append(f"- [{entry['time'][:19]}] {entry['agent']}: {entry['message']}")

    final_report = f"""# Project Complete: {state['project_name']}

**Built:** {datetime.now().strftime('%Y-%m-%d %H:%M')}
**Type:** {state['project_type']}
**Files:** {len(src_files)} source | {len(doc_files)} docs

---

## Summary
{resp.content}

---

## Files Created
{chr(10).join(['- ' + f for f in src_files])}

---

## Agent Execution Log
{chr(10).join(agent_log_lines)}

---

## Errors Encountered & Resolved
{chr(10).join(['- ' + e for e in state.get('errors', [])]) or 'None'}

---

## Known Issues
{chr(10).join(['- ' + e for e in state.get('known_issues', [])]) or 'None — all review/test issues resolved.'}
{f"{chr(10)}These issues were still open when the pipeline's retry limits ran out (review patch rounds, decision fix attempts or planner backtracks) and were not resolved automatically. Any last fixes applied for them are untested. Manual follow-up recommended." if state.get('known_issues') else ""}
"""

    write_doc(state["project_path"], "10_final_report.md", final_report)

    log_agent(state, "final_review", f"Project complete. {len(src_files)} files.")
    console.print(f"[bold green]🎉 Final review complete — {len(src_files)} source files[/bold green]")
    blog("final_review", f"Done — {len(src_files)} source files, {len(doc_files)} docs")
    bstate({"current_agent": "final_review", "phase": "git_commit"})
    state["phase"] = "git_commit"
    return state
