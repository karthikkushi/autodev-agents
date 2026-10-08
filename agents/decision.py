from langchain_core.messages import AIMessage, HumanMessage
from pipeline.state import AgentState, log_agent, MAX_REVIEW_ROUNDS, MAX_DECISION_CALLS, MAX_BACKTRACKS
from tools.file_ops import write_doc, write_project_file, python_syntax_error
from tools.file_blocks import FILE_FORMAT, parse_file_blocks, parse_meta_json
from agents.coder import _existing_code_context, _damage, REVIEW_CODE_BUDGET
from agents.tester_debugger import pytest_counts, _error_query
from tools.web_search import search_web, search_github
from tools.memory import AgentMemory
from server.bridge import log as blog, state as bstate
from router import router
from rich.console import Console

console = Console()


def _apply_code_fixes(project_path: str, code_fixes) -> int:
    """Write the model's {path: content} fixes (src/, tests/ or root config)
    and return how many were written. Junk paths are skipped."""
    if not isinstance(code_fixes, dict):
        return 0
    written = 0
    for filepath, content in code_fixes.items():
        if not isinstance(content, str):
            continue
        bad = python_syntax_error(filepath or "", content)
        if bad:
            console.print(f"[yellow]  ⚠️ Kept the old {filepath} — the fix doesn't compile ({bad[:80]})[/yellow]")
            continue
        why = _damage(project_path, filepath or "", content)
        if why:
            console.print(f"[yellow]  ⚠️ Kept the old {filepath} — the fix would damage it: {why[:80]}[/yellow]")
            continue
        if write_project_file(project_path, filepath or "", content):
            written += 1
    return written


def _apply_decision(state: AgentState, decision_data: dict, fix_only: bool = False) -> None:
    """Loop bookkeeping for one decision call: counters, stop rules, applying
    fixes, and the phase that route_after_decision reads. Kept apart from the
    LLM call so the stop rules can be exercised without a model.

    This has to live in a node — LangGraph throws away state changes made in
    a routing function, which is why the old counters there never advanced
    and the tester <-> decision loop ran forever.

    fix_only: the build nearly passes, so a "change approach" answer is not
    allowed to throw it away — fixes are applied, and with no fixes the build
    finishes with the failures listed as known issues."""
    decision = decision_data.get("decision", "change_approach")
    code_fixes = decision_data.get("code_fixes") or {}
    escalated = state.get("coder_round", 0) >= MAX_REVIEW_ROUNDS

    state["decision_count"] = state.get("decision_count", 0) + 1
    if escalated:
        state["escalation_decision_count"] = state.get("escalation_decision_count", 0) + 1
    elif not (decision == "fix_code" and code_fixes) and not fix_only:
        # Every non-escalated outcome other than "here are the fixes" goes
        # back to the planner, so it counts as a backtrack.
        state["backtrack_count"] = state.get("backtrack_count", 0) + 1

    stop_reason = ""
    if escalated and state["escalation_decision_count"] >= 2:
        stop_reason = "round 5's final decision attempt is done"
    elif state["decision_count"] >= MAX_DECISION_CALLS:
        stop_reason = f"decision has run {state['decision_count']}/{MAX_DECISION_CALLS} times"
    elif state.get("backtrack_count", 0) >= MAX_BACKTRACKS:
        stop_reason = f"{state['backtrack_count']}/{MAX_BACKTRACKS} backtracks used"

    if stop_reason:
        # No more tester, decision or planner passes. Any fixes returned now
        # never get tested, so the errors they target go to known_issues for
        # final_review to report.
        applied = _apply_code_fixes(state["project_path"], code_fixes)
        unresolved = list(state.get("errors", []))
        state["known_issues"] = state.get("known_issues", []) + unresolved
        state["errors"] = []
        state["phase"] = "finalizing"
        console.print(
            f"[yellow]⚠️  Decision loop stopped ({stop_reason}) — applied {applied} unverified "
            f"fix(es), proceeding with {len(unresolved)} known issue(s)[/yellow]"
        )
        blog("decision", f"Loop limit reached ({stop_reason}) — finalizing with "
                         f"{len(unresolved)} known issue(s)")
    elif escalated:
        # Round 4: fix-only, no matter what the model's "decision" field says —
        # we're past the point of backtracking to planner (that would
        # eventually route back to coder, which the escalation ladder forbids).
        # Tester runs next as round 5.
        applied = _apply_code_fixes(state["project_path"], code_fixes)
        if applied:
            console.print(f"[green]🔧 Decision: Applied fixes to {applied} file(s)[/green]")
        else:
            console.print("[yellow]Decision: no code fixes returned — proceeding as-is[/yellow]")
        state["errors"] = []
        state["phase"] = "testing"
    elif fix_only:
        applied = _apply_code_fixes(state["project_path"], code_fixes)
        if applied:
            state["errors"] = []
            state["phase"] = "testing"
            console.print(f"[green]🔧 Decision: Applied fixes to {applied} file(s) — retrying tests[/green]")
        else:
            unresolved = list(state.get("errors", []))
            state["known_issues"] = state.get("known_issues", []) + unresolved
            state["errors"] = []
            state["phase"] = "finalizing"
            console.print("[yellow]Decision: no fix for a nearly-passing build — finalizing with "
                          f"{len(unresolved)} known issue(s)[/yellow]")
    elif decision == "fix_code" and code_fixes:
        _apply_code_fixes(state["project_path"], code_fixes)
        state["errors"] = []
        state["phase"] = "testing"
        console.print("[green]🔧 Decision: Applied code fixes — retrying tests[/green]")
    elif decision in ("change_approach", "change_stack"):
        console.print("[yellow]🔄 Decision: Changing approach — backtracking to Planner[/yellow]")
        new_plan = decision_data.get("new_plan", "")
        if new_plan:
            state["plan"] = new_plan
        state["files_written"] = []
        state["errors"] = []
        state["phase"] = "planning"
    else:
        state["phase"] = "planning"

    if state["phase"] == "planning":
        # Every route back to the planner starts the build fresh: the planner
        # writes a new task list, so a leftover task index would make the
        # coder skip most of it, and old review rounds would escalate early.
        state["task_list"] = []
        state["current_task_index"] = 0
        state["retry_count"] = 0
        state["coder_round"] = 0
        state["review_history"] = []
        state["escalation_decision_count"] = 0


def run_decision(state: AgentState) -> AgentState:
    console.print("\n[bold red]🧠 DECISION AGENT starting (escalation)...[/bold red]")
    state["current_agent"] = "decision"
    state["phase"] = "decision"
    blog("decision", "Escalated — analyzing errors and deciding next action")
    bstate({"current_agent": "decision", "phase": "decision"})

    llm = router.get_llm("reasoning", temperature=0.2)
    memory = AgentMemory(state["project_path"])

    escalated = state.get("coder_round", 0) >= MAX_REVIEW_ROUNDS
    review_history = state.get("review_history", [])

    errors = "\n\n".join(state["errors"][-3:])
    # A build where most tests pass is never re-planned from scratch: one bad
    # test or rounding bug is a fix, not a reason to throw the build away.
    passed, failed = pytest_counts(state["errors"][-1] if state["errors"] else "")
    near_green = bool(passed and failed and failed / (passed + failed) <= 0.25)
    past_failures = memory.get_failures()
    failure_context = "\n".join([f"- {f.get('detail', '')}" for f in past_failures[-5:]])

    # Deep web research on the specific errors
    error_searches = []
    # From the most telling line of each error — error[:80] of pytest output
    # was a row of progress dots, which made a useless search query.
    for query in dict.fromkeys(_error_query(e) for e in state["errors"][-2:]):
        error_searches.append(search_web(f"python {query}", max_results=3))

    # Also search GitHub for similar issues
    gh_result = search_github(f"{state['project_type']} {state['project_name']} error fix")
    error_searches.append(gh_result)

    combined_research = "\n---\n".join(error_searches)
    # It is asked to fix code, so it has to see the code.
    code_view = _existing_code_context(state["project_path"], REVIEW_CODE_BUDGET)

    if escalated or near_green:
        # Rounds 1-3 of reviewer<->coder patching didn't clear the issues.
        # This is a fix-only call: apply the best possible direct fix using
        # the full review history for context. No strategy change, no
        # backtrack to planner — the pipeline must move forward from here.
        review_history_text = "\n\n---\n\n".join(
            f"### Review round {i+1}\n{r}" for i, r in enumerate(review_history)
        )
        situation = ("Three rounds of code review + patching did not resolve the issues — this is the "
                     "final direct-fix attempt." if escalated else
                     f"The build nearly works: {passed} tests pass and {failed} fail. If a failing test "
                     "expects behaviour the design never asked for, fix the test instead of the code.")
        decision_prompt = f"""You are the most senior engineer. {situation} Do NOT propose changing the
approach or tech stack; only fix the code.

PROJECT: {state['project_name']} ({state['project_type']})
ORIGINAL PLAN: {state['plan'][:800]}
ERRORS ENCOUNTERED: {errors}
{"FULL REVIEW HISTORY (all " + str(len(review_history)) + " rounds):" if review_history else ""}
{review_history_text[:4000]}
PAST FAILURES: {failure_context}
RESEARCH FINDINGS: {combined_research[:2000]}

PROJECT CODE (src/):
{code_view}

Think deeply about the pattern across all review rounds — what keeps getting flagged? Fix it properly
this time, not another surface patch.

First a short JSON object:
{{"decision": "fix_code", "reasoning": "What the root cause is and how you're fixing it", "confidence": 0.9}}
Then every file you change, complete, as a FILE block.

{FILE_FORMAT.format(root="the project root — src/app.py, tests/test_app.py")}"""
    else:
        decision_prompt = f"""You are the most senior engineering decision maker. The project is stuck.
You must decide: either fix the current approach or completely change strategy.

PROJECT: {state['project_name']} ({state['project_type']})
ORIGINAL PLAN: {state['plan'][:800]}
ERRORS ENCOUNTERED: {errors}
PAST FAILURES: {failure_context}
RESEARCH FINDINGS: {combined_research[:3000]}

PROJECT CODE (src/):
{code_view}

Think deeply. Make a decision:
1. Can we fix the current code? Prefer this — most failures are fixable bugs.
2. Only if the approach itself can't work, change it and write a new plan.

First a short JSON object:
{{"decision": "fix_code | change_approach | change_stack", "reasoning": "Why", "new_plan": "only for change_approach/change_stack: the complete new implementation plan", "confidence": 0.9}}
If you chose fix_code, follow it with every file you change, complete, as a FILE block.

{FILE_FORMAT.format(root="the project root — src/app.py, tests/test_app.py")}"""

    messages = [HumanMessage(content=decision_prompt)]
    raw = llm.invoke(messages).content.strip()
    decision_data, fixes = parse_meta_json(raw), parse_file_blocks(raw)
    if "decision" not in decision_data and not fixes:
        # One retry before an unreadable reply is allowed to cost the build.
        console.print("[yellow]Decision reply unreadable — asking once more[/yellow]")
        messages += [AIMessage(content=raw), HumanMessage(content=(
            "Your reply could not be read. Answer again: first the JSON object on its own, then each "
            "changed file as a block that starts with `=== FILE: path ===` and ends with `=== END FILE ===`."))]
        raw = llm.invoke(messages).content.strip()
        decision_data, fixes = parse_meta_json(raw), parse_file_blocks(raw)
    if fixes:
        decision_data["code_fixes"] = fixes
    if "decision" not in decision_data:
        # Unreadable summary: use the fixes if it wrote any, rather than
        # throwing the whole build away with a backtrack.
        decision_data["decision"] = "fix_code" if fixes else "change_approach"
        decision_data.setdefault("reasoning", raw[:500])

    decision = decision_data.get("decision", "change_approach")
    reasoning = decision_data.get("reasoning", "")
    confidence = decision_data.get("confidence", 0.7)

    state["decision_log"] = f"Decision: {decision}\nReasoning: {reasoning}\nConfidence: {confidence}"
    write_doc(state["project_path"], "09_decision.md",
              f"# Decision Agent Log\n\n**Decision:** {decision}\n**Confidence:** {confidence}\n\n"
              f"## Reasoning\n{reasoning}")

    log_agent(state, "decision", f"Decision: {decision} (confidence: {confidence})")
    memory.store("decision", state["decision_log"], {"outcome": "decision_made"})
    blog("decision", f"Decision: {decision} (confidence: {confidence:.0%})")
    bstate({"current_agent": "decision", "phase": state.get("phase", "decision")})

    _apply_decision(state, decision_data, fix_only=near_green)
    if state["phase"] == "planning" and decision in ("change_approach", "change_stack"):
        memory.record_outcome("decision", "backtrack", f"Changed approach: {reasoning[:100]}")

    return state
