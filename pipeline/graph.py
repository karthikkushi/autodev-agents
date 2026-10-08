import functools
from langgraph.graph import StateGraph, END
from pipeline import control
from pipeline.state import AgentState
from agents.intake import run_intake
from agents.planner import run_planner
from agents.researcher import run_research_web, run_research_github, run_research_libraries, run_research_merge
from agents.reflect import reflect_after_intake, reflect_after_planner, reflect_after_researcher
from agents.critic import run_critic_gate, route_critic
from agents.environment import run_environment
from agents.ui_designer import run_ui_designer
from agents.coder import run_coder
from agents.security import run_security
from agents.reviewer import run_reviewer
from agents.optimizer import run_optimizer
from agents.code_merge import run_code_merge
from agents.test_writer import run_test_writer
from agents.tester_debugger import run_tester_debugger
from agents.ui_review import run_ui_review
from agents.decision import run_decision
from agents.doc_writer import run_doc_writer_draft, run_doc_writer_final
from agents.final_review import run_final_review
from agents.git_github import run_git_commit, wait_for_github_approval, run_github_push
from tools.safety import get_health
from server.bridge import state as bstate
from rich.console import Console

console = Console()

# Coder<->reviewer escalation ladder — old pipeline had no cap on this loop
# and once ran it 26 times in a row. Now: rounds 1-3 patch via coder+reviewer
# (state["coder_round"], tracked in reviewer.py / checked in coder.py).
# Round 4 = decision fixes directly (route_after_coder below). Round 5 =
# tester, then at most one more decision attempt (escalation_decision_count),
# then the pipeline always proceeds. The tester<->decision loop outside the
# ladder is capped too (decision_count / backtrack_count). All of that
# counting lives in agents/decision.py — routing functions here only read.
#
# Parallel branches (see individual agent files for why each return value is
# shaped the way it is — LangGraph raises InvalidUpdateError if two nodes in
# the same step both write the same state key, including implicitly via a
# full `return state`, so every parallel node here returns either {} or a
# small partial dict of keys nothing else in its step touches):
#   intake        -> {planner, reflect_intake}       -> reflect_planner
#   reflect_planner (or a critic_gate retry)          -> research_dispatch
#   research_dispatch -> {research_web, research_github, research_libraries} -> research_merge
#   coder         -> code_review_dispatch -> {security, optimizer} -> code_merge -> reviewer
#   reviewer(pass)-> post_review_dispatch -> {test_writer, doc_writer_draft}  -> post_review_join -> tester
#   tester/decision (terminal) -> pre_final_dispatch -> {doc_writer_final, final_review} -> pre_git_commit_join -> git_commit


def _step(name: str, fn):
    """Wraps every node: wait first if the user paused or the internet is
    down, then record where the pipeline is so the dashboard can show it
    (and show where a restarted worker will pick up again)."""
    @functools.wraps(fn)
    def run(state):
        control.wait_if_paused(name)
        progress = {"step": name, "task_index": state.get("current_task_index", 0),
                    "task_total": len(state.get("task_list") or [])}
        control.write_status(state["project_path"], status="building", phase=state.get("phase", ""), **progress)
        bstate({**progress, "project_name": state.get("project_name", "")})
        return fn(state)
    return run


def health_check(state: AgentState) -> AgentState:
    # One-shot snapshot for the dashboard — never blocks. Every agent call is
    # a cloud API request, not local inference, so there's no local compute
    # load to wait out.
    health = get_health()
    state["cpu_percent"] = health["cpu_percent"]
    state["ram_percent"] = health["ram_percent"]
    console.print(
        f"[dim]💻 CPU {health['cpu_percent']}% | RAM {health['ram_percent']}% "
        f"| RAM free {health['ram_available_gb']}GB[/dim]"
    )
    return state


# ── trivial fan-out / join nodes — each runs alone (no concurrent sibling at
# its own point in the graph), so a normal full-state return is safe here.

def research_dispatch(state: AgentState) -> AgentState:
    return state


def code_review_dispatch(state: AgentState) -> AgentState:
    return state


def post_review_dispatch(state: AgentState) -> AgentState:
    return state


def post_review_join(state: AgentState) -> AgentState:
    console.print("[dim]✓ test_writer + doc_writer draft both done — starting tester[/dim]")
    return state


def pre_final_dispatch(state: AgentState) -> AgentState:
    return state


def pre_git_commit_join(state: AgentState) -> AgentState:
    console.print("[dim]✓ doc_writer final + final_review both done — committing[/dim]")
    state["phase"] = "git_commit"
    return state


def route_after_coder(state: AgentState) -> str:
    # coder.py writes one task per step and leaves phase="coding" while tasks
    # remain, so a checkpoint lands after every task — a restart mid-coding
    # resumes at the next task instead of redoing them all.
    if state.get("phase") == "coding":
        return "coder"
    # coder.py sets phase="decision_escalation" itself when it just finished
    # round 3's patch — skip security+reviewer entirely for round 4.
    if state.get("phase") == "decision_escalation":
        return "decision"
    return "code_review_dispatch"


def route_after_review(state: AgentState) -> str:
    # Round counting lives in reviewer.py (state["coder_round"]) and the
    # round-3→decision jump is decided by coder.py/route_after_coder above —
    # this just reads the pass/fail phase reviewer already set.
    if state["phase"] == "coding":
        return "coder"
    return "post_review_dispatch"


def route_after_test(state: AgentState) -> str:
    if state["phase"] == "decision":
        return "decision"
    return "ui_review"  # visual QA for web projects; a pass-through for everything else


def route_after_decision(state: AgentState) -> str:
    # Read-only on purpose: LangGraph discards any state change made in a
    # routing function. run_decision already did the counting and set phase
    # to "finalizing" if a loop cap was hit.
    phase = state.get("phase")
    if phase == "finalizing":
        # Still get the visual QA pass: a leftover failing unit test shouldn't
        # cancel the UI review (it copes if the app won't start at all).
        return "ui_review"
    if phase == "planning":
        return "planner"
    if phase == "testing":
        return "tester"
    return "pre_final_dispatch"


def route_after_commit(state: AgentState) -> str:
    # Always go to wait_approval first
    return "wait_approval"


def route_after_approval(state: AgentState) -> str:
    if state.get("github_approved"):
        return "github_push"
    return END


def build_graph(checkpointer=None) -> StateGraph:
    """checkpointer: a LangGraph saver (main.py passes SqliteSaver) — each
    step's state is saved so the run survives restarts and power cuts."""
    g = StateGraph(AgentState)

    # All nodes — each wrapped by _step (pause/offline check + status file)
    nodes = {
        "health_check": health_check,
        "intake": run_intake,
        "reflect_intake": reflect_after_intake,
        "planner": run_planner,
        "reflect_planner": reflect_after_planner,
        "research_dispatch": research_dispatch,
        "research_web": run_research_web,
        "research_github": run_research_github,
        "research_libraries": run_research_libraries,
        "research_merge": run_research_merge,
        "reflect_researcher": reflect_after_researcher,
        "critic_gate": run_critic_gate,
        "environment": run_environment,
        "ui_designer": run_ui_designer,
        "coder": run_coder,
        "code_review_dispatch": code_review_dispatch,
        "security": run_security,
        "optimizer": run_optimizer,
        "code_merge": run_code_merge,
        "reviewer": run_reviewer,
        "post_review_dispatch": post_review_dispatch,
        "test_writer": run_test_writer,
        "doc_writer_draft": run_doc_writer_draft,
        "post_review_join": post_review_join,
        "tester": run_tester_debugger,
        "decision": run_decision,
        "ui_review": run_ui_review,
        "pre_final_dispatch": pre_final_dispatch,
        "doc_writer_final": run_doc_writer_final,
        "final_review": run_final_review,
        "pre_git_commit_join": pre_git_commit_join,
        "git_commit": run_git_commit,
        "wait_approval": wait_for_github_approval,
        "github_push": run_github_push,
    }
    for name, fn in nodes.items():
        g.add_node(name, _step(name, fn))

    # Entry
    g.set_entry_point("health_check")
    g.add_edge("health_check", "intake")

    # intake -> {planner, reflect_intake} in parallel, join at reflect_planner
    g.add_edge("intake", "planner")
    g.add_edge("intake", "reflect_intake")
    g.add_edge("planner", "reflect_planner")
    g.add_edge("reflect_intake", "reflect_planner")

    g.add_edge("reflect_planner", "research_dispatch")

    # research -> 3-way parallel split, join at research_merge
    g.add_edge("research_dispatch", "research_web")
    g.add_edge("research_dispatch", "research_github")
    g.add_edge("research_dispatch", "research_libraries")
    g.add_edge("research_web", "research_merge")
    g.add_edge("research_github", "research_merge")
    g.add_edge("research_libraries", "research_merge")

    g.add_edge("research_merge", "reflect_researcher")
    g.add_edge("reflect_researcher", "critic_gate")

    # Critic gate — pass → environment, fail → re-run the research fan-out
    g.add_conditional_edges("critic_gate", route_critic, {
        "environment": "environment",
        "researcher": "research_dispatch",
    })

    # ui_designer is a pass-through for non-web projects
    g.add_edge("environment", "ui_designer")
    g.add_edge("ui_designer", "coder")

    # Coder — normal write/patch → security+optimizer in parallel. Just
    # finished round 3's patch with reviewer still failing → skip straight
    # to decision (round 4).
    g.add_conditional_edges("coder", route_after_coder, {
        "coder": "coder",
        "code_review_dispatch": "code_review_dispatch",
        "decision": "decision",
    })
    g.add_edge("code_review_dispatch", "security")
    g.add_edge("code_review_dispatch", "optimizer")
    g.add_edge("security", "code_merge")
    g.add_edge("optimizer", "code_merge")
    g.add_edge("code_merge", "reviewer")

    # Reviewer — pass → test_writer + doc_writer draft in parallel, fail →
    # back to coder (round cap enforced by coder.py + route_after_coder)
    g.add_conditional_edges("reviewer", route_after_review, {
        "coder": "coder",
        "post_review_dispatch": "post_review_dispatch",
    })
    g.add_edge("post_review_dispatch", "test_writer")
    g.add_edge("post_review_dispatch", "doc_writer_draft")
    g.add_edge("test_writer", "post_review_join")
    g.add_edge("doc_writer_draft", "post_review_join")
    g.add_edge("post_review_join", "tester")

    # Tester — pass → doc_writer final + final_review in parallel, fail → decision
    g.add_conditional_edges("tester", route_after_test, {
        "decision": "decision",
        "ui_review": "ui_review",
    })
    g.add_edge("ui_review", "pre_final_dispatch")

    # Decision — can backtrack to planner, retry tests, or proceed to the
    # final parallel pair
    g.add_conditional_edges("decision", route_after_decision, {
        "planner": "planner",
        "tester": "tester",
        "ui_review": "ui_review",
        "pre_final_dispatch": "pre_final_dispatch",
    })

    # doc_writer final pass + final_review in parallel, join before git commit
    g.add_edge("pre_final_dispatch", "doc_writer_final")
    g.add_edge("pre_final_dispatch", "final_review")
    g.add_edge("doc_writer_final", "pre_git_commit_join")
    g.add_edge("final_review", "pre_git_commit_join")

    g.add_edge("pre_git_commit_join", "git_commit")
    g.add_edge("git_commit", "wait_approval")
    g.add_conditional_edges("wait_approval", route_after_approval, {
        "github_push": "github_push",
        END: END,
    })
    g.add_edge("github_push", END)

    return g.compile(checkpointer=checkpointer)
