from typing import TypedDict, List, Optional, Any
from datetime import datetime

from pipeline.control import PROJECTS_DIR

# Coder<->reviewer escalation ladder (see pipeline/graph.py): each round the
# reviewer finds issues and the coder patches just the broken lines. Up to 20
# rounds (user decision 2026-09-30: 3 cut builds off while they were still
# improving), but agents/reviewer.py ends the loop early once two rounds in a
# row haven't reduced the high-severity issues. After the last patch round the
# decision agent fixes directly with the full review history; the tester runs,
# decision gets one final fix attempt, then the pipeline always proceeds.
MAX_REVIEW_ROUNDS = 20
# Caps for the tester <-> decision loop, enforced in agents/decision.py. The
# decision node itself does the counting: LangGraph throws away state changes
# made inside a routing function, so counters kept there never advance.
MAX_DECISION_CALLS = 3
MAX_BACKTRACKS = 3


class AgentState(TypedDict):
    # Project info
    project_name: str
    project_type: str
    project_path: str
    design_file: str
    design_content: str

    # Planning outputs
    thinking: str
    plan: str
    architecture: str
    flowchart: str
    task_list: List[dict]
    current_task_index: int
    # Design system from agents/ui_designer.py (web projects only, else "")
    ui_spec: str
    # Last vision-model design score from agents/ui_review.py (0 = not reviewed)
    ui_review_score: float

    # Research outputs (the three _prefixed fields are scratch space written
    # by parallel research_web/research_github/research_libraries branches
    # and consumed only by research_merge — see pipeline/graph.py)
    research_notes: str
    libraries: List[str]
    _web_results: str
    _github_results: str
    _library_candidates: List[str]
    # Critic gate + research reflection (agents/critic.py, agents/reflect.py).
    # Undeclared, these were dropped between nodes, so a critic FAIL was
    # silently ignored and the re-research cap never counted.
    _research_passes: int
    _critic_pass: bool
    _critic_confidence: float
    _research_reflect_score: float

    # Security/optimizer scratch space — both run read-only in parallel after
    # coder, writing findings/suggestions here instead of to disk; a merge
    # node applies them afterward (avoids two branches racing to write the
    # same file). See agents/security.py, agents/optimizer.py, agents/code_merge.py.
    _security_findings: List[dict]
    _security_fixed_files: dict
    _optimization_notes: List[str]
    _optimized_files: dict

    # doc_writer runs twice — a draft in parallel with test_writer/tester,
    # a final pass in parallel with final_review. See agents/doc_writer.py.
    _readme_draft: str

    # Code outputs
    files_written: List[str]
    current_file: str

    # Execution state
    current_agent: str
    agent_logs: List[dict]
    errors: List[str]
    retry_count: int
    decision_log: str

    # Coder<->reviewer escalation ladder
    coder_round: int
    review_history: List[str]
    escalation_decision_count: int
    known_issues: List[str]

    # Tester <-> decision loop caps (MAX_DECISION_CALLS / MAX_BACKTRACKS).
    # Per-project state, not module globals, so --watch mode starts every
    # new design at zero.
    decision_count: int
    backtrack_count: int

    # Health
    cpu_percent: float
    ram_percent: float
    is_throttled: bool

    # Status
    phase: str
    is_complete: bool
    github_ready: bool
    github_approved: bool


def new_state(design_file: str, design_content: str) -> AgentState:
    project_name = design_file.replace(".md", "").replace(" ", "_").lower()
    project_path = str(PROJECTS_DIR / project_name)
    return AgentState(
        project_name=project_name,
        project_type="",
        project_path=project_path,
        design_file=design_file,
        design_content=design_content,
        thinking="",
        plan="",
        architecture="",
        flowchart="",
        task_list=[],
        current_task_index=0,
        ui_spec="",
        ui_review_score=0.0,
        research_notes="",
        libraries=[],
        _web_results="",
        _github_results="",
        _library_candidates=[],
        _research_passes=0,
        _critic_pass=True,
        _critic_confidence=0.0,
        _research_reflect_score=0.0,
        _security_findings=[],
        _security_fixed_files={},
        _optimization_notes=[],
        _optimized_files={},
        _readme_draft="",
        files_written=[],
        current_file="",
        current_agent="",
        agent_logs=[],
        errors=[],
        retry_count=0,
        decision_log="",
        coder_round=0,
        review_history=[],
        escalation_decision_count=0,
        known_issues=[],
        decision_count=0,
        backtrack_count=0,
        cpu_percent=0.0,
        ram_percent=0.0,
        is_throttled=False,
        phase="intake",
        is_complete=False,
        github_ready=False,
        github_approved=False,
    )


def log_agent(state: AgentState, agent: str, message: str) -> None:
    state["agent_logs"].append({
        "agent": agent,
        "message": message,
        "time": datetime.now().isoformat(),
    })
