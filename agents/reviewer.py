from langchain_core.messages import HumanMessage
from pipeline.state import AgentState, log_agent, MAX_REVIEW_ROUNDS
from tools.file_ops import write_doc, list_files
from agents.ui_designer import WEB_PROJECT_TYPES
from agents.coder import _existing_code_context, REVIEW_CODE_BUDGET
from server.bridge import log as blog, state as bstate
from tools import static_checks
from tools.llm_json import parse_llm_json
from tools.pyenv import install_requirements
from router import router
from rich.console import Console

console = Console()

# What the coder should do about each kind of tool finding (its patch task).
TOOL_FIX_HINTS = {
    "F821": "Define or import the name before it is used.",
    "name-defined": "Define or import the name before it is used.",
    "import-not-found": "Import a module that exists in the project, or add the package to requirements.txt.",
    "attr-defined": "Import a name the module really defines, or add it to that module.",
    "invalid-syntax": "Fix the syntax error.",
    "css-needs-build": "Nothing compiles Tailwind here: stop linking this file and load Tailwind with the Play CDN "
                       "(<script src=\"https://cdn.tailwindcss.com\"></script> in <head>), or rewrite it as plain CSS.",
    "tailwind-classes": "Load Tailwind with its Play CDN in the page's <head>: "
                        "<script src=\"https://cdn.tailwindcss.com\"></script> plus an inline tailwind.config script "
                        "with the design tokens. Remove any link to a compiled file such as dist/output.css.",
    "missing-file": "Create the linked file or fix the path.",
    "empty-section": "Make the section show its content: if JavaScript fills it, the page must load that script "
                     "and its data (a <script src> for each file, data before the code that uses it) and the "
                     "script must actually render into this section; otherwise write the content into the HTML.",
    "js-error": "Fix the JavaScript error so the page's scripts run.",
    "failed-load": "Fix the path, or create the file the page asks for.",
    "app-wont-start": "Fix the error that stops the app from starting.",
    "no-entry-page": "Write the site as plain files a browser opens: src/index.html with its css/ and js/ "
                     "(or a Flask/FastAPI src/app.py). Nothing here builds .astro/.jsx/.vue pages.",
    "needs-build": "Move this into plain HTML/CSS/JS under src/ (index.html, css/, js/): nothing here compiles "
                   "framework files, so browsers never see them.",
    "unused-script": "Load the script from the page that needs it (<script src>), or delete it if it isn't needed.",
    "csp-blocked": "Remove the Content-Security-Policy <meta> tag: it blocks what the page itself needs "
                   "(the Tailwind Play CDN runs an inline config script and injects inline styles).",
    "flat-headings": "Size the headings with classes on the tags themselves, e.g. <h1 class=\"text-4xl md:text-5xl "
                     "font-bold\"> and <h2 class=\"text-3xl font-semibold\">. With the Tailwind Play CDN, sizes in "
                     "a .css file never apply: its reset (h1-h6 { font-size: inherit }) loads after your stylesheet.",
    "invisible-text": "Give this text a colour that stands out from its background. If its classes are custom "
                      "theme names (bg-accent, text-text-muted, …), define them — with the Tailwind Play CDN, in "
                      "an inline tailwind.config script in <head> — or use standard Tailwind colours. "
                      "\"(dark mode)\" means only the dark theme is broken: the text colour switches but the "
                      "background is a fixed colour (or the other way round). Take both from the same CSS "
                      "variables, e.g. tailwind.config colors: { surface: 'var(--surface)', text: 'var(--text-main)' }.",
}


STALL_ROUNDS = 3


def _stalled(history: list, highs_now: int) -> bool:
    """True when the last STALL_ROUNDS rounds, this one included, haven't
    beaten the fewest high-severity issues seen before them (each round's
    report lists them as "- [HIGH]"). Two rounds stopped a build at round 3
    while the page check was still finding new problems to fix."""
    counts = [text.count("- [HIGH]") for text in history] + [highs_now]
    if len(counts) <= STALL_ROUNDS:
        return False
    return min(counts[-STALL_ROUNDS:]) >= min(counts[:-STALL_ROUNDS])


def run_reviewer(state: AgentState) -> AgentState:
    console.print("\n[bold blue]🔎 REVIEWER AGENT starting...[/bold blue]")
    state["current_agent"] = "reviewer"
    state["phase"] = "review"
    blog("reviewer", "Starting — reviewing code quality")
    bstate({"current_agent": "reviewer", "phase": "review"})

    llm = router.get_llm("review")

    # Same whole-project view the coder gets (full files up to a budget, then
    # signatures). The old view — first 8 files, 1500 chars each — hid most
    # of the project, so the reviewer flagged existing files as "missing" and
    # burned escalation rounds on non-issues.
    all_files = sorted(f for f in list_files(f"{state['project_path']}/src") if "__pycache__" not in f)
    full_code = _existing_code_context(state["project_path"], REVIEW_CODE_BUDGET)

    if not full_code:
        console.print("[yellow]No source files to review[/yellow]")
        state["phase"] = "testing"
        return state

    # Certain bugs first, found for free: syntax errors, undefined names,
    # imports that can't work, stylesheets browsers can't read and links to
    # missing files. They become high-severity issues — facts, not opinions —
    # and the model is told not to repeat them. Requirements are installed
    # first so a declared dependency isn't reported as missing.
    install_requirements(state["project_path"])
    checks = static_checks.code_checks(state["project_path"])
    if state.get("project_type") in WEB_PROJECT_TYPES:
        # The built page in a real browser: JS errors, failed loads, empty sections.
        checks["page"] = static_checks.page_check(state["project_path"])
    static_checks.write_report(state["project_path"], checks)
    tool_findings = [f for tool in ("ruff", "mypy", "web", "page") for f in checks.get(tool, [])]
    # HIGH findings are certain bugs; the rest (e.g. a script nothing loads)
    # are hints that shouldn't send the coder round on their own.
    tool_issues = [{"file": f["file"], "line": f["line"], "severity": "high" if f["severity"] == "HIGH" else "medium",
                    "issue": f"{f['tool']} {f['code']}: {f['message']}",
                    "fix": TOOL_FIX_HINTS.get(f["code"], "Fix what the tool reports.")} for f in tool_findings]
    tools_block = ""
    if tool_findings:
        console.print(f"[yellow]  🔧 The free checks found {len(tool_findings)} issue(s)[/yellow]")
        blog("reviewer", f"Static checks: {len(tool_findings)} issue(s) (ruff/mypy/web/page)")
        tools_block = ("\nFOUND BY TOOLS (certain, already reported — don't repeat these):\n"
                       + static_checks.as_lines(tool_findings[:40]) + "\n")

    ui_review = ""
    if state.get("project_type") in WEB_PROJECT_TYPES:
        # Medium unless actually broken: "high" routes back to the coder, and
        # taste-level UI notes shouldn't burn escalation rounds.
        ui_review = """6. UI/UX — does the page follow the project design system (CSS variables, type scale,
   dark mode), work on mobile widths, give feedback for every action (inline errors, toasts,
   empty and loading states) and stay accessible (labels, focus rings, keyboard use)?
   Report UI/UX problems as "medium" unless the page is broken or unusable."""

    review_prompt = f"""You are a senior code reviewer. Review this codebase thoroughly.

PROJECT: {state['project_name']} ({state['project_type']})
ARCHITECTURE: {state['architecture'][:600]}

DESIGN DOCUMENT (what was asked for — source of truth):
{state['design_content'][:3500]}

ALL FILES IN src/: {', '.join(all_files)}
Only call a file missing if it is not in this list.

CODEBASE:
{full_code}
{tools_block}
Review for:
1. Completeness — go through every page section and feature the DESIGN DOCUMENT lists. Each one
   that is missing, or present only as a heading or placeholder without its real content, is a
   "high" issue naming the file that should contain it. Content must be the design's own: the
   names, roles, services, projects, numbers, skills and contact details it gives (placeholder
   values included). Invented or generic content in their place is a "high" issue too.
2. Correctness — does it match the architecture?
3. Security — any obvious vulnerabilities?
4. Error handling — are errors handled properly?
5. Code quality — naming, structure, readability
{ui_review}

Return JSON:
{{
  "overall_score": 0.85,
  "issues": [
    {{"file": "main.py", "line": 10, "severity": "high|medium|low", "issue": "...", "fix": "..."}}
  ],
  "missing_files": ["file1.py"],
  "verdict": "pass|fail",
  "summary": "..."
}}
Return ONLY valid JSON."""

    resp = llm.invoke([HumanMessage(content=review_prompt)])
    data = parse_llm_json(resp.content, dict)

    verdict = "pass"
    issues = []
    summary = ""
    score = "—"

    if data:
        try:
            verdict = data.get("verdict", "pass")
            issues = [i for i in (data.get("issues") or []) if isinstance(i, dict)]
            summary = data.get("summary", "")
            score = data.get("overall_score", 0.8)
        except Exception as e:
            console.print(f"[yellow]Review parse warning: {e}[/yellow]")

    if tool_issues:
        # A HIGH tool finding is a certain bug, so the review fails whatever the
        # model said; MAX_REVIEW_ROUNDS still caps the reviewer <-> coder loop.
        issues = tool_issues + issues
        if any(i["severity"] == "high" for i in tool_issues):
            verdict = "fail"

    review_text = (
        f"# Code Review — round {state.get('coder_round', 0) + 1}\n\n"
        f"**Score:** {score}\n**Verdict:** {verdict}\n\n"
        f"## Summary\n{summary}\n\n"
        f"## Issues\n" + "\n".join(f"- [{str(i.get('severity', '?')).upper()}] {i.get('file', '?')}: "
                                   f"{i.get('issue', '')}" for i in issues)
    )
    write_doc(state["project_path"], "07_review.md", review_text)

    high_issues = [i for i in issues if i.get("severity") == "high"]

    if high_issues and verdict == "fail":
        stalled = _stalled(state.get("review_history", []), len(high_issues))
        state["coder_round"] = state.get("coder_round", 0) + 1
        state["review_history"] = state.get("review_history", []) + [review_text or summary]

        console.print(
            f"[red]⚠️  Review round {state['coder_round']}/{MAX_REVIEW_ROUNDS} — "
            f"{len(high_issues)} high-severity issues — routing to coder to patch[/red]"
        )
        if stalled and state["coder_round"] < MAX_REVIEW_ROUNDS:
            # Up to 20 rounds only pays while they help: past this point more
            # rounds re-argue the same points (or churn the code) on free quota.
            # One last patch round, then the ladder's decision step.
            console.print("[yellow]  ⏹ No fewer high-severity issues for three rounds — last patch round, "
                          "then the decision agent[/yellow]")
            blog("reviewer", "No progress for three rounds — last patch round, then the decision agent")
            state["coder_round"] = MAX_REVIEW_ROUNDS
        # Patch tasks — coder.py treats type="fix" as a minimal-edit patch on
        # the named file, not a full rewrite (see coder.py's patch-mode prompt).
        fix_tasks = []
        for iss in high_issues:
            fix_tasks.append({
                "id": 9000 + len(fix_tasks),
                "name": f"Fix: {iss['issue'][:50]}",
                "description": f"In {iss['file']}: {iss['issue']}. Fix: {iss.get('fix', 'Fix the issue')}",
                "file": iss.get("file", ""),
                "type": "fix",
            })
        state["task_list"].extend(fix_tasks)
        state["current_task_index"] = len(state["task_list"]) - len(fix_tasks)
        state["phase"] = "coding"
    else:
        console.print(f"[green]✅ Review passed — {len(issues)} minor issues noted[/green]")
        state["phase"] = "testing"

    log_agent(state, "reviewer", f"Review: {verdict}. Issues: {len(issues)}")
    blog("reviewer", f"Done — verdict: {verdict}, {len(issues)} issues found")
    bstate({"current_agent": "reviewer", "phase": state["phase"]})
    return state
