"""
Visual QA — web projects only, right after the tests pass. Runs the built
app, screenshots it at desktop and phone width in headless Chrome, and has a
vision model judge the render against the design system. If it scores below
UI_PASS_SCORE, a fix round — by a model that sees the screenshots — rewrites
the page files; the tests run again and the fix is rolled back if it broke
them. A second round runs only if the first one raised the score.

Unit tests can't see a page that renders badly: tonight's half-built tip
calculator passed its checks while its card spilled off a phone screen.
Screenshots and findings land in docs/screenshots/ and docs/11_ui_review.md.
"""
import base64
from pathlib import Path

from pipeline import control
from pipeline.state import AgentState, log_agent
from tools import static_checks, webapp
from tools.file_ops import write_doc, write_project_file
from tools.file_blocks import FILE_FORMAT, parse_files
from tools.llm_json import parse_llm_json
from tools.safety import run_safe_command
from tools.pyenv import project_python
from agents.ui_designer import WEB_PROJECT_TYPES
from agents.tester_debugger import head_tail
from agents.coder import _damage
from server.bridge import log as blog, state as bstate
from router import router, NoProviderAvailableError
from rich.console import Console

console = Console()
UI_PASS_SCORE = 7.5
MAX_UI_FIX_ROUNDS = 2
VIEWPORTS = {"desktop": (1280, 900), "mobile": (390, 844)}
_UI_EXTS = (".html", ".css", ".js", ".jinja", ".j2")


def _shoot(project_path: str, tag: str) -> dict:
    """{viewport: png path} for the running app, or {} if it won't start."""
    shots_dir = Path(project_path) / "docs" / "screenshots"
    shots_dir.mkdir(parents=True, exist_ok=True)
    proc, url, err = webapp.start_app(project_path, str(Path(project_path) / "docs" / "app_run.log"))
    if not proc:
        console.print(f"[yellow]  ⚠️ Visual QA: {err[:200]}[/yellow]")
        return {}
    shots = {}
    try:
        for name, (w, h) in VIEWPORTS.items():
            out = str(shots_dir / f"{tag}_{name}.png")
            if webapp.screenshot(url + "/", out, w, h):
                shots[name] = out
    finally:
        webapp.stop(proc)
    return shots


def _critique(state: AgentState, shots: dict) -> dict:
    content = [{"type": "text", "text": f"""You are a senior product designer reviewing a web app's real rendered screens.

PROJECT: {state['project_name']}
DESIGN DOCUMENT (UI section is binding):
{state['design_content'][:2500]}

DESIGN SYSTEM THE CODE SHOULD FOLLOW:
{(state.get('ui_spec') or '')[:2500]}

The screenshots are, in order: {', '.join(f'{n} ({VIEWPORTS[n][0]}px wide)' for n in shots)}.
Judge only what you can actually see: layout and spacing, overflow or cut-off content on the
phone screen, visual hierarchy, typography, alignment, contrast, unstyled browser-default
controls, things that should be hidden but show (e.g. skip links), and how close it is to an
award-winning, premium feel.

Return ONLY JSON:
{{"score": 0-10, "summary": "one sentence", "issues": [{{"severity": "high|medium|low", "issue": "what is wrong, where", "fix": "concrete CSS/HTML change"}}]}}"""}]
    for path in shots.values():
        b64 = base64.b64encode(Path(path).read_bytes()).decode()
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
    resp = router.get_llm("vision").invoke([{"role": "user", "content": content}])
    data = parse_llm_json(resp.content, dict) or {}
    try:
        data["score"] = float(data.get("score", 0))
    except (TypeError, ValueError):
        data["score"] = 0.0
    data.setdefault("issues", [])
    return data


def _ui_files(project_path: str) -> dict:
    src = Path(project_path) / "src"
    return {str(p.relative_to(Path(project_path))): p.read_text(errors="replace")
            for p in sorted(src.rglob("*")) if p.is_file() and p.suffix in _UI_EXTS
            and "__pycache__" not in p.parts}


def _tests_pass(project_path: str) -> bool:
    if not any(Path(project_path, "tests").glob("test_*.py")):
        return True
    ok, _, _ = run_safe_command(f"{project_python(project_path)} -m pytest tests/ -q", cwd=project_path,
                                env={"PYTHONPATH": f"{project_path}/src"}, sandbox_dir=project_path)
    return ok


def _restore(project_path: str, originals: dict) -> None:
    """Put the page files back as they were, removing any the fix added."""
    for path in set(_ui_files(project_path)) - set(originals):
        Path(project_path, path).unlink(missing_ok=True)
    for path, content in originals.items():
        write_project_file(project_path, path, content)


def _fix_round(state: AgentState, review: dict, shots: dict, tests_were_passing: bool = True):
    """One targeted rewrite of the page files. Returns the files as they were
    before it (so a fix that scores worse can be undone), or None if nothing
    was changed. The tests only veto a fix if they passed before it — tests
    that were already failing used to undo every fix."""
    project_path = state["project_path"]
    originals = _ui_files(project_path)
    if not originals:
        return None
    # Whole files: shown only the first 8,000 characters of a 25 KB page, the
    # fixer returned that part as "the complete file" — a portfolio went out
    # as its header, hero and one line of About.
    files_view = "\n\n".join(f"### {p}\n```\n{c}\n```" for p, c in originals.items())
    issues = "\n".join(f"- [{i.get('severity', '?')}] {i.get('issue', '')} → {i.get('fix', '')}"
                       for i in review["issues"] if isinstance(i, dict))
    prompt = f"""You are a senior front-end engineer. The attached screenshots show how this app renders
right now ({', '.join(shots)}). A designer found these problems (score {review['score']}/10):
{issues}

DESIGN SYSTEM TO FOLLOW:
{(state.get('ui_spec') or '')[:3000]}

CURRENT PAGE FILES:
{files_view}

Fix every high and medium problem. Look at the screenshots to check that your CSS selectors really
match the elements you see (e.g. browser-default inputs mean no rule reaches them). Keep every element
id, class and name that the JavaScript or the Python code relies on. You may change the page's
JavaScript where a problem is about behaviour — e.g. an error shown before the user has typed
anything, or an empty value that renders as a blank shape — but never what the app calculates or
sends to the server. Return only the files you change, each complete.

{FILE_FORMAT.format(root="the project folder, e.g. src/static/style.css")}"""
    # A blind fixer patched CSS that never matched the rendered elements, so
    # the score didn't move. Models that read images see the actual result.
    images = [{"type": "image_url", "image_url": {"url": "data:image/png;base64,"
               + base64.b64encode(Path(p).read_bytes()).decode()}} for p in shots.values()]
    try:
        resp = router.get_llm("ui_fix", temperature=0.2).invoke(
            [{"role": "user", "content": [{"type": "text", "text": prompt}, *images]}])
    except NoProviderAvailableError:
        resp = router.get_llm("coding", temperature=0.2).invoke(prompt)
    fixes = _undamaging(project_path, parse_files(resp.content))
    if not fixes:
        console.print("[yellow]  Visual QA: the fix reply had no usable page files[/yellow]")
        return None
    page_before = _page_problems(project_path)
    for path, content in fixes.items():
        write_project_file(project_path, path, content)
    kept = not tests_were_passing or _tests_pass(project_path)
    if not kept:
        # The fix broke tests that passed before. Undoing it at once threw
        # away every design fix (a 2/10 page stayed at 2/10), so the fixer
        # gets one more try with the failing tests in front of it.
        console.print("[yellow]  🔁 UI fix broke the tests — one more try with the failing tests shown[/yellow]")
        kept = _repair_for_tests(project_path) and _tests_pass(project_path)
        if not kept:
            console.print("[yellow]  ↩️ UI fix still breaks the tests — restoring the previous page files[/yellow]")
    if kept:
        # A nicer-looking page with its content gone is not a fix: a round
        # once emptied four sections and still raised the design score.
        broken = _page_problems(project_path) - page_before
        if broken:
            console.print(f"[yellow]  ↩️ UI fix broke the page ({'; '.join(sorted(broken)[:2])}) — "
                          f"restoring the previous page files[/yellow]")
            kept = False
    if kept:
        return originals
    _restore(project_path, originals)
    return None


def _undamaging(project_path: str, files) -> dict:
    """The reply's page files, minus any that would lose content (sections,
    functions, most of their lines). A design score only sees the first
    screen, so it rose while everything below it was being deleted."""
    kept = {}
    for path, content in (files or {}).items():
        if not path.endswith(_UI_EXTS):
            continue
        why = _damage(project_path, path, content)
        if why:
            console.print(f"[yellow]  ↩️ Visual QA: kept {path} as it was — the fix {why}[/yellow]")
            continue
        kept[path] = content
    return kept


def _page_problems(project_path: str) -> set:
    """The page's certain problems in a real browser (empty sections, JS
    errors, failed loads), to tell whether a fix made the page worse."""
    return {f"{f['code']}: {f['message']}" for f in static_checks.page_check(project_path) if f["severity"] == "HIGH"}


def _failing_tests(project_path: str) -> str:
    _, out, err = run_safe_command(f"{project_python(project_path)} -m pytest tests/ -q", cwd=project_path,
                                   env={"PYTHONPATH": f"{project_path}/src"}, sandbox_dir=project_path)
    return f"{err.strip()}\n{out.strip()}".strip()


def _repair_for_tests(project_path: str) -> bool:
    """Ask for the page files again so the tests pass, keeping the design
    fixes. The tests themselves are never handed over to be changed."""
    current = _ui_files(project_path)
    files_view = "\n\n".join(f"### {p}\n```\n{c}\n```" for p, c in current.items())
    prompt = f"""Your design fix made tests fail that passed before it:
{head_tail(_failing_tests(project_path), 3000)}

CURRENT PAGE FILES (with your fix):
{files_view}

Keep the design improvements, but change the page files so these tests pass again — usually by
putting back an element id, class, text or structure a test looks for. Don't change the tests.
Return only the files you change, each complete.

{FILE_FORMAT.format(root="the project folder, e.g. src/static/style.css")}"""
    resp = router.get_llm("coding", temperature=0.2).invoke(prompt)
    fixes = _undamaging(project_path, parse_files(resp.content))
    for path, content in fixes.items():
        write_project_file(project_path, path, content)
    return bool(fixes)


def _report(project_path: str, rounds: list) -> None:
    lines = ["# Visual QA", ""]
    for r in rounds:
        lines += [f"## {r['label']} — score {r['review']['score']:.1f}/10", "", r["review"].get("summary", ""), ""]
        lines += [f"- [{i.get('severity', '?').upper()}] {i.get('issue', '')} — fix: {i.get('fix', '')}"
                  for i in r["review"]["issues"] if isinstance(i, dict)]
        lines += ["", *[f"![{n}](screenshots/{Path(p).name})" for n, p in r["shots"].items()], ""]
    write_doc(project_path, "11_ui_review.md", "\n".join(lines))


def run_ui_review(state: AgentState) -> AgentState:
    if state.get("project_type") not in WEB_PROJECT_TYPES:
        return state
    if not webapp.browser():
        console.print("[yellow]Visual QA skipped: no Chrome/Edge installed[/yellow]")
        return state

    console.print("\n[bold magenta]👁️  VISUAL QA starting...[/bold magenta]")
    state["current_agent"] = "ui_review"
    blog("ui_review", "Starting — screenshotting the app and reviewing the design")
    bstate({"current_agent": "ui_review", "phase": "ui_review"})
    project_path = state["project_path"]

    shots = _shoot(project_path, "before")
    if not shots:
        state["known_issues"] = state.get("known_issues", []) + ["Visual QA: the web app didn't start for screenshots"]
        return state
    review = _critique(state, shots)
    rounds = [{"label": "First render", "review": review, "shots": shots}]
    console.print(f"[cyan]  🎨 Design score {review['score']:.1f}/10 — {len(review['issues'])} issue(s)[/cyan]")
    blog("ui_review", f"Design score {review['score']:.1f}/10 — {len(review['issues'])} issue(s)")

    fix_rounds = 0
    tests_were_passing = None
    while review["score"] < UI_PASS_SCORE and review["issues"] and fix_rounds < MAX_UI_FIX_ROUNDS:
        fix_rounds += 1
        blog("ui_review", f"Below {UI_PASS_SCORE} — fixing the page (round {fix_rounds})")
        if tests_were_passing is None:
            tests_were_passing = _tests_pass(project_path)
        before = _fix_round(state, review, shots, tests_were_passing)
        if before is None:
            break
        best_review, best_shots = review, shots
        previous = review["score"]
        shots = _shoot(project_path, f"after{fix_rounds}")
        if not shots:
            # The fixed page no longer renders — that's worse than any score.
            console.print("[yellow]  ↩️ The app didn't render after the UI fix — restoring the previous page[/yellow]")
            _restore(project_path, before)
            review, shots = best_review, best_shots
            break
        review = _critique(state, shots)
        label = f"After fix round {fix_rounds}"
        console.print(f"[cyan]  🎨 Design score after fixes {review['score']:.1f}/10 (was {previous:.1f})[/cyan]")
        blog("ui_review", f"After fixes: {review['score']:.1f}/10 (was {previous:.1f})")
        if review["score"] < previous:
            # Keep the best version: a round once took a page from 6.0 to 4.0
            # and the worse page was what got shipped.
            console.print("[yellow]  ↩️ The fix scored lower — restoring the previous page files[/yellow]")
            blog("ui_review", f"Fix scored lower ({review['score']:.1f}) — kept the {previous:.1f} version")
            _restore(project_path, before)
            rounds.append({"label": label + " — scored lower, rolled back", "review": review, "shots": shots})
            review, shots = best_review, best_shots
            break
        rounds.append({"label": label, "review": review, "shots": shots})
        # A level score doesn't end the loop: scores move in whole steps, a
        # round can fix real problems without moving one, and a round that
        # scores lower is rolled back anyway.

    _report(project_path, rounds)
    if review["score"] < UI_PASS_SCORE:
        state["known_issues"] = state.get("known_issues", []) + [
            f"Visual QA {review['score']:.1f}/10: " + "; ".join(
                i.get("issue", "") for i in review["issues"][:3] if isinstance(i, dict))]
    state["ui_review_score"] = review["score"]
    control.write_status(project_path, ui_score=round(review["score"], 1))
    log_agent(state, "ui_review", f"Design score {review['score']:.1f}/10 after {fix_rounds} fix round(s)")
    console.print(f"[green]✅ Visual QA done — {review['score']:.1f}/10[/green]")
    return state
