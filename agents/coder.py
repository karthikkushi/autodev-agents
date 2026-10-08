from langchain_core.messages import HumanMessage
from pipeline.state import AgentState, log_agent, MAX_REVIEW_ROUNDS
from tools.file_ops import write_project_file, list_files, read_file, project_file_target
from tools.memory import AgentMemory
from tools.file_blocks import FILE_FORMAT, EDIT_FORMAT, parse_files, parse_edits, apply_edits, was_cut_off, is_stub
from tools import static_checks
from agents.ui_designer import WEB_PROJECT_TYPES
from server.bridge import log as blog, state as bstate
from router import router
from rich.console import Console
import os, py_compile, re, shutil, subprocess, tempfile

console = Console()

# At 12,000 a finished 12.8 KB portfolio page (and every file after it) showed
# up as "(no top-level definitions)": the reviewer called all its sections
# missing for two rounds and the fixes cut it back to a skeleton. Agents that
# judge or rewrite whole files (reviewer, security, optimizer, decision) get
# the larger view.
EXISTING_CODE_BUDGET = 32000
REVIEW_CODE_BUDGET = 48000
_CODE_EXTS = {".py", ".js", ".jsx", ".ts", ".tsx", ".swift", ".go", ".rs", ".java", ".kt",
              ".rb", ".html", ".css", ".sql", ".sh", ".toml", ".yaml", ".yml", ".ini", ".cfg"}
_TOP_LEVEL_DEF = re.compile(r"^(?:async\s+def|def|class|function|export\s+\w+)\b.*$", re.MULTILINE)

# The design doc says what to build; this sets how good anything shown in a
# browser has to look and feel. Only added for web project types, together
# with the project-specific design system from agents/ui_designer.py.
UI_UX_GUIDELINES = """UI/UX QUALITY BAR (web project — the interface must feel like an award-winning product, yet stay simple):
- Visual: neutral background, one accent color used sparingly, cards with soft borders and generous
  whitespace on an 8px grid. All colors, spacing and radii as CSS variables. Automatic dark mode via
  prefers-color-scheme.
- Typography: one clean font (e.g. Inter from Google Fonts, system-ui fallback), a clear size
  hierarchy, tabular numerals for money and data.
- Layout: responsive from 360px phones to wide desktops, never any horizontal scroll, centered
  content with a max width around 1120px.
- Interaction: every action gives feedback — hover/press states, inline validation under fields,
  toasts instead of alert(), friendly empty states with a call to action, loading skeletons,
  150-250ms ease-out transitions, and respect prefers-reduced-motion.
- Accessibility: semantic HTML, a label for every input, visible focus rings, full keyboard use,
  WCAG AA contrast, aria-live for status messages.
- Simplicity: plain HTML/CSS/vanilla JS unless the design document says otherwise — no frontend
  framework.
- Content: use the design document's own text, names, lists and numbers exactly as given — even
  where the design calls them placeholder content (the owner replaces them later). Never invent
  other content in their place, and no lorem ipsum.
- No build step exists (nothing runs npm, the Tailwind CLI, PostCSS or Sass). If the design asks
  for Tailwind, load it with its Play CDN in <head> — <script src="https://cdn.tailwindcss.com"></script>
  followed by <script>tailwind.config = {theme: {extend: {...design tokens...}}}</script> — and use
  its utility classes. Never link a compiled file (dist/output.css) and never put @tailwind or
  @apply in a .css file. Without Tailwind in the design, write plain CSS with custom properties.
  With the Play CDN its reset loads after your stylesheet: size headings with classes on the tags
  (text-4xl …), not in a .css file. Map the config's colors to the CSS variables (surface:
  'var(--surface)') so dark mode switches backgrounds and text together — no fixed bg-white or
  text-black on themed sections.
- Images: only link files that exist. A photo the design mentions but doesn't supply gets a
  placeholder sized like the photo (the person's initials in a circle, or an inline SVG) — an <img>
  pointing at a missing file shows as a broken image with its alt text.
- Scripts: every file a page needs gets its own <script src> (data files before the code that
  uses them); use type="module" whenever a script uses import/export."""


def _existing_code_context(project_path: str, budget: int = EXISTING_CODE_BUDGET) -> str:
    """Current src/ code, read from disk rather than files_written (which only
    has names, and is wiped on a backtrack). Without it every task was
    written blind and re-created models, storage and entry points. Full text
    for every file that fits in `budget` chars; the rest as top-level
    def/class lines (or, for a page or stylesheet, its size and ids) so the
    model still sees every module that exists — and that it isn't empty."""
    src = f"{project_path}/src"
    if not os.path.isdir(src):
        return ""
    paths = sorted(
        f for f in list_files(src)
        if "__pycache__" not in f and os.path.splitext(f)[1] in _CODE_EXTS
    )
    full, outlines, used = [], [], 0
    for rel in paths:
        try:
            content = read_file(f"{src}/{rel}")
        except Exception:
            continue
        if used + len(content) <= budget:
            full.append(f"### src/{rel}\n```\n{content}\n```")
            used += len(content)
        else:
            sigs = [m.group().rstrip() for m in _TOP_LEVEL_DEF.finditer(content)]
            if not sigs:
                ids = re.findall(r"""\bid\s*=\s*["']([^"']+)""", content)
                sigs = [f"({len(content):,} characters, not shown here. It is NOT empty"
                        + (f" — element ids: {', '.join('#' + i for i in ids[:30])}" if ids else "") + ")"]
            outlines.append(f"### src/{rel}\n" + "\n".join(sigs))
    if outlines:
        full.append("OTHER EXISTING FILES (signatures only — over the context budget):\n\n"
                    + "\n\n".join(outlines))
    return "\n\n".join(full)


def _syntax_errors(paths: list[str]) -> list[str]:
    """Real syntax errors in the files just written: Python via py_compile,
    JavaScript via `node --check` (as an ES module and as a plain script — a
    browser file is valid if either parses).

    Replaces an LLM self-review of every task: models can't reliably judge
    their own code without external feedback (Huang et al., ICLR 2024), and
    its low scores kept triggering slow full rewrites on opinion alone.
    Python files that compile also get ruff's fatal rules, so an undefined
    name gets the same one fix pass as a syntax error."""
    errors = []
    node = shutil.which("node")
    compiled = []
    for path in paths:
        name = path.split("/src/")[-1].split("/tests/")[-1]
        if path.endswith(".py"):
            try:
                py_compile.compile(path, doraise=True)
                compiled.append(path)
            except py_compile.PyCompileError as e:
                errors.append(f"{name}: {str(e.msg).strip()[:400]}")
        elif path.endswith(".js") and node:
            results = []
            for suffix in (".mjs", ".cjs"):
                with tempfile.NamedTemporaryFile("w", suffix=suffix, delete=False) as tmp:
                    tmp.write(open(path, errors="replace").read())
                r = subprocess.run([node, "--check", tmp.name], capture_output=True, text=True, timeout=20)
                os.unlink(tmp.name)
                results.append(r)
            if all(r.returncode for r in results):
                errors.append(f"{name}: {results[0].stderr.strip()[-400:]}")
    for f in static_checks.ruff_check(compiled):
        name = f["file"].split("/src/")[-1].split("/tests/")[-1]
        errors.append(f"{name}: line {f['line']}: {f['message']} (ruff {f['code']})")
    return errors


_FUNCTION_DEF = re.compile(
    r"^\s*(?:export\s+)?(?:async\s+)?function\s*\*?\s*([A-Za-z_$][\w$]*)"
    r"|^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s+)?"
    r"(?:function\b|\([^)]*\)\s*=>|[A-Za-z_$][\w$]*\s*=>)"
    r"|^(?:async\s+)?def\s+(\w+)|^class\s+(\w+)", re.MULTILINE)


def _function_names(text: str) -> set:
    return {name for groups in _FUNCTION_DEF.findall(text) for name in groups if name}


def _dropped(old: str, new: str, path: str) -> str:
    """What a rewrite of an existing file would lose, or "" if it keeps it.
    "Update index.html to add X" tasks used to return a page holding only X:
    a portfolio lost six of its eight sections, one task at a time — and on
    the rebuild, its card-rendering functions vanished from main.js."""
    if path.endswith((".html", ".htm")):
        ids = lambda text: set(re.findall(r"""\bid\s*=\s*["']([^"']+)""", text))
        lost = sorted(ids(old) - ids(new))
        if len(lost) >= 2:
            return "sections " + ", ".join(f"#{i}" for i in lost[:6])
    if path.endswith((".js", ".mjs", ".ts", ".py")):
        lost = sorted(_function_names(old) - _function_names(new))
        if lost:
            return "functions " + ", ".join(lost[:6])
    old_lines = [line for line in old.splitlines() if line.strip()]
    new_lines = [line for line in new.splitlines() if line.strip()]
    if len(old_lines) >= 20 and len(new_lines) < 0.6 * len(old_lines):
        return f"{len(old_lines) - len(new_lines)} of its {len(old_lines)} lines"
    return ""


def _damage(project_path: str, path: str, content: str) -> str:
    """Why writing this whole file over the current one would damage it, or "".
    For agents that return complete files outside the coder's own checks: a
    security "patch" swapped a portfolio's sections for "<!-- Sections content
    remains identical -->", and a decision fix cut the page to a skeleton."""
    if is_stub(content, path):
        return "it leaves content out"
    target = project_file_target(project_path, path)
    if target and os.path.exists(target):
        lost = _dropped(read_file(target), content, path)
        if lost:
            return f"it drops {lost}"
    return ""


def _as_edit(llm, task: dict, path: str, old: str, lost: str):
    """Ask again for this task's change to `path` as search/replace edits of
    the current file, which can't silently drop anything. New content, or None."""
    resp = llm.invoke([HumanMessage(content=(
        f"TASK: {task['name']}\nDESCRIPTION: {task['description']}\n\n"
        f"Your reply rewrote {path} and dropped {lost} that earlier tasks built. Everything already in the "
        f"file must stay. Return only EDIT blocks that add or change what this task needs.\n\n"
        f"CURRENT {path} (complete):\n{old}\n\n" + EDIT_FORMAT.format(path=path)))])
    edits = parse_edits(resp.content)
    pairs = edits.get(path) or next(iter(edits.values()), [])
    new, failed = apply_edits(old, pairs) if pairs else (old, 1)
    return new if pairs and not failed and new != old else None


def run_coder(state: AgentState) -> AgentState:
    """Writes ONE task per call and leaves phase="coding" while tasks remain —
    graph.py loops coder -> coder so every finished task is checkpointed."""
    task_list = state["task_list"]
    start_idx = state["current_task_index"]
    first_of_pass = start_idx == 0 or (
        start_idx < len(task_list) and task_list[start_idx].get("type") == "fix"
        and task_list[start_idx - 1].get("type") != "fix"
    )
    if first_of_pass:
        console.print("\n[bold blue]💻 CODER AGENT starting...[/bold blue]")
        blog("coder", f"Starting — writing code for {len(task_list) - start_idx} tasks")
    state["current_agent"] = "coder"
    state["phase"] = "coding"
    bstate({"current_agent": "coder", "phase": "coding", "task_index": start_idx, "task_total": len(task_list),
            "task_name": task_list[start_idx]["name"] if start_idx < len(task_list) else ""})

    llm = router.get_llm("coding", temperature=0.1)
    memory = AgentMemory(state["project_path"])

    context = f"""
PROJECT: {state['project_name']} ({state['project_type']})
ARCHITECTURE: {state['architecture'][:800]}
RESEARCH: {state['research_notes'][:600]}
"""

    for i in range(start_idx, len(task_list)):
        task = task_list[i]
        if task.get("type") == "setup":
            state["current_task_index"] = i + 1
            continue

        is_fix = task.get("type") == "fix"
        if is_fix:
            if not task.get("file"):
                # Reviewer's JSON omitted which file this issue is in — nothing
                # to patch. Skip rather than crash on an empty write path.
                console.print(f"[yellow]  ⚠️ Fix task has no target file, skipping: {task['name']}[/yellow]")
                state["current_task_index"] = i + 1
                continue

        console.print(f"\n[cyan]📝 Task {i+1}/{len(task_list)}: {task['name']}[/cyan]")
        blog("coder", f"Task {i+1}/{len(task_list)}: {task['name']}")

        if is_fix:
            # Patch mode — reviewer flagged a specific issue in a specific file.
            # The model sees the WHOLE file: it used to see only the first 4000
            # characters yet had to return the complete file "byte-for-byte",
            # which silently truncated any larger file.
            target_file = task.get("file", "")
            rel = target_file.removeprefix("src/")
            on_disk = (f"{state['project_path']}/{target_file}" if target_file.startswith("tests/")
                       else f"{state['project_path']}/src/{rel}")
            try:
                current_content = read_file(on_disk)
            except Exception:
                current_content = ""
            # Big files get search/replace edits (only the changed lines come
            # back — far fewer tokens, nothing to truncate); small ones and new
            # files are returned whole.
            patch_by_edits = current_content.count("\n") > 120
            design = ""
            if state.get("project_type") in WEB_PROJECT_TYPES:
                # "Missing section" fixes need the design's own content: without
                # it, fix after fix wrote "<!-- Sections omitted for brevity -->"
                # into a portfolio's page instead of its sections.
                design = (f"\nDESIGN DOCUMENT (the content to use: names, text, services, numbers):\n"
                          f"{state['design_content'][:4000]}\n")
            header = f"""You are patching an existing file to fix ONE specific issue. Do not rewrite
unrelated code, do not reformat, do not change anything except what's needed for this fix.
Write everything the fix adds in full: never stand in a comment such as "omitted for brevity".

FILE: {target_file}
ISSUE TO FIX: {task['description']}
{design}
CURRENT FILE CONTENT (complete):
```
{current_content[:40000] if current_content else '(file not found — write the corrected file from scratch, minimal and focused on this fix)'}
```
"""
            whole_file_prompt = header + f"""
Return exactly one FILE block for {target_file} holding the COMPLETE file with ONLY the lines
needed for this fix changed. Everything else must be byte-for-byte identical to the content above.

{FILE_FORMAT.format(root="the project folder — src/... for code, tests/... for tests")}"""
            code_prompt = (header + "\n" + EDIT_FORMAT.format(path=target_file)) if patch_by_edits else whole_file_prompt
        else:
            # Check memory for similar code
            memory_hint = memory.search(task['description'])

            # Past failures to avoid
            failures = memory.get_failures()
            failure_hint = "\n".join([f"- Avoid: {f['detail']}" for f in failures[-3:]]) if failures else ""

            existing_code = _existing_code_context(state["project_path"])
            ui_rules = ""
            if state.get("project_type") in WEB_PROJECT_TYPES:
                ui_rules = UI_UX_GUIDELINES
                if state.get("ui_spec"):
                    ui_rules += (
                        "\n\nPROJECT DESIGN SYSTEM (from the UI/UX designer — follow it exactly; copy its CSS "
                        "tokens verbatim into the stylesheet and use its microcopy):\n"
                        + state["ui_spec"][:7000]
                    )

            code_prompt = f"""You are an expert software engineer. Write production-quality code for this task.

TASK: {task['name']}
DESCRIPTION: {task['description']}

{context}
DESIGN DOCUMENT (source of truth for features and UI):
{state['design_content'][:4000]}

{ui_rules}

EXISTING CODE IN src/ (read from disk just now — this is the project as it stands):
{existing_code or '(none yet — this is the first code task)'}

{'SIMILAR CODE FROM MEMORY (adapt if useful):' + memory_hint if memory_hint else ''}
{'KNOWN FAILURES TO AVOID:' + failure_hint if failure_hint else ''}

RULES:
- Import from and extend the existing modules above. Never create a second version of a model,
  storage layer, CLI or entry point that already exists — change that file instead.
- Follow the file layout in the architecture above.
- Return only the files you create or change; a changed file must be returned complete.
- Keep this reply small: at most 3 files and about 300 lines per file. If the task needs more,
  write the core files now — later tasks add the rest. Be concise: short docstrings, no
  commented-out code. (Longer replies get cut off.)

Write complete, working, production-quality code. No placeholders or TODOs.

{FILE_FORMAT.format(root="the project folder — src/... for code, tests/... for tests")}"""

        resp = llm.invoke([HumanMessage(content=code_prompt)])
        if is_fix and patch_by_edits:
            pairs = [pair for pairs in parse_edits(resp.content).values() for pair in pairs]
            patched, failed = apply_edits(current_content, pairs) if pairs else (current_content, 1)
            if pairs and not failed and patched != current_content:
                files = {target_file: patched}
            else:
                console.print("[yellow]  ⚠️ Edit didn't match the file — asking for the whole file instead[/yellow]")
                resp = llm.invoke([HumanMessage(content=whole_file_prompt)])
                files = parse_files(resp.content)
        else:
            files = parse_files(resp.content)
        if files is None:
            # Usually a reply cut off before its first file finished — ask
            # once more for a smaller answer before giving up on this task.
            console.print("[yellow]  ⚠️ No complete file in the reply — retrying once, asking for fewer files[/yellow]")
            retry_resp = llm.invoke([HumanMessage(content=(
                code_prompt + "\n\nYour previous reply had no complete FILE block (it was probably cut "
                "off). Reply again with fewer, shorter files — only the ones this task needs most."
            ))])
            resp = retry_resp
            files = parse_files(retry_resp.content)
        if files and was_cut_off(resp.content):
            # File blocks let a truncated reply still count: keep what finished.
            console.print(f"[yellow]  ✂️ Reply was cut off — kept the {len(files)} complete file(s)[/yellow]")
        if files is None:
            # Recorded as an error instead of silently marking the task done,
            # so tester/decision/final_review can see the gap.
            msg = f"coder: task '{task['name']}' produced no parseable output"
            console.print(f"[red]  ❌ {msg}[/red]")
            blog("coder", msg)
            state["errors"].append(msg)
            state["current_task_index"] = i + 1
            break

        # Write files — write_project_file returns "" for rejected paths
        written_now = []
        for filepath, content in files.items():
            target = project_file_target(state["project_path"], filepath)
            # Fixes too: a review fix adding SEO tags rewrote the page without
            # the sections the fix before it had just restored.
            if target and os.path.exists(target):
                old = read_file(target)
                lost = _dropped(old, content, filepath)
                if lost:
                    console.print(f"[yellow]  ⚠️ {filepath}: the new version drops {lost} — asking for edits "
                                  f"instead[/yellow]")
                    content = _as_edit(llm, task, filepath, old, lost)
                    if content is None:
                        msg = f"coder: '{task['name']}' would have dropped {lost} from {filepath}; kept the old file"
                        console.print(f"[yellow]  ↩️ {msg}[/yellow]")
                        state["errors"].append(msg)
                        continue
            written = write_project_file(state["project_path"], filepath, content)
            if written:
                written_now.append(written)
                state["files_written"].append(written)

        # Check with the real compilers; one fix pass only if there are real errors.
        errors = _syntax_errors(written_now)
        if errors:
            console.print(f"[yellow]  🔍 {len(errors)} syntax error(s) or undefined name(s) — one fix pass[/yellow]")
            current = "\n\n".join(f"=== FILE: {p.split(state['project_path'] + '/')[-1]} ===\n"
                                  f"{open(p, errors='replace').read()}\n=== END FILE ===" for p in written_now)
            fix_resp = llm.invoke([HumanMessage(content=(
                "These files have errors (found by the compiler and ruff):\n" + "\n".join(f"- {e}" for e in errors) +
                f"\n\nCURRENT FILES:\n{current}\n\nFix the errors and return the corrected files complete.\n\n"
                + FILE_FORMAT.format(root="the project folder — src/... for code, tests/... for tests")
            ))])
            for filepath, content in (parse_files(fix_resp.content) or {}).items():
                write_project_file(state["project_path"], filepath, content)
            errors = _syntax_errors(written_now)
            if errors:
                state["errors"].append(f"coder: syntax errors after '{task['name']}': " + "; ".join(errors)[:400])

        all_code = "\n\n".join(f"# {k}\n{v}" for k, v in files.items())
        memory.store("coder", all_code[:1000], {"task": task["name"], "outcome": "written"})
        state["current_task_index"] = i + 1
        log_agent(state, "coder", f"Completed task: {task['name']} ({len(written_now)} file(s)"
                                  f"{', syntax errors remain' if errors else ''})")
        break

    if state["current_task_index"] < len(task_list):
        return state  # phase stays "coding" — graph loops back for the next task

    console.print(f"[green]✅ Coding done — {len(state['files_written'])} files written[/green]")
    blog("coder", f"Done — {len(state['files_written'])} files written")

    # Round >= 3 means this pass patched reviewer's round-3 fix tasks (the
    # coder never runs again after that escalation).
    if state.get("coder_round", 0) >= MAX_REVIEW_ROUNDS:
        # Just finished round 3's patch and reviewer has failed 3 times —
        # per the escalation ladder, skip reviewer and go straight to decision
        # with the full review history instead of a 4th review pass.
        console.print(
            f"[red]🚨 Round {state['coder_round']} patch done — escalating to decision "
            f"agent instead of another review pass[/red]"
        )
        bstate({"current_agent": "coder", "phase": "decision_escalation"})
        state["phase"] = "decision_escalation"
    else:
        bstate({"current_agent": "coder", "phase": "review"})
        state["phase"] = "review"
    return state
