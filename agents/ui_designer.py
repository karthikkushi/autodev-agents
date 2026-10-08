"""
UI/UX designer — runs between environment and coder, for web project types
only. Turns the design doc + plan into one concrete design system (CSS
tokens, type scale, layout, component states, motion, accessibility, copy)
that every coder task follows. Without it, pages written in separate coder
tasks each invented their own look. Non-web projects pass straight through.
"""
from langchain_core.messages import HumanMessage
from pipeline.state import AgentState, log_agent
from tools.file_ops import write_doc
from server.bridge import log as blog, state as bstate
from router import router
from rich.console import Console

console = Console()
WEB_PROJECT_TYPES = ("webapp", "fullstack")


def run_ui_designer(state: AgentState) -> AgentState:
    if state.get("project_type") not in WEB_PROJECT_TYPES:
        return state

    console.print("\n[bold magenta]🎨 UI/UX DESIGNER starting...[/bold magenta]")
    state["current_agent"] = "ui_designer"
    blog("ui_designer", "Starting — designing the UI system and UX flows")
    bstate({"current_agent": "ui_designer", "phase": "ui_design"})

    llm = router.get_llm("reasoning", temperature=0.3)

    prompt = f"""You are a senior product designer whose work wins design awards (Awwwards, Apple
Design Awards). Design the UI and UX for this web project. Your spec is handed straight to
engineers, so every value must be concrete — no "choose a nice color".

PROJECT: {state['project_name']} ({state['project_type']})
DESIGN DOCUMENT:
{state['design_content'][:4000]}

PLAN (summary):
{state['plan'][:1500]}

Aim for calm, premium and effortless — beautiful but simple. Buildable with plain HTML, CSS and
vanilla JS (no frameworks) unless the design document says otherwise. Nothing ever builds the
project (no npm, Tailwind CLI, PostCSS or Sass): if the design asks for Tailwind it is loaded from
the Play CDN (cdn.tailwindcss.com), so give the tokens as a tailwind.config theme.extend object
(colors, fontFamily, spacing, borderRadius) whose colors point at the CSS variables below
(surface: 'var(--surface)'): fixed hex values there never switch in dark mode, and a portfolio's
cards stayed white under white dark-mode text. Otherwise specify plain CSS custom properties.

Write the spec in markdown with exactly these sections:
1. Design direction — 3 adjectives and one short paragraph on the feel.
2. Design tokens — a complete CSS `:root {{ }}` block (colors incl. background, surface, text,
   muted text, border, one accent + hover, success, warning, danger; spacing scale on an 8px grid;
   radii; shadows; font family; font sizes; line heights; transition durations/easing), then a
   `@media (prefers-color-scheme: dark) {{ :root {{ }} }}` block overriding the colors.
   All text/background pairs must meet WCAG AA contrast.
3. Typography — the type scale and where each size is used; tabular numerals for numbers.
4. Layout — page structure for desktop and for mobile (< 768px), max content width, grid.
5. Components — for each component the page needs: anatomy plus default, hover, active, focus,
   disabled, loading and error states.
6. UX flows — the main user task step by step, with the feedback shown at each step.
7. Motion — what animates, duration and easing; the prefers-reduced-motion fallback.
8. Accessibility — semantic structure, labels, focus order, keyboard shortcuts, aria-live use.
9. Microcopy — exact text for the title, buttons, empty state, errors and success messages.

Return ONLY the markdown spec."""

    resp = llm.invoke([HumanMessage(content=prompt)])
    state["ui_spec"] = resp.content.strip()
    write_doc(state["project_path"], "04b_ui_design.md", f"# UI/UX Design System\n\n{state['ui_spec']}")

    log_agent(state, "ui_designer", f"Design system written ({len(state['ui_spec'])} chars)")
    console.print("[green]✅ UI/UX design system ready[/green]")
    blog("ui_designer", "Done — design system ready for the coder")
    bstate({"current_agent": "ui_designer", "phase": "coding"})
    return state
