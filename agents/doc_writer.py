"""
Doc writer — runs twice:
  draft  (run_doc_writer_draft): right after reviewer passes, in parallel
          with test_writer/tester. Writes a first README from the plan/
          architecture — doesn't wait on tests, so it doesn't slow down the
          test_writer -> tester chain.
  final  (run_doc_writer_final): after tester/decision resolves, in parallel
          with final_review. Refines the draft with the finished file list.
Both write README.md directly — safe because they're never scheduled
concurrently with each other, only with unrelated branches (test_writer/
tester for the draft, final_review for the final pass).
"""
from langchain_core.messages import HumanMessage
from pipeline.state import AgentState
from tools.file_ops import write_doc, write_root_file, list_files
from server.bridge import log as blog, state as bstate
from router import router
from rich.console import Console

console = Console()


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
    return text


def run_doc_writer_draft(state: AgentState) -> AgentState:
    console.print("\n[bold blue]📚 DOC WRITER (draft) starting...[/bold blue]")
    blog("doc_writer", "Starting — drafting README structure")
    bstate({"current_agent": "doc_writer", "phase": "documenting"})

    llm = router.get_llm("review", model_override="gemini-2.0-flash")

    draft_prompt = f"""Write a README.md for this project — it just passed code review, tests
haven't run yet, so keep run/test instructions general.

PROJECT: {state['project_name']} ({state['project_type']})
PLAN: {state['plan'][:1000]}
ARCHITECTURE: {state['architecture'][:800]}
LIBRARIES: {', '.join(state.get('libraries', []))}

Include:
1. Project title and one-line description
2. Features
3. Setup / installation instructions (with exact commands)
4. How to run it
5. Project structure overview
6. Tech stack

Return ONLY the README.md content in markdown. No preamble, no code fences around the whole thing."""

    resp = llm.invoke([HumanMessage(content=draft_prompt)])
    draft = _strip_fences(resp.content)
    write_root_file(state["project_path"], "README.md", draft)

    console.print("[green]✅ Draft README written[/green]")
    blog("doc_writer", "Draft README written")
    # Partial return — parallel sibling of test_writer (see pipeline/graph.py).
    return {"_readme_draft": draft}


def run_doc_writer_final(state: AgentState) -> AgentState:
    console.print("\n[bold blue]📚 DOC WRITER (final) starting...[/bold blue]")
    blog("doc_writer", "Finalizing README with tested file list")
    bstate({"current_agent": "doc_writer", "phase": "documenting"})

    llm = router.get_llm("review", model_override="gemini-2.0-flash")

    all_files = list_files(state["project_path"])
    src_files = [f for f in all_files if f.startswith("src/")]

    final_prompt = f"""Refine this README draft now that the project is fully built and tested.

DRAFT:
{state.get('_readme_draft', '')[:3000]}

FINAL FILE LIST: {', '.join(src_files[:25])}
KNOWN ISSUES: {', '.join(state.get('known_issues', [])) or 'None'}

Update setup/run instructions to match the real file list, and add a "Known Issues" section
only if there are any listed above. Keep everything else from the draft that's still accurate.

Return ONLY the complete, final README.md content in markdown. No preamble, no code fences
around the whole thing."""

    resp = llm.invoke([HumanMessage(content=final_prompt)])
    final_readme = _strip_fences(resp.content)

    write_root_file(state["project_path"], "README.md", final_readme)
    write_doc(state["project_path"], "09b_docs.md",
              f"# Documentation\n\nGenerated README.md at project root.\n\n---\n\n{final_readme}")

    console.print("[green]✅ Final README written[/green]")
    blog("doc_writer", "Done — README.md finalized")
    # Partial return ({}) — runs in parallel with final_review, which keeps
    # the normal full-state return as the "primary" sibling in this step.
    return {}
