"""
Join node after security + optimizer run in parallel (both read-only).
Applies security's fixes first — they matter more than style — then
optimizer's suggestions, skipping any file security already touched this
round so optimizer's pre-fix version can't clobber a security patch.
Both return whole files, so each one gets the coder's check first: a
version that leaves content out or drops sections is refused.
"""
from pipeline.state import AgentState, log_agent
from tools.file_ops import write_doc, write_project_file
from agents.coder import _damage
from server.bridge import log as blog, state as bstate
from rich.console import Console

console = Console()


def run_code_merge(state: AgentState) -> AgentState:
    console.print("\n[bold magenta]🔀 Merging security + optimizer results...[/bold magenta]")
    state["current_agent"] = "code_merge"
    state["phase"] = "review"
    bstate({"current_agent": "code_merge", "phase": "review"})

    findings = state.get("_security_findings", [])
    fixed_files = state.get("_security_fixed_files", {})
    notes = state.get("_optimization_notes", [])
    optimized_files = state.get("_optimized_files", {})

    fixed_count = 0
    security_touched = set()
    refused = []
    for filepath, content in fixed_files.items():
        rel = filepath.replace("src/", "")
        why = _damage(state["project_path"], rel, content)
        if why:
            console.print(f"[yellow]  ↩️ Kept {rel} — refused the security patch: {why}[/yellow]")
            refused.append(f"security patch for {rel}: {why}")
            continue
        if not write_project_file(state["project_path"], rel, content):
            continue
        fixed_count += 1
        security_touched.add(rel)
        console.print(f"[cyan]  🔧 Security patch: {rel}[/cyan]")

    changed_count = 0
    skipped = []
    for filepath, content in optimized_files.items():
        rel = filepath.replace("src/", "")
        if rel in security_touched:
            skipped.append(rel)
            continue
        why = _damage(state["project_path"], rel, content)
        if why:
            console.print(f"[yellow]  ↩️ Kept {rel} — refused the optimization: {why}[/yellow]")
            refused.append(f"optimization of {rel}: {why}")
            continue
        if not write_project_file(state["project_path"], rel, content):
            continue
        changed_count += 1
        console.print(f"[cyan]  ⚡ Optimized: {rel}[/cyan]")

    high = [f for f in findings if f.get("severity") == "high"]
    write_doc(state["project_path"], "05b_security.md",
        f"# Security Scan\n\n**Findings:** {len(findings)} ({len(high)} high), {fixed_count} patched\n\n"
        f"## Findings\n" + "\n".join(
            [f"- [{f.get('severity','?').upper()}] {f.get('file','?')}: {f.get('issue','')}" for f in findings]
        ) + "".join(f"\n- (refused the {r})" for r in refused if r.startswith("security")))

    opt_note_lines = notes[:]
    if skipped:
        opt_note_lines.append(f"(skipped optimizing {', '.join(skipped)} — security patched it this round)")
    opt_note_lines += [f"(refused the {r})" for r in refused if r.startswith("optimization")]
    write_doc(state["project_path"], "06b_optimizations.md",
        f"# Optimization Pass\n\n**Files changed:** {changed_count}\n\n"
        f"## Notes\n" + "\n".join([f"- {n}" for n in opt_note_lines]))

    log_agent(state, "code_merge",
              f"Applied {fixed_count} security fixes, {changed_count} optimizations "
              f"({len(skipped)} skipped for overlap, {len(refused)} refused)")
    console.print(f"[green]✅ Merge done — {fixed_count} security fixes, {changed_count} optimizations[/green]")
    blog("code_merge", f"Done — {fixed_count} security fixes, {changed_count} optimizations applied"
                       + (f", {len(refused)} refused (would have lost content)" if refused else ""))
    return state
