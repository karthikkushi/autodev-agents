"""
Security agent — runs after coder, in parallel with optimizer (both are
read-only: they only look at code, never write it, so two branches can
safely read the same files at once). Findings and suggested fixes go into
state for agents/code_merge.py to apply afterward — writing here would race
with optimizer's own writes if both ran concurrently.
"""
from langchain_core.messages import HumanMessage
from pipeline.state import AgentState
from tools.file_blocks import FILE_FORMAT, parse_file_blocks, parse_meta_json
from tools import static_checks
from agents.coder import _existing_code_context, REVIEW_CODE_BUDGET
from server.bridge import log as blog, state as bstate
from router import router
from rich.console import Console

console = Console()


def run_security(state: AgentState) -> AgentState:
    console.print("\n[bold red]🛡️  SECURITY AGENT starting...[/bold red]")
    blog("security", "Starting — scanning code for vulnerabilities")
    bstate({"current_agent": "security", "phase": "security"})

    llm = router.get_llm("coding")

    full_code = _existing_code_context(state["project_path"], REVIEW_CODE_BUDGET)
    if not full_code:
        console.print("[yellow]No source files to scan[/yellow]")
        blog("security", "No files to scan — skipping")
        return {"_security_findings": [], "_security_fixed_files": {}}

    # Free checks first: bandit reads the code, pip-audit the installed
    # dependencies. Their findings are certain, so the model is asked to fix
    # the HIGH ones rather than to find them again.
    checks = static_checks.security_checks(state["project_path"])
    static_checks.write_report(state["project_path"], checks)
    tool_findings = checks["bandit"] + checks["pip-audit"]
    tools_block = ""
    if tool_findings:
        console.print(f"[yellow]  🔧 bandit/pip-audit found {len(tool_findings)} issue(s)[/yellow]")
        tools_block = ("\nFOUND BY TOOLS (certain — fix every HIGH one in your FILE blocks; for a vulnerable "
                       "package, pin a fixed version in requirements.txt; don't list these again):\n"
                       + static_checks.as_lines(tool_findings[:40]) + "\n")

    scan_prompt = f"""You are a security auditor. Scan this codebase for vulnerabilities.

PROJECT: {state['project_name']} ({state['project_type']})

CODEBASE (src/):
{full_code}
{tools_block}
Look for: SQL injection, XSS, hardcoded secrets/API keys, missing input validation,
missing auth checks, unsafe deserialization, path traversal, insecure defaults.
A missing Content-Security-Policy is "low" at most. Never add or tighten one: these sites load
Tailwind from its Play CDN (it runs an inline config script and injects inline styles), web
fonts and form endpoints, and a policy that blocks any of them breaks the page.

First a JSON object with your findings:
{{"findings": [{{"file": "app.py", "severity": "high|medium|low", "issue": "...", "fix": "..."}}]}}
Then, only for files with a HIGH severity finding (yours or the tools'), the complete fixed file as a FILE block:
every line the fix doesn't need stays exactly as it is. Never stand a comment in for content.

{FILE_FORMAT.format(root="the project folder, e.g. src/app.py")}"""

    resp = llm.invoke([HumanMessage(content=scan_prompt)])
    raw = resp.content.strip()
    data = parse_meta_json(raw)
    findings = data.get("findings", []) if isinstance(data.get("findings"), list) else []
    fixed_files = parse_file_blocks(raw) or {
        fp: c for fp, c in (data.get("fixed_files") or {}).items() if fp and isinstance(c, str)
    }

    # Same shape code_merge reads (severity in lower case), plus the tool name.
    findings = [{"tool": f["tool"], "file": f["file"], "line": f["line"], "severity": f["severity"].lower(),
                 "issue": f"{f['code']}: {f['message']}", "fix": "see the tool's message"}
                for f in tool_findings] + [f for f in findings if isinstance(f, dict)]
    console.print(f"[green]✅ Security scan done — {len(findings)} findings, {len(fixed_files)} suggested fixes[/green]")
    blog("security", f"Done — {len(findings)} findings, {len(fixed_files)} suggested fixes")
    # Partial return — runs in parallel with optimizer (see pipeline/graph.py
    # and agents/code_merge.py, which applies both branches' results and logs
    # the combined outcome; log_agent here would be silently dropped anyway
    # since "agent_logs" isn't in this return value).
    return {"_security_findings": findings, "_security_fixed_files": fixed_files}
