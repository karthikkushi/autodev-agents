"""Shadow Auditor — runs as a background daemon thread alongside the main pipeline.

100% API, same as the rest of the pipeline — no local model. Uses the "fast"
role (Gemini primary) through the same router every agent uses, which shares
the router's rate limiter, so audit calls count against the same per-provider
budget rather than adding a separate untracked load. Watches the same /state
the phone/dashboard watch, detects when an agent finishes, reads its output
file, asks the model "find issues, missing cases, improvements", and pushes
any finding to the phone via the bridge. Never blocks or modifies the main
pipeline — if a call fails, it just skips that audit.
"""

import os
import time
import threading
import urllib.request
import urllib.error
import json

from auditor.audit_prompts import AUDIT_PROMPTS, AUDIT_TARGETS
from server.bridge import log as blog
from router import router
from pipeline.control import PROJECTS_DIR

STATE_URL = "http://localhost:8081/state"
POLL_INTERVAL = 5  # seconds
PROJECTS_BASE = str(PROJECTS_DIR)


class ShadowAuditor:
    def __init__(self):
        self._thread = None
        self._audited = set()  # (project_name, agent) pairs already audited this run

    def start(self):
        self._thread = threading.Thread(target=self._watch_loop, daemon=True, name="shadow-auditor")
        self._thread.start()
        print(f"[shadow_auditor] started — watching {STATE_URL} every {POLL_INTERVAL}s")

    # ── watch loop ───────────────────────────────────────────────────────

    def _watch_loop(self):
        last_agent = None
        last_project = None
        while True:
            try:
                state = self._poll_state()
                agent = state.get("current_agent")
                project = state.get("project_name")

                if agent != last_agent and last_agent and last_project:
                    self._audit_agent_output(last_agent, last_project)

                last_agent = agent
                last_project = project or last_project
            except Exception as e:
                print(f"[shadow_auditor] watch loop error (non-fatal): {e}")

            time.sleep(POLL_INTERVAL)

    def _poll_state(self) -> dict:
        try:
            with urllib.request.urlopen(STATE_URL, timeout=2) as resp:
                return json.loads(resp.read())
        except Exception:
            return {}

    # ── audit ────────────────────────────────────────────────────────────

    def _audit_agent_output(self, agent: str, project_name: str):
        if agent not in AUDIT_PROMPTS:
            return  # not an agent we audit (e.g. intake, researcher, system)

        key = (project_name, agent)
        if key in self._audited:
            return
        self._audited.add(key)

        project_path = f"{PROJECTS_BASE}/{project_name}"
        output = self._read_agent_output(agent, project_path)
        if not output:
            return

        try:
            llm = router.get_llm("fast")
            prompt = AUDIT_PROMPTS[agent].format(output=output[:4000])
            resp = llm.invoke(prompt)
            findings = (resp.content if hasattr(resp, "content") else str(resp)).strip()
        except Exception as e:
            print(f"[shadow_auditor] audit failed for {agent} (non-fatal): {e}")
            return

        if findings and findings.upper() != "NONE":
            print(f"[shadow_auditor] [{agent}] {findings}")
            blog("shadow_auditor", f"[{agent}] {findings}")

    def _read_agent_output(self, agent: str, project_path: str) -> str:
        if agent == "coder":
            return self._read_src_files(project_path)

        rel_path = AUDIT_TARGETS.get(agent)
        if not rel_path:
            return ""
        full_path = f"{project_path}/{rel_path}"
        if not os.path.exists(full_path):
            return ""
        try:
            with open(full_path) as f:
                return f.read()
        except Exception:
            return ""

    def _read_src_files(self, project_path: str) -> str:
        src_dir = f"{project_path}/src"
        if not os.path.isdir(src_dir):
            return ""
        chunks = []
        total = 0
        for root, _, files in os.walk(src_dir):
            for fname in files:
                if total > 6000:
                    break
                fpath = os.path.join(root, fname)
                rel = os.path.relpath(fpath, project_path)
                try:
                    with open(fpath) as f:
                        content = f.read()
                except Exception:
                    continue
                chunk = f"### {rel}\n{content[:1500]}\n"
                chunks.append(chunk)
                total += len(chunk)
        return "\n".join(chunks)


shadow_auditor = ShadowAuditor()
