#!/usr/bin/env python3
"""
Benchmark free models on this account with two real pipeline-style tasks:

  coding — write a module + pytest file in the pipeline's FILE-block format;
           scored by actually running pytest on what came back
  json   — turn a short plan into the planner's task-list JSON

Usage:  python3 tools/benchmark_models.py            # default candidate list
        python3 tools/benchmark_models.py MODEL ...  # litellm ids, e.g. nvidia_nim/qwen/qwen3-coder-480b-a35b-instruct

Results go to logs/model_benchmark.json. Provider catalogs change often and
list models an account can't call (a third of NVIDIA's listing returned 404
here), so re-run this before reordering router/roles.py.
"""
import concurrent.futures
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
from dotenv import load_dotenv  # noqa: E402
load_dotenv(BASE / ".env")
import litellm  # noqa: E402
from router.llm_router import (  # noqa: E402
    _clear_cooldown, _failure_info, _log_provider_error, _no_thinking_params, _provider_params, _sampling_params,
    _start_cooldown, resting_reason,
)
from tools.file_blocks import FILE_FORMAT, parse_files  # noqa: E402

litellm.suppress_debug_info = True
TIMEOUT = 150

CANDIDATES = [
    "gemini/gemini-3.8-flash", "gemini/gemini-3.5-flash", "gemini/gemini-3.1-flash-lite",
    "nvidia_nim/qwen/qwen3-coder-480b-a35b-instruct",
    "nvidia_nim/mistralai/devstral-2-123b-instruct-2512",
    "nvidia_nim/minimaxai/minimax-m2.7",
    "nvidia_nim/qwen/qwen3-next-80b-a3b-instruct",
    "nvidia_nim/z-ai/glm-5.3", "nvidia_nim/z-ai/glm-5.3-flash",
    "nvidia_nim/moonshotai/kimi-k3",
    "nvidia_nim/openai/gpt-oss-20b",
    "nvidia_nim/google/gemma-4-31b-it",
    "nvidia_nim/poolside/laguna-xs-2.1",
    "nvidia_nim/meta/llama-3.3-70b-instruct",
    "nvidia_nim/nvidia/nemotron-3-super-120b-a12b",
    "nvidia_nim/deepseek-ai/deepseek-v4.1-flash",
    "openrouter/qwen/qwen3.8-27b:free",
    "openrouter/poolside/laguna-s-2.1:free",
    "openrouter/cohere/north-mini-code:free",
    "openrouter/google/gemma-4-31b-it:free",
    "openrouter/thinkingmachines/inkling:free",
    "openrouter/nvidia/nemotron-3-ultra-550b-a55b:free",
    "openrouter/nvidia/nemotron-3-super-120b-a12b:free",
    # More of NVIDIA's 81-model catalogue and OpenRouter's :free list (2026-09-30):
    # the chat/code models not benchmarked before.
    "nvidia_nim/moonshotai/kimi-k2.6", "nvidia_nim/nvidia/nemotron-3-ultra-550b-a55b",
    "nvidia_nim/nvidia/nemotron-3.5-lightning-30b-a3b", "nvidia_nim/nvidia/nemotron-nano-3-30b-a3b",
    "nvidia_nim/nvidia/llama-3.1-nemotron-ultra-253b-v1", "nvidia_nim/nvidia/llama-3.1-nemotron-70b-instruct",
    "nvidia_nim/mistralai/mistral-large-2-instruct", "nvidia_nim/mistralai/codestral-22b-instruct-v0.1",
    "nvidia_nim/ibm/granite-34b-code-instruct", "nvidia_nim/google/gemma-3-12b-it",
    "nvidia_nim/google/diffusiongemma-26b-a4b-it", "nvidia_nim/meta/muse-glimmer-30b",
    "nvidia_nim/01-ai/yi-large", "nvidia_nim/ai21labs/jamba-1.5-large-instruct",
    "nvidia_nim/microsoft/phi-3.5-moe-instruct", "nvidia_nim/nvidia/nemotron-4-340b-instruct",
    "openrouter/google/gemma-4-26b-a4b-it:free", "openrouter/nvidia/nemotron-3.5-lightning:free",
    "openrouter/poolside/laguna-xs-2.1:free", "openrouter/thinkingmachines/inkling-small:free",
    "openrouter/dots-studio/dots-3-note-preview:free", "openrouter/inclusionai/ling-3.0-flash-sante:free",
    "openrouter/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    # No-card providers found 2026-09-30, skipped until their keys are in .env.
    # Ollama Cloud's model names come from https://ollama.com/api/tags once a
    # key exists (e.g. "ollama_chat/gpt-oss:120b").
    "cloudflare/@cf/openai/gpt-oss-120b", "cloudflare/@cf/zai-org/glm-5.3-flash",
    "cloudflare/@cf/qwen/qwen2.5-coder-32b-instruct", "cloudflare/@cf/meta/llama-4-scout-17b-16e-instruct",
    "zai/glm-4.7-flash", "zai/glm-4.5-flash",
    "mistral/codestral-latest", "mistral/devstral-small-latest", "mistral/mistral-large-latest",
    # Added 2026-09-30 from each provider's /models listing (chat models only).
    # Cerebras isn't here: its sign-up asks for a credit card (free-only rule).
    "groq/openai/gpt-oss-120b", "groq/openai/gpt-oss-20b", "groq/qwen/qwen3.8-27b",
    "sambanova/DeepSeek-V3.1", "sambanova/DeepSeek-V3.2", "sambanova/Meta-Llama-3.3-70B-Instruct",
    "sambanova/MiniMax-M2.7", "sambanova/MiniMax-M3", "sambanova/gemma-4-31B-it", "sambanova/gpt-oss-120b",
]
KEYS = {"gemini": "GOOGLE_API_KEY", "nvidia_nim": "NVIDIA_API_KEY", "openrouter": "OPENROUTER_API_KEY",
        "groq": "GROQ_API_KEY", "cerebras": "CEREBRAS_API_KEY", "mistral": "MISTRAL_API_KEY",
        "sambanova": "SAMBANOVA_API_KEY", "cloudflare": "CLOUDFLARE_API_KEY", "zai": "ZAI_API_KEY",
        "ollama_chat": "OLLAMA_API_KEY"}

CODING_PROMPT = f"""You are an expert software engineer. Write production-quality code for this task.

TASK: Implement the bill splitter
DESCRIPTION: Write src/calculator.py with split_bill(bill, tip_percent, people) returning a dict
with tip_total, total, tip_per_person and total_per_person, each rounded to 2 decimals. Raise
ValueError when bill <= 0, people < 1, or tip_percent is outside 0-100. Also write
tests/test_calculator.py with pytest tests (import with `from calculator import split_bill`).

RULES: at most 3 files, concise code.

{FILE_FORMAT.format(root="the project folder — src/... for code, tests/... for tests")}"""

JSON_PROMPT = """Extract the CODING tasks from this plan, in build order.

PLAN:
1. Flask app with a JSON endpoint POST /api/split that validates input and calls calculator.split_bill.
2. calculator.py with the bill-splitting math.
3. templates/index.html with the form (bill, tip presets, people stepper) and a results panel.
4. static/style.css with CSS variables, dark mode and a mobile-first layout.
5. static/app.js that calls /api/split (debounced) and renders results and inline errors.

Return ONLY a JSON array like:
[{"id": 1, "name": "Create calculator", "description": "Write src/calculator.py: ...", "type": "code"}]"""


# Module-level pool, never a `with` block: leaving `with ThreadPoolExecutor`
# waits for the hung call, which silently defeats the timeout.
_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=32)


def _call(model: str, prompt: str, max_tokens: int):
    # Same cooldown file as the pipeline: don't spend a request on a model
    # that is out of its daily quota, and tell the pipeline what we learn.
    resting = resting_reason(model)
    if resting:
        return None, 0.0, f"skipped — {resting}"
    extra = _provider_params(model)
    if extra is None:
        return None, 0.0, "skipped — provider setting missing (e.g. CLOUDFLARE_ACCOUNT_ID)"
    key = os.environ.get(KEYS[model.split("/", 1)[0]], "")
    started = time.time()
    fut = _POOL.submit(litellm.completion, model=model, messages=[{"role": "user", "content": prompt}],
                       api_key=key, max_tokens=max_tokens, timeout=TIMEOUT, **extra,
                       **_sampling_params(model, 0.1), **_no_thinking_params(model))
    try:
        resp = fut.result(timeout=TIMEOUT)
    except concurrent.futures.TimeoutError:
        return None, time.time() - started, f"no reply in {TIMEOUT}s"
    except Exception as e:
        info = _failure_info(e)
        _log_provider_error(model, "benchmark", e)
        _start_cooldown(model, e, info)
        return None, time.time() - started, f"{type(e).__name__}: {info['detail'][:90]}"
    _clear_cooldown(model)
    content = resp.choices[0].message.content or ""
    return content, time.time() - started, "" if content.strip() else "empty reply"


def _score_coding(content: str) -> tuple[bool, str]:
    files = parse_files(content) or {}
    if not files:
        return False, "no files"
    d = Path(tempfile.mkdtemp())
    for path, text in files.items():
        p = path.strip().removeprefix("./").removeprefix("../")
        target = d / (p if p.startswith(("src/", "tests/")) else f"src/{p}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    if not (d / "tests").is_dir():
        return False, f"files {list(files)} but no tests/"
    r = subprocess.run([sys.executable, "-m", "pytest", "tests", "-q"], cwd=d, capture_output=True, text=True,
                       timeout=60, env={**os.environ, "PYTHONPATH": str(d / "src")})
    last = (r.stdout.strip().splitlines() or ["?"])[-1]
    return r.returncode == 0, last


def _score_json(content: str) -> tuple[bool, str]:
    from agents.planner import _parse_task_list
    tasks = _parse_task_list(content)
    return (bool(tasks) and 3 <= len(tasks) <= 12), f"{len(tasks) if tasks else 0} tasks"


def bench(model: str) -> dict:
    row = {"model": model}
    content, secs, err = _call(model, CODING_PROMPT, 8192)
    row["coding_s"] = round(secs, 1)
    row["coding_ok"], row["coding_note"] = (_score_coding(content) if content else (False, err))
    if err and "NotFound" in err:  # the account can't call it — skip the second task
        row.update(json_s=None, json_ok=False, json_note="not available")
        return row
    if err.startswith("skipped"):  # resting on a daily quota — the second task would fail too
        row.update(json_s=None, json_ok=False, json_note="skipped")
        return row
    content, secs, err = _call(model, JSON_PROMPT, 2000)
    row["json_s"] = round(secs, 1)
    row["json_ok"], row["json_note"] = (_score_json(content) if content else (False, err))
    return row


def main():
    models = sys.argv[1:] or CANDIDATES
    no_key = [m for m in models if not os.environ.get(KEYS.get(m.split("/", 1)[0], ""), "")]
    if no_key:
        print(f"Skipping {len(no_key)} model(s) with no API key set: {', '.join(no_key)}", flush=True)
        models = [m for m in models if m not in no_key]
    print(f"Benchmarking {len(models)} models (coding + json)…", flush=True)
    rows = []
    runner = concurrent.futures.ThreadPoolExecutor(max_workers=4)
    for fut in concurrent.futures.as_completed([runner.submit(bench, m) for m in models]):
        row = fut.result()
        row["at"] = time.strftime("%Y-%m-%d %H:%M")
        rows.append(row)
        print(f"{row['model']:58} code {'PASS' if row['coding_ok'] else 'fail':4} {row['coding_s']:6}s  "
              f"json {'PASS' if row['json_ok'] else 'fail':4} {row.get('json_s') or '-':>6}s  "
              f"{row['coding_note'][:40]} | {row.get('json_note', '')[:30]}", flush=True)
    # Merge into the saved results: benchmarking a few new models used to
    # overwrite the numbers for every model not in this run.
    out = BASE / "logs" / "model_benchmark.json"
    try:
        saved = json.loads(out.read_text())
        kept = [dict(r, at=r.get("at", saved.get("at"))) for r in saved.get("results", [])
                if r.get("model") not in {row["model"] for row in rows}]
    except (OSError, ValueError, AttributeError):
        kept = []
    out.write_text(json.dumps({"at": time.strftime("%Y-%m-%d %H:%M"), "results": kept + rows}, indent=2))
    print(f"\nSaved {out} ({len(rows)} new, {len(kept)} kept from earlier runs)", flush=True)
    os._exit(0)  # don't wait for abandoned (timed-out) calls


if __name__ == "__main__":
    main()
