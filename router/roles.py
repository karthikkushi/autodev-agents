"""Role -> ordered list of (litellm model id, env var, rpm) deployments.

Order comes from tools/benchmark_models.py run on this account (2026-09-30,
logs/model_benchmark.json): a coding task scored by actually running pytest on
the generated files, plus a planner-style JSON task.

  coding PASS  nvidia nemotron-3-super 5.2s · gemini-3.1-flash-lite 7.3s ·
               nvidia deepseek-v4.1-flash 12.6s · openrouter nemotron-3-super:free 16.5s ·
               openrouter nemotron-3-ultra:free 84s · nvidia kimi-k3 86s
  json PASS    gemini-3.1-flash-lite 3.1s · openrouter nemotron-3-super:free 5.4s ·
               deepseek-v4.1-flash 6.0s · nemotron-3-ultra:free 18s  (nvidia nemotron-3-super failed)
  not usable   timeouts: glm-5.3, glm-5.3-flash, gemma-4 (nvidia) · wrong format / broken tests:
               north-mini-code, gpt-oss-20b · not on this account: qwen3-coder-480b,
               devstral-2, minimax-m2.7, qwen3-next, llama-3.3-70b

Gemini 3.8 / 3.5 stay first where quality matters most: they are the strongest
when their free daily quota is available, and the router's cooldown skips them
instantly once it's used up. OpenRouter IDs must end in ":free" — anything else
bills credits. Re-run the benchmark before changing this order.
"""

ROLE_DEPLOYMENTS = {
    "fast": [
        ("gemini/gemini-3.8-flash", "GOOGLE_API_KEY", 15),
        ("gemini/gemini-3.5-flash", "GOOGLE_API_KEY", 15),
        ("gemini/gemini-3.1-flash-lite", "GOOGLE_API_KEY", 15),
        ("nvidia_nim/deepseek-ai/deepseek-v4.1-flash", "NVIDIA_API_KEY", 40),
        ("openrouter/nvidia/nemotron-3-super-120b-a12b:free", "OPENROUTER_API_KEY", 20),
    ],
    # Plans, critic, decision — reliable JSON matters here.
    "reasoning": [
        ("gemini/gemini-3.8-flash", "GOOGLE_API_KEY", 15),
        ("gemini/gemini-3.5-flash", "GOOGLE_API_KEY", 15),
        ("gemini/gemini-3.1-flash-lite", "GOOGLE_API_KEY", 15),
        ("nvidia_nim/deepseek-ai/deepseek-v4.1-flash", "NVIDIA_API_KEY", 40),
        ("openrouter/nvidia/nemotron-3-super-120b-a12b:free", "OPENROUTER_API_KEY", 20),
        ("nvidia_nim/moonshotai/kimi-k3", "NVIDIA_API_KEY", 40),
        ("openrouter/nvidia/nemotron-3-ultra-550b-a55b:free", "OPENROUTER_API_KEY", 20),
    ],
    # Nemotron-3-Super before Flash-Lite: fastest passing coder, and it keeps
    # the shared Gemini quota for vision (only Gemini can fix pages from screenshots).
    "coding": [
        ("gemini/gemini-3.8-flash", "GOOGLE_API_KEY", 15),
        ("gemini/gemini-3.5-flash", "GOOGLE_API_KEY", 15),
        ("nvidia_nim/nvidia/nemotron-3-super-120b-a12b", "NVIDIA_API_KEY", 40),
        ("gemini/gemini-3.1-flash-lite", "GOOGLE_API_KEY", 15),
        ("nvidia_nim/deepseek-ai/deepseek-v4.1-flash", "NVIDIA_API_KEY", 40),
        ("openrouter/nvidia/nemotron-3-super-120b-a12b:free", "OPENROUTER_API_KEY", 20),
        ("openrouter/nvidia/nemotron-3-ultra-550b-a55b:free", "OPENROUTER_API_KEY", 20),
    ],
    "review": [
        ("gemini/gemini-3.8-flash", "GOOGLE_API_KEY", 15),
        ("gemini/gemini-3.5-flash", "GOOGLE_API_KEY", 15),
        ("gemini/gemini-3.1-flash-lite", "GOOGLE_API_KEY", 15),
        ("nvidia_nim/deepseek-ai/deepseek-v4.1-flash", "NVIDIA_API_KEY", 40),
        ("openrouter/nvidia/nemotron-3-super-120b-a12b:free", "OPENROUTER_API_KEY", 20),
    ],
    # Screenshot critique (agents/ui_review.py). All five read images; checked
    # 2026-09-30 with a real screenshot — Gemini Flash-Lite answered in ~7s.
    "vision": [
        ("gemini/gemini-3.1-flash-lite", "GOOGLE_API_KEY", 15),
        ("gemini/gemini-3.5-flash", "GOOGLE_API_KEY", 15),
        ("gemini/gemini-3.8-flash", "GOOGLE_API_KEY", 15),
        ("nvidia_nim/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning", "NVIDIA_API_KEY", 40),
        ("nvidia_nim/meta/llama-3.2-11b-vision-instruct", "NVIDIA_API_KEY", 40),
    ],
    # Fixing a page from its screenshots needs a model that reads images AND
    # writes solid code — the Gemini Flash family. agents/ui_review.py falls
    # back to the text-only "coding" role if all of these are unavailable.
    "ui_fix": [
        ("gemini/gemini-3.8-flash", "GOOGLE_API_KEY", 15),
        ("gemini/gemini-3.5-flash", "GOOGLE_API_KEY", 15),
        ("gemini/gemini-3.1-flash-lite", "GOOGLE_API_KEY", 15),
    ],
    "review_flash": [
        ("gemini/gemini-3.8-flash", "GOOGLE_API_KEY", 15),
        ("gemini/gemini-3.5-flash", "GOOGLE_API_KEY", 15),
        ("gemini/gemini-3.1-flash-lite", "GOOGLE_API_KEY", 15),
        ("openrouter/nvidia/nemotron-3-super-120b-a12b:free", "OPENROUTER_API_KEY", 20),
        ("nvidia_nim/deepseek-ai/deepseek-v4.1-flash", "NVIDIA_API_KEY", 40),
    ],
}

# Agent -> role.
AGENT_ROLES = {
    "intake": "fast",
    "planner": "reasoning",
    "reflect": "fast",
    "researcher": "fast",
    "critic": "reasoning",
    "environment": "fast",
    "ui_designer": "reasoning",
    "coder": "coding",
    "security": "coding",
    "reviewer": "review",
    "optimizer": "coding",
    "test_writer": "coding",
    "tester": "coding",
    "decision": "reasoning",
    "doc_writer": "review",
    "final_review": "review",
}

# get_llm(role, model_override=...) — when model_override matches a key here,
# use that group instead of `role`.
MODEL_OVERRIDE_GROUPS = {
    "gemini-2.0-flash": "review_flash",
}
