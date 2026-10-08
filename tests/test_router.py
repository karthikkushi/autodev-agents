"""Router cooldowns and call events — offline, no model calls."""
import json
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pytest

import router.llm_router as r

# A real Gemini free-tier 429 (captured 2026-09-30, trimmed).
GEMINI_DAILY_429 = '''litellm.RateLimitError: litellm.RateLimitError: geminiException - {
  "error": {
    "code": 429,
    "message": "You exceeded your current quota, please check your plan and billing details. \\n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 20, model: gemini-3.8-flash\\nPlease retry in 27.328079588s.",
    "status": "RESOURCE_EXHAUSTED",
    "details": [
      {"@type": "type.googleapis.com/google.rpc.QuotaFailure",
       "violations": [{"quotaMetric": "generativelanguage.googleapis.com/generate_content_free_tier_requests",
                       "quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier",
                       "quotaDimensions": {"location": "global", "model": "gemini-3.8-flash"},
                       "quotaValue": "20"}]},
      {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "27s"}
    ]
  }
}
'''
GEMINI_MINUTE_429 = (GEMINI_DAILY_429.replace("PerDay", "PerMinute").replace('"27s"', '"34s"')
                     .replace("27.328079588s", "34.1s"))


class RateLimitError(Exception):
    status_code = 429


@pytest.fixture
def fresh(tmp_path, monkeypatch):
    """Router cooldown state isolated in a temp file."""
    monkeypatch.setattr(r, "COOLDOWN_FILE", tmp_path / "provider_cooldowns.json")
    monkeypatch.setattr(r, "PROVIDER_ERRORS_LOG", tmp_path / "provider_errors.jsonl")
    monkeypatch.setattr(r, "_cooldown_file_mtime", None)
    state = (r._cooldown_until, r._rate_backoff, r._cooldown_kind, r._cooldown_reason)
    for d in state:
        d.clear()
    yield r
    for d in state:
        d.clear()


def _forget_memory():
    """What a new process (worker restart, benchmark) starts with."""
    for d in (r._cooldown_until, r._rate_backoff, r._cooldown_kind, r._cooldown_reason):
        d.clear()
    r._cooldown_file_mtime = None


def test_daily_quota_cools_down_until_midnight_pacific(fresh):
    model = "gemini/gemini-3.8-flash"
    info = r._failure_info(RateLimitError(GEMINI_DAILY_429))
    assert info["daily"] and info["quota_value"] == "20"
    assert "GenerateRequestsPerDayPerProjectPerModel-FreeTier" in info["detail"]
    assert "retryDelay=27s" in info["detail"] and len(info["detail"]) <= 200
    r._start_cooldown(model, RateLimitError(GEMINI_DAILY_429))
    pacific = ZoneInfo("America/Los_Angeles")
    tomorrow = datetime.now(pacific).date().toordinal() + 1
    midnight = datetime.combine(datetime.fromordinal(tomorrow).date(), datetime.min.time(), tzinfo=pacific)
    assert abs(r._cooldown_until[model] - (midnight.timestamp() + 60)) < 5
    assert r._cooldown_kind[model] == "daily"


def test_daily_reset_follows_daylight_saving():
    # 2026-03-08 10:00 UTC is 03:00 PDT, just after the clocks went forward:
    # the next reset is 2026-03-09 00:00 PDT = 07:00 UTC, not a fixed -08:00.
    now = datetime(2026, 3, 8, 10, 0, tzinfo=timezone.utc).timestamp()
    reset = r._next_daily_reset("gemini/gemini-3.5-flash", now)
    assert reset == datetime(2026, 3, 9, 7, 0, tzinfo=timezone.utc).timestamp() + 60


def test_retry_delay_is_used_as_given(fresh):
    info = r._failure_info(RateLimitError(GEMINI_MINUTE_429))
    assert not info["daily"] and info["wait"] == 34
    with r._cooldown_lock:
        seconds, kind, _ = r._cooldown_plan("gemini/gemini-3.5-flash", RateLimitError(GEMINI_MINUTE_429), info)
    assert kind == "retry" and 34 <= seconds <= 40


def test_retry_after_header_and_plain_errors(fresh):
    class Err(Exception):
        status_code = 429
        litellm_response_headers = {"Retry-After": "12"}
    info = r._failure_info(Err("litellm.RateLimitError: GroqException - too many requests"))
    assert info["wait"] == 12 and "Retry-After=12" in info["detail"]
    plain = r._failure_info(TimeoutError("no reply within 180s"))
    assert plain["wait"] is None and plain["detail"] == "no reply within 180s"


def test_payment_required_rests_like_a_missing_model(fresh):
    class APIError(Exception):
        status_code = 402
    err = APIError("litellm.APIError: APIError: SambanovaException - A payment method is required. "
                   "Add one at https://cloud.sambanova.ai/plans/billing to continue.")
    assert r._failure_info(err)["detail"].startswith("402 · A payment method is required")
    assert r._cooldown_seconds("sambanova/x", err) == r._COOLDOWN["permanent"]


def test_cooldowns_survive_a_reload_from_the_file(fresh):
    model = "gemini/gemini-3.5-flash"
    r._start_cooldown(model, RateLimitError(GEMINI_DAILY_429))
    until = r._cooldown_until[model]
    _forget_memory()
    with r._cooldown_lock:
        r._load_cooldowns_locked()
    assert r._cooldown_until[model] == pytest.approx(until, abs=0.2)
    assert r._cooldown_kind[model] == "daily" and "daily quota" in r._cooldown_reason[model]
    # The backoff ladder survives too (it restarted at "1 min" on every restart).
    r._start_cooldown("nvidia_nim/x", RateLimitError("litellm.RateLimitError: slow down"))
    r._start_cooldown("nvidia_nim/x", RateLimitError("litellm.RateLimitError: slow down"))
    _forget_memory()
    with r._cooldown_lock:
        r._load_cooldowns_locked()
    assert r._rate_backoff["nvidia_nim/x"] == 120


def test_corrupt_and_expired_files_are_ignored(fresh):
    r.COOLDOWN_FILE.write_text("{not json")
    with r._cooldown_lock:
        r._load_cooldowns_locked()          # no exception, nothing loaded
    assert r._cooldown_until == {}
    old = time.time() - r.EXPIRED_AFTER - 60
    r.COOLDOWN_FILE.write_text(json.dumps({"models": {
        "m/old": {"until": old, "kind": "backoff"},
        "m/new": {"until": time.time() + 60, "kind": "backoff", "backoff": 60}}}))
    _forget_memory()
    with r._cooldown_lock:
        r._load_cooldowns_locked()
    assert list(r._cooldown_until) == ["m/new"]


def test_one_probe_at_a_time(fresh):
    model = "nvidia_nim/y"
    r._cooldown_until[model], r._cooldown_kind[model] = time.time() - 1, "backoff"   # cooldown just ended
    assert r._claim(model) is True                 # first caller probes
    assert r._cooldown_kind[model] == "probe"
    assert r._claim(model) is False                # second caller treats it as cooling
    assert r._claim(model, last_resort=True) is False
    assert r._claim("nvidia_nim/healthy") is True  # models with no cooldown aren't gated
    r._clear_cooldown(model)                       # the probe succeeded
    assert r._claim(model) is True


def test_every_attempt_of_a_call_shares_a_call_id(fresh, monkeypatch):
    events = []
    monkeypatch.setattr(r, "llm_event", events.append)

    def fake_completion(model, **_):
        if "first" in model:
            raise RateLimitError(GEMINI_MINUTE_429)
        msg = type("M", (), {"content": "ok"})()
        return type("R", (), {"choices": [type("C", (), {"message": msg, "finish_reason": "stop"})()],
                              "usage": None})()
    monkeypatch.setattr(r.litellm, "completion", fake_completion)
    llm = r.RoutedLLM.__new__(r.RoutedLLM)
    llm.role, llm.temperature, llm.max_tokens = "fast", 0, 100
    llm._deployments = [{"model": "nvidia_nim/first", "api_key": "k"}, {"model": "nvidia_nim/second", "api_key": "k"}]
    assert llm._call_providers([{"role": "user", "content": "hi"}]).content == "ok"
    assert [e["ok"] for e in events] == [False, True]
    assert events[0]["call_id"] == events[1]["call_id"]
    assert [e["final"] for e in events] == [False, True]
    assert "PerMinute" in events[0]["error_detail"]
    # The failed model is now resting on the provider's own retry delay.
    assert r._cooldown_kind["nvidia_nim/first"] == "retry"

    events.clear()
    r._cooldown_until.clear()
    monkeypatch.setattr(r.litellm, "completion", lambda **_: (_ for _ in ()).throw(RateLimitError("busy")))
    with pytest.raises(r.NoProviderAvailableError):
        llm._call_providers([{"role": "user", "content": "hi"}])
    assert [e["final"] for e in events] == [False, True] and not any(e["ok"] for e in events)


def test_providers_that_need_more_than_a_key(monkeypatch):
    monkeypatch.setattr(r, "ROLE_DEPLOYMENTS", {"x": [
        ("cloudflare/@cf/openai/gpt-oss-120b", "CLOUDFLARE_API_KEY", 0),
        ("ollama_chat/gpt-oss:120b", "OLLAMA_API_KEY", 0),
        ("groq/openai/gpt-oss-120b", "GROQ_API_KEY", 0)]})
    for k, v in {"CLOUDFLARE_API_KEY": "cf", "OLLAMA_API_KEY": "ol", "GROQ_API_KEY": "gq"}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("CLOUDFLARE_ACCOUNT_ID", raising=False)
    deps = r.RoutedLLM._load_deployments("x")
    # Cloudflare without its account id is skipped like a missing key.
    assert [d["model"] for d in deps] == ["ollama_chat/gpt-oss:120b", "groq/openai/gpt-oss-120b"]
    assert deps[0]["api_base"] == "https://ollama.com" and "api_base" not in deps[1]
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "abc123")
    cf = r.RoutedLLM._load_deployments("x")[0]
    assert cf["api_base"] == "https://api.cloudflare.com/client/v4/accounts/abc123/ai/v1"


def test_thinking_switches_per_provider():
    assert r._no_thinking_params("groq/qwen/qwen3.8-27b") == {"extra_body": {"reasoning_effort": "none"}}
    assert r._no_thinking_params("groq/openai/gpt-oss-120b")["extra_body"]["reasoning_effort"] == "low"
    assert r._no_thinking_params("zai/glm-4.7-flash") == {"extra_body": {"thinking": {"type": "disabled"}}}
    assert r._no_thinking_params("gemini/gemini-3.5-flash") == {}
