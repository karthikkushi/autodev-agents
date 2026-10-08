"""Our own try-each-provider loop on top of plain litellm.completion().

LiteLLM's Router can fall back between deployments too; this exists for the
things the pipeline needs in one place:
- per-model cooldowns shared by every role (a model that used up its daily
  quota in "coding" is skipped by "fast" and "review" as well), set from what
  the provider actually said (daily quota, retry delay) and saved to
  logs/provider_cooldowns.json so worker restarts, the server and the
  benchmark all see them;
- a hard wall-clock limit per call (litellm's timeout= didn't bound it);
- per-provider settings: thinking switched off for NVIDIA/OpenRouter
  reasoning models, Gemini 3 left at its default temperature;
- waiting out an outage (every free quota used up) instead of failing a
  multi-day run.
Any exception from one provider moves the call to the next one in the
role's deployment list.
"""

import email.utils
import json
import math
import os
import re
import time
import threading
import uuid
import concurrent.futures
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
import litellm

from router.roles import ROLE_DEPLOYMENTS, AGENT_ROLES, MODEL_OVERRIDE_GROUPS
from pipeline import control
from server.bridge import llm_event, log as llm_log, state as llm_state

_PROVIDER_NAMES = {"nvidia_nim": "nvidia", "ollama_chat": "ollama"}


def _provider_of(model: str) -> str:
    """The provider name the dashboard uses ("nvidia_nim/x" -> "nvidia")."""
    prefix = model.split("/", 1)[0]
    return _PROVIDER_NAMES.get(prefix, prefix)

# litellm's timeout= didn't bound the total wait: an NVIDIA call sat on an
# open connection for 5+ minutes and froze the whole pipeline. Each call now
# runs in a pool thread with a hard wall-clock limit; on timeout the router
# moves to the next provider (the abandoned request finishes or dies on its own).
# 180s leaves room for long legitimate replies (8192-token code took ~107s).
HARD_CALL_TIMEOUT = 180
_CALL_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=16, thread_name_prefix="llm-call")

# Per-model cooldown after a failure. Without it every call spent two instant
# 429s on the daily-exhausted Gemini models before reaching one that works —
# hundreds of wasted requests a day, and a 180s wait on every call to a model
# that keeps timing out. Rate limits back off 60s → 30 min; permanent errors
# (404 / no access) sit out an hour.
_COOLDOWN = {"rate": 60, "rate_max": 1800, "timeout": 600, "permanent": 3600, "other": 300}
OUTAGE_BACKOFF = [120, 300, 600, 1200, 1800]   # seconds between retries when every provider fails
PROVIDER_OUTAGE_MAX_WAIT = 6 * 3600             # then give up and let the project fail visibly


def _sleep_unless_paused(seconds: int) -> None:
    """Sleep, but in short steps so a Pause from the dashboard still applies."""
    end = time.time() + seconds
    while time.time() < end:
        time.sleep(min(10, max(0, end - time.time())))
        if control.is_paused():
            return


LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"
PROVIDER_ERRORS_LOG = LOGS_DIR / "provider_errors.jsonl"
PROVIDER_ERRORS_MAX_BYTES = 2_000_000

# Anything that looks like a credential, in case a provider echoes the request
# URL (Gemini keys can travel as ?key=...) or a header back in its error text.
_SECRET_PATTERNS = [
    (re.compile(r"(?i)(\bkey=)[^&\s\"']+"), r"\1***"),
    (re.compile(r"(?i)(bearer\s+)[\w\-.~+/]+=*"), r"\1***"),
    (re.compile(r"\b(AIza[\w\-]{20,}|nvapi-[\w\-]{10,}|sk-[\w\-]{16,}|gsk_\w{16,}|csk-\w{16,}"
                r"|github_pat_\w{20,}|gh[pousr]_\w{20,})"), "***"),
]


def _redact(text: str) -> str:
    """Strip API keys from error text before it is logged or shown."""
    for name, value in os.environ.items():
        if value and len(value) >= 12 and name.endswith(("_API_KEY", "_TOKEN", "_KEY")):
            text = text.replace(value, "***")
    for pattern, replacement in _SECRET_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def _error_headers(error: Exception) -> dict:
    """Response headers LiteLLM kept on the exception (lower-case names)."""
    for source in (getattr(error, "litellm_response_headers", None),
                   getattr(getattr(error, "response", None), "headers", None),
                   getattr(error, "headers", None)):
        try:
            if source:
                return {str(k).lower(): str(v) for k, v in dict(source).items()}
        except Exception:
            continue
    return {}


def _log_provider_error(model: str, role: str, error: Exception) -> None:
    """Keep the full error text of every failed attempt. The console line only
    ever showed LiteLLM's first 120 characters, which never said whether a
    429 was a per-minute or a per-day quota."""
    entry = {"t": time.time(), "time": time.strftime("%Y-%m-%d %H:%M:%S"), "model": model, "role": role,
             "type": type(error).__name__, "status": getattr(error, "status_code", None),
             "headers": _error_headers(error), "error": str(error)[:8000]}
    try:
        LOGS_DIR.mkdir(exist_ok=True)
        if PROVIDER_ERRORS_LOG.exists() and PROVIDER_ERRORS_LOG.stat().st_size > PROVIDER_ERRORS_MAX_BYTES:
            PROVIDER_ERRORS_LOG.replace(PROVIDER_ERRORS_LOG.with_suffix(".jsonl.1"))
        with open(PROVIDER_ERRORS_LOG, "a") as f:
            f.write(_redact(json.dumps(entry)) + "\n")
    except OSError:
        pass


def _json_error_body(text: str) -> dict:
    """The provider's JSON {"error": {...}} inside LiteLLM's message, or {}."""
    decoder = json.JSONDecoder()
    text = text[:20000]
    for m in re.finditer(r"\{", text):
        try:
            body, _ = decoder.raw_decode(text, m.start())
        except ValueError:
            continue
        if isinstance(body, dict) and isinstance(body.get("error"), dict):
            return body["error"]
    return {}


_DURATION = re.compile(r"(\d+(?:\.\d+)?)(ms|h|m|s)")


def _duration_seconds(value) -> float | None:
    """'27s', '26.8s', '250ms', '2m59.56s', '1h2m' or a bare number of seconds."""
    if value is None:
        return None
    text = str(value).strip().lower()
    try:
        return float(text)
    except ValueError:
        pass
    parts = _DURATION.findall(text)
    if not parts or _DURATION.sub("", text).strip():
        return None
    scale = {"ms": 0.001, "s": 1, "m": 60, "h": 3600}
    return sum(float(n) * scale[unit] for n, unit in parts)


def _reset_wait_seconds(value, now: float) -> float | None:
    """Seconds until a Retry-After (seconds or an HTTP date) or
    X-RateLimit-Reset (epoch seconds/milliseconds or a duration)."""
    if value is None:
        return None
    text = str(value).strip()
    try:
        n = float(text)
    except ValueError:
        try:
            return max(0.0, email.utils.parsedate_to_datetime(text).timestamp() - now)
        except (TypeError, ValueError, IndexError):
            return _duration_seconds(text)
    if n > 1e12:  # epoch milliseconds (OpenRouter's free-model limits)
        return max(0.0, n / 1000 - now)
    if n > 1e9:   # epoch seconds
        return max(0.0, n - now)
    return n


def _failure_info(error: Exception) -> dict:
    """What a failed attempt really said, read from the provider's own error.

    Gemini's 429 body carries a QuotaFailure (quotaId, e.g.
    GenerateRequestsPerDayPerProjectPerModel-FreeTier, and quotaValue, the
    limit) and a RetryInfo (retryDelay). The retryDelay is only seconds even
    when the *daily* quota is gone, which is why the old backoff kept retrying
    models that could not answer until midnight Pacific. Other providers give
    a status, a message and sometimes Retry-After / X-RateLimit-Reset
    (OpenRouter puts the latter inside the body's metadata.headers).

    "detail" is a short one-line summary for the console and the dashboard."""
    text = str(error)
    body = _json_error_body(text)
    headers = _error_headers(error)
    meta = body.get("metadata") if isinstance(body.get("metadata"), dict) else {}
    meta_headers = {str(k).lower(): v for k, v in (meta.get("headers") or {}).items()} \
        if isinstance(meta.get("headers"), dict) else {}
    status = body.get("code") if isinstance(body.get("code"), int) else getattr(error, "status_code", None)
    quota_ids, quota_value, retry_delay = [], None, None
    for item in body.get("details") or []:
        kind = str(item.get("@type", "")) if isinstance(item, dict) else ""
        if kind.endswith("QuotaFailure"):
            for v in item.get("violations") or []:
                if isinstance(v, dict) and v.get("quotaId"):
                    quota_ids.append(str(v["quotaId"]))
                    quota_value = v.get("quotaValue", quota_value)
        elif kind.endswith("RetryInfo"):
            retry_delay = item.get("retryDelay")
    if not body:
        # LiteLLM cut or re-quoted the body: read the same fields from the text.
        quota_ids = re.findall(r'"quotaId"\s*:\s*"([^"]+)"', text)
        m = re.search(r'"quotaValue"\s*:\s*"?(\d+)', text)
        quota_value = m.group(1) if m else None
        m = re.search(r'"retryDelay"\s*:\s*"([^"]+)"', text) or re.search(r"retry in (\d+(?:\.\d+)?s)", text)
        retry_delay = m.group(1) if m else None
        m = re.search(r'"code"\s*:\s*(\d{3})\b', text)
        if m and not status:
            status = int(m.group(1))
    retry_after = headers.get("retry-after") or meta_headers.get("retry-after")
    reset = headers.get("x-ratelimit-reset") or meta_headers.get("x-ratelimit-reset")
    now = time.time()
    wait = _duration_seconds(retry_delay)
    if wait is None:
        wait = _reset_wait_seconds(retry_after, now)
    if wait is None:
        wait = _reset_wait_seconds(reset, now)

    # "litellm.APIError: APIError: SambanovaException - A payment method is required..."
    message = body.get("message") or re.sub(r"^\w+Exception\s*-\s*", "",
                                            re.sub(r"^(?:(?:litellm\.)?\w+(?:Error|Exception):\s*)+", "", text))
    parts = [str(status)] if status else []
    if quota_ids:
        parts.append(quota_ids[0] + (f" quotaValue={quota_value}" if quota_value else ""))
    else:
        parts.append(" ".join(str(message).split())[:120])
    if retry_delay:
        parts.append(f"retryDelay={retry_delay}")
    if retry_after:
        parts.append(f"Retry-After={retry_after}")
    if reset:
        parts.append(f"X-RateLimit-Reset={reset}")
    return {"status": status, "quota_ids": quota_ids, "quota_value": quota_value,
            "daily": any("perday" in q.lower() for q in quota_ids), "wait": wait,
            "detail": _redact(" · ".join(p for p in parts if p))[:200]}


def _human(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 90:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{round(seconds / 60)} min"
    return f"{seconds // 3600} h {seconds % 3600 // 60} min"


# ── Cooldowns ────────────────────────────────────────────────────────────────
# Kept in logs/provider_cooldowns.json as well as in memory. In memory only,
# every worker restart forgot them (the backoff restarted at "1 min" 8 times
# in one day) and the benchmark never knew about them.
COOLDOWN_FILE = LOGS_DIR / "provider_cooldowns.json"
EXPIRED_AFTER = _COOLDOWN["rate_max"]   # entries this long past their end are dropped (backoff is stale)
RETRY_MARGIN = 3                         # seconds added to a provider's own "retry after"
# Free-tier requests-per-day reset at midnight in the provider's own zone.
DAILY_RESET_TZ = {"gemini": "America/Los_Angeles"}
# Cooldowns that come from a fact rather than a guess: a daily quota, the
# provider's own retry delay, or another caller's probe. Even the "every model
# is cooling, try them anyway" last resort skips these — the call would fail.
HARD_KINDS = ("daily", "retry", "probe")

_cooldown_until: dict[str, float] = {}
_rate_backoff: dict[str, int] = {}
_cooldown_kind: dict[str, str] = {}
_cooldown_reason: dict[str, str] = {}
_cooldown_lock = threading.Lock()
_cooldown_file_mtime = None


def _load_cooldowns_locked() -> None:
    """Pick up cooldowns another process (worker, server, benchmark) saved.
    Re-reads only when the file changed; a corrupt or half-written file is
    ignored. Call with _cooldown_lock held."""
    global _cooldown_file_mtime
    try:
        mtime = COOLDOWN_FILE.stat().st_mtime_ns
    except OSError:
        return
    if mtime == _cooldown_file_mtime:
        return
    _cooldown_file_mtime = mtime  # a corrupt file isn't re-read on every call; the next save replaces it
    try:
        models = json.loads(COOLDOWN_FILE.read_text()).get("models")
        if not isinstance(models, dict):
            raise ValueError("no models")
        now = time.time()
        loaded = {}
        for model, entry in models.items():
            until = float(entry["until"])
            if until + EXPIRED_AFTER >= now:
                loaded[model] = (until, int(entry.get("backoff") or 0), str(entry.get("kind") or "backoff"),
                                 str(entry.get("reason") or ""))
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return
    for d in (_cooldown_until, _rate_backoff, _cooldown_kind, _cooldown_reason):
        d.clear()
    for model, (until, backoff, kind, reason) in loaded.items():
        _cooldown_until[model], _cooldown_kind[model], _cooldown_reason[model] = until, kind, reason
        if backoff:
            _rate_backoff[model] = backoff


def _save_cooldowns_locked() -> None:
    """Write every live cooldown to the file (temp file + rename, so a reader
    never sees half of it). Call with _cooldown_lock held."""
    global _cooldown_file_mtime
    now = time.time()
    models = {m: {"until": round(until, 1), "until_local": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(until)),
                  "backoff": _rate_backoff.get(m, 0), "kind": _cooldown_kind.get(m, "backoff"),
                  "reason": _cooldown_reason.get(m, ""),
                  # Lets the dashboard hide models only the benchmark tried.
                  "in_roles": any(m == d[0] for deps in ROLE_DEPLOYMENTS.values() for d in deps)}
              for m, until in _cooldown_until.items() if until + EXPIRED_AFTER >= now}
    tmp = COOLDOWN_FILE.with_name(f".{COOLDOWN_FILE.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        COOLDOWN_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps({"updated": round(now, 1), "models": models}, indent=1))
        os.replace(tmp, COOLDOWN_FILE)
        _cooldown_file_mtime = COOLDOWN_FILE.stat().st_mtime_ns
    except OSError:
        tmp.unlink(missing_ok=True)


def _next_daily_reset(model: str, now: float) -> float | None:
    """When the model's provider resets its per-day quota (midnight in its
    zone, DST included), plus a minute for clock differences."""
    try:
        tz = ZoneInfo(DAILY_RESET_TZ.get(model.split("/", 1)[0], "UTC"))
    except Exception:  # no time-zone database: use the normal backoff instead
        return None
    local = datetime.fromtimestamp(now, tz)
    midnight = datetime.combine(local.date() + timedelta(days=1), datetime.min.time(), tzinfo=tz)
    return midnight.timestamp() + 60


def _cooldown_seconds(model: str, error: Exception) -> int:
    """The backoff guess for failures that don't say how long to wait."""
    name = type(error).__name__
    # LiteLLM sometimes wraps Gemini's quota 429 as a BadRequestError.
    text = str(error)[:400]
    if name == "RateLimitError" or '"code": 429' in text or "RESOURCE_EXHAUSTED" in text:
        seconds = min(_COOLDOWN["rate_max"], _rate_backoff.get(model, _COOLDOWN["rate"] // 2) * 2)
        _rate_backoff[model] = seconds
        return seconds
    # 401/402/403/404 don't fix themselves in minutes: bad key, "a payment
    # method is required" (SambaNova), no access, model gone.
    if name in ("NotFoundError", "AuthenticationError", "PermissionDeniedError") \
            or getattr(error, "status_code", None) in (401, 402, 403, 404):
        return _COOLDOWN["permanent"]
    if name in ("TimeoutError", "Timeout"):
        return _COOLDOWN["timeout"]
    if name == "ServiceUnavailableError":
        return _COOLDOWN["rate"]
    return _COOLDOWN["other"]


def _cooldown_plan(model: str, error: Exception, info: dict) -> tuple[int, str, str]:
    """(seconds, kind, reason): until the daily reset when the daily quota is
    gone, exactly the provider's own retry delay when it gave one, otherwise
    the backoff guess. Call with _cooldown_lock held (the backoff is shared)."""
    now = time.time()
    if info["daily"]:
        reset = _next_daily_reset(model, now)
        if reset:
            limit = f" ({info['quota_value']}/day)" if info.get("quota_value") else ""
            return (math.ceil(reset - now), "daily",
                    f"daily quota used up{limit} — resets {time.strftime('%H:%M', time.localtime(reset))}")
    if info["wait"] is not None:
        return (int(min(info["wait"], 86400)) + RETRY_MARGIN, "retry",
                f"provider asked to wait {_human(info['wait'])}")
    seconds = _cooldown_seconds(model, error)
    return seconds, "backoff", f"{type(error).__name__} — backing off {_human(seconds)}"


def _start_cooldown(model: str, error: Exception, info: dict | None = None) -> None:
    info = info or _failure_info(error)
    with _cooldown_lock:
        _load_cooldowns_locked()
        seconds, kind, reason = _cooldown_plan(model, error, info)
        _cooldown_until[model] = time.time() + seconds
        _cooldown_kind[model], _cooldown_reason[model] = kind, reason
        _save_cooldowns_locked()
    print(f"[router] {model} resting {_human(seconds)} — {reason}")


def _clear_cooldown(model: str) -> None:
    with _cooldown_lock:
        _load_cooldowns_locked()
        if model not in _cooldown_until and model not in _rate_backoff:
            return  # healthy already: no file write on every successful call
        for d in (_cooldown_until, _rate_backoff, _cooldown_kind, _cooldown_reason):
            d.pop(model, None)
        _save_cooldowns_locked()


def _claim(model: str, last_resort: bool = False) -> bool:
    """May this caller try `model` now?

    Healthy models (no cooldown on record): always. A model whose cooldown has
    just ended gets one probe at a time: the first caller holds it for
    HARD_CALL_TIMEOUT and everyone else treats it as still cooling until that
    probe succeeds (_clear_cooldown) or fails (_start_cooldown) — parallel
    branches (research x3, security + optimizer) used to hit the same
    rate-limited model within the same second. A model still cooling is only
    tried as a last resort, and never under a HARD_KINDS cooldown."""
    with _cooldown_lock:
        _load_cooldowns_locked()
        until = _cooldown_until.get(model)
        if until is None:
            return True
        now = time.time()
        if until <= now:
            _cooldown_until[model] = now + HARD_CALL_TIMEOUT
            _cooldown_reason[model] = f"probe after: {_cooldown_reason.get(model, '')}"[:200]
            _cooldown_kind[model] = "probe"
            _save_cooldowns_locked()
            return True
        return last_resort and _cooldown_kind.get(model) not in HARD_KINDS


def _ready_order(deployments: list) -> tuple[list, bool]:
    """(deployments, last_resort). Normally the ones not cooling down, in the
    role's order. If every one is cooling down, all of them, soonest-to-recover
    first, with last_resort=True so _claim can still let the soft ones through."""
    with _cooldown_lock:
        _load_cooldowns_locked()
        now = time.time()
        ready = [d for d in deployments if _cooldown_until.get(d["model"], 0) <= now]
        if ready:
            return ready, False
        return sorted(deployments, key=lambda d: _cooldown_until.get(d["model"], 0)), True


def resting_reason(model: str) -> str:
    """Why `model` is resting on a known daily quota or retry delay, or "".
    For callers outside the router (the benchmark) that shouldn't spend a
    request on a call that can only fail."""
    with _cooldown_lock:
        _load_cooldowns_locked()
        if _cooldown_until.get(model, 0) > time.time() and _cooldown_kind.get(model) in ("daily", "retry"):
            return _cooldown_reason.get(model) or "resting"
    return ""


with _cooldown_lock:
    _load_cooldowns_locked()

# Suppress litellm's verbose success/failure logging
litellm.suppress_debug_info = True

DEFAULT_MAX_TOKENS = 4000
# Coding-role replies are whole files wrapped in JSON; 4000 tokens cuts them
# off mid-string and json.loads fails. A provider that rejects 8192 just
# fails over to the next one in the role's list.
ROLE_MAX_TOKENS = {"coding": 8192, "ui_fix": 8192}

class NoProviderAvailableError(Exception):
    pass


def _normalize_messages(messages):
    if isinstance(messages, str):
        return [{"role": "user", "content": messages}]
    normalized = []
    for m in messages:
        if isinstance(m, dict):
            normalized.append(m)
        else:
            normalized.append({"role": "user", "content": getattr(m, "content", str(m))})
    return normalized


def _no_thinking_params(model: str) -> dict:
    """Switch off built-in chain-of-thought for reasoning models (Nemotron,
    DeepSeek). On long prompts the thinking runs past max_tokens and lands in
    the reply instead of the answer — the planner got pages of "We need to
    extract..." prose and no JSON task list. Thinking tokens also eat the
    free NVIDIA budget. Each provider has its own switch (checked 2026-09-30)."""
    if model.startswith("nvidia_nim/"):
        return {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
    if model.startswith("openrouter/"):
        return {"extra_body": {"reasoning": {"enabled": False}}}
    if model.startswith(("groq/", "cerebras/", "sambanova/")):
        # gpt-oss can't switch reasoning off, only down to "low" (Groq,
        # Cerebras docs); Qwen accepts "none"; SambaNova's DeepSeek thinks
        # only when asked to.
        if "gpt-oss" in model:
            extra = {"reasoning_effort": "low"}
            if model.startswith("groq/"):
                extra["include_reasoning"] = False
            return {"extra_body": extra}
        if "qwen" in model.lower():
            return {"extra_body": {"reasoning_effort": "none"}}
        if model.startswith("sambanova/") and "deepseek" in model.lower():
            return {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
    if model.startswith("zai/"):  # GLM-4.5 and later (docs.z.ai core parameters)
        return {"extra_body": {"thinking": {"type": "disabled"}}}
    return {}


def _provider_params(model: str) -> dict | None:
    """What a provider needs besides its API key, or None if that setting is
    missing — the deployment is then skipped like one without a key (a
    Cloudflare call without the account id can only fail)."""
    prefix = model.split("/", 1)[0]
    if prefix == "cloudflare":
        account = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
        return {"api_base": f"https://api.cloudflare.com/client/v4/accounts/{account}/ai/v1"} if account else None
    if prefix == "ollama_chat":  # Ollama Cloud, not a local Ollama server
        return {"api_base": "https://ollama.com"}
    return {}


def _sampling_params(model: str, temperature: float) -> dict:
    """Gemini 3 models run at their default temperature (1.0): Google's
    Gemini 3 guide warns that lower values can cause looping and weaker
    reasoning, and LiteLLM warns the parameter is being removed for them.
    Everyone else gets the temperature the agent asked for."""
    if "gemini-3" in model:
        return {}
    return {"temperature": temperature}


CUT_OFF_DIR = Path(__file__).resolve().parent.parent / "logs" / "cut_off_replies"
CUT_OFF_KEEP = 20


def _finish_reason(resp) -> str:
    try:
        return resp.choices[0].finish_reason or ""
    except (AttributeError, IndexError, TypeError):
        return ""


def _keep_cut_off_reply(model: str, role: str, content: str) -> None:
    """Save a reply that hit max_tokens, so a runaway answer can be looked at
    afterwards (one filled 8192 tokens and wrote a placeholder instead of a
    test file, and nothing was left to show why). Keeps the latest 20."""
    try:
        CUT_OFF_DIR.mkdir(parents=True, exist_ok=True)
        name = f"{time.strftime('%Y%m%d-%H%M%S')}_{role}_{model.rsplit('/', 1)[-1]}.txt"
        (CUT_OFF_DIR / name).write_text(content)
        for old in sorted(CUT_OFF_DIR.glob("*.txt"))[:-CUT_OFF_KEEP]:
            old.unlink(missing_ok=True)
    except OSError:
        pass


class _Response:
    def __init__(self, content: str):
        self.content = content


class RoutedLLM:
    """Tries each deployment in the role's list in order.
    Any exception — 404, 402, 429, timeout, auth — moves to the next provider.
    Only raises NoProviderAvailableError if every provider fails.
    """
    def __init__(self, role: str, temperature: float = 0, max_tokens: int = DEFAULT_MAX_TOKENS):
        self.role = role
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._deployments = self._load_deployments(role)

    @staticmethod
    def _load_deployments(role: str) -> list:
        deps = []
        for model_id, env_key, _rpm in ROLE_DEPLOYMENTS.get(role, []):
            api_key = os.environ.get(env_key)
            extra = _provider_params(model_id)
            if api_key and extra is not None:
                deps.append({"model": model_id, "api_key": api_key, **extra})
        return deps

    def invoke(self, messages):
        msgs = _normalize_messages(messages)
        outage_started, attempt = None, 0
        while True:
            control.wait_if_paused(f"{self.role} call")
            try:
                result = self._call_providers(msgs)
                if outage_started:
                    llm_log("system", f"▶️ AI providers are answering again — continuing ({self.role})")
                return result
            except NoProviderAvailableError:
                # Offline: wait_if_paused() above waits for the connection.
                if not control.is_online(force=True):
                    continue
                # Online but every provider failed — usually free quotas used
                # up for the day. They reset, so a multi-day run waits and
                # retries instead of marking the whole project failed.
                outage_started = outage_started or time.time()
                if time.time() - outage_started > PROVIDER_OUTAGE_MAX_WAIT:
                    raise
                wait = OUTAGE_BACKOFF[min(attempt, len(OUTAGE_BACKOFF) - 1)]
                attempt += 1
                msg = (f"⏳ Every AI provider for '{self.role}' is failing (rate limits?) — "
                       f"retrying in {wait // 60} min")
                print(f"[router] {msg}")
                llm_log("system", msg)
                if attempt == 1:
                    control.notify("AutoDev — waiting for AI providers",
                                   f"All providers for {self.role} are failing; retrying automatically.", "hourglass")
                _sleep_unless_paused(wait)

    def _event(self, dep: dict, ok: bool, started: float, call_id: str, resp=None,
               error: Exception = None, detail: str = "") -> dict:
        usage = getattr(resp, "usage", None)
        return {
            "provider": _provider_of(dep["model"]),
            "model": dep["model"].split("/", 1)[-1],
            "role": self.role,
            "ok": ok,
            "latency_ms": int((time.time() - started) * 1000),
            "tokens_in": getattr(usage, "prompt_tokens", 0) or 0,
            "tokens_out": getattr(usage, "completion_tokens", 0) or 0,
            "error": type(error).__name__ if error else "",
            "error_detail": detail,
            "cut_off": _finish_reason(resp) == "length",
            "call_id": call_id,
        }

    def _call_providers(self, msgs):
        # Every attempt of this call shares call_id, and the attempt that ends
        # the call (the answer, or the last failure) is marked final — so the
        # dashboard can tell a bounce to the next provider (the call was still
        # answered) from a call nobody answered. A failed attempt is held back
        # until we know which of the two it was.
        call_id, last_error, pending = uuid.uuid4().hex[:12], None, None

        def settle(final: bool) -> None:
            nonlocal pending
            if pending:
                llm_event({**pending, "final": final})
                pending = None

        ordered, last_resort = _ready_order(self._deployments)
        for dep in ordered:
            if not _claim(dep["model"], last_resort):
                continue
            # 503 (overloaded) gets one retry; all other errors move immediately to next provider
            attempts = 2 if "gemini" in dep["model"] else 1
            for attempt in range(attempts):
                started = time.time()
                # The dashboard's "Now working" tile shows which model the agent
                # is waiting on; the attempt's result ends it (same call_id).
                llm_state({"llm_active": {"model": dep["model"].split("/", 1)[-1], "provider": _provider_of(dep["model"]),
                                          "role": self.role, "call_id": call_id, "since": started}})
                try:
                    future = _CALL_POOL.submit(
                        litellm.completion,
                        model=dep["model"],
                        messages=msgs,
                        api_key=dep["api_key"],
                        **{k: v for k, v in dep.items() if k == "api_base"},
                        **_sampling_params(dep["model"], self.temperature),
                        timeout=120,
                        max_tokens=self.max_tokens,
                        **_no_thinking_params(dep["model"]),
                    )
                    try:
                        resp = future.result(timeout=HARD_CALL_TIMEOUT)
                    except concurrent.futures.TimeoutError:
                        future.cancel()
                        raise TimeoutError(f"no reply within {HARD_CALL_TIMEOUT}s")
                    content = resp.choices[0].message.content or ""
                    if not content.strip():
                        raise ValueError("Empty response content from model")
                    settle(False)
                    llm_event({**self._event(dep, True, started, call_id, resp=resp), "final": True})
                    if _finish_reason(resp) == "length":
                        _keep_cut_off_reply(dep["model"], self.role, content)
                    _clear_cooldown(dep["model"])
                    return _Response(content)
                except litellm.ServiceUnavailableError as e:
                    last_error, info = e, _failure_info(e)
                    settle(False)
                    pending = self._event(dep, False, started, call_id, error=e, detail=info["detail"])
                    _log_provider_error(dep["model"], self.role, e)
                    if attempt < attempts - 1:
                        print(f"[router] {dep['model']} 503, retrying...")
                        time.sleep(3)
                    else:
                        print(f"[router] {dep['model']} failed ({type(e).__name__}: {info['detail']}), trying next...")
                        _start_cooldown(dep["model"], e, info)
                except Exception as e:
                    last_error, info = e, _failure_info(e)
                    settle(False)
                    pending = self._event(dep, False, started, call_id, error=e, detail=info["detail"])
                    _log_provider_error(dep["model"], self.role, e)
                    print(f"[router] {dep['model']} failed ({type(e).__name__}: {info['detail']}), trying next...")
                    if isinstance(e, ValueError):
                        # An empty reply is a one-off, not a rate limit: the
                        # model answered, so a probe it was holding is over.
                        _clear_cooldown(dep["model"])
                    else:
                        _start_cooldown(dep["model"], e, info)
                    break  # non-503 errors move to next provider immediately
        if pending:
            settle(True)
        else:
            # Nothing was tried: every provider is resting on a known quota or
            # a probe. Still one call nobody answered, so it has to show up.
            llm_event({"provider": "none", "model": "", "role": self.role, "ok": False, "latency_ms": 0,
                       "tokens_in": 0, "tokens_out": 0, "error": "NoProviderReady",
                       "error_detail": "every provider for this role is resting (quota or probe)",
                       "cut_off": False, "call_id": call_id, "final": True})
        raise NoProviderAvailableError(
            f"All providers failed for role '{self.role}'. Last error: {last_error}"
        )


class LLMRouter:
    def get_llm(self, role: str, temperature: float = 0, model_override: str = None,
                max_tokens: int = None):
        group = MODEL_OVERRIDE_GROUPS.get(model_override, role)
        if max_tokens is None:
            max_tokens = ROLE_MAX_TOKENS.get(group, DEFAULT_MAX_TOKENS)
        if group not in ROLE_DEPLOYMENTS:
            raise ValueError(f"Unknown role group '{group}'")
        # Check at least one key is configured
        has_key = any(
            os.environ.get(env_key)
            for _, env_key, _ in ROLE_DEPLOYMENTS.get(group, [])
        )
        if not has_key:
            raise NoProviderAvailableError(
                f"No API key configured for any provider in role '{group}'"
            )
        return RoutedLLM(group, temperature, max_tokens)

    def get_llm_for_agent(self, agent_name: str, temperature: float = 0, model_override: str = None,
                          max_tokens: int = None):
        role = AGENT_ROLES.get(agent_name)
        if not role:
            raise ValueError(f"No role mapped for agent '{agent_name}'")
        return self.get_llm(role, temperature, model_override, max_tokens)


router = LLMRouter()
