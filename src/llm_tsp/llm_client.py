from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import requests


class LLMClientError(RuntimeError):
    pass


def get_secret(key: str) -> str | None:
    val = os.environ.get(key)
    if val:
        return val
    try:
        from google.colab import userdata  # type: ignore
        return userdata.get(key)
    except Exception:
        return None


@dataclass
class _GroqStickyKeyState:
    """Stateful Groq key handling, matching the clustering pipeline style.

    The important behavior is sticky-key selection: keep using the current usable
    key across logical LLM calls. Only switch keys when the current key returns
    HTTP 429 or has a repeated request/timeout failure. The proactive per-key
    limiter cools down the current key before a call instead of rotating every
    request.
    """

    key_envs_signature: tuple[str, ...] = ()
    keys: list[tuple[str, str]] = field(default_factory=list)
    key_index: int = 0
    call_times_by_key: dict[str, list[float]] = field(default_factory=dict)

    def refresh(self, api_key_envs: list[str]) -> None:
        signature = tuple(api_key_envs)
        if signature == self.key_envs_signature and self.keys:
            return

        loaded: list[tuple[str, str]] = []
        seen_values: set[str] = set()
        for name in api_key_envs:
            value = get_secret(name)
            if not value or value in seen_values:
                continue
            loaded.append((name, value))
            seen_values.add(value)

        self.key_envs_signature = signature
        self.keys = loaded
        self.key_index = 0
        self.call_times_by_key = {name: [] for name, _ in loaded}

    def current(self) -> tuple[str, str]:
        if not self.keys:
            raise LLMClientError("No Groq API key found. Expected GROQ_API_KEY_1 or similar.")
        return self.keys[self.key_index % len(self.keys)]

    def switch_to_next(self) -> tuple[str, str]:
        if not self.keys:
            raise LLMClientError("No Groq API key available to switch to.")
        self.key_index = (self.key_index + 1) % len(self.keys)
        return self.current()

    def rate_limit_wait_for_current_key(self, calls_per_minute_per_key: float) -> None:
        key_name, _ = self.current()
        limit = max(1, int(calls_per_minute_per_key))
        now = time.time()
        times = [t for t in self.call_times_by_key.get(key_name, []) if now - t < 60.0]
        self.call_times_by_key[key_name] = times
        if len(times) >= limit:
            wait_s = 60.0 - (now - times[0]) + 0.5
            wait_s = max(0.0, wait_s)
            print(
                f"[rate-limit] {len(times)} calls already made for {key_name} in last 60s. "
                f"Cooling down {wait_s:.1f}s...",
                flush=True,
            )
            time.sleep(wait_s)

    def mark_success_for_current_key(self) -> None:
        key_name, _ = self.current()
        self.call_times_by_key.setdefault(key_name, []).append(time.time())


_GROQ_STATE = _GroqStickyKeyState()


def reset_groq_key_state() -> None:
    """Reset sticky-key/rate-limiter state. Mostly useful for tests."""

    global _GROQ_STATE
    _GROQ_STATE = _GroqStickyKeyState()


def loaded_groq_key_names(api_key_envs: list[str] | None = None) -> list[str]:
    """Return loaded Groq key names using the same de-duplication as calls."""

    api_key_envs = api_key_envs or ["GROQ_API_KEY"] + [f"GROQ_API_KEY_{i}" for i in range(1, 11)]
    state = _GroqStickyKeyState()
    state.refresh(api_key_envs)
    return [name for name, _ in state.keys]


def call_groq_chat(
    messages: list[dict[str, str]],
    model: str,
    api_key_envs: list[str] | None = None,
    timeout_s: float = 60,
    max_retries: int | None = None,
    max_429_retries: int | None = None,
    max_request_error_retries: int | None = None,
    calls_per_minute_per_key: float = 2,
    temperature: float = 0.8,
    top_p: float = 1.0,
) -> str:
    """Call Groq with clustering-style sticky key handling.

    Behavior:
    - Load all available keys once and de-duplicate identical secret values.
    - Keep using the current key for subsequent logical LLM calls.
    - Proactively wait when the current key has already used the configured
      calls-per-minute budget.
    - Switch to the next key only after HTTP 429 or request/timeout errors.
    - Retries caused by 429/request errors remain part of the same logical LLM
      call and therefore do not count as heuristic attempts.
    """

    api_key_envs = api_key_envs or ["GROQ_API_KEY", "GROQ_API_KEY_1", "GROQ_API_KEY_2"]
    url = "https://api.groq.com/openai/v1/chat/completions"
    _GROQ_STATE.refresh(api_key_envs)
    if not _GROQ_STATE.keys:
        raise LLMClientError("No Groq API key found. Expected GROQ_API_KEY_1 or similar.")

    if max_retries is not None:
        # Backward-compatible cap when older code passes only max_retries.
        max_429_retries = int(max_retries) if max_429_retries is None else int(max_429_retries)
        max_request_error_retries = int(max_retries) if max_request_error_retries is None else int(max_request_error_retries)
    max_429_retries = 100 if max_429_retries is None else int(max_429_retries)
    max_request_error_retries = 5 if max_request_error_retries is None else int(max_request_error_retries)

    retry_429 = 0
    retry_request_error = 0
    last_error: Exception | None = None

    while True:
        key_name, api_key = _GROQ_STATE.current()
        _GROQ_STATE.rate_limit_wait_for_current_key(calls_per_minute_per_key)
        try:
            print(
                f"[llm] PATCHED_API_TIMEOUT calling Groq with key={key_name} timeout={int(timeout_s)}s",
                flush=True,
            )
            resp = requests.post(
                url,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": float(temperature),
                    "top_p": float(top_p),
                },
                timeout=(10, timeout_s),
            )
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError, requests.exceptions.RequestException) as e:
            last_error = e
            retry_request_error += 1
            print(
                f"[request-timeout/error] {type(e).__name__}: {e}. "
                "Retrying same logical LLM call; this will not count as a heuristic attempt.",
                flush=True,
            )
            if len(_GROQ_STATE.keys) > 1:
                next_name, _ = _GROQ_STATE.switch_to_next()
                print(f"[api-key] Switching to {next_name}", flush=True)
                time.sleep(2.0)
            else:
                time.sleep(5.0)
            if retry_request_error >= max_request_error_retries:
                raise LLMClientError(
                    f"Too many repeated Groq request errors/timeouts ({retry_request_error})."
                ) from e
            continue

        if resp.status_code == 429:
            retry_429 += 1
            print(
                f"[api-key] HTTP 429 on {key_name}. "
                "Retrying same logical LLM call; this will not count as a heuristic attempt.",
                flush=True,
            )
            if len(_GROQ_STATE.keys) > 1:
                next_name, _ = _GROQ_STATE.switch_to_next()
                print(f"[api-key] Switching to {next_name}", flush=True)
                time.sleep(2.0)
            else:
                time.sleep(65.0)
            if retry_429 >= max_429_retries:
                raise LLMClientError("Too many repeated HTTP 429 retries.")
            continue

        if resp.status_code >= 400:
            text = resp.text[:2000]
            raise LLMClientError(f"Groq HTTP {resp.status_code}: {text}")

        try:
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
        except Exception as e:  # pragma: no cover - defensive against malformed API responses
            last_error = e
            raise LLMClientError(f"Malformed Groq response: {e}") from e

        _GROQ_STATE.mark_success_for_current_key()
        return content
