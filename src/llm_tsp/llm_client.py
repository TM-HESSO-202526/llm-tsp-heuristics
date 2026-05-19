from __future__ import annotations

import os
import time
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


def call_groq_chat(
    messages: list[dict[str, str]],
    model: str,
    api_key_envs: list[str] | None = None,
    timeout_s: float = 60,
    max_retries: int = 5,
) -> str:
    api_key_envs = api_key_envs or ["GROQ_API_KEY", "GROQ_API_KEY_1", "GROQ_API_KEY_2"]
    url = "https://api.groq.com/openai/v1/chat/completions"
    last_error: Exception | None = None
    for attempt in range(max_retries):
        key_name = api_key_envs[attempt % len(api_key_envs)]
        api_key = get_secret(key_name)
        if not api_key:
            continue
        try:
            resp = requests.post(
                url,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"model": model, "messages": messages, "temperature": 0.8},
                timeout=timeout_s,
            )
            if resp.status_code == 429:
                wait_s = min(60, 5 * (attempt + 1))
                print(f"[api-key] HTTP 429 on {key_name}. Retrying same logical LLM call after {wait_s}s.")
                time.sleep(wait_s)
                continue
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            last_error = e
            time.sleep(min(30, 2 * (attempt + 1)))
    raise LLMClientError(f"LLM call failed after retries: {last_error}")
