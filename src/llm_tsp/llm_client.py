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
    temperature: float = 0.8,
    top_p: float = 1.0,
) -> str:
    api_key_envs = api_key_envs or ["GROQ_API_KEY", "GROQ_API_KEY_1", "GROQ_API_KEY_2"]
    url = "https://api.groq.com/openai/v1/chat/completions"
    last_error: Exception | None = None
    usable_keys = [k for k in api_key_envs if get_secret(k)]
    if not usable_keys:
        raise LLMClientError("No Groq API key found. Expected GROQ_API_KEY_1 or similar.")

    for attempt in range(max_retries):
        key_name = usable_keys[attempt % len(usable_keys)]
        api_key = get_secret(key_name)
        try:
            print(f"[llm] calling Groq model={model} key={key_name} attempt={attempt + 1}/{max_retries}")
            resp = requests.post(
                url,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": float(temperature),
                    "top_p": float(top_p),
                },
                timeout=timeout_s,
            )
            if resp.status_code == 429:
                wait_s = min(60, 5 * (attempt + 1))
                print(f"[api-key] HTTP 429 on {key_name}. Retrying same logical LLM call after {wait_s}s.")
                time.sleep(wait_s)
                continue
            if resp.status_code >= 400:
                print(f"[llm] HTTP {resp.status_code}: {resp.text[:500]}")
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            last_error = e
            wait_s = min(30, 2 * (attempt + 1))
            print(f"[llm] request error on {key_name}: {type(e).__name__}: {e}. Waiting {wait_s}s.")
            time.sleep(wait_s)
    raise LLMClientError(f"LLM call failed after retries: {last_error}")
