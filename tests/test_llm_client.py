from __future__ import annotations

from dataclasses import dataclass

from llm_tsp import llm_client


@dataclass
class FakeResponse:
    status_code: int
    content: str = "ok"
    text: str = ""

    def json(self):
        return {"choices": [{"message": {"content": self.content}}]}


def _auth_key(headers):
    return headers["Authorization"].replace("Bearer ", "")


def test_groq_sticky_key_keeps_same_key_until_429(monkeypatch):
    llm_client.reset_groq_key_state()
    monkeypatch.setenv("GROQ_API_KEY_1", "key-one")
    monkeypatch.setenv("GROQ_API_KEY_2", "key-two")
    monkeypatch.setattr(llm_client.time, "sleep", lambda *_args, **_kwargs: None)

    used = []

    def fake_post(_url, headers, json, timeout):
        used.append(_auth_key(headers))
        return FakeResponse(200, content=f"response-{len(used)}")

    monkeypatch.setattr(llm_client.requests, "post", fake_post)
    messages = [{"role": "user", "content": "x"}]

    assert llm_client.call_groq_chat(messages, "m", ["GROQ_API_KEY_1", "GROQ_API_KEY_2"], calls_per_minute_per_key=99) == "response-1"
    assert llm_client.call_groq_chat(messages, "m", ["GROQ_API_KEY_1", "GROQ_API_KEY_2"], calls_per_minute_per_key=99) == "response-2"
    assert used == ["key-one", "key-one"]


def test_groq_switches_key_after_429(monkeypatch):
    llm_client.reset_groq_key_state()
    monkeypatch.setenv("GROQ_API_KEY_1", "key-one")
    monkeypatch.setenv("GROQ_API_KEY_2", "key-two")
    monkeypatch.setattr(llm_client.time, "sleep", lambda *_args, **_kwargs: None)

    used = []

    def fake_post(_url, headers, json, timeout):
        used.append(_auth_key(headers))
        if len(used) == 1:
            return FakeResponse(429, text="rate limited")
        return FakeResponse(200, content="ok-after-switch")

    monkeypatch.setattr(llm_client.requests, "post", fake_post)
    messages = [{"role": "user", "content": "x"}]

    assert llm_client.call_groq_chat(messages, "m", ["GROQ_API_KEY_1", "GROQ_API_KEY_2"], calls_per_minute_per_key=99) == "ok-after-switch"
    assert used == ["key-one", "key-two"]
