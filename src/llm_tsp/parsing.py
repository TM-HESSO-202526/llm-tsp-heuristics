from __future__ import annotations

import re


def extract_first_python_block(text: str) -> str:
    match = re.search(r"```(?:python|py)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def reject_forbidden_code(code: str) -> None:
    forbidden = [
        "import os",
        "import subprocess",
        "subprocess.",
        "open(",
        "requests.",
        "socket",
        "urllib",
        "eval(",
        "exec(",
    ]
    lowered = code.lower()
    for token in forbidden:
        if token.lower() in lowered:
            raise ValueError(f"Forbidden code token: {token}")
