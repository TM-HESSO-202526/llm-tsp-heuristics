from __future__ import annotations

from typing import Any


SYSTEM_PROMPT = """You generate constructive heuristics for large TSP instances. Return only one fenced Python block."""


def build_tsp_prompt(
    config: dict[str, Any],
    previous_feedback: str | None = None,
    parent_code: str | None = None,
) -> str:
    pop = config.get("popmusic", {})
    feedback = config.get("feedback", {})
    lines = [
        "Write a Python constructive TSP heuristic.",
        "Implement exactly:",
        "def construct_tour(problem, rng=None):",
        "    # return a permutation of 0..problem.n-1",
        "",
        "Available problem interface:",
        "- problem.n",
        "- problem.edge_cost(i, j)",
        "- problem.full_edge_cost(i, j)",
        "- problem.neighbors(i)",
        "- problem.prior(i, j)",
        "- problem.coords",
        "",
        "Constraints:",
        "- Return a valid full tour containing every node exactly once.",
        "- Do not read files or call external solvers.",
        "- Avoid cubic behavior; instances have about 1000 to 1800 nodes.",
    ]
    if pop.get("use_popmusic_candidates"):
        lines += [
            "",
            "POPMUSIC/LKH candidate mode is active.",
            "Prefer problem.neighbors(i) for local choices.",
        ]
    if pop.get("use_popmusic_edge_prior"):
        lines += [
            "POPMUSIC edge-prior mode is active.",
            f"Prior mode: {pop.get('prior_mode', 'frequency')}",
            "Use problem.prior(i, j) as a helpful signal, but still control distance and tour validity.",
        ]
    if previous_feedback:
        lines += ["", "Previous evaluation feedback:", previous_feedback]
    if parent_code and feedback.get("include_parent_code_in_mutation_prompt", True):
        lines += ["", "Current best/parent code to improve:", "```python", parent_code, "```"]
    lines += ["", "Return only the fenced Python code block."]
    return "\n".join(lines)
