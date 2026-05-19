from __future__ import annotations

from typing import Any


SYSTEM_PROMPT = """You generate constructive heuristics for large TSP instances. Return only one fenced Python block."""


def _strategy_text(strategy: str) -> str:
    if strategy == "1+1":
        return (
            "Search strategy: 1+1 elitist evolution. The parent is the best full-valid "
            "candidate seen so far. Your job is to produce a child that improves it."
        )
    if strategy == "1,1":
        return (
            "Search strategy: 1,1 sequential evolution. The parent is the latest previous "
            "candidate, even if it is not the global best. Your job is to produce the next child."
        )
    return f"Search strategy: {strategy}."


def build_tsp_prompt(
    config: dict[str, Any],
    previous_feedback: str | None = None,
    parent_code: str | None = None,
    history_text: str | None = None,
    prompt_mode: str = "initial",
    parent_is_invalid: bool = False,
) -> str:
    """Build the TSP LLaMEA prompt.

    The search-control wording mirrors the clustering repo pattern:
    - 1+1 uses the best full-valid parent;
    - 1,1 uses the latest sequential parent;
    - invalid-parent redesign can expose or hide the failed code depending on
      INCLUDE_INVALID_CODE_IN_FEEDBACK and HIDE_INVALID_PARENT_CODE.
    """
    pop = config.get("popmusic", {})
    feedback = config.get("feedback", {})
    search = config.get("search", {})
    strategy = search.get("selection_strategy", "1+1")

    lines = [
        "Write a Python constructive TSP heuristic.",
        _strategy_text(strategy),
        "",
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
        "- Use numpy/math/list/dict/set only; do not import sklearn, scipy, networkx, OR-Tools, LKH, Concorde, or tsplib libraries.",
    ]

    if pop.get("use_popmusic_candidates"):
        lines += [
            "",
            "POPMUSIC/LKH candidate mode is active.",
            "Use problem.neighbors(i) as the main sparse neighborhood for local choices.",
            "When candidate restriction is active, problem.edge_cost(i, j) may only be available for candidate edges; problem.full_edge_cost(i, j) exists for diagnostics but should not become a dense O(n^2) scan.",
        ]
    if pop.get("use_popmusic_edge_prior"):
        lines += [
            "",
            "POPMUSIC edge-prior mode is active.",
            f"Prior mode: {pop.get('prior_mode', 'frequency')}",
            "problem.prior(i, j) gives an operational edge-support signal derived from POPMUSIC/LKH candidate behavior.",
            "Use the prior as a helpful signal, but still control distance, degree balance, tour validity, and runtime.",
        ]

    if history_text:
        lines += ["", "Recent attempt history:", history_text]

    if previous_feedback:
        title = "Previous evaluation feedback:"
        if prompt_mode == "redesign_invalid_parent":
            title = "Invalid/partial parent diagnosis:"
        lines += ["", title, previous_feedback]

    if prompt_mode == "redesign_invalid_parent":
        lines += [
            "",
            "Redesign instruction:",
            "The selected parent is invalid, partially valid, duplicated, or timed out before any reliable full-valid parent was available.",
            "Do not make a tiny patch only. Redesign the heuristic so that it is robust first: deterministic fallback, explicit unvisited-set handling, no empty reductions, no shape assumptions, and valid full-tour return on every instance.",
        ]
    elif parent_code:
        lines += [
            "",
            "Mutation instruction:",
            "Use the parent as a starting point, but make a meaningful algorithmic improvement rather than a superficial rename.",
        ]

    include_parent_code = bool(feedback.get("include_parent_code_in_mutation_prompt", True))
    include_invalid_code = bool(feedback.get("include_invalid_code_in_feedback", True))
    hide_invalid_code = bool(search.get("hide_invalid_parent_code", False))

    if parent_code:
        should_show_valid_parent = (not parent_is_invalid) and include_parent_code
        should_show_invalid_parent = parent_is_invalid and include_invalid_code and not hide_invalid_code
        if should_show_valid_parent:
            lines += ["", "Current parent code to improve:", "```python", parent_code, "```"]
        elif should_show_invalid_parent:
            lines += ["", "Invalid/partial parent code shown for diagnosis:", "```python", parent_code, "```"]
        elif parent_is_invalid and hide_invalid_code:
            lines += ["", "The invalid/partial parent code is intentionally hidden; use only the diagnosis above."]

    lines += ["", "Return only the fenced Python code block."]
    return "\n".join(lines)
