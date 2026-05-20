from __future__ import annotations

import json
from typing import Any


SYSTEM_PROMPT = (
    "You generate executable Python class-based TSP heuristics. "
    "Follow the requested interface exactly. Use only numpy/math in generated code; "
    "no external TSP, graph, optimization, or solver libraries."
)


def normalized_selection_strategy(selection_strategy: str) -> str:
    raw = str(selection_strategy).strip().lower().replace(" ", "")
    if raw in {"1,1", "one,one", "onecommaone", "1comma1"}:
        return "1,1"
    return "1+1"


def objective_prompt_block(config: dict[str, Any]) -> str:
    """TSP-specific objective/interface context, analogous to clustering objective_prompt_block."""
    pop = config.get("popmusic", {})
    lines = [
        "Active objective: Traveling Salesman Problem / permutation tour construction.",
        "Problem:",
        "Given a TSPLIB instance with n cities, return one Hamiltonian tour as a permutation of city indices.",
        "The evaluator closes the tour automatically by adding the edge from the last city back to the first city.",
        "Do not append the start city at the end of the returned tour.",
        "",
        "Official evaluation objective:",
        "Minimize total tour length. If an optimum is available, the evaluator reports the percentage gap versus the known optimum.",
        "",
        "Problem object seen by your code:",
        "- problem.n: number of cities.",
        "- problem.edge_cost(i, j): true TSPLIB edge-cost query for one edge at a time.",
        "- problem.neighbors(i): sparse POPMUSIC/LKH candidate-neighbor list for city i when candidate mode is active.",
        "- problem.prior(i, j): optional POPMUSIC/LKH tour-frequency edge-support signal.",
        "- problem.coords: city coordinates as a numpy array when available.",
        "The full dense distance matrix is not part of the public interface; use bounded edge_cost queries and candidate lists.",
    ]
    if pop.get("use_popmusic_candidates"):
        lines += [
            "",
            "POPMUSIC/LKH candidate mode is active.",
            "Use problem.neighbors(i) as the main sparse neighborhood for local choices.",
            "Candidate lists are guidance, not a hard final-tour feasibility constraint.",
            "The LLM receives the sparse candidate list, not a full dense distance matrix.",
            "problem.edge_cost(i, j) is an oracle-style query for individual edge costs; avoid dense all-pairs scans.",
            "The final returned tour is a normal full TSP permutation and is evaluated on the true full TSPLIB distance.",
        ]
    if pop.get("use_popmusic_edge_prior"):
        lines += [
            "",
            "POPMUSIC edge-prior mode is active.",
            f"Prior mode: {pop.get('prior_mode', 'frequency')}",
            "problem.prior(i, j) gives an operational edge-support signal built from 30 short LKH/POPMUSIC tour runs by counting edge frequencies.",
            "Use the prior as a helpful construction signal, while still controlling distance, degree balance, tour validity, and runtime.",
        ]
    return "\n".join(lines).strip()


def base_task_prompt(config: dict[str, Any]) -> str:
    """Base task prompt with no search-strategy wording, mirroring clustering."""
    objective_block = objective_prompt_block(config)
    return f"""
Your task is to design a novel heuristic algorithm for the following TSP optimization problem.

{objective_block}

Interface:
The generated Python code must define exactly one class named TSPHeuristic:

class TSPHeuristic:
    def __call__(self, problem, rng=None):
        ...

The evaluator will call:

algo = TSPHeuristic()
tour = algo(problem, rng)

Inputs:
- problem is a TSP problem object exposing the interface described above.
- rng is an optional numpy.random.Generator.

Output:
- Return exactly n city indices as an array-like permutation of 0..problem.n-1.
- Each city must appear exactly once.
- Do not append the first city at the end; the evaluator closes the tour itself.
- The algorithm must be self-contained and executable with numpy/math available.

Rules:
- You may use numpy as np, math, lists, dictionaries, sets, and bounded loops.
- Do not import or call sklearn, scipy, pandas, joblib, numba, torch, tensorflow, jax, networkx, OR-Tools, LKH, Concorde, tsplib libraries, multiprocessing, threading, or external optimization libraries.
- Do not read/write files.
- Do not use global hidden state.
- Keep the method scalable for n around 1000 to 1800 cities.
- Avoid cubic or unbounded all-pairs local search.
- When drawing an index, use rng.integers(problem.n) or rng.integers(0, problem.n); do not use rng.randint.

Objective separation:
The official evaluator computes the tour cost and gap outside your code after your algorithm returns a tour.
No hidden evaluator-side local search, 2-opt, repair, or post-processing is applied beyond validation.
Do not hard-code any instance names, coordinates, optima, reference values, or file paths.

Diversity/novelty:
Do not merely rename the previous algorithm or only tune constants.
Prefer meaningful structural changes when redesigning, while still optimizing the active TSP objective.

Return format:
# Name:

# Code:
```python
# your code here
```
""".strip()


def historical_family_avoidance_block(config: dict[str, Any]) -> str:
    """Static historical family-memory block, off by default like the clustering control."""
    search = config.get("search", {})
    if not bool(search.get("historical_family_avoidance", False)):
        return ""
    return (
        "Historical family memory from previous TSP runs:\n"
        "The following mechanism families were repeatedly observed in older TSP artifacts. "
        "Use this as prior context, not as a hard ban. Avoid weak or stagnant families as minor variants, "
        "but preserve/refine historically strong families if the selected parent genuinely belongs to one.\n"
        "Do not merely add words such as enhanced, adaptive, hybrid, regularized, improved, or V2 "
        "while keeping the same main mechanism.\n\n"
        "For TSP: repeated weak families often include nearest-neighbor-only constructors, random restarts "
        "without a structural rule, and expensive unbounded 2-opt/LK-like loops that time out. "
        "Historically useful families are not banned: nearest-neighbor/regret/insertion mechanisms with bounded "
        "improvement or POPMUSIC candidate/prior usage may still be refined if they are genuinely improving.\n\n"
        "Your next heuristic should make a structural change in the main tour-construction mechanism unless "
        "the selected parent is already from a strong/improving family. Do not merely rename or decorate a weak family."
    )


def build_family_memory_block(records: list[Any] | None, parent: Any | None, config: dict[str, Any]) -> str:
    """Dynamic current-run family novelty block using clustering variable names.

    This is deliberately prompt-silent when family_novelty_mode is False.
    """
    search = config.get("search", {})
    if not bool(search.get("family_novelty_mode", False)):
        return ""
    if not records:
        return ""

    min_attempts = int(search.get("min_family_attempts_before_avoid", 5))
    memory_limit = int(search.get("family_memory_limit", 8))
    weak_threshold = float(search.get("weak_family_score_threshold", 20.0))
    allow_strong = bool(search.get("allow_strong_family_exploitation", True))
    parent_family = str(getattr(parent, "family", "") or "") if parent is not None else ""

    by_family: dict[str, dict[str, Any]] = {}
    for r in records:
        fam = str(getattr(r, "family", "") or "other")
        d = by_family.setdefault(fam, {"attempts": 0, "best_score": None, "valid": 0})
        d["attempts"] += 1
        if bool(getattr(r, "full_valid", False)):
            d["valid"] += 1
        score = getattr(r, "selection_score", None)
        if score is not None:
            try:
                score_f = float(score)
                d["best_score"] = score_f if d["best_score"] is None else min(float(d["best_score"]), score_f)
            except Exception:
                pass

    rows = []
    for fam, d in by_family.items():
        attempts = int(d["attempts"])
        best_score = d["best_score"]
        weak_or_repeated = attempts >= min_attempts and (best_score is None or float(best_score) > weak_threshold)
        strong_parent = allow_strong and fam == parent_family and best_score is not None and float(best_score) <= weak_threshold
        if weak_or_repeated and not strong_parent:
            rows.append((attempts, fam, best_score, int(d["valid"])))
    if not rows:
        return ""
    rows = sorted(rows, reverse=True)[:memory_limit]
    lines = [
        "Current-run family novelty memory:",
        "The following families have appeared repeatedly without strong improvement. Avoid producing another minor variant unless you make a real structural change.",
    ]
    for attempts, fam, best_score, valid_count in rows:
        score_txt = "NA" if best_score is None else f"{float(best_score):.4f}"
        lines.append(f"- family={fam} | attempts={attempts} | full_valid_attempts={valid_count} | best_selection_score={score_txt}")
    return "\n".join(lines)


def _redesign_instruction(parent_timed_out: bool = False) -> str:
    return (
        "Selection mode: invalid/timeout-aware redesign fallback.\n"
        "No fully valid heuristic has been found yet, and the selected parent is not fully valid"
        + (" and appears to have timeout/runtime failures.\n" if parent_timed_out else ".\n")
        + "Do not continue the same broken or expensive structure.\n"
        + "Use the current-run feedback and parent code below only to understand the failure mode.\n"
        + "The parent code is shown for diagnosis, but do not blindly mutate or continue the same broken/expensive structure.\n"
        + "Redesign from scratch if the parent structure is the source of the failure.\n"
        + "The first priority is to become valid on all search instances; then improve the active objective."
    )


def _selection_instruction(strategy: str, parent_is_valid: bool) -> str:
    strategy = normalized_selection_strategy(strategy)
    if strategy == "1+1":
        if parent_is_valid:
            return (
                "Selection mode: 1+1 elitist improvement.\n"
                "The selected parent below is the current best-so-far full-valid heuristic under the active objective. "
                "Your goal is to improve on this parent while preserving useful mechanisms, keeping the class valid and scalable, "
                "and avoiding changes that only add complexity without lowering the score."
            )
        return (
            "Selection mode: 1+1 with partial-validity fallback.\n"
            "No fully valid heuristic has been found yet. The selected parent below is the best partial/latest candidate available. "
            "Your first priority is to make it valid on all search instances; then improve the active objective."
        )
    if parent_is_valid:
        return (
            "Selection mode: 1,1 sequential mutation chain.\n"
            "The selected parent below is the most recent heuristic in the chain, not necessarily the best-so-far. "
            "Your goal is to explore a meaningful variation while keeping the heuristic valid and scalable. "
            "Larger structural changes are acceptable, but use the feedback to avoid repeating known failures."
        )
    return (
        "Selection mode: 1,1 sequential mutation chain.\n"
        "The selected parent below is the most recent heuristic in the chain and it may be invalid or only partially valid. "
        "Your first priority is to repair validity issues while still exploring a meaningful variation. "
        "Use the feedback to avoid repeating known failures."
    )


def build_tsp_prompt(
    config: dict[str, Any],
    parent_code: str | None = None,
    history_text: str | None = None,
    prompt_mode: str = "initial",
    parent_is_invalid: bool = False,
    parent_summary: dict[str, Any] | None = None,
    parent_timed_out: bool = False,
    historical_memory: str | None = None,
    family_memory: str | None = None,
) -> str:
    """Build the TSP LLaMEA prompt using the same structure as clustering."""
    base = base_task_prompt(config)
    search = config.get("search", {})
    strategy = normalized_selection_strategy(search.get("selection_strategy", "1+1"))

    if parent_summary is None:
        parent_summary = {}

    if prompt_mode == "initial" or not parent_summary:
        return f"""
{base}

Generate the first heuristic for this active objective now.
""".strip()

    parent_json = json.dumps(parent_summary, indent=2, ensure_ascii=False)
    historical_memory = historical_memory or ""
    family_memory = family_memory or ""

    if prompt_mode == "redesign_invalid_parent":
        instruction = _redesign_instruction(parent_timed_out=parent_timed_out)
        code_block = ""
        if parent_code:
            code_block = f"""
Invalid/partial parent full code, shown only for diagnosis:
```python
{parent_code}
```
"""
        return f"""
{base}

{instruction}

{historical_memory}

{family_memory}

Current-run invalid/partial parent summary:
```json
{parent_json}
```

{code_block}
Important: the parent above is not fully valid.
Use it to understand what failed, but do not simply continue the same broken or expensive structure.
If the parent appears to time out, crash, return wrong tour lengths, duplicate cities, omit cities, or use an expensive mechanism, redesign from scratch while avoiding that failure mode.
Generate a fresh redesigned heuristic for the active objective.
Keep the generated code class-based, self-contained, and compatible with the TSPHeuristic interface.
Return the answer in the required # Name / # Code format.
""".strip()

    parent_is_valid = not bool(parent_is_invalid)
    instruction = _selection_instruction(strategy, parent_is_valid=parent_is_valid)
    history = history_text or "No previous attempts."
    code_block = ""
    if parent_code:
        code_block = f"""
Selected parent full code:
```python
{parent_code}
```
"""

    return f"""
{base}

Previously generated heuristics for this active objective:
{history}

{historical_memory}

{family_memory}

{instruction}

Selected parent summary:
```json
{parent_json}
```

{code_block}
Repair, modify, or redesign the heuristic to improve the active objective.
Use the score, runtime, error feedback, and instance-level feedback above.
If the parent failed on an instance, fix that issue.
If the parent was valid, try to lower the mean cost / mean gap versus the active reference.
Keep the generated code class-based, self-contained, and compatible with the TSPHeuristic interface.
Return the answer in the required # Name / # Code format.
""".strip()
