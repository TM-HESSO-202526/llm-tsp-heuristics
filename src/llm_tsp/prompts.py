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
    """Static historical family-avoidance block, off by default."""
    search = config.get("search", {})
    if not bool(search.get("historical_family_avoidance", False)):
        return ""
    return """Historical family avoidance is ACTIVE.

This run is not only trying to improve the best TSP gap. It is explicitly testing whether the LLM can be pushed away from over-produced heuristic families and generate structurally different constructive mechanisms.

From previous TSP runs, the following families were heavily over-generated and must NOT be used again as the main mechanism:

1. Nearest-neighbor / closest-unvisited constructors
   - Do not build a tour by repeatedly going to the nearest or cheapest next city.
   - Do not disguise this as “adaptive”, “hybrid”, “enhanced”, or “priority” nearest-neighbor.

2. Cheapest-insertion / regret-insertion constructors
   - Do not construct the tour mainly by inserting one unvisited city into the cheapest position.
   - Do not generate another randomized cheapest insertion, regret insertion, farthest insertion, or sampled insertion variant.

3. Simple candidate-list greedy constructors
   - Do not merely use problem.neighbors(i) to choose the nearest or highest-prior neighbor.
   - Candidate lists may be used, but the global construction logic must be different from greedy walk or greedy insertion.

4. Prior-as-linear-score variants
   - Do not simply score edges as distance - alpha * prior, distance / (1 + prior), or another small weighted mixture.
   - If problem.prior(i, j) is used, it must change the structure of the construction, not just the edge score.

5. Standard 2-opt / relocate / LK-like cleanup as the main idea
   - Do not produce a base tour and then rely mainly on 2-opt, segment reversal, relocate, swap, or variable-depth exchange.
   - Bounded cleanup is allowed only as a small final repair step, not as the core heuristic.

6. Random restarts / multi-start wrappers
   - Do not create diversity only by trying many random starts of the same known constructor.
   - Randomness is allowed only if the deterministic mechanism is structurally new.

Your next heuristic must choose a genuinely different construction family. Strict novelty requirement:
- The main construction mechanism must be different from nearest-neighbor, cheapest/regret insertion, and 2-opt-centered improvement.
- Do not merely rename an old method.
- Do not just add extra constants, thresholds, restarts, or a final local search to an old family.
- In the generated code comments, briefly indicate the intended mechanism family.

Still obey all TSP interface rules:
- Define exactly one class named TSPHeuristic.
- Return one permutation of 0..problem.n-1.
- Do not append the start city at the end.
- Use only numpy/math/basic Python.
- Do not use external solvers or libraries.
- Keep the method scalable for n around 1000 to 1800.
- Use problem.neighbors(i) and problem.prior(i, j) when available, but avoid dense all-pairs scans."""


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
) -> str:
    """Build the TSP LLaMEA prompt using the same structure as clustering."""
    base = base_task_prompt(config)
    search = config.get("search", {})
    strategy = normalized_selection_strategy(search.get("selection_strategy", "1+1"))

    if parent_summary is None:
        parent_summary = {}

    historical_memory = historical_memory or ""

    if prompt_mode == "initial" or not parent_summary:
        return f"""
{base}

{historical_memory}

Generate the first heuristic for this active objective now.
""".strip()

    parent_json = json.dumps(parent_summary, indent=2, ensure_ascii=False)

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
