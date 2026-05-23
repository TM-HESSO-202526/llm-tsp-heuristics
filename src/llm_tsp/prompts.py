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
    use_candidates = bool(pop.get("use_popmusic_candidates"))
    use_prior = bool(pop.get("use_popmusic_edge_prior"))

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
        "- problem.coords: city coordinates as a numpy array when available.",
        "The full dense distance matrix is not part of the public interface; use bounded edge_cost queries.",
    ]
    if use_candidates:
        lines += [
            "- problem.neighbors(i): sparse POPMUSIC/LKH candidate-neighbor list for city i.",
            "",
            "POPMUSIC/LKH candidate mode is active.",
            "Use problem.neighbors(i) as the main sparse neighborhood for local choices.",
            "Candidate lists are guidance, not a hard final-tour feasibility constraint.",
            "The LLM receives the sparse candidate list, not a full dense distance matrix.",
            "problem.edge_cost(i, j) is an oracle-style query for individual edge costs; avoid dense all-pairs scans.",
            "The final returned tour is a normal full TSP permutation and is evaluated on the true full TSPLIB distance.",
        ]
    if use_prior:
        lines += [
            "- problem.prior(i, j): POPMUSIC/LKH tour-frequency edge-support signal.",
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

    pop = config.get("popmusic", {})
    use_candidates = bool(pop.get("use_popmusic_candidates"))
    use_prior = bool(pop.get("use_popmusic_edge_prior"))

    families = [
        "1. Nearest-neighbor / closest-unvisited constructors\n"
        "   - Do not build a tour by repeatedly going to the nearest or cheapest next city.\n"
        "   - Do not disguise this as \"adaptive\", \"hybrid\", \"enhanced\", or \"priority\" nearest-neighbor.",
        "2. Cheapest-insertion / regret-insertion constructors\n"
        "   - Do not construct the tour mainly by inserting one unvisited city into the cheapest position.\n"
        "   - Do not generate another randomized cheapest insertion, regret insertion, farthest insertion, or sampled insertion variant.",
    ]
    if use_candidates:
        families.append(
            "3. Simple candidate-list greedy constructors\n"
            "   - Do not merely use problem.neighbors(i) to choose the nearest or highest-prior neighbor.\n"
            "   - Candidate lists may be used, but the global construction logic must be different from greedy walk or greedy insertion."
        )
    if use_prior:
        families.append(
            f"{len(families) + 1}. Prior-as-linear-score variants\n"
            "   - Do not simply score edges as distance - alpha * prior, distance / (1 + prior), or another small weighted mixture.\n"
            "   - If problem.prior(i, j) is used, it must change the structure of the construction, not just the edge score."
        )
    families += [
        f"{len(families) + 1}. Standard 2-opt / relocate / LK-like cleanup as the main idea\n"
        "   - Do not produce a base tour and then rely mainly on 2-opt, segment reversal, relocate, swap, or variable-depth exchange.\n"
        "   - Bounded cleanup is allowed only as a small final repair step, not as the core heuristic.",
        f"{len(families) + 2}. Random restarts / multi-start wrappers\n"
        "   - Do not create diversity only by trying many random starts of the same known constructor.\n"
        "   - Randomness is allowed only if the deterministic mechanism is structurally new.",
    ]

    interface_rules = [
        "- Define exactly one class named TSPHeuristic.",
        "- Return one permutation of 0..problem.n-1.",
        "- Do not append the start city at the end.",
        "- Use only numpy/math/basic Python.",
        "- Do not use external solvers or libraries.",
        "- Keep the method scalable for n around 1000 to 1800.",
    ]
    if use_candidates and use_prior:
        interface_rules.append("- Use problem.neighbors(i) and problem.prior(i, j), but avoid dense all-pairs scans.")
    elif use_candidates:
        interface_rules.append("- Use problem.neighbors(i) when helpful, but avoid dense all-pairs scans.")
    elif use_prior:
        interface_rules.append("- Use problem.prior(i, j) when helpful, but avoid dense all-pairs scans.")
    else:
        interface_rules.append("- Use bounded problem.edge_cost(i, j) queries and problem.coords when helpful; avoid dense all-pairs scans.")

    return f"""Historical family avoidance is ACTIVE.

This run is not only trying to improve the best TSP gap. It is explicitly testing whether the LLM can be pushed away from over-produced heuristic families and generate structurally different constructive mechanisms.

From previous TSP runs, the following families were heavily over-generated and must NOT be used again as the main mechanism:

{chr(10).join(families)}

Your next heuristic must choose a genuinely different construction family. Strict novelty requirement:
- The main construction mechanism must be different from nearest-neighbor, cheapest/regret insertion, and 2-opt-centered improvement.
- Do not merely rename an old method.
- Do not just add extra constants, thresholds, restarts, or a final local search to an old family.
- In the generated code comments, briefly indicate the intended mechanism family.

Still obey all TSP interface rules:
{chr(10).join(interface_rules)}"""


def family_focus_block(
    config: dict[str, Any],
    family_spec: dict[str, Any] | None = None,
    *,
    family_step: int | None = None,
    calls_per_family: int | None = None,
    family_index: int | None = None,
    total_families: int | None = None,
) -> str:
    """Build the optional family-focus block from launcher-provided family text.

    The backend intentionally does not hard-code the family names/objectives.
    The Colab launcher owns the FAMILY_FOCUS_PLAN so the user can edit, comment,
    reorder, or disable families quickly without touching source code.
    """
    search = config.get("search", {})
    if not bool(search.get("family_focus_mode", False)) or not family_spec:
        return ""

    name = str(family_spec.get("name") or family_spec.get("id") or "Unnamed focus family").strip()
    fid = str(family_spec.get("id") or name).strip()
    objective = str(
        family_spec.get("objective")
        or family_spec.get("family_objective")
        or family_spec.get("description")
        or "Improve this focused TSP construction family."
    ).strip()
    description = str(family_spec.get("description") or "").strip()

    raw_constraints = family_spec.get("strict_constraints") or family_spec.get("constraints") or []
    if isinstance(raw_constraints, str):
        constraints = [line.strip(" -") for line in raw_constraints.splitlines() if line.strip()]
    else:
        constraints = [str(x).strip(" -") for x in raw_constraints if str(x).strip()]

    raw_instructions = family_spec.get("instructions") or family_spec.get("family_specific_instructions") or []
    if isinstance(raw_instructions, str):
        instructions = [line.strip(" -") for line in raw_instructions.splitlines() if line.strip()]
    else:
        instructions = [str(x).strip(" -") for x in raw_instructions if str(x).strip()]

    default_constraints = [
        "Your task is to improve this family, not to switch families.",
        "The declared family must be the main construction mechanism, not a cosmetic wrapper.",
        "Do not compute the family-specific structure and then ignore it.",
        "Do not fall back to global nearest-neighbor as the main construction.",
        "Do not use cheapest/regret insertion as the main construction.",
        "Do not rely on 2-opt, swaps, relocate moves, or segment reversal as the main source of quality.",
        "Bounded cleanup is allowed only after the family-specific constructive tour has been built.",
        "The method must scale to n around 1000 to 1800 cities.",
    ]

    seen: set[str] = set()
    merged_constraints: list[str] = []
    for item in [*constraints, *instructions, *default_constraints]:
        key = item.lower()
        if key not in seen:
            merged_constraints.append(item)
            seen.add(key)

    progress = []
    if family_index is not None and total_families is not None:
        progress.append(f"Family block: {int(family_index) + 1}/{int(total_families)}")
    if family_step is not None and calls_per_family is not None:
        progress.append(f"Call inside this family block: {int(family_step)}/{int(calls_per_family)}")
    progress_text = "\n".join(progress)

    description_block = ""
    if description and description != objective:
        description_block = f"\nFamily description from launcher:\n{description}\n"

    constraints_text = "\n".join(f"- {c}" for c in merged_constraints)
    return f"""Family-focus mode is ACTIVE.

For the next generated heuristic, you are locked to the following family:

Family id:
{fid}

Family name:
{name}

Family objective:
{objective}
{description_block}
{progress_text}

Strict constraints:
{constraints_text}

Only use the local parent and local history from this same family block. Ignore successful heuristics from other family blocks as mechanisms to preserve. At the end of the run, the backend will compare the best candidate from each family separately.""".strip()


def _redesign_instruction(
    parent_timed_out: bool = False,
    historical_avoidance_active: bool = False,
    family_focus_active: bool = False,
) -> str:
    base = (
        "Selection mode: invalid/timeout-aware redesign fallback.\n"
        "No fully valid heuristic has been found yet, and the selected parent is not fully valid"
        + (" and appears to have timeout/runtime failures.\n" if parent_timed_out else ".\n")
        + "Do not continue the same broken or expensive structure.\n"
        + "Use the current-run feedback and parent code below only to understand the failure mode.\n"
        + "The parent code is shown for diagnosis, but do not blindly mutate or continue the same broken/expensive structure.\n"
        + "Redesign from scratch if the parent structure is the source of the failure.\n"
        + "The first priority is to become valid on all search instances; then improve the active objective."
    )
    if historical_avoidance_active:
        base += (
            "\nHistorical family avoidance is active, so validity repair must not collapse back to a banned family. "
            "If the invalid parent uses nearest-neighbor, cheapest/regret insertion, or 2-opt-centered cleanup, treat that code as a failure example rather than as a template."
        )
    if family_focus_active:
        base += (
            "\nFamily-focus mode is active. Repair validity while staying inside the currently locked family. "
            "Do not escape the family block just because the parent is invalid, slow, or low quality."
        )
    return base


def _selection_instruction(
    strategy: str,
    parent_is_valid: bool,
    historical_avoidance_active: bool = False,
    family_focus_active: bool = False,
) -> str:
    strategy = normalized_selection_strategy(strategy)
    if family_focus_active:
        if strategy == "1+1":
            if parent_is_valid:
                return (
                    "Selection mode: 1+1 family-focused exploitation.\n"
                    "The selected parent below is the current best-so-far full-valid heuristic within this same focus family only. "
                    "Use it as a local score/validity reference for this family block. Improve or redesign it while preserving the locked family as the main mechanism. "
                    "Do not switch to nearest-neighbor, cheapest/regret insertion, simple greedy construction, or 2-opt-centered cleanup as the core idea. "
                    "A lower score is useful, but this block is primarily testing whether this specific family can be made valid, scalable, and competitive."
                )
            return (
                "Selection mode: 1+1 family-focused partial-validity fallback.\n"
                "No fully valid heuristic has been found yet inside this focus family. The selected parent below is only a partial/latest candidate from this same family block. "
                "Your first priority is to return a valid permutation on all search instances while staying inside the locked family. "
                "Do not repair validity by switching to nearest-neighbor, cheapest/regret insertion, or 2-opt-centered cleanup."
            )
        if parent_is_valid:
            return (
                "Selection mode: 1,1 family-focused sequential chain.\n"
                "The selected parent below is the most recent heuristic inside this same focus family block. "
                "Continue the chain by improving the locked family, not by changing family. "
                "Do not merely rename the parent, tune constants, add restarts, or add a cleanup step while abandoning the declared mechanism."
            )
        return (
            "Selection mode: 1,1 family-focused invalid-parent repair.\n"
            "The selected parent below is the most recent heuristic inside this same focus family block and it may be invalid or only partially valid. "
            "Use the feedback to repair validity while preserving the locked family as the main mechanism. "
            "Do not escape to nearest-neighbor, cheapest/regret insertion, or 2-opt-centered cleanup."
        )

    if historical_avoidance_active:
        if strategy == "1+1":
            if parent_is_valid:
                return (
                    "Selection mode: 1+1 elitist improvement with historical family avoidance.\n"
                    "The selected parent below is the current best-so-far full-valid heuristic under the active objective, "
                    "but in this run it is mainly a score/validity reference, not a mechanism to preserve. "
                    "Do not keep the parent structure merely because it is currently best. "
                    "If the parent belongs to a banned historical family, such as nearest-neighbor, cheapest/regret insertion, "
                    "simple greedy construction, or 2-opt-centered cleanup, redesign the main construction mechanism instead of mutating it. "
                    "A lower score is useful, but the primary experimental goal is to test whether a genuinely different family can be generated while staying valid and scalable."
                )
            return (
                "Selection mode: 1+1 with partial-validity fallback and historical family avoidance.\n"
                "No fully valid heuristic has been found yet. The selected parent below is only a partial/latest candidate and must not anchor the search. "
                "Your first priority is to return a valid permutation on all search instances, but do so with a main mechanism that respects the historical family-avoidance constraints. "
                "Do not repair validity by falling back to nearest-neighbor, cheapest/regret insertion, or 2-opt-centered cleanup."
            )
        if parent_is_valid:
            return (
                "Selection mode: 1,1 sequential mutation chain with historical family avoidance.\n"
                "The selected parent below is the most recent heuristic in the chain, not necessarily the best-so-far, "
                "and it is a reference point rather than a structure to preserve. "
                "Make a genuine family-level change when the current parent belongs to a banned or over-produced mechanism. "
                "Do not merely rename the parent, tune constants, add restarts, or add a small cleanup step to the same family. "
                "The goal is to continue the chain with a valid, scalable heuristic from a structurally different construction family."
            )
        return (
            "Selection mode: 1,1 sequential mutation chain with invalid-parent repair and historical family avoidance.\n"
            "The selected parent below is the most recent heuristic in the chain and it may be invalid or only partially valid. "
            "Use the feedback to understand the failure, but do not preserve a banned or over-produced family while repairing it. "
            "Your first priority is validity; your second priority is to keep the main mechanism structurally different from nearest-neighbor, cheapest/regret insertion, and 2-opt-centered cleanup."
        )

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
    family_focus_memory: str | None = None,
) -> str:
    """Build the TSP LLaMEA prompt using the same structure as clustering."""
    base = base_task_prompt(config)
    search = config.get("search", {})
    strategy = normalized_selection_strategy(search.get("selection_strategy", "1+1"))

    if parent_summary is None:
        parent_summary = {}

    historical_memory = historical_memory or ""
    family_focus_memory = family_focus_memory or ""
    historical_avoidance_active = bool(historical_memory.strip())
    family_focus_active = bool(family_focus_memory.strip())

    if prompt_mode == "initial" or not parent_summary:
        return f"""
{base}

{historical_memory}

{family_focus_memory}

Generate the first heuristic for this active objective now.
""".strip()

    parent_json = json.dumps(parent_summary, indent=2, ensure_ascii=False)

    if prompt_mode == "redesign_invalid_parent":
        instruction = _redesign_instruction(
            parent_timed_out=parent_timed_out,
            historical_avoidance_active=historical_avoidance_active,
            family_focus_active=family_focus_active,
        )
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

{family_focus_memory}

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
    instruction = _selection_instruction(
        strategy,
        parent_is_valid=parent_is_valid,
        historical_avoidance_active=historical_avoidance_active,
        family_focus_active=family_focus_active,
    )
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

{family_focus_memory}

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
