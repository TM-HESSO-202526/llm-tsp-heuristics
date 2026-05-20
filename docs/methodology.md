# Methodology

This repository keeps the cleaned TSP part of the thesis.

## Scope

The repository focuses on large TSPLIB instances, not the early small 100-city experiments. The default suite is:

- train: `dsj1000`, `pr1002`, `d1291`
- validation: `fl1400`, `pcb1173`
- test: `rl1304`, `u1817`

## Loop

The loop follows a LLaMEA-like pattern:

1. Build a prompt describing the TSP interface and previous results.
2. Ask the LLM to generate one Python heuristic.
3. Extract the first fenced Python block.
4. Validate the code and instantiate the required class.
5. Evaluate raw/final tour gaps on a split.
6. Save the code, raw response, metrics, and error trace.
7. Feed useful feedback into the next call.

## Standard generated-code interface

Generated candidates must follow the same class-based style as the clustering repo. They should return a response with `# Name:` and `# Code:` and the code must define:

```python
class TSPHeuristic:
    def __call__(self, problem, rng=None):
        """Return a permutation of nodes 0..problem.n-1."""
```

The backend parses the `# Name:` field and infers a mechanism family from the name and code for logging, exactly as the clustering pipeline does. The LLM is not asked to provide a separate family variable.

The `problem` object exposes:

- `problem.n`
- `problem.edge_cost(i, j)`
- `problem.neighbors(i)`
- `problem.prior(i, j)`
- `problem.coords`

When POPMUSIC candidate mode is active, the LLM receives sparse candidate lists through `problem.neighbors(i)`, not a full dense distance matrix. `problem.edge_cost(i, j)` remains a true TSPLIB edge-cost oracle for individual edges. Candidate lists guide construction, but the returned tour is a normal TSP permutation and is evaluated on the true full TSPLIB distance.

## Why POPMUSIC matters

The main TSP thesis lesson was that unrestricted LLM generation often rediscovers standard constructive families. The POPMUSIC/LKH layer provides operational structure: candidate lists and tour-frequency edge priors. The LLM can then use a more informative construction environment rather than starting from an unconstrained dense graph.

## LLaMEA parent-selection strategies

The cleaned TSP loop keeps the same parent-selection controls as the clustering launcher:

- `SELECTION_STRATEGY = "1+1"`: elitist mode. The parent is the best full-valid candidate seen so far; before a full-valid candidate exists, the loop falls back to the best partial candidate by penalized score, then the latest candidate.
- `SELECTION_STRATEGY = "1,1"`: sequential mode. The parent is the latest previous candidate.

Before any full-valid parent exists, `INVALID_PARENT_REDESIGN` can trigger a redesign prompt using the latest invalid or partially valid candidate as diagnostic material. The exposed controls match the clustering pipeline: `INVALID_PARENT_REDESIGN`, `REDESIGN_ON_ANY_INVALID_BEFORE_FULL_VALID`, `REDESIGN_ON_TIMEOUT_PARENT`, and `HIDE_INVALID_PARENT_CODE`. Parent-code exposure is therefore controlled by `HIDE_INVALID_PARENT_CODE`, rather than by extra TSP-specific feedback switches.

The family-memory controls also use the clustering terminology: `HISTORICAL_FAMILY_AVOIDANCE`, `FAMILY_NOVELTY_MODE`, `FAMILY_MEMORY_LIMIT`, `MIN_FAMILY_ATTEMPTS_BEFORE_AVOID`, `WEAK_FAMILY_SCORE_THRESHOLD`, and `ALLOW_STRONG_FAMILY_EXPLOITATION`. They are implemented but disabled by default, so no family-memory block is injected into the prompt unless explicitly enabled.
