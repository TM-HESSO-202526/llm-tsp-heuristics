# Prompt design

The TSP prompt mirrors the clustering repo methodology as closely as possible while keeping the problem-specific TSP interface.

## Required generated-code interface

The LLM returns the answer in the same `# Name:` / `# Code:` format used by the clustering pipeline. The code must define exactly one class:

```python
class TSPHeuristic:
    def __call__(self, problem, rng=None):
        ...
```

The evaluator instantiates the class and calls:

```python
algo = TSPHeuristic()
tour = algo(problem, rng)
```

The returned `tour` must be a permutation of `0..problem.n-1`. The evaluator closes the cycle automatically.

## Alignment with clustering

The TSP prompt follows the same high-level methodology as the clustering prompt:

- the base prompt contains the objective, interface, constraints, and `# Name` / `# Code` return format;
- the LLM provides the name, while the backend infers the family from the name and code;
- selection-strategy wording is not embedded in the base prompt;
- `1+1`, `1,1`, and invalid-parent redesign wording is injected dynamically only when a parent exists;
- invalid/partial parent code is shown once, in the same prompt section as clustering, and only hidden when `HIDE_INVALID_PARENT_CODE=True` in invalid-redesign mode;
- `HISTORICAL_FAMILY_AVOIDANCE` optionally injects a fixed historical avoidance block into the prompt.

## TSP-specific objective block

The TSP-specific block tells the LLM that it receives a `problem` object with:

- `problem.n`
- `problem.edge_cost(i, j)`
- `problem.neighbors(i)`
- `problem.prior(i, j)`
- `problem.coords`

When POPMUSIC candidates are active, the prompt tells the LLM that it receives sparse candidate lists through `problem.neighbors(i)`, not a full dense distance matrix. The final tour may still contain non-candidate edges and is evaluated on the true TSPLIB distance. When the edge prior is active, `problem.prior(i, j)` comes from the historical 30-run LKH/POPMUSIC tour-frequency cache, not from the candidate list alone.

## Family-focus mode

`FAMILY_FOCUS_MODE` is an optional exploitation mode for the historical-family-avoidance experiments. Instead of giving the LLM a single mixed prompt containing many alternative families, the run is split into one local block per family. Each family block has its own local parent and local prompt history. At the end of the run, the backend writes `family_focus_summary.csv` and compares the best candidate found in each family.

The family names, objectives, and strict constraints are intentionally kept in the Colab launcher as `FAMILY_FOCUS_PLAN`, so they can be edited quickly without changing backend code. The backend only formats the currently active family into the prompt.

When active, the total number of LLM calls is derived as:

```text
FAMILY_FOCUS_CALLS_PER_FAMILY × number of enabled families
```

For example, with 20 calls and 5 enabled families, the run uses 100 LLM calls. Each block receives a prompt beginning with `Family-focus mode is ACTIVE` and tells the LLM that it is locked to one family, must improve that family, and must not switch back to nearest-neighbor, regret/cheapest insertion, or 2-opt-centered cleanup as the main mechanism.
