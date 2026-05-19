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
4. Validate the code and execute the required function.
5. Evaluate raw/final tour gaps on a split.
6. Save the code, raw response, metrics, and error trace.
7. Feed useful feedback into the next call.

## Standard generated-code interface

Generated candidates should define:

```python
def construct_tour(problem, rng=None):
    """Return a permutation of nodes 0..problem.n-1."""
```

The `problem` object exposes:

- `problem.n`
- `problem.edge_cost(i, j)`
- `problem.neighbors(i)`
- `problem.prior(i, j)`
- `problem.coords`

If candidate restriction is active, `edge_cost(i, j)` is only guaranteed for candidate edges. This encourages heuristics to stay within sparse candidate sets.

## Why POPMUSIC matters

The main TSP thesis lesson was that unrestricted LLM generation often rediscovers standard constructive families. The POPMUSIC/LKH layer provides operational structure: candidate edges and edge-frequency priors. The LLM can then use a more informative construction environment rather than starting from an unconstrained dense graph.
