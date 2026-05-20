# POPMUSIC / LKH candidate sets

This repo supports a switchable POPMUSIC/LKH candidate layer. If candidate mode is active and a candidate file is missing, the main pipeline generates it immediately with LKH/POPMUSIC and saves it to the cache. There is no silent kNN fallback.

Typical thesis settings:

```yaml
max_candidates: 20
popmusic_sample_size: 14
popmusic_solutions: 20
popmusic_max_neighbors: 5
popmusic_trials: 1
popmusic_initial_tour: false
```

The candidate cache default is:

```text
/content/drive/MyDrive/TM/LKH_candidate_cache
```

## Modes

- `none`: dense TSP access.
- `candidates_only`: candidate lists are available, but no frequency prior is exposed.
- `frequency`: expose edge-frequency priors from POPMUSIC/LKH outputs.
- `binary_topk`: keep only top-k prior edges.
- `shuffled`: sanity check / weakened prior.

Candidate mode is guidance for construction. The generated heuristic should use `problem.neighbors(i)` and the prior signal when available, but the final returned tour is a normal Hamiltonian cycle and may include non-candidate edges. Final cost is always computed on the true full TSPLIB distance.
