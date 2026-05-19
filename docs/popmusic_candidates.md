# POPMUSIC / LKH candidate sets

This repo supports a switchable POPMUSIC/LKH candidate layer.

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

The clean repo should allow these modes to be changed by config only, just like the clustering repo allows sampling/decomposition modes to be toggled by config.
