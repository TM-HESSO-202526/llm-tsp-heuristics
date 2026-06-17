# POPMUSIC / LKH candidate sets and edge priors

This repo supports the same POPMUSIC/LKH methodology used in the historical TSP notebooks. If candidate mode is active and a candidate file is missing, the main pipeline generates the official LKH `CANDIDATE_FILE` immediately with `CANDIDATE_SET_TYPE = POPMUSIC` and saves it to the Drive cache. There is no silent kNN fallback.

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


## Historical edge-prior cache

When `USE_POPMUSIC_EDGE_PRIOR=True`, the prior is not derived from the candidate list itself. It is built using the historical LKH/POPMUSIC tour-frequency procedure:

```text
CANDIDATE_SET_TYPE = POPMUSIC
MAX_CANDIDATES = 20
POPMUSIC_SAMPLE_SIZE = 14
POPMUSIC_SOLUTIONS = 20
POPMUSIC_MAX_NEIGHBORS = 5
POPMUSIC_TRIALS = 1
POPMUSIC_INITIAL_TOUR = NO
RUNS = 1
MOVE_TYPE = 5
PATCHING_A = 2
PATCHING_C = 3
SEED = ...
TIME_LIMIT = 1.0
TRACE_LEVEL = 0
OUTPUT_TOUR_FILE = ...
```

The backend runs this short LKH configuration 30 times by default, parses each successful output tour, counts tour edges symmetrically, normalizes by the number of successful runs, keeps the historical top-k cache convention, and stores the result as:

```text
/content/drive/MyDrive/TM/LKH_edge_prior_cache/{instance}_popmusic_edge_prior_runs30_topk5.npz
```

This gives two separate caches:

1. `/content/drive/MyDrive/TM/LKH_candidate_cache` for POPMUSIC candidate files.
2. `/content/drive/MyDrive/TM/LKH_edge_prior_cache` for 30-run LKH tour-frequency priors.


## External input manifest

The final repository does not bundle `.tsp`, `.cand`, `.npz`, `.tour`, `.par`, or LKH log files. The expected filenames for reproducing the final evaluator are listed in `data/external_inputs_manifest.csv`.
