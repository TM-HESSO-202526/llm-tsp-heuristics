# TSP server evaluation

This folder mirrors the clustering `server_eval/` workflow for the final TSP evaluation.

## What it evaluates

Selected heuristics are read from:

```text
experiments/selected_tsp_heuristics_final_by_signal/
```

Signal modes:

```text
distance_only
candidate_list
edge_prior
edge_prior_plus_candidate_list
all
```

The evaluator writes:

```text
raw_results.csv
summary_by_heuristic.csv
summary_by_heuristic_instance.csv
summary_by_instance_size.csv
complexity_fit.csv
run_config.json
```

It supports resume and per-case timeout.

## External input files needed

Keep these private outside GitHub, for example:

```text
D:\Users\antho\TM\server_eval_tsp_inputs\
├── TSP_instances\
│   ├── dsj1000.tsp
│   ├── pr1002.tsp
│   ├── d1291.tsp
│   ├── fl1400.tsp
│   ├── pcb1173.tsp
│   ├── rl1304.tsp
│   └── u1817.tsp
├── LKH_candidate_cache\
│   ├── dsj1000_cand-popmusic-k20-s14-sol20-nn5-tr1.cand
│   └── ...
└── LKH_edge_prior_cache\
    ├── dsj1000_popmusic_edge_prior_runs30_topk5.npz
    └── ...
```

The optimum/reference values are already in the repo:

```text
data/tsp_instances_opt.csv
```

So you do not need a separate reference file unless you want to override that CSV.

Candidate files are needed for:

```text
candidate_list
edge_prior_plus_candidate_list
```

Edge-prior `.npz` files are needed for:

```text
edge_prior
edge_prior_plus_candidate_list
```

Distance-only runs need only the `.tsp` files and the optima CSV.
