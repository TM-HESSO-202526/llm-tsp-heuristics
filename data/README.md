# Data policy

This repository does **not** commit TSPLIB `.tsp` files, POPMUSIC/LKH `.cand` files, edge-prior `.npz` files, LKH `.tour` files, `.par` files, logs, or server work folders. These files are external inputs or generated cache artifacts.

The repo keeps only lightweight manifests and reference metadata:

- `tsp_instances_opt.csv` — instance names, splits, and reference tour lengths used for gap computation.
- `instance_suite_1kplus.csv` — cleaned instance-suite metadata used by the pipeline.
- `external_inputs_manifest.csv` — expected external filenames for TSPLIB instances, POPMUSIC candidate files, and edge-prior caches.

## Expected local layout

Place TSPLIB `.tsp` files in either:

```text
data/raw/
```

or in the Drive path used by the configs:

```text
/content/drive/MyDrive/TM/TSP_instances
```

Place POPMUSIC candidate files in:

```text
/content/drive/MyDrive/TM/LKH_candidate_cache
```

Place edge-prior `.npz` files in:

```text
/content/drive/MyDrive/TM/LKH_edge_prior_cache
```

The server evaluator accepts equivalent local paths through `--instance-root`, `--candidate-cache-dir`, and `--edge-prior-cache-dir`.

## Instance coverage

The cleaned final TSP metadata includes the original train/validation/test instances and extended larger instances. Distance-only evaluation requires only the corresponding `.tsp` file. Candidate-list evaluation additionally requires the POPMUSIC `.cand` file. Edge-prior evaluation requires the corresponding `.npz` prior cache. See `external_inputs_manifest.csv` for the exact expected filenames and for cache availability notes.

The external cache files are intentionally not bundled in the final repository. This keeps the public repo focused on source code, selected heuristics, prompt material, metadata, and reproducibility instructions rather than redistributing TSPLIB inputs or LKH-generated artifacts.
