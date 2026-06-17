# TSP server evaluation

This folder contains the cleaned final Python evaluator for the selected TSP heuristics, plus a supporting cache builder for large POPMUSIC/LKH candidate and prior files.

## Main script

```bash
python server_eval/run_selected_tsp_eval.py --help
```

The evaluator writes raw per-instance results and summary CSV files to the chosen output directory. It supports the final distance-only, candidate-list, and edge-prior heuristic groups in `experiments/selected_tsp_heuristics_final_by_signal/`.

## Required inputs

- TSPLIB `.tsp` instance files.
- `data/tsp_instances_opt.csv` for reference values.
- POPMUSIC candidate files and edge-prior `.npz` files when evaluating signal-based heuristics; see `data/external_inputs_manifest.csv` for expected filenames.

The helper `build_large_tsp_caches.py` is kept only for preparing large-instance candidate/prior caches. It is not the main final evaluator. The final selected-method evaluator itself is `run_selected_tsp_eval.py`.

## Final-evaluation discovery

The selected TSP evaluator uses `experiments/selected_tsp_heuristics_final_by_signal/INDEX_selected_tsp_heuristics.csv` as the method manifest. By default it runs only methods marked `include_in_final_eval=true`; pass `--include-appendix` only for manual inspection runs of appendix-only methods.
