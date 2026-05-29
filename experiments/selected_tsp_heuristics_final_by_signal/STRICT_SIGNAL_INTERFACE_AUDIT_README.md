# Strict signal-interface audit

The selected TSP folders are historical artifacts from several LLM runs. A final
server evaluation should compare methods under a strict interface: a heuristic in
`distance_only` must not call `problem.cand`, `problem.neighbors()`, or
`problem.prior()`. The audit file in this folder records the static scan used by
`server_eval/run_selected_tsp_eval.py`.

In strict mode, the evaluator skips selected heuristics whose code requests
signals unavailable in their nominal folder. At the time of this audit:

- `distance_only/01_legacy_llamea_best_generalized_ba4` accesses
  `problem.cand`, so it is not a strict distance-only heuristic.
- `distance_only/06_hist_raw_novel_best_124447_iter019` accesses
  `problem.neighbors()` and `problem.prior()`, so it is not strict distance-only.

They are kept in the repository for provenance, but excluded by default from the
strict final comparison. Use `ALLOW_INTERFACE_MISMATCH=1` only for diagnostic
replay of the historical selected set.
