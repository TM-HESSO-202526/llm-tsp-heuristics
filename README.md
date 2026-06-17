# LLM TSP Heuristics

This repository contains the code used for the TSP branch of the thesis experiments on LLM-generated constructive heuristics. It includes the generation pipeline, prompt material, selected/generated heuristic implementations, reference summaries, and the final Python evaluator used for server-side evaluation.

This repository is a final thesis artifact, not a general-purpose TSP solver library. It contains the final selected heuristics evaluated in the report, appendix and diagnostic heuristics kept for traceability, prompt-reference material, and scripts needed to reproduce the reported evaluations. By default, the final evaluator only runs the report-selected methods.

## Repository structure

- `src/llm_tsp/` — reusable Python package for TSP instances, distances, candidate sets, priors, prompts, and evaluation helpers.
- `configs/` — YAML configurations for the LLM-generation runs and large-instance suite.
- `notebooks/` — Colab launcher used during generation.
- `docs/` — methodology notes and exact prompt-reference notebooks.
- `experiments/selected_tsp_heuristics_final_by_signal/` — curated Python heuristic implementations grouped by signal regime.
- `server_eval/run_selected_tsp_eval.py` — final Python evaluator for selected heuristics.
- `data/` — instance-suite manifests, reference optima metadata, and an external-input manifest for required `.tsp`, `.cand`, and `.npz` files.

## Selected and appendix methods

The selected TSP heuristics are stored under:

```text
experiments/selected_tsp_heuristics_final_by_signal/
```

The file `APPENDIX_METHODS_INDEX.csv` maps the TSP appendix method cards to their corresponding Python source files. Appendix-only exploratory methods are included in the same folder tree for traceability, but they are not all part of the final competitive selected set.

## Final evaluation

The cleaned server evaluator is:

```bash
python server_eval/run_selected_tsp_eval.py --help
```

The evaluator expects the TSPLIB instance files and, for signal-based regimes, the precomputed POPMUSIC candidate and edge-prior files described in `data/external_inputs_manifest.csv`, `docs/popmusic_candidates.md`, and `data/README.md`.

## Tests

Run the repository tests from the project root with:

```bash
python -m pytest -q
```


## Prompt material

The exact prompt blocks referenced by the report are provided in:

```text
docs/prompt_reference/exact_tsp_prompt_blocks.ipynb
```

## Notes

The repository keeps the Python implementations used for the thesis report. Earlier evaluator prototypes and server-specific launch wrappers were removed from this final-submission version to keep the repository focused and reproducible.
