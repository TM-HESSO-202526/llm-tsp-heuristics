# LLM TSP Heuristics

This repository contains the code used for the TSP branch of the thesis experiments on LLM-generated constructive heuristics. It includes the generation pipeline, exact prompt material, selected/generated heuristic implementations, reference summaries, and the final Python evaluator used for server-side evaluation.

This repository is a final thesis artifact, not a general-purpose TSP solver library. It contains the final selected heuristics evaluated in the report, appendix and diagnostic heuristics kept for traceability, prompt-reference material, and scripts needed to reproduce the reported evaluations. By default, the final evaluator only runs the report-selected methods.

## Quick Colab launcher

The TSP generation pipeline can be launched from the unified Colab notebook:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TM-HESSO-202526/llm-tsp-heuristics/blob/main/notebooks/00_tsp_colab_launcher.ipynb)

The notebook is the easiest way to inspect and test the generation setup. It lets the user configure the TSP signal regime, generation variables, historical family-avoidance options, family-focus settings, and evaluation parameters from a single place.

To run the LLM-generation cells, a Groq API key is required. In Colab, add the key as an environment variable or secret named:

```text
GROQ_API_KEY_1
```

Additional keys can be provided as `GROQ_API_KEY_2`, `GROQ_API_KEY_3`, and so on if the user wants to distribute calls across several keys. Groq API keys can be created from the Groq developer console:

```text
https://console.groq.com/keys
```

Without a Groq key, the notebook can still be inspected, but the LLM-generation calls will not run. The notebook also checks for the required TSPLIB `.tsp`, POPMUSIC `.cand`, and edge-prior `.npz` files. If a required file is missing, it either attempts to generate the cache from the available TSPLIB file or opens a Colab upload prompt and copies the uploaded file to the expected Drive location.

## Repository structure

- `src/llm_tsp/` — reusable Python package for TSP instances, distances, candidate sets, priors, prompt construction, and evaluation helpers.
- `configs/` — YAML configurations for the LLM-generation runs and large-instance suite.
- `notebooks/00_tsp_colab_launcher.ipynb` — clean Colab launcher for the TSP generation pipeline.
- `scripts/` — supporting scripts used by the generation/cache workflow.
- `docs/` — prompt-reference notebooks.
- `experiments/selected_tsp_heuristics_final_by_signal/` — curated Python heuristic implementations grouped by signal regime and named with the report IDs.
- `server_eval/run_selected_tsp_eval.py` — final Python evaluator for selected TSP heuristics.
- `data/` — instance-suite manifests, reference optima metadata, and an external-input for required `.tsp`, `.cand`, and `.npz` files.

## Selected heuristic code

The Python code for the report-selected and appendix TSP heuristics is stored in:

```text
experiments/selected_tsp_heuristics_final_by_signal/
```

The folder is organized by signal regime:

```text
experiments/selected_tsp_heuristics_final_by_signal/
├── distance_only/
├── candidate_list/
└── edge_prior/
```

Each heuristic Python file starts with the corresponding report ID, such as `D1`, `C1a`, or `P2`. This makes it possible to map the implementation directly to the method cards and result tables in the thesis report.

The selected-heuristic index files provide compact mappings between report IDs, signal regimes, direct source files, final-evaluation inclusion flags, and short method descriptions:

```text
experiments/selected_tsp_heuristics_final_by_signal/INDEX_selected_tsp_heuristics.csv
```

## Signal regimes

- `distance_only` — generated heuristics using only TSPLIB coordinates/distances.
- `candidate_list` — generated heuristics using POPMUSIC/LKH candidate-neighbour lists.
- `edge_prior` — generated heuristics using sparse edge-prior information derived from repeated short LKH/POPMUSIC runs.

## Final evaluation

The cleaned selected-heuristic evaluator is:

```bash
python server_eval/run_selected_tsp_eval.py --help
```

It evaluates the selected heuristics over the TSP benchmark instances and writes raw result and summary CSV files to the chosen output directory. By default, it uses `INDEX_selected_tsp_heuristics.csv` and only evaluates rows marked `include_in_final_eval=true`. Appendix-only and failed diagnostic methods remain available for inspection, but they are not run by a normal full evaluation command.

The evaluator expects the TSPLIB instance files and, for signal-based regimes, the precomputed POPMUSIC candidate and edge-prior files described in:

```text
data/external_inputs_manifest.csv
data/README.md
docs/popmusic_candidates.md
```

## Tests

Run the repository tests from the project root with:

```bash
python -m pytest -q
```

## Prompt material

The exact TSP prompt blocks referenced by the report are provided in:

```text
docs/prompt_reference/exact_tsp_prompt_blocks.ipynb
```

These files document the system prompt, task prompts, required interface, historical family-avoidance instructions, and family-focus prompt material used during generation.

## Data inputs

The repository does not store TSPLIB `.tsp` files, POPMUSIC `.cand` caches, edge-prior `.npz` caches, `.tour` files, `.par` files, logs, or server work folders. These are treated as external inputs or generated artifacts.

The required filenames and expected locations are documented in:

```text
data/external_inputs_manifest.csv
```

The generated run artifacts themselves are not stored in this final-submission repository.
