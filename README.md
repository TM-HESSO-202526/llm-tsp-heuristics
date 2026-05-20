# LLM TSP Heuristics

Colab-first research code for generating and evaluating LLM-generated constructive heuristics for large Traveling Salesman Problem instances.

This repository is the TSP counterpart of the clustering experiments in `llm-clustering-heuristics`. It focuses on a cleaned version of the thesis TSP work: a LLaMEA-style loop for constructive TSP heuristics, optionally combined with LKH/POPMUSIC candidate sets and edge-frequency priors.

The goal is **not** to package every historical notebook or every failed run. The goal is to preserve the thesis-relevant TSP pipeline in a reproducible form.

## Core idea

An LLM proposes executable Python heuristic code. The code is evaluated automatically on a fixed TSPLIB split. The result is fed back to the next LLM call using the same invalid-parent redesign controls as the clustering repo. This is the same research pattern used later in the clustering repo: generate, evaluate, summarize, and iterate.

The TSP-specific addition is a switchable POPMUSIC/LKH layer inside the same LLaMEA loop:

```python
USE_POPMUSIC_CANDIDATES = True
USE_POPMUSIC_EDGE_PRIOR = True
POPMUSIC_PRIOR_MODE = "frequency"  # none, frequency, binary_topk, shuffled
MAX_CANDIDATES = 20
```

With these flags, the same LLaMEA loop can be run in dense mode, candidate-guided mode, or candidate-guided mode with a POPMUSIC edge prior. When candidate mode is active, missing POPMUSIC/LKH candidate files are generated automatically into the cache before the LLaMEA loop evaluates candidates. The LLM receives sparse candidate lists through `problem.neighbors(i)`, not a full dense distance matrix. Candidate edges guide construction, but final returned tours are normal TSP tours and may include non-candidate edges; they are always evaluated on the true full TSPLIB distance.

When edge-prior mode is active, the prior is generated using the historical procedure: 30 short LKH/POPMUSIC runs with `MOVE_TYPE = 5`, `PATCHING_A = 2`, `PATCHING_C = 3`, `TIME_LIMIT = 1.0`, and `OUTPUT_TOUR_FILE`; successful tours are parsed, tour edges are counted symmetrically, and the cache is saved as `/content/drive/MyDrive/TM/LKH_edge_prior_cache/{instance}_popmusic_edge_prior_runs30_topk5.npz`.

## Instance policy

This cleaned repo intentionally ignores the early small-instance experiments. The default split uses only the 1k+ TSPLIB instances used in the later thesis weeks:

| Split | Instances |
|---|---|
| Train | `dsj1000`, `pr1002`, `d1291` |
| Validation | `fl1400`, `pcb1173` |
| Test | `rl1304`, `u1817` |

Optimum values and split metadata are stored in `data/tsp_instances_opt.csv` and `configs/tsp_large_suite.yaml`.

## Recommended Colab workflow

```python
from google.colab import drive
drive.mount('/content/drive')

!git clone https://github.com/TM-HESSO-202526/llm-tsp-heuristics.git
%cd llm-tsp-heuristics
!pip install -r requirements.txt

!python scripts/run_unified_tsp_pipeline.py --config configs/run_llamea_popmusic_candidates.yaml
```

For a smoke test without TSPLIB files:

```bash
python scripts/run_unified_tsp_pipeline.py --config configs/run_llamea_dense.yaml --dry-run
```

## Important configuration variables

The unified runner and notebook launcher expose the same LLaMEA-style controls used in the clustering work:

```python
LLM_PROVIDER = "groq"
LLM_MODEL = "llama-3.3-70b-versatile"
MAX_LLM_CALLS = 40
GLOBAL_SEED = 12345
CANDIDATE_TIMEOUT_S = 60
EVALUATION_TIMEOUT_S = 120

SELECTION_STRATEGY = "1+1"       # "1+1" = elitist best-so-far parent; "1,1" = latest sequential parent
HISTORY_LIMIT = 20
INVALID_PARENT_REDESIGN = True
REDESIGN_ON_ANY_INVALID_BEFORE_FULL_VALID = True
REDESIGN_ON_TIMEOUT_PARENT = True
HIDE_INVALID_PARENT_CODE = False

HISTORICAL_FAMILY_AVOIDANCE = False
FAMILY_NOVELTY_MODE = False
FAMILY_MEMORY_LIMIT = 8
MIN_FAMILY_ATTEMPTS_BEFORE_AVOID = 5
WEAK_FAMILY_SCORE_THRESHOLD = 20.0
ALLOW_STRONG_FAMILY_EXPLOITATION = True
```

There is intentionally no separate experiment-mode variable anymore. This repo always runs the TSP LLaMEA loop; dense mode, candidate mode, and POPMUSIC-prior mode are controlled by the POPMUSIC flags.

The intent is that the TSP loop remains flexible enough to reproduce early LLaMEA-style experiments, while using the same cleaner runtime-control style as the clustering repo.

## Repository layout

```text
configs/       YAML configs for dense LLaMEA and POPMUSIC-enabled LLaMEA runs.
data/          Instance split/optimum metadata and placeholders for local TSPLIB files.
docs/          Methodology, prompt design, POPMUSIC notes, and relation to clustering.
experiments/   Small curated summaries only; full generated artifacts should stay in Drive.
notebooks/     Colab launcher and archived reference notebooks.
scripts/       CLI entrypoints for unified runs, candidate building, and summaries.
src/llm_tsp/   Reusable Python package.
tests/         Small tests/smoke checks.
```

## Generated artifacts

Real experiment outputs should go to Google Drive, typically:

```text
/content/drive/MyDrive/TM/llm-tsp-runs/
```

Each run folder stores prompts, raw LLM responses, generated code, per-instance details, summaries, and a zipped artifact bundle. Artifact names intentionally mirror the clustering repo style:

```text
codes/iter_*.py
prompts/prompt_iter_*.txt
raw_responses/raw_iter_*.txt
llm_attempts.csv
llm_search_instance_rows.csv
search_detail_iter_*.csv
llm_best_attempts_top20.csv
llm_family_summary.csv
best_candidate_code.py
best_candidate_summary.json
```

Legacy aliases such as `generated_attempts.csv`, `generated_code/`, and `raw_llm_responses/` are still written for convenience. The `experiments/runs/` directory is ignored by git by default.

## What this repo deliberately excludes

This first clean version does **not** include the fixed scaffold/hook experiments. The public TSP story is centered on:

1. the LLaMEA-style generation loop;
2. the 1k+ TSPLIB split;
3. optional POPMUSIC/LKH candidate sets generated/cached with the historical LKH-3.0.8 workflow;
4. optional POPMUSIC/LKH tour-frequency edge-prior information generated from 30 short LKH runs;
5. selected clean summaries from historical prior/candidate experiments.

## Thesis framing

The TSP experiments showed that unrestricted LLM generation tends to rediscover common constructive families such as nearest-neighbor, insertion, and regret-style heuristics. The more useful thesis signal came from combining LLM generation with operational structure: large-instance splits, candidate-guided construction, POPMUSIC/LKH candidate sets, and explicit edge-prior information. These lessons motivated the later clustering repo design.


## Unified Colab launcher

The main notebook is `notebooks/00_tsp_colab_launcher.ipynb`. It follows the same launcher philosophy as the clustering repository: one control panel, Drive mounting, repo refresh, editable install, runtime-config generation, file checks, live logs, artifact summaries, and optional artifact zip download.
