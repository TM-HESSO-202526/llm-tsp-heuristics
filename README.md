# LLM TSP Heuristics

Colab-first research code for generating and evaluating LLM-generated constructive heuristics for large Traveling Salesman Problem instances.

This repository is the TSP counterpart of the clustering experiments in `llm-clustering-heuristics`. It focuses on a cleaned version of the thesis TSP work: a LLaMEA-style loop for constructive TSP heuristics, optionally combined with LKH/POPMUSIC candidate sets and edge-frequency priors.

The goal is **not** to package every historical notebook or every failed run. The goal is to preserve the thesis-relevant TSP pipeline in a reproducible form.

## Core idea

An LLM proposes executable Python heuristic code. The code is evaluated automatically on a fixed TSPLIB split. The result is fed back to the next LLM call, optionally including invalid code and error traces. This is the same research pattern used later in the clustering repo: generate, evaluate, summarize, and iterate.

The TSP-specific addition is a switchable POPMUSIC/LKH layer:

```python
USE_POPMUSIC_CANDIDATES = True
USE_POPMUSIC_EDGE_PRIOR = True
POPMUSIC_PRIOR_MODE = "frequency"  # none, candidates_only, frequency, binary_topk, shuffled
MAX_CANDIDATES = 20
```

With these flags, the same LLaMEA loop can be run in dense mode, candidate-restricted mode, or candidate-restricted mode with a POPMUSIC edge prior.

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

The unified runner and notebook launcher expose the variables that were useful during the clustering work too:

```python
LLM_PROVIDER = "groq"
LLM_MODEL = "llama-3.3-70b-versatile"
MAX_LLM_CALLS = 40
GLOBAL_SEED = 12345
CANDIDATE_TIMEOUT_S = 60
EVALUATION_TIMEOUT_S = 120

INCLUDE_INVALID_CODE_IN_FEEDBACK = True
INCLUDE_INVALID_ERROR_TRACE = True
INCLUDE_PARENT_CODE_IN_MUTATION_PROMPT = True
SAVE_RAW_LLM_RESPONSES = True
SAVE_GENERATED_ATTEMPTS = True
```

The intent is that the TSP loop remains flexible enough to reproduce early LLaMEA-style experiments, while using the same cleaner runtime-control style as the clustering repo.

## Repository layout

```text
configs/       YAML configs for dense LLaMEA, POPMUSIC LLaMEA, and prior ablation runs.
data/          Instance split/optimum metadata and placeholders for local TSPLIB files.
docs/          Methodology, prompt design, POPMUSIC notes, and relation to clustering.
experiments/   Small curated summaries only; full generated artifacts should stay in Drive.
notebooks/     Colab launcher and archived reference notebooks.
scripts/       CLI entrypoints for unified runs, candidate building, ablations, summaries.
src/llm_tsp/   Reusable Python package.
tests/         Small tests/smoke checks.
```

## Generated artifacts

Real experiment outputs should go to Google Drive, typically:

```text
/content/drive/MyDrive/TM/llm-tsp-runs/
```

Each run folder should store prompts, raw LLM responses, generated code, per-instance details, summaries, and a zipped artifact bundle. The `experiments/runs/` directory is ignored by git by default.

## What this repo deliberately excludes

This first clean version does **not** include the fixed scaffold/hook experiments. The public TSP story is centered on:

1. the LLaMEA-style generation loop;
2. the 1k+ TSPLIB split;
3. optional POPMUSIC/LKH candidate sets;
4. optional POPMUSIC edge-prior information;
5. prior ablations and selected clean summaries.

## Thesis framing

The TSP experiments showed that unrestricted LLM generation tends to rediscover common constructive families such as nearest-neighbor, insertion, and regret-style heuristics. The more useful thesis signal came from combining LLM generation with operational structure: large-instance splits, candidate restrictions, POPMUSIC/LKH candidate sets, and explicit edge-prior information. These lessons motivated the later clustering repo design.
