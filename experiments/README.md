# Experiments

Use this directory for small curated summaries only.

Full generated run outputs should go to Google Drive and are ignored by git by default:

```text
/content/drive/MyDrive/TM/llm-tsp-runs/
```

Recommended artifact contents per run:

- `candidates.jsonl`
- `generated_attempts.csv`
- `raw_llm_responses/`
- `generated_code/`
- `instance_level_summary.csv`
- `split_level_summary.csv`
- `overall_summary.csv`
- zipped bundle for final preservation

## Selected TSP heuristics for final evaluation

Final selected TSP heuristics are stored in `selected_tsp_heuristics_final_by_signal/`, grouped by available signal: distance-only, candidate-list, edge-prior, and edge-prior + candidate-list.

