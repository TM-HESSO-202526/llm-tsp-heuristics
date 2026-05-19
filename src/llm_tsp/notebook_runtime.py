from __future__ import annotations

from pathlib import Path
from typing import Any
import tempfile
import yaml


DEFAULTS: dict[str, Any] = {
    "RUN_NAME": "tsp_notebook_run",
    "EXPERIMENT_MODE": "llamea",
    "LLM_PROVIDER": "groq",
    "LLM_MODEL": "llama-3.3-70b-versatile",
    "MAX_LLM_CALLS": 40,
    "GLOBAL_SEED": 12345,
    "CANDIDATE_TIMEOUT_S": 60,
    "EVALUATION_TIMEOUT_S": 120,
    "EVAL_SPLIT": "train",
    "USE_POPMUSIC_CANDIDATES": False,
    "USE_POPMUSIC_EDGE_PRIOR": False,
    "POPMUSIC_PRIOR_MODE": "none",
    "MAX_CANDIDATES": 20,
    "RESTRICT_EDGE_COST_TO_CANDIDATES": False,
    "INCLUDE_INVALID_CODE_IN_FEEDBACK": True,
    "INCLUDE_INVALID_ERROR_TRACE": True,
    "INCLUDE_PARENT_CODE_IN_MUTATION_PROMPT": True,
    "SAVE_RAW_LLM_RESPONSES": True,
    "SAVE_GENERATED_ATTEMPTS": True,
    "INSTANCE_ROOT": "/content/drive/MyDrive/TM/TSP_instances",
    "CANDIDATE_CACHE_DIR": "/content/drive/MyDrive/TM/LKH_candidate_cache",
    "ARTIFACT_ROOT": "/content/drive/MyDrive/TM/llm-tsp-runs",
}


def build_runtime_config_from_notebook_globals(globals_dict: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Build a temporary YAML config from notebook-level variables.

    This mirrors the robust pattern used in the clustering repo: all important
    experiment toggles can be set at the top of the notebook, converted into an
    effective config, and printed before launching the run.
    """
    values = {k: globals_dict.get(k, v) for k, v in DEFAULTS.items()}
    effective = {
        "run_name": values["RUN_NAME"],
        "experiment_mode": values["EXPERIMENT_MODE"],
        "llm": {
            "provider": values["LLM_PROVIDER"],
            "model": values["LLM_MODEL"],
            "max_llm_calls": values["MAX_LLM_CALLS"],
        },
        "runtime": {
            "global_seed": values["GLOBAL_SEED"],
            "candidate_timeout_s": values["CANDIDATE_TIMEOUT_S"],
            "evaluation_timeout_s": values["EVALUATION_TIMEOUT_S"],
            "eval_split": values["EVAL_SPLIT"],
        },
        "feedback": {
            "include_invalid_code_in_feedback": values["INCLUDE_INVALID_CODE_IN_FEEDBACK"],
            "include_invalid_error_trace": values["INCLUDE_INVALID_ERROR_TRACE"],
            "include_parent_code_in_mutation_prompt": values["INCLUDE_PARENT_CODE_IN_MUTATION_PROMPT"],
            "save_raw_llm_responses": values["SAVE_RAW_LLM_RESPONSES"],
            "save_generated_attempts": values["SAVE_GENERATED_ATTEMPTS"],
        },
        "suite": {
            "instance_root": values["INSTANCE_ROOT"],
            "candidate_cache_dir": values["CANDIDATE_CACHE_DIR"],
            "artifact_root": values["ARTIFACT_ROOT"],
            "splits": {
                "train": ["dsj1000", "pr1002", "d1291"],
                "val": ["fl1400", "pcb1173"],
                "test": ["rl1304", "u1817"],
            },
            "optima": {
                "dsj1000": 18659688,
                "pr1002": 259045,
                "d1291": 50801,
                "fl1400": 20127,
                "pcb1173": 56892,
                "rl1304": 252948,
                "u1817": 57201,
            },
        },
        "popmusic": {
            "use_popmusic_candidates": values["USE_POPMUSIC_CANDIDATES"],
            "use_popmusic_edge_prior": values["USE_POPMUSIC_EDGE_PRIOR"],
            "prior_mode": values["POPMUSIC_PRIOR_MODE"],
            "max_candidates": values["MAX_CANDIDATES"],
            "restrict_edge_cost_to_candidates": values["RESTRICT_EDGE_COST_TO_CANDIDATES"],
        },
    }
    out = Path(tempfile.gettempdir()) / f"{values['RUN_NAME']}_runtime_config.yaml"
    with out.open("w", encoding="utf-8") as f:
        yaml.safe_dump(effective, f, sort_keys=False)
    return str(out), effective


def print_effective_config(effective: dict[str, Any]) -> None:
    print("Effective TSP runtime config")
    print("-" * 72)
    print(f"run_name: {effective.get('run_name')}")
    print(f"experiment_mode: {effective.get('experiment_mode')}")
    llm = effective.get("llm", {})
    runtime = effective.get("runtime", {})
    pop = effective.get("popmusic", {})
    print(f"llm: {llm.get('provider')} / {llm.get('model')}")
    print(f"max_llm_calls: {llm.get('max_llm_calls')}")
    print(f"eval_split: {runtime.get('eval_split')}")
    print(f"candidate_timeout_s: {runtime.get('candidate_timeout_s')}")
    print(f"evaluation_timeout_s: {runtime.get('evaluation_timeout_s')}")
    print(f"use_popmusic_candidates: {pop.get('use_popmusic_candidates')}")
    print(f"use_popmusic_edge_prior: {pop.get('use_popmusic_edge_prior')}")
    print(f"popmusic_prior_mode: {pop.get('prior_mode')}")
    print(f"max_candidates: {pop.get('max_candidates')}")
    print(f"restrict_edge_cost_to_candidates: {pop.get('restrict_edge_cost_to_candidates')}")
    print("-" * 72)
