from __future__ import annotations

from pathlib import Path
from typing import Any
import tempfile
import yaml


DEFAULTS: dict[str, Any] = {
    # Run identity
    "RUN_NAME": "tsp_llamea_popmusic_train",
    "EXPERIMENT_MODE": "llamea",
    "SMOKE_TEST": False,
    "DRY_RUN": False,

    # LLM provider
    "LLM_PROVIDER": "groq",
    "LLM_MODEL": "llama-3.3-70b-versatile",
    "MAX_LLM_CALLS": 40,
    "TEMPERATURE": 0.8,
    "TOP_P": 1.0,

    # Provider/key/rate-limit behavior
    "GROQ_MAX_KEYS": 10,
    "LLM_CALLS_PER_MINUTE_PER_KEY": 2,
    "LLM_REQUEST_TIMEOUT_S": 60,
    "MAX_429_RETRIES": 100,
    "MAX_REQUEST_ERROR_RETRIES": 5,

    # Runtime and evaluation
    "GLOBAL_SEED": 12345,
    "CANDIDATE_TIMEOUT_S": 60,
    "EVALUATION_TIMEOUT_S": 120,
    "EVAL_SPLIT": "train",

    # LLaMEA-style feedback controls
    "INCLUDE_INVALID_CODE_IN_FEEDBACK": True,
    "INCLUDE_INVALID_ERROR_TRACE": True,
    "INCLUDE_PARENT_CODE_IN_MUTATION_PROMPT": True,
    "SAVE_RAW_LLM_RESPONSES": True,
    "SAVE_GENERATED_ATTEMPTS": True,

    # TSP data / artifact paths
    "INSTANCE_ROOT": "/content/drive/MyDrive/TM/TSP_instances",
    "CANDIDATE_CACHE_DIR": "/content/drive/MyDrive/TM/LKH_candidate_cache",
    "ARTIFACT_ROOT": "/content/drive/MyDrive/TM/llm-tsp-runs",

    # POPMUSIC/candidate-prior controls
    "USE_POPMUSIC_CANDIDATES": True,
    "USE_POPMUSIC_EDGE_PRIOR": True,
    "POPMUSIC_PRIOR_MODE": "frequency",
    "MAX_CANDIDATES": 20,
    "RESTRICT_EDGE_COST_TO_CANDIDATES": True,
    "ALLOW_NON_CANDIDATE_EDGES_IN_FINAL_TOUR": False,
    "LKH_BINARY_PATH": "/content/tools/lkh/LKH",

    # Reference suite: 1k+ TSPLIB instances only
    "TRAIN_INSTANCES": ["dsj1000", "pr1002", "d1291"],
    "VAL_INSTANCES": ["fl1400", "pcb1173"],
    "TEST_INSTANCES": ["rl1304", "u1817"],
    "OPTIMA": {
        "dsj1000": 18659688,
        "pr1002": 259045,
        "d1291": 50801,
        "fl1400": 20127,
        "pcb1173": 56892,
        "rl1304": 252948,
        "u1817": 57201,
    },
}


def _get(globals_dict: dict[str, Any], key: str) -> Any:
    return globals_dict.get(key, DEFAULTS[key])


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def build_runtime_config_from_notebook_globals(globals_dict: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Build a temporary YAML config from notebook-level variables.

    The launcher notebook is intentionally only a control panel. This function
    converts the variables declared in the notebook into the nested YAML-style
    config consumed by the backend scripts, mirroring the clustering repo pattern.
    """
    values = {k: globals_dict.get(k, v) for k, v in DEFAULTS.items()}
    smoke_test = _as_bool(values["SMOKE_TEST"])
    dry_run = _as_bool(values["DRY_RUN"])
    max_calls = 1 if smoke_test else int(values["MAX_LLM_CALLS"])

    effective = {
        "run_name": values["RUN_NAME"],
        "experiment_mode": values["EXPERIMENT_MODE"],
        "llm": {
            "provider": values["LLM_PROVIDER"],
            "model": values["LLM_MODEL"],
            "max_llm_calls": max_calls,
            "temperature": float(values["TEMPERATURE"]),
            "top_p": float(values["TOP_P"]),
            "groq_max_keys": int(values["GROQ_MAX_KEYS"]),
            "calls_per_minute_per_key": float(values["LLM_CALLS_PER_MINUTE_PER_KEY"]),
            "request_timeout_s": float(values["LLM_REQUEST_TIMEOUT_S"]),
            "max_429_retries": int(values["MAX_429_RETRIES"]),
            "max_request_error_retries": int(values["MAX_REQUEST_ERROR_RETRIES"]),
        },
        "runtime": {
            "global_seed": int(values["GLOBAL_SEED"]),
            "candidate_timeout_s": float(values["CANDIDATE_TIMEOUT_S"]),
            "evaluation_timeout_s": float(values["EVALUATION_TIMEOUT_S"]),
            "eval_split": values["EVAL_SPLIT"],
            "smoke_test": smoke_test,
            "dry_run": dry_run,
        },
        "feedback": {
            "include_invalid_code_in_feedback": _as_bool(values["INCLUDE_INVALID_CODE_IN_FEEDBACK"]),
            "include_invalid_error_trace": _as_bool(values["INCLUDE_INVALID_ERROR_TRACE"]),
            "include_parent_code_in_mutation_prompt": _as_bool(values["INCLUDE_PARENT_CODE_IN_MUTATION_PROMPT"]),
            "save_raw_llm_responses": _as_bool(values["SAVE_RAW_LLM_RESPONSES"]),
            "save_generated_attempts": _as_bool(values["SAVE_GENERATED_ATTEMPTS"]),
        },
        "suite": {
            "instance_root": values["INSTANCE_ROOT"],
            "candidate_cache_dir": values["CANDIDATE_CACHE_DIR"],
            "artifact_root": values["ARTIFACT_ROOT"],
            "splits": {
                "train": list(values["TRAIN_INSTANCES"]),
                "val": list(values["VAL_INSTANCES"]),
                "test": list(values["TEST_INSTANCES"]),
            },
            "optima": dict(values["OPTIMA"]),
        },
        "popmusic": {
            "use_popmusic_candidates": _as_bool(values["USE_POPMUSIC_CANDIDATES"]),
            "use_popmusic_edge_prior": _as_bool(values["USE_POPMUSIC_EDGE_PRIOR"]),
            "prior_mode": values["POPMUSIC_PRIOR_MODE"],
            "max_candidates": int(values["MAX_CANDIDATES"]),
            "restrict_edge_cost_to_candidates": _as_bool(values["RESTRICT_EDGE_COST_TO_CANDIDATES"]),
            "allow_non_candidate_edges_in_final_tour": _as_bool(values["ALLOW_NON_CANDIDATE_EDGES_IN_FINAL_TOUR"]),
            "lkh_binary_path": values["LKH_BINARY_PATH"],
        },
    }
    out = Path(tempfile.gettempdir()) / f"{values['RUN_NAME']}_runtime_config.yaml"
    with out.open("w", encoding="utf-8") as f:
        yaml.safe_dump(effective, f, sort_keys=False)
    return str(out), effective


def print_effective_config(effective: dict[str, Any]) -> None:
    print("Effective TSP runtime config")
    print("-" * 80)
    print(f"run_name: {effective.get('run_name')}")
    print(f"experiment_mode: {effective.get('experiment_mode')}")
    llm = effective.get("llm", {})
    runtime = effective.get("runtime", {})
    suite = effective.get("suite", {})
    pop = effective.get("popmusic", {})
    feedback = effective.get("feedback", {})
    print(f"llm: {llm.get('provider')} / {llm.get('model')}")
    print(f"max_llm_calls: {llm.get('max_llm_calls')}")
    print(f"temperature: {llm.get('temperature')}  top_p: {llm.get('top_p')}")
    print(f"smoke_test: {runtime.get('smoke_test')}  dry_run: {runtime.get('dry_run')}")
    print(f"eval_split: {runtime.get('eval_split')}")
    print(f"global_seed: {runtime.get('global_seed')}")
    print(f"candidate_timeout_s: {runtime.get('candidate_timeout_s')}")
    print(f"evaluation_timeout_s: {runtime.get('evaluation_timeout_s')}")
    print(f"instance_root: {suite.get('instance_root')}")
    print(f"candidate_cache_dir: {suite.get('candidate_cache_dir')}")
    print(f"artifact_root: {suite.get('artifact_root')}")
    print(f"use_popmusic_candidates: {pop.get('use_popmusic_candidates')}")
    print(f"use_popmusic_edge_prior: {pop.get('use_popmusic_edge_prior')}")
    print(f"popmusic_prior_mode: {pop.get('prior_mode')}")
    print(f"max_candidates: {pop.get('max_candidates')}")
    print(f"restrict_edge_cost_to_candidates: {pop.get('restrict_edge_cost_to_candidates')}")
    print(f"allow_non_candidate_edges_in_final_tour: {pop.get('allow_non_candidate_edges_in_final_tour')}")
    print(f"include_invalid_code_in_feedback: {feedback.get('include_invalid_code_in_feedback')}")
    print(f"include_invalid_error_trace: {feedback.get('include_invalid_error_trace')}")
    print("-" * 80)


def selected_instance_names(effective: dict[str, Any]) -> list[str]:
    split = effective.get("runtime", {}).get("eval_split", "train")
    splits = effective.get("suite", {}).get("splits", {})
    if split == "all":
        names: list[str] = []
        for part in ("train", "val", "test"):
            names.extend(splits.get(part, []))
        return names
    return list(splits.get(split, []))


def candidate_file_candidates(instance_name: str, candidate_cache_dir: str | Path) -> list[Path]:
    root = Path(candidate_cache_dir)
    return [
        root / f"{instance_name}.cand",
        root / f"{instance_name}.candidates",
        root / f"{instance_name}_candidates.txt",
        root / f"{instance_name}.txt",
    ]


def tsplib_file_candidates(instance_name: str, instance_root: str | Path) -> list[Path]:
    root = Path(instance_root)
    return [
        root / f"{instance_name}.tsp",
        root / f"{instance_name}.TSP",
        root / instance_name / f"{instance_name}.tsp",
        root / instance_name / f"{instance_name}.TSP",
    ]
