from __future__ import annotations

from pathlib import Path
from typing import Any
import tempfile
import yaml


DEFAULTS: dict[str, Any] = {
    # Run identity. The TSP repo is always LLaMEA-mode; no separate mode switch.
    "RUN_NAME": "tsp_llamea_popmusic_train",
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

    # LLaMEA evolution/search behavior, aligned with clustering launcher.
    "SELECTION_STRATEGY": "1+1",  # "1+1" = elitist best-so-far parent; "1,1" = latest sequential parent
    "HISTORY_LIMIT": 20,

    # Invalid-parent redesign behavior, aligned with clustering launcher.
    "INVALID_PARENT_REDESIGN": True,
    "REDESIGN_ON_ANY_INVALID_BEFORE_FULL_VALID": True,
    "REDESIGN_ON_TIMEOUT_PARENT": True,
    "HIDE_INVALID_PARENT_CODE": False,

    # Family memory / novelty controls, same terminology as clustering.
    # They are implemented but kept off by default, so nothing is injected
    # into the prompt unless these are explicitly enabled.
    "HISTORICAL_FAMILY_AVOIDANCE": False,
    "FAMILY_NOVELTY_MODE": False,
    "FAMILY_MEMORY_LIMIT": 8,
    "MIN_FAMILY_ATTEMPTS_BEFORE_AVOID": 5,
    "WEAK_FAMILY_SCORE_THRESHOLD": 20.0,
    "ALLOW_STRONG_FAMILY_EXPLOITATION": True,

    # Runtime and evaluation
    "GLOBAL_SEED": 12345,
    "CANDIDATE_TIMEOUT_S": 60,
    "EVALUATION_TIMEOUT_S": 120,
    "EVAL_SPLIT": "train",

    # TSP data / artifact paths
    "INSTANCE_ROOT": "/content/drive/MyDrive/TM/TSP_instances",
    "CANDIDATE_CACHE_DIR": "/content/drive/MyDrive/TM/LKH_candidate_cache",
    "EDGE_PRIOR_CACHE_DIR": "/content/drive/MyDrive/TM/LKH_edge_prior_cache",
    "ARTIFACT_ROOT": "/content/drive/MyDrive/TM/llm-tsp-runs",

    # POPMUSIC/candidate-prior controls
    "USE_POPMUSIC_CANDIDATES": True,
    "USE_POPMUSIC_EDGE_PRIOR": True,
    "POPMUSIC_PRIOR_MODE": "frequency",
    "MAX_CANDIDATES": 20,
    "LKH_BINARY_PATH": "/content/tools/lkh/LKH",
    "EDGE_PRIOR_RUNS": 30,
    "EDGE_PRIOR_TIME_LIMIT_S": 1.0,
    "EDGE_PRIOR_TOPK": 5,
    "EDGE_PRIOR_FORCE_REBUILD": False,

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
    config consumed by the backend scripts, mirroring the clustering repo
    pattern. TSP has no experiment-mode switch: every run uses the LLaMEA loop.
    """
    values = {k: globals_dict.get(k, v) for k, v in DEFAULTS.items()}
    smoke_test = _as_bool(values["SMOKE_TEST"])
    dry_run = _as_bool(values["DRY_RUN"])
    max_calls = 1 if smoke_test else int(values["MAX_LLM_CALLS"])

    strategy = str(values["SELECTION_STRATEGY"])
    if strategy not in {"1+1", "1,1"}:
        raise ValueError(f"SELECTION_STRATEGY must be '1+1' or '1,1', got {strategy!r}")

    effective = {
        "run_name": values["RUN_NAME"],
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
        "search": {
            "selection_strategy": strategy,
            "history_limit": int(values["HISTORY_LIMIT"]),
            "invalid_parent_redesign": _as_bool(values["INVALID_PARENT_REDESIGN"]),
            "redesign_on_any_invalid_before_full_valid": _as_bool(values["REDESIGN_ON_ANY_INVALID_BEFORE_FULL_VALID"]),
            "redesign_on_timeout_parent": _as_bool(values["REDESIGN_ON_TIMEOUT_PARENT"]),
            "hide_invalid_parent_code": _as_bool(values["HIDE_INVALID_PARENT_CODE"]),
            "historical_family_avoidance": _as_bool(values["HISTORICAL_FAMILY_AVOIDANCE"]),
            "family_novelty_mode": _as_bool(values["FAMILY_NOVELTY_MODE"]),
            "family_memory_limit": int(values["FAMILY_MEMORY_LIMIT"]),
            "min_family_attempts_before_avoid": int(values["MIN_FAMILY_ATTEMPTS_BEFORE_AVOID"]),
            "weak_family_score_threshold": float(values["WEAK_FAMILY_SCORE_THRESHOLD"]),
            "allow_strong_family_exploitation": _as_bool(values["ALLOW_STRONG_FAMILY_EXPLOITATION"]),
        },
        "runtime": {
            "global_seed": int(values["GLOBAL_SEED"]),
            "candidate_timeout_s": float(values["CANDIDATE_TIMEOUT_S"]),
            "evaluation_timeout_s": float(values["EVALUATION_TIMEOUT_S"]),
            "eval_split": values["EVAL_SPLIT"],
            "smoke_test": smoke_test,
            "dry_run": dry_run,
        },
        "suite": {
            "instance_root": values["INSTANCE_ROOT"],
            "candidate_cache_dir": values["CANDIDATE_CACHE_DIR"],
            "edge_prior_cache_dir": values["EDGE_PRIOR_CACHE_DIR"],
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
            # Fixed policy, matching the historical TSP notebooks:
            # candidate lists guide the heuristic; final tours are evaluated on full TSPLIB distance.
            "lkh_binary_path": values["LKH_BINARY_PATH"],
            "edge_prior_cache_dir": values["EDGE_PRIOR_CACHE_DIR"],
            "edge_prior_runs": int(values["EDGE_PRIOR_RUNS"]),
            "edge_prior_time_limit_s": float(values["EDGE_PRIOR_TIME_LIMIT_S"]),
            "edge_prior_topk": int(values["EDGE_PRIOR_TOPK"]),
            "edge_prior_force_rebuild": _as_bool(values["EDGE_PRIOR_FORCE_REBUILD"]),
        },
        "edge_prior": {
            "runs": int(values["EDGE_PRIOR_RUNS"]),
            "time_limit_s": float(values["EDGE_PRIOR_TIME_LIMIT_S"]),
            "topk": int(values["EDGE_PRIOR_TOPK"]),
            "move_type": 5,
            "patching_a": 2,
            "patching_c": 3,
            "force_rebuild": _as_bool(values["EDGE_PRIOR_FORCE_REBUILD"]),
        },
    }
    out = Path(tempfile.gettempdir()) / f"{values['RUN_NAME']}_runtime_config.yaml"
    with out.open("w", encoding="utf-8") as f:
        yaml.safe_dump(effective, f, sort_keys=False)
    return str(out), effective


def print_effective_config(effective: dict[str, Any]) -> None:
    llm = effective.get("llm", {})
    runtime = effective.get("runtime", {})
    search = effective.get("search", {})
    pop = effective.get("popmusic", {})
    suite = effective.get("suite", {})

    print("-" * 80)
    print("Effective TSP runtime config")
    print("-" * 80)
    print(f"run_name: {effective.get('run_name')}")
    print("mode: LLaMEA only")
    print(f"llm: {llm.get('provider')} / {llm.get('model')}")
    print(f"max_llm_calls: {llm.get('max_llm_calls')}")
    print(f"selection_strategy: {search.get('selection_strategy')}")
    print(f"history_limit: {search.get('history_limit')}")
    print(
        "Invalid-parent redesign: "
        f"{search.get('invalid_parent_redesign')} | any-invalid: {search.get('redesign_on_any_invalid_before_full_valid')} | "
        f"timeout: {search.get('redesign_on_timeout_parent')} | expose-invalid-code: {not search.get('hide_invalid_parent_code', False)}"
    )
    print(
        "Family novelty mode: "
        f"{search.get('family_novelty_mode')} | memory limit: {search.get('family_memory_limit')} | "
        f"min attempts before avoid: {search.get('min_family_attempts_before_avoid')} | "
        f"weak threshold: {search.get('weak_family_score_threshold')} | "
        f"allow strong exploitation: {search.get('allow_strong_family_exploitation')}"
    )
    print(f"smoke_test: {runtime.get('smoke_test')}")
    print(f"dry_run: {runtime.get('dry_run')}")
    print(f"eval_split: {runtime.get('eval_split')}")
    print(f"global_seed: {runtime.get('global_seed')}")
    print(f"candidate_timeout_s: {runtime.get('candidate_timeout_s')}")
    print(f"evaluation_timeout_s: {runtime.get('evaluation_timeout_s')}")
    print(f"use_popmusic_candidates: {pop.get('use_popmusic_candidates')}")
    print(f"use_popmusic_edge_prior: {pop.get('use_popmusic_edge_prior')}")
    print(f"popmusic_prior_mode: {pop.get('prior_mode')}")
    print(f"max_candidates: {pop.get('max_candidates')}")
    print(f"edge_prior_cache_dir: {suite.get('edge_prior_cache_dir')}")
    print(f"edge_prior_runs: {pop.get('edge_prior_runs')} | time_limit_s: {pop.get('edge_prior_time_limit_s')} | topk: {pop.get('edge_prior_topk')}")
    print("candidate_edge_policy: guidance_only_full_tour_allowed")
    print("problem interface: sparse candidates + edge-cost oracle; no public dense distance matrix")
    print(f"instance_root: {suite.get('instance_root')}")
    print(f"candidate_cache_dir: {suite.get('candidate_cache_dir')}")
    print(f"artifact_root: {suite.get('artifact_root')}")
    print("splits:")
    for split, names in suite.get("splits", {}).items():
        print(f"  {split}: {names}")
    print("-" * 80)


def selected_instance_names(effective: dict[str, Any]) -> list[str]:
    """Return the TSP instance names selected by ``runtime.eval_split``.

    This helper is used by the Colab launcher before the backend pipeline is
    executed. It mirrors the split-selection logic used by the backend scripts
    but keeps the notebook cell short and robust.
    
    Supported values are:
    - ``train``: training split only
    - ``val``: validation split only
    - ``test``: final test split only
    - ``all``: train + val + test, preserving order and removing duplicates
    """
    runtime = effective.get("runtime", {})
    suite = effective.get("suite", {})
    split = str(runtime.get("eval_split", "train")).strip().lower()
    splits = suite.get("splits", {}) or {}

    if split == "all":
        names: list[str] = []
        seen: set[str] = set()
        for key in ("train", "val", "test"):
            for name in splits.get(key, []) or []:
                if name not in seen:
                    names.append(str(name))
                    seen.add(str(name))
        # Include any extra split keys at the end, in case a user extends the config.
        for key, values in splits.items():
            if key in {"train", "val", "test"}:
                continue
            for name in values or []:
                if name not in seen:
                    names.append(str(name))
                    seen.add(str(name))
        return names

    if split not in splits:
        raise ValueError(
            f"Unknown EVAL_SPLIT {split!r}. Expected one of {sorted(splits.keys()) + ['all']}."
        )
    return [str(name) for name in (splits.get(split, []) or [])]


def tsplib_file_candidates(instance_name: str, instance_root: str | Path) -> list[Path]:
    """Possible TSPLIB file locations for one instance.

    The project stores raw TSPLIB files outside git, usually in Drive. This
    accepts both flat folders and one-subfolder-per-instance layouts.
    """
    root = Path(instance_root)
    name = str(instance_name)
    variants = [name, name.upper(), name.lower()]
    paths: list[Path] = []
    for stem in variants:
        paths.extend([
            root / f"{stem}.tsp",
            root / f"{stem}.TSP",
            root / stem / f"{stem}.tsp",
            root / stem / f"{stem}.TSP",
        ])
    # Deduplicate while preserving order.
    out: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key not in seen:
            out.append(path)
            seen.add(key)
    return out


def candidate_file_candidates(instance_name: str, candidate_root: str | Path) -> list[Path]:
    """Possible POPMUSIC/LKH candidate-cache file locations for one instance.

    Historical notebooks used slightly different suffixes while experimenting
    with LKH/POPMUSIC. The launcher checks a permissive set of names so the
    user does not have to rename an existing cache unnecessarily.
    """
    root = Path(candidate_root)
    name = str(instance_name)
    variants = [name, name.upper(), name.lower()]
    suffixes = [
        "_cand-popmusic-k20-s14-sol20-nn5-tr1.cand",
        ".cand",
        ".candidates",
        "_candidates.txt",
        "_candidates.csv",
        "_popmusic_candidates.txt",
        "_popmusic.cand",
        ".popmusic",
        ".txt",
    ]
    paths: list[Path] = []
    for stem in variants:
        for suffix in suffixes:
            paths.append(root / f"{stem}{suffix}")
        # Also accept candidate files inside a per-instance directory.
        for suffix in suffixes:
            paths.append(root / stem / f"{stem}{suffix}")
    out: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key not in seen:
            out.append(path)
            seen.add(key)
    return out


def edge_prior_file_candidates(instance_name: str, edge_prior_root: str | Path) -> list[Path]:
    """Historical LKH/POPMUSIC edge-prior cache locations for one instance."""
    root = Path(edge_prior_root)
    name = str(instance_name)
    variants = [name, name.upper(), name.lower()]
    suffixes = [
        "_popmusic_edge_prior_runs30_topk5.npz",
        "_edge_prior_runs30_topk5.npz",
        "_edge_prior.npz",
    ]
    paths: list[Path] = []
    for stem in variants:
        for suffix in suffixes:
            paths.append(root / f"{stem}{suffix}")
        for suffix in suffixes:
            paths.append(root / stem / f"{stem}{suffix}")
    out: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key not in seen:
            out.append(path)
            seen.add(key)
    return out
