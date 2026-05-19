from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_run_config(path: str | Path) -> dict[str, Any]:
    cfg = load_yaml(path)
    suite_path = cfg.get("suite_config")
    if suite_path:
        suite_cfg = load_yaml(suite_path)
        cfg["suite"] = suite_cfg
    return cfg


@dataclass
class RuntimeConfig:
    run_name: str = "tsp_run"
    experiment_mode: str = "llamea"
    global_seed: int = 12345
    max_llm_calls: int = 40
    eval_split: str = "train"
    candidate_timeout_s: float = 60.0
    evaluation_timeout_s: float = 120.0
    smoke_test: bool = False
    dry_run: bool = False
    llm_provider: str = "groq"
    llm_model: str = "llama-3.3-70b-versatile"
    temperature: float = 0.8
    top_p: float = 1.0
    use_popmusic_candidates: bool = False
    use_popmusic_edge_prior: bool = False
    popmusic_prior_mode: str = "none"
    max_candidates: int = 20
    restrict_edge_cost_to_candidates: bool = False
    allow_non_candidate_edges_in_final_tour: bool = True
    include_invalid_code_in_feedback: bool = True
    include_invalid_error_trace: bool = True
    include_parent_code_in_mutation_prompt: bool = True
    instance_root: str = "/content/drive/MyDrive/TM/TSP_instances"
    candidate_cache_dir: str = "/content/drive/MyDrive/TM/LKH_candidate_cache"
    artifact_root: str = "/content/drive/MyDrive/TM/llm-tsp-runs"
    extra: dict[str, Any] = field(default_factory=dict)


def flatten_runtime_config(cfg: dict[str, Any]) -> RuntimeConfig:
    llm = cfg.get("llm", {})
    runtime = cfg.get("runtime", {})
    feedback = cfg.get("feedback", {})
    pop = cfg.get("popmusic", {})
    suite = cfg.get("suite", {})
    return RuntimeConfig(
        run_name=cfg.get("run_name", "tsp_run"),
        experiment_mode=cfg.get("experiment_mode", "llamea"),
        global_seed=int(runtime.get("global_seed", 12345)),
        max_llm_calls=int(llm.get("max_llm_calls", 40)),
        eval_split=runtime.get("eval_split", "train"),
        candidate_timeout_s=float(runtime.get("candidate_timeout_s", 60)),
        evaluation_timeout_s=float(runtime.get("evaluation_timeout_s", 120)),
        smoke_test=bool(runtime.get("smoke_test", False)),
        dry_run=bool(runtime.get("dry_run", False)),
        llm_provider=str(llm.get("provider", "groq")),
        llm_model=str(llm.get("model", "llama-3.3-70b-versatile")),
        temperature=float(llm.get("temperature", 0.8)),
        top_p=float(llm.get("top_p", 1.0)),
        use_popmusic_candidates=bool(pop.get("use_popmusic_candidates", False)),
        use_popmusic_edge_prior=bool(pop.get("use_popmusic_edge_prior", False)),
        popmusic_prior_mode=str(pop.get("prior_mode", "none")),
        max_candidates=int(pop.get("max_candidates", 20)),
        restrict_edge_cost_to_candidates=bool(pop.get("restrict_edge_cost_to_candidates", False)),
        allow_non_candidate_edges_in_final_tour=bool(pop.get("allow_non_candidate_edges_in_final_tour", True)),
        include_invalid_code_in_feedback=bool(feedback.get("include_invalid_code_in_feedback", True)),
        include_invalid_error_trace=bool(feedback.get("include_invalid_error_trace", True)),
        include_parent_code_in_mutation_prompt=bool(feedback.get("include_parent_code_in_mutation_prompt", True)),
        instance_root=str(suite.get("instance_root", "/content/drive/MyDrive/TM/TSP_instances")),
        candidate_cache_dir=str(suite.get("candidate_cache_dir", "/content/drive/MyDrive/TM/LKH_candidate_cache")),
        artifact_root=str(suite.get("artifact_root", "/content/drive/MyDrive/TM/llm-tsp-runs")),
        extra=cfg,
    )
