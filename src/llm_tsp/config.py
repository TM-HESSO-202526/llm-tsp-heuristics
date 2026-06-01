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
    """Flattened runtime view used by scripts and notebooks.

    """

    run_name: str = "tsp_llamea_run"
    global_seed: int = 12345
    max_llm_calls: int = 40
    eval_split: str = "train"
    candidate_timeout_s: float = 60.0
    smoke_test: bool = False

    # LLM/provider controls
    llm_provider: str = "groq"
    llm_model: str = "llama-3.3-70b-versatile"
    temperature: float = 0.8
    top_p: float = 1.0

    # LLaMEA search controls, kept aligned with the clustering repo launcher.
    selection_strategy: str = "1+1"  # "1+1" elitist best-so-far; "1,1" latest sequential parent
    history_limit: int = 20
    invalid_parent_redesign: bool = True
    redesign_on_any_invalid_before_full_valid: bool = True
    redesign_on_timeout_parent: bool = True
    hide_invalid_parent_code: bool = False
    historical_family_avoidance: bool = False
    strict_constructive_only: bool = False
    family_focus_mode: bool = False
    family_focus_calls_per_family: int = 20

    # POPMUSIC/candidate-prior layer.
    use_popmusic_candidates: bool = False
    use_popmusic_edge_prior: bool = False
    popmusic_prior_mode: str = "none"
    max_candidates: int = 20
    # Candidate mode policy is fixed:
    # POPMUSIC/LKH candidates are exposed as guidance through problem.neighbors(i).
    # Final tours are normal TSP permutations and are always evaluated on the true full TSPLIB distance.
    lkh_binary_path: str = "/content/tools/lkh/LKH"
    edge_prior_cache_dir: str = "/content/drive/MyDrive/TM/LKH_edge_prior_cache"
    edge_prior_runs: int = 30
    edge_prior_time_limit_s: float = 1.0
    edge_prior_topk: int = 5
    edge_prior_force_rebuild: bool = False

    # Paths.
    instance_root: str = "/content/drive/MyDrive/TM/TSP_instances"
    candidate_cache_dir: str = "/content/drive/MyDrive/TM/LKH_candidate_cache"
    artifact_root: str = "/content/drive/MyDrive/TM/llm-tsp-runs"
    extra: dict[str, Any] = field(default_factory=dict)


def flatten_runtime_config(cfg: dict[str, Any]) -> RuntimeConfig:
    llm = cfg.get("llm", {})
    runtime = cfg.get("runtime", {})
    pop = cfg.get("popmusic", {})
    suite = cfg.get("suite", {})
    search = cfg.get("search", {})

    return RuntimeConfig(
        run_name=cfg.get("run_name", "tsp_llamea_run"),
        global_seed=int(runtime.get("global_seed", 12345)),
        max_llm_calls=int(llm.get("max_llm_calls", 40)),
        eval_split=runtime.get("eval_split", "train"),
        candidate_timeout_s=float(runtime.get("candidate_timeout_s", 60)),
        smoke_test=bool(runtime.get("smoke_test", False)),
        llm_provider=str(llm.get("provider", "groq")),
        llm_model=str(llm.get("model", "llama-3.3-70b-versatile")),
        temperature=float(llm.get("temperature", 0.8)),
        top_p=float(llm.get("top_p", 1.0)),
        selection_strategy=str(search.get("selection_strategy", "1+1")),
        history_limit=int(search.get("history_limit", 20)),
        invalid_parent_redesign=bool(search.get("invalid_parent_redesign", True)),
        redesign_on_any_invalid_before_full_valid=bool(search.get("redesign_on_any_invalid_before_full_valid", True)),
        redesign_on_timeout_parent=bool(search.get("redesign_on_timeout_parent", True)),
        hide_invalid_parent_code=bool(search.get("hide_invalid_parent_code", False)),
        historical_family_avoidance=bool(search.get("historical_family_avoidance", False)),
        strict_constructive_only=bool(search.get("strict_constructive_only", False)),
        family_focus_mode=bool(search.get("family_focus_mode", False)),
        family_focus_calls_per_family=int(search.get("family_focus_calls_per_family", 20)),
        use_popmusic_candidates=bool(pop.get("use_popmusic_candidates", False)),
        use_popmusic_edge_prior=bool(pop.get("use_popmusic_edge_prior", False)),
        popmusic_prior_mode=str(pop.get("prior_mode", "none")),
        max_candidates=int(pop.get("max_candidates", 20)),
        lkh_binary_path=str(pop.get("lkh_binary_path", cfg.get("lkh", {}).get("lkh_binary", "/content/tools/lkh/LKH"))),
        edge_prior_cache_dir=str(suite.get("edge_prior_cache_dir", pop.get("edge_prior_cache_dir", "/content/drive/MyDrive/TM/LKH_edge_prior_cache"))),
        edge_prior_runs=int(pop.get("edge_prior_runs", cfg.get("edge_prior", {}).get("runs", 30))),
        edge_prior_time_limit_s=float(pop.get("edge_prior_time_limit_s", cfg.get("edge_prior", {}).get("time_limit_s", 1.0))),
        edge_prior_topk=int(pop.get("edge_prior_topk", cfg.get("edge_prior", {}).get("topk", 5))),
        edge_prior_force_rebuild=bool(pop.get("edge_prior_force_rebuild", cfg.get("edge_prior", {}).get("force_rebuild", False))),
        instance_root=str(suite.get("instance_root", "/content/drive/MyDrive/TM/TSP_instances")),
        candidate_cache_dir=str(suite.get("candidate_cache_dir", "/content/drive/MyDrive/TM/LKH_candidate_cache")),
        artifact_root=str(suite.get("artifact_root", "/content/drive/MyDrive/TM/llm-tsp-runs")),
        extra=cfg,
    )
