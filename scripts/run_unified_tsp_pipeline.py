#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm_tsp.config import load_run_config, flatten_runtime_config
from llm_tsp.artifacts import make_run_dir, save_json
from llm_tsp.distance import euclidean_matrix, tsplib_distance_matrix, tour_cost_from_matrix
from llm_tsp.sparse_problem import SparseTSPProblem
from llm_tsp.candidate_sets import (
    k_nearest_candidates,
    parse_simple_candidate_file,
    prior_from_candidate_frequency,
)
from llm_tsp.priors import transform_prior
from llm_tsp.lkh_popmusic import (
    PopmusicParams, EdgePriorParams,
    popmusic_candidate_file_name, popmusic_edge_prior_file_name,
    run_popmusic_candidate_generation, run_popmusic_edge_prior_generation,
    load_prior_npz,
)
from llm_tsp.baselines import nearest_neighbor, prior_greedy
from llm_tsp.suite import specs_from_suite_config, filter_specs, InstanceSpec
from llm_tsp.tsplib_io import read_tsplib_coords
from llm_tsp.llm_client import call_groq_chat, get_secret, loaded_groq_key_names as _loaded_groq_key_names
from llm_tsp.llamea_loop import run_llamea_search
from llm_tsp.prompts import objective_prompt_block


def make_toy_problem(n: int = 100, seed: int = 0, use_candidates: bool = False, max_k: int = 20) -> SparseTSPProblem:
    rng = np.random.default_rng(seed)
    coords = rng.random((n, 2)) * 1000.0
    dist = euclidean_matrix(coords)
    cands = k_nearest_candidates(dist, max_k=max_k) if use_candidates else None
    prior = prior_from_candidate_frequency([cands], n=n) if cands else None
    return SparseTSPProblem(
        coords=coords,
        dist=dist,
        candidate_neighbors=cands,
        prior_map=prior,
    )


def run_dry_smoke(cfg: dict, artifact_dir: Path) -> None:
    pop = cfg.get("popmusic", {})
    runtime = cfg.get("runtime", {})
    problem = make_toy_problem(
        n=100,
        seed=int(runtime.get("global_seed", 0)),
        use_candidates=bool(pop.get("use_popmusic_candidates", False)),
        max_k=int(pop.get("max_candidates", 20)),
    )
    tour = nearest_neighbor(problem)
    nn_cost = tour_cost_from_matrix(tour, problem.distance_matrix_for_evaluator())
    rows = [{"method": "nearest_neighbor", "n": problem.n, "cost": float(nn_cost)}]
    print("Dry-run smoke test")
    print(f"n={problem.n} nearest_neighbor_cost={nn_cost:.3f}")
    if pop.get("use_popmusic_edge_prior"):
        tour2 = prior_greedy(problem)
        prior_cost = tour_cost_from_matrix(tour2, problem.distance_matrix_for_evaluator())
        rows.append({"method": "prior_greedy_placeholder", "n": problem.n, "cost": float(prior_cost)})
        print(f"prior_greedy_cost={prior_cost:.3f}")
    pd.DataFrame(rows).to_csv(artifact_dir / "dry_run_smoke_results.csv", index=False)


def candidate_paths_for_instance(instance_name: str, root: Path, params: PopmusicParams | None = None) -> list[Path]:
    params = params or PopmusicParams()
    return [
        popmusic_candidate_file_name(instance_name, root, params),
        root / f"{instance_name}.cand",
        root / f"{instance_name}.candidates",
        root / f"{instance_name}_candidates.txt",
        root / f"{instance_name}.txt",
    ]



def edge_prior_paths_for_instance(instance_name: str, root: Path) -> list[Path]:
    return [
        root / f"{instance_name}_popmusic_edge_prior_runs30_topk5.npz",
        root / f"{instance_name}_edge_prior_runs30_topk5.npz",
        root / f"{instance_name}_edge_prior.npz",
    ]


def tsplib_paths_for_instance(instance_name: str, root: Path) -> list[Path]:
    return [
        root / f"{instance_name}.tsp",
        root / f"{instance_name}.TSP",
        root / instance_name / f"{instance_name}.tsp",
        root / instance_name / f"{instance_name}.TSP",
    ]


def select_existing_path(paths: list[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


def select_existing(paths: list[Path]) -> str:
    p = select_existing_path(paths)
    return "" if p is None else str(p)


def selected_specs(cfg: dict, eval_split: str) -> list[InstanceSpec]:
    suite = cfg.get("suite", {})
    specs = specs_from_suite_config(suite)
    return specs if eval_split == "all" else filter_specs(specs, split=eval_split)


def write_selected_instances(cfg: dict, artifact_dir: Path, eval_split: str) -> pd.DataFrame:
    suite = cfg.get("suite", {})
    instance_root = Path(suite.get("instance_root", ""))
    candidate_root = Path(suite.get("candidate_cache_dir", ""))
    edge_prior_root = Path(suite.get("edge_prior_cache_dir", "/content/drive/MyDrive/TM/LKH_edge_prior_cache"))
    selected = selected_specs(cfg, eval_split)
    rows = []
    for spec in selected:
        tsplib_candidates = tsplib_paths_for_instance(spec.name, instance_root)
        candidate_candidates = candidate_paths_for_instance(spec.name, candidate_root)
        edge_prior_candidates = edge_prior_paths_for_instance(spec.name, edge_prior_root)
        rows.append({
            "instance": spec.name,
            "split": spec.split,
            "optimum": spec.optimum,
            "tsplib_file_found": select_existing(tsplib_candidates),
            "tsplib_first_expected": str(tsplib_candidates[0]),
            "candidate_file_found": select_existing(candidate_candidates),
            "candidate_first_expected": str(candidate_candidates[0]),
            "edge_prior_file_found": select_existing(edge_prior_candidates),
            "edge_prior_first_expected": str(edge_prior_candidates[0]),
        })
    df = pd.DataFrame(rows)
    df.to_csv(artifact_dir / "selected_instances.csv", index=False)
    return df


def load_problem_for_spec(cfg: dict, spec: InstanceSpec, artifact_dir: Path) -> tuple[str, SparseTSPProblem, float | None]:
    suite = cfg.get("suite", {})
    pop = cfg.get("popmusic", {})
    runtime = cfg.get("runtime", {})

    instance_root = Path(suite.get("instance_root", ""))
    candidate_root = Path(suite.get("candidate_cache_dir", ""))

    tsp_path = select_existing_path(tsplib_paths_for_instance(spec.name, instance_root))
    if tsp_path is None:
        raise FileNotFoundError(f"Missing TSPLIB file for {spec.name} under {instance_root}")

    coords, meta = read_tsplib_coords(tsp_path)
    dist = tsplib_distance_matrix(coords, meta.get("EDGE_WEIGHT_TYPE"))
    n = int(dist.shape[0])
    max_k = int(pop.get("max_candidates", 20))
    requested_use_candidates = bool(pop.get("use_popmusic_candidates", False))
    use_prior = bool(pop.get("use_popmusic_edge_prior", False))
    # Historical POPMUSIC-prior experiments used both structures: the LLM sees
    # POPMUSIC candidate lists as sparse guidance, and edge priors are an extra
    # score signal built from short LKH tours. If the prior is enabled, candidate
    # lists are therefore exposed even if the launcher flag was accidentally left
    # off.
    use_candidates = bool(requested_use_candidates or use_prior)

    candidate_map = None
    raw_prior = {}
    candidate_source = "none"
    edge_prior_source = "none"
    edge_prior_meta = {}

    lkh_cfg = cfg.get("lkh", {})
    edge_prior_cfg = cfg.get("edge_prior", {})
    params = PopmusicParams(
        max_candidates=int(pop.get("max_candidates", lkh_cfg.get("max_candidates", max_k))),
        popmusic_sample_size=int(lkh_cfg.get("popmusic_sample_size", 14)),
        popmusic_solutions=int(lkh_cfg.get("popmusic_solutions", 20)),
        popmusic_max_neighbors=int(lkh_cfg.get("popmusic_max_neighbors", 5)),
        popmusic_trials=int(lkh_cfg.get("popmusic_trials", 1)),
        popmusic_initial_tour=bool(lkh_cfg.get("popmusic_initial_tour", False)),
    )
    prior_params = EdgePriorParams(
        runs=int(pop.get("edge_prior_runs", edge_prior_cfg.get("runs", 30))),
        time_limit_s=float(pop.get("edge_prior_time_limit_s", edge_prior_cfg.get("time_limit_s", 1.0))),
        topk=int(pop.get("edge_prior_topk", edge_prior_cfg.get("topk", 5))),
        move_type=int(edge_prior_cfg.get("move_type", 5)),
        patching_a=int(edge_prior_cfg.get("patching_a", 2)),
        patching_c=int(edge_prior_cfg.get("patching_c", 3)),
        force_rebuild=bool(pop.get("edge_prior_force_rebuild", edge_prior_cfg.get("force_rebuild", False))),
    )
    lkh_binary = pop.get("lkh_binary_path") or lkh_cfg.get("lkh_binary", "/content/tools/lkh/LKH")

    if use_candidates or use_prior:
        cand_path = select_existing_path(candidate_paths_for_instance(spec.name, candidate_root, params))
        if cand_path is None:
            # Historical behavior: if the POPMUSIC cache is absent, build the official
            # LKH CANDIDATE_FILE immediately and store it under the historical cache name.
            cand_path = popmusic_candidate_file_name(spec.name, candidate_root, params)
            print(f"[candidate cache] miss for {spec.name}; generating official LKH/POPMUSIC candidate file.")
            print(f"[candidate cache] tsp_file={tsp_path}")
            print(f"[candidate cache] candidate_file={cand_path}")
            cand_path = run_popmusic_candidate_generation(
                tsp_path,
                cand_path,
                lkh_binary,
                params=params,
                timeout_s=float(lkh_cfg.get("generation_timeout_s", 900)),
            )
            print(f"[candidate cache] wrote {cand_path}")
        else:
            print(f"[candidate cache] hit for {spec.name}: {cand_path}")

        try:
            parsed = parse_simple_candidate_file(cand_path, n=n)
            from llm_tsp.candidate_sets import normalize_candidates
            candidate_map = normalize_candidates(parsed, n=n, max_k=max_k, dist=dist)
            candidate_source = str(cand_path)
        except Exception as e:
            raise RuntimeError(f"Could not parse POPMUSIC candidate file for {spec.name}: {cand_path}") from e

    if use_prior:
        edge_prior_root = Path(suite.get("edge_prior_cache_dir", pop.get("edge_prior_cache_dir", "/content/drive/MyDrive/TM/LKH_edge_prior_cache")))
        prior_path = popmusic_edge_prior_file_name(spec.name, edge_prior_root, prior_params)
        if prior_params.force_rebuild or (not prior_path.exists()):
            print(f"[edge prior cache] miss for {spec.name}; generating LKH/POPMUSIC tour-frequency prior.")
            print(
                f"[edge prior cache] runs={prior_params.runs} time_limit={prior_params.time_limit_s}s "
                f"topk={prior_params.topk} output={prior_path}"
            )
            prior_path = run_popmusic_edge_prior_generation(
                tsp_path,
                prior_path,
                lkh_binary,
                n=n,
                base_seed=int(runtime.get("global_seed", 0)),
                popmusic=params,
                prior_params=prior_params,
                timeout_s=float(edge_prior_cfg.get("generation_timeout_s", max(1800.0, prior_params.runs * 60.0))),
            )
            print(f"[edge prior cache] wrote {prior_path}")
        else:
            print(f"[edge prior cache] hit for {spec.name}: {prior_path}")
        raw_prior, edge_prior_meta = load_prior_npz(prior_path)
        edge_prior_source = str(prior_path)

    prior_map = None
    if use_prior:
        prior_map = transform_prior(
            raw_prior,
            mode=str(pop.get("prior_mode", "frequency")),
            n=n,
            seed=int(runtime.get("global_seed", 0)),
            topk=int(prior_params.topk),
        )

    problem = SparseTSPProblem(
        coords=coords,
        dist=dist,
        candidate_neighbors=candidate_map if use_candidates else None,
        prior_map=prior_map,
    )

    row = {
        "instance": spec.name,
        "n": n,
        "edge_weight_type": meta.get("EDGE_WEIGHT_TYPE", ""),
        "tsplib_path": str(tsp_path),
        "candidate_source": candidate_source,
        "requested_use_candidates": requested_use_candidates,
        "use_candidates": use_candidates,
        "use_prior": use_prior,
        "edge_prior_source": edge_prior_source,
        "edge_prior_success_runs": edge_prior_meta.get("success_runs"),
        "edge_prior_attempted_runs": edge_prior_meta.get("attempted_runs"),
        "edge_prior_topk": edge_prior_meta.get("topk"),
        "candidate_edge_policy": "guidance_only_full_tour_allowed",
    }
    pd.DataFrame([row]).to_csv(artifact_dir / f"problem_{spec.name}_load_info.csv", index=False)
    print(
        f"[load] {spec.name}: n={n}, type={row['edge_weight_type']}, "
        f"candidates={candidate_source}, prior={edge_prior_source if prior_map else 'off'}"
    )
    return spec.name, problem, spec.optimum


def load_selected_problems(cfg: dict, artifact_dir: Path, eval_split: str) -> list[tuple[str, SparseTSPProblem, float | None]]:
    problems = []
    for spec in selected_specs(cfg, eval_split):
        problems.append(load_problem_for_spec(cfg, spec, artifact_dir))
    if not problems:
        raise ValueError(f"No problems selected for eval_split={eval_split!r}")
    return problems


def make_llm_call(cfg: dict):
    llm = cfg.get("llm", {})
    max_keys = int(llm.get("groq_max_keys", 10))
    key_envs = ["GROQ_API_KEY"] + [f"GROQ_API_KEY_{i}" for i in range(1, max_keys + 1)]
    provider = str(llm.get("provider", "groq")).lower()
    if provider != "groq":
        raise ValueError(f"Only provider='groq' is implemented in this public TSP launcher, got {provider!r}")

    def _call(messages: list[dict[str, str]]) -> str:
        return call_groq_chat(
            messages,
            model=str(llm.get("model", "llama-3.3-70b-versatile")),
            api_key_envs=key_envs,
            timeout_s=float(llm.get("request_timeout_s", 60)),
            max_429_retries=int(llm.get("max_429_retries", 100)),
            max_request_error_retries=int(llm.get("max_request_error_retries", 5)),
            calls_per_minute_per_key=float(llm.get("calls_per_minute_per_key", 2)),
            temperature=float(llm.get("temperature", 0.8)),
            top_p=float(llm.get("top_p", 1.0)),
        )

    return _call




def jsonable(value):
    if value is None:
        return None
    try:
        import numpy as np
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.bool_):
            return bool(value)
    except Exception:
        pass
    if isinstance(value, Path):
        return str(value)
    return value

def loaded_groq_key_names(cfg: dict) -> list[str]:
    llm = cfg.get("llm", {})
    max_keys = int(llm.get("groq_max_keys", 10))
    key_envs = ["GROQ_API_KEY"] + [f"GROQ_API_KEY_{i}" for i in range(1, max_keys + 1)]
    return _loaded_groq_key_names(key_envs)


def print_tsp_prompt_excerpt(cfg: dict, max_chars: int = 2600) -> None:
    """Print only the objective block, matching the clustering launcher convention."""
    try:
        prompt = objective_prompt_block(cfg)
    except Exception as e:
        print(f"[warning] Could not build prompt excerpt: {type(e).__name__}: {e}")
        return
    excerpt = prompt[:max_chars].rstrip()
    print("\n--- Objective prompt excerpt ---")
    print(excerpt)
    if len(prompt) > max_chars:
        print("...")


def make_search_instances_table(problems: list[tuple[str, SparseTSPProblem, float | None]]) -> pd.DataFrame:
    rows = []
    for name, problem, optimum in problems:
        rows.append({
            "name": name,
            "n": int(problem.n),
            "optimum": None if optimum is None else float(optimum),
            "has_candidate_neighbors": problem.candidate_neighbors is not None,
            "has_prior": problem.prior_map is not None,
            "candidate_edge_policy": "guidance_only_full_tour_allowed",
        })
    return pd.DataFrame(rows)


def write_final_run_summaries(df: pd.DataFrame, artifact_dir: Path, cfg: dict) -> dict | None:
    """Write final summary aliases and best-code artifacts in clustering-style names."""
    save_json(artifact_dir / "llm_final_config.json", cfg)
    best = None
    if df is not None and not df.empty and "selection_score" in df:
        valid_df = df[df["full_valid"].astype(bool) & df["selection_score"].notna()]
        if len(valid_df):
            best_row = valid_df.sort_values(["selection_score", "attempt"]).iloc[0].to_dict()
            best = {k: jsonable(v) for k, v in best_row.items()}
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="YAML run config")
    parser.add_argument("--dry-run", action="store_true", help="Run a toy smoke test without TSPLIB files or LLM calls")
    args = parser.parse_args()

    cfg = load_run_config(args.config)
    if args.dry_run:
        cfg.setdefault("runtime", {})["dry_run"] = True
    rc = flatten_runtime_config(cfg)

    artifact_dir = make_run_dir(rc.artifact_root, rc.run_name)
    save_json(artifact_dir / "effective_config.json", cfg)
    save_json(artifact_dir / "llm_final_config.json", cfg)

    print("OBJECTIVE_MODE: tsp")
    print("CENTER_CONSTRAINT: permutation_tour")
    print(f"ARTIFACT_DIR: {artifact_dir}")
    print(f"run_name: {rc.run_name}")
    print("mode: LLaMEA only")
    print(f"llm: {rc.llm_provider} / {rc.llm_model}")
    print(f"max_llm_calls: {rc.max_llm_calls}")
    print(f"eval_split: {rc.eval_split}")
    print(f"candidate_timeout_s: {rc.candidate_timeout_s}")
    print(f"evaluation_timeout_s: {rc.evaluation_timeout_s}")
    print(f"use_popmusic_candidates: {rc.use_popmusic_candidates}")
    print(f"use_popmusic_edge_prior: {rc.use_popmusic_edge_prior}")
    print(f"effective_candidate_lists_exposed: {bool(rc.use_popmusic_candidates or rc.use_popmusic_edge_prior)}")
    print(f"popmusic_prior_mode: {rc.popmusic_prior_mode}")
    print(f"max_candidates: {rc.max_candidates}")
    print(f"edge_prior_cache_dir: {rc.edge_prior_cache_dir}")
    print(f"edge_prior_runs: {rc.edge_prior_runs} | time_limit_s: {rc.edge_prior_time_limit_s} | topk: {rc.edge_prior_topk}")
    print("candidate_edge_policy: guidance_only_full_tour_allowed")
    print("problem interface: sparse candidates + edge-cost oracle; no public dense distance matrix")

    selected_df = write_selected_instances(cfg, artifact_dir, rc.eval_split)
    print("\nSelected TSP instances:")
    print(selected_df[["instance", "split", "optimum", "tsplib_file_found", "candidate_file_found", "edge_prior_file_found"]])

    if rc.use_popmusic_candidates or rc.use_popmusic_edge_prior:
        missing_candidates = selected_df[selected_df["candidate_file_found"].astype(str) == ""]
        print(f"POPMUSIC candidate lists are exposed. Candidate files missing for {len(missing_candidates)}/{len(selected_df)} selected instance(s).")
        if len(missing_candidates):
            print("Missing candidate files will be generated immediately with LKH/POPMUSIC during problem loading.")

    if rc.dry_run:
        run_dry_smoke(cfg, artifact_dir)
        status = {
            "status": "dry_run_completed",
            "artifact_dir": str(artifact_dir),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        save_json(artifact_dir / "run_status.json", status)
        (artifact_dir / "pipeline_status.txt").write_text("dry_run_completed\n", encoding="utf-8")
        print("Dry-run completed.")
        print(f"Artifacts initialized in: {artifact_dir}")
        return

    status = {
        "status": "running_llamea",
        "artifact_dir": str(artifact_dir),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    save_json(artifact_dir / "run_status.json", status)
    (artifact_dir / "pipeline_status.txt").write_text("running_llamea\n", encoding="utf-8")

    print("\nLoading selected TSP problems...")
    problems = load_selected_problems(cfg, artifact_dir, rc.eval_split)
    search_instances = make_search_instances_table(problems)
    search_instances.to_csv(artifact_dir / "search_instances.csv", index=False)

    print("\nSearch instances:")
    print(search_instances)
    print("Final eval instances: 0")
    print("Final evaluation instance table is empty; final evaluation will be skipped.")

    print("Parser and safety checks ready.")
    loaded_keys = loaded_groq_key_names(cfg)
    print(f"Groq keys loaded: {loaded_keys}")
    print("LLM helpers ready. Groq timeout/retry helper is active.")
    print("Unified prompt builder ready for objective: tsp")
    print(f"Selection strategy: {rc.selection_strategy}")
    print(f"Historical family avoidance: {rc.historical_family_avoidance}")
    print(
        "Invalid-parent redesign: "
        f"{rc.invalid_parent_redesign} | any-invalid: {rc.redesign_on_any_invalid_before_full_valid} | "
        f"timeout: {rc.redesign_on_timeout_parent} | expose-invalid-code: {not rc.hide_invalid_parent_code}"
    )
    print_tsp_prompt_excerpt(cfg)
    print("Candidate evaluator ready.")

    llm_call = make_llm_call(cfg)
    df = run_llamea_search(cfg, problems, llm_call, artifact_dir)
    df.to_csv(artifact_dir / "generated_attempts.csv", index=False)
    if not df.empty:
        df.to_csv(artifact_dir / "llm_attempts.csv", index=False)

    best = write_final_run_summaries(df, artifact_dir, cfg)

    status = {
        "status": "llamea_completed",
        "artifact_dir": str(artifact_dir),
        "attempts": int(len(df)),
        "best": best,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    save_json(artifact_dir / "run_status.json", status)
    (artifact_dir / "pipeline_status.txt").write_text("llamea_completed\n", encoding="utf-8")

    print("Final evaluation skipped.")
    if best:
        print(f"Best full-valid attempt: {best['attempt']} score={best['selection_score']:.4f} gap={best['mean_gap_percent']:.4f}%")
    else:
        print("No full-valid candidate yet. Check llm_attempts.csv, candidates.jsonl, prompts/, codes/, and raw_responses/.")

    print(f"Artifacts directory: {artifact_dir}")


if __name__ == "__main__":
    main()
