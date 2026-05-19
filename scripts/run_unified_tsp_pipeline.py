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
from llm_tsp.distance import euclidean_matrix, tour_cost_from_matrix
from llm_tsp.sparse_problem import SparseTSPProblem
from llm_tsp.candidate_sets import k_nearest_candidates
from llm_tsp.baselines import nearest_neighbor, prior_greedy
from llm_tsp.suite import specs_from_suite_config, filter_specs


def make_toy_problem(n: int = 100, seed: int = 0, use_candidates: bool = False, max_k: int = 20) -> SparseTSPProblem:
    rng = np.random.default_rng(seed)
    coords = rng.random((n, 2)) * 1000.0
    dist = euclidean_matrix(coords)
    cands = k_nearest_candidates(dist, max_k=max_k) if use_candidates else None
    return SparseTSPProblem(coords=coords, dist=dist, candidate_neighbors=cands, restrict_edge_cost_to_candidates=False)


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
    nn_cost = tour_cost_from_matrix(tour, problem.dist)
    rows = [{"method": "nearest_neighbor", "n": problem.n, "cost": float(nn_cost)}]
    print("Dry-run smoke test")
    print(f"n={problem.n} nearest_neighbor_cost={nn_cost:.3f}")
    if pop.get("use_popmusic_edge_prior"):
        tour2 = prior_greedy(problem)
        prior_cost = tour_cost_from_matrix(tour2, problem.dist)
        rows.append({"method": "prior_greedy_placeholder", "n": problem.n, "cost": float(prior_cost)})
        print(f"prior_greedy_cost={prior_cost:.3f}")
    pd.DataFrame(rows).to_csv(artifact_dir / "dry_run_smoke_results.csv", index=False)


def candidate_paths_for_instance(instance_name: str, root: Path) -> list[Path]:
    return [
        root / f"{instance_name}.cand",
        root / f"{instance_name}.candidates",
        root / f"{instance_name}_candidates.txt",
        root / f"{instance_name}.txt",
    ]


def tsplib_paths_for_instance(instance_name: str, root: Path) -> list[Path]:
    return [
        root / f"{instance_name}.tsp",
        root / f"{instance_name}.TSP",
        root / instance_name / f"{instance_name}.tsp",
        root / instance_name / f"{instance_name}.TSP",
    ]


def select_existing(paths: list[Path]) -> str:
    for p in paths:
        if p.exists():
            return str(p)
    return ""


def write_selected_instances(cfg: dict, artifact_dir: Path, eval_split: str) -> pd.DataFrame:
    suite = cfg.get("suite", {})
    instance_root = Path(suite.get("instance_root", ""))
    candidate_root = Path(suite.get("candidate_cache_dir", ""))
    specs = specs_from_suite_config(suite)
    selected = specs if eval_split == "all" else filter_specs(specs, split=eval_split)
    rows = []
    for spec in selected:
        tsplib_candidates = tsplib_paths_for_instance(spec.name, instance_root)
        candidate_candidates = candidate_paths_for_instance(spec.name, candidate_root)
        rows.append({
            "instance": spec.name,
            "split": spec.split,
            "optimum": spec.optimum,
            "tsplib_file_found": select_existing(tsplib_candidates),
            "tsplib_first_expected": str(tsplib_candidates[0]),
            "candidate_file_found": select_existing(candidate_candidates),
            "candidate_first_expected": str(candidate_candidates[0]),
        })
    df = pd.DataFrame(rows)
    df.to_csv(artifact_dir / "selected_instances.csv", index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="YAML run config")
    parser.add_argument("--dry-run", action="store_true", help="Run a toy smoke test without TSPLIB files or LLM calls")
    args = parser.parse_args()

    cfg = load_run_config(args.config)
    if args.dry_run:
        cfg.setdefault("runtime", {})["dry_run"] = True
    rc = flatten_runtime_config(cfg)

    print("Unified TSP pipeline")
    print("-" * 80)
    print(f"run_name: {rc.run_name}")
    print(f"experiment_mode: {rc.experiment_mode}")
    print(f"llm: {rc.llm_provider} / {rc.llm_model}")
    print(f"max_llm_calls: {rc.max_llm_calls}")
    print(f"smoke_test: {rc.smoke_test}")
    print(f"dry_run: {rc.dry_run}")
    print(f"eval_split: {rc.eval_split}")
    print(f"candidate_timeout_s: {rc.candidate_timeout_s}")
    print(f"evaluation_timeout_s: {rc.evaluation_timeout_s}")
    print(f"use_popmusic_candidates: {rc.use_popmusic_candidates}")
    print(f"use_popmusic_edge_prior: {rc.use_popmusic_edge_prior}")
    print(f"popmusic_prior_mode: {rc.popmusic_prior_mode}")
    print(f"max_candidates: {rc.max_candidates}")
    print(f"restrict_edge_cost_to_candidates: {rc.restrict_edge_cost_to_candidates}")
    print("-" * 80)

    artifact_dir = make_run_dir(rc.artifact_root, rc.run_name)
    print(f"ARTIFACT_DIR: {artifact_dir}")
    save_json(artifact_dir / "effective_config.json", cfg)

    selected_df = write_selected_instances(cfg, artifact_dir, rc.eval_split)
    print(f"Selected {len(selected_df)} instance(s) for split={rc.eval_split}:")
    for _, row in selected_df.iterrows():
        found = "found" if row["tsplib_file_found"] else "missing"
        print(f"  - {row['instance']} opt={row['optimum']} tsplib={found}")

    if rc.use_popmusic_candidates:
        missing_candidates = selected_df[selected_df["candidate_file_found"].astype(str) == ""]
        print(f"POPMUSIC candidate mode is ON. Candidate files missing for {len(missing_candidates)}/{len(selected_df)} selected instance(s).")
        if len(missing_candidates):
            print("Expected candidate-file naming examples are saved in selected_instances.csv.")

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
        "status": "harness_initialized",
        "note": "This public first version exposes the cleaned TSP control surface. Plug in historical LLaMEA provider calls/candidate parsers as needed.",
        "artifact_dir": str(artifact_dir),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    save_json(artifact_dir / "run_status.json", status)
    (artifact_dir / "pipeline_status.txt").write_text("harness_initialized\n", encoding="utf-8")

    print("\nThis first repo version sets up the cleaned harness/config surface.")
    print("Next step: plug in the selected historical TSP LLaMEA call loop and POPMUSIC candidate-cache parser for your exact artifact format.")
    print(f"Artifacts initialized in: {artifact_dir}")


if __name__ == "__main__":
    main()
