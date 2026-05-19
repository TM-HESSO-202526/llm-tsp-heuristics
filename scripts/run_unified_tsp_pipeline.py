#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys
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


def run_dry_smoke(cfg: dict) -> None:
    pop = cfg.get("popmusic", {})
    problem = make_toy_problem(use_candidates=bool(pop.get("use_popmusic_candidates", False)), max_k=int(pop.get("max_candidates", 20)))
    tour = nearest_neighbor(problem)
    cost = tour_cost_from_matrix(tour, problem.dist)
    print("Dry-run smoke test")
    print(f"n={problem.n} nearest_neighbor_cost={cost:.3f}")
    if pop.get("use_popmusic_edge_prior"):
        tour2 = prior_greedy(problem)
        print(f"prior_greedy_cost={tour_cost_from_matrix(tour2, problem.dist):.3f}")


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
    print("-" * 72)
    print(f"run_name: {rc.run_name}")
    print(f"experiment_mode: {rc.experiment_mode}")
    print(f"eval_split: {rc.eval_split}")
    print(f"max_llm_calls: {rc.max_llm_calls}")
    print(f"use_popmusic_candidates: {rc.use_popmusic_candidates}")
    print(f"use_popmusic_edge_prior: {rc.use_popmusic_edge_prior}")
    print(f"popmusic_prior_mode: {rc.popmusic_prior_mode}")
    print(f"max_candidates: {rc.max_candidates}")
    print("-" * 72)

    if cfg.get("runtime", {}).get("dry_run", False):
        run_dry_smoke(cfg)
        return

    artifact_dir = make_run_dir(rc.artifact_root, rc.run_name)
    save_json(artifact_dir / "effective_config.json", cfg)

    suite = cfg.get("suite", {})
    specs = filter_specs(specs_from_suite_config(suite), split=rc.eval_split)
    print(f"Selected {len(specs)} instance(s) for split={rc.eval_split}:")
    for s in specs:
        print(f"  - {s.name} opt={s.optimum}")

    print("\nThis first repo version sets up the harness and config surface.")
    print("Next step: plug in the selected historical LLaMEA call loop and POPMUSIC candidate cache parser for your exact artifact format.")
    print(f"Artifacts initialized in: {artifact_dir}")


if __name__ == "__main__":
    main()
