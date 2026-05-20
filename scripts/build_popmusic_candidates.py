#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm_tsp.lkh_popmusic import (
    PopmusicParams,
    EdgePriorParams,
    popmusic_candidate_file_name,
    popmusic_edge_prior_file_name,
    run_popmusic_candidate_generation,
    run_popmusic_edge_prior_generation,
)
from llm_tsp.tsplib_io import read_tsplib_coords


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate historical LKH/POPMUSIC TSP candidate and edge-prior caches.")
    parser.add_argument("--instance-name", required=True, help="TSPLIB instance stem, e.g. d1291")
    parser.add_argument("--tsp-file", required=True)
    parser.add_argument("--candidate-cache-dir", default="/content/drive/MyDrive/TM/LKH_candidate_cache")
    parser.add_argument("--edge-prior-cache-dir", default="/content/drive/MyDrive/TM/LKH_edge_prior_cache")
    parser.add_argument("--lkh-binary", default="/content/tools/lkh/LKH")
    parser.add_argument("--max-candidates", type=int, default=20)
    parser.add_argument("--popmusic-sample-size", type=int, default=14)
    parser.add_argument("--popmusic-solutions", type=int, default=20)
    parser.add_argument("--popmusic-max-neighbors", type=int, default=5)
    parser.add_argument("--popmusic-trials", type=int, default=1)
    parser.add_argument("--build-edge-prior", action="store_true")
    parser.add_argument("--edge-prior-runs", type=int, default=30)
    parser.add_argument("--edge-prior-time-limit-s", type=float, default=1.0)
    parser.add_argument("--edge-prior-topk", type=int, default=5)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()

    tsp_file = Path(args.tsp_file)
    pop = PopmusicParams(
        max_candidates=args.max_candidates,
        popmusic_sample_size=args.popmusic_sample_size,
        popmusic_solutions=args.popmusic_solutions,
        popmusic_max_neighbors=args.popmusic_max_neighbors,
        popmusic_trials=args.popmusic_trials,
        popmusic_initial_tour=False,
    )
    candidate_file = popmusic_candidate_file_name(args.instance_name, args.candidate_cache_dir, pop)
    print("Generating POPMUSIC candidate file:", candidate_file)
    run_popmusic_candidate_generation(tsp_file, candidate_file, args.lkh_binary, params=pop)
    print("candidate_file:", candidate_file)

    if args.build_edge_prior:
        coords, _ = read_tsplib_coords(tsp_file)
        n = int(len(coords))
        prior_params = EdgePriorParams(
            runs=args.edge_prior_runs,
            time_limit_s=args.edge_prior_time_limit_s,
            topk=args.edge_prior_topk,
            move_type=5,
            patching_a=2,
            patching_c=3,
            force_rebuild=True,
        )
        prior_file = popmusic_edge_prior_file_name(args.instance_name, args.edge_prior_cache_dir, prior_params)
        print("Generating LKH/POPMUSIC tour-frequency edge prior:", prior_file)
        run_popmusic_edge_prior_generation(
            tsp_file,
            prior_file,
            args.lkh_binary,
            n=n,
            base_seed=args.seed,
            popmusic=pop,
            prior_params=prior_params,
        )
        print("edge_prior_file:", prior_file)


if __name__ == "__main__":
    main()
