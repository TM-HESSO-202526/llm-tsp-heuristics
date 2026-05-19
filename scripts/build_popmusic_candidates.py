#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm_tsp.lkh_popmusic import PopmusicParams, write_lkh_parameter_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Create LKH parameter files for POPMUSIC candidate generation.")
    parser.add_argument("--tsp-file", required=True)
    parser.add_argument("--candidate-file", required=True)
    parser.add_argument("--par-file", required=True)
    parser.add_argument("--max-candidates", type=int, default=20)
    args = parser.parse_args()

    params = PopmusicParams(max_candidates=args.max_candidates)
    par = write_lkh_parameter_file(args.tsp_file, args.candidate_file, args.par_file, params=params)
    print(f"Wrote LKH parameter file: {par}")
    print("Run LKH manually, e.g.: /content/tools/lkh/LKH", par)


if __name__ == "__main__":
    main()
