#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm_tsp.config import load_run_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/run_prior_ablation.yaml")
    args = parser.parse_args()
    cfg = load_run_config(args.config)
    print("Prior ablation config loaded.")
    print("Methods:")
    for m in cfg.get("methods", []):
        print(f"  - {m}")
    print("This first version keeps the CLI placeholder; plug in the selected cleaned ablation implementation from the historical runs.")


if __name__ == "__main__":
    main()
