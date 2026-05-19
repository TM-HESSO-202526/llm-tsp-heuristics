#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize generated_attempts.csv files under an artifact root.")
    parser.add_argument("root", help="Run artifact root")
    parser.add_argument("--out", default="tsp_run_summary.csv")
    args = parser.parse_args()

    rows = []
    for path in Path(args.root).rglob("generated_attempts.csv"):
        df = pd.read_csv(path)
        if df.empty:
            continue
        valid = df[df.get("valid", False) == True]
        best_gap = valid["mean_gap_percent"].min() if not valid.empty and "mean_gap_percent" in valid else None
        rows.append({"run_dir": str(path.parent), "attempts": len(df), "valid": len(valid), "best_gap_percent": best_gap})
    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    print(f"Wrote {args.out} with {len(out)} row(s).")


if __name__ == "__main__":
    main()
