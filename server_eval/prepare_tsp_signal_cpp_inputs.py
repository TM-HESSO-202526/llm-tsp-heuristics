#!/usr/bin/env python3
"""Prepare lightweight text edge-prior files for the C++ TSP signal evaluator.

The Python evaluator can load `.npz` prior caches directly.  The C++ evaluator
uses a simple text format instead: one undirected edge per line, `i j weight`,
with 0-based node ids.  This script converts all requested instances before the
C++ 50-repetition signal run.

It supports both normal prior caches such as
`<instance>_popmusic_edge_prior_runs30_topk5.npz` and the historical `_work`
folder containing LKH `.tour` files, which is useful for the large usa13509
cache artifact.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm_tsp.lkh_popmusic import load_prior_npz, parse_lkh_tour_file
from llm_tsp.priors import transform_prior
from llm_tsp.tsplib_io import read_tsplib_coords


def tsp_path_for(instance: str, root: Path) -> Path:
    candidates = [
        root / f"{instance}.tsp",
        root / f"{instance}.TSP",
        root / instance / f"{instance}.tsp",
        root / instance / f"{instance}.TSP",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Missing TSPLIB file for {instance} under {root}")


def prior_npz_candidates(instance: str, root: Path) -> list[Path]:
    return [
        root / f"{instance}_popmusic_edge_prior_runs30_topk5.npz",
        root / f"{instance}_edge_prior_runs30_topk5.npz",
        root / f"{instance}_edge_prior.npz",
    ]


def prior_work_candidates(instance: str, root: Path) -> list[Path]:
    return [
        root / f"{instance}_popmusic_edge_prior_runs30_topk5_work",
        root / f"{instance}_edge_prior_runs30_topk5_work",
        root / f"{instance}_edge_prior_work",
    ]


def build_prior_from_tours(work_dir: Path, n: int, topk: int = 5) -> dict[tuple[int, int], float]:
    counts: dict[tuple[int, int], float] = defaultdict(float)
    success_runs = 0
    for tour_file in sorted(work_dir.glob("*.tour")):
        tour = parse_lkh_tour_file(tour_file)
        if len(tour) != n or len(set(tour)) != n:
            continue
        success_runs += 1
        for i, a in enumerate(tour):
            b = tour[(i + 1) % n]
            if a == b:
                continue
            e = tuple(sorted((int(a), int(b))))
            counts[e] += 1.0
    if success_runs <= 0 or not counts:
        return {}

    raw = {e: c / float(success_runs) for e, c in counts.items()}
    by_node: dict[int, list[tuple[int, float]]] = {i: [] for i in range(n)}
    for (a, b), w in raw.items():
        by_node[a].append((b, w))
        by_node[b].append((a, w))

    keep: dict[tuple[int, int], float] = {}
    for i, vals in by_node.items():
        vals.sort(key=lambda x: (-x[1], x[0]))
        for j, _w in vals[:topk]:
            e = tuple(sorted((i, j)))
            keep[e] = raw[e]
    return keep


def write_prior_text(out_path: Path, prior: dict[tuple[int, int], float]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("# 0-based undirected prior edges: i j weight\n")
        for (a, b), w in sorted(prior.items()):
            f.write(f"{int(a)} {int(b)} {float(w):.12g}\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", required=True, help="Comma-separated instance names")
    ap.add_argument("--instance-root", required=True)
    ap.add_argument("--edge-prior-cache-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--prior-mode", default="frequency")
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()

    instances = [x.strip() for x in args.instances.split(",") if x.strip()]
    inst_root = Path(args.instance_root).expanduser().resolve()
    prior_root = Path(args.edge_prior_cache_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Preparing prior text files for {len(instances)} instances")
    print(f"instance_root={inst_root}")
    print(f"edge_prior_cache_dir={prior_root}")
    print(f"out_dir={out_dir}")

    for name in instances:
        coords, _meta = read_tsplib_coords(tsp_path_for(name, inst_root))
        n = int(coords.shape[0])
        prior = None
        source = None

        for p in prior_npz_candidates(name, prior_root):
            if p.exists():
                raw_prior, meta = load_prior_npz(p)
                prior = transform_prior(raw_prior, mode=args.prior_mode, n=n, seed=0, topk=args.topk)
                source = f"npz:{p.name}:{meta}"
                break

        if prior is None:
            for d in prior_work_candidates(name, prior_root):
                if d.exists() and d.is_dir():
                    prior = build_prior_from_tours(d, n=n, topk=args.topk)
                    source = f"tour_work:{d.name}"
                    break

        if prior is None:
            raise FileNotFoundError(
                f"No prior cache found for {name} under {prior_root}. "
                f"Expected .npz or *_work tour directory."
            )

        out_path = out_dir / f"{name}_popmusic_edge_prior_runs30_topk5.prior.txt"
        write_prior_text(out_path, prior)
        print(f"OK {name}: n={n} edges={len(prior)} source={source} -> {out_path}")


if __name__ == "__main__":
    main()
