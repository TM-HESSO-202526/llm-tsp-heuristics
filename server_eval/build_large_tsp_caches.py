#!/usr/bin/env python3
"""Build LKH/POPMUSIC candidate and edge-prior cache files for large TSPLIB instances.

Default behavior matches the historical project cache generation:

  .tsp -> LKH POPMUSIC CANDIDATE_FILE (.cand)
  .tsp -> 30 short LKH/POPMUSIC tours -> edge-frequency prior (.npz)

The output filenames are the same historical filenames used by the notebooks:
  <instance>_cand-popmusic-k20-s14-sol20-nn5-tr1.cand
  <instance>_popmusic_edge_prior_runs30_topk5.npz

This script is intended for long-running server/tmux use.
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from llm_tsp.lkh_popmusic import (
    EdgePriorParams,
    PopmusicParams,
    ensure_lkh_binary,
    parse_lkh_tour_file,
    popmusic_candidate_file_name,
    popmusic_edge_prior_file_name,
    run_popmusic_candidate_generation,
    run_popmusic_edge_prior_generation,
    save_prior_npz,
    write_edge_prior_lkh_parameter_file,
)


def parse_dimension(tsp_file: Path) -> int:
    for raw in tsp_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        upper = line.upper()
        if upper.startswith("DIMENSION"):
            # Supports both "DIMENSION: 7397" and "DIMENSION 7397".
            cleaned = line.replace(":", " ")
            parts = cleaned.split()
            for tok in reversed(parts):
                try:
                    return int(tok)
                except ValueError:
                    pass
    raise ValueError(f"Could not parse DIMENSION from {tsp_file}")


def write_cached_candidate_prior_par_file(
    tsp_file: Path,
    candidate_file: Path,
    par_file: Path,
    tour_file: Path,
    *,
    seed: int,
    max_candidates: int,
    move_type: int,
    patching_a: int,
    patching_c: int,
    time_limit_s: float,
) -> None:
    """Write an LKH run that uses an already-generated candidate file.

    This is not the default historical method. It is a robust fallback for very
    large instances where rebuilding POPMUSIC candidates inside each short prior
    run may fail. It still produces a real tour-frequency prior from LKH tours.
    """
    par_file.parent.mkdir(parents=True, exist_ok=True)
    tour_file.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"PROBLEM_FILE = {tsp_file}",
        f"CANDIDATE_FILE = {candidate_file}",
        f"MAX_CANDIDATES = {int(max_candidates)}",
        "RUNS = 1",
        f"MOVE_TYPE = {int(move_type)}",
        f"PATCHING_A = {int(patching_a)}",
        f"PATCHING_C = {int(patching_c)}",
        f"SEED = {int(seed)}",
        f"TIME_LIMIT = {float(time_limit_s)}",
        "TRACE_LEVEL = 1",
        f"OUTPUT_TOUR_FILE = {tour_file}",
    ]
    par_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_cached_candidate_tour_frequency_prior(
    *,
    tsp_file: Path,
    candidate_file: Path,
    prior_file: Path,
    lkh_binary: Path,
    n: int,
    base_seed: int,
    runs: int,
    topk: int,
    max_candidates: int,
    time_limit_s: float,
    subprocess_timeout_s: float,
    move_type: int,
    patching_a: int,
    patching_c: int,
) -> Path:
    """Fallback edge prior generation using an existing .cand file.

    Output is the same sparse .npz schema used by the evaluator, but the work
    log records that cached candidates were supplied explicitly.
    """
    prior_file.parent.mkdir(parents=True, exist_ok=True)
    binary = ensure_lkh_binary(lkh_binary, build_if_missing=True, timeout_s=900.0)
    work_dir = prior_file.parent / (prior_file.stem + "_cached_candidate_work")
    log_dir = work_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    counts: dict[tuple[int, int], float] = defaultdict(float)
    success_runs = 0
    rows: list[dict[str, object]] = []

    for r in range(int(runs)):
        seed = int(base_seed) + r
        par_file = work_dir / f"run_{r+1:03d}.par"
        tour_file = work_dir / f"run_{r+1:03d}.tour"
        log_file = log_dir / f"run_{r+1:03d}.log"
        write_cached_candidate_prior_par_file(
            tsp_file,
            candidate_file,
            par_file,
            tour_file,
            seed=seed,
            max_candidates=max_candidates,
            move_type=move_type,
            patching_a=patching_a,
            patching_c=patching_c,
            time_limit_s=time_limit_s,
        )

        t0 = time.perf_counter()
        try:
            proc = subprocess.run(
                [str(binary), str(par_file)],
                text=True,
                capture_output=True,
                timeout=float(subprocess_timeout_s),
            )
            elapsed = time.perf_counter() - t0
            log_file.write_text(
                "STDOUT\n======\n" + proc.stdout + "\n\nSTDERR\n======\n" + proc.stderr,
                encoding="utf-8",
            )
            tour = parse_lkh_tour_file(tour_file)
            ok = proc.returncode == 0 and len(tour) == n and len(set(tour)) == n
            print(
                f"[{r+1:02d}/{runs}] cached-cand prior {tsp_file.stem}: "
                f"return={proc.returncode} tour_len={len(tour)} ok={ok} time={elapsed:.1f}s",
                flush=True,
            )
            if ok:
                success_runs += 1
                for k, a in enumerate(tour):
                    b = tour[(k + 1) % n]
                    if a == b:
                        continue
                    edge = tuple(sorted((int(a), int(b))))
                    counts[edge] += 1.0
            rows.append(
                {
                    "run": r + 1,
                    "seed": seed,
                    "returncode": int(proc.returncode),
                    "tour_len": len(tour),
                    "success": bool(ok),
                    "elapsed_s": elapsed,
                }
            )
        except subprocess.TimeoutExpired:
            elapsed = time.perf_counter() - t0
            print(f"[{r+1:02d}/{runs}] cached-cand prior {tsp_file.stem}: TIMEOUT after {elapsed:.1f}s", flush=True)
            (log_dir / f"run_{r+1:03d}.timeout").write_text("timeout\n", encoding="utf-8")
            rows.append(
                {
                    "run": r + 1,
                    "seed": seed,
                    "returncode": -999,
                    "tour_len": 0,
                    "success": False,
                    "elapsed_s": elapsed,
                }
            )

    with (work_dir / "edge_prior_runs.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["run", "seed", "returncode", "tour_len", "success", "elapsed_s"])
        writer.writeheader()
        writer.writerows(rows)

    if success_runs <= 0:
        raise RuntimeError(f"No successful cached-candidate LKH tours for {tsp_file.name}. Check {work_dir}")

    raw = {edge: float(val) / float(success_runs) for edge, val in counts.items()}

    # Keep top-k per node using the observed frequencies.
    incident: list[list[tuple[float, tuple[int, int]]]] = [[] for _ in range(n)]
    for edge, weight in raw.items():
        a, b = edge
        incident[a].append((weight, edge))
        incident[b].append((weight, edge))

    keep: set[tuple[int, int]] = set()
    for i in range(n):
        incident[i].sort(key=lambda x: x[0], reverse=True)
        for _, edge in incident[i][: int(topk)]:
            keep.add(edge)
    prior = {edge: raw[edge] for edge in keep}

    save_prior_npz(prior_file, prior, success_runs=success_runs, attempted_runs=int(runs), topk=int(topk))
    print(
        f"WROTE fallback cached-candidate tour-frequency prior: {prior_file} "
        f"success_runs={success_runs}/{runs} edges={len(prior)}",
        flush=True,
    )
    return prior_file


def _valid_lkh_tour(path: Path, n: int) -> bool:
    tour = parse_lkh_tour_file(path)
    return len(tour) == int(n) and len(set(tour)) == int(n) and (not tour or (min(tour) == 0 and max(tour) == int(n) - 1))


def _aggregate_tours_to_prior_npz(
    *,
    work_dir: Path,
    prior_file: Path,
    n: int,
    runs: int,
    topk: int,
    base_seed: int,
) -> Path:
    """Aggregate existing run_XXX.tour files into the sparse edge-prior .npz schema."""
    counts: dict[tuple[int, int], float] = defaultdict(float)
    rows: list[dict[str, object]] = []
    success_runs = 0

    for r in range(1, int(runs) + 1):
        tour_file = work_dir / f"run_{r:03d}.tour"
        tour = parse_lkh_tour_file(tour_file)
        ok = len(tour) == int(n) and len(set(tour)) == int(n) and (not tour or (min(tour) == 0 and max(tour) == int(n) - 1))
        rows.append({
            "run": r,
            "seed": int(base_seed) + r - 1,
            "returncode": 0 if ok else -1,
            "tour_len": len(tour),
            "success": bool(ok),
        })
        print(f"aggregate run_{r:03d}: len={len(tour)} ok={ok}", flush=True)
        if not ok:
            raise RuntimeError(f"Missing or invalid tour before aggregation: {tour_file}")

        success_runs += 1
        for k, a in enumerate(tour):
            b = tour[(k + 1) % int(n)]
            if a == b:
                continue
            edge = tuple(sorted((int(a), int(b))))
            counts[edge] += 1.0

    raw = {edge: float(val) / float(success_runs) for edge, val in counts.items()}

    incident: list[list[tuple[float, tuple[int, int]]]] = [[] for _ in range(int(n))]
    for edge, weight in raw.items():
        a, b = edge
        incident[a].append((weight, edge))
        incident[b].append((weight, edge))

    keep: set[tuple[int, int]] = set()
    for i in range(int(n)):
        incident[i].sort(key=lambda x: x[0], reverse=True)
        for _, edge in incident[i][: int(topk)]:
            keep.add(edge)
    prior = {edge: raw[edge] for edge in keep}

    save_prior_npz(prior_file, prior, success_runs=success_runs, attempted_runs=int(runs), topk=int(topk))

    with (work_dir / "edge_prior_runs.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["run", "seed", "returncode", "tour_len", "success"])
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"WROTE parallel edge prior: {prior_file} success_runs={success_runs}/{runs} edges={len(prior)}",
        flush=True,
    )
    return prior_file


def _parse_core_list(core_list: str | None, parallel: int) -> list[int] | None:
    if not core_list:
        return None
    cores = [int(x.strip()) for x in str(core_list).split(",") if x.strip()]
    if not cores:
        return None
    return cores[: max(1, int(parallel))]


def build_parallel_tour_frequency_prior(
    *,
    tsp_file: Path,
    prior_file: Path,
    lkh_binary: Path,
    n: int,
    base_seed: int,
    runs: int,
    topk: int,
    parallel_runs: int,
    core_list: str | None,
    method: str,
    popmusic: PopmusicParams,
    prior_params: EdgePriorParams,
    candidate_file: Path | None = None,
    max_candidates: int = 20,
    subprocess_timeout_s: float = 86400.0,
) -> Path:
    """Build or resume edge-prior tours in parallel, then aggregate them into the final .npz.

    This is intended for very large instances.  It preserves the historical work directory for
    historical_popmusic, so already-generated tours such as usa13509 run_001..run_013 are reused.
    """
    prior_file.parent.mkdir(parents=True, exist_ok=True)
    binary = ensure_lkh_binary(lkh_binary, build_if_missing=True, timeout_s=900.0)

    if method == "historical_popmusic":
        work_dir = prior_file.parent / (prior_file.stem + "_work")
    elif method == "cached_candidate_lkh":
        if candidate_file is None or not candidate_file.exists() or candidate_file.stat().st_size <= 0:
            raise FileNotFoundError(f"cached_candidate_lkh needs an existing candidate file: {candidate_file}")
        work_dir = prior_file.parent / (prior_file.stem + "_cached_candidate_work")
    else:
        raise ValueError(f"Unsupported parallel prior method: {method}")

    log_dir = work_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    missing: list[int] = []
    print("=== DETECT EXISTING PRIOR TOURS ===", flush=True)
    for r in range(1, int(runs) + 1):
        tour = work_dir / f"run_{r:03d}.tour"
        if _valid_lkh_tour(tour, int(n)):
            print(f"run_{r:03d} OK existing", flush=True)
        else:
            print(f"run_{r:03d} missing/invalid -> will run", flush=True)
            missing.append(r)

    print(f"Missing/invalid prior runs: {len(missing)}/{runs}", flush=True)
    cores = _parse_core_list(core_list, int(parallel_runs))
    max_parallel = max(1, int(parallel_runs))
    active: list[tuple[int, subprocess.Popen, Path, float]] = []
    failures: list[int] = []

    def write_par_for_run(r: int) -> Path:
        seed = int(base_seed) + r - 1
        par_file = work_dir / f"run_{r:03d}.par"
        tour_file = work_dir / f"run_{r:03d}.tour"
        if method == "historical_popmusic":
            write_edge_prior_lkh_parameter_file(
                tsp_file,
                par_file,
                tour_file,
                seed=seed,
                popmusic=popmusic,
                prior=prior_params,
            )
        else:
            write_cached_candidate_prior_par_file(
                tsp_file,
                candidate_file,  # type: ignore[arg-type]
                par_file,
                tour_file,
                seed=seed,
                max_candidates=max_candidates,
                move_type=prior_params.move_type,
                patching_a=prior_params.patching_a,
                patching_c=prior_params.patching_c,
                time_limit_s=prior_params.time_limit_s,
            )
        return par_file

    def launch(r: int) -> None:
        par = write_par_for_run(r)
        log = log_dir / f"run_{r:03d}.parallel.log"
        core = None
        if cores:
            core = cores[(r - 1) % len(cores)]
        cmd = [str(binary), str(par)]
        if core is not None:
            cmd = ["taskset", "-c", str(core)] + cmd
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] START run_{r:03d} core={core} cmd={' '.join(cmd)}", flush=True)
        fh = log.open("w", encoding="utf-8")
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, text=True)
        # Keep file handle attached to proc object so it is not garbage collected too early.
        proc._llm_tsp_log_fh = fh  # type: ignore[attr-defined]
        active.append((r, proc, log, time.perf_counter()))

    def reap_one(block: bool) -> None:
        while True:
            for i, (r, proc, log, t0) in enumerate(list(active)):
                code = proc.poll()
                if code is None:
                    # Optional hard outer timeout to avoid hidden infinite jobs.
                    if subprocess_timeout_s and (time.perf_counter() - t0) > float(subprocess_timeout_s):
                        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] TIMEOUT run_{r:03d}; killing", flush=True)
                        proc.kill()
                        code = proc.wait()
                    else:
                        continue
                try:
                    proc._llm_tsp_log_fh.close()  # type: ignore[attr-defined]
                except Exception:
                    pass
                active.pop(i)
                ok = code == 0 and _valid_lkh_tour(work_dir / f"run_{r:03d}.tour", int(n))
                elapsed = time.perf_counter() - t0
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] DONE run_{r:03d} return={code} ok={ok} time={elapsed:.1f}s log={log}", flush=True)
                if not ok:
                    failures.append(r)
                return
            if not block:
                return
            time.sleep(2.0)

    for r in missing:
        while len(active) >= max_parallel:
            reap_one(block=True)
        launch(r)

    while active:
        reap_one(block=True)

    if failures:
        raise RuntimeError(f"Some LKH prior runs failed validation: {failures}. Check {log_dir}")

    return _aggregate_tours_to_prior_npz(
        work_dir=work_dir,
        prior_file=prior_file,
        n=int(n),
        runs=int(runs),
        topk=int(topk),
        base_seed=int(base_seed),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build large TSP candidate and edge-prior caches.")
    parser.add_argument("--instances", default="pla7397,usa13509", help="Comma-separated instance names.")
    parser.add_argument("--instance-root", required=True, type=Path)
    parser.add_argument("--candidate-cache-dir", required=True, type=Path)
    parser.add_argument("--edge-prior-dir", required=True, type=Path)
    parser.add_argument("--lkh-binary", default="/home/anthony.atallah/data-local/TM/tools/lkh/LKH", type=Path)

    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=12345)
    parser.add_argument("--time-limit-s", type=float, default=600.0, help="LKH TIME_LIMIT per prior run.")
    parser.add_argument("--subprocess-timeout-s", type=float, default=900.0, help="Outer subprocess timeout per prior run.")
    parser.add_argument("--candidate-timeout-s", type=float, default=14400.0, help="Timeout for candidate-file generation.")

    parser.add_argument("--max-candidates", type=int, default=20)
    parser.add_argument("--popmusic-sample-size", type=int, default=14)
    parser.add_argument("--popmusic-solutions", type=int, default=20)
    parser.add_argument("--popmusic-max-neighbors", type=int, default=5)
    parser.add_argument("--popmusic-trials", type=int, default=1)
    parser.add_argument("--move-type", type=int, default=5)
    parser.add_argument("--patching-a", type=int, default=2)
    parser.add_argument("--patching-c", type=int, default=3)

    parser.add_argument("--force-candidate", action="store_true")
    parser.add_argument("--force-prior", action="store_true")
    parser.add_argument(
        "--prior-method",
        choices=["historical_popmusic", "cached_candidate_lkh"],
        default="historical_popmusic",
        help=(
            "historical_popmusic matches the original smaller-instance prior builder. "
            "cached_candidate_lkh is a robust tour-frequency fallback using an existing .cand file."
        ),
    )
    parser.add_argument(
        "--fallback-cached-candidate",
        action="store_true",
        help="If historical_popmusic fails, retry with cached_candidate_lkh.",
    )
    parser.add_argument(
        "--parallel-prior-runs",
        type=int,
        default=1,
        help="Run missing LKH edge-prior tours in parallel. Use 10 on zeus when cores are available.",
    )
    parser.add_argument(
        "--core-list",
        default="",
        help="Optional comma-separated CPU core list used with taskset for parallel prior runs, e.g. 0,1,2,3,4,5,6,7,8,9.",
    )
    args = parser.parse_args()

    pop = PopmusicParams(
        max_candidates=args.max_candidates,
        popmusic_sample_size=args.popmusic_sample_size,
        popmusic_solutions=args.popmusic_solutions,
        popmusic_max_neighbors=args.popmusic_max_neighbors,
        popmusic_trials=args.popmusic_trials,
        popmusic_initial_tour=False,
    )
    prior_params = EdgePriorParams(
        runs=args.runs,
        time_limit_s=args.time_limit_s,
        topk=args.topk,
        move_type=args.move_type,
        patching_a=args.patching_a,
        patching_c=args.patching_c,
        force_rebuild=args.force_prior,
    )

    args.candidate_cache_dir.mkdir(parents=True, exist_ok=True)
    args.edge_prior_dir.mkdir(parents=True, exist_ok=True)

    instances = [x.strip() for x in args.instances.split(",") if x.strip()]
    print("Instances:", instances, flush=True)
    print("Candidate method: LKH POPMUSIC .cand generation", flush=True)
    print("Prior method:", args.prior_method, flush=True)
    print("LKH binary target:", args.lkh_binary, flush=True)

    # Build LKH early so failures happen before a long run starts.
    lkh_binary = ensure_lkh_binary(args.lkh_binary, build_if_missing=True, timeout_s=900.0)
    print("Using LKH:", lkh_binary, flush=True)

    for idx, name in enumerate(instances, start=1):
        print("=" * 100, flush=True)
        print(f"[{idx}/{len(instances)}] {name}", flush=True)
        tsp_file = args.instance_root / f"{name}.tsp"
        if not tsp_file.exists():
            raise FileNotFoundError(f"Missing TSP file: {tsp_file}")
        n = parse_dimension(tsp_file)
        print(f"tsp_file: {tsp_file} n={n}", flush=True)

        candidate_file = popmusic_candidate_file_name(name, args.candidate_cache_dir, pop)
        prior_file = popmusic_edge_prior_file_name(name, args.edge_prior_dir, prior_params)

        if candidate_file.exists() and candidate_file.stat().st_size > 0 and not args.force_candidate:
            print(f"candidate exists: {candidate_file}", flush=True)
        else:
            print(f"generating candidate: {candidate_file}", flush=True)
            t0 = time.perf_counter()
            run_popmusic_candidate_generation(
                tsp_file,
                candidate_file,
                lkh_binary,
                params=pop,
                timeout_s=args.candidate_timeout_s,
            )
            print(f"candidate generated: {candidate_file} time={time.perf_counter()-t0:.1f}s", flush=True)

        if prior_file.exists() and prior_file.stat().st_size > 0 and not args.force_prior:
            print(f"edge prior exists: {prior_file}", flush=True)
            continue

        print(
            f"generating edge prior: {prior_file} n={n}, runs={args.runs}, "
            f"topk={args.topk}, time_limit_s={args.time_limit_s}, method={args.prior_method}",
            flush=True,
        )
        t0 = time.perf_counter()
        try:
            if args.parallel_prior_runs and int(args.parallel_prior_runs) > 1:
                build_parallel_tour_frequency_prior(
                    tsp_file=tsp_file,
                    prior_file=prior_file,
                    lkh_binary=lkh_binary,
                    n=n,
                    base_seed=args.base_seed,
                    runs=args.runs,
                    topk=args.topk,
                    parallel_runs=args.parallel_prior_runs,
                    core_list=args.core_list,
                    method=args.prior_method,
                    popmusic=pop,
                    prior_params=prior_params,
                    candidate_file=candidate_file,
                    max_candidates=args.max_candidates,
                    subprocess_timeout_s=args.subprocess_timeout_s,
                )
            elif args.prior_method == "historical_popmusic":
                run_popmusic_edge_prior_generation(
                    tsp_file,
                    prior_file,
                    lkh_binary,
                    n=n,
                    base_seed=args.base_seed,
                    popmusic=pop,
                    prior_params=prior_params,
                    timeout_s=max(args.subprocess_timeout_s, args.time_limit_s + 60.0),
                )
            else:
                build_cached_candidate_tour_frequency_prior(
                    tsp_file=tsp_file,
                    candidate_file=candidate_file,
                    prior_file=prior_file,
                    lkh_binary=lkh_binary,
                    n=n,
                    base_seed=args.base_seed,
                    runs=args.runs,
                    topk=args.topk,
                    max_candidates=args.max_candidates,
                    time_limit_s=args.time_limit_s,
                    subprocess_timeout_s=args.subprocess_timeout_s,
                    move_type=args.move_type,
                    patching_a=args.patching_a,
                    patching_c=args.patching_c,
                )
        except Exception as exc:
            if args.prior_method == "historical_popmusic" and args.fallback_cached_candidate:
                print(f"historical prior failed for {name}: {type(exc).__name__}: {exc}", flush=True)
                print("Retrying with cached_candidate_lkh fallback...", flush=True)
                build_cached_candidate_tour_frequency_prior(
                    tsp_file=tsp_file,
                    candidate_file=candidate_file,
                    prior_file=prior_file,
                    lkh_binary=lkh_binary,
                    n=n,
                    base_seed=args.base_seed,
                    runs=args.runs,
                    topk=args.topk,
                    max_candidates=args.max_candidates,
                    time_limit_s=args.time_limit_s,
                    subprocess_timeout_s=args.subprocess_timeout_s,
                    move_type=args.move_type,
                    patching_a=args.patching_a,
                    patching_c=args.patching_c,
                )
            else:
                raise
        print(f"edge prior generated: {prior_file} time={time.perf_counter()-t0:.1f}s", flush=True)

    print("=" * 100, flush=True)
    print("DONE building requested TSP caches.", flush=True)


if __name__ == "__main__":
    main()
