#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import csv
import importlib.util
import inspect
import io
import json
import math
import os
from pathlib import Path
import platform
import signal
import socket
import sys
import time
import traceback
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm_tsp.candidate_sets import normalize_candidates, parse_simple_candidate_file
from llm_tsp.distance import tour_cost_from_matrix, make_tsplib_distance, validate_tour
from llm_tsp.lkh_popmusic import (
    EdgePriorParams,
    PopmusicParams,
    popmusic_candidate_file_name,
    popmusic_edge_prior_file_name,
    load_prior_npz,
)
from llm_tsp.priors import transform_prior
from llm_tsp.sparse_problem import SparseTSPProblem
from llm_tsp.tsplib_io import read_tsplib_coords

# NOTE: tsplib_paths_for_instance lives in scripts/run_unified_tsp_pipeline.py, not src.
# We keep a local copy here so this server evaluator is independent of the LLM launcher.


RAW_COLUMNS = [
    "signal_category",
    "heuristic_id",
    "heuristic_label",
    "code_path",
    "instance_name",
    "split",
    "n",
    "rep",
    "seed",
    "objective_value",
    "reference_value",
    "gap_ref_pct",
    "runtime_s",
    "status",
    "error_type",
    "error_message",
    "uses_only_candidates",
    "candidate_edge_count",
    "total_edges",
    "candidate_edge_share",
    "stdout_tail",
    "hostname",
]


class CandidateTimeoutError(TimeoutError):
    pass


@contextlib.contextmanager
def time_limit(timeout_s: float | None):
    """Hard-ish Unix timeout for generated heuristic calls.

    This is effective for normal Python loops. Very long C/NumPy calls may only
    be interrupted when control returns to Python, but that is still enough for
    the generated TSP code in this repository.
    """
    if timeout_s is None or timeout_s <= 0 or not hasattr(signal, "SIGALRM"):
        yield
        return

    def _handler(signum, frame):
        raise CandidateTimeoutError(f"candidate timed out after {float(timeout_s):.1f}s")

    old_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, float(timeout_s))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, old_handler)


def parse_csv_filter(value: str | None, *, cast=str) -> list[Any] | None:
    if value is None:
        return None
    value = str(value).strip()
    if not value or value.upper() == "ALL":
        return None
    return [cast(x.strip()) for x in value.split(",") if x.strip()]


def tsp_paths_for_instance(instance_name: str, root: Path) -> list[Path]:
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


def load_optima(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "instance" not in df.columns:
        raise ValueError(f"{path} must contain an instance column")
    if "optimum" not in df.columns:
        raise ValueError(f"{path} must contain an optimum column")
    if "split" not in df.columns:
        df["split"] = "all"
    return df


def candidate_paths_for_instance(instance_name: str, root: Path, params: PopmusicParams) -> list[Path]:
    return [
        popmusic_candidate_file_name(instance_name, root, params),
        root / f"{instance_name}.cand",
        root / f"{instance_name}.candidates",
        root / f"{instance_name}_candidates.txt",
        root / f"{instance_name}.txt",
    ]


def edge_prior_paths_for_instance(instance_name: str, root: Path, params: EdgePriorParams) -> list[Path]:
    return [
        popmusic_edge_prior_file_name(instance_name, root, params),
        root / f"{instance_name}_popmusic_edge_prior_runs30_topk5.npz",
        root / f"{instance_name}_edge_prior_runs30_topk5.npz",
        root / f"{instance_name}_edge_prior.npz",
    ]


@dataclass
class TSPInstance:
    name: str
    split: str
    n: int
    optimum: float
    tsp_path: Path
    coords: np.ndarray
    dist: Any
    edge_weight_type: str


@dataclass
class HeuristicSpec:
    signal_category: str
    heuristic_id: str
    heuristic_label: str
    code_path: Path


def load_instances(
    *,
    instance_root: Path,
    optima_csv: Path,
    instances_filter: str,
    split_filter: str,
    max_instances: int,
    dense_distance_threshold: int,
) -> list[TSPInstance]:
    opt = load_optima(optima_csv)
    wanted_instances = parse_csv_filter(instances_filter, cast=str)
    wanted_splits = parse_csv_filter(split_filter, cast=str)

    if wanted_instances is not None:
        opt = opt[opt["instance"].astype(str).isin(set(wanted_instances))]
    if wanted_splits is not None:
        opt = opt[opt["split"].astype(str).isin(set(wanted_splits))]

    rows = []
    for _, r in opt.iterrows():
        name = str(r["instance"])
        tsp_path = select_existing_path(tsp_paths_for_instance(name, instance_root))
        if tsp_path is None:
            raise FileNotFoundError(
                f"Missing TSPLIB file for {name}. Expected one of: "
                + ", ".join(str(p) for p in tsp_paths_for_instance(name, instance_root))
            )
        coords, meta = read_tsplib_coords(tsp_path)
        edge_weight_type = str(meta.get("EDGE_WEIGHT_TYPE", ""))
        dist = make_tsplib_distance(coords, edge_weight_type, dense_threshold=dense_distance_threshold)
        rows.append(
            TSPInstance(
                name=name,
                split=str(r.get("split", "all")),
                n=int(coords.shape[0]),
                optimum=float(r["optimum"]),
                tsp_path=tsp_path,
                coords=coords,
                dist=dist,
                edge_weight_type=edge_weight_type,
            )
        )

    rows.sort(key=lambda x: (x.n, x.name))
    if max_instances and max_instances > 0:
        rows = rows[: int(max_instances)]
    if not rows:
        raise ValueError("No TSP instances selected")
    return rows


def discover_heuristics(
    selected_root: Path,
    signal_mode: str,
    max_heuristics: int,
    heuristic_ids: str | None = None,
) -> list[HeuristicSpec]:
    if not selected_root.exists():
        raise FileNotFoundError(f"Missing selected TSP root: {selected_root}")

    wanted_ids = parse_csv_filter(heuristic_ids, cast=str)
    wanted_set = set(wanted_ids) if wanted_ids is not None else None

    all_modes = ["distance_only", "candidate_list", "edge_prior", "edge_prior_plus_candidate_list"]
    if signal_mode.lower() == "all":
        modes = all_modes
    else:
        modes = [signal_mode]
    specs: list[HeuristicSpec] = []
    for mode in modes:
        mode_dir = selected_root / mode
        if not mode_dir.exists():
            raise FileNotFoundError(f"Missing signal category folder: {mode_dir}")
        for hdir in sorted([p for p in mode_dir.iterdir() if p.is_dir()]):
            if wanted_set is not None and hdir.name not in wanted_set:
                continue
            code = hdir / "heuristic.py"
            if not code.exists():
                py_files = sorted(hdir.glob("*.py"))
                if not py_files:
                    continue
                code = py_files[0]
            specs.append(
                HeuristicSpec(
                    signal_category=mode,
                    heuristic_id=hdir.name,
                    heuristic_label=hdir.name,
                    code_path=code,
                )
            )
    if max_heuristics and max_heuristics > 0:
        specs = specs[: int(max_heuristics)]
    if not specs:
        raise ValueError(
            f"No selected heuristics found for signal_mode={signal_mode!r}, "
            f"heuristic_ids={heuristic_ids!r}, root={selected_root}"
        )
    return specs


def load_candidate_map(inst: TSPInstance, candidate_cache_dir: Path, max_candidates: int) -> tuple[dict[int, list[int]] | None, str]:
    params = PopmusicParams(max_candidates=int(max_candidates))
    cand_path = select_existing_path(candidate_paths_for_instance(inst.name, candidate_cache_dir, params))
    if cand_path is None:
        raise FileNotFoundError(
            f"Missing POPMUSIC candidate file for {inst.name} under {candidate_cache_dir}. "
            f"Expected historical name like {popmusic_candidate_file_name(inst.name, candidate_cache_dir, params).name}"
        )
    parsed = parse_simple_candidate_file(cand_path, n=inst.n)
    return normalize_candidates(parsed, n=inst.n, max_k=max_candidates, dist=inst.dist), str(cand_path)


def load_prior_map(inst: TSPInstance, edge_prior_cache_dir: Path, prior_mode: str, seed: int) -> tuple[dict[tuple[int, int], float] | None, str]:
    params = EdgePriorParams(runs=30, topk=5)
    prior_path = select_existing_path(edge_prior_paths_for_instance(inst.name, edge_prior_cache_dir, params))
    if prior_path is None:
        raise FileNotFoundError(
            f"Missing edge-prior npz for {inst.name} under {edge_prior_cache_dir}. "
            f"Expected historical name like {popmusic_edge_prior_file_name(inst.name, edge_prior_cache_dir, params).name}"
        )
    raw_prior, _meta = load_prior_npz(prior_path)
    prior_map = transform_prior(raw_prior, mode=prior_mode, n=inst.n, seed=seed, topk=5)
    return prior_map, str(prior_path)


def make_problem(
    inst: TSPInstance,
    signal_category: str,
    candidate_cache_dir: Path,
    edge_prior_cache_dir: Path,
    max_candidates: int,
    prior_mode: str,
    seed: int,
) -> tuple[SparseTSPProblem, dict[str, Any]]:
    use_candidates = signal_category in {"candidate_list", "edge_prior_plus_candidate_list"}
    use_prior = signal_category in {"edge_prior", "edge_prior_plus_candidate_list"}

    candidate_map = None
    prior_map = None
    candidate_source = "none"
    prior_source = "none"

    if use_candidates:
        candidate_map, candidate_source = load_candidate_map(inst, candidate_cache_dir, max_candidates)

    if use_prior:
        prior_map, prior_source = load_prior_map(inst, edge_prior_cache_dir, prior_mode, seed)

    problem = SparseTSPProblem(
        coords=inst.coords,
        dist=inst.dist,
        candidate_neighbors=candidate_map,
        prior_map=prior_map,
        edge_weight_type=inst.edge_weight_type,
    )
    info = {
        "use_candidates": use_candidates,
        "use_prior": use_prior,
        "candidate_source": candidate_source,
        "edge_prior_source": prior_source,
    }
    return problem, info


def module_from_path(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def prior_rows_from_problem(problem: SparseTSPProblem) -> list[dict[int, float]]:
    rows = [dict() for _ in range(problem.n)]
    if problem.prior_map is None:
        return rows
    for (a, b), w in problem.prior_map.items():
        rows[int(a)][int(b)] = float(w)
        rows[int(b)][int(a)] = float(w)
    return rows


def call_heuristic_from_module(mod, problem: SparseTSPProblem, rng: np.random.Generator, timeout_s: float, seed: int):
    # Make old notebook-style globals available.
    mod.edge_cost = problem.edge_cost
    mod.full_edge_cost = problem.full_edge_cost
    mod.coords = problem.coords
    mod.D = problem.distance_matrix_for_evaluator()
    mod.dist = problem.distance_matrix_for_evaluator()
    try:
        mod.cand = problem.cand
    except Exception:
        mod.cand = None
    mod.prior_rows = prior_rows_from_problem(problem)
    mod._START_TIME = time.time()
    mod.TIME_LIMIT_S = float(timeout_s or 0)

    with time_limit(timeout_s):
        if hasattr(mod, "TSPHeuristic"):
            algo = mod.TSPHeuristic()
            return algo(problem, rng=rng)

        if hasattr(mod, "construct_tour"):
            fn = mod.construct_tour
            sig = inspect.signature(fn)
            n_params = len(sig.parameters)

            # Common old notebook signatures:
            # construct_tour(problem)
            # construct_tour(D, start_node)
            # construct_tour(D, prior_rows, start_node)
            if n_params == 1:
                return fn(problem)
            if n_params == 2:
                return fn(problem.distance_matrix_for_evaluator(), int(seed % problem.n))
            if n_params >= 3:
                return fn(problem.distance_matrix_for_evaluator(), prior_rows_from_problem(problem), int(seed % problem.n))

        raise ValueError("Heuristic code must define class TSPHeuristic or function construct_tour")


def normalize_tour(raw_tour: Any, n: int) -> np.ndarray:
    arr = np.asarray(raw_tour, dtype=np.int64).reshape(-1)
    validate_tour(arr, n)
    return arr


def run_one(
    h: HeuristicSpec,
    inst: TSPInstance,
    problem: SparseTSPProblem,
    rep: int,
    seed: int,
    timeout_s: float,
) -> dict[str, Any]:
    start = time.perf_counter()
    stdout_tail = ""
    try:
        mod_name = f"tsp_selected_{abs(hash((h.signal_category, h.heuristic_id, rep, inst.name))) % 10**12}"
        mod = module_from_path(h.code_path, mod_name)
        capture = io.StringIO()
        with contextlib.redirect_stdout(capture):
            raw = call_heuristic_from_module(mod, problem, np.random.default_rng(seed), timeout_s, seed)
        out = capture.getvalue()
        stdout_tail = out[-1000:] if out else ""

        tour = normalize_tour(raw, problem.n)
        objective_value = tour_cost_from_matrix(tour, problem.distance_matrix_for_evaluator())
        gap = 100.0 * (objective_value - inst.optimum) / inst.optimum if inst.optimum > 0 else np.nan
        candidate_edge_count, total_edges = problem.tour_candidate_edge_count(tour)
        uses_only_candidates = None
        candidate_edge_share = None
        if candidate_edge_count is not None and total_edges is not None:
            uses_only_candidates = bool(total_edges and candidate_edge_count == total_edges)
            candidate_edge_share = float(candidate_edge_count) / float(total_edges) if total_edges else None

        return {
            "signal_category": h.signal_category,
            "heuristic_id": h.heuristic_id,
            "heuristic_label": h.heuristic_label,
            "code_path": str(h.code_path.relative_to(REPO_ROOT) if h.code_path.is_relative_to(REPO_ROOT) else h.code_path),
            "instance_name": inst.name,
            "split": inst.split,
            "n": inst.n,
            "rep": rep,
            "seed": seed,
            "objective_value": objective_value,
            "reference_value": inst.optimum,
            "gap_ref_pct": gap,
            "runtime_s": time.perf_counter() - start,
            "status": "ok",
            "error_type": "",
            "error_message": "",
            "uses_only_candidates": uses_only_candidates,
            "candidate_edge_count": candidate_edge_count,
            "total_edges": total_edges,
            "candidate_edge_share": candidate_edge_share,
            "stdout_tail": stdout_tail,
            "hostname": socket.gethostname(),
        }
    except CandidateTimeoutError as e:
        return error_row(h, inst, rep, seed, start, "timeout", type(e).__name__, str(e), stdout_tail)
    except Exception as e:
        return error_row(h, inst, rep, seed, start, "error", type(e).__name__, str(e), stdout_tail, traceback.format_exc(limit=8))


def error_row(h, inst, rep, seed, start, status, error_type, error_message, stdout_tail="", tb=""):
    return {
        "signal_category": h.signal_category,
        "heuristic_id": h.heuristic_id,
        "heuristic_label": h.heuristic_label,
        "code_path": str(h.code_path.relative_to(REPO_ROOT) if h.code_path.is_relative_to(REPO_ROOT) else h.code_path),
        "instance_name": inst.name,
        "split": inst.split,
        "n": inst.n,
        "rep": rep,
        "seed": seed,
        "objective_value": np.nan,
        "reference_value": inst.optimum,
        "gap_ref_pct": np.nan,
        "runtime_s": time.perf_counter() - start,
        "status": status,
        "error_type": error_type,
        "error_message": (error_message or "")[:1000],
        "uses_only_candidates": None,
        "candidate_edge_count": None,
        "total_edges": None,
        "candidate_edge_share": None,
        "stdout_tail": (stdout_tail or "")[-1000:],
        "hostname": socket.gethostname(),
    }


def append_row_csv(path: Path, row: dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=RAW_COLUMNS)
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in RAW_COLUMNS})


def read_existing_keys(path: Path) -> set[tuple[str, str, str, int]]:
    if not path.exists():
        return set()
    try:
        df = pd.read_csv(path, usecols=["signal_category", "heuristic_id", "instance_name", "rep"])
    except Exception:
        return set()
    return set((str(r.signal_category), str(r.heuristic_id), str(r.instance_name), int(r.rep)) for r in df.itertuples())


def q(series: pd.Series, p: float) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return np.nan
    return float(np.percentile(s, p))


def safe_exp(x: float) -> float:
    """Exponentiate a fitted log-intercept without crashing summaries.

    In early/partial runs, log-log fits can be ill-conditioned and produce a
    very large intercept. The exponent coefficient is only a descriptive
    constant; it should never crash the evaluator.
    """
    try:
        if not np.isfinite(x):
            return np.nan
        if x > 700:  # exp(709) is near the largest finite float64.
            return np.inf
        if x < -745:  # underflows to zero in float64.
            return 0.0
        return float(math.exp(float(x)))
    except Exception:
        return np.nan


def write_csv_sorted(rows: list[dict[str, Any]], path: Path, sort_cols: list[str], columns: list[str] | None = None):
    """Write a CSV even when rows is empty or columns are partially missing."""
    df = pd.DataFrame(rows, columns=columns) if columns else pd.DataFrame(rows)
    if not df.empty:
        present = [c for c in sort_cols if c in df.columns]
        if present:
            df = df.sort_values(present, na_position="last")
    df.to_csv(path, index=False)


def summarize(raw_path: Path, out_dir: Path):
    if not raw_path.exists():
        return
    df = pd.read_csv(raw_path)
    if df.empty:
        return

    ok = df[df["status"] == "ok"].copy()
    key_cols = ["signal_category", "heuristic_id", "heuristic_label"]
    rows = []
    for key, g_all in df.groupby(key_cols, dropna=False):
        g_ok = g_all[g_all["status"] == "ok"].copy()
        # per heuristic-instance repetition std
        hi = []
        if len(g_ok):
            for inst_key, gi in g_ok.groupby(["instance_name", "n"], dropna=False):
                hi.append({
                    "instance_name": inst_key[0],
                    "n": inst_key[1],
                    "gap_median_reps": q(gi["gap_ref_pct"], 50),
                    "gap_std_reps": float(pd.to_numeric(gi["gap_ref_pct"], errors="coerce").std(ddof=1)) if gi["gap_ref_pct"].notna().sum() >= 2 else 0.0,
                    "runtime_median_reps_s": q(gi["runtime_s"], 50),
                    "runtime_std_reps_s": float(pd.to_numeric(gi["runtime_s"], errors="coerce").std(ddof=1)) if gi["runtime_s"].notna().sum() >= 2 else 0.0,
                })
        hi_df = pd.DataFrame(hi)
        row = {
            "signal_category": key[0],
            "heuristic_id": key[1],
            "heuristic_label": key[2],
            "total_runs": int(len(g_all)),
            "ok_runs": int((g_all["status"] == "ok").sum()),
            "error_runs": int((g_all["status"] == "error").sum()),
            "timeout_runs": int((g_all["status"] == "timeout").sum()),
            "success_rate": float((g_all["status"] == "ok").mean()) if len(g_all) else np.nan,
            "gap_p01": q(g_ok["gap_ref_pct"], 1),
            "gap_p02": q(g_ok["gap_ref_pct"], 2),
            "gap_p05": q(g_ok["gap_ref_pct"], 5),
            "gap_p10": q(g_ok["gap_ref_pct"], 10),
            "gap_median": q(g_ok["gap_ref_pct"], 50),
            "gap_p75": q(g_ok["gap_ref_pct"], 75),
            "gap_p90": q(g_ok["gap_ref_pct"], 90),
            "gap_std_global": float(pd.to_numeric(g_ok["gap_ref_pct"], errors="coerce").std(ddof=1)) if len(g_ok) >= 2 else np.nan,
            "runtime_median_s": q(g_ok["runtime_s"], 50),
            "runtime_p75_s": q(g_ok["runtime_s"], 75),
            "runtime_p90_s": q(g_ok["runtime_s"], 90),
            "runtime_std_global_s": float(pd.to_numeric(g_ok["runtime_s"], errors="coerce").std(ddof=1)) if len(g_ok) >= 2 else np.nan,
        }
        if len(hi_df):
            row.update({
                "gap_rep_std_mean": float(hi_df["gap_std_reps"].mean()),
                "gap_rep_std_median": float(hi_df["gap_std_reps"].median()),
                "gap_rep_std_p90": q(hi_df["gap_std_reps"], 90),
                "gap_instance_median_mean": float(hi_df["gap_median_reps"].mean()),
                "gap_instance_median_std": float(hi_df["gap_median_reps"].std(ddof=1)) if len(hi_df) >= 2 else 0.0,
                "gap_instance_median_p10": q(hi_df["gap_median_reps"], 10),
                "gap_instance_median_p90": q(hi_df["gap_median_reps"], 90),
                "gap_instance_median_iqr": q(hi_df["gap_median_reps"], 75) - q(hi_df["gap_median_reps"], 25),
                "runtime_rep_std_mean_s": float(hi_df["runtime_std_reps_s"].mean()),
                "runtime_rep_std_median_s": float(hi_df["runtime_std_reps_s"].median()),
                "runtime_rep_std_p90_s": q(hi_df["runtime_std_reps_s"], 90),
                "runtime_instance_median_std_s": float(hi_df["runtime_median_reps_s"].std(ddof=1)) if len(hi_df) >= 2 else 0.0,
            })
        rows.append(row)
    pd.DataFrame(rows).sort_values(["signal_category", "gap_median"], na_position="last").to_csv(out_dir / "summary_by_heuristic.csv", index=False)

    # detailed heuristic-instance summary
    hi_rows = []
    for keys, g_all in df.groupby(["signal_category", "heuristic_id", "heuristic_label", "instance_name", "split", "n"], dropna=False):
        g_ok = g_all[g_all["status"] == "ok"].copy()
        hi_rows.append({
            "signal_category": keys[0],
            "heuristic_id": keys[1],
            "heuristic_label": keys[2],
            "instance_name": keys[3],
            "split": keys[4],
            "n": keys[5],
            "total_runs": int(len(g_all)),
            "ok_runs": int((g_all["status"] == "ok").sum()),
            "success_rate": float((g_all["status"] == "ok").mean()) if len(g_all) else np.nan,
            "gap_median_reps": q(g_ok["gap_ref_pct"], 50),
            "gap_p10_reps": q(g_ok["gap_ref_pct"], 10),
            "gap_std_reps": float(pd.to_numeric(g_ok["gap_ref_pct"], errors="coerce").std(ddof=1)) if len(g_ok) >= 2 else 0.0,
            "runtime_median_reps_s": q(g_ok["runtime_s"], 50),
            "runtime_std_reps_s": float(pd.to_numeric(g_ok["runtime_s"], errors="coerce").std(ddof=1)) if len(g_ok) >= 2 else 0.0,
        })
    pd.DataFrame(hi_rows).to_csv(out_dir / "summary_by_heuristic_instance.csv", index=False)

    # by size
    size_rows = []
    for keys, g_all in df.groupby(["signal_category", "n"], dropna=False):
        g_ok = g_all[g_all["status"] == "ok"].copy()
        size_rows.append({
            "signal_category": keys[0],
            "n": keys[1],
            "total_runs": int(len(g_all)),
            "ok_runs": int((g_all["status"] == "ok").sum()),
            "success_rate": float((g_all["status"] == "ok").mean()) if len(g_all) else np.nan,
            "gap_median": q(g_ok["gap_ref_pct"], 50),
            "gap_p10": q(g_ok["gap_ref_pct"], 10),
            "gap_p90": q(g_ok["gap_ref_pct"], 90),
            "gap_std_global_size": float(pd.to_numeric(g_ok["gap_ref_pct"], errors="coerce").std(ddof=1)) if len(g_ok) >= 2 else np.nan,
            "runtime_median_s": q(g_ok["runtime_s"], 50),
            "runtime_p90_s": q(g_ok["runtime_s"], 90),
            "runtime_std_global_size_s": float(pd.to_numeric(g_ok["runtime_s"], errors="coerce").std(ddof=1)) if len(g_ok) >= 2 else np.nan,
        })
    pd.DataFrame(size_rows).sort_values(["signal_category", "n"]).to_csv(out_dir / "summary_by_instance_size.csv", index=False)

    # complexity fit per heuristic: median runtime by n, log-log slope.
    comp = []
    if len(ok):
        for keys, gh in ok.groupby(key_cols, dropna=False):
            byn = gh.groupby("n")["runtime_s"].median().reset_index()
            byn = byn[(byn["n"] > 0) & (byn["runtime_s"] > 0)]
            if len(byn) >= 2:
                x = np.log(byn["n"].astype(float).to_numpy())
                y = np.log(byn["runtime_s"].astype(float).to_numpy())
                beta, loga = np.polyfit(x, y, 1)
                yhat = beta * x + loga
                ss_res = float(np.sum((y - yhat) ** 2))
                ss_tot = float(np.sum((y - np.mean(y)) ** 2))
                r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
                speed_class = "fast_le_n1p5" if beta <= 1.5 else ("medium_le_n2" if beta <= 2.0 else "slow_gt_n2")
                comp.append({
                    "signal_category": keys[0],
                    "heuristic_id": keys[1],
                    "heuristic_label": keys[2],
                    "beta": float(beta) if np.isfinite(beta) else np.nan,
                    "loga": float(loga) if np.isfinite(loga) else np.nan,
                    "a": safe_exp(float(loga)),
                    "r2_loglog": r2,
                    "n_points": int(len(byn)),
                    "min_n": int(byn["n"].min()),
                    "max_n": int(byn["n"].max()),
                    "speed_class": speed_class,
                })
            else:
                comp.append({
                    "signal_category": keys[0],
                    "heuristic_id": keys[1],
                    "heuristic_label": keys[2],
                    "beta": np.nan,
                    "loga": np.nan,
                    "a": np.nan,
                    "r2_loglog": np.nan,
                    "n_points": int(len(byn)),
                    "min_n": int(byn["n"].min()) if len(byn) else np.nan,
                    "max_n": int(byn["n"].max()) if len(byn) else np.nan,
                    "speed_class": "insufficient_sizes",
                })
    comp_cols = [
        "signal_category", "heuristic_id", "heuristic_label", "beta", "loga", "a",
        "r2_loglog", "n_points", "min_n", "max_n", "speed_class"
    ]
    write_csv_sorted(comp, out_dir / "complexity_fit.csv", ["signal_category", "beta"], columns=comp_cols)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--signal-mode", default="all", choices=["all", "distance_only", "candidate_list", "edge_prior", "edge_prior_plus_candidate_list"])
    ap.add_argument("--selected-root", default="experiments/selected_tsp_heuristics_final_by_signal")
    ap.add_argument("--heuristic-ids", default="ALL", help="Comma-separated heuristic folder names to evaluate, or ALL")
    ap.add_argument("--instance-root", required=True)
    ap.add_argument("--candidate-cache-dir", default="")
    ap.add_argument("--edge-prior-cache-dir", default="")
    ap.add_argument("--optima-csv", default="data/tsp_instances_opt.csv")
    ap.add_argument("--instances", default="ALL")
    ap.add_argument("--splits", default="all")
    ap.add_argument("--repetitions", type=int, default=2)
    ap.add_argument("--max-heuristics", type=int, default=1000)
    ap.add_argument("--max-instances", type=int, default=1000)
    ap.add_argument("--timeout-s", type=float, default=300.0)
    ap.add_argument("--dense-distance-threshold", type=int, default=20000, help="Use lazy on-the-fly TSPLIB distances above this n to avoid full n x n matrices.")
    ap.add_argument("--global-seed", type=int, default=12345)
    ap.add_argument("--max-candidates", type=int, default=20)
    ap.add_argument("--prior-mode", default="frequency")
    ap.add_argument("--output-root", default="/tmp/tsp_eval_results")
    ap.add_argument("--output-dir", default="")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--allow-interface-mismatch", action="store_true", help="Accepted for compatibility with older launchers; selected folders are still evaluated as discovered.")
    args = ap.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else Path(args.output_root) / f"tsp_{args.signal_mode}_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "raw_results.csv"

    config = vars(args).copy()
    config.update({
        "output_dir": str(out_dir),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "loop_order": "rep_outer_then_heuristic_then_instance",
    })
    (out_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    selected_root = (REPO_ROOT / args.selected_root).resolve()
    optima_csv = (REPO_ROOT / args.optima_csv).resolve() if not Path(args.optima_csv).is_absolute() else Path(args.optima_csv)
    instance_root = Path(args.instance_root).expanduser().resolve()
    candidate_cache_dir = Path(args.candidate_cache_dir).expanduser().resolve() if args.candidate_cache_dir else Path("__missing_candidate_cache__")
    edge_prior_cache_dir = Path(args.edge_prior_cache_dir).expanduser().resolve() if args.edge_prior_cache_dir else Path("__missing_edge_prior_cache__")

    heuristics = discover_heuristics(selected_root, args.signal_mode, args.max_heuristics, args.heuristic_ids)
    instances = load_instances(
        instance_root=instance_root,
        optima_csv=optima_csv,
        instances_filter=args.instances,
        split_filter=args.splits,
        max_instances=args.max_instances,
        dense_distance_threshold=args.dense_distance_threshold,
    )

    print(f"Signal mode: {args.signal_mode}")
    print(f"Heuristics: {len(heuristics)}")
    for h in heuristics:
        print(f"  - {h.signal_category}/{h.heuristic_id} ({h.code_path.relative_to(REPO_ROOT)})")
    print(f"Instances: {len(instances)}")
    for inst in instances:
        print(f"  - {inst.name} split={inst.split} n={inst.n} opt={inst.optimum:g}")
    print(f"Repetitions: {args.repetitions}")
    print(f"Output: {out_dir}")

    existing = read_existing_keys(raw_path) if args.resume else set()
    if existing:
        print(f"Resume enabled: loaded {len(existing)} existing rows from {raw_path}")

    total = len(heuristics) * len(instances) * int(args.repetitions)
    counter = 0
    problem_cache: dict[tuple[str, str], tuple[SparseTSPProblem, dict[str, Any]]] = {}

    for rep in range(1, int(args.repetitions) + 1):
        print(f"\n=== repetition {rep}/{args.repetitions} ===", flush=True)
        for h in heuristics:
            for inst in instances:
                counter += 1
                key = (h.signal_category, h.heuristic_id, inst.name, rep)
                if key in existing:
                    if counter % 50 == 0:
                        print(f"[{counter}/{total}] skip existing {h.signal_category}/{h.heuristic_id} {inst.name} rep={rep}", flush=True)
                    continue

                seed = int(np.random.default_rng(args.global_seed + rep * 1000003 + abs(hash((h.signal_category, h.heuristic_id, inst.name))) % 100000).integers(0, 2**32 - 1))
                try:
                    pkey = (h.signal_category, inst.name)
                    if pkey not in problem_cache:
                        problem_cache[pkey] = make_problem(
                            inst,
                            h.signal_category,
                            candidate_cache_dir,
                            edge_prior_cache_dir,
                            args.max_candidates,
                            args.prior_mode,
                            seed,
                        )
                    problem, _info = problem_cache[pkey]
                    row = run_one(h, inst, problem, rep, seed, args.timeout_s)
                except Exception as e:
                    row = error_row(h, inst, rep, seed, time.perf_counter(), "error", type(e).__name__, str(e), "", traceback.format_exc(limit=8))

                append_row_csv(raw_path, row)
                status = row["status"]
                if status == "ok":
                    print(
                        f"[{counter}/{total}] rep={rep} {h.signal_category}/{h.heuristic_id} {inst.name}: "
                        f"gap={row['gap_ref_pct']:.3f}% time={row['runtime_s']:.3f}s status=ok",
                        flush=True,
                    )
                else:
                    print(
                        f"[{counter}/{total}] rep={rep} {h.signal_category}/{h.heuristic_id} {inst.name}: "
                        f"{status.upper()} {row['error_type']}: {row['error_message']}",
                        flush=True,
                    )

                summarize(raw_path, out_dir)

    summarize(raw_path, out_dir)
    print("\nDone.")
    print("Wrote:")
    for name in ["raw_results.csv", "summary_by_heuristic.csv", "summary_by_heuristic_instance.csv", "summary_by_instance_size.csv", "complexity_fit.csv", "run_config.json"]:
        print(f" - {out_dir / name}")
    print(f"LATEST_RESULT_DIR={out_dir}")


if __name__ == "__main__":
    main()
