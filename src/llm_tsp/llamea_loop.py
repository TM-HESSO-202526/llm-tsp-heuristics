from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Callable, Any
import hashlib
import json
import re
import shutil
from pathlib import Path

import pandas as pd

from .parsing import extract_first_python_block
from .prompts import (
    SYSTEM_PROMPT,
    build_tsp_prompt,
    normalized_selection_strategy,
    historical_family_avoidance_block,
    family_focus_block,
)
from .evaluation import evaluate_code_on_problem, EvaluationResult
from .sparse_problem import SparseTSPProblem


@dataclass
class CandidateRecord:
    attempt: int
    iteration: int
    objective_mode: str
    center_constraint: str
    name: str
    family: str
    code_hash: str
    valid: bool
    full_valid: bool
    valid_instances: int
    total_instances: int
    mean_gap_percent: float | None
    selection_score: float | None
    mean_runtime_s: float | None
    parent_attempt: int | None
    selection_strategy: str
    prompt_mode: str
    error: str | None
    code_path: str | None
    raw_response_path: str | None
    prompt_path: str | None
    family_desc: str | None = None
    feedback_by_instance: str | None = None
    search_cost_mean: float | None = None
    search_gap_ref_mean_pct: float | None = None
    partial_valid_cases: int | None = None
    partial_total_cases: int | None = None
    partial_failed_cases: int | None = None
    family_focus_active: bool = False
    focus_family_id: str | None = None
    focus_family_name: str | None = None
    focus_family_step: int | None = None
    focus_family_index: int | None = None
    focus_family_total: int | None = None
    family_focus_compliant: bool | None = None
    family_focus_compliance_level: str | None = None
    family_focus_mismatch_reason: str | None = None


def stable_hash(text: str, n: int = 16) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:n]


def _is_timeout_error(error: str | None) -> bool:
    if not error:
        return False
    e = error.lower()
    return "timeout" in e or "timed out" in e or "time limit" in e


def _jsonable(value: Any) -> Any:
    """Convert numpy/pandas scalar-ish values to normal JSON values."""
    if value is None:
        return None
    try:
        import numpy as np

        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (np.bool_,)):
            return bool(value)
    except Exception:
        pass
    if isinstance(value, Path):
        return str(value)
    return value


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def summarize_results(
    results: list[EvaluationResult],
    partial_failure_penalty: float = 200.0,
) -> tuple[bool, int, int, float | None, float | None, float | None, float | None, str | None]:
    total = len(results)
    valid_results = [r for r in results if r.valid]
    valid_count = len(valid_results)
    full_valid = valid_count == total and total > 0

    gaps = [float(r.gap_percent) for r in valid_results if r.gap_percent is not None]
    costs = [float(r.cost) for r in valid_results if r.cost is not None]
    runtimes = [float(r.runtime_s) for r in valid_results if r.runtime_s is not None]
    mean_gap = sum(gaps) / len(gaps) if gaps else None
    mean_cost = sum(costs) / len(costs) if costs else None
    mean_runtime = sum(runtimes) / len(runtimes) if runtimes else None

    if full_valid and mean_gap is not None:
        score = mean_gap
        err = None
    elif valid_count > 0 and mean_gap is not None:
        # Partial validity is useful signal but must lose to any fully valid heuristic.
        invalid_count = total - valid_count
        score = mean_gap + partial_failure_penalty * invalid_count
        err = f"partial_validity: {valid_count}/{total} valid"
    else:
        score = None
        err = next((r.error for r in results if r.error), "all invalid")

    return full_valid, valid_count, total, mean_gap, mean_cost, mean_runtime, score, err


def _record_line(record: CandidateRecord) -> str:
    status = "valid" if bool(record.full_valid) else "invalid/partial"
    gap = record.mean_gap_percent
    score = record.selection_score
    family = str(record.family or "").strip()
    family_part = f" | family={family}" if family else ""
    focus_part = ""
    if record.family_focus_active:
        comp = record.family_focus_compliance_level or "unknown"
        reason = str(record.family_focus_mismatch_reason or "").replace("\n", " ")[:180]
        focus_part = f" | focus_family={record.focus_family_id} | focus_compliance={comp}"
        if reason:
            focus_part += f" | focus_feedback={reason}"
    err = str(record.error or "")[:200].replace("\n", " ")
    return (
        f"iter={record.iteration} | {record.name} | {status}{family_part}{focus_part} | "
        f"search_gap={gap} | selection_score={score} | error={err}"
    )


def build_history_text(records: list[CandidateRecord], limit: int = 20) -> str | None:
    if not records or limit <= 0:
        return None
    recent = records[-limit:]
    return "\n".join(_record_line(r) for r in recent)


def select_parent(records: list[CandidateRecord], strategy: str) -> CandidateRecord | None:
    if not records:
        return None
    strategy = normalized_selection_strategy(strategy)
    if strategy == "1,1":
        # Sequential mutation chain: always mutate the most recent generated candidate,
        # whether or not it is the best-so-far.
        return records[-1]
    if strategy == "1+1":
        # Elitist 1+1: mutate the best full-valid candidate so far; if none exists,
        # fall back to the best partial candidate by penalized score, then latest candidate.
        full_valid = [r for r in records if r.full_valid and r.selection_score is not None]
        if full_valid:
            return min(full_valid, key=lambda r: (float(r.selection_score), r.attempt))
        partial = [r for r in records if r.valid_instances > 0 and r.selection_score is not None]
        if partial:
            return min(partial, key=lambda r: (float(r.selection_score), r.attempt))
        return records[-1]
    raise ValueError(f"Unknown selection_strategy {strategy!r}; expected '1+1' or '1,1'.")


def parent_selection_reason(records: list[CandidateRecord], strategy: str, parent: CandidateRecord | None) -> str:
    if parent is None:
        return "no_parent_initial_generation"
    strategy = normalized_selection_strategy(strategy)
    if strategy == "1,1":
        return "latest_candidate_1comma1"
    full_valid = [r for r in records if r.full_valid and r.selection_score is not None]
    if full_valid and parent.attempt == min(full_valid, key=lambda r: (float(r.selection_score), r.attempt)).attempt:
        return "best_full_valid_1plus1"
    partial = [r for r in records if r.valid_instances > 0 and r.selection_score is not None]
    if partial and parent.attempt == min(partial, key=lambda r: (float(r.selection_score), r.attempt)).attempt:
        return "best_partial_penalized_1plus1"
    return "latest_no_valid_or_partial_1plus1"


def _family_focus_enabled(search_cfg: dict) -> bool:
    return bool(search_cfg.get("family_focus_mode", False))


def _family_focus_plan(search_cfg: dict) -> list[dict[str, Any]]:
    plan = search_cfg.get("family_focus_plan") or []
    if not isinstance(plan, list):
        raise ValueError("search.family_focus_plan must be a list of family dictionaries.")
    out: list[dict[str, Any]] = []
    for i, item in enumerate(plan):
        if not isinstance(item, dict):
            raise ValueError(f"search.family_focus_plan[{i}] must be a dict, got {type(item).__name__}.")
        if not bool(item.get("enabled", True)):
            continue
        fid = str(item.get("id") or item.get("name") or f"family_{i+1}").strip()
        if not fid:
            fid = f"family_{i+1}"
        normalized = dict(item)
        normalized.setdefault("id", fid)
        normalized.setdefault("name", fid)
        out.append(normalized)
    return out


def _family_focus_for_attempt(search_cfg: dict, attempt: int) -> tuple[dict[str, Any] | None, int | None, int | None, int | None, int | None]:
    """Return family spec and block metadata for one global attempt.

    Returns: (family_spec, family_step, calls_per_family, family_index, total_families).
    """
    if not _family_focus_enabled(search_cfg):
        return None, None, None, None, None
    plan = _family_focus_plan(search_cfg)
    if not plan:
        raise ValueError("FAMILY_FOCUS_MODE is active but FAMILY_FOCUS_PLAN is empty or all entries are disabled.")
    calls_per_family = max(1, int(search_cfg.get("family_focus_calls_per_family", 20)))
    family_index = min((int(attempt) - 1) // calls_per_family, len(plan) - 1)
    family_step = (int(attempt) - 1) - family_index * calls_per_family + 1
    return plan[family_index], family_step, calls_per_family, family_index, len(plan)


def _records_for_focus(records: list[CandidateRecord], focus_family_id: str | None) -> list[CandidateRecord]:
    if not focus_family_id:
        return records
    return [r for r in records if r.focus_family_id == focus_family_id]



def _contains_any(text: str, needles: list[str]) -> bool:
    return any(n in text for n in needles)


def infer_family_focus_compliance(
    focus_spec: dict[str, Any] | None,
    inferred_family: str,
    algo_name: str,
    code: str,
) -> tuple[bool | None, str, str]:
    """Heuristic, transparent compliance diagnostic for family-focus mode.

    This is intentionally a logging/feedback signal, not a hard rejection gate.
    The generic inferred family can say ``nearest_neighbor_2opt`` even when a real
    grid/Voronoi scaffold exists and uses nearest-neighbor only as a local
    subroutine. Therefore we record three levels:
    - compliant: the locked-family structure appears to be the main mechanism.
    - partial: the locked-family structure appears, but the code also leans on
      nearest-neighbor/insertion/2-opt style components.
    - non_compliant: the code mostly escaped to a banned generic family or did
      not implement the locked-family structure.
    """
    if not focus_spec:
        return None, "not_applicable", ""

    fid = str(focus_spec.get("id") or "").strip().lower()
    name = str(focus_spec.get("name") or fid)
    text = f"{algo_name or ''}\n{code or ''}".lower().replace("-", "_")
    family = str(inferred_family or "other")
    fallback_family = family in {
        "nearest_neighbor",
        "nearest_neighbor_2opt",
        "candidate_2opt",
        "candidate_constructive",
        "candidate_prior_2opt",
        "candidate_prior_constructive",
        "two_opt_improvement",
        "regret_insertion",
    }
    nn_like = _contains_any(text, ["nearest", "closest", "closest_unvisited", "min_distance"])
    opt_like = _contains_any(text, ["2_opt", "two_opt", "segment", "reverse", "reversal"])

    def result(level: str, reason: str) -> tuple[bool, str, str]:
        return level == "compliant", level, reason

    if fid == "mst_skeleton":
        has_skeleton = _contains_any(text, ["mst", "spanning", "tree", "forest", "parent", "adjacency", "preorder", "dfs", "shortcut"])
        has_real_tree = _contains_any(text, ["adjacency", "parent", "preorder", "dfs", "stack", "forest", "tree_edges", "mst_edges"])
        if fallback_family and nn_like and not has_real_tree:
            return result("non_compliant", "Nearest-neighbor-like code generated inside MST skeleton block; no clear tree/forest skeleton is used as the main object.")
        if has_skeleton and has_real_tree and not fallback_family:
            return result("compliant", "Tree/forest skeleton terms and traversal/adjacency logic appear central to the code.")
        if has_skeleton:
            return result("partial", "MST/tree vocabulary appears, but the implementation also looks nearest-neighbor or cleanup dominated.")
        return result("non_compliant", "No clear MST/tree/forest skeleton mechanism detected.")

    if fid == "voronoi_regions":
        has_regions = _contains_any(text, ["voronoi", "region", "regions", "seed", "seeds", "partition", "cell", "centroid"])
        has_bridge = _contains_any(text, ["bridge", "endpoint", "connect", "merge", "region_order", "order_regions"])
        if has_regions and has_bridge and not fallback_family:
            return result("compliant", "Voronoi/region decomposition and region-bridging logic appear central.")
        if has_regions:
            return result("partial", "Region/Voronoi structure appears, but the code also leans on nearest-neighbor, insertion, or 2-opt-style logic.")
        return result("non_compliant", "No clear Voronoi/region decomposition detected; likely escaped the locked family.")

    if fid == "convex_hull_outside_in":
        has_hull = _contains_any(text, ["convex", "hull", "graham", "monotonic", "boundary", "outer"])
        has_insert = _contains_any(text, ["insert", "insertion", "delta", "edge impact", "distance_to_edge", "tour edge"])
        if has_hull and has_insert:
            return result("compliant", "Convex-hull/outside-in insertion structure appears central; insertion_constructive is expected for this family.")
        if has_hull:
            return result("partial", "Convex hull/boundary structure appears, but outside-in insertion is weak or unclear.")
        return result("non_compliant", "No clear convex-hull/outside-in construction detected.")

    if fid == "grid_sector_decomposition":
        has_grid = _contains_any(text, ["grid", "cell", "cells", "sector", "sectors", "snake", "radial", "sweep"])
        has_macro_order = _contains_any(text, ["snake", "radial", "sector", "order_cells", "cell_order", "sort"])
        if has_grid and has_macro_order and not fallback_family:
            return result("compliant", "Grid/sector macro-order appears central to the construction.")
        if has_grid:
            return result("partial", "Grid/sector structure appears, but nearest-neighbor, insertion, or 2-opt-style logic still seems important.")
        return result("non_compliant", "No clear grid/sector decomposition detected.")

    if fid == "sparse_geometric_graph":
        has_graph = _contains_any(text, ["graph", "adjacency", "gabriel", "delaunay", "edge list", "fragment", "component", "bounded_degree"])
        if has_graph and not fallback_family:
            return result("compliant", "Sparse geometric graph / fragment structure appears central.")
        if has_graph:
            return result("partial", "Sparse graph vocabulary appears, but the code may still be nearest-neighbor or repair dominated.")
        return result("non_compliant", "No clear sparse geometric graph mechanism detected.")

    if fid == "clustering_decomposition":
        has_cluster = _contains_any(text, ["cluster", "clusters", "centroid", "center", "assignment", "subproblem"])
        has_bridge = _contains_any(text, ["bridge", "endpoint", "connect", "merge", "cluster_order", "order_clusters"])
        if has_cluster and has_bridge and not fallback_family:
            return result("compliant", "Cluster decomposition and bridge logic appear central.")
        if has_cluster:
            return result("partial", "Cluster decomposition appears, but local nearest-neighbor/cleanup may dominate.")
        return result("non_compliant", "No clear cluster-decomposition mechanism detected.")

    if fid == "spectral_clustering_proxy":
        has_partition = _contains_any(text, ["spectral", "projection", "project", "bisect", "bisection", "cut", "partition", "split"])
        if has_partition and not fallback_family:
            return result("compliant", "Spectral/graph-cut proxy partitioning appears central.")
        if has_partition:
            return result("partial", "Spectral/proxy partitioning appears, but local nearest-neighbor/cleanup may dominate.")
        return result("non_compliant", "No clear spectral/proxy partitioning mechanism detected.")

    if fid == "polar_angle_sweep":
        has_sweep = _contains_any(text, ["angle", "polar", "atan2", "radius", "radial", "sweep", "sector"])
        if has_sweep and not fallback_family:
            return result("compliant", "Polar/radial sweep ordering appears central.")
        if has_sweep:
            return result("partial", "Sweep terms appear, but the code may still select nearest/closest cities or rely on cleanup.")
        return result("non_compliant", "No clear polar-angle/sweep mechanism detected.")

    if fid == "region_growth_endpoint_bridging":
        has_endpoint = _contains_any(text, ["endpoint", "fragment", "path", "bridge", "merge", "degree", "subtour", "region"])
        if has_endpoint and not fallback_family:
            return result("compliant", "Endpoint/fragment bridging appears central.")
        if has_endpoint:
            return result("partial", "Endpoint/fragment terminology appears, but it may reduce to a closest-endpoint/nearest-neighbor walk.")
        return result("non_compliant", "No clear endpoint-bridging or fragment-growth mechanism detected.")

    # Unknown user-added focus family: provide conservative diagnostics.
    if fallback_family:
        return result("partial", f"Generated code was classified as {family}; inspect manually for family fidelity to {name}.")
    return result("compliant", f"No obvious banned generic family detected for focus family {name}.")

def _write_family_focus_summary(records: list[CandidateRecord], artifact_dir: Path) -> None:
    if not records or not any(r.family_focus_active for r in records):
        return
    df = pd.DataFrame([asdict(r) for r in records])
    if df.empty or "focus_family_id" not in df:
        return
    rows: list[dict[str, Any]] = []
    for fid, g in df.groupby("focus_family_id", dropna=False):
        full = g[g["full_valid"].astype(bool) & g["selection_score"].notna()].copy()
        best = full.sort_values(["selection_score", "attempt"]).iloc[0].to_dict() if len(full) else {}
        compliant_full = full[full.get("family_focus_compliance_level", "") == "compliant"] if "family_focus_compliance_level" in full else pd.DataFrame()
        partial_full = full[full.get("family_focus_compliance_level", "") == "partial"] if "family_focus_compliance_level" in full else pd.DataFrame()
        best_compliant = compliant_full.sort_values(["selection_score", "attempt"]).iloc[0].to_dict() if len(compliant_full) else {}
        best_partial = partial_full.sort_values(["selection_score", "attempt"]).iloc[0].to_dict() if len(partial_full) else {}
        compliance_counts = g["family_focus_compliance_level"].fillna("unknown").value_counts().to_dict() if "family_focus_compliance_level" in g else {}
        rows.append({
            "focus_family_id": fid,
            "focus_family_name": next((x for x in g["focus_family_name"].dropna().astype(str).tolist() if x), fid),
            "focus_family_index": int(g["focus_family_index"].dropna().iloc[0]) if g["focus_family_index"].notna().any() else None,
            "attempts": int(len(g)),
            "full_valid_attempts": int(len(full)),
            "valid_attempts": int(g["valid"].astype(bool).sum()) if "valid" in g else None,
            "compliant_attempts": int(compliance_counts.get("compliant", 0)),
            "partial_compliance_attempts": int(compliance_counts.get("partial", 0)),
            "non_compliant_attempts": int(compliance_counts.get("non_compliant", 0)),
            "best_attempt": int(best.get("attempt")) if best else None,
            "best_selection_score": float(best.get("selection_score")) if best else None,
            "best_mean_gap_percent": float(best.get("mean_gap_percent")) if best and best.get("mean_gap_percent") is not None else None,
            "best_mean_runtime_s": float(best.get("mean_runtime_s")) if best and best.get("mean_runtime_s") is not None else None,
            "best_name": best.get("name") if best else None,
            "best_inferred_family": best.get("family") if best else None,
            "best_family_focus_compliance_level": best.get("family_focus_compliance_level") if best else None,
            "best_family_focus_compliant": best.get("family_focus_compliant") if best else None,
            "best_family_focus_mismatch_reason": best.get("family_focus_mismatch_reason") if best else None,
            "best_compliant_attempt": int(best_compliant.get("attempt")) if best_compliant else None,
            "best_compliant_mean_gap_percent": float(best_compliant.get("mean_gap_percent")) if best_compliant and best_compliant.get("mean_gap_percent") is not None else None,
            "best_partial_attempt": int(best_partial.get("attempt")) if best_partial else None,
            "best_partial_mean_gap_percent": float(best_partial.get("mean_gap_percent")) if best_partial and best_partial.get("mean_gap_percent") is not None else None,
            "best_code_path": best.get("code_path") if best else None,
        })
    out = pd.DataFrame(rows).sort_values(["focus_family_index", "focus_family_id"], na_position="last")
    _write_csv(out, artifact_dir / "family_focus_summary.csv")
    # Copy the best code per family into a small convenience folder.
    best_dir = artifact_dir / "best_by_focus_family"
    best_dir.mkdir(parents=True, exist_ok=True)
    for _, row in out.iterrows():
        code_path = row.get("best_code_path")
        fid = str(row.get("focus_family_id") or "unknown")
        if code_path and Path(str(code_path)).exists():
            safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", fid)[:80]
            shutil.copy2(str(code_path), best_dir / f"{safe}_best.py")


def _read_code_from_record(record: CandidateRecord | None) -> str | None:
    if not record or not record.code_path:
        return None
    path = Path(record.code_path)
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def parse_field(raw_text: str, field_name: str, default: str = "") -> str:
    pat = rf"^\s*#?\s*{re.escape(field_name)}\s*:\s*(.+?)\s*$"
    m = re.search(pat, str(raw_text), flags=re.IGNORECASE | re.MULTILINE)
    return m.group(1).strip() if m else default


def infer_candidate_name(raw_text: str, code: str, attempt: int) -> str:
    # Match the clustering repo behavior: the LLM must return a # Name field;
    # the backend keeps a deterministic fallback instead of crashing the run.
    name = parse_field(raw_text, "Name", "")
    if name:
        return name[:120]
    if re.search(r"class\s+TSPHeuristic\b", code):
        return "TSPHeuristic"
    return f"UnnamedTSPHeuristic{attempt}"


def infer_family(algo_name: str, code: str) -> str:
    # Family is inferred by the backend, as in the clustering repo.
    s = f"{algo_name or ''}\n{code or ''}".lower().replace("-", "_")
    has_2opt = "2_opt" in s or "two_opt" in s or "two opt" in s
    has_nn = "nearest" in s or "closest" in s
    has_prior = "prior(" in s or "problem.prior" in s
    has_candidates = "neighbors(" in s or "problem.neighbors" in s
    has_insert = "insert" in s or "insertion" in s
    has_savings = "savings" in s or "clarke" in s or "wright" in s
    has_regret = "regret" in s
    has_random = "random" in s or "rng." in s
    has_greedy = "greedy" in s

    if has_prior and has_candidates and has_2opt:
        return "candidate_prior_2opt"
    if has_prior and has_candidates:
        return "candidate_prior_constructive"
    if has_candidates and has_2opt:
        return "candidate_2opt"
    if has_nn and has_2opt:
        return "nearest_neighbor_2opt"
    if has_regret and has_insert:
        return "regret_insertion"
    if has_insert and has_2opt:
        return "insertion_2opt"
    if has_savings:
        return "savings_constructive"
    if has_2opt:
        return "two_opt_improvement"
    if has_nn:
        return "nearest_neighbor"
    if has_insert:
        return "insertion_constructive"
    if has_candidates:
        return "candidate_constructive"
    if has_greedy:
        return "greedy_constructive"
    if has_random:
        return "randomized_constructive"
    return "other"


def family_description(family_sig: str) -> str:
    descriptions = {
        "candidate_prior_2opt": "candidate/prior-guided construction with bounded 2-opt-style improvement",
        "candidate_prior_constructive": "candidate/prior-guided constructive tour builder",
        "candidate_2opt": "candidate-neighborhood construction with bounded 2-opt-style improvement",
        "nearest_neighbor_2opt": "nearest-neighbor construction with bounded 2-opt-style improvement",
        "regret_insertion": "regret/insertion constructive tour builder",
        "insertion_2opt": "insertion construction with bounded 2-opt-style improvement",
        "savings_constructive": "savings-style constructive tour builder",
        "two_opt_improvement": "tour construction dominated by 2-opt-style improvement",
        "nearest_neighbor": "nearest-neighbor constructive tour builder",
        "insertion_constructive": "insertion constructive tour builder",
        "candidate_constructive": "candidate-neighborhood constructive tour builder",
        "greedy_constructive": "generic greedy constructive tour builder",
        "randomized_constructive": "randomized constructive tour builder",
        "llm_call_failure": "LLM call failed before candidate code could be evaluated",
        "other": "other/uncategorized constructive mechanism",
    }
    return descriptions.get(str(family_sig or "other"), str(family_sig or "other"))


def _should_use_redesign_prompt(parent: CandidateRecord | None, records: list[CandidateRecord], search_cfg: dict) -> bool:
    if not parent or not search_cfg.get("invalid_parent_redesign", True):
        return False
    if parent.full_valid:
        return False
    has_full_valid = any(r.full_valid for r in records)
    if (not has_full_valid) and search_cfg.get("redesign_on_any_invalid_before_full_valid", True):
        return True
    if _is_timeout_error(parent.error) and search_cfg.get("redesign_on_timeout_parent", True):
        return True
    return False

def _parent_summary(parent: CandidateRecord | None, records: list[CandidateRecord], strategy: str, prompt_mode: str) -> dict[str, Any]:
    if parent is None:
        return {}
    return {
        "objective_mode": "tsp",
        "center_constraint": "permutation_tour",
        "selection_strategy": normalized_selection_strategy(strategy),
        "selection_strategy_raw": strategy,
        "parent_selection_reason": parent_selection_reason(records, strategy, parent),
        "iteration": int(parent.iteration),
        "name": parent.name,
        "family_sig": parent.family,
        "family_desc": parent.family_desc or family_description(parent.family),
        "family_focus_active": bool(parent.family_focus_active),
        "focus_family_id": parent.focus_family_id,
        "focus_family_name": parent.focus_family_name,
        "family_focus_compliant": parent.family_focus_compliant,
        "family_focus_compliance_level": parent.family_focus_compliance_level,
        "family_focus_mismatch_reason": parent.family_focus_mismatch_reason or "",
        "valid": bool(parent.full_valid),
        "full_valid": bool(parent.full_valid),
        "selection_score": parent.selection_score,
        "search_gap_ref_mean_pct": parent.search_gap_ref_mean_pct,
        "search_gap_ref_mean": parent.mean_gap_percent,
        "search_cost_mean": parent.search_cost_mean,
        "search_gap_opt_mean_pct": parent.mean_gap_percent,
        "search_runtime_mean_s": parent.mean_runtime_s,
        "probe_valid": None,
        "probe_gap_ref_mean_pct": None,
        "valid_instances": int(parent.valid_instances),
        "total_instances": int(parent.total_instances),
        "partial_valid_cases": parent.partial_valid_cases,
        "partial_total_cases": parent.partial_total_cases,
        "partial_failed_cases": parent.partial_failed_cases,
        "feedback_by_instance": parent.feedback_by_instance or "",
        "feedback_by_p": parent.feedback_by_instance or "",
        "probe_feedback_by_p": "",
        "error": str(parent.error or "")[:800],
        "parent_timed_out": bool(_is_timeout_error(parent.error)),
        "full_valid_exists": bool(any(r.full_valid for r in records)),
        "invalid_redesign_mode": bool(prompt_mode == "redesign_invalid_parent"),
    }


def _instance_result_row(
    attempt: int,
    record_seed: dict[str, Any],
    instance_name: str,
    problem: SparseTSPProblem,
    optimum: float | None,
    result: EvaluationResult,
) -> dict[str, Any]:
    return {
        "iteration": attempt,
        "attempt": attempt,
        "objective_mode": "tsp",
        "center_constraint": "permutation_tour",
        "instance": instance_name,
        "n": int(problem.n),
        "optimum": _safe_float(optimum),
        "valid": bool(result.valid),
        "cost": _safe_float(result.cost),
        "gap_percent": _safe_float(result.gap_percent),
        "runtime_s": _safe_float(result.runtime_s),
        "error": result.error,
        "traceback": result.traceback,
        "uses_only_candidates": result.uses_only_candidates,
        "candidate_edge_count": result.candidate_edge_count,
        "total_edges": result.total_edges,
        "candidate_edge_share": _safe_float(result.candidate_edge_share),
        **record_seed,
    }


def _format_instance_feedback(rows: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for row in rows:
        name = row["instance"]
        if row["valid"]:
            gap = row["gap_percent"]
            gap_txt = "NA" if gap is None else f"{float(gap):.3f}%"
            cost = row["cost"]
            cost_txt = "NA" if cost is None else f"{float(cost):.6g}"
            rt = row["runtime_s"]
            rt_txt = "NA" if rt is None else f"{float(rt):.3f}s"
            cand_share = row.get("candidate_edge_share")
            cand_txt = "NA" if cand_share is None else f"{100.0 * float(cand_share):.1f}% candidate-edges"
            lines.append(f"{name}: valid 1/1, gap_vs_opt={gap_txt}, cost={cost_txt}, runtime={rt_txt}, {cand_txt}")
        else:
            err = str(row.get("error") or "invalid")
            tb = str(row.get("traceback") or "")
            if tb:
                tb_short = "\n".join(tb.splitlines()[:4])
                lines.append(f"{name}: valid 0/1, errors={err}\n{tb_short}")
            else:
                lines.append(f"{name}: valid 0/1, errors={err}")
    return "\n".join(lines)


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _write_best_artifacts(records: list[CandidateRecord], artifact_dir: Path) -> None:
    full_valid = [r for r in records if r.full_valid and r.selection_score is not None]
    if not full_valid:
        return
    best = min(full_valid, key=lambda r: (float(r.selection_score), r.attempt))
    best_summary = asdict(best)
    with (artifact_dir / "best_candidate_summary.json").open("w", encoding="utf-8") as f:
        json.dump({k: _jsonable(v) for k, v in best_summary.items()}, f, indent=2, ensure_ascii=False)
    if best.code_path and Path(best.code_path).exists():
        shutil.copy2(best.code_path, artifact_dir / "best_candidate_code.py")


def _write_family_summary(records: list[CandidateRecord], artifact_dir: Path) -> None:
    if not records:
        return
    df = pd.DataFrame([asdict(r) for r in records])
    if df.empty or "family" not in df:
        return
    rows = []
    for family, g in df.groupby("family", dropna=False):
        full = g[g["full_valid"].astype(bool)]
        valid = g[g["valid"].astype(bool)]
        rows.append(
            {
                "family": family,
                "attempts": int(len(g)),
                "valid_attempts": int(len(valid)),
                "full_valid_attempts": int(len(full)),
                "best_selection_score": float(full["selection_score"].min()) if len(full) and full["selection_score"].notna().any() else None,
                "best_gap_percent": float(full["mean_gap_percent"].min()) if len(full) and full["mean_gap_percent"].notna().any() else None,
            }
        )
    _write_csv(pd.DataFrame(rows).sort_values(["best_selection_score", "attempts"], na_position="last"), artifact_dir / "llm_family_summary.csv")


def _write_summary_artifacts(records: list[CandidateRecord], instance_rows: list[dict[str, Any]], artifact_dir: Path, save_generated_attempts: bool = True) -> pd.DataFrame:
    df = pd.DataFrame([asdict(r) for r in records])
    if df.empty:
        return df

    # Clustering-compatible aliases. Keep the concise TSP names too.
    if "name" in df and "algo_name" not in df:
        df["algo_name"] = df["name"]
    if "family" in df and "family_sig" not in df:
        df["family_sig"] = df["family"]
    if "mean_gap_percent" in df and "search_gap_ref_mean" not in df:
        df["search_gap_ref_mean"] = df["mean_gap_percent"]
    if "mean_runtime_s" in df and "search_runtime_mean_s" not in df:
        df["search_runtime_mean_s"] = df["mean_runtime_s"]
    if "valid_instances" in df and "partial_valid_cases" not in df:
        df["partial_valid_cases"] = df["valid_instances"]
    if "total_instances" in df and "partial_total_cases" not in df:
        df["partial_total_cases"] = df["total_instances"]
    if {"valid_instances", "total_instances"}.issubset(df.columns) and "partial_failed_cases" not in df:
        df["partial_failed_cases"] = df["total_instances"] - df["valid_instances"]
    if "family_desc" not in df:
        df["family_desc"] = df.get("family_sig", "other")

    # Clustering-style names.
    _write_csv(df, artifact_dir / "llm_attempts.csv")
    if save_generated_attempts:
        _write_csv(df, artifact_dir / "generated_attempts.csv")

    full = df[df["full_valid"].astype(bool) & df["selection_score"].notna()].copy()
    if not full.empty:
        top = full.sort_values(["selection_score", "attempt"]).head(20)
        _write_csv(top, artifact_dir / "llm_best_attempts_top20.csv")

    if instance_rows:
        inst_df = pd.DataFrame(instance_rows)
        _write_csv(inst_df, artifact_dir / "llm_search_instance_rows.csv")

    _write_family_summary(records, artifact_dir)
    _write_family_focus_summary(records, artifact_dir)
    _write_best_artifacts(records, artifact_dir)
    return df


def run_llamea_search(
    config: dict,
    problems: list[tuple[str, SparseTSPProblem, float | None]],
    llm_call: Callable[[list[dict[str, str]]], str],
    artifact_dir: str | Path,
) -> pd.DataFrame:
    """Run the cleaned TSP LLaMEA loop.

    Search behavior matches the clustering control style:
    - ``SELECTION_STRATEGY = "1+1"``: parent is the best full-valid candidate so far.
      Before a full-valid candidate exists, invalid-parent redesign can use the latest failed
      candidate as diagnostic material.
    - ``SELECTION_STRATEGY = "1,1"``: parent is the latest previous candidate.
    - invalid-parent code exposure follows the clustering controls: invalid/partial
      parent code is shown for diagnosis unless ``hide_invalid_parent_code`` is true
      inside invalid-redesign mode.

    The artifact names intentionally mirror the clustering repo:
    ``codes/iter_*.py``, ``prompts/prompt_iter_*.txt``, ``raw_responses/raw_iter_*.txt``,
    ``llm_attempts.csv``, ``llm_search_instance_rows.csv``, ``search_detail_iter_*.csv``,
    ``llm_best_attempts_top20.csv``, ``llm_family_summary.csv``, plus
    ``best_candidate_code.py`` and ``best_candidate_summary.json``.
    """
    artifact_dir = Path(artifact_dir)
    code_dir = artifact_dir / "codes"
    raw_dir = artifact_dir / "raw_responses"
    prompt_dir = artifact_dir / "prompts"
    code_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    prompt_dir.mkdir(parents=True, exist_ok=True)

    # Keep legacy directory names as aliases for users who already used v4.
    legacy_code_dir = artifact_dir / "generated_code"
    legacy_raw_dir = artifact_dir / "raw_llm_responses"
    legacy_code_dir.mkdir(parents=True, exist_ok=True)
    legacy_raw_dir.mkdir(parents=True, exist_ok=True)

    llm_cfg = config.get("llm", {})
    runtime_cfg = config.get("runtime", {})
    search_cfg = config.get("search", {})
    pop_cfg = config.get("popmusic", {})

    max_calls = int(llm_cfg.get("max_llm_calls", 1))
    family_focus_active = _family_focus_enabled(search_cfg)
    family_focus_plan = _family_focus_plan(search_cfg) if family_focus_active else []
    calls_per_family = max(1, int(search_cfg.get("family_focus_calls_per_family", 20)))
    if family_focus_active:
        if not family_focus_plan:
            raise ValueError("FAMILY_FOCUS_MODE is active but FAMILY_FOCUS_PLAN is empty or all entries are disabled.")
        planned_calls = calls_per_family * len(family_focus_plan)
        if max_calls != planned_calls:
            print(
                f"Family-focus mode active: overriding max_llm_calls={max_calls} with "
                f"{planned_calls} (= {len(family_focus_plan)} families × {calls_per_family} calls).",
                flush=True,
            )
        max_calls = planned_calls
    strategy = normalized_selection_strategy(search_cfg.get("selection_strategy", "1+1"))
    if strategy not in {"1+1", "1,1"}:
        raise ValueError(f"selection_strategy must be '1+1' or '1,1', got {strategy!r}")
    history_limit = int(search_cfg.get("history_limit", 20))
    hide_invalid_code = bool(search_cfg.get("hide_invalid_parent_code", False))
    partial_failure_penalty = float(runtime_cfg.get("partial_failure_penalty", 200.0))

    records: list[CandidateRecord] = []
    instance_rows: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()

    for attempt in range(1, max_calls + 1):
        focus_spec, focus_step, focus_calls, focus_index, focus_total = _family_focus_for_attempt(search_cfg, attempt)
        focus_family_id = str(focus_spec.get("id")) if focus_spec else None
        focus_family_name = str(focus_spec.get("name") or focus_family_id) if focus_spec else None
        local_records = _records_for_focus(records, focus_family_id)

        parent = select_parent(local_records, strategy)
        parent_code = _read_code_from_record(parent)
        prompt_mode = "initial" if parent is None else "mutate_parent"
        parent_is_invalid = bool(parent is not None and not parent.full_valid)
        if _should_use_redesign_prompt(parent, local_records, search_cfg):
            prompt_mode = "redesign_invalid_parent"

        focus_record_fields = {
            "family_focus_active": bool(focus_spec),
            "focus_family_id": focus_family_id,
            "focus_family_name": focus_family_name,
            "focus_family_step": focus_step,
            "focus_family_index": focus_index,
            "focus_family_total": focus_total,
        }

        # Match clustering: selected parent code is shown in normal mutation prompts.
        # In invalid-redesign mode, hide it only when hide_invalid_parent_code=True.
        if prompt_mode == "redesign_invalid_parent" and hide_invalid_code:
            parent_code_for_prompt = None
        else:
            parent_code_for_prompt = parent_code

        parent_summary = _parent_summary(parent, local_records, strategy, prompt_mode)
        if focus_spec:
            parent_summary.update({
                "family_focus_active": True,
                "focus_family_id": focus_family_id,
                "focus_family_name": focus_family_name,
                "focus_family_step": focus_step,
                "focus_family_index": focus_index,
                "focus_family_total": focus_total,
            })
        historical_memory = historical_family_avoidance_block(config)
        focus_memory = family_focus_block(
            config,
            focus_spec,
            family_step=focus_step,
            calls_per_family=focus_calls,
            family_index=focus_index,
            total_families=focus_total,
        )
        prompt = build_tsp_prompt(
            config,
            parent_code=parent_code_for_prompt,
            history_text=build_history_text(local_records, history_limit),
            prompt_mode=prompt_mode,
            parent_is_invalid=parent_is_invalid,
            parent_summary=parent_summary,
            parent_timed_out=bool(_is_timeout_error(parent.error) if parent else False),
            historical_memory=historical_memory,
            family_focus_memory=focus_memory,
        )

        prompt_path = prompt_dir / f"prompt_iter_{attempt:03d}.txt"
        prompt_path.write_text(prompt, encoding="utf-8")
        # Legacy prompt name from v4, so old notebooks still find something.
        (prompt_dir / f"attempt_{attempt:04d}.txt").write_text(prompt, encoding="utf-8")

        print("\n" + "=" * 90, flush=True)
        print(f"[LLM call {attempt}/{max_calls}] objective=tsp constraint=permutation_tour", flush=True)
        if focus_spec:
            print(
                f"  family-focus: {focus_family_id} ({focus_family_name}) "
                f"call {focus_step}/{focus_calls}",
                flush=True,
            )

        try:
            raw = llm_call([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ])
        except Exception as e:
            h = stable_hash(f"llm_call_failed:{attempt}:{e}")
            rec = CandidateRecord(
                attempt=attempt,
                iteration=attempt,
                objective_mode="tsp",
                center_constraint="permutation_tour",
                name=f"LLMCallFailed{attempt}",
                family="llm_call_failure",
                code_hash=h,
                valid=False,
                full_valid=False,
                valid_instances=0,
                total_instances=len(problems),
                mean_gap_percent=None,
                selection_score=None,
                mean_runtime_s=None,
                parent_attempt=parent.attempt if parent else None,
                selection_strategy=strategy,
                prompt_mode=prompt_mode,
                error=f"{type(e).__name__}: {e}",
                code_path=None,
                raw_response_path=None,
                prompt_path=str(prompt_path),
                family_desc=family_description("llm_call_failure"),
                **focus_record_fields,
                partial_valid_cases=0,
                partial_total_cases=len(problems),
                partial_failed_cases=len(problems),
            )
            records.append(rec)
            print(f"  failed: {rec.error}", flush=True)
            with (artifact_dir / "candidates.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
            _write_summary_artifacts(records, instance_rows, artifact_dir)
            continue

        raw_path = raw_dir / f"raw_iter_{attempt:03d}.txt"
        raw_path.write_text(raw, encoding="utf-8")
        (legacy_raw_dir / f"attempt_{attempt:04d}.txt").write_text(raw, encoding="utf-8")

        code = extract_first_python_block(raw)
        h = stable_hash(code)
        name = infer_candidate_name(raw, code, attempt)
        family = infer_family(name, code)
        fam_desc = family_description(family)
        family_focus_compliant, family_focus_compliance_level, family_focus_mismatch_reason = infer_family_focus_compliance(
            focus_spec,
            family,
            name,
            code,
        )
        code_path = code_dir / f"iter_{attempt:03d}_{h}.py"
        code_path.write_text(code, encoding="utf-8")
        legacy_code_path = legacy_code_dir / f"attempt_{attempt:04d}_{h}.py"
        legacy_code_path.write_text(code, encoding="utf-8")

        if h in seen_hashes:
            rec = CandidateRecord(
                attempt=attempt,
                iteration=attempt,
                objective_mode="tsp",
                center_constraint="permutation_tour",
                name=name,
                family=family,
                code_hash=h,
                valid=False,
                full_valid=False,
                valid_instances=0,
                total_instances=len(problems),
                mean_gap_percent=None,
                selection_score=None,
                mean_runtime_s=None,
                parent_attempt=parent.attempt if parent else None,
                selection_strategy=strategy,
                prompt_mode=prompt_mode,
                error="duplicate_code",
                code_path=str(code_path),
                raw_response_path=str(raw_path),
                prompt_path=str(prompt_path),
                family_desc=fam_desc,
                family_focus_compliant=family_focus_compliant,
                family_focus_compliance_level=family_focus_compliance_level,
                family_focus_mismatch_reason=family_focus_mismatch_reason,
                **focus_record_fields,
                partial_valid_cases=0,
                partial_total_cases=len(problems),
                partial_failed_cases=len(problems),
            )
            records.append(rec)
            print(f"  failed: ValueError: duplicate_code", flush=True)
            with (artifact_dir / "candidates.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
            _write_summary_artifacts(records, instance_rows, artifact_dir)
            continue
        seen_hashes.add(h)

        print(f"  name: {name}", flush=True)
        print(f"  family: {family}", flush=True)
        if focus_spec:
            print(
                f"  family-focus compliance: {family_focus_compliance_level}"
                + (f" | {family_focus_mismatch_reason}" if family_focus_mismatch_reason else ""),
                flush=True,
            )
        print("  search feedback:", flush=True)

        eval_results: list[EvaluationResult] = []
        last_trace = None
        current_instance_rows: list[dict[str, Any]] = []

        record_seed = {
            "code_hash": h,
            "name": name,
            "family": family,
            "parent_attempt": parent.attempt if parent else None,
            "selection_strategy": strategy,
            "prompt_mode": prompt_mode,
            "code_path": str(code_path),
            "family_focus_active": bool(focus_spec),
            "focus_family_id": focus_family_id,
            "focus_family_name": focus_family_name,
            "focus_family_step": focus_step,
            "focus_family_index": focus_index,
            "focus_family_total": focus_total,
            "family_focus_compliant": family_focus_compliant,
            "family_focus_compliance_level": family_focus_compliance_level,
            "family_focus_mismatch_reason": family_focus_mismatch_reason,
        }

        for problem_index, (instance_name, problem, optimum) in enumerate(problems):
            res = evaluate_code_on_problem(
                code,
                problem,
                optimum=optimum,
                seed=int(runtime_cfg.get("global_seed", 0)) + 1000 * attempt + problem_index,
                # Final tour validity is normal TSP permutation validity.
                # Candidate lists guide construction; cost is measured on full TSPLIB distance.
                timeout_s=float(runtime_cfg.get("candidate_timeout_s", 0) or 0),
            )
            eval_results.append(res)
            if res.traceback:
                last_trace = res.traceback
            row = _instance_result_row(attempt, record_seed, instance_name, problem, optimum, res)
            current_instance_rows.append(row)

        print(_format_instance_feedback(current_instance_rows), flush=True)

        feedback_by_instance = _format_instance_feedback(current_instance_rows)
        full_valid, valid_count, total, mean_gap, mean_cost, mean_runtime, score, err = summarize_results(
            eval_results,
            partial_failure_penalty=partial_failure_penalty,
        )
        rec = CandidateRecord(
            attempt=attempt,
            iteration=attempt,
            objective_mode="tsp",
            center_constraint="permutation_tour",
            name=name,
            family=family,
            code_hash=h,
            valid=valid_count > 0,
            full_valid=full_valid,
            valid_instances=valid_count,
            total_instances=total,
            mean_gap_percent=mean_gap,
            selection_score=score,
            mean_runtime_s=mean_runtime,
            parent_attempt=parent.attempt if parent else None,
            selection_strategy=strategy,
            prompt_mode=prompt_mode,
            error=err,
            code_path=str(code_path),
            raw_response_path=str(raw_path),
            prompt_path=str(prompt_path),
            family_desc=fam_desc,
            family_focus_compliant=family_focus_compliant,
            family_focus_compliance_level=family_focus_compliance_level,
            family_focus_mismatch_reason=family_focus_mismatch_reason,
            **focus_record_fields,
            feedback_by_instance=feedback_by_instance,
            search_cost_mean=mean_cost,
            search_gap_ref_mean_pct=mean_gap,
            partial_valid_cases=valid_count,
            partial_total_cases=total,
            partial_failed_cases=total - valid_count,
        )
        records.append(rec)

        # Add record-level values to all per-instance rows and write the per-iteration detail file.
        enriched_rows = []
        for row in current_instance_rows:
            enriched = dict(row)
            enriched.update(
                {
                    "full_valid": rec.full_valid,
                    "valid_instances": rec.valid_instances,
                    "total_instances": rec.total_instances,
                    "mean_gap_percent": rec.mean_gap_percent,
                    "search_gap_ref_mean_pct": rec.search_gap_ref_mean_pct,
                    "search_cost_mean": rec.search_cost_mean,
                    "selection_score": rec.selection_score,
                    "mean_runtime_s": rec.mean_runtime_s,
                    "record_error": rec.error,
                    "family_focus_compliant": rec.family_focus_compliant,
                    "family_focus_compliance_level": rec.family_focus_compliance_level,
                    "family_focus_mismatch_reason": rec.family_focus_mismatch_reason,
                }
            )
            enriched_rows.append(enriched)
        instance_rows.extend(enriched_rows)
        _write_csv(pd.DataFrame(enriched_rows), artifact_dir / f"search_detail_iter_{attempt:03d}.csv")

        if not full_valid:
            print(f"  invalid/partial: {err}", flush=True)
            if last_trace:
                print("\n".join(str(last_trace).splitlines()[:8]), flush=True)

        with (artifact_dir / "candidates.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")

        _write_summary_artifacts(records, instance_rows, artifact_dir)

    print("\nFinished LLM search loop.", flush=True)
    df = _write_summary_artifacts(records, instance_rows, artifact_dir)

    if not df.empty:
        cols = [
            "iteration",
            "objective_mode",
            "center_constraint",
            "name",
            "family",
            "full_valid",
            "valid_instances",
            "mean_gap_percent",
            "selection_score",
            "mean_runtime_s",
            "family_focus_compliance_level",
            "family_focus_mismatch_reason",
            "code_path",
        ]
        existing = [c for c in cols if c in df.columns]
        show = df.copy()
        if "selection_score" in show:
            show = show.sort_values(["selection_score", "iteration"], na_position="last")
        print(show[existing].head(10), flush=True)

        if "code_path" in df:
            print(df[["iteration", "name", "family", "selection_score", "mean_gap_percent", "code_path"]].tail(min(20, len(df))), flush=True)

    return df
