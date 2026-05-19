from __future__ import annotations

from dataclasses import dataclass
import time
import traceback
import numpy as np

from .distance import validate_tour, tour_cost_from_matrix
from .sparse_problem import SparseTSPProblem
from .parsing import reject_forbidden_code


@dataclass
class EvaluationResult:
    valid: bool
    cost: float | None = None
    gap_percent: float | None = None
    runtime_s: float | None = None
    error: str | None = None
    traceback: str | None = None
    uses_only_candidates: bool | None = None


def compile_candidate(code: str):
    reject_forbidden_code(code)
    ns: dict[str, object] = {}
    safe_globals = {
        "np": np,
        "numpy": np,
        "math": __import__("math"),
        "random": __import__("random"),
        "__builtins__": {
            "abs": abs,
            "all": all,
            "any": any,
            "bool": bool,
            "dict": dict,
            "enumerate": enumerate,
            "float": float,
            "int": int,
            "len": len,
            "list": list,
            "max": max,
            "min": min,
            "range": range,
            "reversed": reversed,
            "round": round,
            "set": set,
            "sorted": sorted,
            "sum": sum,
            "tuple": tuple,
            "zip": zip,
        },
    }
    exec(code, safe_globals, ns)
    fn = ns.get("construct_tour") or safe_globals.get("construct_tour")
    if not callable(fn):
        raise ValueError("Generated code must define construct_tour(problem, rng=None)")
    return fn


def evaluate_code_on_problem(
    code: str,
    problem: SparseTSPProblem,
    optimum: float | None = None,
    seed: int = 0,
    require_candidate_tour: bool = False,
) -> EvaluationResult:
    start = time.time()
    try:
        fn = compile_candidate(code)
        rng = np.random.default_rng(seed)
        tour = fn(problem, rng=rng)
        validate_tour(tour, problem.n)
        uses_only_candidates = problem.tour_uses_only_candidates(tour)
        if require_candidate_tour and not uses_only_candidates:
            raise ValueError("Final tour uses at least one non-candidate edge")
        cost = tour_cost_from_matrix(tour, problem.dist)
        gap = None
        if optimum and optimum > 0:
            gap = 100.0 * (cost - float(optimum)) / float(optimum)
        return EvaluationResult(
            valid=True,
            cost=cost,
            gap_percent=gap,
            runtime_s=time.time() - start,
            uses_only_candidates=uses_only_candidates,
        )
    except Exception as e:
        return EvaluationResult(
            valid=False,
            runtime_s=time.time() - start,
            error=str(e),
            traceback=traceback.format_exc(limit=8),
        )
