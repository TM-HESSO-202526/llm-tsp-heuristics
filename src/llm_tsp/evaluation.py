from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
import signal
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


class CandidateTimeoutError(TimeoutError):
    pass


@contextmanager
def _time_limit(timeout_s: float | None):
    """Best-effort Unix timeout for generated heuristic evaluation."""
    if timeout_s is None or timeout_s <= 0 or not hasattr(signal, "SIGALRM"):
        yield
        return

    def _handler(signum, frame):  # pragma: no cover - timing dependent
        raise CandidateTimeoutError(f"candidate timed out after {timeout_s:.1f}s")

    old_handler = signal.getsignal(signal.SIGALRM)
    old_timer = signal.setitimer(signal.ITIMER_REAL, float(timeout_s))
    signal.signal(signal.SIGALRM, _handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)
        # Preserve any pre-existing timer if there was one.
        if old_timer and old_timer[0] > 0:  # pragma: no cover
            signal.setitimer(signal.ITIMER_REAL, old_timer[0], old_timer[1])


def _restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = name.split(".")[0].lower()
    allowed = {"math", "random", "numpy"}
    if root not in allowed:
        raise ImportError(f"Imports are restricted; attempted to import {name!r}")
    return __import__(name, globals, locals, fromlist, level)


def compile_candidate(code: str):
    reject_forbidden_code(code)
    ns: dict[str, object] = {}
    safe_builtins = {
        "__build_class__": __build_class__,
        "__import__": _restricted_import,
        "abs": abs,
        "all": all,
        "any": any,
        "bool": bool,
        "dict": dict,
        "enumerate": enumerate,
        "Exception": Exception,
        "float": float,
        "int": int,
        "isinstance": isinstance,
        "len": len,
        "list": list,
        "map": map,
        "max": max,
        "min": min,
        "object": object,
        "print": print,
        "range": range,
        "reversed": reversed,
        "round": round,
        "set": set,
        "slice": slice,
        "sorted": sorted,
        "sum": sum,
        "tuple": tuple,
        "ValueError": ValueError,
        "zip": zip,
    }
    safe_globals = {
        "np": np,
        "numpy": np,
        "math": __import__("math"),
        "random": __import__("random"),
        "__builtins__": safe_builtins,
        "__name__": "generated_tsp_candidate",
    }
    exec(code, safe_globals, ns)
    cls = ns.get("TSPHeuristic") or safe_globals.get("TSPHeuristic")
    if cls is None:
        raise ValueError("Generated code must define class TSPHeuristic")
    algo = cls()
    if not callable(algo):
        raise ValueError("TSPHeuristic instance is not callable")

    def _call(problem, rng=None):
        return algo(problem, rng)

    return _call


def evaluate_code_on_problem(
    code: str,
    problem: SparseTSPProblem,
    optimum: float | None = None,
    seed: int = 0,
    timeout_s: float | None = None,
) -> EvaluationResult:
    start = time.time()
    try:
        with _time_limit(timeout_s):
            fn = compile_candidate(code)
            rng = np.random.default_rng(seed)
            tour = fn(problem, rng=rng)
            validate_tour(tour, problem.n)
            uses_only_candidates = problem.tour_uses_only_candidates(tour)
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
