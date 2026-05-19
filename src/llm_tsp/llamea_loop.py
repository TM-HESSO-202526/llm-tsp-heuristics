from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Callable
import hashlib
import json
from pathlib import Path
import pandas as pd

from .parsing import extract_first_python_block
from .prompts import SYSTEM_PROMPT, build_tsp_prompt
from .evaluation import evaluate_code_on_problem, EvaluationResult
from .sparse_problem import SparseTSPProblem


@dataclass
class CandidateRecord:
    attempt: int
    code_hash: str
    valid: bool
    mean_gap_percent: float | None
    mean_runtime_s: float | None
    error: str | None
    code_path: str | None
    raw_response_path: str | None


def stable_hash(text: str, n: int = 16) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:n]


def summarize_results(results: list[EvaluationResult]) -> tuple[float | None, float | None, str | None]:
    valid = [r for r in results if r.valid]
    if not valid:
        err = next((r.error for r in results if r.error), "all invalid")
        return None, None, err
    gaps = [r.gap_percent for r in valid if r.gap_percent is not None]
    runtimes = [r.runtime_s for r in valid if r.runtime_s is not None]
    mean_gap = sum(gaps) / len(gaps) if gaps else None
    mean_runtime = sum(runtimes) / len(runtimes) if runtimes else None
    return mean_gap, mean_runtime, None


def make_feedback(record: CandidateRecord, last_error_trace: str | None = None, include_trace: bool = True) -> str:
    if record.valid:
        return f"Last valid candidate mean gap: {record.mean_gap_percent:.4f}% ; mean runtime: {record.mean_runtime_s:.3f}s. Improve the gap without increasing complexity too much."
    msg = f"Last candidate was invalid: {record.error}"
    if include_trace and last_error_trace:
        msg += "\nTraceback:\n" + last_error_trace
    return msg


def run_llamea_search(
    config: dict,
    problems: list[tuple[str, SparseTSPProblem, float | None]],
    llm_call: Callable[[list[dict[str, str]]], str],
    artifact_dir: str | Path,
) -> pd.DataFrame:
    """Run a minimal LLaMEA-style search loop.

    This is intentionally generic. The production Colab workflow can wrap this
    with provider-specific rate limits, richer prompt variants, or additional
    run metadata while keeping the same artifact format.
    """
    artifact_dir = Path(artifact_dir)
    code_dir = artifact_dir / "generated_code"
    raw_dir = artifact_dir / "raw_llm_responses"
    code_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    max_calls = int(config.get("llm", {}).get("max_llm_calls", 1))
    feedback_cfg = config.get("feedback", {})
    include_trace = bool(feedback_cfg.get("include_invalid_error_trace", True))
    parent_code: str | None = None
    previous_feedback: str | None = None
    records: list[CandidateRecord] = []
    seen_hashes: set[str] = set()

    for attempt in range(1, max_calls + 1):
        prompt = build_tsp_prompt(config, previous_feedback=previous_feedback, parent_code=parent_code)
        raw = llm_call([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ])
        raw_path = raw_dir / f"attempt_{attempt:04d}.txt"
        raw_path.write_text(raw, encoding="utf-8")
        code = extract_first_python_block(raw)
        h = stable_hash(code)
        code_path = code_dir / f"attempt_{attempt:04d}_{h}.py"
        code_path.write_text(code, encoding="utf-8")

        if h in seen_hashes:
            rec = CandidateRecord(attempt, h, False, None, None, "duplicate_code", str(code_path), str(raw_path))
            records.append(rec)
            previous_feedback = make_feedback(rec, include_trace=include_trace)
            continue
        seen_hashes.add(h)

        eval_results = []
        last_trace = None
        for name, problem, optimum in problems:
            res = evaluate_code_on_problem(
                code,
                problem,
                optimum=optimum,
                seed=int(config.get("runtime", {}).get("global_seed", 0)) + attempt,
                require_candidate_tour=bool(config.get("popmusic", {}).get("allow_non_candidate_edges_in_final_tour", True) is False and config.get("popmusic", {}).get("restrict_edge_cost_to_candidates", False)),
            )
            eval_results.append(res)
            if res.traceback:
                last_trace = res.traceback
        mean_gap, mean_runtime, err = summarize_results(eval_results)
        valid = err is None and mean_gap is not None
        rec = CandidateRecord(attempt, h, valid, mean_gap, mean_runtime, err, str(code_path), str(raw_path))
        records.append(rec)

        with (artifact_dir / "candidates.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")

        if valid:
            if parent_code is None or mean_gap < min([r.mean_gap_percent for r in records if r.valid and r.mean_gap_percent is not None], default=float("inf")):
                parent_code = code
        previous_feedback = make_feedback(rec, last_error_trace=last_trace, include_trace=include_trace)

    df = pd.DataFrame([asdict(r) for r in records])
    df.to_csv(artifact_dir / "generated_attempts.csv", index=False)
    return df
