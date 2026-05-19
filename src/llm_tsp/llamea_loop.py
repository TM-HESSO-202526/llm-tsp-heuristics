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


def stable_hash(text: str, n: int = 16) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:n]


def _is_timeout_error(error: str | None) -> bool:
    if not error:
        return False
    e = error.lower()
    return "timeout" in e or "timed out" in e or "time limit" in e


def summarize_results(results: list[EvaluationResult], partial_failure_penalty: float = 200.0) -> tuple[bool, int, int, float | None, float | None, float | None, str | None]:
    total = len(results)
    valid_results = [r for r in results if r.valid]
    valid_count = len(valid_results)
    full_valid = valid_count == total and total > 0

    gaps = [r.gap_percent for r in valid_results if r.gap_percent is not None]
    runtimes = [r.runtime_s for r in valid_results if r.runtime_s is not None]
    mean_gap = sum(gaps) / len(gaps) if gaps else None
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

    return full_valid, valid_count, total, mean_gap, mean_runtime, score, err


def _record_line(record: CandidateRecord) -> str:
    status = "full-valid" if record.full_valid else ("partial" if record.valid_instances else "invalid")
    score = "NA" if record.selection_score is None else f"{record.selection_score:.4f}"
    gap = "NA" if record.mean_gap_percent is None else f"{record.mean_gap_percent:.4f}%"
    err = f" error={record.error}" if record.error else ""
    return (
        f"attempt {record.attempt}: {status}, valid={record.valid_instances}/{record.total_instances}, "
        f"gap={gap}, selection_score={score}, hash={record.code_hash}{err}"
    )


def build_history_text(records: list[CandidateRecord], limit: int = 20) -> str | None:
    if not records or limit <= 0:
        return None
    recent = records[-limit:]
    return "\n".join(_record_line(r) for r in recent)


def select_parent(records: list[CandidateRecord], strategy: str) -> CandidateRecord | None:
    if not records:
        return None
    if strategy == "1+1":
        full_valid = [r for r in records if r.full_valid and r.selection_score is not None]
        if full_valid:
            return min(full_valid, key=lambda r: (r.selection_score, r.attempt))
        # Before a full-valid candidate exists, the invalid-parent redesign logic may use latest feedback.
        return records[-1]
    if strategy == "1,1":
        return records[-1]
    raise ValueError(f"Unknown selection_strategy {strategy!r}; expected '1+1' or '1,1'.")


def _read_code_from_record(record: CandidateRecord | None) -> str | None:
    if not record or not record.code_path:
        return None
    path = Path(record.code_path)
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def make_feedback(
    record: CandidateRecord,
    code: str | None = None,
    last_error_trace: str | None = None,
    include_trace: bool = True,
    include_invalid_code: bool = True,
    hide_invalid_code: bool = False,
) -> str:
    if record.full_valid:
        return (
            f"Last candidate was full-valid on {record.valid_instances}/{record.total_instances} instance(s). "
            f"Mean gap: {record.mean_gap_percent:.4f}% ; mean runtime: {record.mean_runtime_s:.3f}s. "
            "Improve the gap without increasing complexity too much."
        )

    if record.valid_instances > 0:
        msg = (
            f"Last candidate was only partially valid: {record.valid_instances}/{record.total_instances} instance(s) passed. "
            f"Selection score: {record.selection_score}. Error: {record.error}"
        )
    else:
        msg = f"Last candidate was invalid on all evaluated instances: {record.error}"

    if include_trace and last_error_trace:
        msg += "\nTraceback:\n" + last_error_trace

    if include_invalid_code and not hide_invalid_code and code:
        msg += "\nInvalid/partial candidate code for diagnosis:\n```python\n" + code + "\n```"

    return msg


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
    - invalid-code/error-trace exposure is controlled by the feedback flags.
    """
    artifact_dir = Path(artifact_dir)
    code_dir = artifact_dir / "generated_code"
    raw_dir = artifact_dir / "raw_llm_responses"
    code_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    llm_cfg = config.get("llm", {})
    runtime_cfg = config.get("runtime", {})
    feedback_cfg = config.get("feedback", {})
    search_cfg = config.get("search", {})
    pop_cfg = config.get("popmusic", {})

    max_calls = int(llm_cfg.get("max_llm_calls", 1))
    strategy = str(search_cfg.get("selection_strategy", "1+1"))
    if strategy not in {"1+1", "1,1"}:
        raise ValueError(f"selection_strategy must be '1+1' or '1,1', got {strategy!r}")
    history_limit = int(search_cfg.get("history_limit", 20))
    include_trace = bool(feedback_cfg.get("include_invalid_error_trace", True))
    include_invalid_code = bool(feedback_cfg.get("include_invalid_code_in_feedback", True))
    hide_invalid_code = bool(search_cfg.get("hide_invalid_parent_code", False))
    partial_failure_penalty = float(runtime_cfg.get("partial_failure_penalty", 200.0))

    records: list[CandidateRecord] = []
    seen_hashes: set[str] = set()
    previous_feedback: str | None = None
    previous_feedback_code: str | None = None
    previous_trace: str | None = None

    for attempt in range(1, max_calls + 1):
        parent = select_parent(records, strategy)
        parent_code = _read_code_from_record(parent)
        prompt_mode = "initial" if parent is None else "mutate_parent"
        parent_is_invalid = bool(parent is not None and not parent.full_valid)
        if _should_use_redesign_prompt(parent, records, search_cfg):
            prompt_mode = "redesign_invalid_parent"

        # Do not expose invalid parent code unless the two clustering-style flags allow it.
        if parent_is_invalid and (hide_invalid_code or not include_invalid_code):
            parent_code_for_prompt = None
        else:
            parent_code_for_prompt = parent_code

        prompt = build_tsp_prompt(
            config,
            previous_feedback=make_feedback(
                parent,
                code=previous_feedback_code,
                last_error_trace=previous_trace,
                include_trace=include_trace,
                include_invalid_code=include_invalid_code,
                hide_invalid_code=hide_invalid_code,
            ) if parent is not None else previous_feedback,
            parent_code=parent_code_for_prompt,
            history_text=build_history_text(records, history_limit),
            prompt_mode=prompt_mode,
            parent_is_invalid=parent_is_invalid,
        )

        prompt_path = artifact_dir / "prompts" / f"attempt_{attempt:04d}.txt"
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_path.write_text(prompt, encoding="utf-8")

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
            rec = CandidateRecord(
                attempt=attempt,
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
            )
            records.append(rec)
            previous_feedback = make_feedback(rec, code=code, include_trace=include_trace, include_invalid_code=include_invalid_code, hide_invalid_code=hide_invalid_code)
            previous_feedback_code = code
            previous_trace = None
            with (artifact_dir / "candidates.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
            continue
        seen_hashes.add(h)

        eval_results = []
        last_trace = None
        for problem_index, (name, problem, optimum) in enumerate(problems):
            res = evaluate_code_on_problem(
                code,
                problem,
                optimum=optimum,
                seed=int(runtime_cfg.get("global_seed", 0)) + 1000 * attempt + problem_index,
                require_candidate_tour=bool(
                    pop_cfg.get("allow_non_candidate_edges_in_final_tour", True) is False
                    and pop_cfg.get("restrict_edge_cost_to_candidates", False)
                ),
            )
            eval_results.append(res)
            if res.traceback:
                last_trace = res.traceback

        full_valid, valid_count, total, mean_gap, mean_runtime, score, err = summarize_results(
            eval_results,
            partial_failure_penalty=partial_failure_penalty,
        )
        rec = CandidateRecord(
            attempt=attempt,
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
        )
        records.append(rec)

        with (artifact_dir / "candidates.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")

        previous_feedback = make_feedback(
            rec,
            code=code,
            last_error_trace=last_trace,
            include_trace=include_trace,
            include_invalid_code=include_invalid_code,
            hide_invalid_code=hide_invalid_code,
        )
        previous_feedback_code = code
        previous_trace = last_trace

    df = pd.DataFrame([asdict(r) for r in records])
    if bool(feedback_cfg.get("save_generated_attempts", True)):
        df.to_csv(artifact_dir / "generated_attempts.csv", index=False)
    return df
