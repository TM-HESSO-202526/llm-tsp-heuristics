from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass


@dataclass(frozen=True)
class PopmusicParams:
    max_candidates: int = 20
    popmusic_sample_size: int = 14
    popmusic_solutions: int = 20
    popmusic_max_neighbors: int = 5
    popmusic_trials: int = 1
    popmusic_initial_tour: bool = False


def write_lkh_parameter_file(
    tsp_file: str | Path,
    candidate_file: str | Path,
    par_file: str | Path,
    params: PopmusicParams = PopmusicParams(),
) -> Path:
    tsp_file = Path(tsp_file)
    candidate_file = Path(candidate_file)
    par_file = Path(par_file)
    par_file.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"PROBLEM_FILE = {tsp_file}",
        f"CANDIDATE_FILE = {candidate_file}",
        f"MAX_CANDIDATES = {params.max_candidates}",
        f"POPMUSIC_SAMPLE_SIZE = {params.popmusic_sample_size}",
        f"POPMUSIC_SOLUTIONS = {params.popmusic_solutions}",
        f"POPMUSIC_MAX_NEIGHBORS = {params.popmusic_max_neighbors}",
        f"POPMUSIC_TRIALS = {params.popmusic_trials}",
        f"POPMUSIC_INITIAL_TOUR = {'YES' if params.popmusic_initial_tour else 'NO'}",
        "RUNS = 1",
    ]
    par_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return par_file
