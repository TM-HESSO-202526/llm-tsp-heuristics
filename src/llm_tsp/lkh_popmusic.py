from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
import shutil
import subprocess


@dataclass(frozen=True)
class PopmusicParams:
    max_candidates: int = 20
    popmusic_sample_size: int = 14
    popmusic_solutions: int = 20
    popmusic_max_neighbors: int = 5
    popmusic_trials: int = 1
    popmusic_initial_tour: bool = False


def popmusic_candidate_file_name(
    instance_name: str,
    candidate_root: str | Path,
    params: PopmusicParams = PopmusicParams(),
) -> Path:
    """Historical POPMUSIC candidate-cache filename used by the original notebooks."""
    root = Path(candidate_root)
    stem = (
        f"{instance_name}"
        f"_cand-popmusic-k{params.max_candidates}"
        f"-s{params.popmusic_sample_size}"
        f"-sol{params.popmusic_solutions}"
        f"-nn{params.popmusic_max_neighbors}"
        f"-tr{params.popmusic_trials}.cand"
    )
    return root / stem


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
    candidate_file.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"PROBLEM_FILE = {tsp_file}",
        f"CANDIDATE_FILE = {candidate_file}",
        "CANDIDATE_SET_TYPE = POPMUSIC",
        f"MAX_CANDIDATES = {params.max_candidates}",
        f"POPMUSIC_SAMPLE_SIZE = {params.popmusic_sample_size}",
        f"POPMUSIC_SOLUTIONS = {params.popmusic_solutions}",
        f"POPMUSIC_MAX_NEIGHBORS = {params.popmusic_max_neighbors}",
        f"POPMUSIC_TRIALS = {params.popmusic_trials}",
        f"POPMUSIC_INITIAL_TOUR = {'YES' if params.popmusic_initial_tour else 'NO'}",
        "RUNS = 1",
        "TRACE_LEVEL = 1",
    ]
    par_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return par_file


def ensure_lkh_binary(lkh_binary: str | Path, *, build_if_missing: bool = True, timeout_s: float = 600.0) -> Path:
    """Return a usable LKH binary, building it in Colab if it is missing.

    The main pipeline uses this only when POPMUSIC candidate files are missing
    and candidate mode is active. Existing binaries are never rebuilt.
    """
    target = Path(lkh_binary)
    if target.exists():
        return target

    found = shutil.which("LKH")
    if found:
        return Path(found)

    if not build_if_missing:
        raise FileNotFoundError(f"LKH binary not found at {target}")

    tools_dir = Path("/content/tools") if str(target).startswith("/content/") else target.parent.parent
    tools_dir.mkdir(parents=True, exist_ok=True)
    target.parent.mkdir(parents=True, exist_ok=True)

    # Try both historical LKH hosting URLs used in Colab notebooks.
    script = f"""
set -euo pipefail
cd {tools_dir}
if [ ! -d LKH-3.0.8 ]; then
  rm -f LKH-3.0.8.tgz
  (wget -q -O LKH-3.0.8.tgz http://akira.ruc.dk/~keld/research/LKH-3/LKH-3.0.8.tgz \
   || wget -q -O LKH-3.0.8.tgz http://webhotel4.ruc.dk/~keld/research/LKH-3/LKH-3.0.8.tgz)
  tar xzf LKH-3.0.8.tgz
fi
make -C LKH-3.0.8 >/tmp/lkh_make_stdout.txt 2>/tmp/lkh_make_stderr.txt
cp LKH-3.0.8/LKH {target}
chmod +x {target}
"""
    proc = subprocess.run(
        ["bash", "-lc", script],
        text=True,
        capture_output=True,
        timeout=float(timeout_s),
    )
    if proc.returncode != 0 or not target.exists():
        raise RuntimeError(
            "Could not build LKH automatically.\n"
            f"stdout:\n{proc.stdout[-4000:]}\n"
            f"stderr:\n{proc.stderr[-4000:]}"
        )
    return target


def run_popmusic_candidate_generation(
    tsp_file: str | Path,
    candidate_file: str | Path,
    lkh_binary: str | Path,
    *,
    params: PopmusicParams = PopmusicParams(),
    timeout_s: float = 900.0,
) -> Path:
    """Generate a POPMUSIC/LKH candidate file and return its path."""
    tsp_file = Path(tsp_file)
    candidate_file = Path(candidate_file)
    candidate_file.parent.mkdir(parents=True, exist_ok=True)
    par_file = candidate_file.with_suffix(candidate_file.suffix + ".par")
    write_lkh_parameter_file(tsp_file, candidate_file, par_file, params=params)

    binary = ensure_lkh_binary(lkh_binary, build_if_missing=True, timeout_s=timeout_s)
    proc = subprocess.run(
        [str(binary), str(par_file)],
        text=True,
        capture_output=True,
        timeout=float(timeout_s),
    )
    log_file = candidate_file.with_suffix(candidate_file.suffix + ".lkh.log")
    log_file.write_text(
        "STDOUT\n======\n" + proc.stdout + "\n\nSTDERR\n======\n" + proc.stderr,
        encoding="utf-8",
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"LKH POPMUSIC generation failed for {tsp_file.name} with return code {proc.returncode}. "
            f"See {log_file}"
        )
    if not candidate_file.exists() or candidate_file.stat().st_size == 0:
        raise RuntimeError(
            f"LKH finished but did not create a non-empty candidate file at {candidate_file}. See {log_file}"
        )
    return candidate_file
