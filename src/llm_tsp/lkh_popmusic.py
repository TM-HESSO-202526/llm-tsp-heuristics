from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict
import shutil
import subprocess
import numpy as np

from .candidate_sets import PriorMap, binary_topk_prior


@dataclass(frozen=True)
class PopmusicParams:
    max_candidates: int = 20
    popmusic_sample_size: int = 14
    popmusic_solutions: int = 20
    popmusic_max_neighbors: int = 5
    popmusic_trials: int = 1
    popmusic_initial_tour: bool = False


@dataclass(frozen=True)
class EdgePriorParams:
    runs: int = 30
    time_limit_s: float = 1.0
    topk: int = 5
    move_type: int = 5
    patching_a: int = 2
    patching_c: int = 3
    force_rebuild: bool = False


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


def popmusic_edge_prior_file_name(
    instance_name: str,
    edge_prior_root: str | Path,
    params: EdgePriorParams = EdgePriorParams(),
) -> Path:
    """Historical LKH/POPMUSIC tour-frequency prior-cache filename."""
    return Path(edge_prior_root) / (
        f"{instance_name}_popmusic_edge_prior_runs{params.runs}_topk{params.topk}.npz"
    )


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


def write_edge_prior_lkh_parameter_file(
    tsp_file: str | Path,
    par_file: str | Path,
    output_tour_file: str | Path,
    *,
    seed: int,
    popmusic: PopmusicParams = PopmusicParams(),
    prior: EdgePriorParams = EdgePriorParams(),
) -> Path:
    """Write the historical short-run LKH .par file used to build edge priors."""
    tsp_file = Path(tsp_file)
    par_file = Path(par_file)
    output_tour_file = Path(output_tour_file)
    par_file.parent.mkdir(parents=True, exist_ok=True)
    output_tour_file.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"PROBLEM_FILE = {tsp_file}",
        "CANDIDATE_SET_TYPE = POPMUSIC",
        f"MAX_CANDIDATES = {popmusic.max_candidates}",
        f"POPMUSIC_SAMPLE_SIZE = {popmusic.popmusic_sample_size}",
        f"POPMUSIC_SOLUTIONS = {popmusic.popmusic_solutions}",
        f"POPMUSIC_MAX_NEIGHBORS = {popmusic.popmusic_max_neighbors}",
        f"POPMUSIC_TRIALS = {popmusic.popmusic_trials}",
        f"POPMUSIC_INITIAL_TOUR = {'YES' if popmusic.popmusic_initial_tour else 'NO'}",
        "RUNS = 1",
        f"MOVE_TYPE = {prior.move_type}",
        f"PATCHING_A = {prior.patching_a}",
        f"PATCHING_C = {prior.patching_c}",
        f"SEED = {int(seed)}",
        f"TIME_LIMIT = {float(prior.time_limit_s)}",
        "TRACE_LEVEL = 0",
        f"OUTPUT_TOUR_FILE = {output_tour_file}",
    ]
    par_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return par_file


def ensure_lkh_binary(lkh_binary: str | Path, *, build_if_missing: bool = True, timeout_s: float = 600.0) -> Path:
    """Return a usable LKH 3.0.8 binary, building it in Colab if missing."""
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

    # Historical Colab workflow: install tools, download LKH-3.0.8, make, copy binary.
    script = f"""
set -euo pipefail
cd {tools_dir}
if [ ! -d LKH-3.0.8 ]; then
  rm -f LKH-3.0.8.tgz
  (wget -q -O LKH-3.0.8.tgz http://akira.ruc.dk/~keld/research/LKH-3/LKH-3.0.8.tgz \
   || wget -q -O LKH-3.0.8.tgz https://webhotel4.ruc.dk/~keld/research/LKH-3/LKH-3.0.8.tgz \
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
    """Generate a historical LKH/POPMUSIC CANDIDATE_FILE and return its path."""
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


def parse_lkh_tour_file(path: str | Path) -> list[int]:
    """Parse a 1-based LKH TOUR_SECTION output file and return a 0-based tour."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return []
    text = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    in_section = False
    nodes: list[int] = []
    for raw in text:
        line = raw.strip()
        upper = line.upper()
        if not line:
            continue
        if upper.startswith("TOUR_SECTION"):
            in_section = True
            continue
        if upper.startswith("EOF"):
            break
        if not in_section:
            continue
        for tok in line.replace(",", " ").split():
            try:
                val = int(tok)
            except ValueError:
                continue
            if val == -1:
                return [x - 1 for x in nodes]
            if val > 0:
                nodes.append(val)
    return [x - 1 for x in nodes]


def _topk_symmetrized_prior(counts: dict[tuple[int, int], float], n: int, success_runs: int, topk: int) -> PriorMap:
    if success_runs <= 0 or not counts:
        return {}
    raw = {edge: float(val) / float(success_runs) for edge, val in counts.items()}
    if topk and topk > 0:
        return binary_topk_prior(raw, k_per_node=int(topk), n=n) | {
            edge: raw[edge] for edge in binary_topk_prior(raw, k_per_node=int(topk), n=n).keys()
        }
    return raw


def save_prior_npz(path: str | Path, prior: PriorMap, *, success_runs: int, attempted_runs: int, topk: int) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    edges = np.array(list(prior.keys()), dtype=np.int64) if prior else np.zeros((0, 2), dtype=np.int64)
    weights = np.array([prior[tuple(edge)] for edge in map(tuple, edges)], dtype=np.float64) if len(edges) else np.zeros((0,), dtype=np.float64)
    np.savez_compressed(
        path,
        edge_i=edges[:, 0] if len(edges) else np.zeros((0,), dtype=np.int64),
        edge_j=edges[:, 1] if len(edges) else np.zeros((0,), dtype=np.int64),
        weight=weights,
        success_runs=np.array([success_runs], dtype=np.int64),
        attempted_runs=np.array([attempted_runs], dtype=np.int64),
        topk=np.array([topk], dtype=np.int64),
    )
    return path


def _npz_scalar(data: np.lib.npyio.NpzFile, key: str, default=None):
    if key not in data:
        return default
    value = data[key]
    try:
        arr = np.asarray(value)
        if arr.shape == ():
            return arr.item()
        if arr.size == 1:
            return arr.reshape(-1)[0].item()
    except Exception:
        pass
    return value


def _dense_prior_matrix_to_map(matrix: np.ndarray) -> PriorMap:
    """Convert the historical dense `prior` matrix cache to sparse PriorMap.

    The original notebooks saved edge priors as a dense float32 matrix under the
    key `prior`, together with `success_runs` and `method`.  Existing Drive
    caches therefore do not contain the newer `edge_i`/`edge_j`/`weight` arrays.
    We keep only strictly positive off-diagonal entries and store undirected
    edges using sorted `(a, b)` keys, which is what `SparseTSPProblem.prior(i,j)`
    expects.
    """
    arr = np.asarray(matrix, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"Legacy prior matrix must be square, got shape {arr.shape}")
    rows, cols = np.nonzero(arr > 0)
    prior: PriorMap = {}
    for i, j in zip(rows, cols):
        i = int(i)
        j = int(j)
        if i == j:
            continue
        a, b = sorted((i, j))
        w = float(arr[i, j])
        if w > prior.get((a, b), 0.0):
            prior[(a, b)] = w
    return prior


def load_prior_npz(path: str | Path) -> tuple[PriorMap, dict[str, int | str]]:
    """Load an LKH/POPMUSIC edge-prior cache.

    Supports both formats used across the project:

    1. Historical notebook cache:
       `prior=<dense float32 matrix>`, `success_runs=<int>`, `method=<str>`

    2. Newer sparse cache:
       `edge_i`, `edge_j`, `weight`, `success_runs`, `attempted_runs`, `topk`

    This compatibility is important because the user's Drive already contains
    historical `.npz` files such as
    `{instance}_popmusic_edge_prior_runs30_topk5.npz`.
    """
    path = Path(path)
    data = np.load(path, allow_pickle=True)

    if "prior" in data:
        prior = _dense_prior_matrix_to_map(data["prior"])
        success_runs = _npz_scalar(data, "success_runs", -1)
        attempted_runs = _npz_scalar(data, "attempted_runs", -1)
        topk = _npz_scalar(data, "topk", -1)
        method = _npz_scalar(data, "method", "legacy_dense_prior")
        if isinstance(method, bytes):
            method = method.decode("utf-8", errors="ignore")
        else:
            method = str(method)
        meta = {
            "format": "legacy_dense_prior",
            "method": method,
            "success_runs": int(success_runs) if success_runs is not None else -1,
            "attempted_runs": int(attempted_runs) if attempted_runs is not None else -1,
            "topk": int(topk) if topk is not None else -1,
        }
        return prior, meta

    required = {"edge_i", "edge_j", "weight"}
    missing = sorted(required.difference(set(data.files)))
    if missing:
        raise KeyError(
            f"Edge-prior cache {path} has unsupported schema. "
            f"Missing {missing}; available keys are {list(data.files)}"
        )

    edge_i = data["edge_i"].astype(int)
    edge_j = data["edge_j"].astype(int)
    weights = data["weight"].astype(float)
    prior = {(int(i), int(j)): float(w) for i, j, w in zip(edge_i, edge_j, weights)}
    meta = {
        "format": "sparse_edge_list",
        "method": str(_npz_scalar(data, "method", "sparse_edge_list")),
        "success_runs": int(_npz_scalar(data, "success_runs", -1)),
        "attempted_runs": int(_npz_scalar(data, "attempted_runs", -1)),
        "topk": int(_npz_scalar(data, "topk", -1)),
    }
    return prior, meta


def run_popmusic_edge_prior_generation(
    tsp_file: str | Path,
    prior_file: str | Path,
    lkh_binary: str | Path,
    *,
    n: int,
    base_seed: int,
    popmusic: PopmusicParams = PopmusicParams(),
    prior_params: EdgePriorParams = EdgePriorParams(),
    timeout_s: float = 1800.0,
) -> Path:
    """Generate the historical LKH/POPMUSIC tour-frequency edge prior cache.

    For each short LKH run, this writes a separate .par file containing:
    CANDIDATE_SET_TYPE=POPMUSIC, MAX_CANDIDATES, POPMUSIC_*, RUNS=1,
    MOVE_TYPE=5, PATCHING_A=2, PATCHING_C=3, SEED, TIME_LIMIT=1.0,
    TRACE_LEVEL=0, and OUTPUT_TOUR_FILE.

    Successful tours are parsed and edge frequencies are counted symmetrically.
    The resulting prior is saved as the historical .npz cache file.
    """
    tsp_file = Path(tsp_file)
    prior_file = Path(prior_file)
    prior_file.parent.mkdir(parents=True, exist_ok=True)
    binary = ensure_lkh_binary(lkh_binary, build_if_missing=True, timeout_s=min(float(timeout_s), 900.0))

    work_dir = prior_file.parent / (prior_file.stem + "_work")
    work_dir.mkdir(parents=True, exist_ok=True)
    log_dir = work_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    counts: dict[tuple[int, int], float] = defaultdict(float)
    success_runs = 0
    attempted = int(prior_params.runs)
    per_run_rows = []

    for r in range(attempted):
        seed = int(base_seed) + r
        par_file = work_dir / f"run_{r+1:03d}.par"
        tour_file = work_dir / f"run_{r+1:03d}.tour"
        write_edge_prior_lkh_parameter_file(
            tsp_file,
            par_file,
            tour_file,
            seed=seed,
            popmusic=popmusic,
            prior=prior_params,
        )
        try:
            proc = subprocess.run(
                [str(binary), str(par_file)],
                text=True,
                capture_output=True,
                timeout=max(30.0, float(prior_params.time_limit_s) + 20.0),
            )
            (log_dir / f"run_{r+1:03d}.log").write_text(
                "STDOUT\n======\n" + proc.stdout + "\n\nSTDERR\n======\n" + proc.stderr,
                encoding="utf-8",
            )
            tour = parse_lkh_tour_file(tour_file)
            ok = proc.returncode == 0 and len(tour) == n and len(set(tour)) == n
            if ok:
                success_runs += 1
                for k, a in enumerate(tour):
                    b = tour[(k + 1) % n]
                    if a == b:
                        continue
                    edge = tuple(sorted((int(a), int(b))))
                    counts[edge] += 1.0
            per_run_rows.append((r + 1, seed, int(proc.returncode), len(tour), bool(ok)))
        except subprocess.TimeoutExpired:
            (log_dir / f"run_{r+1:03d}.timeout").write_text("timeout\n", encoding="utf-8")
            per_run_rows.append((r + 1, seed, -999, 0, False))

    if success_runs == 0:
        raise RuntimeError(
            f"No successful LKH tours were produced while generating edge prior for {tsp_file.name}. "
            f"Check {work_dir}."
        )

    prior = _topk_symmetrized_prior(counts, n=n, success_runs=success_runs, topk=int(prior_params.topk))
    save_prior_npz(prior_file, prior, success_runs=success_runs, attempted_runs=attempted, topk=int(prior_params.topk))

    # Lightweight CSV log without a pandas dependency in this module.
    lines = ["run,seed,returncode,tour_len,success"]
    lines += [f"{a},{b},{c},{d},{e}" for a, b, c, d, e in per_run_rows]
    (work_dir / "edge_prior_runs.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return prior_file
