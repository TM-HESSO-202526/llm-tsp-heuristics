from __future__ import annotations

from pathlib import Path
from collections import defaultdict
import random
import numpy as np


CandidateMap = dict[int, list[int]]
PriorMap = dict[tuple[int, int], float]


def normalize_candidates(cands: CandidateMap, n: int, max_k: int = 20, dist: np.ndarray | None = None) -> CandidateMap:
    """Make candidate lists bidirectional, sorted, deduplicated, and truncated."""
    sets = {i: set(cands.get(i, [])) for i in range(n)}
    for i, neighs in list(sets.items()):
        for j in list(neighs):
            if 0 <= j < n and j != i:
                sets.setdefault(j, set()).add(i)
    out: CandidateMap = {}
    for i in range(n):
        neighs = [j for j in sets.get(i, set()) if 0 <= j < n and j != i]
        if dist is not None:
            neighs.sort(key=lambda j: float(dist[i, j]))
        else:
            neighs.sort()
        out[i] = neighs[:max_k]
    return out


def k_nearest_candidates(dist: np.ndarray, max_k: int = 20) -> CandidateMap:
    n = dist.shape[0]
    out = {}
    for i in range(n):
        order = np.argsort(dist[i])
        out[i] = [int(j) for j in order if int(j) != i][:max_k]
    return normalize_candidates(out, n=n, max_k=max_k, dist=dist)


def parse_simple_candidate_file(path: str | Path, n: int | None = None) -> CandidateMap:
    """Parse normalized candidate rows or LKH CANDIDATE_FILE output.

    Supported formats:
    - normalized 0-based rows: ``0 12 94 103``
    - normalized 1-based rows: ``1 13 95 104``
    - LKH candidate-set rows inside ``CANDIDATE_SET_SECTION`` where rows are
      ``node degree neighbor alpha neighbor alpha ...`` using 1-based node ids.

    The returned map is always 0-based and normalized/truncated later by the
    caller when needed.
    """
    path = Path(path)
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    has_lkh_section = any("CANDIDATE_SET_SECTION" in line.upper() for line in lines)

    cands: CandidateMap = {}
    max_node = -1
    raw_rows: list[list[int]] = []
    in_lkh = not has_lkh_section

    for raw in lines:
        line = raw.strip()
        upper = line.upper()
        if not line or line.startswith("#"):
            continue
        if "CANDIDATE_SET_SECTION" in upper:
            in_lkh = True
            continue
        if upper.startswith("EOF") or line == "-1":
            if has_lkh_section:
                break
            continue
        if has_lkh_section and not in_lkh:
            continue

        parts: list[int] = []
        for tok in line.replace(",", " ").split():
            try:
                parts.append(int(tok))
            except ValueError:
                pass
        if len(parts) >= 2:
            raw_rows.append(parts)
            max_node = max(max_node, *parts)

    if n is None:
        # LKH uses 1-based ids; simple files may use 0-based ids. The normalize
        # step below will correct once we detect which style is present.
        n = max_node + 1

    # Detect LKH rows only when the file explicitly has the LKH section marker.
    # This avoids misreading simple rows like "0 1 2 3" as degree/pair rows.
    looks_lkh = has_lkh_section
    min_first = min((r[0] for r in raw_rows), default=0)
    max_first = max((r[0] for r in raw_rows), default=-1)
    one_based = has_lkh_section or (min_first >= 1 and max_first <= n)

    for parts in raw_rows:
        if looks_lkh and len(parts) >= 4:
            i_raw = parts[0]
            degree = max(0, parts[1])
            pair_tokens = parts[2:]
            neighs_raw = pair_tokens[0 : 2 * degree : 2] if degree else pair_tokens[::2]
        else:
            i_raw = parts[0]
            neighs_raw = parts[1:]

        i = i_raw - 1 if one_based else i_raw
        neighs = [(j - 1 if one_based else j) for j in neighs_raw]
        if i < 0:
            continue
        cands.setdefault(i, []).extend(j for j in neighs if j != i)

    return normalize_candidates(cands, n=n)

def prior_from_candidate_frequency(candidate_runs: list[CandidateMap], n: int) -> PriorMap:
    counts: dict[tuple[int, int], float] = defaultdict(float)
    for cands in candidate_runs:
        for i, neighs in cands.items():
            for j in neighs:
                if i == j:
                    continue
                a, b = sorted((int(i), int(j)))
                counts[(a, b)] += 1.0
    if not counts:
        return {}
    max_count = max(counts.values())
    return {edge: val / max_count for edge, val in counts.items()}


def binary_topk_prior(prior: PriorMap, k_per_node: int, n: int) -> PriorMap:
    by_node: dict[int, list[tuple[int, float]]] = {i: [] for i in range(n)}
    for (a, b), w in prior.items():
        by_node[a].append((b, w))
        by_node[b].append((a, w))
    out: PriorMap = {}
    for i, vals in by_node.items():
        vals.sort(key=lambda x: -x[1])
        for j, _ in vals[:k_per_node]:
            a, b = sorted((i, j))
            out[(a, b)] = 1.0
    return out


def shuffled_prior(prior: PriorMap, n: int, seed: int = 0) -> PriorMap:
    rng = random.Random(seed)
    weights = list(prior.values())
    edges = list(prior.keys())
    rng.shuffle(weights)
    return {edge: w for edge, w in zip(edges, weights)}
