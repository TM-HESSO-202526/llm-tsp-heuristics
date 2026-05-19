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
    """Parse a permissive candidate file.

    This parser accepts simple text rows like:

    ```text
    0 12 94 103
    1 3 8 90
    ```

    i.e. first token is the source node, remaining integer tokens are neighbors.
    Historical LKH candidate files may need a specialized converter before this
    normalized form is produced.
    """
    path = Path(path)
    cands: CandidateMap = {}
    max_node = -1
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = []
        for tok in line.replace(",", " ").split():
            try:
                parts.append(int(tok))
            except ValueError:
                pass
        if len(parts) < 2:
            continue
        i, neighs = parts[0], parts[1:]
        cands.setdefault(i, []).extend(j for j in neighs if j != i)
        max_node = max(max_node, i, *neighs)
    if n is None:
        n = max_node + 1
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
