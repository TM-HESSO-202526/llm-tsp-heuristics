from __future__ import annotations

import math
from typing import Iterable

import numpy as np


# Scalable distance-only baselines for large Euclidean TSPLIB instances.
# They deliberately avoid full n x n matrices and exact insertion / full 2-opt.
# Cost queries are made through problem.edge_cost(i, j), and ordering rules use
# only coordinates.  The target is practical scaling to pla33810/pla85900.


def _coords(problem) -> np.ndarray:
    return np.asarray(problem.coords, dtype=float)


def _edge(problem, i: int, j: int) -> float:
    return float(problem.edge_cost(int(i), int(j)))


def _tour_cost(problem, tour: np.ndarray) -> float:
    t = np.asarray(tour, dtype=np.int64)
    return float(sum(_edge(problem, int(t[i]), int(t[(i + 1) % len(t)])) for i in range(len(t))))


def _normalize_key(v: np.ndarray, bits: int = 21) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    lo = float(np.min(v))
    hi = float(np.max(v))
    if hi <= lo:
        return np.zeros_like(v, dtype=np.uint64)
    scale = (1 << bits) - 1
    return np.asarray(np.clip(np.rint((v - lo) / (hi - lo) * scale), 0, scale), dtype=np.uint64)


def _part1by1(n: np.ndarray) -> np.ndarray:
    x = n.astype(np.uint64) & np.uint64(0x1fffff)
    x = (x | (x << np.uint64(32))) & np.uint64(0x001f00000000ffff)
    x = (x | (x << np.uint64(16))) & np.uint64(0x001f0000ff0000ff)
    x = (x | (x << np.uint64(8))) & np.uint64(0x100f00f00f00f00f)
    x = (x | (x << np.uint64(4))) & np.uint64(0x10c30c30c30c30c3)
    x = (x | (x << np.uint64(2))) & np.uint64(0x1249249249249249)
    return x


def _morton_order(coords: np.ndarray) -> np.ndarray:
    # Use first two coordinates. The TSP instances here are planar TSPLIB.
    x = _normalize_key(coords[:, 0])
    y = _normalize_key(coords[:, 1])
    # 2D Morton code with bit interleaving. The masks are 3D-safe but still
    # produce a stable spatial ordering for two coordinates.
    code = _part1by1(x) | (_part1by1(y) << np.uint64(1))
    return np.argsort(code, kind="mergesort").astype(np.int64)


def _pca_projection_order(coords: np.ndarray) -> np.ndarray:
    centered = coords - coords.mean(axis=0, keepdims=True)
    if coords.shape[1] == 1:
        score = centered[:, 0]
    else:
        # 2x2 covariance for TSPLIB planar instances; cheap and stable.
        _, _, vh = np.linalg.svd(centered[: min(len(centered), 200000)], full_matrices=False)
        direction = vh[0]
        score = centered @ direction
    return np.argsort(score, kind="mergesort").astype(np.int64)


def _x_axis_order(coords: np.ndarray) -> np.ndarray:
    if coords.shape[1] >= 2:
        return np.lexsort((coords[:, 1], coords[:, 0])).astype(np.int64)
    return np.argsort(coords[:, 0], kind="mergesort").astype(np.int64)


def _angular_order(coords: np.ndarray) -> np.ndarray:
    c = coords.mean(axis=0)
    x = coords[:, 0] - c[0]
    y = coords[:, 1] - c[1] if coords.shape[1] >= 2 else np.zeros(len(coords))
    angle = np.arctan2(y, x)
    radius = np.sqrt(x * x + y * y)
    return np.lexsort((radius, angle)).astype(np.int64)


def _grid_serpentine_order(coords: np.ndarray) -> np.ndarray:
    n = len(coords)
    if n <= 1:
        return np.arange(n, dtype=np.int64)
    gx = max(2, int(math.sqrt(n)))
    xkey = _normalize_key(coords[:, 0], bits=20).astype(np.int64)
    ykey = _normalize_key(coords[:, 1] if coords.shape[1] >= 2 else np.zeros(n), bits=20).astype(np.int64)
    xbin = np.minimum(gx - 1, (xkey * gx) // (1 << 20))
    rows = []
    for b in range(gx):
        idx = np.flatnonzero(xbin == b)
        if len(idx) == 0:
            continue
        idx = idx[np.argsort(ykey[idx], kind="mergesort")]
        if b % 2:
            idx = idx[::-1]
        rows.append(idx)
    return np.concatenate(rows).astype(np.int64) if rows else np.arange(n, dtype=np.int64)


def _recursive_bisection_order(coords: np.ndarray, bucket_size: int = 256) -> np.ndarray:
    n = len(coords)
    order: list[np.ndarray] = []
    stack = [np.arange(n, dtype=np.int64)]
    while stack:
        idx = stack.pop()
        if len(idx) <= bucket_size:
            sub = coords[idx]
            axis = int(np.argmax(np.ptp(sub, axis=0)))
            order.append(idx[np.argsort(sub[:, axis], kind="mergesort")])
            continue
        sub = coords[idx]
        axis = int(np.argmax(np.ptp(sub, axis=0)))
        sorted_idx = idx[np.argsort(sub[:, axis], kind="mergesort")]
        mid = len(sorted_idx) // 2
        # Push right first so left is processed first.
        stack.append(sorted_idx[mid:])
        stack.append(sorted_idx[:mid])
    return np.concatenate(order).astype(np.int64)


def _kdtree_nn(problem, starts: Iterable[int], candidate_k: int = 64, sample_fallback: int = 128) -> np.ndarray:
    coords = _coords(problem)
    n = len(coords)
    if n <= 1:
        return np.arange(n, dtype=np.int64)
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(coords)
    except Exception:
        # Deterministic spatial-order fallback.
        return _morton_order(coords)

    best_tour = None
    best_cost = math.inf
    starts = list(starts)
    for start in starts:
        visited = np.zeros(n, dtype=bool)
        tour = np.empty(n, dtype=np.int64)
        cur = int(start) % n
        for pos in range(n):
            tour[pos] = cur
            visited[cur] = True
            if pos == n - 1:
                break
            next_city = -1
            # Query a bounded number of geometric neighbours. Increase only a
            # little if all returned neighbours were already visited.
            k = min(n, candidate_k)
            while k <= min(n, 512):
                _dist, idx = tree.query(coords[cur], k=k)
                idx = np.atleast_1d(idx).astype(np.int64)
                avail = idx[~visited[idx]]
                if len(avail):
                    # Pick true shortest among the small candidate set.
                    next_city = int(min(avail, key=lambda j: _edge(problem, cur, int(j))))
                    break
                if k == n:
                    break
                k = min(n, k * 2)
            if next_city < 0:
                # Bounded fallback: sample remaining vertices rather than scan
                # all unvisited vertices.
                rem = np.flatnonzero(~visited)
                if len(rem) > sample_fallback:
                    # Deterministic slice is intentional for reproducibility.
                    step = max(1, len(rem) // sample_fallback)
                    cand = rem[::step][:sample_fallback]
                else:
                    cand = rem
                next_city = int(min(cand, key=lambda j: _edge(problem, cur, int(j))))
            cur = next_city
        cost = _tour_cost(problem, tour)
        if cost < best_cost:
            best_cost = cost
            best_tour = tour.copy()
    assert best_tour is not None
    return best_tour


def _bounded_window_2opt(problem, tour: np.ndarray, passes: int = 2, window: int = 48) -> np.ndarray:
    t = np.asarray(tour, dtype=np.int64).copy()
    n = len(t)
    if n < 4:
        return t
    for _ in range(int(passes)):
        improved = False
        for i in range(n - 3):
            a = int(t[i])
            b = int(t[(i + 1) % n])
            jmax = min(n - 1, i + window)
            best_j = -1
            best_gain = 0.0
            old_ab = _edge(problem, a, b)
            for j in range(i + 2, jmax):
                if i == 0 and j == n - 1:
                    continue
                c = int(t[j])
                d = int(t[(j + 1) % n])
                gain = old_ab + _edge(problem, c, d) - _edge(problem, a, c) - _edge(problem, b, d)
                if gain > best_gain:
                    best_gain = gain
                    best_j = j
            if best_j >= 0:
                t[i + 1:best_j + 1] = t[i + 1:best_j + 1][::-1]
                improved = True
        if not improved:
            break
    return t


class KDTreeNearestNeighborFixedStart:
    def __call__(self, problem, rng=None):
        return _kdtree_nn(problem, starts=[0])


class KDTreeNearestNeighborMultistart:
    def __call__(self, problem, rng=None):
        n = int(problem.n)
        rng = rng or np.random.default_rng(0)
        base = [0, n // 4, n // 2, (3 * n) // 4]
        extra = list(map(int, rng.choice(n, size=min(4, n), replace=False)))
        return _kdtree_nn(problem, starts=base + extra)


class XAxisSweep:
    def __call__(self, problem, rng=None):
        return _x_axis_order(_coords(problem))


class PCASweep:
    def __call__(self, problem, rng=None):
        return _pca_projection_order(_coords(problem))


class AngularSweep:
    def __call__(self, problem, rng=None):
        return _angular_order(_coords(problem))


class MortonZOrder:
    def __call__(self, problem, rng=None):
        return _morton_order(_coords(problem))


class GridSerpentine:
    def __call__(self, problem, rng=None):
        return _grid_serpentine_order(_coords(problem))


class MortonBoundedLocalTwoOpt:
    def __call__(self, problem, rng=None):
        tour = _morton_order(_coords(problem))
        return _bounded_window_2opt(problem, tour, passes=2, window=48)
