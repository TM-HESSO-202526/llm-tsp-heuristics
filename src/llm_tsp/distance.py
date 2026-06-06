from __future__ import annotations

import math
from typing import Any

import numpy as np


def euclidean_matrix(coords: np.ndarray, round_to_int: bool = False) -> np.ndarray:
    coords = np.asarray(coords, dtype=float)
    diff = coords[:, None, :] - coords[None, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=2))
    if round_to_int:
        d = np.rint(d).astype(float)
    return d


class LazyTSPLIBDistanceMatrix:
    """Memory-safe TSPLIB distance object for large Euclidean instances.

    It deliberately behaves like a small subset of a NumPy distance matrix:
    it exposes ``shape`` and supports scalar, row, and vectorized edge lookup
    through ``D[i, j]``.  It does *not* materialize the full n x n matrix, so it
    remains usable for pla33810/pla85900.  Code that explicitly calls
    ``np.asarray(D)`` will still fail or become infeasible; that is useful in the
    final evaluation because it exposes non-scalable generated logic instead of
    silently allocating tens of GB per job.
    """

    def __init__(self, coords: np.ndarray, edge_weight_type: str | None = None) -> None:
        self.coords = np.asarray(coords, dtype=float)
        self.edge_weight_type = (edge_weight_type or "").strip().upper()
        self.shape = (int(self.coords.shape[0]), int(self.coords.shape[0]))
        self.ndim = 2
        self.dtype = np.dtype(float)

    def __len__(self) -> int:
        return self.shape[0]

    def _format(self, d: np.ndarray | float) -> np.ndarray | float:
        typ = self.edge_weight_type
        if typ == "EUC_2D":
            return np.rint(d).astype(float) if isinstance(d, np.ndarray) else float(round(float(d)))
        if typ == "CEIL_2D":
            return np.ceil(d).astype(float) if isinstance(d, np.ndarray) else float(math.ceil(float(d)))
        return d.astype(float) if isinstance(d, np.ndarray) else float(d)

    def edge(self, i: int, j: int) -> float:
        a = self.coords[int(i)]
        b = self.coords[int(j)]
        d = float(np.linalg.norm(a - b))
        return float(self._format(d))

    def row(self, i: int) -> np.ndarray:
        diff = self.coords - self.coords[int(i)]
        d = np.sqrt(np.sum(diff * diff, axis=1))
        return np.asarray(self._format(d), dtype=float)

    def edges(self, i: Any, j: Any) -> np.ndarray | float:
        ii = np.asarray(i)
        jj = np.asarray(j)
        if ii.ndim == 0 and jj.ndim == 0:
            return self.edge(int(ii), int(jj))
        a = self.coords[ii.astype(int)]
        b = self.coords[jj.astype(int)]
        d = np.sqrt(np.sum((a - b) * (a - b), axis=-1))
        return np.asarray(self._format(d), dtype=float)

    def __getitem__(self, key: Any) -> np.ndarray | float:
        # D[i] -> full row, for old notebook-style nearest-neighbor code.
        if isinstance(key, (int, np.integer)):
            return self.row(int(key))
        if not isinstance(key, tuple) or len(key) != 2:
            raise TypeError("LazyTSPLIBDistanceMatrix expects D[i] or D[i, j] indexing")
        i, j = key
        if isinstance(i, (int, np.integer)) and isinstance(j, slice):
            if j == slice(None):
                return self.row(int(i))
        if isinstance(i, slice) and i == slice(None) and isinstance(j, (int, np.integer)):
            return self.row(int(j))
        return self.edges(i, j)

    def copy(self):
        # Some old generated code does D[row].copy(); support row copy through
        # explicit indexing only.  A full matrix copy is intentionally refused.
        raise MemoryError("Refusing to materialize a full lazy TSPLIB distance matrix")



def tsplib_distance_matrix(coords: np.ndarray, edge_weight_type: str | None = None) -> np.ndarray:
    """Build the dense TSPLIB-style distance matrix for moderate instances."""
    coords = np.asarray(coords, dtype=float)
    diff = coords[:, None, :] - coords[None, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=2))
    typ = (edge_weight_type or "").strip().upper()
    if typ == "EUC_2D":
        return np.rint(d).astype(float)
    if typ == "CEIL_2D":
        return np.ceil(d).astype(float)
    return d.astype(float)


def make_tsplib_distance(coords: np.ndarray, edge_weight_type: str | None = None, *, dense_threshold: int = 20000):
    """Return a dense matrix for moderate n and a lazy object for large n."""
    n = int(np.asarray(coords).shape[0])
    if dense_threshold and n > int(dense_threshold):
        return LazyTSPLIBDistanceMatrix(coords, edge_weight_type)
    return tsplib_distance_matrix(coords, edge_weight_type)


def tsplib_edge_cost(coords: np.ndarray, i: int, j: int, edge_weight_type: str | None = None) -> float:
    d = float(np.linalg.norm(np.asarray(coords, dtype=float)[int(i)] - np.asarray(coords, dtype=float)[int(j)]))
    typ = (edge_weight_type or "").strip().upper()
    if typ == "EUC_2D":
        return float(round(d))
    if typ == "CEIL_2D":
        return float(math.ceil(d))
    return d


def tour_cost_from_matrix(tour: list[int] | np.ndarray, dist: Any) -> float:
    t = np.asarray(tour, dtype=int)
    if t.ndim != 1:
        raise ValueError("tour must be one-dimensional")
    if len(t) == 0:
        return 0.0
    nxt = np.roll(t, -1)
    # Works both for dense ndarray and LazyTSPLIBDistanceMatrix.
    return float(np.asarray(dist[t, nxt], dtype=float).sum())


def validate_tour(tour: list[int] | np.ndarray, n: int) -> None:
    t = list(map(int, tour))
    if len(t) != n:
        raise ValueError(f"tour length {len(t)} != n={n}")
    if set(t) != set(range(n)):
        raise ValueError("tour is not a valid permutation of 0..n-1")
