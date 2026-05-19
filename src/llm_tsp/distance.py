from __future__ import annotations

import numpy as np


def euclidean_matrix(coords: np.ndarray, round_to_int: bool = False) -> np.ndarray:
    coords = np.asarray(coords, dtype=float)
    diff = coords[:, None, :] - coords[None, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=2))
    if round_to_int:
        d = np.rint(d).astype(float)
    return d


def tsplib_distance_matrix(coords: np.ndarray, edge_weight_type: str | None = None) -> np.ndarray:
    """Build the TSPLIB-style distance matrix for the coordinate instances used here.

    Supported/expected types:
    - EUC_2D: nearest integer Euclidean distance.
    - CEIL_2D: ceiling Euclidean distance.
    - default/unknown: raw Euclidean distance.

    The large 1k+ instances in the thesis suite are coordinate-based TSPLIB
    instances, so this lightweight implementation is enough for the public repo.
    """
    coords = np.asarray(coords, dtype=float)
    diff = coords[:, None, :] - coords[None, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=2))
    typ = (edge_weight_type or "").strip().upper()
    if typ == "EUC_2D":
        return np.rint(d).astype(float)
    if typ == "CEIL_2D":
        return np.ceil(d).astype(float)
    return d.astype(float)


def tour_cost_from_matrix(tour: list[int] | np.ndarray, dist: np.ndarray) -> float:
    t = np.asarray(tour, dtype=int)
    if t.ndim != 1:
        raise ValueError("tour must be one-dimensional")
    if len(t) == 0:
        return 0.0
    nxt = np.roll(t, -1)
    return float(dist[t, nxt].sum())


def validate_tour(tour: list[int] | np.ndarray, n: int) -> None:
    t = list(map(int, tour))
    if len(t) != n:
        raise ValueError(f"tour length {len(t)} != n={n}")
    if set(t) != set(range(n)):
        raise ValueError("tour is not a valid permutation of 0..n-1")
