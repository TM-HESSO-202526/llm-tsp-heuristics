from __future__ import annotations

from typing import Any

import numpy as np

from .candidate_sets import CandidateMap, PriorMap
from .distance import LazyTSPLIBDistanceMatrix, tsplib_edge_cost


class SparseTSPProblem:
    """TSP problem object exposed to generated heuristics.

    The object can use either a dense distance matrix or a memory-safe lazy
    distance object.  The lazy path is essential for very large TSPLIB instances
    such as pla33810 and pla85900, where materializing the full n x n matrix is
    not compatible with running many jobs concurrently.
    """

    def __init__(
        self,
        coords: np.ndarray,
        dist: Any | None,
        candidate_neighbors: CandidateMap | None = None,
        prior_map: PriorMap | None = None,
        edge_weight_type: str | None = None,
    ) -> None:
        self.coords = np.asarray(coords)
        self._dist = dist
        self.edge_weight_type = (edge_weight_type or "").strip().upper()
        self.candidate_neighbors = candidate_neighbors
        self.prior_map = prior_map

    @property
    def n(self) -> int:
        if self._dist is not None:
            return int(self._dist.shape[0])
        return int(self.coords.shape[0])

    @property
    def has_dense_distance_matrix(self) -> bool:
        return isinstance(self._dist, np.ndarray)

    @property
    def has_candidate_neighbors(self) -> bool:
        return self.candidate_neighbors is not None

    @property
    def has_edge_prior(self) -> bool:
        return self.prior_map is not None

    @property
    def cand(self) -> list[np.ndarray] | None:
        if self.candidate_neighbors is None:
            raise AttributeError(
                "problem.cand is unavailable because POPMUSIC candidate mode is disabled; "
                "use problem.edge_cost(i, j) and problem.coords instead."
            )
        return [np.asarray(self.candidate_neighbors.get(i, []), dtype=np.int64) for i in range(self.n)]

    def neighbors(self, i: int) -> list[int]:
        if self.candidate_neighbors is None:
            raise AttributeError(
                "problem.neighbors(i) is unavailable because POPMUSIC candidate mode is disabled; "
                "use bounded problem.edge_cost(i, j) queries and problem.coords instead."
            )
        return list(self.candidate_neighbors.get(int(i), []))

    def is_candidate_edge(self, i: int, j: int) -> bool | None:
        if self.candidate_neighbors is None:
            return None
        return int(j) in set(self.candidate_neighbors.get(int(i), []))

    def edge_cost(self, i: int, j: int) -> float:
        if self._dist is not None:
            return float(self._dist[int(i), int(j)])
        return tsplib_edge_cost(self.coords, int(i), int(j), self.edge_weight_type)

    def full_edge_cost(self, i: int, j: int) -> float:
        return self.edge_cost(i, j)

    def distance_matrix_for_evaluator(self):
        if self._dist is not None:
            return self._dist
        return LazyTSPLIBDistanceMatrix(self.coords, self.edge_weight_type)

    def prior(self, i: int, j: int) -> float:
        if self.prior_map is None:
            raise AttributeError("problem.prior(i, j) is unavailable because POPMUSIC edge-prior mode is disabled.")
        a, b = sorted((int(i), int(j)))
        return float(self.prior_map.get((a, b), 0.0))

    def tour_candidate_edge_count(self, tour: list[int] | np.ndarray) -> tuple[int | None, int | None]:
        if self.candidate_neighbors is None:
            return None, None
        t = list(map(int, tour))
        total = len(t)
        if not total:
            return 0, 0
        count = sum(1 for k in range(total) if self.is_candidate_edge(t[k], t[(k + 1) % total]))
        return int(count), int(total)

    def tour_uses_only_candidates(self, tour: list[int] | np.ndarray) -> bool | None:
        count, total = self.tour_candidate_edge_count(tour)
        if count is None or total is None:
            return None
        return bool(total and count == total)
