from __future__ import annotations

import numpy as np

from .candidate_sets import CandidateMap, PriorMap


class SparseTSPProblem:
    """TSP problem object exposed to generated heuristics.

    This mirrors the historical TSP notebooks used in the thesis:

    * POPMUSIC/LKH candidate lists are exposed as sparse guidance through
      ``problem.neighbors(i)`` and ``problem.cand``.
    * The generated heuristic is not given a public dense distance matrix.
      It can query distances through ``problem.edge_cost(i, j)``.
    * Final tours are normal TSP permutations and are evaluated externally on
      the true full TSPLIB distance matrix. Non-candidate final edges are not
      rejected; they are simply part of the returned tour cost.
    """

    def __init__(
        self,
        coords: np.ndarray,
        dist: np.ndarray,
        candidate_neighbors: CandidateMap | None = None,
        prior_map: PriorMap | None = None,
    ) -> None:
        self.coords = np.asarray(coords)
        self._dist = np.asarray(dist)
        self.candidate_neighbors = candidate_neighbors
        self.prior_map = prior_map

    @property
    def n(self) -> int:
        return int(self._dist.shape[0])

    @property
    def cand(self) -> list[np.ndarray] | None:
        """Historical notebook-compatible candidate-list view."""
        if self.candidate_neighbors is None:
            return None
        return [
            np.asarray(self.candidate_neighbors.get(i, []), dtype=np.int64)
            for i in range(self.n)
        ]

    def neighbors(self, i: int) -> list[int]:
        if self.candidate_neighbors is None:
            return [j for j in range(self.n) if j != i]
        return list(self.candidate_neighbors.get(int(i), []))

    def is_candidate_edge(self, i: int, j: int) -> bool:
        if self.candidate_neighbors is None:
            return True
        return int(j) in set(self.candidate_neighbors.get(int(i), []))

    def edge_cost(self, i: int, j: int) -> float:
        """True full TSPLIB edge cost queried through an oracle-style method.

        Candidate mode does not make non-candidate edges illegal. Heuristics
        should use ``neighbors(i)`` as sparse POPMUSIC guidance, but they can
        still query individual edge costs when constructing or repairing a tour.
        The full dense matrix itself is deliberately not exposed as ``problem.dist``.
        """
        return float(self._dist[int(i), int(j)])

    def full_edge_cost(self, i: int, j: int) -> float:
        return self.edge_cost(i, j)

    def distance_matrix_for_evaluator(self) -> np.ndarray:
        """Internal evaluator access to the true full distance matrix."""
        return self._dist

    def prior(self, i: int, j: int) -> float:
        if not self.prior_map:
            return 0.0
        a, b = sorted((int(i), int(j)))
        return float(self.prior_map.get((a, b), 0.0))

    def tour_candidate_edge_count(self, tour: list[int] | np.ndarray) -> tuple[int, int]:
        t = list(map(int, tour))
        total = len(t)
        if not total:
            return 0, 0
        count = sum(1 for k in range(total) if self.is_candidate_edge(t[k], t[(k + 1) % total]))
        return int(count), int(total)

    def tour_uses_only_candidates(self, tour: list[int] | np.ndarray) -> bool:
        count, total = self.tour_candidate_edge_count(tour)
        return bool(total and count == total)
