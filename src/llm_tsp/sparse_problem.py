from __future__ import annotations

import numpy as np

from .candidate_sets import CandidateMap, PriorMap


class SparseTSPProblem:
    """TSP problem object exposed to generated heuristics.

    The public interface is intentionally conditional:

    * ``problem.edge_cost(i, j)`` and ``problem.coords`` are always available.
    * ``problem.neighbors(i)`` and ``problem.cand`` are available only when
      POPMUSIC/LKH candidate mode is enabled.
    * ``problem.prior(i, j)`` is available only when POPMUSIC edge-prior mode is
      enabled.

    Final tours are normal TSP permutations and are evaluated externally on the
    true full TSPLIB distance matrix. Candidate lists are guidance only when
    enabled; non-candidate final edges are not rejected.
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
    def has_candidate_neighbors(self) -> bool:
        return self.candidate_neighbors is not None

    @property
    def has_edge_prior(self) -> bool:
        return self.prior_map is not None

    @property
    def cand(self) -> list[np.ndarray] | None:
        """Historical notebook-compatible candidate-list view.

        This property is intentionally unavailable when POPMUSIC candidate mode
        is disabled, so generated code cannot silently treat the full graph as a
        candidate list.
        """
        if self.candidate_neighbors is None:
            raise AttributeError(
                "problem.cand is unavailable because POPMUSIC candidate mode is disabled; "
                "use problem.edge_cost(i, j) and problem.coords instead."
            )
        return [
            np.asarray(self.candidate_neighbors.get(i, []), dtype=np.int64)
            for i in range(self.n)
        ]

    def neighbors(self, i: int) -> list[int]:
        """Return sparse POPMUSIC/LKH candidate neighbors for city i.

        When candidate mode is disabled, this method raises instead of returning
        all cities. This prevents generated heuristics from accidentally using
        ``problem.neighbors(i)`` as a hidden dense-neighborhood interface.
        """
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
        """True full TSPLIB edge cost queried through an oracle-style method.

        Candidate mode does not make non-candidate edges illegal. Heuristics
        should use ``neighbors(i)`` as sparse POPMUSIC guidance only when that
        mode is enabled, but they can still query individual edge costs when
        constructing or repairing a tour. The full dense matrix itself is
        deliberately not exposed as ``problem.dist``.
        """
        return float(self._dist[int(i), int(j)])

    def full_edge_cost(self, i: int, j: int) -> float:
        return self.edge_cost(i, j)

    def distance_matrix_for_evaluator(self) -> np.ndarray:
        """Internal evaluator access to the true full distance matrix."""
        return self._dist

    def prior(self, i: int, j: int) -> float:
        """Return POPMUSIC/LKH edge-support prior for one edge.

        When edge-prior mode is disabled, this method raises instead of silently
        returning 0.0. That keeps the executed interface aligned with the prompt.
        """
        if self.prior_map is None:
            raise AttributeError(
                "problem.prior(i, j) is unavailable because POPMUSIC edge-prior mode is disabled."
            )
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
