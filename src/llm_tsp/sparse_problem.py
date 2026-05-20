from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .candidate_sets import CandidateMap, PriorMap


@dataclass
class SparseTSPProblem:
    """TSP problem object exposed to generated heuristics.

    Candidate mode is guidance, not a hard feasibility restriction. This mirrors
    the historical TSP notebooks: POPMUSIC/LKH candidate lists are exposed through
    ``problem.neighbors(i)``/``problem.cand``, but a returned tour is evaluated as a
    normal TSP tour on the true full TSPLIB distance matrix.
    """

    coords: np.ndarray
    dist: np.ndarray
    candidate_neighbors: CandidateMap | None = None
    prior_map: PriorMap | None = None

    @property
    def n(self) -> int:
        return int(self.dist.shape[0])

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
        """True full TSPLIB edge cost.

        Candidate mode does not make non-candidate edges illegal. Heuristics
        should use ``neighbors(i)`` as sparse guidance, but they can still query
        distances for bounded fallbacks and must return a normal full tour.
        """
        return float(self.dist[int(i), int(j)])

    def full_edge_cost(self, i: int, j: int) -> float:
        return self.edge_cost(i, j)

    def prior(self, i: int, j: int) -> float:
        if not self.prior_map:
            return 0.0
        a, b = sorted((int(i), int(j)))
        return float(self.prior_map.get((a, b), 0.0))

    def tour_uses_only_candidates(self, tour: list[int] | np.ndarray) -> bool:
        t = list(map(int, tour))
        return all(self.is_candidate_edge(t[k], t[(k + 1) % len(t)]) for k in range(len(t)))
