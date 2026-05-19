from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .candidate_sets import CandidateMap, PriorMap


@dataclass
class SparseTSPProblem:
    coords: np.ndarray
    dist: np.ndarray
    candidate_neighbors: CandidateMap | None = None
    prior_map: PriorMap | None = None
    restrict_edge_cost_to_candidates: bool = False

    @property
    def n(self) -> int:
        return int(self.dist.shape[0])

    def neighbors(self, i: int) -> list[int]:
        if self.candidate_neighbors is None:
            return [j for j in range(self.n) if j != i]
        return list(self.candidate_neighbors.get(int(i), []))

    def is_candidate_edge(self, i: int, j: int) -> bool:
        if self.candidate_neighbors is None:
            return True
        return int(j) in set(self.candidate_neighbors.get(int(i), []))

    def edge_cost(self, i: int, j: int) -> float:
        i, j = int(i), int(j)
        if i == j:
            return 0.0
        if self.restrict_edge_cost_to_candidates and not self.is_candidate_edge(i, j):
            raise ValueError(f"edge_cost({i},{j}) is only available for candidate edges")
        return float(self.dist[i, j])

    def full_edge_cost(self, i: int, j: int) -> float:
        return float(self.dist[int(i), int(j)])

    def prior(self, i: int, j: int) -> float:
        if not self.prior_map:
            return 0.0
        a, b = sorted((int(i), int(j)))
        return float(self.prior_map.get((a, b), 0.0))

    def tour_uses_only_candidates(self, tour: list[int] | np.ndarray) -> bool:
        t = list(map(int, tour))
        return all(self.is_candidate_edge(t[k], t[(k + 1) % len(t)]) for k in range(len(t)))
