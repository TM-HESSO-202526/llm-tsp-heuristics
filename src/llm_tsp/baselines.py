from __future__ import annotations

import numpy as np
from .sparse_problem import SparseTSPProblem


def nearest_neighbor(problem: SparseTSPProblem, start: int = 0) -> list[int]:
    n = problem.n
    unvisited = set(range(n))
    tour = [int(start)]
    unvisited.remove(int(start))
    while unvisited:
        i = tour[-1]
        if problem.candidate_neighbors is None:
            choices = list(unvisited)
        else:
            choices = [j for j in problem.candidate_neighbors.get(i, []) if j in unvisited]
            if not choices:
                choices = list(unvisited)
        j = min(choices, key=lambda x: problem.full_edge_cost(i, x))
        tour.append(int(j))
        unvisited.remove(int(j))
    return tour


def prior_greedy(problem: SparseTSPProblem, start: int = 0, alpha: float = 0.7) -> list[int]:
    n = problem.n
    unvisited = set(range(n))
    tour = [int(start)]
    unvisited.remove(int(start))
    max_d = float(np.max(problem.distance_matrix_for_evaluator())) or 1.0
    while unvisited:
        i = tour[-1]
        if problem.candidate_neighbors is None:
            choices = list(unvisited)
        else:
            choices = [j for j in problem.candidate_neighbors.get(i, []) if j in unvisited]
            if not choices:
                choices = list(unvisited)
        def score(j: int) -> float:
            distance_term = problem.full_edge_cost(i, j) / max_d
            if problem.prior_map is None:
                prior_value = 0.0
            else:
                a, b = sorted((int(i), int(j)))
                prior_value = float(problem.prior_map.get((a, b), 0.0))
            prior_term = 1.0 - prior_value
            return alpha * distance_term + (1.0 - alpha) * prior_term
        j = min(choices, key=score)
        tour.append(int(j))
        unvisited.remove(int(j))
    return tour
