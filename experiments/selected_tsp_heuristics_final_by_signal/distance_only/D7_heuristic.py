import numpy as np


def _construct_tour(D, start_node):
    n = D.shape[0]
    tour = [int(start_node)]
    visited = np.zeros(n, dtype=bool)
    visited[int(start_node)] = True
    for _ in range(n - 1):
        current = tour[-1]
        best = None
        best_dist = float('inf')
        for j in range(n):
            if visited[j]:
                continue
            dist = float(D[current, j])
            if dist < best_dist:
                best_dist = dist
                best = j
        tour.append(int(best))
        visited[int(best)] = True
    return np.asarray(tour, dtype=int)


class TSPHeuristic:
    def __call__(self, problem, rng=None):
        D = problem.distance_matrix_for_evaluator()
        rng = np.random.default_rng() if rng is None else rng
        start_node = int(rng.integers(problem.n))
        return _construct_tour(D, start_node)
