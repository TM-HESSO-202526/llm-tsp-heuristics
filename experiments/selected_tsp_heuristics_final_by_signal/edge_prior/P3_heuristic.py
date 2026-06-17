import numpy as np


def _prior_rows(problem):
    rows = [dict() for _ in range(problem.n)]
    prior = getattr(problem, 'prior_map', None)
    if prior is None:
        return rows
    for (a, b), w in prior.items():
        rows[int(a)][int(b)] = float(w)
        rows[int(b)][int(a)] = float(w)
    return rows


def _construct_tour(D, prior_rows, start_node):
    n = D.shape[0]
    tour = [int(start_node)]
    visited = {int(start_node)}
    current = int(start_node)
    while len(tour) < n:
        best = None
        best_score = float('-inf')
        for j, score in prior_rows[current].items():
            if j not in visited and float(score) > best_score:
                best_score = float(score)
                best = int(j)
        if best is None:
            best_dist = float('inf')
            for j in range(n):
                if j not in visited:
                    dist = float(D[current, j])
                    if dist < best_dist:
                        best_dist = dist
                        best = j
        tour.append(int(best))
        visited.add(int(best))
        current = int(best)
    return np.asarray(tour, dtype=int)


class TSPHeuristic:
    def __call__(self, problem, rng=None):
        D = problem.distance_matrix_for_evaluator()
        rng = np.random.default_rng() if rng is None else rng
        start_node = int(rng.integers(problem.n))
        return _construct_tour(D, _prior_rows(problem), start_node)
