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
    n = len(prior_rows)
    tour = [int(start_node)]
    unvisited = set(range(n))
    unvisited.remove(int(start_node))
    while unvisited:
        current = tour[-1]
        best = None
        best_score = float('-inf')
        best_dist = float('inf')
        for j in unvisited:
            score = float(prior_rows[current].get(j, 0.0))
            dist = float(D[current, j])
            if score > best_score or (score == best_score and dist < best_dist):
                best_score = score
                best_dist = dist
                best = j
        tour.append(int(best))
        unvisited.remove(int(best))
    return np.asarray(tour, dtype=int)


class TSPHeuristic:
    def __call__(self, problem, rng=None):
        D = problem.distance_matrix_for_evaluator()
        rng = np.random.default_rng() if rng is None else rng
        start_node = int(rng.integers(problem.n))
        return _construct_tour(D, _prior_rows(problem), start_node)
