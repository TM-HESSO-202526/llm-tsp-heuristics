import numpy as np
from llm_tsp.distance import euclidean_matrix
from llm_tsp.sparse_problem import SparseTSPProblem
from llm_tsp.evaluation import evaluate_code_on_problem


def test_evaluate_simple_class_candidate():
    coords = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    problem = SparseTSPProblem(coords=coords, dist=euclidean_matrix(coords))
    code = """
class TSPHeuristic:
    def __call__(self, problem, rng=None):
        return list(range(problem.n))
"""
    res = evaluate_code_on_problem(code, problem, optimum=4.0)
    assert res.valid
    assert abs(res.cost - 4.0) < 1e-9


def test_function_interface_is_rejected():
    coords = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    problem = SparseTSPProblem(coords=coords, dist=euclidean_matrix(coords))
    code = """
def construct_tour(problem, rng=None):
    return list(range(problem.n))
"""
    res = evaluate_code_on_problem(code, problem, optimum=4.0)
    assert not res.valid
    assert "TSPHeuristic" in str(res.error)


def test_non_candidate_final_edges_are_allowed():
    coords = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    problem = SparseTSPProblem(
        coords=coords,
        dist=euclidean_matrix(coords),
        candidate_neighbors={0: [1], 1: [0], 2: [3], 3: [2]},
    )
    code = """
class TSPHeuristic:
    def __call__(self, problem, rng=None):
        # This tour uses non-candidate edges 1-2 and 3-0.
        return [0, 1, 2, 3]
"""
    res = evaluate_code_on_problem(code, problem, optimum=4.0)
    assert res.valid
    assert res.uses_only_candidates is False
    assert abs(res.cost - 4.0) < 1e-9


def test_candidate_metrics_are_none_when_candidates_disabled():
    coords = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    problem = SparseTSPProblem(coords=coords, dist=euclidean_matrix(coords))
    code = """
class TSPHeuristic:
    def __call__(self, problem, rng=None):
        return list(range(problem.n))
"""
    res = evaluate_code_on_problem(code, problem, optimum=4.0)
    assert res.valid
    assert res.uses_only_candidates is None
    assert res.candidate_edge_count is None
    assert res.total_edges is None
    assert res.candidate_edge_share is None


def test_neighbors_unavailable_when_candidates_disabled():
    coords = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    problem = SparseTSPProblem(coords=coords, dist=euclidean_matrix(coords))
    code = """
class TSPHeuristic:
    def __call__(self, problem, rng=None):
        return [0] + list(problem.neighbors(0))
"""
    res = evaluate_code_on_problem(code, problem, optimum=4.0)
    assert not res.valid
    assert "problem.neighbors(i) is unavailable" in str(res.error)


def test_prior_unavailable_when_edge_prior_disabled():
    coords = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    problem = SparseTSPProblem(coords=coords, dist=euclidean_matrix(coords))
    code = """
class TSPHeuristic:
    def __call__(self, problem, rng=None):
        _ = problem.prior(0, 1)
        return list(range(problem.n))
"""
    res = evaluate_code_on_problem(code, problem, optimum=4.0)
    assert not res.valid
    assert "problem.prior(i, j) is unavailable" in str(res.error)


def test_neighbors_and_candidate_metrics_work_when_candidates_enabled():
    coords = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    problem = SparseTSPProblem(
        coords=coords,
        dist=euclidean_matrix(coords),
        candidate_neighbors={0: [1], 1: [2], 2: [3], 3: [0]},
    )
    code = """
class TSPHeuristic:
    def __call__(self, problem, rng=None):
        return [0, 1, 2, 3]
"""
    res = evaluate_code_on_problem(code, problem, optimum=4.0)
    assert res.valid
    assert res.uses_only_candidates is True
    assert res.candidate_edge_count == 4
    assert res.total_edges == 4
    assert res.candidate_edge_share == 1.0
