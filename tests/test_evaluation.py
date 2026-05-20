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
