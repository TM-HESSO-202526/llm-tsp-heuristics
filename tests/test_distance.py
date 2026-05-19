import numpy as np
from llm_tsp.distance import euclidean_matrix, validate_tour, tour_cost_from_matrix


def test_tour_cost_square():
    coords = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    dist = euclidean_matrix(coords)
    tour = [0, 1, 2, 3]
    validate_tour(tour, 4)
    assert abs(tour_cost_from_matrix(tour, dist) - 4.0) < 1e-9
