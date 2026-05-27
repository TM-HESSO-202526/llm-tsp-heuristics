import numpy as np
import math

class TSPHeuristic:
    def __call__(self, problem, rng=None):
        if rng is None:
            rng = np.random.default_rng()

        # Initialize empty tour
        tour = np.full(problem.n, -1, dtype=int)

        # Choose a random starting city
        start_city = rng.integers(0, problem.n)
        tour[0] = start_city

        # Create a set of unvisited cities
        unvisited_cities = set(range(problem.n))
        unvisited_cities.remove(start_city)

        # Create communities based on prior
        communities = {}
        for city in unvisited_cities:
            max_prior = -1
            max_prior_neighbor = None
            for neighbor in problem.neighbors(city):
                if neighbor in unvisited_cities or neighbor == start_city:
                    prior = problem.prior(neighbor, city)
                    if prior > max_prior:
                        max_prior = prior
                        max_prior_neighbor = neighbor
            if max_prior_neighbor not in communities:
                communities[max_prior_neighbor] = []
            communities[max_prior_neighbor].append(city)

        # Perform prior-guided edge selection within communities
        selected_edges = {}
        for city, community in communities.items():
            for neighbor in community:
                prior = problem.prior(city, neighbor)
                if prior > 0:
                    if neighbor not in selected_edges:
                        selected_edges[neighbor] = []
                    selected_edges[neighbor].append((city, prior))

        # Construct tour by visiting cities in order of selected edges
        current_city = start_city
        next_city_idx = 1
        while next_city_idx < problem.n:
            max_prior = -1
            next_city = None
            for neighbor in problem.neighbors(current_city):
                if neighbor in unvisited_cities:
                    prior = problem.prior(current_city, neighbor)
                    if prior > max_prior:
                        max_prior = prior
                        next_city = neighbor
            if next_city is None:
                # If no high-priority neighbor is found, choose the closest unvisited city
                min_cost = float('inf')
                next_city = None
                for city in unvisited_cities:
                    cost = problem.edge_cost(current_city, city)
                    if cost < min_cost:
                        min_cost = cost
                        next_city = city
            tour[next_city_idx] = next_city
            unvisited_cities.remove(next_city)
            current_city = next_city
            next_city_idx += 1

        # Perform a bounded local refinement step
        for i in range(5):
            for j in range(i + 1, problem.n - 1):
                if problem.edge_cost(tour[i], tour[j]) + problem.edge_cost(tour[j + 1], tour[i + 1]) < problem.edge_cost(tour[i], tour[i + 1]) + problem.edge_cost(tour[j], tour[j + 1]):
                    tour[i + 1:j + 1] = tour[i + 1:j + 1][::-1]

        return tour

# Intended mechanism family: Community detection with prior-guided edge selection and local refinement