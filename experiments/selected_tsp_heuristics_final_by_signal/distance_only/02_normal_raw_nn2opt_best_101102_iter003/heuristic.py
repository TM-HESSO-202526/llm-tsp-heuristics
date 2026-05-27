import numpy as np

class TSPHeuristic:
    def __call__(self, problem, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        
        # Initialize the tour with a random city
        tour = [rng.integers(problem.n)]
        
        # Create a set of unvisited cities
        unvisited = set(range(problem.n))
        unvisited.remove(tour[0])
        
        while unvisited:
            # Find the nearest unvisited city to the last city in the tour
            nearest_city = min(unvisited, key=lambda city: problem.edge_cost(tour[-1], city))
            tour.append(nearest_city)
            unvisited.remove(nearest_city)
        
        # Apply 2-opt improvement
        improved = True
        while improved:
            improved = False
            for i in range(len(tour) - 1):
                for j in range(i + 1, len(tour)):
                    old_cost = problem.edge_cost(tour[i], tour[i + 1]) + problem.edge_cost(tour[j], tour[(j + 1) % len(tour)])
                    new_cost = problem.edge_cost(tour[i], tour[j]) + problem.edge_cost(tour[i + 1], tour[(j + 1) % len(tour)])
                    if new_cost < old_cost:
                        tour[i + 1:j + 1] = tour[i + 1:j + 1][::-1]
                        improved = True
        
        return np.array(tour)