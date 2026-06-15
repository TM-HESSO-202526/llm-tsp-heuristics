import numpy as np

class TSPHeuristic:
    def __call__(self, problem, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        
        # Number of cities
        n = problem.n
        
        # City coordinates
        coords = problem.coords
        
        # Dimensionality of the coordinates
        dim = coords.shape[1]
        
        # Project coordinates onto a random line
        num_projections = 5
        lines = rng.standard_normal((num_projections, dim))
        lines = lines / np.linalg.norm(lines, axis=1, keepdims=True)
        projections = np.dot(coords, lines.T)
        
        # Sort cities by their projections and select the best split
        best_split = None
        best_cost = np.inf
        for i in range(num_projections):
            sorted_indices = np.argsort(projections[:, i])
            mid = n // 2
            left_half = sorted_indices[:mid]
            right_half = sorted_indices[mid:]
            left_path = self.build_path(problem, left_half, rng)
            right_path = self.build_path(problem, right_half, rng)
            tour = np.concatenate((left_path, right_path))
            cost = self.tour_cost(problem, tour)
            if cost < best_cost:
                best_cost = cost
                best_split = tour
        
        # Local refining
        best_split = self.local_refine(problem, best_split, rng)
        
        return best_split
    
    def build_path(self, problem, cities, rng):
        # Start with a random city
        path = [rng.choice(cities)]
        
        # Greedily add the closest unvisited city
        unvisited_cities = set(cities)
        unvisited_cities.remove(path[0])
        
        while unvisited_cities:
            current_city = path[-1]
            closest_city = min(unvisited_cities, key=lambda city: problem.edge_cost(current_city, city))
            path.append(closest_city)
            unvisited_cities.remove(closest_city)
        
        return np.array(path)
    
    def local_refine(self, problem, tour, rng):
        # Local 2-opt refinement
        for _ in range(len(tour) // 2):
            i = rng.integers(0, len(tour) - 1)
            j = rng.integers(0, len(tour) - 1)
            if i > j:
                i, j = j, i
            if i < j - 1:
                new_tour = np.concatenate((tour[:i], tour[i:j][::-1], tour[j:]))
                if self.tour_cost(problem, new_tour) < self.tour_cost(problem, tour):
                    tour = new_tour
        
        return tour
    
    def tour_cost(self, problem, tour):
        cost = 0
        for i in range(len(tour) - 1):
            cost += problem.edge_cost(tour[i], tour[i + 1])
        return cost