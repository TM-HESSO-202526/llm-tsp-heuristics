import numpy as np
import math

class TSPHeuristic:
    def __call__(self, problem, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        
        # Randomly select a subset of cities to serve as Voronoi seeds
        num_seeds = max(2, int(math.sqrt(problem.n)))
        seeds_idx = rng.integers(0, problem.n, size=num_seeds)
        seeds_coords = problem.coords[seeds_idx]
        
        # Assign each city to the nearest seed, forming Voronoi regions
        assignments = np.argmin(np.linalg.norm(problem.coords[:, np.newaxis] - seeds_coords, axis=2), axis=1)
        
        # Construct a local path inside each region using a more efficient nearest-neighbor approach
        region_tours = []
        for i in range(num_seeds):
            region_cities = np.where(assignments == i)[0]
            if len(region_cities) == 0:  # Handle empty regions
                continue
            tour = [region_cities[0]]
            unvisited = set(region_cities[1:])
            while unvisited:
                current_city = tour[-1]
                next_city = min(unvisited, key=lambda x: problem.edge_cost(current_city, x))
                tour.append(next_city)
                unvisited.remove(next_city)
            region_tours.append(tour)
        
        # Connect region endpoints into a single Hamiltonian tour
        tour = []
        last_region_city = None
        for i in range(len(region_tours)):
            if i == 0:
                tour += region_tours[i]
                last_region_city = region_tours[i][-1]
            else:
                # Connect the last city of the previous region to the first city of the current region
                # Try to find a shorter connection between regions
                best_city = None
                best_cost = np.inf
                best_idx = None
                for j, city in enumerate(region_tours[i]):
                    cost = problem.edge_cost(last_region_city, city)
                    if cost < best_cost:
                        best_cost = cost
                        best_city = city
                        best_idx = j
                tour.append(best_city)
                tour += [city for j, city in enumerate(region_tours[i]) if j != best_idx]
                last_region_city = region_tours[i][-1]
        
        # Add any remaining cities that were not assigned to a region
        remaining_cities = set(range(problem.n)) - set(tour)
        for city in remaining_cities:
            # Find the nearest city in the tour and insert the remaining city after it
            nearest_city = min(tour, key=lambda x: problem.edge_cost(x, city))
            nearest_idx = tour.index(nearest_city)
            tour.insert(nearest_idx + 1, city)
        
        # Perform a more efficient 2-opt exchange to improve the tour
        for _ in range(max(10, int(math.log(problem.n)))):
            improved = False
            for i in range(len(tour) - 1):
                for j in range(i + 2, len(tour)):
                    cost1 = problem.edge_cost(tour[i], tour[i + 1]) + problem.edge_cost(tour[j - 1], tour[j])
                    cost2 = problem.edge_cost(tour[i], tour[j - 1]) + problem.edge_cost(tour[i + 1], tour[j])
                    if cost2 < cost1:
                        tour[i + 1:j] = tour[i + 1:j][::-1]
                        improved = True
            if not improved:
                break
        
        return tour