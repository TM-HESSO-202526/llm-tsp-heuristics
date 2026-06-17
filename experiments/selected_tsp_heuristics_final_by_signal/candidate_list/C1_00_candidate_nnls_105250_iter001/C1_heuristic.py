import numpy as np

class TSPHeuristic:
    def __call__(self, problem, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        
        # Start with a random city
        current_city = rng.integers(problem.n)
        tour = [current_city]
        visited = set([current_city])
        
        # Repeatedly choose the closest unvisited city
        for _ in range(problem.n - 1):
            # Get the neighbors of the current city
            neighbors = problem.neighbors(current_city)
            unvisited_neighbors = [neighbor for neighbor in neighbors if neighbor not in visited]
            
            # If there are no unvisited neighbors, choose any unvisited city
            if not unvisited_neighbors:
                unvisited_cities = [city for city in range(problem.n) if city not in visited]
                next_city = rng.choice(unvisited_cities)
            else:
                # Choose the closest unvisited neighbor
                costs = [problem.edge_cost(current_city, neighbor) for neighbor in unvisited_neighbors]
                next_city = unvisited_neighbors[np.argmin(costs)]
            
            # Add the next city to the tour and mark it as visited
            tour.append(next_city)
            visited.add(next_city)
            current_city = next_city
        
        return np.array(tour)