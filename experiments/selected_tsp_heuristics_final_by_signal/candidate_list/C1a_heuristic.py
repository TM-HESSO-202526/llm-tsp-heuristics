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
                # Choose the closest unvisited city
                costs = [problem.full_edge_cost(current_city, city) for city in unvisited_cities]
                next_city = unvisited_cities[np.argmin(costs)]
            else:
                # Choose the closest unvisited neighbor
                costs = [problem.edge_cost(current_city, neighbor) for neighbor in unvisited_neighbors]
                next_city = unvisited_neighbors[np.argmin(costs)]
            
            # Add the next city to the tour and mark it as visited
            tour.append(next_city)
            visited.add(next_city)
            current_city = next_city
        
        # Perform local search to improve the tour
        for _ in range(problem.n):
            i = rng.integers(problem.n - 1)
            j = rng.integers(problem.n - 1)
            if i > j:
                i, j = j, i
            # Check if swapping the two edges improves the tour
            edge1 = problem.full_edge_cost(tour[i], tour[i+1])
            edge2 = problem.full_edge_cost(tour[j], tour[j+1] if j + 1 < problem.n else tour[0])
            edge3 = problem.full_edge_cost(tour[i], tour[j+1] if j + 1 < problem.n else tour[0])
            edge4 = problem.full_edge_cost(tour[j], tour[i+1])
            if edge3 + edge4 < edge1 + edge2:
                # Swap the two edges
                tour[i+1:j+1] = tour[i+1:j+1][::-1]
        
        # Perform adaptive 2-opt to further improve the tour
        improved = True
        max_iterations = problem.n // 2
        iteration = 0
        while improved and iteration < max_iterations:
            improved = False
            for i in range(problem.n - 1):
                for j in range(i + 2, problem.n):
                    edge1 = problem.full_edge_cost(tour[i], tour[i+1])
                    edge2 = problem.full_edge_cost(tour[j-1], tour[j])
                    edge3 = problem.full_edge_cost(tour[i], tour[j-1])
                    edge4 = problem.full_edge_cost(tour[j], tour[i+1])
                    if edge3 + edge4 < edge1 + edge2:
                        # Swap the two edges
                        tour[i+1:j] = tour[i+1:j][::-1]
                        improved = True
            iteration += 1
        
        # Neighborhood exploration
        for _ in range(problem.n):
            i = rng.integers(problem.n - 1)
            neighbors = problem.neighbors(tour[i])
            for neighbor in neighbors:
                if neighbor not in tour:
                    j = tour.index(tour[i])
                    edge1 = problem.full_edge_cost(tour[j], tour[j+1] if j + 1 < problem.n else tour[0])
                    edge2 = problem.full_edge_cost(tour[j], neighbor)
                    edge3 = problem.full_edge_cost(neighbor, tour[j+1] if j + 1 < problem.n else tour[0])
                    if edge2 + edge3 < edge1:
                        # Replace the edge with the new one
                        tour[j+1] = neighbor
        
        return np.array(tour)