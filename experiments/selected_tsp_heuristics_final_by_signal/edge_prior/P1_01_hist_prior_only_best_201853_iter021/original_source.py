import numpy as np

class TSPHeuristic:
    def __call__(self, problem, rng=None):
        # Intended mechanism: Iterative Centroid-Based Tour Construction with Edge Prior
        if rng is None:
            rng = np.random.default_rng()
        
        n = problem.n
        tour = []
        
        # Initialize a set to keep track of visited cities
        visited = set()
        
        # Calculate the centroid of all cities
        centroid = np.mean(problem.coords, axis=0)
        
        # Start with the city closest to the centroid
        current_city = np.argmin(np.linalg.norm(problem.coords - centroid, axis=1))
        visited.add(current_city)
        tour.append(current_city)
        
        # Grow the tour by adding the most promising unvisited city
        for _ in range(n - 1):
            max_prior = -1
            next_city = None
            max_edge_cost = np.inf
            for i in range(n):
                if i not in visited:
                    prior = problem.prior(current_city, i)
                    cost = problem.edge_cost(current_city, i)
                    if prior > max_prior and cost < np.inf:
                        max_prior = prior
                        next_city = i
                        max_edge_cost = cost
                    elif prior == max_prior and cost < max_edge_cost:
                        next_city = i
                        max_edge_cost = cost
            
            # If no promising city is found, choose a random unvisited city
            if next_city is None:
                unvisited = [i for i in range(n) if i not in visited]
                next_city = rng.choice(unvisited)
            
            tour.append(next_city)
            visited.add(next_city)
            current_city = next_city
        
        # Perform a bounded iterative refinement to improve the tour
        max_refine_attempts = 10
        for _ in range(max_refine_attempts):
            # Choose two random cities in the tour
            idx1, idx2 = rng.integers(0, n-1, size=2)
            if idx1 == idx2:
                continue
            
            # Extract the sub-tours
            sub_tour1 = tour[idx1:]
            sub_tour2 = tour[:idx1]
            
            # Check if reversing sub-tour1 improves the tour
            new_tour = sub_tour2 + sub_tour1[::-1]
            if self.evaluate_tour(new_tour, problem) < self.evaluate_tour(tour, problem):
                tour = new_tour
        
        return tour
    
    def evaluate_tour(self, tour, problem):
        # Calculate the total cost of the tour
        total_cost = 0
        for i in range(len(tour) - 1):
            total_cost += problem.edge_cost(tour[i], tour[i+1])
        total_cost += problem.edge_cost(tour[-1], tour[0])
        return total_cost