import numpy as np

class TSPHeuristic:
    def __call__(self, problem, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        
        # Initialize the best tour and its cost
        best_tour = None
        best_cost = np.inf
        
        # Perform multiple random starts
        for _ in range(min(10, problem.n)):
            # Choose a random starting city
            current_city = rng.integers(problem.n)
            
            # Initialize the tour
            tour = [current_city]
            
            # Initialize the set of unvisited cities
            unvisited_cities = set(range(problem.n))
            unvisited_cities.remove(current_city)
            
            # Construct the tour
            while unvisited_cities:
                # Find the nearest unvisited city
                nearest_city = min(unvisited_cities, key=lambda city: problem.edge_cost(current_city, city))
                
                # Add the nearest city to the tour
                tour.append(nearest_city)
                
                # Update the current city and unvisited cities
                current_city = nearest_city
                unvisited_cities.remove(current_city)
            
            # Calculate the tour cost
            tour_cost = sum(problem.edge_cost(tour[i], tour[(i+1) % len(tour)]) for i in range(len(tour)))
            
            # Update the best tour if the current tour has a lower cost
            if tour_cost < best_cost:
                best_tour = tour
                best_cost = tour_cost
        
        return np.array(best_tour)