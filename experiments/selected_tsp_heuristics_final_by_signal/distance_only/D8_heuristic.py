import numpy as np

class TSPHeuristic:
    def __call__(self, problem, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        
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
        
        return np.array(tour)