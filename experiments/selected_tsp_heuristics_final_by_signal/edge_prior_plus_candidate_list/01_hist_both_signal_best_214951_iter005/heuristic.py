import numpy as np
import math

class TSPHeuristic:
    # Intended mechanism family: Region-based partitioning TSP heuristic
    def __call__(self, problem, rng=None):
        if rng is None:
            rng = np.random.default_rng()

        # Initialize the set of unvisited cities
        unvisited = set(range(problem.n))

        # Choose a random starting city
        current_city = rng.integers(problem.n)
        unvisited.remove(current_city)

        # Initialize the tour with the starting city
        tour = [current_city]

        # Repeat until all cities are visited
        while unvisited:
            # Divide the remaining cities into regions based on their distance to the current city
            regions = {}
            for city in unvisited:
                distance = problem.edge_cost(current_city, city)
                if distance not in regions:
                    regions[distance] = []
                regions[distance].append(city)

            # Sort the regions by distance
            sorted_regions = sorted(regions.items())

            # Choose the closest region with the highest prior
            max_prior = float('-inf')
            next_city = None
            for distance, cities in sorted_regions:
                for city in cities:
                    prior = problem.prior(current_city, city)
                    if prior > max_prior:
                        max_prior = prior
                        next_city = city

            # Add the chosen city to the tour and remove it from the unvisited set
            tour.append(next_city)
            unvisited.remove(next_city)
            current_city = next_city

        return tour