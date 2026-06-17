import numpy as np

class TSPHeuristic:
    def __call__(self, problem, rng=None):
        if rng is None:
            rng = np.random.default_rng()

        n = problem.n
        tour = []
        open_endpoints = set(range(n))

        # Start with a random city
        current_city = rng.integers(n)
        tour.append(current_city)
        open_endpoints.remove(current_city)

        # Initialize a list to store the fragments
        fragments = [[current_city]]

        # Grow the tour by adding fragments and bridging endpoints
        while len(tour) < n:
            # Find the closest unvisited city to each endpoint
            closest_cities = []
            for fragment in fragments:
                min_distance = np.inf
                closest_city = None
                for city in open_endpoints:
                    distance = problem.edge_cost(fragment[-1], city)
                    if distance < min_distance:
                        min_distance = distance
                        closest_city = city
                closest_cities.append(closest_city)

            # Find the endpoint with the closest unvisited city
            min_distance = np.inf
            next_fragment_index = None
            next_city = None
            for i, (closest_city, fragment) in enumerate(zip(closest_cities, fragments)):
                distance = problem.edge_cost(fragment[-1], closest_city)
                if distance < min_distance:
                    min_distance = distance
                    next_fragment_index = i
                    next_city = closest_city

            # Add the next city to the corresponding fragment and remove it from the open endpoints
            fragments[next_fragment_index].append(next_city)
            open_endpoints.remove(next_city)

            # Check if the fragment is complete
            if len(fragments[next_fragment_index]) > 1 and next_city in [fragment[0] for fragment in fragments]:
                # Remove the fragment from the list of open endpoints
                fragments.pop(next_fragment_index)

            # Update the tour
            tour = [city for fragment in fragments for city in fragment]

            # Introduce a risk-awareness mechanism to avoid sub-tours
            if len(fragments) > 1:
                # Find the fragment with the fewest cities
                min_length = np.inf
                min_fragment_index = None
                for i, fragment in enumerate(fragments):
                    if len(fragment) < min_length:
                        min_length = len(fragment)
                        min_fragment_index = i

                # Merge the smallest fragment with the largest fragment
                max_length = 0
                max_fragment_index = None
                for i, fragment in enumerate(fragments):
                    if len(fragment) > max_length:
                        max_length = len(fragment)
                        max_fragment_index = i

                # Merge the fragments
                fragments[max_fragment_index].extend(fragments[min_fragment_index])
                fragments.pop(min_fragment_index)

        return tour