import numpy as np
import math

class TSPHeuristic:
    def __call__(self, problem, rng=None):
        # Initialize the random number generator if not provided
        if rng is None:
            rng = np.random.default_rng()

        # Initialize the number of cities
        n = problem.n

        # Initialize a set to keep track of visited cities
        visited = set()

        # Initialize the current city
        current_city = 0

        # Add the current city to the visited set
        visited.add(current_city)

        # Initialize the tour
        tour = [current_city]

        # Initialize the minimum spanning tree
        mst = {}

        # Create a list of all cities
        cities = list(range(n))

        # While not all cities have been visited
        while len(visited) < n:
            # Initialize the minimum cost and next city
            min_cost = float('inf')
            next_city = None

            # For each unvisited city
            for city in cities:
                # If the city has not been visited
                if city not in visited:
                    # Calculate the cost of traveling from the current city to the city
                    cost = problem.edge_cost(current_city, city)

                    # If the cost is less than the minimum cost
                    if cost < min_cost:
                        # Update the minimum cost and next city
                        min_cost = cost
                        next_city = city

            # Add the next city to the tour and the visited set
            tour.append(next_city)
            visited.add(next_city)

            # Update the current city
            current_city = next_city

            # Add the edge to the minimum spanning tree
            if current_city not in mst:
                mst[current_city] = []
            mst[current_city].append(tour[-2])  # Store the edge in both directions
            if tour[-2] not in mst:
                mst[tour[-2]] = []
            mst[tour[-2]].append(current_city)

        # Convert the minimum spanning tree into a Hamiltonian tour
        # Start at the first city in the tour
        current_city = tour[0]

        # Initialize the Hamiltonian tour
        hamiltonian_tour = [current_city]

        # While not all cities have been visited in the Hamiltonian tour
        while len(hamiltonian_tour) < n:
            # Get the list of neighboring cities that have not been visited yet
            unvisited_neighbors = [neighbor for neighbor in mst.get(current_city, []) if neighbor not in hamiltonian_tour]

            # If there are unvisited neighbors
            if unvisited_neighbors:
                # Choose the closest unvisited neighbor
                next_city = min(unvisited_neighbors, key=lambda neighbor: problem.edge_cost(current_city, neighbor))

                # Add the next city to the Hamiltonian tour
                hamiltonian_tour.append(next_city)

                # Update the current city
                current_city = next_city
            else:
                # If there are no unvisited neighbors, choose the closest unvisited city
                unvisited_cities = [city for city in range(n) if city not in hamiltonian_tour]
                next_city = min(unvisited_cities, key=lambda city: problem.edge_cost(current_city, city))

                # Add the next city to the Hamiltonian tour
                hamiltonian_tour.append(next_city)

                # Update the current city
                current_city = next_city

        # Perform a bounded 2-opt cleanup with more iterations and a better selection strategy
        for _ in range(int(n * 0.2)):  # Increase the number of iterations
            i = rng.integers(1, n - 1)  # Avoid swapping the first city
            j = rng.integers(i + 1, n)
            # Check if the 2-opt swap improves the tour
            if problem.edge_cost(hamiltonian_tour[i - 1], hamiltonian_tour[j]) + problem.edge_cost(hamiltonian_tour[i], hamiltonian_tour[j - 1]) < problem.edge_cost(hamiltonian_tour[i - 1], hamiltonian_tour[i]) + problem.edge_cost(hamiltonian_tour[j - 1], hamiltonian_tour[j]):
                # Perform the 2-opt swap
                hamiltonian_tour[i:j] = hamiltonian_tour[i:j][::-1]

        return np.array(hamiltonian_tour)