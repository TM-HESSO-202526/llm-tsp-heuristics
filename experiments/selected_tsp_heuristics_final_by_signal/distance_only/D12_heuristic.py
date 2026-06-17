import numpy as np
import math

class TSPHeuristic:
    def __call__(self, problem, rng=None):
        if rng is None:
            rng = np.random.default_rng()

        n = problem.n
        if not hasattr(problem, 'coords'):
            coords = np.zeros((n, 2))
            for i in range(n):
                coords[i, 0] = rng.integers(0, 100)
                coords[i, 1] = rng.integers(0, 100)
        else:
            coords = problem.coords

        # Build a sparse geometric graph using a scalable local approximation
        graph = self.build_sparse_graph(coords, n, rng)

        # Find a Hamiltonian tour in the graph using a greedy approach
        tour = self.find_hamiltonian_tour(graph, n, coords)

        # Local cleanup to avoid duplicates and ensure Hamiltonian property
        tour = self.local_cleanup(tour, n)

        return tour

    def build_sparse_graph(self, coords, n, rng):
        graph = {i: [] for i in range(n)}

        for i in range(n):
            nearest_neighbors = []
            for j in range(n):
                if i != j:
                    distance = np.linalg.norm(coords[i] - coords[j])
                    nearest_neighbors.append((distance, j))

            nearest_neighbors.sort(key=lambda x: x[0])
            for _, neighbor in nearest_neighbors[:3]:
                graph[i].append(neighbor)

        return graph

    def find_hamiltonian_tour(self, graph, n, coords):
        tour = []
        visited = set()

        current_node = rng.integers(0, n)
        tour.append(current_node)
        visited.add(current_node)

        while len(tour) < n:
            next_nodes = [node for node in graph[current_node] if node not in visited]
            if next_nodes:
                next_node = min(next_nodes, key=lambda node: np.linalg.norm(coords[current_node] - coords[node]))
                tour.append(next_node)
                visited.add(next_node)
                current_node = next_node
            else:
                unvisited_nodes = [node for node in range(n) if node not in visited]
                if unvisited_nodes:
                    current_node = unvisited_nodes[0]
                    tour.append(current_node)
                    visited.add(current_node)
                else:
                    break

        return tour

    def local_cleanup(self, tour, n):
        tour = list(set(tour))

        missing_cities = [i for i in range(n) if i not in tour]
        for city in missing_cities:
            closest_city = min(tour, key=lambda x: min([self.distance(x, city, n) for x in tour]))
            index = tour.index(closest_city)
            tour.insert(index + 1, city)

        return tour

    def distance(self, i, j, n, problem):
        return problem.edge_cost(i, j)