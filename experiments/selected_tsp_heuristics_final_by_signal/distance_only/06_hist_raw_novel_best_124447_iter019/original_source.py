import numpy as np

class TSPHeuristic:
    def __init__(self):
        pass

    def __call__(self, problem, rng=None):
        # Initialize the current node and the tour
        current_node = 0
        tour = [current_node]
        visited = set([current_node])

        # Loop through all nodes
        while len(tour) < problem.n:
            # Find the closest unvisited neighbor with a higher prior
            min_cost = float('inf')
            next_node = None
            max_prior = 0
            for neighbor in problem.neighbors(current_node):
                if neighbor not in visited:
                    cost = problem.edge_cost(current_node, neighbor)
                    prior = problem.prior(current_node, neighbor)
                    if cost < min_cost and prior > max_prior:
                        min_cost = cost
                        next_node = neighbor
                        max_prior = prior
                    elif cost < min_cost and prior == max_prior:
                        min_cost = cost
                        next_node = neighbor

            # If no unvisited neighbors are found, start a new cluster
            if next_node is None:
                # Find the closest unvisited node with a higher prior
                min_cost = float('inf')
                for i in range(problem.n):
                    if i not in visited:
                        cost = problem.edge_cost(current_node, i)
                        prior = problem.prior(current_node, i)
                        if cost < min_cost and prior > max_prior:
                            min_cost = cost
                            next_node = i
                            max_prior = prior
                        elif cost < min_cost and prior == max_prior:
                            min_cost = cost
                            next_node = i

            # Add the next node to the tour and mark it as visited
            tour.append(next_node)
            visited.add(next_node)
            current_node = next_node

        # Perform a limited 2-opt improvement
        improved = True
        while improved:
            improved = False
            for i in range(len(tour) - 1):
                for j in range(i + 2, len(tour)):
                    cost_before = problem.edge_cost(tour[i], tour[i + 1]) + problem.edge_cost(tour[j - 1], tour[j])
                    cost_after = problem.edge_cost(tour[i], tour[j - 1]) + problem.edge_cost(tour[i + 1], tour[j])
                    if cost_after < cost_before:
                        # Reverse the segment
                        tour[i + 1:j] = tour[i + 1:j][::-1]
                        improved = True

        return tour