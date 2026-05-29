import numpy as np
import time
import random

def construct_tour(problem):
    n = int(problem.n)
    cand = problem.cand
    if n <= 1:
        return np.arange(n, dtype=np.int64)

    start_time = globals().get("_START_TIME", time.time())
    time_limit = float(globals().get("TIME_LIMIT_S", 1.0))

    def time_up(frac=0.9):
        return (time.time() - start_time) > frac * time_limit

    def ins_cost(a, b, x):
        return edge_cost(a, x) + edge_cost(x, b) - edge_cost(a, b)

    def bounded_cleanup(tour):
        # Define a tiny bounded improver that checks if the last two nodes in the tour are adjacent
        # If they are, remove the last node and return the updated tour
        if len(tour) > 1 and edge_cost(tour[-1], tour[-2]) == 0:
            return np.delete(tour, -1)
        return tour

    tour = np.array([0])  # Initialize the tour with the first node
    current_node = 0
    while len(tour) < n:
        # Choose a node from the candidate frontier of the last node in the tour
        next_node = random.choice(cand[tour[-1]])
        # Insert it into the best cycle edge
        best_insertion = None
        best_cost = float('inf')
        for i in range(len(tour) - 1):
            cost = ins_cost(tour[i], tour[i + 1], next_node)
            if cost < best_cost:
                best_cost = cost
                best_insertion = i
        if best_insertion is not None:
            tour = np.insert(tour, best_insertion + 1, next_node)
        else:
            # If no good insertion is found, append the node to the end of the tour
            tour = np.append(tour, next_node)
        current_node = next_node
        if time_up():
            break
    # Finish by cheapest insertion into the existing cycle
    for i in range(len(tour) - 1):
        cost = ins_cost(tour[i], tour[i + 1], tour[-1])
        if cost < 0:
            tour = np.insert(tour, i + 1, tour[-1])
            break
    tour = bounded_cleanup(tour)
    # Define a tiny bounded improver that checks if the tour can be improved by swapping two adjacent nodes
    def bounded_improver(tour):
        n = len(tour)
        for i in range(n - 1):
            for j in range(i + 1, n - 1):
                cost = edge_cost(tour[i], tour[i + 1]) + edge_cost(tour[j], tour[j + 1]) - edge_cost(tour[i], tour[j]) - edge_cost(tour[i + 1], tour[j + 1])
                if cost < 0:
                    tour = np.array([tour[j], tour[i], tour[i + 1], tour[j + 1]] if i == j else tour)
                    return tour
        return tour
    tour = bounded_improver(tour)
    return np.asarray(tour, dtype=np.int64)