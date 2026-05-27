import numpy as np

def construct_tour(D, prior_rows, start_node):
    n = D.shape[0]
    tour = [start_node]
    visited = set([start_node])
    current_node = start_node

    while len(tour) < n:
        max_score = float('-inf')
        next_node = None

        for neighbor, score in prior_rows[current_node].items():
            if neighbor not in visited and score > max_score:
                max_score = score
                next_node = neighbor

        if next_node is None:
            # If no neighbor has a higher score, select the neighbor with the minimum distance
            min_distance = float('inf')
            next_node = None
            for neighbor in range(n):
                if neighbor not in visited and D[current_node, neighbor] < min_distance:
                    min_distance = D[current_node, neighbor]
                    next_node = neighbor

        tour.append(next_node)
        visited.add(next_node)
        current_node = next_node

    return np.array(tour)
