import numpy as np

def construct_tour(D, prior_rows, start_node):
    n = len(prior_rows)
    tour = [start_node]
    unvisited = set(range(n))
    unvisited.remove(start_node)

    while unvisited:
        current_node = tour[-1]
        max_score = float('-inf')
        next_node = None

        for neighbor in unvisited:
            score = prior_rows[current_node].get(neighbor, 0)
            if score > max_score:
                max_score = score
                next_node = neighbor
            elif score == max_score:
                if D[current_node, neighbor] < D[current_node, next_node]:
                    next_node = neighbor

        tour.append(next_node)
        unvisited.remove(next_node)

    return tour
