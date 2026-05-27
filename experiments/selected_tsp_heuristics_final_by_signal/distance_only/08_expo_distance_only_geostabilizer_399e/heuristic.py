import numpy as np

def construct_tour(D, start_node):
    n = D.shape[0]
    tour = [start_node]
    visited = [False] * n
    visited[start_node] = True

    for _ in range(n - 1):
        min_dist = np.inf
        next_node = None
        for i in range(n):
            if not visited[i]:
                dist = D[tour[-1], i]
                if dist < min_dist:
                    min_dist = dist
                    next_node = i
                elif dist == min_dist:
                    # Geometric stabilization: choose the node that is closest to the centroid of the tour
                    centroid = np.mean([tour], axis=0)
                    dist_to_centroid = np.linalg.norm(centroid - i)
                    next_node_dist_to_centroid = np.linalg.norm(centroid - next_node)
                    if dist_to_centroid < next_node_dist_to_centroid:
                        next_node = i
        tour.append(next_node)
        visited[next_node] = True

    return tour
