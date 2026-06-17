import numpy as np

class TSPHeuristic:
    def __call__(self, problem, rng=None):
        if rng is None:
            rng = np.random.default_rng()

        # Number of clusters (k)
        k = max(2, int(np.sqrt(problem.n)))

        # Initialize cluster assignments
        clusters = np.zeros(problem.n, dtype=int)

        # Randomly assign each city to a cluster
        for i in range(problem.n):
            clusters[i] = rng.integers(k)

        # Recalculate cluster centers
        centers = np.zeros((k, 2))
        for i in range(k):
            cluster_cities = np.where(clusters == i)[0]
            if len(cluster_cities) > 0:
                center_coords = problem.coords[cluster_cities].mean(axis=0)
                centers[i] = center_coords

        # Reassign cities to clusters based on proximity to cluster centers
        for i in range(problem.n):
            city_coords = problem.coords[i]
            min_dist = float('inf')
            closest_cluster = -1
            for j in range(k):
                dist = np.linalg.norm(city_coords - centers[j])
                if dist < min_dist:
                    min_dist = dist
                    closest_cluster = j
            clusters[i] = closest_cluster

        # Build local tours within each cluster using a nearest neighbor approach
        local_tours = []
        used_cities = set()
        for i in range(k):
            cluster_cities = np.where(clusters == i)[0]
            if len(cluster_cities) > 0:
                tour = [cluster_cities[0]]
                used_cities.add(cluster_cities[0])
                remaining_cities = list(cluster_cities[1:])
                while remaining_cities:
                    current_city = tour[-1]
                    min_dist = float('inf')
                    next_city_index = -1
                    for j, city in enumerate(remaining_cities):
                        dist = problem.edge_cost(current_city, city)
                        if dist < min_dist:
                            min_dist = dist
                            next_city_index = j
                    if next_city_index == -1:
                        break
                    tour.append(remaining_cities.pop(next_city_index))
                    used_cities.add(tour[-1])
                local_tours.append(tour)

        # Collect any remaining unused cities and add them to the local tours
        remaining_cities = [city for city in range(problem.n) if city not in used_cities]
        if remaining_cities:
            for city in remaining_cities:
                min_dist = float('inf')
                closest_tour_index = -1
                for i, tour in enumerate(local_tours):
                    dist = problem.edge_cost(tour[-1], city)
                    if dist < min_dist:
                        min_dist = dist
                        closest_tour_index = i
                local_tours[closest_tour_index].append(city)

        # Refine local tours by reversing segments if it improves the tour cost
        for tour in local_tours:
            for i in range(len(tour) - 1):
                for j in range(i + 2, len(tour) + 1):
                    segment = tour[i:j]
                    reversed_segment = segment[::-1]
                    original_cost = sum(problem.edge_cost(tour[k], tour[k + 1]) for k in range(i, j - 1))
                    new_cost = sum(problem.edge_cost(tour[k], tour[k + 1]) for k in range(i, i + len(reversed_segment) - 1))
                    if new_cost < original_cost:
                        tour[i:j] = reversed_segment

        # Connect clusters through endpoint-aware bridges
        tour = local_tours[0]
        remaining_tours = local_tours[1:]
        while remaining_tours:
            last_city = tour[-1]
            min_dist = float('inf')
            next_tour_index = -1
            for i, local_tour in enumerate(remaining_tours):
                dist = problem.edge_cost(last_city, local_tour[0])
                if dist < min_dist:
                    min_dist = dist
                    next_tour_index = i
            if next_tour_index == -1:
                break
            tour.extend(remaining_tours.pop(next_tour_index))

        return tour