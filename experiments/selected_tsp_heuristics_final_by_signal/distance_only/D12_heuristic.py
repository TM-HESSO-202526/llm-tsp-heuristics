import numpy as np

class TSPHeuristic:
    def __call__(self, problem, rng=None):
        if rng is None:
            rng = np.random.default_rng()

        # Choose two random centers
        center_idx1 = rng.integers(problem.n)
        center_idx2 = rng.integers(problem.n)
        while center_idx2 == center_idx1:
            center_idx2 = rng.integers(problem.n)
        center1 = problem.coords[center_idx1]
        center2 = problem.coords[center_idx2]

        # Calculate the polar angles and radii of all cities with respect to both centers
        angles1 = np.arctan2(problem.coords[:, 1] - center1[1], problem.coords[:, 0] - center1[0])
        radii1 = np.linalg.norm(problem.coords - center1, axis=1)
        angles2 = np.arctan2(problem.coords[:, 1] - center2[1], problem.coords[:, 0] - center2[0])
        radii2 = np.linalg.norm(problem.coords - center2, axis=1)

        # Weight the angles by the radii to handle radius jumps
        weighted_angles1 = angles1 + radii1 / np.max(radii1)
        weighted_angles2 = angles2 + radii2 / np.max(radii2)

        # Adjust the weighted angles to reduce the impact of radius jumps
        min_radius1 = np.min(radii1)
        max_radius1 = np.max(radii1)
        radius_range1 = max_radius1 - min_radius1
        adjusted_angles1 = weighted_angles1 - (radii1 - min_radius1) / radius_range1 * np.pi / 4  
        min_radius2 = np.min(radii2)
        max_radius2 = np.max(radii2)
        radius_range2 = max_radius2 - min_radius2
        adjusted_angles2 = weighted_angles2 - (radii2 - min_radius2) / radius_range2 * np.pi / 4  

        # Sort cities by adjusted angles with respect to both centers
        sorted_indices1 = np.argsort(adjusted_angles1)
        sorted_indices2 = np.argsort(adjusted_angles2)

        # Select the better ordering based on the total length of the tour
        tour1 = sorted_indices1
        tour2 = sorted_indices2
        tour_length1 = 0
        for i in range(len(tour1) - 1):
            tour_length1 += problem.edge_cost(tour1[i], tour1[i + 1])
        tour_length2 = 0
        for i in range(len(tour2) - 1):
            tour_length2 += problem.edge_cost(tour2[i], tour2[i + 1])
        if tour_length1 < tour_length2:
            tour = tour1
        else:
            tour = tour2

        # Endpoint bridge: try to bridge the tour between the first and last city
        min_bridge_cost = problem.edge_cost(tour[0], tour[-1])
        for i in range(len(tour)):
            for j in range(i + 1, len(tour)):
                bridge_cost = problem.edge_cost(tour[i], tour[j])
                if bridge_cost < min_bridge_cost:
                    min_bridge_cost = bridge_cost
                    bridge_idx1 = i
                    bridge_idx2 = j
        if min_bridge_cost < problem.edge_cost(tour[0], tour[-1]):
            # Reverse the segment between the two bridge points
            tour[bridge_idx1:bridge_idx2 + 1] = tour[bridge_idx1:bridge_idx2 + 1][::-1]

        # Angle adjustment: try to adjust the angles of the tour to reduce the total length
        for _ in range(30):
            i = rng.integers(len(tour))
            j = rng.integers(len(tour))
            if i > j:
                i, j = j, i
            new_tour = tour.copy()
            new_tour[i:j + 1] = tour[i:j + 1][::-1]
            new_tour_length = 0
            for k in range(len(new_tour) - 1):
                new_tour_length += problem.edge_cost(new_tour[k], new_tour[k + 1])
            if new_tour_length < tour_length1:
                tour = new_tour
                tour_length1 = new_tour_length

        # Additional multi-step angle adjustment
        for _ in range(10):
            for i in range(len(tour) - 1):
                new_tour = tour.copy()
                new_tour[i:i + 2] = tour[i:i + 2][::-1]
                new_tour_length = 0
                for k in range(len(new_tour) - 1):
                    new_tour_length += problem.edge_cost(new_tour[k], new_tour[k + 1])
                if new_tour_length < tour_length1:
                    tour = new_tour
                    tour_length1 = new_tour_length

        return tour