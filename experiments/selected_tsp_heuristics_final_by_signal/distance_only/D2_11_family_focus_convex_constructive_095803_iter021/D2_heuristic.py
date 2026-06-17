import numpy as np

class TSPHeuristic:
    def __call__(self, problem, rng=None):
        if rng is None:
            rng = np.random.default_rng()

        # Calculate convex hull using Graham's scan algorithm
        points = problem.coords
        n = len(points)
        hull = self._convex_hull(points)

        # Initialize tour with convex hull
        tour = hull

        # Insert interior cities in a structured outside-in way
        interior_cities = [i for i in range(n) if i not in tour]
        for city in interior_cities:
            best_insertion_index = None
            best_insertion_cost = float('inf')
            for i in range(len(tour)):
                # Calculate insertion cost
                cost = problem.edge_cost(tour[i-1], city) + problem.edge_cost(city, tour[i]) - problem.edge_cost(tour[i-1], tour[i])
                if cost < best_insertion_cost:
                    best_insertion_index = i
                    best_insertion_cost = cost
            tour.insert(best_insertion_index, city)

        return tour

    def _convex_hull(self, points):
        n = len(points)
        hull = []
        l = 0
        for i in range(1, n):
            if points[i, 0] < points[l, 0]:
                l = i

        p = l
        q = 0
        while True:
            hull.append(p)
            q = (p + 1) % n

            for i in range(n):
                if self._orientation(points[p], points[i], points[q]) == 2:
                    q = i

            p = q

            if p == l:
                break

        return hull

    def _orientation(self, p, q, r):
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if val == 0:
            return 0  # Collinear
        elif val > 0:
            return 1  # Clockwise
        else:
            return 2  # Counterclockwise