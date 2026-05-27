import numpy as np
import math

class TSPHeuristic:
    def __call__(self, problem, rng=None):
        if rng is None:
            rng = np.random.default_rng()

        # Get city coordinates
        coords = problem.coords

        # Calculate grid cell size
        min_x, max_x = np.min(coords[:, 0]), np.max(coords[:, 0])
        min_y, max_y = np.min(coords[:, 1]), np.max(coords[:, 1])
        num_cells = int(np.sqrt(problem.n))
        cell_size_x = (max_x - min_x) / num_cells
        cell_size_y = (max_y - min_y) / num_cells

        # Initialize grid cells
        grid = {}
        for i, coord in enumerate(coords):
            cell_x = min(int((coord[0] - min_x) / cell_size_x), num_cells - 1)
            cell_y = min(int((coord[1] - min_y) / cell_size_y), num_cells - 1)
            if (cell_x, cell_y) not in grid:
                grid[(cell_x, cell_y)] = []
            grid[(cell_x, cell_y)].append(i)

        # Order grid cells in a radial pattern
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        ordered_cells = sorted(grid.keys(), key=lambda cell: math.atan2(cell[1] - (num_cells - 1) / 2, cell[0] - (num_cells - 1) / 2))

        # Build local paths within grid cells
        tour = []
        visited_cities = set()
        for cell in ordered_cells:
            if cell in grid:
                # Find the nearest unvisited city in the current cell
                nearest_city = None
                min_distance = float('inf')
                for city in grid[cell]:
                    if city not in visited_cities:
                        distance = problem.edge_cost(tour[-1] if tour else rng.integers(problem.n), city) if tour else 0
                        if distance < min_distance:
                            min_distance = distance
                            nearest_city = city
                if nearest_city is not None:
                    tour.append(nearest_city)
                    visited_cities.add(nearest_city)
                else:
                    # If no unvisited city is found, move to the next cell
                    continue

        # Ensure all cities are visited
        for i in range(problem.n):
            if i not in visited_cities:
                # Find the nearest city in the tour to the current unvisited city
                nearest_city = None
                min_distance = float('inf')
                for j in range(len(tour)):
                    distance = problem.edge_cost(i, tour[j])
                    if distance < min_distance:
                        min_distance = distance
                        nearest_city = j
                # Insert the unvisited city after the nearest city in the tour
                tour.insert(nearest_city + 1, i)

        # Perform a simple 2-opt to improve the tour
        improved = True
        while improved:
            improved = False
            for i in range(len(tour) - 1):
                for j in range(i + 1, len(tour)):
                    if problem.edge_cost(tour[i], tour[j]) + problem.edge_cost(tour[(i + 1) % len(tour)], tour[(j + 1) % len(tour)]) < problem.edge_cost(tour[i], tour[(i + 1) % len(tour)]) + problem.edge_cost(tour[j], tour[(j + 1) % len(tour)]):
                        tour[i + 1:j + 1] = tour[i + 1:j + 1][::-1]
                        improved = True

        return tour