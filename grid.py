from evaluate import manhattan, findAreas
from utils import _crossOver_one_point, _crossOver_two_points, _calcPath, MOVES, ANGLES, HIGH
from utils import plot_path_fitness
import numpy as np
import time
import matplotlib.pyplot as plt


class Grid:
    def __init__(
            self, start: tuple, pop_size: int, mu: float, limit: tuple,
            cross_points: int = 1, elitism: float = 0.0, num_drones: int = 1
    ):
        self.start = start  # Starting point's coordinates (x, y)
        self.limit = limit  # Map border's limit
        self.p_size = pop_size    # Population's size for each drone
        self.mu = mu        # Mutation Rate
        self.num_drones = int(num_drones)    # Number of drones that scan the area simultaneously

        self.total_target_length = limit[0] * limit[1]    # Total targeted number of scanned cells
        self.target_length = round(self.total_target_length / num_drones)  # Targeted number of moves for each drone

        self.max_angle_sum = 180 * (self.target_length - 1)
        
        # Maximum possible distance of a given point on the map
        self.max_dist = max(self.start[0], abs(self.start[0] - self.limit[0]))
        self.max_dist += max(self.start[1], abs(self.start[1] - self.limit[1]))
        
        # Generating initial population randomly
        self.population = np.random.randint(0, HIGH, (self.num_drones, self.p_size, self.target_length), dtype=np.int32)
        self.paths = None

        self.best_populations = None
        self.best_fitness = None
        self.best_paths = None
        self.best_norm_fitness = None
        self.mean_fitness = None
        self.mean_norm_fitness = None
        self.best_areas = None
        self.mean_areas = None

        self.cross_points = cross_points    # Number of cross points
        self.elitism = round(elitism * pop_size)    # Number of elite parents to be taken to the next generation

    # Generates a path for the given moves
    def calcPath(self):

        moves = MOVES[self.population].astype(np.int32)
        paths = np.zeros_like(moves, dtype=np.int32)

        paths[:, :, 0, :] = self.start

        _calcPath(moves, np.int32(self.limit[0]), np.int32(self.limit[1]), paths)
        self.paths = paths.copy()

        return paths

    # Calculates the final fitness scores of given paths and a given population
    def calcFitness(self, paths, population=None):

        if population is None:
            population = self.population

        stacked_paths = np.hstack(tuple(paths))[np.newaxis]
        areas = findAreas(stacked_paths)

        # Normalizing the calculated areas
        areas = areas / self.total_target_length

        # Calculating the difference in angle between each move and it's successor
        angles_0 = ANGLES[population[:, :, 0:-1]]
        angles_1 = ANGLES[population[:, :, 1:]]
        
        diff = np.abs(angles_1 - angles_0)
        
        # Rescaling the angles greater than 180
        mask = diff > 180
        diff[mask] = 360 - diff[mask]
        
        angle_fit = diff.sum(axis=-1)

        # Summing the angle fitness value found by each drone
        angle_fit = angle_fit.sum(axis=0)

        # Normalizing the total sum of angles
        angle_fit = angle_fit / (self.max_angle_sum * self.num_drones)

        # Inverting the MAX's and MIN's
        # (This is done to make the maximization problem common among all fitness function's components)
        angle_fit = 1 - angle_fit

        last_cells = paths[:, :, -1, :]
        
        # Calculating the manhattan distance between the last cell of each path and the starting point
        last_distances = manhattan(last_cells[:, :, 0], last_cells[:, :, 1], *self.start)

        # Summing the last distances found by each drone
        last_distances = last_distances.sum(axis=0)

        last_distances = last_distances / (self.max_dist * self.num_drones)

        # Inverting the MAX's and MIN's
        # (This is done to make the maximization problem common among all fitness function's components)
        last_distances = 1 - last_distances

        fitness = areas + angle_fit + last_distances

        # Normalizing the total fitness sum,
        # To create a probability distribution that will be used later in the selection process
        norm_fitness = fitness / fitness.sum()

        return norm_fitness, fitness, areas
    
    # This function randomly selects a pair of populations
    # from the current population based on a probability distribution
    def selection(self, prob_dist):

        parents1 = np.empty((self.num_drones, self.p_size, self.target_length), dtype=np.int32)
        parents2 = np.empty((self.num_drones, self.p_size, self.target_length), dtype=np.int32)
        for i in range(self.num_drones):
            parent1_indices = np.random.choice(self.p_size, self.p_size, True, prob_dist)
            parent2_indices = np.random.choice(self.p_size, self.p_size, True, prob_dist)

            parents1[i] = self.population[i, parent1_indices, :]
            parents2[i] = self.population[i, parent2_indices, :]

        return parents1, parents2

    # This function performs a cross-over operation between two parents (individuals)
    # To create a new generation (children)
    def crossOver(self, parents1, parents2, sorted_ind):

        children = np.zeros_like(self.population, dtype=np.int32)

        if self.cross_points == 1:
            _crossOver_one_point(parents1, parents2, children)
        else:
            _crossOver_two_points(parents1, parents2, children)

        if self.elitism != 0:
            elites_indices = sorted_ind[self.p_size - self.elitism:]
            elites = self.population[:, elites_indices]
            children[:, self.p_size - self.elitism:, :] = elites

        return children

    def mutate(self, children):
        prob_mask = np.random.random(children.shape) < self.mu
        children[prob_mask] = np.random.randint(0, HIGH, np.count_nonzero(prob_mask))

    def run(self, G):

        self.best_populations = np.empty((self.num_drones, G, *self.population.shape[2:]), dtype=np.int32)
        self.best_paths = np.empty((self.num_drones, G, self.target_length, 2), dtype=np.int32)
        self.best_norm_fitness = np.empty(G, dtype=float)
        self.best_fitness = np.empty(G, dtype=float)
        self.mean_fitness = np.empty(G, dtype=float)
        self.mean_norm_fitness = np.empty(G, dtype=float)
        self.best_areas = np.empty(G, dtype=float)
        self.mean_areas = np.empty(G, dtype=float)

        for i in range(G):
            paths = self.calcPath()
            norm_fitness, fitness, areas = self.calcFitness(paths)
            
            # Taking the index of the best population in the current generation
            best_population = np.argsort(norm_fitness)[-1]
            
            # Performing the `natural selection` process, based on the normalized fitness scores
            parents = self.selection(norm_fitness)
            best_norm_fitness_ind = None

            if self.elitism != 0:
                best_norm_fitness_ind = np.argsort(norm_fitness)
            
            # Performing cross-over operation between the chosen parents
            children = self.crossOver(*parents, best_norm_fitness_ind)
            # Mutating then new generation
            self.mutate(children)

            self.best_populations[:, i] = self.population[:, best_population]
            self.best_paths[:, i] = paths[:, best_population]
            self.best_norm_fitness[i] = norm_fitness[best_population]
            self.best_fitness[i] = fitness[best_population]
            self.mean_fitness[i] = np.mean(fitness)
            self.mean_norm_fitness[i] = np.mean(norm_fitness)
            self.best_areas[i] = areas[best_population]
            self.mean_areas[i] = np.mean(areas)
            
            # Assigning the new generation to the old generation
            self.population = children.copy()


if __name__ == "__main__":
    b = time.time()
    lim = (9, 9)
    p = Grid(start=(8, 4), pop_size=200, mu=0.0, limit=lim, cross_points=2, elitism=0.3, num_drones=2)
    # p.calcPath()
    # old_p = p.paths.copy()
    p.run(500)
    e = time.time()
    print(e - b)
    nf, f, ar = p.calcFitness(p.best_paths, p.best_populations)
    a = findAreas(np.hstack(tuple(p.best_paths))[np.newaxis])
    plot_path_fitness(p)

