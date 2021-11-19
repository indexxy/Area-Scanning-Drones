import numpy as np
from numba import guvectorize
import matplotlib.pyplot as plt
from evaluate import findAreas

MOVES = np.array([[0, 0], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1], [1, 0], [1, 1]], dtype=np.int32)

# Increasing the number of moves that are not [0, 0] in the pool
# This is done in order to make the probability of the drone `choosing` to move higher,
# instead of staying in its place.
MOVES = np.append(MOVES, MOVES[1:]).reshape(17, 2)
MOVES = np.append(MOVES, MOVES[1:]).reshape(33, 2)
HIGH = 33

ANGLES = np.array([0, 0, 45, 90, 135, 180, 225, 270, 315])
ANGLES = np.append(ANGLES, ANGLES[1:])
ANGLES = np.append(ANGLES, ANGLES[1:])


def plot_path_fitness(grid):
    stacked_paths = np.hstack(tuple(grid.best_paths))[np.newaxis]
    areas = findAreas(stacked_paths)
    figure, (ax1, ax2) = plt.subplots(1, 2)
    best_index = np.argwhere(grid.best_fitness == grid.best_fitness.max())[-1][-1]
    last_cells = tuple(tuple(c) for c in grid.best_paths[:, best_index, -1, :])
    plt.figtext(
        0.5, 0.03,
        f"Generation with the highest fitness: {best_index+1}, area={areas[best_index]}" +
        f", last cells: {last_cells}, start cell: {grid.start}",
        ha="center"
    )

    ax2.invert_yaxis()
    ax2.set_title(
        f"Most fit path generated by {grid.best_paths.shape[0]} drones",
        fontsize=10
    )
    for pa in grid.best_paths:
        ax2.plot(pa[best_index, :, 0], pa[best_index, :, 1])

    ax2.grid()

    ax1.plot(grid.best_fitness, label='best fitness')
    ax1.plot(grid.mean_fitness, label='mean fitness')
    ax1.legend()
    ax1.grid()
    ax1.set_xlabel('Generations')
    ax1.set_ylabel('Fitness Value (Higher is better)')
    figure.subplots_adjust(left=0.075, bottom=0.15, right=0.95, top=None, wspace=0.075, hspace=0.075)
    figure.set_dpi(100)

    area_figure, area_ax = plt.subplots(1)
    area_ax.plot(grid.mean_areas, label='mean area')
    area_ax.legend()
    area_ax.grid()
    area_ax.set_xlabel('Generations')
    area_ax.set_ylabel('Normalized Area (Higher is better)')

    figure.show()
    area_figure.show()

    plt.show(block=True)


# The following functions are universal/vectorized functions,
# they provide a significant boost in the runtime performance
@guvectorize(
    "void(int32[:], int32[:], int32[:])",
    "(n), (n) -> (n)",
    target="cpu",
    cache=True
)
def _crossOver_one_point(parent1, parent2, child):
    sep_point = np.random.randint(0, child.shape[0] + 1)
    child[:] = np.hstack((parent1[:sep_point + 1], parent2[sep_point + 1:]))


@guvectorize(
    "void(int32[:], int32[:], int32[:])",
    "(n), (n) -> (n)",
    target="cpu",
    cache=True
)
def _crossOver_two_points(parent1, parent2, child):
    sep_point = np.random.randint(0, child.shape[0] + 1)
    sep_point2 = np.random.randint(0, child.shape[0] + 1)

    if sep_point2 <= sep_point:
        tmp = sep_point
        sep_point = sep_point2
        sep_point2 = tmp

    child[:] = np.hstack(
        (
            parent2[:sep_point + 1],
            parent1[sep_point + 1: sep_point2 + 1],
            parent2[sep_point2 + 1:]
        )
    )


@guvectorize(
    "void(int32[:, :], int32, int32, int32[:, :])",
    "(n, m), (), () -> (n, m)",
    target="cpu",
    cache=True
)
def _calcPath(moves, limitX, limitY, paths):

    for i in range(moves.shape[0]-1):
        sumX = paths[i, 0] + moves[i, 0]
        sumY = paths[i, 1] + moves[i, 1]

        while sumX < 0 or sumX >= limitX or sumY < 0 or sumY >= limitY:
            # In order to prevent getting stuck in the corners of the grid,
            # If an illegal move is encountered (outside of the grid), the drone does not necessarily stay in its place
            # Instead, random moves are generated until a legal one is encountered.
            sumX = paths[i, 0] + np.random.randint(-1, 2)
            sumY = paths[i, 1] + np.random.randint(-1, 2)

        paths[i+1, 0] = sumX
        paths[i+1, 1] = sumY
