from grid import Grid
from evaluate import findAreas
from utils import plot_path_fitness
import time



if __name__ == "__main__":

    limit = (9, 9)
    print("- SPX: X Coordinates of Starting Point (should be between 0 and 9)")
    print("- SPY: Y Coordinates of Starting Point (should be between 0 and 9)")
    print("- MU: Mutation Rate")
    print("- P: Population Size")
    print("- C: Number of Cross Points (it should be either 1 or 2)")
    print("- (Optional) E: Elitism Rate, (it should be between 0 and 1) (default is 0) (leave empty for default)")
    print("-"*20)
    print("Please enter the parameters with the following format")
    
    inp = str(input("SPX, SPY, MU, P, C, E\n>>"))

    split_inp = inp.split(',')
    sp = (int(split_inp[0]), int(split_inp[1]))
    mu = float(split_inp[2])
    p = int(split_inp[3])
    c = int(split_inp[4])
    e = 0 if not len(split_inp) == 6 else float(split_inp[5])

    assert 0 <= sp[0] < limit[0] and 0 <= sp[1] < limit[1], "Starting point coordinates should be within [0, 9) "

    assert c in [1, 2], "Number of Cross Points should be either 1 or 2"

    assert 0 <= e < 1, "Elitism rate should be within range [0, 1)"

    num_drones = int(input("Please Enter the number of drones in the arena: "))

    fits = []
    areas = []

    grid = Grid(sp, p, mu, limit, cross_points=c, elitism=e, num_drones=num_drones)

    g = int(input("Please Enter the number of generations: "))

    print("Genetic algorithm is running... (*THIS COULD TAKE SOME TIME*)")

    b = time.time()
    grid.run(g)
    a = findAreas(grid.best_paths)
    areas.append(a)

    e = time.time()

    print("Total time taken {:.2f} seconds".format(e - b))

    plot_path_fitness(grid)


