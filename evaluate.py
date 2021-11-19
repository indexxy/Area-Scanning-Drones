import numpy as np
from numba import guvectorize


def manhattan(x0, y0, xf, yf):
    return np.abs(x0 - xf) + np.abs(y0 - yf)


def findAreas(paths):
    paths = paths.reshape(paths.shape[1], -1, paths.shape[-1])
    areas = np.empty(paths.shape[0], dtype=np.int32)

    X = paths[:, :, 0]

    __findAreas(paths, X, areas)

    return areas


@guvectorize(
    "void(int32[:, :], int32[:], int32[:])",
    "(n, m), (n) -> ()",
    cache=True
)
def __findAreas(paths, X, out):
    res = 0
    for x in np.unique(X):
        masked_arr = paths[paths[:, 0] == x]
        res += np.unique(masked_arr[:, -1]).size

    out[0] = res
