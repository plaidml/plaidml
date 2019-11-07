import timeit

import numpy as np

from min_parallel import gridintegrate

N = 128  # number of grid points along each coord direction
MINVAL, MAXVAL = -1.25, 1.25  # coordinate bounding values


def main():
    coordvals = np.linspace(MINVAL, MAXVAL, N, dtype=np.float64)
    delta = (MAXVAL - MINVAL) / (N - 1)  # grid spacing

    def run():
        return gridintegrate(coordvals, delta)

    result = run()
    print('Value of the integral is ', result)

    ITERATIONS = 100
    elapsed = timeit.timeit(run, number=ITERATIONS)
    print('runtime:', elapsed / ITERATIONS)


if __name__ == '__main__':
    main()
