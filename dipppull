import sys
import time
from numba import jit, prange, set_num_threads, get_num_threads
from numpy import sqrt


@jit(nopython=True, parallel=True)
def main(nmax):
    pibyfour = 0.0
    dx = 1.0 / nmax

    for i in prange(nmax):
        pibyfour += sqrt(1 - pow((i * dx), 2))

    pi = 4.0 * pibyfour * dx

    return pi


if __name__ == '__main__':

    if len(sys.argv) == 3:

        nmax = int(sys.argv[1])
        nthreads = int(sys.argv[2])

        # Set the number of Numba threads
        set_num_threads(nthreads)

        # First call: compile the function with Numba.
        # This is deliberately NOT timed.
        main(nmax)

        # Second call: the function is already compiled.
        # This is the calculation we actually benchmark.
        initial = time.perf_counter()

        pi = main(nmax)

        final = time.perf_counter()

        print("Requested threads:", nthreads)
        print("Numba threads used:", get_num_threads())
        print("Pi = {:18.16f}".format(pi))
        print("Elapsed time: {:8.6f} s".format(final - initial))

    else:
        print("Usage: python {} <ITERATIONS> <THREADS>".format(sys.argv[0]))
