import timeit
from collections import OrderedDict

import numpy as np
import math
import matplotlib.pyplot as plt
import plaidml2 as plaidml
import plaidml2.edsl as edsl
import plaidml2.exec as plaidml_exec


def sq(X):
    return X * X


def sum(R):
    idxs = edsl.TensorIndexes(3)
    O = edsl.TensorOutput()
    O[()] += R[idxs]
    return O


def meshgrid(
        n,  # number of grid points along each coord direction
        minval,  # coordinate bounding values
        maxval):  # coordinate bounding values
    coordvals = np.linspace(minval, maxval, n, dtype=np.float32)
    delta = (maxval - minval) / (n - 1)  # grid spacing
    X_data = coordvals.reshape(n, 1, 1)
    Y_data = coordvals.reshape(1, n, 1)
    Z_data = coordvals.reshape(1, 1, n)
    X = edsl.Tensor(edsl.LogicalShape(plaidml.DType.FLOAT32, X_data.shape))
    Y = edsl.Tensor(edsl.LogicalShape(plaidml.DType.FLOAT32, Y_data.shape))
    Z = edsl.Tensor(edsl.LogicalShape(plaidml.DType.FLOAT32, Z_data.shape))
    return X, Y, Z, X_data, Y_data, Z_data


def partial(
        F,  # F: differentiable function tensor
        wrt,  # wrt: variable with respect to which we need to differentiate ('x' | 'y' | 'z')
        delta):  # delta: grid spacing
    F_neg = -F
    dims = edsl.TensorDims(3)
    x, y, z = edsl.TensorIndexes(3)
    F.bind_dims(*dims)
    O = edsl.TensorOutput(*dims)
    if wrt == 'x':
        O[x, y, z] = F[x + 1, y, z] + F_neg[x - 1, y, z]
    elif wrt == 'y':
        O[x, y, z] = F[x, y + 1, z] + F_neg[x, y - 1, z]
    elif wrt == 'z':
        O[x, y, z] = F[x, y, z + 1] + F_neg[x, y, z - 1]
    return O / (2.0 * delta)


def grad(
        F,  # F: differentiable function tensor
        delta):  # delta: grid spacing
    return partial(F, 'x', delta) + partial(F, 'y', delta) + partial(F, 'z', delta)


# f-rep of torodial shell f(x, y, z) = (sqrt(x^2 + y^2) - 1)^2 + z^2 + (0.1)^2
def frep_torus(X, Y, Z, R, r):
    F = sq(edsl.sqrt(sq(X) + sq(Y)) - R) + sq(Z) - sq(r)
    return F


def toroidal_shell_integral_moment_of_innertia_exact(R, r):
    return 2 * (math.pi**2) * r * R * ((2 * (R**2)) + (3 * (r**2)))


def torus_volume_exact(R, r):
    return 2 * (math.pi**2) * (r**2) * R


def torus_surface_area_exact(R, r):
    return 4 * (math.pi**2) * r * R


def run_program(X, Y, Z, X_data, Y_data, Z_data, O, benchmark=False):
    program = edsl.Program('torus_volume', [O])
    binder = plaidml_exec.Binder(program)
    executable = binder.compile()

    def run():
        binder.input(X).copy_from_ndarray(X_data)
        binder.input(Y).copy_from_ndarray(Y_data)
        binder.input(Z).copy_from_ndarray(Z_data)
        executable.run()
        return binder.output(O).as_ndarray()

    if benchmark:
        # the first run will compile and run
        print('compiling...')
        result = run()

        # subsequent runs should not include compile time
        print('running...')
        ITERATIONS = 100
        elapsed = timeit.timeit(run, number=ITERATIONS)
        print('runtime:', elapsed / ITERATIONS)
    else:
        result = run()
    return result


def toroidal_shell_moment_of_inertia(
        n,  # number of grid points along each coord direction
        minval,  # coordinate bounding values
        maxval,  # coordinate bounding values
        eps,  # Threshold for trivial gradient
        R,  # major radius
        r,  # minor radius
        benchmark=False):  # benchmark: get timing information

    delta = (maxval - minval) / (n - 1)  # grid spacing

    X, Y, Z, X_data, Y_data, Z_data = meshgrid(n, minval, maxval)
    F = frep_torus(X, Y, Z, R, r)

    # moment of inertia about z axis at each point g(x, y, z) = x^2 + y^2
    G = sq(X) + sq(Y)

    DFDX = partial(F, 'x', delta)
    DFDY = partial(F, 'y', delta)
    DFDZ = partial(F, 'z', delta)

    # chi: occupancy function: 1 inside the region (f<0), 0 outside the region (f>0)
    CHI = edsl.select(F > 0, 0, 1)
    DCHIDX = partial(CHI, 'x', delta)
    DCHIDY = partial(CHI, 'y', delta)
    DCHIDZ = partial(CHI, 'z', delta)

    NUMER = DFDX * DCHIDX + DFDY * DCHIDY + DFDZ * DCHIDZ
    DENOM = edsl.sqrt(sq(DFDX) + sq(DFDY) + sq(DFDZ))
    H = edsl.select(DENOM < eps, 0, NUMER / DENOM)
    O = sum(-G * H)

    result = run_program(X, Y, Z, X_data, Y_data, Z_data, O, benchmark)

    return result * (delta**3)


def torus_surface_area(
        n,  # number of grid points along each coord direction
        minval,  # coordinate bounding values
        maxval,  # coordinate bounding values
        eps,  # Threshold for trivial gradient
        R,  # major radius
        r,  # minor radius
        G,  # integrand functino TODO: pull out integrand
        benchmark=False):  # benchmark: get timing information

    delta = (maxval - minval) / (n - 1)  # grid spacing

    X, Y, Z, X_data, Y_data, Z_data = meshgrid(n, minval, maxval)
    F = frep_torus(X, Y, Z, R, r)

    DFDX = partial(F, 'x', delta)
    DFDY = partial(F, 'y', delta)
    DFDZ = partial(F, 'z', delta)

    # chi: occupancy function: 1 inside the region (f<0), 0 outside the region (f>0)
    CHI = edsl.select(F > 0, 0, 1)
    DCHIDX = partial(CHI, 'x', delta)
    DCHIDY = partial(CHI, 'y', delta)
    DCHIDZ = partial(CHI, 'z', delta)

    NUMER = DFDX * DCHIDX + DFDY * DCHIDY + DFDZ * DCHIDZ
    DENOM = edsl.sqrt(sq(DFDX) + sq(DFDY) + sq(DFDZ))
    H = edsl.select(DENOM < eps, 0, NUMER / DENOM)
    O = sum(-H * G)

    result = run_program(X, Y, Z, X_data, Y_data, Z_data, O, benchmark)
    return result * (delta**3)


def torus_volume(
        n,  # number of grid points along each coord direction
        minval,  # coordinate bounding values
        maxval,  # coordinate bounding values
        eps,  # Threshold for trivial gradient
        R,  # major radius
        r,  # minor radius
        benchmark=False):  # benchmark: get timing information
    # coordvals = np.linspace(minval, maxval, n, dtype=np.float32)
    delta = (maxval - minval) / (n - 1)  # grid spacing

    X, Y, Z, X_data, Y_data, Z_data = meshgrid(n, minval, maxval)
    F = frep_torus(X, Y, Z, R, r)

    PHI = (X + Y + Z) / 3.0
    # chi: occupancy function: 1 inside the region (f<0), 0 outside the region (f>0)
    CHI = edsl.select(F > 0, 0, 1)

    DelCHI = grad(CHI, delta)
    O = sum(-PHI * DelCHI)

    result = run_program(X, Y, Z, X_data, Y_data, Z_data, O, benchmark)

    return result * (delta**3)


def main(
):  #TODO:work in progress to generate graphs will clean up after documentation is completed
    R = 1.0  # major radius
    r = 0.1  # minor radius
    #N = 128  # number of grid points
    error_chart = np.array([])
    error_chart_N_by_delta = np.array([])
    for N in range(32, 512, 64):
        print("evaluating for N: {}".format(N))
        minval = -1.25 * R
        maxval = 1.25 * R
        delta = (maxval - minval) / (N - 1)  # grid spacing
        eps = 1.0e-8
        #compute exact value
        #exact_value = toroidal_shell_integral_moment_of_innertia_exact(R, r)
        exact_value = torus_volume_exact(R, r)
        print(exact_value)
        #run integral computation
        print("Exact value: {}".format(exact_value))
        #compare the result
        #result = toroidal_shell_moment_of_inertia(N, minval, maxval, eps, R, r)
        result = torus_volume(N, minval, maxval, eps, R, r)
        print("computed result using integral: {}".format(result))
        error = (abs(result - exact_value) / exact_value) * 100
        print("error: {} %".format(error))
        error_chart = np.append(error_chart, math.log(error))
        error_chart_N_by_delta = np.append(error_chart_N_by_delta, math.log(R / delta))

    fig = plt.figure()
    ax = plt.axes()
    ax.set(
        xlabel='log(R/delta)',
        ylabel='log(error percentage)',
        title=
        'Result of convergence study genus 1 grid-based evaluationof moment of innertia of toroidal shell'
    )
    ax.scatter(error_chart_N_by_delta, error_chart)
    plt.show()


if __name__ == '__main__':
    main()
