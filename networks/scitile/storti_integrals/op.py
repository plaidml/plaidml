import timeit
from collections import OrderedDict

import numpy as np
import math
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


def sum_4D(R):
    idxs = edsl.TensorIndexes(4)
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


def meshgrid_4D(
        n,  # number of grid points along each coord direction
        minval,  # coordinate bounding values
        maxval):  # coordinate bounding values
    coordvals = np.linspace(minval, maxval, n, dtype=np.float32)
    delta = (maxval - minval) / (n - 1)  # grid spacing
    X_data = coordvals.reshape(n, 1, 1, 1)
    Y_data = coordvals.reshape(1, n, 1, 1)
    Z_data = coordvals.reshape(1, 1, n, 1)
    W_data = coordvals.reshape(1, 1, 1, n)

    X = edsl.Tensor(edsl.LogicalShape(plaidml.DType.FLOAT32, X_data.shape))
    Y = edsl.Tensor(edsl.LogicalShape(plaidml.DType.FLOAT32, Y_data.shape))
    Z = edsl.Tensor(edsl.LogicalShape(plaidml.DType.FLOAT32, Z_data.shape))
    W = edsl.Tensor(edsl.LogicalShape(plaidml.DType.FLOAT32, W_data.shape))
    return X, Y, Z, W, X_data, Y_data, Z_data, W_data


def partial(
        F,  # F: differentiable function tensor
        wrt,  # wrt: ('x' | 'y' | 'z')
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


def partial_4D(
        F,  # F: differentiable function tensor
        wrt,  # wrt: ('x' | 'y' | 'z')
        delta):  # delta: grid spacing
    F_neg = -F
    dims = edsl.TensorDims(4)
    x, y, z, w = edsl.TensorIndexes(4)
    F.bind_dims(*dims)
    O = edsl.TensorOutput(*dims)
    if wrt == 'x':
        O[x, y, z, w] = F[x + 1, y, z, w] + F_neg[x - 1, y, z, w]
    elif wrt == 'y':
        O[x, y, z, w] = F[x, y + 1, z, w] + F_neg[x, y - 1, z, w]
    elif wrt == 'z':
        O[x, y, z, w] = F[x, y, z + 1, w] + F_neg[x, y, z - 1, w]
    elif wrt == 'w':
        O[x, y, z, w] = F[x, y, z, w + 1] + F_neg[x, y, z, w - 1]
    return O / (2.0 * delta)


def grad(
        F,  # F: differentiable function tensor
        delta):  # delta: grid spacing
    return partial(F, 'x', delta) + partial(F, 'y', delta) + partial(F, 'z', delta)


def grad_4D(
        F,  # F: differentiable function tensor
        delta):  # delta: grid spacing
    return partial_4D(F, 'x', delta) + partial_4D(F, 'y', delta) + partial_4D(
        F, 'z', delta) + partial_4D(F, 'w', delta)


def integrand_empty(X, Y, Z):
    return 1


def integrand_empty_4D(X, Y, Z, W):
    return 1


def integral_surface_area(
        n,  # number of grid points along each coord direction
        minval,  # coordinate bounding values
        maxval,  # coordinate bounding values
        eps,  # Threshold for trivial gradient
        frep,  # function 
        frep_vars,  #functno rep variables
        integrand,  # integrand 
        benchmark=False):  # benchmark: get timing information

    delta = (maxval - minval) / (n - 1)  # grid spacing

    X, Y, Z, X_data, Y_data, Z_data = meshgrid(n, minval, maxval)
    F = frep(X, Y, Z, frep_vars)
    G = integrand(X, Y, Z)

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


def integral_surface_area_4D(
        n,  # number of grid points along each coord direction
        minval,  # coordinate bounding values
        maxval,  # coordinate bounding values
        eps,  # Threshold for trivial gradient
        frep,  # function 
        frep_vars,  #functno rep variables
        integrand,  # integrand 
        benchmark=False):  # benchmark: get timing information

    delta = (maxval - minval) / (n - 1)  # grid spacing

    X, Y, Z, W, X_data, Y_data, Z_data, W_data = meshgrid_4D(n, minval, maxval)
    F = frep(X, Y, Z, W, frep_vars)
    G = integrand(X, Y, Z, W)

    DFDX = partial_4D(F, 'x', delta)
    DFDY = partial_4D(F, 'y', delta)
    DFDZ = partial_4D(F, 'z', delta)
    DFDW = partial_4D(F, 'w', delta)

    # chi: occupancy function: 1 inside the region (f<0), 0 outside the region (f>0)
    CHI = edsl.select(F > 0, 0, 1)
    DCHIDX = partial_4D(CHI, 'x', delta)
    DCHIDY = partial_4D(CHI, 'y', delta)
    DCHIDZ = partial_4D(CHI, 'z', delta)
    DCHIDW = partial_4D(CHI, 'w', delta)

    NUMER = DFDX * DCHIDX + DFDY * DCHIDY + DFDZ * DCHIDZ + DFDW * DCHIDW
    DENOM = edsl.sqrt(sq(DFDX) + sq(DFDY) + sq(DFDZ) + sq(DFDW))
    H = edsl.select(DENOM < eps, 0, NUMER / DENOM)
    O = sum_4D(-H * G)

    result = run_program_4D(X, Y, Z, W, X_data, Y_data, Z_data, W_data, O, benchmark)
    return result * (delta**4)


def integral_volume(
        n,  # number of grid points along each coord direction
        minval,  # coordinate bounding values
        maxval,  # coordinate bounding values
        eps,  # Threshold for trivial gradient
        frep,  # function rep
        frep_vars,  # function rep variables
        benchmark=False):  # benchmark: get timing information

    delta = (maxval - minval) / (n - 1)  # grid spacing

    X, Y, Z, X_data, Y_data, Z_data = meshgrid(n, minval, maxval)
    F = frep(X, Y, Z, frep_vars)

    PHI = (X + Y + Z) / 3.0
    # chi: occupancy function: 1 inside the region (f<0), 0 outside the region (f>0)
    CHI = edsl.select(F > 0, 0, 1)

    DelCHI = grad(CHI, delta)
    O = sum(-PHI * DelCHI)

    result = run_program(X, Y, Z, X_data, Y_data, Z_data, O, benchmark)

    return result * (delta**3)


def integral_volume_4D(
        n,  # number of grid points along each coord direction
        minval,  # coordinate bounding values
        maxval,  # coordinate bounding values
        eps,  # Threshold for trivial gradient
        frep,  # function rep
        frep_vars,  # function rep variables
        benchmark=False):  # benchmark: get timing information

    delta = (maxval - minval) / (n - 1)  # grid spacing

    X, Y, Z, W, X_data, Y_data, Z_data, W_data = meshgrid_4D(n, minval, maxval)
    F = frep(X, Y, Z, W, frep_vars)

    PHI = (X + Y + Z + W) / 4.0
    # chi: occupancy function: 1 inside the region (f<0), 0 outside the region (f>0)
    CHI = edsl.select(F > 0, 0, 1)

    DelCHI = grad_4D(CHI, delta)
    O = sum_4D(-PHI * DelCHI)

    result = run_program_4D(X, Y, Z, W, X_data, Y_data, Z_data, W_data, O, benchmark)

    return result * (delta**4)


def run_program(X, Y, Z, X_data, Y_data, Z_data, O, benchmark=False):
    program = edsl.Program('integral_program', [O])
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


def run_program_4D(X, Y, Z, W, X_data, Y_data, Z_data, W_data, O, benchmark=False):
    program = edsl.Program('4D_integral_program', [O])
    binder = plaidml_exec.Binder(program)
    executable = binder.compile()

    def run():
        binder.input(X).copy_from_ndarray(X_data)
        binder.input(Y).copy_from_ndarray(Y_data)
        binder.input(Z).copy_from_ndarray(Z_data)
        binder.input(W).copy_from_ndarray(W_data)
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