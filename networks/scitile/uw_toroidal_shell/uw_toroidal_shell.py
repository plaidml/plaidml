import timeit
from collections import OrderedDict

import numpy as np

import plaidml
import plaidml.edsl as edsl
import plaidml.exec


def sq(X):
    return X * X


def partial(F, wrt, delta):
    F_neg = -F
    dims = edsl.TensorDims(3)
    x, y, z = edsl.TensorIndexes(3)
    F.bind_dims(*dims)
    OC = edsl.Contraction().outShape(*dims)
    if wrt == 'x':
        O = OC.outAccess(x, y, z).assign(F[x + 1, y, z] + F_neg[x - 1, y, z]).build()
    elif wrt == 'y':
        O = OC.outAccess(x, y, z).assign(F[x, y + 1, z] + F_neg[x, y - 1, z]).build()
    elif wrt == 'z':
        O = OC.outAccess(x, y, z).assign(F[x, y, z + 1] + F_neg[x, y, z - 1]).build()
    return O / (2.0 * delta)


def partial_chi(F, wrt, delta):
    dims = edsl.TensorDims(3)
    x, y, z = edsl.TensorIndexes(3)
    F.bind_dims(*dims)
    DF_left_C = edsl.Contraction().outShape(*dims)
    DF_right_C = edsl.Contraction().outShape(*dims)

    if wrt == 'x':
        DF_right = DF_right_C.outAccess(x, y, z).assign(F[x + 1, y, z]).build()
        DF_left = DF_left_C.outAccess(x, y, z).assign(F[x - 1, y, z]).build()
    elif wrt == 'y':
        DF_right = DF_right_C.outAccess(x, y, z).assign(F[x, y + 1, z]).build()
        DF_left = DF_left_C.outAccess(x, y, z).assign(F[x, y - 1, z]).build()
    elif wrt == 'z':
        DF_right = DF_right_C.outAccess(x, y, z).assign(F[x, y, z + 1]).build()
        DF_left = DF_left_C.outAccess(x, y, z).assign(F[x, y, z - 1]).build()

    one = edsl.cast(1, F.dtype)
    zero = edsl.cast(1, F.dtype)
    neg_one = edsl.cast(-1, F.dtype)

    DF_chi_right = edsl.select(DF_right < 0, one, zero)
    DF_chi_left = edsl.select(DF_left < 0, neg_one, zero)
    return (DF_chi_right + DF_chi_left) / (2.0 * delta)


def sum(R):
    idxs = edsl.TensorIndexes(3)
    return edsl.Contraction().sum(R[idxs]).build()


# n: number of grid points along each coord direction
# minval, maxval: coordinate bounding values
# eps: Threshold for trivial gradient
def toroidal_shell_integral(n, minval, maxval, eps, benchmark=False):
    coordvals = np.linspace(minval, maxval, n, dtype=np.float32)
    delta = (maxval - minval) / (n - 1)  # grid spacing

    X_data = coordvals.reshape(n, 1, 1)
    Y_data = coordvals.reshape(1, n, 1)
    Z_data = coordvals.reshape(1, 1, n)
    X = edsl.Placeholder(plaidml.DType.FLOAT32, X_data.shape)
    Y = edsl.Placeholder(plaidml.DType.FLOAT32, Y_data.shape)
    Z = edsl.Placeholder(plaidml.DType.FLOAT32, Z_data.shape)

    # f-rep of torodial shell f(x, y, z) = (sqrt(x^2 + y^2) - 1)^2 + z^2 + (0.1)^2
    F = sq(edsl.sqrt(sq(X) + sq(Y)) - 1.0) + sq(Z) - sq(0.1)
    # moment of inertia about z axis at each point g(x, y, z) = x^2 + y^2
    G = sq(X) + sq(Y)

    DFDX = partial(F, 'x', delta)
    DFDY = partial(F, 'y', delta)
    DFDZ = partial(F, 'z', delta)

    # chi: occupancy function: 1 inside the region (f<0), 0 outside the region (f>0)
    DCHIDX = partial_chi(F, 'x', delta)
    DCHIDY = partial_chi(F, 'y', delta)
    DCHIDZ = partial_chi(F, 'z', delta)

    NUMER = DFDX * DCHIDX + DFDY * DCHIDY + DFDZ * DCHIDZ
    DENOM = edsl.sqrt(sq(DFDX) + sq(DFDY) + sq(DFDZ))
    zero = edsl.cast(0, NUMER.dtype)
    H = edsl.select(DENOM < eps, zero, NUMER / DENOM)
    O = sum(-G * H)

    program = plaidml.Program('toroidal_shell_integral', [X, Y, Z], [O])
    runner = plaidml.exec.Runner(program)

    def run():
        outputs = runner.run([X_data, Y_data, Z_data])
        return outputs[0]

    if benchmark:
        print('running...')
        ITERATIONS = 100
        elapsed = timeit.timeit(run, number=ITERATIONS)
        print('runtime:', elapsed / ITERATIONS)

    result = run()
    return result * (delta**3)


def main():
    result = toroidal_shell_integral(128, -1.25, 1.25, 1.0e-8, True)
    print("computed result: {}".format(result))


if __name__ == '__main__':
    main()
