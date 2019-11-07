import os
from collections import OrderedDict

import numpy as np

os.environ['KERAS_BACKEND'] = 'plaidml2.bridge.keras'

import keras.backend as K
import plaidml2.edsl as edsl


def function_g(x, y, z):
    X = x.tensor
    Y = y.tensor
    O = X * X + Y * Y
    return K._KerasNode('function_g', tensor=O)


def function_f(x, y, z):
    X = x.tensor
    Y = y.tensor
    Z = z.tensor
    Z_square = (Z * Z) - 0.01
    O = edsl.pow(edsl.sqrt(X * X + Y * Y) - 1.0, 2) + (Z_square)
    return K._KerasNode('function_f', tensor=O)


def partial(f, wrt, delta):
    F = f.tensor
    F_neg = -F
    dims = edsl.TensorDims(F.shape.ndims)
    x, y, z = edsl.TensorIndexes(3)
    F.bind_dims(*dims)
    O = edsl.TensorOutput(*dims)
    if wrt == 'x':
        O[x, y, z] = F[x + 1, y, z] + F_neg[x - 1, y, z]
    elif wrt == 'y':
        O[x, y, z] = F[x, y + 1, z] + F_neg[x, y - 1, z]
    elif wrt == 'z':
        O[x, y, z] = F[x, y, z + 1] + F_neg[x, y, z - 1]
    O = O / (2.0 * delta)
    return K._KerasNode('df_dx', tensor=O)


def partial_chi(f, wrt, delta):
    F = f.tensor
    dims = edsl.TensorDims(F.shape.ndims)
    x, y, z = edsl.TensorIndexes(3)
    F.bind_dims(*dims)
    DF_left = edsl.TensorOutput(*dims)
    DF_right = edsl.TensorOutput(*dims)

    if wrt == 'x':
        DF_right[x, y, z] = F[x + 1, y, z]
        DF_left[x, y, z] = F[x - 1, y, z]
    elif wrt == 'y':
        DF_right[x, y, z] = F[x, y + 1, z]
        DF_left[x, y, z] = F[x, y - 1, z]
    elif wrt == 'z':
        DF_right[x, y, z] = F[x, y, z + 1]
        DF_left[x, y, z] = F[x, y, z - 1]

    DF_chi_right = edsl.select(DF_right < 0, 1, 0)
    DF_chi_left = edsl.select(DF_left < 0, -1, 0)
    Intermediate = DF_chi_right + DF_chi_left
    O = Intermediate / (2.0 * delta)
    return K._KerasNode('df_chi_dx', tensor=O)


def dot_self(dfdx, dfdy, dfdz):
    DFDX = dfdx.tensor
    DFDY = dfdy.tensor
    DFDZ = dfdz.tensor
    O = DFDX * DFDX + DFDY * DFDY + DFDZ * DFDZ
    return K._KerasNode('dot', tensor=O)


def dot(a, b, c, d, e, f):
    A = a.tensor
    B = b.tensor
    C = c.tensor
    D = d.tensor
    E = e.tensor
    F = f.tensor
    O = A * D + B * E + C * F
    return K._KerasNode('dottwo', tensor=O)


def root(f):
    F = f.tensor
    F_neg = -F
    dims = edsl.TensorDims(F.shape.ndims)
    O = edsl.sqrt(F)
    return K._KerasNode('root', tensor=O)


def divide(num, denom, eps):
    NUM = num.tensor
    DEN = denom.tensor
    idxs = edsl.TensorIndexes(3)
    empty_idxs = edsl.TensorIndexes(0)
    O = edsl.select(DEN < eps, 0, NUM / DEN)
    return K._KerasNode('divide', tensor=O)


def rho(g, h):
    G = g.tensor
    H = h.tensor
    G_neg = -G
    dims = edsl.TensorDims(H.shape.ndims)
    O = edsl.TensorOutput(dims)
    O = G_neg * H
    return K._KerasNode('rho', tensor=O)


def sumall(rho):
    R = rho.tensor
    dims = edsl.TensorDims(R.shape.ndims)
    R.bind_dims(*dims)
    idxs = edsl.TensorIndexes(3)
    O = edsl.TensorOutput()
    O[[]] += R[idxs]
    return K._KerasNode('sumall', tensor=O)


def toroidal_shell_integral(
        n,  #number of grid points along each coord direction
        minval,
        maxval,  #minval , maxval : coordinate bounding values
        eps):  # eps : Threshold for trivial gradient
    coordvals = np.linspace(minval, maxval, n, dtype=np.float32)
    delta = (maxval - minval) / (n - 1)  # grid spacing

    X = K.variable(coordvals.reshape(n, 1, 1))
    Y = K.variable(coordvals.reshape(1, n, 1))
    Z = K.variable(coordvals.reshape(1, 1, n))

    F = function_f(X, Y,
                   Z)  # f-rep of torodial shell f(x,y,z) = (sqrt(x^2+y^2)-1)^2 + z^2 + (0.1)^2
    G = function_g(X, Y, Z)  # moment of innertia about z axis at each point g(x,y,z) = x^2 + y^2

    DFDX = partial(F, 'x', delta)
    DFDY = partial(F, 'y', delta)
    DFDZ = partial(F, 'z', delta)

    DOT = dot_self(DFDX, DFDY, DFDZ)
    DENOM = root(DOT)

    # chi: occupancy function: 1 inside the region (f<0), 0 outside the region (f>0)
    DCHIDX = partial_chi(F, 'x', delta)
    DCHIDY = partial_chi(F, 'y', delta)
    DCHIDZ = partial_chi(F, 'z', delta)

    NUMER = dot(DFDX, DFDY, DFDZ, DCHIDX, DCHIDY, DCHIDZ)
    H = divide(NUMER, DENOM, eps)
    RHO = rho(G, H)
    ANS = sumall(RHO)
    result = ANS.eval()
    result = result * (delta**3)
    return result


def main():
    n = 128
    minval, maxval = -1.25, 1.25
    eps = 1.0e-8
    result = toroidal_shell_integral(n, minval, maxval, eps)
    print("computed result: ")
    print(result)


if __name__ == '__main__':
    main()
