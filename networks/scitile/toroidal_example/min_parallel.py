import math
import numpy as np
from numba import jit, cuda, float32
EPS = 1.0e-8  # Threshold for trivial gradient
RAD = 1  # stencil radius


# f(x,y,z)=0 implicitly defines domain of integration
@cuda.jit(device=True)
def f(x0, y0, z0):
    #return x0**2 + y0**2 + z0**2 -1. # unit sphere
    return (math.sqrt(x0**2 + y0**2) - 1.)**2 + z0**2 - 0.1**2  #torus with a=1, b=.1


# integrand g(x,y,z). Here choose g=x^2+y^2 for moment of inertia about z-axis
@cuda.jit(device=True)
def g(x0, y0, z0):
    return x0**2 + y0**2


# occupancy function: 1 inside the region (f<0), 0 outside the region (f>0)
@cuda.jit(device=True)
def chi(f):
    return f < 0


# kernel to evaluate and store grid of values for f and g
@cuda.jit("void(float64[:,:,:],float64[:,:,:],float64[:], float64[:], float64[:])")
def fgKernel3D(d_f, d_g, d_x, d_y, d_z):
    i, j, k = cuda.grid(3)
    m, n, p = d_f.shape
    if i < m and j < n and k < p:
        d_f[i, j, k] = f(d_x[i], d_y[j], d_z[k])
        d_g[i, j, k] = g(d_x[i], d_y[j], d_z[k])


# kernel to compute integral of g on f=0 surface
@cuda.jit("void(float64[:,:,:],float64[:,:,:],float64[:,:,:], float64)")
def stencilKernel3D(d_rho, d_f, d_g, delta):
    i, j, k = cuda.grid(3)
    m, n, p = d_f.shape
    # set contributions to zero for edge cases
    if i == 0 or j == 0 or k == 0 or i == m or j == n or k == p:
        d_rho[i, j, k] = 0.
        return
    if i >= RAD and j >= RAD and k >= RAD and i < m - RAD and j < n - RAD and k < p - RAD:
        # access neighbor data along each axis
        left, right = d_f[i - 1, j, k], d_f[i + 1, j, k]
        front, back = d_f[i, j - 1, k], d_f[i, j + 1, k]
        down, up = d_f[i, j, k - 1], d_f[i, j, k + 1]
        # compute central difference estimate of first derivatives of f
        dfdx = (right - left) / (2 * delta)
        dfdy = (back - front) / (2 * delta)
        dfdz = (up - down) / (2 * delta)
        # compute central difference estimate of first derivatives of occupancy
        dchidx = (chi(right) - chi(left)) / (2 * delta)
        dchidy = (chi(back) - chi(front)) / (2 * delta)
        dchidz = (chi(up) - chi(down)) / (2 * delta)
        # denominator is magnitude of grad(f)
        denom = math.sqrt(dfdx * dfdx + dfdy * dfdy + dfdz * dfdz)
        if denom < EPS:
            d_rho[
                i, j,
                k] = 0.  # Avoid singularities: if denominator would vanish, set contribution to zero
        else:
            numer = dfdx * dchidx + dfdy * dchidy + dfdz * dchidz  # numerator is grad(f).grad(chi)
            d_rho[i, j, k] = -1. * d_g[
                i, j, k] * numer / denom  # grid point contribution for non-zero denominator


# reduction to sum contributions from the grid points
@cuda.reduce
def sum_reduce(a, b):
    return a + b


# wrapper function to set up and call integration kernel
def gridintegrate(coordvals, delta):
    m = coordvals.shape[0]  # grid dimensions
    # copy coordinate arrays to device
    d_x = cuda.to_device(coordvals)
    d_y = cuda.to_device(coordvals)
    d_z = cuda.to_device(coordvals)
    # create device arrays for values of inputs f, g and output rho
    d_f = cuda.device_array(shape=[m, m, m], dtype=np.float64)
    d_g = cuda.device_array(shape=[m, m, m], dtype=np.float64)
    d_rho = cuda.device_array(shape=[m, m, m], dtype=np.float64)
    TPB = 8  # threads per block
    B = (m + TPB - 1) // TPB  # blocks in the grid
    tpb = TPB, TPB, TPB
    bpg = B, B, B
    fgKernel3D[bpg, tpb](d_f, d_g, d_x, d_y, d_z)  #compute and store values of inputs
    stencilKernel3D[bpg, tpb](d_rho, d_f, d_g, delta)  # compute and store grid point contributions
    return sum_reduce(np.reshape(d_rho, m * m * m)) * delta**3  # sum grid point contributions
