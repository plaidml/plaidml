import unittest
import numpy.testing as npt
import os
import math
from networks.scitile.storti_integrals.torus import toroidal_shell_integral_moment_of_innertia_exact
from networks.scitile.storti_integrals.torus import torus_surface_area_exact
from networks.scitile.storti_integrals.torus import torus_volume_exact
from networks.scitile.storti_integrals.torus import torus
from networks.scitile.storti_integrals.torus import integrand_inertia
from networks.scitile.storti_integrals.hypersphere import hypersphere
from networks.scitile.storti_integrals.hypersphere import hypersphere_area_exact
from networks.scitile.storti_integrals.hypersphere import hypersphere_volume_exact
from networks.scitile.storti_integrals.op import *

DEFAULT_TOL = 1e-2
DEFAULT_ATOL = 1e-8


class TorusTest(unittest.TestCase):

    def test_toroidal_shell_moment_of_inertia(self):
        R = 10.0  # major radius
        r = 2.0  # minor radius
        N = 128  # number of grid points
        minval = -1.25 * R
        maxval = 1.25 * R
        # G = 0  # Daubechies wavelet genus ( 1 <= G <= 7 ) #TODO: add genus
        eps = 1.0e-8
        exact_value = toroidal_shell_integral_moment_of_innertia_exact(R, r)
        result = integral_surface_area(N, minval, maxval, eps, torus, [R, r], integrand_inertia)
        npt.assert_allclose(result, exact_value, rtol=DEFAULT_TOL, atol=DEFAULT_ATOL)

    def test_torus_surface_area(self):
        R = 10.0  # major radius
        r = 2.0  # minor radius
        N = 128  # number of grid points
        minval = -1.25 * R
        maxval = 1.25 * R
        # G = 0  # Daubechies wavelet genus ( 1 <= G <= 7 ) #TODO: add genus
        eps = 1.0e-8
        exact_value = torus_surface_area_exact(R, r)
        result = integral_surface_area(N, minval, maxval, eps, torus, [R, r], integrand_empty)
        npt.assert_allclose(result, exact_value, rtol=DEFAULT_TOL, atol=DEFAULT_ATOL)

    def test_torus_volume(self):
        R = 10.0  # major radius
        r = 2.0  # minor radius
        N = 128  # number of grid points
        minval = -1.25 * R
        maxval = 1.25 * R
        # G = 0  # Daubechies wavelet genus ( 1 <= G <= 7 ) #TODO: add genus
        eps = 1.0e-8
        exact_value = torus_volume_exact(R, r)
        result = integral_volume(N, minval, maxval, eps, torus, [R, r])
        npt.assert_allclose(result, exact_value, rtol=DEFAULT_TOL, atol=DEFAULT_ATOL)

    def test_hypershpere_area(self):
        R = 1.0  # radius
        p = 2.0  # Lp distance measure
        N = 32
        minval = -1.25 * R
        maxval = 1.25 * R
        eps = 1.0e-8
        exact_value = hypersphere_area_exact(R)
        result = integral_surface_area_4D(N, minval, maxval, eps, hypersphere, [R, p],
                                          integrand_empty_4D)
        npt.assert_allclose(result, exact_value, rtol=DEFAULT_TOL, atol=DEFAULT_ATOL)

    def test_hypershpere_volume(self):
        R = 1.0  # radius
        p = 2.0  # Lp distance measure
        N = 32
        minval = -1.25 * R
        maxval = 1.25 * R
        eps = 1.0e-8
        exact_value = hypersphere_volume_exact(R)
        result = integral_volume_4D(N, minval, maxval, eps, hypersphere, [R, p])
        npt.assert_allclose(result, exact_value, rtol=DEFAULT_TOL, atol=DEFAULT_ATOL)


if __name__ == '__main__':
    unittest.main()
