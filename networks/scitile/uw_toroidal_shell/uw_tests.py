import unittest
import numpy.testing as npt
import os
import math
from networks.scitile.uw_toroidal_shell.uw_toroidal_shell import toroidal_shell_integral_moment_of_innertia_exact
from networks.scitile.uw_toroidal_shell.uw_toroidal_shell import torus_surface_area_exact
from networks.scitile.uw_toroidal_shell.uw_toroidal_shell import integral_surface_area
from networks.scitile.uw_toroidal_shell.uw_toroidal_shell import torus_volume_exact
from networks.scitile.uw_toroidal_shell.uw_toroidal_shell import integral_volume
from networks.scitile.uw_toroidal_shell.uw_toroidal_shell import torus
from networks.scitile.uw_toroidal_shell.uw_toroidal_shell import integrand_empty
from networks.scitile.uw_toroidal_shell.uw_toroidal_shell import integrand_inertia

DEFAULT_TOL = 1e-2
DEFAULT_ATOL = 1e-8


class UWTest(unittest.TestCase):

    def test_toroidal_shell_moment_of_inertia(self):
        R = 10.0  # major radius
        r = 2.0  # minor radius
        N = 128  # number of grid points
        minval = -1.25 * R
        maxval = 1.25 * R
        # G = 0  # Daubechies wavelet genus ( 1 <= G <= 7 ) #TODO: add genus
        eps = 1.0e-8
        exact_value = toroidal_shell_integral_moment_of_innertia_exact(R, r)
        result = integral_surface_area(N, minval, maxval, eps, torus, [R, r], integrand_inertia, 1)
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
        result = integral_surface_area(N, minval, maxval, eps, torus, [R, r], integrand_empty, 1)
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


if __name__ == '__main__':
    unittest.main()
