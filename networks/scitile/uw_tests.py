import unittest
import numpy.testing as npt
import os

from networks.scitile.uw_toroidal_shell import toroidal_shell_integral

DEFAULT_TOL = 1e-3
DEFAULT_ATOL = 1e-8


class UWTest(unittest.TestCase):

    def test_torodial_shell_integral(self):
        result = toroidal_shell_integral(128, -1.25, 1.25, 1.0e-8)
        npt.assert_allclose(result, 3.9926786915581705, rtol=DEFAULT_TOL, atol=DEFAULT_ATOL)


if __name__ == '__main__':
    unittest.main()
