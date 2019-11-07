import unittest
import numpy.testing as npt
import os

from networks.scitile.uw_toroidal_shell import toroidal_shell_integral

DEFAULT_TOL = 1e-3
DEFAULT_ATOL = 1e-8


class UWTest(unittest.TestCase):

    def test_torodial_shell_integral(self):

        n = 128
        minval, maxval = -1.25, 1.25
        eps = 1.0e-8
        result = toroidal_shell_integral(n, minval, maxval, eps)
        npt.assert_allclose(result, 3.9926786915581705, rtol=DEFAULT_TOL, atol=DEFAULT_ATOL)
        #self.assertEqual(result, 3.9926786915581705)


if __name__ == '__main__':
    unittest.main()
