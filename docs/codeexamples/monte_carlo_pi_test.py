import unittest
import numpy.testing as npt
import os

from networks.scitile.monte_carlo.pi import monte_carlo_pi

DEFAULT_TOL = 1e-3
DEFAULT_ATOL = 1e-8


class PITest(unittest.TestCase):

    def test_monte_carlo_pi(self):
        result = monte_carlo_pi(10000)
        npt.assert_allclose(result, 3.14159265359, rtol=DEFAULT_TOL, atol=DEFAULT_ATOL)


if __name__ == '__main__':
    unittest.main()