import unittest
import numpy.testing as npt
import numpy as np
import os

from conjgrad import cg_solve

DEFAULT_TOL = 2e-2
DEFAULT_ATOL = 2e-2


def make_sparse_inputs(A, b):
    N = len(b)
    spA = np.zeros((N, 18))
    C = np.zeros(18 * N, dtype=int)
    for i in range(N):
        filt = A[i, :] != 0
        Ai = A[i, filt]
        conn = np.where(filt)[0]
        spA[i, :len(Ai)] = Ai
        C[18 * i:18 * i + len(conn)] = conn
    return spA, C


class LASTest(unittest.TestCase):

    def test_cg_eye_rand(self):
        A = np.eye(10)
        b = np.random.rand(10)
        res = cg_solve(A, b)
        sol = np.linalg.solve(A, b)
        filt = np.abs(sol) > 0.01 * np.mean(sol)
        npt.assert_allclose(res[filt], sol[filt], rtol=DEFAULT_TOL, atol=DEFAULT_ATOL)

    # def test_cg_20x20(self):
    #     A = np.array([[2,-1,0,0],[-1,2,-1,0],[0,-1,2,-1],[0,0,-1.,1.]])
    #     b = np.array([0,0,0,1.])
    #     res = cg_solve(A, b)
    #     sol = np.linalg.solve(A, b)
    #     filt = np.abs(sol) > 0.01 * np.mean(sol)
    #     npt.assert_allclose(res[filt], sol[filt], rtol=DEFAULT_TOL, atol=DEFAULT_ATOL)


if __name__ == '__main__':
    unittest.main()
