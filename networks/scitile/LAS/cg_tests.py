import unittest
import numpy.testing as npt
import numpy as np
import os

from networks.scitile.LAS.conjgrad import cg_solve
from networks.scitile.LAS.conjgrad import cg_solve_sp

DEFAULT_TOL = 2e-2
DEFAULT_ATOL = 2e-2

def make_sparse_inputs(A, b):
    N = len(b)
    spA = np.zeros((N, 18))
    C = np.zeros(18*N, dtype=int)
    for i in range(N):
        filt = A[i,:] != 0
        Ai = A[i,filt]
        conn = np.where(filt)[0]
        spA[i,:len(Ai)] = Ai
        C[18*i:18*i+len(conn)] = conn
    return spA, C


class UWTest(unittest.TestCase):

    def test_cg_10x10(self):
        testdat = np.load('networks/scitile/LAS/test_data/10x10.npz')
        A, b = testdat['A'], testdat['b']
        testdat.close()
        res = cg_solve(A,b)
        sol = np.linalg.solve(A,b)
        filt = np.abs(sol)>0.01*np.mean(sol)
        npt.assert_allclose(res[filt], sol[filt], rtol=DEFAULT_TOL, atol=DEFAULT_ATOL)

    def test_cg_20x20(self):
        testdat = np.load('networks/scitile/LAS/test_data/20x20.npz')
        A, b = testdat['A'], testdat['b']
        testdat.close()
        res = cg_solve(A,b)
        sol = np.linalg.solve(A,b)
        filt = np.abs(sol)>0.01*np.mean(sol)
        npt.assert_allclose(res[filt], sol[filt], rtol=DEFAULT_TOL, atol=DEFAULT_ATOL)

    def test_cg_40x40(self):
        testdat = np.load('networks/scitile/LAS/test_data/40x40.npz')
        A, b = testdat['A'], testdat['b']
        testdat.close()
        res = cg_solve(A,b)
        sol = np.linalg.solve(A,b)
        filt = np.abs(sol)>0.01*np.mean(sol)
        npt.assert_allclose(res[filt], sol[filt], rtol=DEFAULT_TOL, atol=DEFAULT_ATOL)

    def test_cg_80x80(self):
        testdat = np.load('networks/scitile/LAS/test_data/80x80.npz')
        A, b = testdat['A'], testdat['b']
        testdat.close()
        res = cg_solve(A,b)
        sol = np.linalg.solve(A,b)
        filt = np.abs(sol)>0.01*np.mean(sol)
        npt.assert_allclose(res[filt], sol[filt], rtol=DEFAULT_TOL, atol=DEFAULT_ATOL)

    def test_spcg_10x10(self):
        testdat = np.load('networks/scitile/LAS/test_data/10x10.npz')
        A, b = testdat['A'], testdat['b']
        spA, C = make_sparse_inputs(A, b)
        testdat.close()
        res = cg_solve_sp(spA,C,b)
        sol = np.linalg.solve(A,b)
        filt = np.abs(sol)>0.01*np.mean(sol)
        npt.assert_allclose(res[filt], sol[filt], rtol=DEFAULT_TOL, atol=DEFAULT_ATOL)

    def test_spcg_20x20(self):
        testdat = np.load('networks/scitile/LAS/test_data/20x20.npz')
        A, b = testdat['A'], testdat['b']
        spA, C = make_sparse_inputs(A, b)
        testdat.close()
        res = cg_solve_sp(spA,C,b)
        sol = np.linalg.solve(A,b)
        filt = np.abs(sol)>0.01*np.mean(sol)
        npt.assert_allclose(res[filt], sol[filt], rtol=DEFAULT_TOL, atol=DEFAULT_ATOL)

    def test_spcg_40x40(self):
        testdat = np.load('networks/scitile/LAS/test_data/40x40.npz')
        A, b = testdat['A'], testdat['b']
        spA, C = make_sparse_inputs(A, b)
        testdat.close()
        res = cg_solve_sp(spA,C,b)
        sol = np.linalg.solve(A,b)
        filt = np.abs(sol)>0.01*np.mean(sol)
        npt.assert_allclose(res[filt], sol[filt], rtol=DEFAULT_TOL, atol=DEFAULT_ATOL)

    def test_spcg_80x80(self):
        testdat = np.load('networks/scitile/LAS/test_data/80x80.npz')
        A, b = testdat['A'], testdat['b']
        spA, C = make_sparse_inputs(A, b)
        testdat.close()
        res = cg_solve_sp(spA,C,b)
        sol = np.linalg.solve(A,b)
        filt = np.abs(sol)>0.01*np.mean(sol)
        npt.assert_allclose(res[filt], sol[filt], rtol=DEFAULT_TOL, atol=DEFAULT_ATOL)


if __name__ == '__main__':
    unittest.main()
