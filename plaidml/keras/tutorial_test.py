import argparse
from collections import OrderedDict
import functools
import numpy as np
import numpy.testing as npt
import operator
import sys
import unittest

import plaidml
import plaidml.keras
plaidml.keras.install_backend()
import keras.backend as K
import plaidml.context
import plaidml.exceptions
import plaidml
import testing.plaidml_config


def m(*args, **kwargs):
    """Makes a test matrix whose dimensions are the supplied arguments."""
    total = functools.reduce(operator.mul, args, 1)
    arr = np.array(range(-2, total - 2), dtype=float)
    arr = np.reshape(arr, args)
    return arr


class TestTileTutorial(unittest.TestCase):

    def setUp(self):
        plaidml.settings.config = None
        plaidml.settings.setup = True
        plaidml.settings.experimental = True

    def testSumOverAxis(self):
        I = K.variable(np.array([[1., 2., 3.], [4., 5., 6.]]))
        code = """function (I[M, N]) -> (O) {
                      O[n: N] = +(I[m, n]);
                  }"""
        op = K._Op('sum_over_axis', I.dtype, I.shape[1], code, OrderedDict([('I', I)]), ['O'])
        reference = K.sum(I, axis=0)
        npt.assert_allclose(op.eval(), reference.eval())

    def testMatMul(self):
        A = K.variable(np.array([[1., 2., 3.], [4., 5., 6.]]))
        B = K.variable(np.array([[1., -2.], [-3., 4.], [5., -6.]]))
        code = """function (A[M, L], B[L, N]) -> (C) {
                      C[i, j: M, N] = +(A[i, k] * B[k, j]);
                  }"""
        op = K._Op('matmul', A.dtype, (2, 2), code, OrderedDict([('A', A), ('B', B)]), ['C'])
        reference = K.dot(A, B)
        npt.assert_allclose(op.eval(), reference.eval())

    def testMaxOverAxis(self):
        I = K.variable(np.array([[1., 2., 3.], [4., 5., 6.]]))
        code = """function (I[M, N]) -> (O) {
                      O[n: N] = >(I[m, n]);
                  }"""
        op = K._Op('max_over_axis', I.dtype, I.shape[1], code, OrderedDict([('I', I)]), ['O'])
        reference = K.max(I, axis=0)
        npt.assert_allclose(op.eval(), reference.eval())

    def testGlobalMin(self):
        I = K.variable(np.array([[[1., 2., 3.], [4., 5., 6.]]]))
        code = """function (I) -> (O) {
                      Neg = -I;
                      O_Neg[] = >(Neg[i, j, k]);
                      O = -O_Neg;
                  }"""
        op = K._Op('global_min', I.dtype, tuple(), code, OrderedDict([('I', I)]), ['O'])
        reference = K.min(I, axis=[0, 1, 2])
        npt.assert_allclose(op.eval(), reference.eval())

    def testMeanOverAxis(self):
        I = K.variable(np.array([[1., 2., 3.], [4., 5., 6.]]))
        code = """function (I[X, Y]) -> (O) {
                      Sum[y: Y] = +(I[x, y]);
                      O = Sum / X;
                  }"""
        op = K._Op('mean_over_axis', I.dtype, I.shape[1], code, OrderedDict([('I', I)]), ['O'])
        reference = K.mean(I, axis=0)
        npt.assert_allclose(op.eval(), reference.eval())

    def testGlobalMeanAwkward(self):
        I = K.variable(np.array([[1., 2., 3.], [4., 5., 6.]]))
        code = """function (I[X, Y]) -> (O) {
                      Sum[] = +(I[x, y]);
                      PartialMean = Sum / X;
                      O = PartialMean / Y;
                  }"""
        op = K._Op('global_mean_awkward', I.dtype, tuple(), code, OrderedDict([('I', I)]), ['O'])
        reference = K.mean(I, axis=[0, 1])
        npt.assert_allclose(op.eval(), reference.eval())

    def testGlobalMean(self):
        I = K.variable(np.array([[1., 2., 3.], [4., 5., 6.]]))
        code = """function (I[X, Y]) -> (O) {
                      Sum[] = +(I[x, y]);
                      O = Sum / (X * Y);
                  }"""
        op = K._Op('global_mean', I.dtype, tuple(), code, OrderedDict([('I', I)]), ['O'])
        reference = K.mean(I, axis=[0, 1])
        npt.assert_allclose(op.eval(), reference.eval())

    def testBrokenMaxPool(self):
        I = K.variable(np.array([1., 2., 3., 4., 5.]))
        code = """function (I[N]) -> (O) {
                      O[i: N / 2] = >(I[2 * i + j]);
                  }"""
        op = K._Op('broken_maxpool', I.dtype, (I.shape[0] / 2,), code,
                   OrderedDict([('I', I)]), ['O'])
        reference = K.max(I)
        npt.assert_allclose(op.eval(), [5.] * 2)

    def testValidMaxPool(self):
        I = K.variable(np.array([1., 2., 3., 4., 5.]))
        code = """function (I[N]) -> (O) {
                      O[i: N / 2] = >(I[2 * i + j]), j < 2;
                  }"""
        op = K._Op('valid_maxpool', I.dtype, (I.shape[0] / 2,), code,
                   OrderedDict([('I', I)]), ['O'])
        npt.assert_allclose(op.eval(), [2., 4.])

    def testSameMaxPool(self):
        I = K.variable(np.array([1., 2., 3., 4., 5.]))
        code = """function (I[N]) -> (O) {
                      O[i: (N + 1) / 2] = >(I[2 * i + j]), j < 2;
                  }"""
        op = K._Op('valid_maxpool', I.dtype, ((I.shape[0] + 1) / 2,), code,
                   OrderedDict([('I', I)]), ['O'])
        npt.assert_allclose(op.eval(), np.array([2., 4., 5.]))

    def testSkipping(self):
        I = K.variable(np.array([[1., 2.], [3., 4.], [5., -4.], [-5., 6.], [-7., 9.]]))
        code = """function (I[N, M]) -> (O) {
                      O[2 * i: N] = +(I[2 * i, j]);
                  }"""
        op = K._Op('skip', I.dtype, (5,), code, OrderedDict([('I', I)]), ['O'])
        npt.assert_allclose(op.eval(), np.array([3., 0., 1., 0., 2.]))

    def testCumSum(self):
        I = K.variable(np.array([1., 2., 3., 4., 5., 6.]))
        code = """function (I[N]) -> (O) {
                      O[i: N] = +(I[i - j]), j < N;
                  }"""
        op = K._Op('cumulative_sum', I.dtype, (6,), code, OrderedDict([('I', I)]), ['O'])
        code2 = """function (I[N]) -> (O) {
                       O[i: N] = +(I[k]), i - k < N;
                   }"""
        op2 = K._Op('cumulative_sum2', I.dtype, (6,), code, OrderedDict([('I', I)]), ['O'])
        reference = K.cumsum(I)
        npt.assert_allclose(op.eval(), reference.eval())
        npt.assert_allclose(op2.eval(), reference.eval())

    def testConv1D(self):
        I = K.variable(m(2, 8, 3))
        kernel = K.variable(m(3, 3, 2))
        code = """function (I[N, L, CI], K[LK, CI, CO]) -> (O) {
                      O[n, x, co: N, L - LK + 1, CO] = +(I[n, x + k, ci] * K[k, ci, co]);
                  }"""
        op = K._Op('cumulative_sum', I.dtype, (2, 6, 3), code,
                   OrderedDict([('I', I), ('K', kernel)]), ['O'])
        reference = K.conv1d(I, kernel, padding='valid')
        npt.assert_allclose(op.eval(), reference.eval())

    def testDilatedConv2D(self):
        I = K.variable(m(2, 6, 10, 3))
        kernel = K.variable(m(3, 2, 3, 2))
        code = """function (I[N, Lx, Ly, CI], K[LKx, LKy, CI, CO]) -> (O) {
                      O[n, x, y, co: N, Lx - 2 * (LKx - 1), Ly - 3 * (LKy - 1), CO] =
                              +(I[n, x + 2 * kx, y + 3 * ky, ci] * K[kx, ky, ci, co]);
                  }"""
        op = K._Op('cumulative_sum', I.dtype, (2, 2, 7, 2), code,
                   OrderedDict([('I', I), ('K', kernel)]), ['O'])
        reference = K.conv2d(I, kernel, padding='valid', dilation_rate=(2, 3))
        npt.assert_allclose(op.eval(), reference.eval())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action='count', default=0)
    args, remainder = parser.parse_known_args()

    plaidml._internal_set_vlog(args.verbose)
    np.set_printoptions(threshold=np.nan)
    unittest.main(argv=sys.argv[:1] + remainder, verbosity=args.verbose + 1)
