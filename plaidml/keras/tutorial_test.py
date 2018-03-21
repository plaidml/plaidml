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
import plaidml.tile as tile
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
        value = tile.Operation(code,
                               [('I', I)],
                               [('O', tile.Shape(I.shape.dtype, (I.shape.dims[1],)))],
                               name='SumOverAxis') \
                    .sole_output()
        reference = K.sum(I, axis=0)
        npt.assert_allclose(value.eval(), reference.eval())

    def testMatMul(self):
        A = K.variable(np.array([[1., 2., 3.], [4., 5., 6.]]))
        B = K.variable(np.array([[1., -2.], [-3., 4.], [5., -6.]]))
        code = """function (A[M, L], B[L, N]) -> (C) {
                      C[i, j: M, N] = +(A[i, k] * B[k, j]);
                  }"""
        value = tile.Operation(code,
                               [('A', A), ('B', B)],
                               [('C', tile.Shape(A.shape.dtype, (2, 2)))],
                               name='MatMul') \
                    .sole_output()
        reference = K.dot(A, B)
        npt.assert_allclose(value.eval(), reference.eval())

    def testMaxOverAxis(self):
        I = K.variable(np.array([[1., 2., 3.], [4., 5., 6.]]))
        code = """function (I[M, N]) -> (O) {
                      O[n: N] = >(I[m, n]);
                  }"""
        value = tile.Operation(code,
                               [('I', I)],
                               [('O', tile.Shape(I.shape.dtype, (I.shape.dims[1],)))],
                               name='MaxOverAxis') \
                    .sole_output()
        reference = K.max(I, axis=0)
        npt.assert_allclose(value.eval(), reference.eval())

    def testGlobalMin(self):
        I = K.variable(np.array([[[1., 2., 3.], [4., 5., 6.]]]))
        code = """function (I) -> (O) {
                      Neg = -I;
                      O_Neg[] = >(Neg[i, j, k]);
                      O = -O_Neg;
                  }"""
        value = tile.Operation(code,
                               [('I', I)],
                               [('O', tile.Shape(I.shape.dtype, tuple()))],
                               name='GlobalMin') \
                    .sole_output()
        reference = K.min(I, axis=[0, 1, 2])
        npt.assert_allclose(value.eval(), reference.eval())

    def testMeanOverAxis(self):
        I = K.variable(np.array([[1., 2., 3.], [4., 5., 6.]]))
        code = """function (I[X, Y]) -> (O) {
                      Sum[y: Y] = +(I[x, y]);
                      O = Sum / X;
                  }"""
        value = tile.Operation(code,
                               [('I', I)],
                               [('O', tile.Shape(I.shape.dtype, (I.shape.dims[1],)))],
                               name='MeanOverAxis') \
                    .sole_output()
        reference = K.mean(I, axis=0)
        npt.assert_allclose(value.eval(), reference.eval())

    def testGlobalMeanAwkward(self):
        I = K.variable(np.array([[1., 2., 3.], [4., 5., 6.]]))
        code = """function (I[X, Y]) -> (O) {
                      Sum[] = +(I[x, y]);
                      PartialMean = Sum / X;
                      O = PartialMean / Y;
                  }"""
        value = tile.Operation(code,
                               [('I', I)],
                               [('O', tile.Shape(I.shape.dtype, tuple()))],
                               name='GlobalMeanAwkward') \
                    .sole_output()
        reference = K.mean(I, axis=[0, 1])
        npt.assert_allclose(value.eval(), reference.eval())

    def testGlobalMean(self):
        I = K.variable(np.array([[1., 2., 3.], [4., 5., 6.]]))
        code = """function (I[X, Y]) -> (O) {
                      Sum[] = +(I[x, y]);
                      O = Sum / (X * Y);
                  }"""
        value = tile.Operation(code,
                               [('I', I)],
                               [('O', tile.Shape(I.shape.dtype, tuple()))],
                               name='GlobalMean') \
                    .sole_output()
        reference = K.mean(I, axis=[0, 1])
        npt.assert_allclose(value.eval(), reference.eval())

    def testBrokenMaxPool(self):
        I = K.variable(np.array([1., 2., 3., 4., 5.]))
        code = """function (I[N]) -> (O) {
                      O[i: N / 2] = >(I[2 * i + j]);
                  }"""
        value = tile.Operation(code,
                               [('I', I)],
                               [('O', tile.Shape(I.shape.dtype, (I.shape.dims[0] / 2,)))],
                               name='BrokenMaxpool') \
                    .sole_output()
        reference = K.max(I)
        npt.assert_allclose(value.eval(), [5.] * 2)

    def testValidMaxPool(self):
        I = K.variable(np.array([1., 2., 3., 4., 5.]))
        code = """function (I[N]) -> (O) {
                      O[i: N / 2] = >(I[2 * i + j]), j < 2;
                  }"""
        value = tile.Operation(code,
                               [('I', I)],
                               [('O', tile.Shape(I.shape.dtype, (I.shape.dims[0] / 2,)))],
                               name='ValidMaxpool') \
                    .sole_output()
        npt.assert_allclose(value.eval(), [2., 4.])

    def testSameMaxPool(self):
        I = K.variable(np.array([1., 2., 3., 4., 5.]))
        code = """function (I[N]) -> (O) {
                      O[i: (N + 1) / 2] = >(I[2 * i + j]), j < 2;
                  }"""
        value = tile.Operation(code,
                               [('I', I)],
                               [('O', tile.Shape(I.shape.dtype, ((I.shape.dims[0] + 1) / 2,)))],
                               name='ValidMaxpool') \
                    .sole_output()
        npt.assert_allclose(value.eval(), np.array([2., 4., 5.]))

    def testSkipping(self):
        I = K.variable(np.array([[1., 2.], [3., 4.], [5., -4.], [-5., 6.], [-7., 9.]]))
        code = """function (I[N, M]) -> (O) {
                      O[2 * i: N] = +(I[2 * i, j]);
                  }"""
        value = tile.Operation(code,
                               [('I', I)],
                               [('O', tile.Shape(I.shape.dtype, (5,)))],
                               name='Skip') \
                    .sole_output()
        npt.assert_allclose(value.eval(), np.array([3., 0., 1., 0., 2.]))

    def testCumSum(self):
        I = K.variable(np.array([1., 2., 3., 4., 5., 6.]))
        code = """function (I[N]) -> (O) {
                      O[i: N] = +(I[i - j]), j < N;
                  }"""
        value = tile.Operation(code,
                               [('I', I)],
                               [('O', tile.Shape(I.shape.dtype, (6,)))],
                               name='CumulativeSum') \
                    .sole_output()
        code2 = """function (I[N]) -> (O) {
                       O[i: N] = +(I[k]), i - k < N;
                   }"""
        value2 = tile.Operation(code,
                                [('I', I)],
                                [('O', tile.Shape(I.shape.dtype, (6,)))],
                                name='CumulativeSum2') \
                     .sole_output()
        reference = K.cumsum(I)
        npt.assert_allclose(value.eval(), reference.eval())
        npt.assert_allclose(value2.eval(), reference.eval())

    def testConv1D(self):
        I = K.variable(m(2, 8, 3))
        kernel = K.variable(m(3, 3, 2))
        code = """function (I[N, L, CI], K[LK, CI, CO]) -> (O) {
                      O[n, x, co: N, L - LK + 1, CO] = +(I[n, x + k, ci] * K[k, ci, co]);
                  }"""
        value = tile.Operation(code,
                               [('I', I), ('K', kernel)],
                               [('O', tile.Shape(I.shape.dtype, (2, 6, 2)))],
                               name='CumulativeSum') \
                    .sole_output()
        reference = K.conv1d(I, kernel, padding='valid')
        npt.assert_allclose(value.eval(), reference.eval())

    def testDilatedConv2D(self):
        I = K.variable(m(2, 6, 10, 3))
        kernel = K.variable(m(3, 2, 3, 2))
        code = """function (I[N, Lx, Ly, CI], K[LKx, LKy, CI, CO]) -> (O) {
                      O[n, x, y, co: N, Lx - 2 * (LKx - 1), Ly - 3 * (LKy - 1), CO] =
                              +(I[n, x + 2 * kx, y + 3 * ky, ci] * K[kx, ky, ci, co]);
                  }"""
        value = tile.Operation(code,
                               [('I', I), ('K', kernel)],
                               [('O', tile.Shape(I.shape.dtype, (2, 2, 7, 2)))],
                               name='CumulativeSum') \
                    .sole_output()
        reference = K.conv2d(I, kernel, padding='valid', dilation_rate=(2, 3))
        npt.assert_allclose(value.eval(), reference.eval())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action='count', default=0)
    args, remainder = parser.parse_known_args()

    plaidml._internal_set_vlog(args.verbose)
    np.set_printoptions(threshold=np.nan)
    unittest.main(argv=sys.argv[:1] + remainder, verbosity=args.verbose + 1)
