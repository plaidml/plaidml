# Copyright 2020 Intel Corporation
# RUN: py_test -v | FileCheck %s

import functools
import platform
import sys
import unittest

import plaidml
import plaidml.exec
from plaidml.edsl import *
import numpy as np

DEFAULT_DEVICE = 'llvm_cpu.0'
DEFAULT_TARGET = 'llvm_cpu'

# HACKs for getting lit/FileCheck to integrate with python

# Redirect stderr to stdout
sys.stderr = sys.stdout


# Ensure that the order that tests run are the order that they are specified in the source file.
def suiteFactory(*testcases):

    ln = lambda f: getattr(tc, f).__code__.co_firstlineno
    lncmp = lambda a, b: ln(a) - ln(b)

    test_suite = unittest.TestSuite()
    for tc in testcases:
        test_suite.addTest(unittest.makeSuite(tc, sortUsing=lncmp))

    return test_suite


def caseFactory():

    from inspect import findsource

    g = globals().copy()

    cases = [
        g[obj] for obj in g if obj.startswith("Test") and issubclass(g[obj], unittest.TestCase)
    ]

    ordered_cases = sorted(cases, key=lambda f: findsource(f)[1])

    return ordered_cases


def dot(X, Y):
    I, J, K = TensorDims(3)
    i, j, k = TensorIndexes(3)
    X.bind_dims(I, K)
    Y.bind_dims(K, J)
    R = TensorOutput(I, J)
    R[i, j] += X[i, k] * Y[k, j]
    return R


def relu(I):
    return select(I < 0.0, 0.0, I)


def softmax(X):
    I, J = TensorDims(2)
    i, j = TensorIndexes(2)
    X.bind_dims(I, J)
    M = TensorOutput(I, 1)
    M[i, 0] >= X[i, j]
    E = exp(X - M)
    N = TensorOutput(I, 1)
    N[i, 0] += E[i, j]
    return E / N


def conv_1d(I, K):
    N, X, KX, CI, CO = TensorDims(5)
    n, x, k, ci, co = TensorIndexes(5)
    I.bind_dims(N, X, CI)
    K.bind_dims(KX, CI, CO)
    O = TensorOutput(N, X - KX + 1, CO)
    O[n, x, co] += I[n, x + k, ci] * K[k, ci, co]
    return O


def conv_2d_dilated(I, K):
    N, X, Y, KX, KY, CI, CO = TensorDims(7)
    n, x, y, kx, ky, ci, co = TensorIndexes(7)
    I.bind_dims(N, X, Y, CI)
    K.bind_dims(KX, KY, CI, CO)
    O = TensorOutput(N, X - 2 * (KX - 1), Y - 3 * (KY - 1), CO)
    O[n, x, y, co] += I[n, x + 2 * kx, y + 3 * ky, ci] * K[kx, ky, ci, co]
    return O


def conv_2d(I, K):
    CI, CO, K0, K1, N, X0, X1 = TensorDims(7)
    n, x0, x1, k0, k1, ci, co = TensorIndexes(7)
    I.bind_dims(N, X0, X1, CI)
    K.bind_dims(K0, K1, CI, CO)
    R = TensorOutput(N, X0 - (K0 - 1), X1 - (K1 - 1), CO)
    R[n, x0, x1, co] += I[n, x0 + k0 - (K0 // 2), x1 + k1 - (K1 // 2), ci] * K[k0, k1, ci, co]
    return R


def complex_conv_2d(
        I,
        K,
        s0,
        s1,  # stride coeffs
        d0,
        d1  # dilation coeffs
):
    # "same-lower" autopadding will be applied
    N, G, GCI, GCO = TensorDims(4)
    X0, X1 = TensorDims(2)
    K0, K1 = TensorDims(2)
    n, g, gci, gco = TensorIndexes(4)
    x0, x1 = TensorIndexes(2)
    k0, k1 = TensorIndexes(2)
    I.bind_dims(N, X0, X1, G, GCI)
    K.bind_dims(K0, K1, G, GCI, GCO)

    # Compute output spatial dimensions
    Y0, Y1 = TensorDims(2)
    Y0 = (X0 + s0 - 1) // s0
    Y1 = (X1 + s1 - 1) // s1

    #Compute the effective kernel size after dilation
    EK0, EK1 = TensorDims(2)
    EK0 = d0 * (K0 - 1) + 1
    EK1 = d1 * (K1 - 1) + 1

    #Compute the padding offset
    P0, P1 = TensorDims(2)
    P0 = ((Y0 - 1) * s0 + EK0 - X0) // 2
    P1 = ((Y1 - 1) * s1 + EK1 - X1) // 2

    # Specify the output size
    O = TensorOutput(N, Y0, Y1, G, GCO)

    # Compute the convolution
    O[n, x0, x1, g, gco] += I[n, s0 * x1 + d0 * k0 - P0, s1 * x1 + d1 * k1 -
                              P1, g, gci] * K[k0, k1, g, gci, gco]
    return O


def max_pool_1d(I):
    N = TensorDim()
    i, j = TensorIndexes(2)
    I.bind_dims(N)
    O = TensorOutput(N // 2)
    O[i] >= I[2 * i + j]
    O.add_constraint(j < 2)
    return O


def max_pool_2d(I):
    N, X0, X1, C = TensorDims(4)
    n, x0, x1, i, j, c = TensorIndexes(6)
    I.bind_dims(N, X0, X1, C)
    R = TensorOutput(N, (X0 + 1) // 2, (X1 + 1) // 2, C)
    R[n, x0, x1, c] >= I[n, 2 * x0 + i, 2 * x1 + j, c]
    R.add_constraints([i < 2, j < 2])
    return R


def flatten(X):
    X_dims = TensorDims(X.rank)
    X.bind_dims(*X_dims)
    product = functools.reduce(lambda x, y: x * y, X_dims[1:-1])
    return reshape(X, (1, product))


def normalize(X):
    idxs = TensorIndexes(X.rank)
    XSqr = X * X
    X_MS = TensorOutput()
    X_MS[()] += XSqr[idxs]
    return sqrt(X_MS)


def lars_momentum(X, Grad, Veloc, LR, lars_coeff, lars_weight_decay, momentum):
    XNorm = normalize(X)
    GradNorm = normalize(Grad)
    LocLR = LR * lars_coeff * XNorm / (GradNorm + lars_weight_decay * XNorm)
    NewVeloc = momentum * Veloc + LocLR * (Grad + lars_weight_decay * X)
    return (X - NewVeloc, NewVeloc)


def arg_max(I):
    X0, X1, X2 = TensorDims(3)
    x0, x1, x2 = TensorIndexes(3)
    I.bind_dims(X0, X1, X2)
    Max = TensorOutput(X0, X2)
    Max[x0, x2] >= I[x0, x1, x2]
    One = Placeholder(plaidml.DType.FLOAT32)
    T = TensorOutput(X1)
    T[x1] = One[()]
    IX = index(T, 0)
    O = TensorOutput(X0, X2)
    O[x0, x2] >= cond(I[x0, x1, x2], Max[x0, x2], IX[x1])
    return cast(O, DType.UINT32)


def sum_over_axis(I):
    M, N = TensorDims(2)
    m, n = TensorIndexes(2)
    I.bind_dims(M, N)
    O = TensorOutput(N)
    # contraction
    O[n] += I[m, n]
    return O


def max_over_axis(I):
    M, N = TensorDims(2)
    m, n = TensorIndexes(2)
    I.bind_dims(M, N)
    O = TensorOutput(N)
    O[n] >= I[m, n]
    return O


def matmul(A, B):
    I, J, K = TensorDims(3)
    i, j, k = TensorIndexes(3)
    A.bind_dims(I, K)
    B.bind_dims(K, J)
    C = TensorOutput(I, J)
    C[i, j] += A[i, k] * B[k, j]
    return C


def global_min(I):
    i, j, k = TensorIndexes(3)
    Neg = -I
    O_Neg = TensorOutput()
    O_Neg[()] >= Neg[i, j, k]
    O = -O_Neg
    return O


def avg(I):
    X, Y = TensorDims(2)
    x, y = TensorIndexes(2)
    I.bind_dims(X, Y)
    Sum = TensorOutput()
    Sum[y] += I[x, y]
    return Sum / X


def avg_stages(I):
    X, Y = TensorDims(2)
    x, y = TensorIndexes(2)
    I.bind_dims(X, Y)
    Sum = TensorOutput()
    Sum[()] += I[x, y]
    PartialMean = Sum / X
    return PartialMean / Y


def avg_merge(I):
    X, Y = TensorDims(2)
    x, y = TensorIndexes(2)
    I.bind_dims(X, Y)
    Sum = TensorOutput()
    Sum[()] += I[x, y]
    return Sum / (X * Y)


def skip(I):
    M, N = TensorDims(2)
    i, j = TensorIndexes(2)
    I.bind_dims(M, N)
    O = TensorOutput(N)
    O[2 * i] += I[2 * i, j]
    return O


def csum(I):
    N = TensorDim()
    i, k = TensorIndexes(2)
    I.bind_dims(N)
    O = TensorOutput(N)
    O[i] += I[k]
    O.add_constraint(i - k < N)
    return O

def constant_add():
    a = np.array([4,3,2,1])
    b = np.array([1,2,3,4])
    Buffer_A = plaidml.Buffer(plaidml.TensorShape(plaidml.DType.INT32, [4]), device = DEFAULT_DEVICE)
    Buffer_B = plaidml.Buffer(plaidml.TensorShape(plaidml.DType.INT32, [4]), device = DEFAULT_DEVICE)
    Buffer_A.copy_from_ndarray(a)
    Buffer_B.copy_from_ndarray(b)
    A = Constant(LogicalShape(plaidml.DType.INT32, [4]), Buffer_A, name="A")
    B = Constant(LogicalShape(plaidml.DType.INT32, [4]), Buffer_B, name="B")
    return A + B

class TestEdsl(unittest.TestCase):
    maxDiff = None

    def checkProgram(self, program, inputs, expected):
        if platform.system() == 'Windows':
            # the Orc JIT in LLVM is currently broken on windows.
            return
        outputs = plaidml.exec.run(program, inputs)
        for i in range(len(expected)):
            self.assertEqual(outputs[i].tolist(), expected[i])

    def test_broadcast_cmp(self):
        A = Tensor(LogicalShape(plaidml.DType.UINT64, [3, 4]))
        B = Tensor(LogicalShape(plaidml.DType.UINT64, [3, 1]))
        O = cast(A >= B, DType.UINT64)

        program = Program('broadcast_cmp', [O])
        self.checkProgram(program, [
            (A, np.array([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]])),
            (B, np.array([[0], [6], [12]])),
        ], [
            [[1, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 0]],
        ])

    def test_higher_precision_constants_invalid_negative(self):
        I = Tensor(LogicalShape(plaidml.DType.FLOAT32, [3, 3]))
        O = I * (-2)

        try:
            program = Program('higher_precision_constants', [O],
                              floatx=plaidml.DType.FLOAT64,
                              intx=plaidml.DType.UINT64)
        except Exception:
            return
        self.fail("expected exception")

    def test_higher_precision_constants(self):
        I = Tensor(LogicalShape(plaidml.DType.FLOAT32, [3, 3]))
        O = I + 1 + 2.0

        program = Program('higher_precision_constants', [O],
                          floatx=plaidml.DType.FLOAT64,
                          intx=plaidml.DType.UINT64)
        # CHECK-LABEL: test_higher_precision_constants
        expected = '''
module {
  func @higher_precision_constants(%arg0: tensor<3x3xf32>) -> tensor<3x3xf64> {
    %c1 = "eltwise.sconst"() {value = 1 : i64} : () -> ui64
    %cst = "eltwise.sconst"() {value = 2.000000e+00 : f64} : () -> f64
    %0 = "eltwise.add"(%arg0, %c1) : (tensor<3x3xf32>, ui64) -> tensor<3x3xf32>
    %1 = "eltwise.add"(%0, %cst) : (tensor<3x3xf32>, f64) -> tensor<3x3xf64>
    return %1 : tensor<3x3xf64>
  }
}
'''
        self.checkProgram(program, [
            (I, np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])),
        ], [
            [[4, 5, 6], [7, 8, 9], [10, 11, 12]],
        ])

    def test_sum_over_axis(self):
        I = Placeholder(plaidml.DType.FLOAT32, [1, 784])
        O = sum_over_axis(I)
        program = Program('sum_over_axis', [O])
        # CHECK-LABEL: test_sum_over_axis
        expected = '''
#map0 = affine_map<(d0, d1) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d1, d0)>


module {
  func @sum_over_axis(%arg0: tensor<1x784xf32>) -> tensor<784xf32> {
    %cst = "eltwise.sconst"() {value = 0.000000e+00 : f64} : () -> f32
    %0 = tile.contract add, none, %cst, %arg0 {sink = #map0, srcs = [#map1]} : f32, tensor<1x784xf32> -> tensor<784xf32>
    return %0 : tensor<784xf32>
  }
}
'''

    def test_max_over_axis(self):
        I = Placeholder(plaidml.DType.FLOAT32, [1, 784])
        O = max_over_axis(I)
        program = Program('max_over_axis', [O])
        # CHECK-LABEL: test_max_over_axis
        expected = '''
#map0 = affine_map<(d0, d1) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d1, d0)>


module {
  func @max_over_axis(%arg0: tensor<1x784xf32>) -> tensor<784xf32> {
    %cst = "eltwise.sconst"() {value = 0.000000e+00 : f64} : () -> f32
    %0 = tile.contract max, none, %cst, %arg0 {sink = #map0, srcs = [#map1]} : f32, tensor<1x784xf32> -> tensor<784xf32>
    return %0 : tensor<784xf32>
  }
}
'''

    def test_matmul(self):
        A = Placeholder(plaidml.DType.FLOAT32, [1, 784])
        B = Placeholder(plaidml.DType.FLOAT32, [784, 784])
        O = matmul(A, B)
        program = Program('matmul', [O])
        # CHECK-LABEL: test_matmul
        expected = '''
#map0 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d2, d1)>


module {
  func @matmul(%arg0: tensor<784x784xf32>, %arg1: tensor<1x784xf32>) -> tensor<1x784xf32> {
    %cst = "eltwise.sconst"() {value = 0.000000e+00 : f64} : () -> f32
    %0 = tile.contract add, mul, %cst, %arg1, %arg0 {sink = #map0, srcs = [#map1, #map2]} : f32, tensor<1x784xf32>, tensor<784x784xf32> -> tensor<1x784xf32>
    return %0 : tensor<1x784xf32>
  }
}
'''

    def test_avg(self):
        I = Placeholder(plaidml.DType.FLOAT32, [1, 784])
        O = avg(I)
        # CHECK-LABEL: test_avg
        program = Program('avg', [O], target='')
        expected = '''
#map0 = affine_map<(d0, d1) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d1, d0)>


module {
  func @avg(%arg0: tensor<1x784xf32>) -> f32 {
    %cst = "eltwise.sconst"() {value = 0.000000e+00 : f64} : () -> f32
    %0 = tile.contract add, none, %cst, %arg0 {sink = #map0, srcs = [#map1]} : f32, tensor<1x784xf32> -> f32
    return %0 : f32
  }
}
'''

    def test_avg_stages(self):
        I = Placeholder(plaidml.DType.FLOAT32, [1, 784])
        O = avg_stages(I)
        program = Program('avg_stages', [O], target='')
        # CHECK-LABEL: test_avg_stages
        expected = '''
#map0 = affine_map<() -> ()>
#map1 = affine_map<(d0, d1) -> (d0, d1)>


module {
  func @avg_stages(%arg0: tensor<1x784xf32>) -> f32 {
    %cst = "eltwise.sconst"() {value = 0.000000e+00 : f64} : () -> f32
    %c784 = tile.constant 784
    %0 = tile.contract add, none, %cst, %arg0 {sink = #map0, srcs = [#map1]} : f32, tensor<1x784xf32> -> f32
    %1 = "eltwise.div"(%0, %c784) : (f32, index) -> f32
    return %1 : f32
  }
}
'''

    def test_avg_merge(self):
        I = Placeholder(plaidml.DType.FLOAT32, [1, 784])
        O = avg_merge(I)
        program = Program('avg_merge', [O], target='')
        # CHECK-LABEL: test_avg_merge
        expected = '''
#map0 = affine_map<() -> ()>
#map1 = affine_map<(d0, d1) -> (d0, d1)>


module {
  func @avg_merge(%arg0: tensor<1x784xf32>) -> f32 {
    %cst = "eltwise.sconst"() {value = 0.000000e+00 : f64} : () -> f32
    %c784 = tile.constant 784
    %0 = tile.contract add, none, %cst, %arg0 {sink = #map0, srcs = [#map1]} : f32, tensor<1x784xf32> -> f32
    %1 = "eltwise.div"(%0, %c784) : (f32, index) -> f32
    return %1 : f32
  }
}
'''

    def test_max_pool_1d(self):
        I = Placeholder(plaidml.DType.FLOAT32, [10], name='I')
        O = max_pool_1d(I)
        program = Program('max_pool_1d', [O])
        # CHECK-LABEL: test_max_pool_1d
        expected = '''
#map0 = affine_map<(d0, d1) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0 * 2 + d1)>

#set0 = affine_set<(d0, d1) : (d1 >= 0, -d1 + 1 >= 0)>

module {
  func @max_pool_1d(%arg0: tensor<10xf32> {tile.name = "I"}) -> tensor<5xf32> {
    %cst = "eltwise.sconst"() {value = 0.000000e+00 : f64} : () -> f32
    %0 = tile.contract max, none, %cst, %arg0 {cons = #set0, sink = #map0, srcs = [#map1]} : f32, tensor<10xf32> -> tensor<5xf32>
    return %0 : tensor<5xf32>
  }
}
'''

    def test_skip(self):
        I = Placeholder(plaidml.DType.FLOAT32, [1, 784])
        O = skip(I)
        program = Program('skip', [O])
        # CHECK-LABEL: test_skip
        expected = '''
#map0 = affine_map<(d0, d1) -> (d0 * 2)>
#map1 = affine_map<(d0, d1) -> (d0 * 2, d1)>


module {
  func @skip(%arg0: tensor<1x784xf32>) -> tensor<784xf32> {
    %cst = "eltwise.sconst"() {value = 0.000000e+00 : f64} : () -> f32
    %0 = tile.contract add, none, %cst, %arg0 {sink = #map0, srcs = [#map1]} : f32, tensor<1x784xf32> -> tensor<784xf32>
    return %0 : tensor<784xf32>
  }
}
'''

    def test_conv_1d(self):
        I = Placeholder(plaidml.DType.FLOAT32, [1, 224, 3])
        K = Placeholder(plaidml.DType.FLOAT32, [3, 3, 1])
        O = conv_1d(I, K)
        program = Program('conv_1d', [O])
        # CHECK-LABEL: test_conv_1d
        expected = '''
#map0 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1 + d3, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d3, d4, d2)>


module {
  func @conv_1d(%arg0: tensor<3x3x1xf32>, %arg1: tensor<1x224x3xf32>) -> tensor<1x222x1xf32> {
    %cst = "eltwise.sconst"() {value = 0.000000e+00 : f64} : () -> f32
    %0 = tile.contract add, mul, %cst, %arg1, %arg0 {sink = #map0, srcs = [#map1, #map2]} : f32, tensor<1x224x3xf32>, tensor<3x3x1xf32> -> tensor<1x222x1xf32>
    return %0 : tensor<1x222x1xf32>
  }
}
'''

    def test_conv_2d_dilated(self):
        I = Placeholder(plaidml.DType.FLOAT32, [1, 224, 224, 1])
        K = Placeholder(plaidml.DType.FLOAT32, [3, 3, 1, 32])
        O = conv_2d_dilated(I, K)
        program = Program('conv_2d_dilated', [O])
        # CHECK-LABEL: test_conv_2d_dilated
        expected = '''
#map0 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4 * 2, d2 + d5 * 3, d6)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d3)>


module {
  func @conv_2d_dilated(%arg0: tensor<3x3x1x32xf32>, %arg1: tensor<1x224x224x1xf32>) -> tensor<1x220x218x32xf32> {
    %cst = "eltwise.sconst"() {value = 0.000000e+00 : f64} : () -> f32
    %0 = tile.contract add, mul, %cst, %arg1, %arg0 {sink = #map0, srcs = [#map1, #map2]} : f32, tensor<1x224x224x1xf32>, tensor<3x3x1x32xf32> -> tensor<1x220x218x32xf32>
    return %0 : tensor<1x220x218x32xf32>
  }
}
'''

    def test_complex_conv_2d(self):
        I = Placeholder(plaidml.DType.FLOAT32, [1, 224, 224, 3, 3])
        K = Placeholder(plaidml.DType.FLOAT32, [3, 3, 3, 3, 32])
        O = complex_conv_2d(I, K, 1, 2, 1, 2)
        program = Program('complex_conv_2d', [O])
        # CHECK-LABEL: test_complex_conv_2d
        expected = '''
#map0 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d2 + d5 - 1, d2 * 2 + d6 * 2 - 1, d3, d7)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d5, d6, d3, d7, d4)>


module {
  func @complex_conv_2d(%arg0: tensor<3x3x3x3x32xf32>, %arg1: tensor<1x224x224x3x3xf32>) -> tensor<1x224x112x3x32xf32> {
    %cst = "eltwise.sconst"() {value = 0.000000e+00 : f64} : () -> f32
    %0 = tile.contract add, mul, %cst, %arg1, %arg0 {sink = #map0, srcs = [#map1, #map2]} : f32, tensor<1x224x224x3x3xf32>, tensor<3x3x3x3x32xf32> -> tensor<1x224x112x3x32xf32>
    return %0 : tensor<1x224x112x3x32xf32>
  }
}
'''

    @unittest.skip(
        'TODO: currently segfaults mismatched dimensions error needs to be printed correctly')
    def test_complex_conv_2d_dim_mismatch(self):
        I = Placeholder(plaidml.DType.FLOAT32, [1, 1, 1, 1, 1])
        K = Placeholder(plaidml.DType.FLOAT32, [1, 1, 1, 1, 1])
        O = complex_conv_2d(I, K, 1, 2, 1, 2)
        program = Program('complex_conv_2d', [O])

    def test_mnist_mlp(self):
        # model.add(Dense(512, activation='relu', input_shape=(784,)))
        I = Placeholder(plaidml.DType.FLOAT32, [1, 784])
        K1 = Placeholder(plaidml.DType.FLOAT32, [784, 512])
        B1 = Placeholder(plaidml.DType.FLOAT32, [512])
        D1 = relu(dot(I, K1) + B1)
        # model.add(Dense(512, activation='relu'))
        K2 = Placeholder(plaidml.DType.FLOAT32, [512, 512])
        B2 = Placeholder(plaidml.DType.FLOAT32, [512])
        D2 = relu(dot(D1, K2) + B2)
        # model.add(Dense(10, activation='softmax'))
        K3 = Placeholder(plaidml.DType.FLOAT32, [512, 10])
        B3 = Placeholder(plaidml.DType.FLOAT32, [10])
        D3 = softmax(dot(D2, K3) + B3)
        program = Program('mnist_mlp', [D3])
        # CHECK-LABEL: test_mnist_mlp
        expected = '''
#map0 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map3 = affine_map<(d0, d1) -> (d0, 0)>
#map4 = affine_map<(d0, d1) -> (d0, d1)>


module {
  func @mnist_mlp(%arg0: tensor<10xf32>, %arg1: tensor<512x10xf32>, %arg2: tensor<512xf32>, %arg3: tensor<512x512xf32>, %arg4: tensor<512xf32>, %arg5: tensor<784x512xf32>, %arg6: tensor<1x784xf32>) -> tensor<1x10xf32> {
    %cst = "eltwise.sconst"() {value = 0.000000e+00 : f64} : () -> f32
    %0 = tile.contract add, mul, %cst, %arg6, %arg5 {sink = #map0, srcs = [#map1, #map2]} : f32, tensor<1x784xf32>, tensor<784x512xf32> -> tensor<1x512xf32>
    %1 = "eltwise.add"(%0, %arg4) : (tensor<1x512xf32>, tensor<512xf32>) -> tensor<1x512xf32>
    %2 = "eltwise.cmp_lt"(%1, %cst) : (tensor<1x512xf32>, f32) -> tensor<1x512xi1>
    %3 = "eltwise.select"(%2, %cst, %1) : (tensor<1x512xi1>, f32, tensor<1x512xf32>) -> tensor<1x512xf32>
    %4 = tile.contract add, mul, %cst, %3, %arg3 {sink = #map0, srcs = [#map1, #map2]} : f32, tensor<1x512xf32>, tensor<512x512xf32> -> tensor<1x512xf32>
    %5 = "eltwise.add"(%4, %arg2) : (tensor<1x512xf32>, tensor<512xf32>) -> tensor<1x512xf32>
    %6 = "eltwise.cmp_lt"(%5, %cst) : (tensor<1x512xf32>, f32) -> tensor<1x512xi1>
    %7 = "eltwise.select"(%6, %cst, %5) : (tensor<1x512xi1>, f32, tensor<1x512xf32>) -> tensor<1x512xf32>
    %8 = tile.contract add, mul, %cst, %7, %arg1 {sink = #map0, srcs = [#map1, #map2]} : f32, tensor<1x512xf32>, tensor<512x10xf32> -> tensor<1x10xf32>
    %9 = "eltwise.add"(%8, %arg0) : (tensor<1x10xf32>, tensor<10xf32>) -> tensor<1x10xf32>
    %10 = tile.contract max, none, %cst, %9 {sink = #map3, srcs = [#map4]} : f32, tensor<1x10xf32> -> tensor<1x1xf32>
    %11 = "eltwise.sub"(%9, %10) : (tensor<1x10xf32>, tensor<1x1xf32>) -> tensor<1x10xf32>
    %12 = "eltwise.exp"(%11) : (tensor<1x10xf32>) -> tensor<1x10xf32>
    %13 = tile.contract add, none, %cst, %12 {sink = #map3, srcs = [#map4]} : f32, tensor<1x10xf32> -> tensor<1x1xf32>
    %14 = "eltwise.div"(%12, %13) : (tensor<1x10xf32>, tensor<1x1xf32>) -> tensor<1x10xf32>
    return %14 : tensor<1x10xf32>
  }
}
'''

    def test_mnist_cnn(self):
        # model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
        I = Placeholder(plaidml.DType.FLOAT32, [1, 224, 224, 1])
        K1 = Placeholder(plaidml.DType.FLOAT32, [3, 3, 1, 32])
        B1 = Placeholder(plaidml.DType.FLOAT32, [32])
        C1 = relu(conv_2d(I, K1) + B1)
        # model.add(Conv2D(64, (3, 3), activation='relu'))
        K2 = Placeholder(plaidml.DType.FLOAT32, [3, 3, 32, 64])
        B2 = Placeholder(plaidml.DType.FLOAT32, [64])
        C2 = relu(conv_2d(C1, K2) + B2)
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        P1 = max_pool_2d(C2)
        # model.add(Flatten())
        F = flatten(P1)
        self.assertEqual(str(F.compute_shape()), 'tensor<1x12100xf32>')
        K3 = Placeholder(plaidml.DType.FLOAT32, [12100, 128])
        B3 = Placeholder(plaidml.DType.FLOAT32, [128])
        D1 = relu(dot(F, K3) + B3)
        # model.add(Dense(num_classes, activation='softmax'))
        K4 = Placeholder(plaidml.DType.FLOAT32, [128, 100])
        B4 = Placeholder(plaidml.DType.FLOAT32, [100])
        D2 = softmax(dot(D1, K4) + B4)
        program = Program('mnist_cnn', [D2], target='')
        # CHECK-LABEL: test_mnist_cnn
        expected = '''
#map0 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4 - 1, d2 + d5 - 1, d6)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d3)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>
#map4 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1 * 2 + d4, d2 * 2 + d5, d3)>
#map5 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map6 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map7 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map8 = affine_map<(d0, d1) -> (d0, 0)>
#map9 = affine_map<(d0, d1) -> (d0, d1)>

#set0 = affine_set<(d0, d1, d2, d3, d4, d5) : (d4 >= 0, -d4 + 1 >= 0, d5 >= 0, -d5 + 1 >= 0)>

module {
  func @mnist_cnn(%arg0: tensor<100xf32>, %arg1: tensor<128x100xf32>, %arg2: tensor<128xf32>, %arg3: tensor<12100x128xf32>, %arg4: tensor<64xf32>, %arg5: tensor<3x3x32x64xf32>, %arg6: tensor<32xf32>, %arg7: tensor<3x3x1x32xf32>, %arg8: tensor<1x224x224x1xf32>) -> tensor<1x100xf32> {
    %c12100 = tile.constant 12100
    %cst = "eltwise.sconst"() {value = 0.000000e+00 : f64} : () -> f32
    %c1 = "eltwise.sconst"() {value = 1 : i64} : () -> si32
    %0 = tile.contract add, mul, %cst, %arg8, %arg7 {sink = #map0, srcs = [#map1, #map2]} : f32, tensor<1x224x224x1xf32>, tensor<3x3x1x32xf32> -> tensor<1x222x222x32xf32>
    %1 = "eltwise.add"(%0, %arg6) : (tensor<1x222x222x32xf32>, tensor<32xf32>) -> tensor<1x222x222x32xf32>
    %2 = "eltwise.cmp_lt"(%1, %cst) : (tensor<1x222x222x32xf32>, f32) -> tensor<1x222x222x32xi1>
    %3 = "eltwise.select"(%2, %cst, %1) : (tensor<1x222x222x32xi1>, f32, tensor<1x222x222x32xf32>) -> tensor<1x222x222x32xf32>
    %4 = tile.contract add, mul, %cst, %3, %arg5 {sink = #map0, srcs = [#map1, #map2]} : f32, tensor<1x222x222x32xf32>, tensor<3x3x32x64xf32> -> tensor<1x220x220x64xf32>
    %5 = "eltwise.add"(%4, %arg4) : (tensor<1x220x220x64xf32>, tensor<64xf32>) -> tensor<1x220x220x64xf32>
    %6 = "eltwise.cmp_lt"(%5, %cst) : (tensor<1x220x220x64xf32>, f32) -> tensor<1x220x220x64xi1>
    %7 = "eltwise.select"(%6, %cst, %5) : (tensor<1x220x220x64xi1>, f32, tensor<1x220x220x64xf32>) -> tensor<1x220x220x64xf32>
    %8 = tile.contract max, none, %cst, %7 {cons = #set0, sink = #map3, srcs = [#map4]} : f32, tensor<1x220x220x64xf32> -> tensor<1x110x110x64xf32>
    %9 = "tile.reshape"(%8, %c1, %c12100) : (tensor<1x110x110x64xf32>, si32, index) -> tensor<1x12100xf32>
    %10 = tile.contract add, mul, %cst, %9, %arg3 {sink = #map5, srcs = [#map6, #map7]} : f32, tensor<1x12100xf32>, tensor<12100x128xf32> -> tensor<1x128xf32>
    %11 = "eltwise.add"(%10, %arg2) : (tensor<1x128xf32>, tensor<128xf32>) -> tensor<1x128xf32>
    %12 = "eltwise.cmp_lt"(%11, %cst) : (tensor<1x128xf32>, f32) -> tensor<1x128xi1>
    %13 = "eltwise.select"(%12, %cst, %11) : (tensor<1x128xi1>, f32, tensor<1x128xf32>) -> tensor<1x128xf32>
    %14 = tile.contract add, mul, %cst, %13, %arg1 {sink = #map5, srcs = [#map6, #map7]} : f32, tensor<1x128xf32>, tensor<128x100xf32> -> tensor<1x100xf32>
    %15 = "eltwise.add"(%14, %arg0) : (tensor<1x100xf32>, tensor<100xf32>) -> tensor<1x100xf32>
    %16 = tile.contract max, none, %cst, %15 {sink = #map8, srcs = [#map9]} : f32, tensor<1x100xf32> -> tensor<1x1xf32>
    %17 = "eltwise.sub"(%15, %16) : (tensor<1x100xf32>, tensor<1x1xf32>) -> tensor<1x100xf32>
    %18 = "eltwise.exp"(%17) : (tensor<1x100xf32>) -> tensor<1x100xf32>
    %19 = tile.contract add, none, %cst, %18 {sink = #map8, srcs = [#map9]} : f32, tensor<1x100xf32> -> tensor<1x1xf32>
    %20 = "eltwise.div"(%18, %19) : (tensor<1x100xf32>, tensor<1x1xf32>) -> tensor<1x100xf32>
    return %20 : tensor<1x100xf32>
  }
}
'''

    def test_arg_max(self):
        I = Placeholder(plaidml.DType.FLOAT32, [1, 10, 10])
        O = arg_max(I)
        program = Program('arg_max', [O], target='')
        self.assertEqual(str(O.compute_shape()), 'tensor<1x10xui32>')
        # CHECK-LABEL: test_arg_max
        expected = '''
#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<() -> ()>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d2, d1)>
#map4 = affine_map<(d0, d1, d2) -> (d2)>


module {
  func @arg_max(%arg0: f32, %arg1: tensor<1x10x10xf32>) -> tensor<1x10xui32> {
    %cst = "eltwise.sconst"() {value = 0.000000e+00 : f64} : () -> f32
    %0 = tile.contract assign, none, %cst, %arg0 {sink = #map0, srcs = [#map1]} : f32, f32 -> tensor<10xf32>
    %1 = "tile.index"(%0) {dim = 0 : i64} : (tensor<10xf32>) -> tensor<10xsi32>
    %2 = tile.contract max, none, %cst, %arg1 {sink = #map2, srcs = [#map3]} : f32, tensor<1x10x10xf32> -> tensor<1x10xf32>
    %3 = tile.contract max, cond, %cst, %arg1, %2, %1 {sink = #map2, srcs = [#map3, #map2, #map4]} : f32, tensor<1x10x10xf32>, tensor<1x10xf32>, tensor<10xsi32> -> tensor<1x10xsi32>
    %4 = "eltwise.cast"(%3) : (tensor<1x10xsi32>) -> tensor<1x10xui32>
    return %4 : tensor<1x10xui32>
  }
}
'''

    def test_global_min(self):
        I = Placeholder(plaidml.DType.FLOAT32, [10, 10, 10], name='I')
        O = global_min(I)
        program = Program('global_min', [O])
        # CHECK-LABEL: test_global_min
        expected = '''
#map0 = affine_map<() -> ()>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>


module {
  func @global_min(%arg0: tensor<10x10x10xf32> {tile.name = "I"}) -> f32 {
    %cst = "eltwise.sconst"() {value = 0.000000e+00 : f64} : () -> f32
    %0 = "eltwise.neg"(%arg0) : (tensor<10x10x10xf32>) -> tensor<10x10x10xf32>
    %1 = tile.contract max, none, %cst, %0 {sink = #map0, srcs = [#map1]} : f32, tensor<10x10x10xf32> -> f32
    %2 = "eltwise.neg"(%1) : (f32) -> f32
    return %2 : f32
  }
}
'''

    def test_cum_sum(self):
        I = Placeholder(plaidml.DType.FLOAT32, [10], name='I')
        O = csum(I)
        program = Program('cum_sum', [O])
        # CHECK-LABEL: test_cum_sum
        expected = '''
#map0 = affine_map<(d0, d1) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d1)>

#set0 = affine_set<(d0, d1) : (d0 - d1 >= 0, -d0 + d1 + 9 >= 0)>

module {
  func @cum_sum(%arg0: tensor<10xf32> {tile.name = "I"}) -> tensor<10xf32> {
    %cst = "eltwise.sconst"() {value = 0.000000e+00 : f64} : () -> f32
    %0 = tile.contract add, none, %cst, %arg0 {cons = #set0, sink = #map0, srcs = [#map1]} : f32, tensor<10xf32> -> tensor<10xf32>
    return %0 : tensor<10xf32>
  }
}
'''

    def test_invalid_shape_error(self):
        O = TensorOutput(TensorDims(3))
        with self.assertRaises(plaidml.Error) as err:
            shape = O.compute_shape()
        self.assertTrue('Cannot compute shape' in str(err.exception))

    def test_unique_names(self):
        A = Placeholder(plaidml.DType.FLOAT32, name='A')
        B = Placeholder(plaidml.DType.FLOAT32, name='B')
        C0 = Placeholder(plaidml.DType.FLOAT32, name='C')
        C1 = Placeholder(plaidml.DType.FLOAT32, name='C')
        program = Program('unique_names', [A + B + C0 + C1])
        # CHECK-LABEL: test_unique_names
        expected = '''

module {
  func @unique_names(%arg0: f32 {tile.name = "C"}, %arg1: f32 {tile.name = "C_0"}, %arg2: f32 {tile.name = "B"}, %arg3: f32 {tile.name = "A"}) -> f32 {
    %0 = "eltwise.add"(%arg3, %arg2) : (f32, f32) -> f32
    %1 = "eltwise.add"(%0, %arg1) : (f32, f32) -> f32
    %2 = "eltwise.add"(%1, %arg0) : (f32, f32) -> f32
    return %2 : f32
  }
}
'''

    def test_lars_momentum4d(self):
        X_shape = LogicalShape(plaidml.DType.FLOAT32, [4, 7, 3, 9])
        LR_Shape = LogicalShape(plaidml.DType.FLOAT32)
        X = Tensor(X_shape)
        Grad = Tensor(X_shape)
        Veloc = Tensor(X_shape)
        LR = Tensor(LR_Shape)
        R = lars_momentum(X, Grad, Veloc, LR, 1. / 1024., 1. / 2048., 1. / 8.)
        program = Program('lars_momentum4d', R, target='')
        # CHECK-LABEL: test_lars_momentum4d
        expected = '''
#map0 = affine_map<() -> ()>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>


module {
  func @lars_momentum4d(%arg0: tensor<4x7x3x9xf32>, %arg1: tensor<4x7x3x9xf32>, %arg2: f32, %arg3: tensor<4x7x3x9xf32>) -> (tensor<4x7x3x9xf32>, tensor<4x7x3x9xf32>) {
    %cst = "eltwise.sconst"() {value = 1.250000e-01 : f64} : () -> f32
    %cst_0 = "eltwise.sconst"() {value = 9.765625E-4 : f64} : () -> f32
    %cst_1 = "eltwise.sconst"() {value = 4.8828125E-4 : f64} : () -> f32
    %cst_2 = "eltwise.sconst"() {value = 0.000000e+00 : f64} : () -> f32
    %0 = "eltwise.mul"(%arg0, %cst_1) : (tensor<4x7x3x9xf32>, f32) -> tensor<4x7x3x9xf32>
    %1 = "eltwise.add"(%arg1, %0) : (tensor<4x7x3x9xf32>, tensor<4x7x3x9xf32>) -> tensor<4x7x3x9xf32>
    %2 = "eltwise.mul"(%arg0, %arg0) : (tensor<4x7x3x9xf32>, tensor<4x7x3x9xf32>) -> tensor<4x7x3x9xf32>
    %3 = tile.contract add, none, %cst_2, %2 {sink = #map0, srcs = [#map1]} : f32, tensor<4x7x3x9xf32> -> f32
    %4 = "eltwise.sqrt"(%3) : (f32) -> f32
    %5 = "eltwise.mul"(%4, %cst_1) : (f32, f32) -> f32
    %6 = "eltwise.mul"(%arg1, %arg1) : (tensor<4x7x3x9xf32>, tensor<4x7x3x9xf32>) -> tensor<4x7x3x9xf32>
    %7 = tile.contract add, none, %cst_2, %6 {sink = #map0, srcs = [#map1]} : f32, tensor<4x7x3x9xf32> -> f32
    %8 = "eltwise.sqrt"(%7) : (f32) -> f32
    %9 = "eltwise.add"(%8, %5) : (f32, f32) -> f32
    %10 = "eltwise.mul"(%arg2, %cst_0) : (f32, f32) -> f32
    %11 = "eltwise.mul"(%10, %4) : (f32, f32) -> f32
    %12 = "eltwise.div"(%11, %9) : (f32, f32) -> f32
    %13 = "eltwise.mul"(%12, %1) : (f32, tensor<4x7x3x9xf32>) -> tensor<4x7x3x9xf32>
    %14 = "eltwise.mul"(%arg3, %cst) : (tensor<4x7x3x9xf32>, f32) -> tensor<4x7x3x9xf32>
    %15 = "eltwise.add"(%14, %13) : (tensor<4x7x3x9xf32>, tensor<4x7x3x9xf32>) -> tensor<4x7x3x9xf32>
    %16 = "eltwise.sub"(%arg0, %15) : (tensor<4x7x3x9xf32>, tensor<4x7x3x9xf32>) -> tensor<4x7x3x9xf32>
    return %16, %15 : tensor<4x7x3x9xf32>, tensor<4x7x3x9xf32>
  }
}
'''

    def test_repeat_elts(self):
        I = Placeholder(plaidml.DType.FLOAT32, [10, 10, 10])
        N0, N1, N2 = TensorDims(3)
        n0, n1, n2, k = TensorIndexes(4)
        I.bind_dims(N0, N1, N2)
        O = TensorOutput(N0, 3 * N1, N2)
        O[n0, 3 * n1 + k, n2] = I[n0, n1, n2]
        O.add_constraint(k < 3)
        O.no_reduce()
        program = Program('repeat_elts', [O])
        # CHECK-LABEL: test_repeat_elts
        expected = '''
#map0 = affine_map<(d0, d1, d2, d3) -> (d0, d1 * 3 + d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>

#set0 = affine_set<(d0, d1, d2, d3) : (d2 >= 0, -d2 + 2 >= 0)>

module {
  func @repeat_elts(%arg0: tensor<10x10x10xf32>) -> tensor<10x30x10xf32> {
    %cst = "eltwise.sconst"() {value = 0.000000e+00 : f64} : () -> f32
    %0 = tile.contract assign, none, %cst, %arg0 {cons = #set0, no_reduce, sink = #map0, srcs = [#map1]} : f32, tensor<10x10x10xf32> -> tensor<10x30x10xf32>
    return %0 : tensor<10x30x10xf32>
  }
}
'''

    def test_use_default(self):
        P = Placeholder(plaidml.DType.FLOAT32, [1, 7, 10, 10])
        I = Placeholder(plaidml.DType.FLOAT32, [1, 10, 10])
        B, N1, N2 = TensorDims(3)
        b, i1, i2 = TensorIndexes(3)
        I.bind_dims(B, N1, N2)
        O = TensorOutput(B, 7, N1, N2)
        O[b, 3, i1, i2] = I[b, i1, i2]
        O.use_default(P)
        program = Program('use_default', [O])
        # CHECK-LABEL: test_use_default
        expected = '''
#map0 = affine_map<(d0, d1, d2) -> (d0, 3, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>


module {
  func @use_default(%arg0: tensor<1x10x10xf32>, %arg1: tensor<1x7x10x10xf32>) -> tensor<1x7x10x10xf32> {
    %0 = tile.contract assign, none, %arg1, %arg0 {sink = #map0, srcs = [#map1]} : tensor<1x7x10x10xf32>, tensor<1x10x10xf32> -> tensor<1x7x10x10xf32>
    return %0 : tensor<1x7x10x10xf32>
  }
}
'''

    def test_defract(self):
        I = Placeholder(plaidml.DType.FLOAT32, [3], name='I')
        K = Placeholder(plaidml.DType.FLOAT32, [3], name='K')
        i, j = TensorIndexes(2)
        O = TensorOutput(5)
        O[i] += (I[(i - j + 1) // 2] * K[j])
        program = Program('defract_test', [O])
        # CHECK-LABEL: test_defract
        expected = '''
#map0 = affine_map<(d0, d1) -> (d0)>
#map1 = affine_map<(d0, d1) -> ((d0 - d1 + 1) floordiv 2)>
#map2 = affine_map<(d0, d1) -> (d1)>


module {
  func @defract_test(%arg0: tensor<3xf32> {tile.name = "K"}, %arg1: tensor<3xf32> {tile.name = "I"}) -> tensor<5xf32> {
    %cst = "eltwise.sconst"() {value = 0.000000e+00 : f64} : () -> f32
    %0 = tile.contract add, mul, %cst, %arg1, %arg0 {sink = #map0, srcs = [#map1, #map2]} : f32, tensor<3xf32>, tensor<3xf32> -> tensor<5xf32>
    return %0 : tensor<5xf32>
  }
}
'''
        self.checkProgram(program, [
            (I, np.array([1, 2, 3])),
            (K, np.array([1, 2, 3])),
        ], [
            [2, 5, 4, 9, 6],
        ])

    @unittest.skip('FIXME: incorrect output')
    def test_defract_short(self):
        I = Placeholder(plaidml.DType.FLOAT32, [3], name='I')
        i, j = TensorIndexes(2)
        O = TensorOutput(6)
        O[i] += (I[(i - 1) // 2])
        program = Program('defract_short_test', [O])
        # SKIP-CHECK-LABEL: test_defract_short
        expected = '''
#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ((d0 - 1) floordiv 2)>


module {
  func @defract_short_test(%arg0: tensor<3xf32> {tile.name = "I"}) -> tensor<6xf32> {
    %cst = "eltwise.sconst"() {value = 0.000000e+00 : f64} : () -> f32
    %0 = tile.contract add, none, %cst, %arg0 {sink = #map0, srcs = [#map1]} : f32, tensor<3xf32> -> tensor<6xf32>
    return %0 : tensor<6xf32>
  }
}
'''
        self.checkProgram(program, [
            (I, np.array([1, 2, 3])),
        ], [
            [0, 1, 0, 2, 0, 3],
        ])

    @unittest.skip('FIXME: incorrect output')
    def test_defract_long(self):
        shape = [1, 3, 3, 1]
        I = Placeholder(plaidml.DType.FLOAT32, shape, name='I')
        K = Placeholder(plaidml.DType.FLOAT32, shape, name='K')
        n, x0, x1, c0, c1, co, ci, k0, k1 = TensorIndexes(9)
        O = TensorOutput(1, 5, 5, 1)
        O[n, x0, x1, co] += (I[n, (x0 + k0 - 1) // 2,
                               (x1 + k1 - 1) // 2, ci] * K[2 - k0, 2 - k1, co, ci])
        program = Program('defract_long', [O])
        # SKIP-CHECK-LABEL: test_defract_long
        expected = '''
#map0 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, (d1 + d4 - 1) floordiv 2, (d2 + d5 - 1) floordiv 2, d6)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (-d4 + 2, -d5 + 2, d3, d6)>


module {
  func @defract_long(%arg0: tensor<1x3x3x1xf32> {tile.name = "K"}, %arg1: tensor<1x3x3x1xf32> {tile.name = "I"}) -> tensor<1x5x5x1xf32> {
    %cst = "eltwise.sconst"() {value = 0.000000e+00 : f64} : () -> f32
    %0 = tile.contract add, mul, %cst, %arg1, %arg0 {sink = #map0, srcs = [#map1, #map2]} : f32, tensor<1x3x3x1xf32>, tensor<1x3x3x1xf32> -> tensor<1x5x5x1xf32>
    return %0 : tensor<1x5x5x1xf32>
  }
}
'''
        self.checkProgram(program, [
            (I, np.array([[
                [[1], [3], [-1]],
                [[0], [2], [4]],
                [[1], [-1], [-2]],
            ]])),
            (K, np.array([[
                [[2], [3], [4]],
                [[6], [-3], [-1]],
                [[-1], [-2], [1]],
            ]])),
        ], [[[
            [[0], [0], [0], [0], [0]],
            [[0], [4], [12], [6], [24]],
            [[0], [0], [0], [0], [0]],
            [[6], [-3], [-6], [-3], [-12]],
            [[0], [0], [0], [0], [0]],
        ]]])

    def test_funky_names(self):
        '''Exercises fix for plaidml bug #241

        Now that we emit keras layer names as 'pid' attribute values, in order
        to help link tile code back to its origin while debugging, we must
        reformat those names as valid tile identifiers. If we are doing that,
        this test will pass, otherwise we'll get a syntax error.'''

        I = Placeholder(plaidml.DType.FLOAT32, [3], name='I')
        K = Placeholder(plaidml.DType.FLOAT32, [3], name='K')
        i, j = TensorIndexes(2)
        O = TensorOutput(5)
        O[i] += (I[(i - j + 1) // 2] * K[j])
        program = Program('this-is-not an identifier', [O])
        # CHECK-LABEL: test_funky_names
        expected = '''
#map0 = affine_map<(d0, d1) -> (d0)>
#map1 = affine_map<(d0, d1) -> ((d0 - d1 + 1) floordiv 2)>
#map2 = affine_map<(d0, d1) -> (d1)>


module {
  func @"this-is-not an identifier"(%arg0: tensor<3xf32> {tile.name = "K"}, %arg1: tensor<3xf32> {tile.name = "I"}) -> tensor<5xf32> {
    %cst = "eltwise.sconst"() {value = 0.000000e+00 : f64} : () -> f32
    %0 = tile.contract add, mul, %cst, %arg1, %arg0 {sink = #map0, srcs = [#map1, #map2]} : f32, tensor<3xf32>, tensor<3xf32> -> tensor<5xf32>
    return %0 : tensor<5xf32>
  }
}
'''
        self.checkProgram(program, [
            (I, np.array([1, 2, 3])),
            (K, np.array([1, 2, 3])),
        ], [
            [2, 5, 4, 9, 6],
        ])

    def test_identity(self):
        I = Placeholder(plaidml.DType.FLOAT32, [3], name='I')
        program = Program('identity', [I])
        # CHECK-LABEL: test_identity
        expected = '''
module {
  func @identity(%arg0: tensor<3xf32> {tile.name = "I"}) -> tensor<3xf32> {
    %0 = "eltwise.ident"(%arg0) : (tensor<3xf32>) -> tensor<3xf32>
    return %0 : tensor<3xf32>
  }
}
'''
        self.checkProgram(program, [
            (I, np.array([(1, 2, 3)])),
        ], [
            [1, 2, 3],
        ])

    @unittest.skip('TODO: exception needs to be thrown')
    def test_assignment_exceptions(self):
        A = Placeholder(plaidml.DType.FLOAT32, [5, 1], name='A')
        B = Placeholder(plaidml.DType.FLOAT32, [1, 5], name='B')
        L, M, N = TensorDims(3)
        i, j, k = TensorIndexes(3)
        A.bind_dims(L, M)
        B.bind_dims(M, N)
        O = TensorOutput(L, N)
        O[i, j] = A[i, k] * B[k, j]
        program = Program('assignment_non_exception', [O])
        self.checkProgram(program, [
            (A, np.array([[1], [2], [3], [4], [5]])),
            (B, np.array([1, 2, 3, 4, 5])),
        ], [[
            [1., 2., 3., 4., 5.],
            [2., 4., 6., 8., 10.],
            [3., 6., 9., 12., 15.],
            [4., 8., 12., 16., 20.],
            [5., 10., 15., 20., 25.],
        ]])

        O = TensorOutput(L, N)
        O[i, j] = B[i, k] * A[k, j]
        with self.assertRaises(plaidml.Error) as cm:
            program = Program('assignment_exception', [O])
        self.assertTrue("illegal assignment aggregation" in str(cm.exception))

    def test_two_outputs(self):
        I = Placeholder(plaidml.DType.FLOAT32, [3], name='I')
        program1 = Program('two_outputs', [I, I])
        # CHECK-LABEL: test_two_outputs
        expected = '''
module {
  func @two_outputs(%arg0: tensor<3xf32> {tile.name = "I"}) -> (tensor<3xf32>, tensor<3xf32>) {
    %0 = "eltwise.ident"(%arg0) : (tensor<3xf32>) -> tensor<3xf32>
    %1 = "eltwise.ident"(%arg0) : (tensor<3xf32>) -> tensor<3xf32>
    return %0, %1 : tensor<3xf32>, tensor<3xf32>
  }
}
'''
        self.checkProgram(program1, [
            (I, np.array([(1, 2, 3)])),
        ], [
            [1, 2, 3],
            [1, 2, 3],
        ])

        O1 = I
        O2 = I
        program2 = Program('two_outputs', [O1, O2])
        self.assertMultiLineEqual(str(program1), str(program2))
        self.checkProgram(program2, [
            (I, np.array([(1, 2, 3)])),
        ], [
            [1, 2, 3],
            [1, 2, 3],
        ])

    def test_placeholder_noshape(self):
        I1 = Placeholder(plaidml.DType.INT32)
        program1 = Program('placeholder_noshape', [I1])
        I2 = Placeholder(LogicalShape(plaidml.DType.INT32))
        program2 = Program('placeholder_noshape', [I2])
        self.assertEqual(str(I1.compute_shape()), "tensor<si32>")
        self.assertEqual(str(I2.compute_shape()), "tensor<si32>")
        self.assertMultiLineEqual(str(program1), str(program2))

    def test_placeholder_noname(self):
        I1 = Placeholder(plaidml.DType.INT32, [1, 1])
        program1 = Program('placeholder_noname', [I1])
        I2 = Placeholder(LogicalShape(plaidml.DType.INT32, [1, 1]))
        program2 = Program('placeholder_noname', [I2])
        self.assertEqual(str(I1.compute_shape()), "tensor<1x1xsi32>")
        self.assertEqual(str(I2.compute_shape()), "tensor<1x1xsi32>")
        self.assertMultiLineEqual(str(program1), str(program2))

    def test_placeholder_with_name(self):
        I1 = Placeholder(plaidml.DType.INT32, [1, 1], name='I')
        program1 = Program('placeholder_with_name', [I1])
        I2 = Placeholder(LogicalShape(plaidml.DType.INT32, [1, 1]), name='I')
        program2 = Program('placeholder_with_name', [I2])
        self.assertEqual(str(I1.compute_shape()), "tensor<1x1xsi32>")
        self.assertEqual(str(I2.compute_shape()), "tensor<1x1xsi32>")
        self.assertMultiLineEqual(str(program1), str(program2))

    def test_constant_add(self):
        constants = np.array([[1,2,3,4], [4,3,2,1]])
        I = constant_add()
        program = Program('const_add', [I])
        self.assertEqual(len(program.args), 3)
        self.assertEqual(len(program.inputs), 2)
        # check inputs to known values
        for i in range(2):
            this_arg = program.inputs[i]
            self.assertEqual(this_arg.buffer.as_ndarray().tolist(), constants[i].tolist())
        expected = '''

module {
  func @const_add(%arg0: tensor<4xsi32> {tile.name = "B"}, %arg1: tensor<4xsi32> {tile.name = "A"}) -> tensor<4xsi32> {
    %0 = "eltwise.add"(%arg1, %arg0) : (tensor<4xsi32>, tensor<4xsi32>) -> tensor<4xsi32>
    return %0 : tensor<4xsi32>
  }
}'''
        self.assertEqual(str(program), expected)

    def test_collect_passes(self):
        A = Placeholder(plaidml.DType.FLOAT32, [10, 10])
        B = Placeholder(plaidml.DType.FLOAT32, [10, 10])
        C = dot(A, B)
        program = Program('collect_passes', [C], debug=True)

        first_pass = program.passes[0]
        self.assertEqual('tile', first_pass[0])
        self.assertEqual(str(program), first_pass[1])


if __name__ == '__main__':
    plaidml.settings.set('PLAIDML_DEVICE', DEFAULT_DEVICE)
    plaidml.settings.set('PLAIDML_TARGET', DEFAULT_TARGET)
    cases = suiteFactory(*caseFactory())
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(cases)
