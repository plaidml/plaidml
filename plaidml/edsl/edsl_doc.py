# Copyright 2020 Intel Corporation.
# Note:
#    This file is being used by sphinx docs to pull in code blocks.
#    Code blocks are pulled into docs/usage/writing_edsl.rs
#    Any changes made here may upset the docs.

import unittest

import plaidml
from plaidml.edsl import *


def sum_over_axis(I):
    M, N = TensorDims(2)
    m, n = TensorIndexes(2)
    I.bind_dims(M, N)
    O = TensorOutput(N)
    O[n] += I[m, n]  # contraction
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


def for_loop_max_pool_1d():
    # for_loop_max_pool_1d_start
    for i in range(1, N // 2):
        curr_max = numpy.finfo(float).eps
        for j in range(1, 2):
            if I[2 * i * j] > curr_max:
                curr_max = I[2 * i + j]
        O[i] = curr_max
    # for_loop_max_pool_1d_end


def wrong_max_pool_1d(I):
    N = TensorDim()
    i, j = TensorIndexes(2)
    I.bind_dims(N)
    O = TensorOutput(N // 2)
    O[i] >= I[2 * i + j]
    return O


def max_pool_1d(I):
    N = TensorDim()
    i, j = TensorIndexes(2)
    I.bind_dims(N)
    O = TensorOutput(N // 2)
    O[i] >= I[2 * i + j]
    O.add_constraint(j < 2)
    return O


def valid_indices(I):
    N = TensorDim()
    i, j = TensorIndexes(2)
    # valid_indices_start
    I.bind_dims(N)
    O = TensorOutput(N + 1 // 2)
    O[i] >= I[2 * i + j]
    O.add_constraint(j < 2)
    # valid_indices_end
    return O


def max_pool_1d_odd(I):
    N = TensorDim()
    i, j = TensorIndexes(2)
    I.bind_dims(N)
    O = TensorOutput(N + 1 // 2)
    O[i] >= I[2 * i + j]
    O.add_constraint(j < 2)
    return O


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


def complex_conv_2d(I, K, s0, s1, d0, d1):
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


class TestEdslDocs(unittest.TestCase):

    def test_sum_over_axis(self):
        I = Tensor(LogicalShape(plaidml.DType.FLOAT32, [1, 78]))
        O = sum_over_axis(I)
        program = Program('sum_over_axis', [O])

    def test_max_over_axis(self):
        I = Tensor(LogicalShape(plaidml.DType.FLOAT32, [1, 78]))
        O = max_over_axis(I)
        program = Program('max_over_axis', [O])

    def test_matmul(self):
        A = Tensor(LogicalShape(plaidml.DType.FLOAT32, [1, 78]))
        B = Tensor(LogicalShape(plaidml.DType.FLOAT32, [78, 78]))
        O = matmul(A, B)
        program = Program('matmul', [O])

    def test_avg(self):
        I = Tensor(LogicalShape(plaidml.DType.FLOAT32, [1, 78]))
        O = avg(I)
        program = Program('avg', [O])

    def test_avg_stages(self):
        I = Tensor(LogicalShape(plaidml.DType.FLOAT32, [1, 78]))
        O = avg_stages(I)
        program = Program('avg_stages', [O])

    def test_avg_merge(self):
        I = Tensor(LogicalShape(plaidml.DType.FLOAT32, [1, 78]))
        O = avg_merge(I)
        program = Program('avg_merge', [O])

    def test_max_pool_1d(self):
        I = Tensor(LogicalShape(plaidml.DType.FLOAT32, [10]), name='I')
        O = max_pool_1d(I)
        program = Program('max_pool_1d', [O])

    def test_max_pool_1d_odd(self):
        I = Tensor(LogicalShape(plaidml.DType.FLOAT32, [10]), name='I')
        O = max_pool_1d_odd(I)
        program = Program('max_pool_1d_odd', [O])

    def test_skip(self):
        I = Tensor(LogicalShape(plaidml.DType.FLOAT32, [1, 78]))
        O = skip(I)
        program = Program('skip', [O])

    def test_conv_1d(self):
        I = Tensor(LogicalShape(plaidml.DType.FLOAT32, [1, 224, 3]))
        K = Tensor(LogicalShape(plaidml.DType.FLOAT32, [3, 3, 1]))
        O = conv_1d(I, K)
        program = Program('conv_1d', [O])

    def test_conv_2d_dilated(self):
        I = Tensor(LogicalShape(plaidml.DType.FLOAT32, [1, 224, 224, 1]))
        K = Tensor(LogicalShape(plaidml.DType.FLOAT32, [3, 3, 1, 32]))
        O = conv_2d_dilated(I, K)
        program = Program('conv_2d_dilated', [O])

    def test_complex_conv_2d(self):
        I = Tensor(LogicalShape(plaidml.DType.FLOAT32, [1, 224, 224, 3, 3]))
        K = Tensor(LogicalShape(plaidml.DType.FLOAT32, [3, 3, 3, 3, 32]))
        O = complex_conv_2d(I, K, 1, 2, 1, 2)
        program = Program('complex_conv_2d', [O])
