# Copyright 2020 Intel Corporation.
# Note:
#    This file is being used by sphinx docs to pull in code blocks.
#    Code blocks are pulled into docs/usage/*.rst
#    Any changes made here may upset the docs.

import unittest

import plaidml
from plaidml.edsl import *


def sum_over_axis(I):
    M, N = TensorDims(2)
    m, n = TensorIndexes(2)
    I.bind_dims(M, N)
    O = TensorOutput(N)
    O = Contraction().outShape(N).outAccess(n).sum(I[m, n])
    return O


def max_over_axis(I):
    M, N = TensorDims(2)
    m, n = TensorIndexes(2)
    I.bind_dims(M, N)
    return Contraction().outShape(N).outAccess(n).max(I[m, n])


def matmul(A, B):
    I, J, K = TensorDims(3)
    i, j, k = TensorIndexes(3)
    A.bind_dims(I, K)
    B.bind_dims(K, J)
    return Contraction().outShape(I, J).outAccess(i, j).sum(A[i, k] * B[k, j])


def global_min(I):
    i, j, k = TensorIndexes(3)
    Neg = -I
    O = -Contraction().max(Neg[i, j, k])
    return O


def avg(I):
    X, Y = TensorDims(2)
    x, y = TensorIndexes(2)
    I.bind_dims(X, Y)
    Sum = TensorOutput()
    Sum = Contraction().outShape(Y).outAccess(y).sum(I[x, y])
    return Sum / X


def avg_stages(I):
    X, Y = TensorDims(2)
    x, y = TensorIndexes(2)
    I.bind_dims(X, Y)
    Sum = TensorOutput()
    Sum = Contraction().sum(I[x, y])
    PartialMean = Sum / X
    return PartialMean / Y


def avg_merge(I):
    X, Y = TensorDims(2)
    x, y = TensorIndexes(2)
    I.bind_dims(X, Y)
    Sum = TensorOutput()
    Sum = Contraction().sum(I[x, y])
    return Sum / (X * Y)


def skip(I):
    M, N = TensorDims(2)
    i, j = TensorIndexes(2)
    I.bind_dims(M, N)
    O = TensorOutput(N)
    return Contraction().outShape(N).outAccess(2 * i).sum(I[2 * i, j])


def cumsum(I):
    N = TensorDim()
    i, k = TensorIndexes(2)
    I.bind_dims(N)
    return Contraction().outShape(N).outAccess(i).sum(I[k]).add_constraint(i - k < N)


def layer():
    # layer() has not been implemented for the Python frontend
    pass


def trace(A, B):
    At = trace(A, "Pre-summation")
    C = At + B
    return trace(C, "Post-summation")


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

    def test_skip(self):
        I = Tensor(LogicalShape(plaidml.DType.FLOAT32, [1, 78]))
        O = skip(I)
        program = Program('skip', [O])

    def test_trace(self):
        A = Tensor(LogicalShape(plaidml.DType.FLOAT32, [3, 3]))
        B = Tensor(LogicalShape(plaidml.DType.FLOAT32, [3, 3]))
        O = trace(A, B)
        program = Program('trace', [O])
