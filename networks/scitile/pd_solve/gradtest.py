import numpy as np
import os
import sys

import plaidml2
import plaidml2.edsl as edsl
import plaidml2.exec as pld_exec
import plaidml2.op as op

import unittest
import numpy.testing as npt


def matmul_2_1(A, B):
    I, J = edsl.TensorDims(2)
    i, j = edsl.TensorIndexes(2)
    A.bind_dims(I, J)
    B.bind_dims(J)
    C = edsl.TensorOutput(I)
    C[i] += A[i, j] * B[j]
    return C


def matmul_1_2(A, B):
    I, J = edsl.TensorDims(2)
    i, j = edsl.TensorIndexes(2)
    A.bind_dims(I)
    B.bind_dims(I, J)
    C = edsl.TensorOutput(I)
    C[j] += A[i] * B[i, j]
    return C


def matmul_1_1(A, B):
    I = edsl.TensorDim()
    i = edsl.TensorIndex()
    A.bind_dims(I)
    B.bind_dims(I)
    C = edsl.TensorOutput()
    C[()] += A[i] * B[i]
    return C


def matmul_2_2(A, B):
    I, J, K = edsl.TensorDims(3)
    i, j, k = edsl.TensorIndexes(3)
    A.bind_dims(I, J)
    B.bind_dims(J, K)
    C = edsl.TensorOutput(I, K)
    C[(i, k)] += A[i, j] * B[j, k]
    return C


def distance(a, b):
    I, J = edsl.TensorDims(2)
    i, j = edsl.TensorIndexes(2)
    neg = -b
    a.bind_dims(I)
    neg.bind_dims(J)
    C = edsl.TensorOutput(I, J)
    C[(i, j)] = a[i] + neg[j]
    return C


def get_jacobian(Is, I_dat, O, wrt):
    dy = edsl.jacobian(O, [wrt])[0]
    program = edsl.Program('program', [O, dy])
    binder = pld_exec.Binder(program)
    executable = binder.compile()
    for i in range(len(Is)):
        binder.input(Is[i]).copy_from_ndarray(I_dat[i])
    executable.run()
    return binder.output(dy).as_ndarray()


class GradTest(unittest.TestCase):

    def test_1(self):
        np_A = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
        np_x = np.array([1, 2, 3])

        dtype = plaidml2.DType.FLOAT32
        A = edsl.Tensor(edsl.LogicalShape(dtype, np_A.shape))
        x = edsl.Tensor(edsl.LogicalShape(dtype, np_x.shape))
        y = matmul_2_1(A, x)
        test_result = get_jacobian([A, x], [np_A, np_x], y, x)

        true_result = np_A

        npt.assert_allclose(test_result, true_result)

    def test_2(self):
        np_A = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
        np_B = np.array([1, 2, 3])
        np_x = np.array([1, 2, 3])

        dtype = plaidml2.DType.FLOAT32
        A = edsl.Tensor(edsl.LogicalShape(dtype, np_A.shape))
        B = edsl.Tensor(edsl.LogicalShape(dtype, np_B.shape))
        x = edsl.Tensor(edsl.LogicalShape(dtype, np_x.shape))
        y = matmul_1_1(B, matmul_2_1(A, x))
        test_result = get_jacobian([A, B, x], [np_A, np_B, np_x], y, x)

        true_result = np.squeeze(np.expand_dims(np_B, 0) @ np_A)

        npt.assert_allclose(test_result, true_result)

    def test_3(self):
        np_A = np.array([[1., 2., 3., 4.], [4., 5., 6., 7.], [7., 8., 9., 10.]])
        np_B = np.array([1., 2., 3.])
        np_x = np.array([1., 2., 3., 4.])

        dtype = plaidml2.DType.FLOAT32
        A = edsl.Tensor(edsl.LogicalShape(dtype, np_A.shape))
        B = edsl.Tensor(edsl.LogicalShape(dtype, np_B.shape))
        x = edsl.Tensor(edsl.LogicalShape(dtype, np_x.shape))
        y = matmul_1_1(B, matmul_2_1(A, x))
        test_result = get_jacobian([A, B, x], [np_A, np_B, np_x], y, x)

        true_result = np.squeeze(np.expand_dims(np_B, 0) @ np_A)

        npt.assert_allclose(test_result, true_result)

    def test_4(self):
        np_A = np.array([[1., 2., 3., 4.], [4., 5., 6., 7.], [7., 8., 9., 10.]])
        np_B = np.array([1., 2., 3.])
        np_x = np.array([[1., 2., 3., 4.], [5., 6., 7., 8.]]).T

        dtype = plaidml2.DType.FLOAT32
        A = edsl.Tensor(edsl.LogicalShape(dtype, np_A.shape))
        B = edsl.Tensor(edsl.LogicalShape(dtype, np_B.shape))
        x = edsl.Tensor(edsl.LogicalShape(dtype, np_x.shape))
        y = matmul_1_2(B, matmul_2_2(A, x))

        with npt.assert_raises(Exception):
            test_result = get_jacobian([A, B, x], [np_A, np_B, np_x], y, x)

        # Expected result when rank>=2 wrt input is supported:
        # true_result = np.expand_dims(np.squeeze(np.expand_dims(np_B, 0) @ np_A), 1)*np.ones(np_x.shape)
        # npt.assert_allclose(test_result, true_result)

    def test_5(self):
        np_A = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
        np_x = np.array([1, 2, 3])

        dtype = plaidml2.DType.FLOAT32
        A = edsl.Tensor(edsl.LogicalShape(dtype, np_A.shape))
        x = edsl.Tensor(edsl.LogicalShape(dtype, np_x.shape))
        y = matmul_2_1(A, x * x)
        test_result = get_jacobian([A, x], [np_A, np_x], y, x)

        true_result = np_A * np.expand_dims(2 * np_x, 0)

        npt.assert_allclose(test_result, true_result)

    def test_6(self):
        np_x = np.array([1., 2., 3.])
        np_b = np.array([1., 2., 3.])

        dtype = plaidml2.DType.FLOAT32
        b = edsl.Tensor(edsl.LogicalShape(dtype, np_b.shape))
        x = edsl.Tensor(edsl.LogicalShape(dtype, np_x.shape))
        y = distance(x, b)

        test_result = get_jacobian([x, b], [np_x, np_b], y, x)

        true_result = np.zeros((3, 3, 3))
        true_result[0, :, 0] = 1
        true_result[1, :, 1] = 1
        true_result[2, :, 2] = 1

        npt.assert_allclose(test_result, true_result)

    def test_7(self):
        np_x = np.array([1., 2., 3.])

        dtype = plaidml2.DType.FLOAT32
        x = edsl.Tensor(edsl.LogicalShape(dtype, np_x.shape))
        y = distance(x, x)

        test_result = get_jacobian([x], [np_x], y, x)

        true_result = np.zeros((3, 3, 3))
        true_result[:, :, 0] = [[0, 1, 1], [-1, 0, 0], [-1, 0, 0]]
        true_result[:, :, 1] = [[0, -1, 0], [1, 0, 1], [0, -1, 0]]
        true_result[:, :, 2] = [[0, 0, -1], [0, 0, -1], [1, 1, 0]]

        npt.assert_allclose(test_result, true_result)

    def test_8(self):
        np_x = np.array([1., 2., 3.])
        np_A = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])

        dtype = plaidml2.DType.FLOAT32
        A = edsl.Tensor(edsl.LogicalShape(dtype, np_A.shape))
        x = edsl.Tensor(edsl.LogicalShape(dtype, np_x.shape))

        y = matmul_2_2(A, distance(x, x))

        J_test = get_jacobian([A, x], [np_A, np_x], y, x)
        J_true = np.zeros((3, 3, 3))
        J_true[:, :, 0] = [[-5, 1, 1], [-11, 4, 4], [-17, 7, 7]]
        J_true[:, :, 1] = [[2, -4, 2], [5, -10, 5], [8, -16, 8]]
        J_true[:, :, 2] = [[3, 3, -3], [6, 6, -9], [9, 9, -15]]
        npt.assert_allclose(J_true, J_test)


if __name__ == '__main__':
    unittest.main()
