import numpy as np
import os
import sys

import plaidml2
import plaidml2.edsl as edsl
import plaidml2.exec as pld_exec
import plaidml2.op as op

import unittest
import numpy.testing as npt


def matmul_2_2(A, B):
    I, J, K = edsl.TensorDims(3)
    i, j, k = edsl.TensorIndexes(3)
    A.bind_dims(I, J)
    B.bind_dims(J, K)
    C = edsl.TensorOutput(I, K)
    C[(i, k)] += A[i, j] * B[j, k]
    return C


def matmul_2_1(A, b):
    I, J = edsl.TensorDims(2)
    i, j = edsl.TensorIndexes(2)
    A.bind_dims(I, J)
    b.bind_dims(J)
    C = edsl.TensorOutput(I)
    C[(i)] += A[i, j] * b[j]
    return C


def dist(a, b):
    I, J = edsl.TensorDims(2)
    i, j = edsl.TensorIndexes(2)
    a.bind_dims(I)
    neg = -b
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

    def test_ident(self):
        np_x = np.array([1, 2, 3])

        dtype = plaidml2.DType.FLOAT32
        x = edsl.Tensor(edsl.LogicalShape(dtype, np_x.shape))

        test_result = get_jacobian([x], [np_x], x, x)
        true_result = np.eye(3)

        npt.assert_allclose(test_result, true_result)

    def test_square(self):
        np_x = np.array([1, 2, 3])

        dtype = plaidml2.DType.FLOAT32
        x = edsl.Tensor(edsl.LogicalShape(dtype, np_x.shape))
        y = op.square(x)

        test_result = get_jacobian([x], [np_x], y, x)
        true_result = np.array([[2, 0, 0], [0, 4, 0], [0, 0, 6]])

        npt.assert_allclose(test_result, true_result)

    def test_assign(self):
        np_x = np.array([1, 2, 3])
        np_b = np.array([1, 1, 1])

        dtype = plaidml2.DType.FLOAT32
        x = edsl.Tensor(edsl.LogicalShape(dtype, np_x.shape))
        b = edsl.Tensor(edsl.LogicalShape(dtype, np_b.shape))
        y = op.square(dist(x, b))

        test_result = get_jacobian([x, b], [np_x, np_b], y, x)
        true_result = np.zeros((3, 3, 3))
        true_result[0, :, 0] = 0
        true_result[1, :, 1] = 2
        true_result[2, :, 2] = 4

        npt.assert_allclose(test_result, true_result)

    def test_matmul_2_1(self):
        np_A = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
        np_x = np.array([1., 2., 3.])

        dtype = plaidml2.DType.FLOAT32
        A = edsl.Tensor(edsl.LogicalShape(dtype, np_A.shape))
        x = edsl.Tensor(edsl.LogicalShape(dtype, np_x.shape))
        y = matmul_2_1(A, x)

        test_result = get_jacobian([A, x], [np_A, np_x], y, x)
        true_result = np_A

        npt.assert_allclose(test_result, true_result)

    def test_matmul_2_2(self):
        np_A = np.array([[1., 2.], [3., 4.]])
        np_x = np.array([[5., 6.], [7., 8.]])

        dtype = plaidml2.DType.FLOAT32
        A = edsl.Tensor(edsl.LogicalShape(dtype, np_A.shape))
        x = edsl.Tensor(edsl.LogicalShape(dtype, np_x.shape))
        y = matmul_2_2(A, x)

        test_result = get_jacobian([A, x], [np_A, np_x], y, x)
        true_result = np.array([[[[1, 0], [2, 0]], [[0, 1], [0, 2]]],
                                [[[3, 0], [4, 0]], [[0, 3], [0, 4]]]])

        npt.assert_allclose(test_result, true_result)

    def test_chain(self):
        np_x = np.array([1., 2., 3.])
        np_A = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])

        dtype = plaidml2.DType.FLOAT32
        A = edsl.Tensor(edsl.LogicalShape(dtype, np_A.shape))
        x = edsl.Tensor(edsl.LogicalShape(dtype, np_x.shape))
        y = matmul_2_2(A, dist(x, x))

        J_test = get_jacobian([A, x], [np_A, np_x], y, x)
        J_true = np.zeros((3, 3, 3))
        J_true[:, :, 0] = [[-5, 1, 1], [-11, 4, 4], [-17, 7, 7]]
        J_true[:, :, 1] = [[2, -4, 2], [5, -10, 5], [8, -16, 8]]
        J_true[:, :, 2] = [[3, 3, -3], [6, 6, -9], [9, 9, -15]]

        npt.assert_allclose(J_true, J_test)


if __name__ == '__main__':
    unittest.main()
