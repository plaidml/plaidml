# Copyright 2020 Intel Corporation

import functools
import platform
import sys
import unittest

import numpy as np

import plaidml
import plaidml.exec
from plaidml.edsl import *

DEFAULT_DEVICE = 'llvm_cpu.0'
DEFAULT_TARGET = 'llvm_cpu'


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
    IX = index([X1], 0)
    O = TensorOutput(X0, X2)
    O[x0, x2] >= cond(I[x0, x1, x2], Max[x0, x2], IX[x1])
    return cast(O, DType.UINT32)


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


class TestEdsl(unittest.TestCase):
    maxDiff = None

    def checkProgram(self, program, inputs, expected):
        outputs = self.runProgram(program, inputs)
        for i in range(len(expected)):
            self.assertEqual(outputs[i].tolist(), expected[i])

    def runProgram(self, program, inputs=[]):
        if platform.system() == 'Windows':
            # the Orc JIT in LLVM is currently broken on windows.
            return
        return plaidml.exec.run(program, inputs)

    def test_broadcast_cmp(self):
        A = Input(plaidml.DType.UINT64, [3, 4])
        B = Input(plaidml.DType.UINT64, [3, 1])
        O = cast(A >= B, DType.UINT64)

        program = Program('broadcast_cmp', [O])
        self.checkProgram(program, [
            (A, np.array([
                [0, 1, 2, 3],
                [4, 5, 6, 7],
                [8, 9, 10, 11],
            ])),
            (B, np.array([
                [0],
                [6],
                [12],
            ])),
        ], [
            [
                [1, 1, 1, 1],
                [0, 0, 1, 1],
                [0, 0, 0, 0],
            ],
        ])

    def test_higher_precision_constants_invalid_negative(self):
        I = Input(plaidml.DType.FLOAT32, [3, 3])
        O = I * (-2)

        try:
            program = Program('higher_precision_constants', [O],
                              floatx=plaidml.DType.FLOAT64,
                              intx=plaidml.DType.UINT64)
        except Exception:
            return
        self.fail("expected exception")

    def test_higher_precision_constants(self):
        I = Input(plaidml.DType.FLOAT32, [3, 3])
        O = I + 1 + 2.0

        program = Program('higher_precision_constants', [O],
                          floatx=plaidml.DType.FLOAT64,
                          intx=plaidml.DType.UINT64)
        self.checkProgram(program, [
            (I, np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])),
        ], [
            [[4, 5, 6], [7, 8, 9], [10, 11, 12]],
        ])

    def test_matmul(self):
        A = Input(plaidml.DType.FLOAT32, [1, 784])
        B = Input(plaidml.DType.FLOAT32, [784, 784])
        O = matmul(A, B)
        program = Program('matmul', [O])
        self.runProgram(program)

    def test_cast_scalar(self):
        I = Input(plaidml.DType.INT32, {})
        O = cast(I, plaidml.DType.FLOAT32)
        program = Program('cast', [O])
        self.checkProgram(program, [
            (I, np.array([(3)])),
        ], [3.0])

    def test_cast_folder(self):
        I = Input(plaidml.DType.INT32, {3})
        O = cast(I, plaidml.DType.INT32)
        program = Program('cast', [O])
        self.checkProgram(program, [
            (I, np.array([(1, 2, 3)])),
        ], [[1, 2, 3]])

    def test_conv_1d(self):
        I = Input(plaidml.DType.FLOAT32, [1, 224, 3])
        K = Input(plaidml.DType.FLOAT32, [3, 3, 1])
        O = conv_1d(I, K)
        program = Program('conv_1d', [O])
        self.runProgram(program)

    def test_conv_2d_dilated(self):
        I = Input(plaidml.DType.FLOAT32, [1, 224, 224, 1])
        K = Input(plaidml.DType.FLOAT32, [3, 3, 1, 32])
        O = conv_2d_dilated(I, K)
        program = Program('conv_2d_dilated', [O])
        self.runProgram(program)

    def test_complex_conv_2d(self):
        I = Input(plaidml.DType.FLOAT32, [1, 224, 224, 3, 3])
        K = Input(plaidml.DType.FLOAT32, [3, 3, 3, 3, 32])
        O = complex_conv_2d(I, K, 1, 2, 1, 2)
        program = Program('complex_conv_2d', [O])
        self.runProgram(program)

    @unittest.skip(
        'TODO: currently segfaults mismatched dimensions error needs to be printed correctly')
    def test_complex_conv_2d_dim_mismatch(self):
        I = Input(plaidml.DType.FLOAT32, [1, 1, 1, 1, 1])
        K = Input(plaidml.DType.FLOAT32, [1, 1, 1, 1, 1])
        O = complex_conv_2d(I, K, 1, 2, 1, 2)
        program = Program('complex_conv_2d', [O])

    def test_mnist_mlp(self):
        # model.add(Dense(512, activation='relu', input_shape=(784,)))
        I = Input(plaidml.DType.FLOAT32, [1, 784])
        K1 = Input(plaidml.DType.FLOAT32, [784, 512])
        B1 = Input(plaidml.DType.FLOAT32, [512])
        D1 = relu(dot(I, K1) + B1)
        # model.add(Dense(512, activation='relu'))
        K2 = Input(plaidml.DType.FLOAT32, [512, 512])
        B2 = Input(plaidml.DType.FLOAT32, [512])
        D2 = relu(dot(D1, K2) + B2)
        # model.add(Dense(10, activation='softmax'))
        K3 = Input(plaidml.DType.FLOAT32, [512, 10])
        B3 = Input(plaidml.DType.FLOAT32, [10])
        D3 = softmax(dot(D2, K3) + B3)
        program = Program('mnist_mlp', [D3])

    def test_mnist_cnn(self):
        # model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
        I = Input(plaidml.DType.FLOAT32, [1, 224, 224, 1])
        K1 = Input(plaidml.DType.FLOAT32, [3, 3, 1, 32])
        B1 = Input(plaidml.DType.FLOAT32, [32])
        C1 = relu(conv_2d(I, K1) + B1)
        # model.add(Conv2D(64, (3, 3), activation='relu'))
        K2 = Input(plaidml.DType.FLOAT32, [3, 3, 32, 64])
        B2 = Input(plaidml.DType.FLOAT32, [64])
        C2 = relu(conv_2d(C1, K2) + B2)
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        P1 = max_pool_2d(C2)
        # model.add(Flatten())
        F = flatten(P1)
        self.assertEqual(str(F.compute_shape()), 'tensor<1x12100xf32>')
        K3 = Input(plaidml.DType.FLOAT32, [12100, 128])
        B3 = Input(plaidml.DType.FLOAT32, [128])
        D1 = relu(dot(F, K3) + B3)
        # model.add(Dense(num_classes, activation='softmax'))
        K4 = Input(plaidml.DType.FLOAT32, [128, 100])
        B4 = Input(plaidml.DType.FLOAT32, [100])
        D2 = softmax(dot(D1, K4) + B4)
        program = Program('mnist_cnn', [D2])

    def test_arg_max(self):
        I = Input(plaidml.DType.FLOAT32, [1, 10, 10])
        O = arg_max(I)
        program = Program('arg_max', [O])
        self.assertEqual(str(O.compute_shape()), 'tensor<1x10xui32>')

    def test_global_min(self):
        I = Input(plaidml.DType.FLOAT32, [10, 10, 10], name='I')
        O = global_min(I)
        program = Program('global_min', [O])

    def test_invalid_shape_error(self):
        O = TensorOutput(TensorDims(3))
        with self.assertRaises(plaidml.Error) as err:
            shape = O.compute_shape()
        self.assertTrue('Cannot compute shape' in str(err.exception))

    def test_unique_names(self):
        A = Input(plaidml.DType.FLOAT32, name='A')
        B = Input(plaidml.DType.FLOAT32, name='B')
        C0 = Input(plaidml.DType.FLOAT32, name='C')
        C1 = Input(plaidml.DType.FLOAT32, name='C')
        program = Program('unique_names', [A + B + C0 + C1])

    def test_lars_momentum4d(self):
        X_shape = LogicalShape(plaidml.DType.FLOAT32, [4, 7, 3, 9])
        LR_Shape = LogicalShape(plaidml.DType.FLOAT32)
        X = Input(X_shape)
        Grad = Input(X_shape)
        Veloc = Input(X_shape)
        LR = Input(LR_Shape)
        R = lars_momentum(X, Grad, Veloc, LR, 1. / 1024., 1. / 2048., 1. / 8.)
        program = Program('lars_momentum4d', R)

    def test_repeat_elts(self):
        I = Input(plaidml.DType.FLOAT32, [10, 10, 10])
        N0, N1, N2 = TensorDims(3)
        n0, n1, n2, k = TensorIndexes(4)
        I.bind_dims(N0, N1, N2)
        O = TensorOutput(N0, 3 * N1, N2)
        O[n0, 3 * n1 + k, n2] = I[n0, n1, n2]
        O.add_constraint(k < 3)
        O.no_reduce()
        program = Program('repeat_elts', [O])

    def test_use_default(self):
        P = Input(plaidml.DType.FLOAT32, [1, 7, 10, 10])
        I = Input(plaidml.DType.FLOAT32, [1, 10, 10])
        B, N1, N2 = TensorDims(3)
        b, i1, i2 = TensorIndexes(3)
        I.bind_dims(B, N1, N2)
        O = TensorOutput(B, 7, N1, N2)
        O[b, 3, i1, i2] = I[b, i1, i2]
        O.use_default(P)
        program = Program('use_default', [O])

    def test_defract(self):
        I = Input(plaidml.DType.FLOAT32, [3], name='I')
        K = Input(plaidml.DType.FLOAT32, [3], name='K')
        i, j = TensorIndexes(2)
        O = TensorOutput(5)
        O[i] += (I[(i - j + 1) // 2] * K[j])
        program = Program('defract_test', [O])
        self.checkProgram(program, [
            (I, np.array([1, 2, 3])),
            (K, np.array([1, 2, 3])),
        ], [
            [2, 5, 4, 9, 6],
        ])

    @unittest.skip('FIXME: incorrect output')
    def test_defract_short(self):
        I = Input(plaidml.DType.FLOAT32, [3], name='I')
        i, j = TensorIndexes(2)
        O = TensorOutput(6)
        O[i] += (I[(i - 1) // 2])
        program = Program('defract_short_test', [O])
        self.checkProgram(program, [
            (I, np.array([1, 2, 3])),
        ], [
            [0, 1, 0, 2, 0, 3],
        ])

    @unittest.skip('FIXME: incorrect output')
    def test_defract_long(self):
        shape = [1, 3, 3, 1]
        I = Input(plaidml.DType.FLOAT32, shape, name='I')
        K = Input(plaidml.DType.FLOAT32, shape, name='K')
        n, x0, x1, c0, c1, co, ci, k0, k1 = TensorIndexes(9)
        O = TensorOutput(1, 5, 5, 1)
        O[n, x0, x1, co] += (I[n, (x0 + k0 - 1) // 2,
                               (x1 + k1 - 1) // 2, ci] * K[2 - k0, 2 - k1, co, ci])
        program = Program('defract_long', [O])
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

        I = Input(plaidml.DType.FLOAT32, [3], name='I')
        K = Input(plaidml.DType.FLOAT32, [3], name='K')
        i, j = TensorIndexes(2)
        O = TensorOutput(5)
        O[i] += (I[(i - j + 1) // 2] * K[j])
        program = Program('this-is-not an identifier', [O])
        self.checkProgram(program, [
            (I, np.array([1, 2, 3])),
            (K, np.array([1, 2, 3])),
        ], [
            [2, 5, 4, 9, 6],
        ])

    def test_identity(self):
        I = Input(plaidml.DType.FLOAT32, [3], name='I')
        program = Program('identity', [I])
        self.checkProgram(program, [
            (I, np.array([(1, 2, 3)])),
        ], [
            [1, 2, 3],
        ])

    @unittest.skip('TODO: exception needs to be thrown')
    def test_assignment_exceptions(self):
        A = Input(plaidml.DType.FLOAT32, [5, 1], name='A')
        B = Input(plaidml.DType.FLOAT32, [1, 5], name='B')
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
        I = Input(plaidml.DType.FLOAT32, [3], name='I')
        program1 = Program('two_outputs', [I, I])
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
        I1 = Input(plaidml.DType.INT32)
        program1 = Program('placeholder_noshape', [I1])
        I2 = Input(LogicalShape(plaidml.DType.INT32))
        program2 = Program('placeholder_noshape', [I2])
        self.assertEqual(str(I1.compute_shape()), "tensor<si32>")
        self.assertEqual(str(I2.compute_shape()), "tensor<si32>")
        self.assertMultiLineEqual(str(program1), str(program2))

    def test_placeholder_noname(self):
        I1 = Input(plaidml.DType.INT32, [1, 1])
        program1 = Program('placeholder_noname', [I1])
        I2 = Input(LogicalShape(plaidml.DType.INT32, [1, 1]))
        program2 = Program('placeholder_noname', [I2])
        self.assertEqual(str(I1.compute_shape()), "tensor<1x1xsi32>")
        self.assertEqual(str(I2.compute_shape()), "tensor<1x1xsi32>")
        self.assertMultiLineEqual(str(program1), str(program2))

    def test_placeholder_with_name(self):
        I1 = Input(plaidml.DType.INT32, [1, 1], name='I')
        program1 = Program('placeholder_with_name', [I1])
        I2 = Input(LogicalShape(plaidml.DType.INT32, [1, 1]), name='I')
        program2 = Program('placeholder_with_name', [I2])
        self.assertEqual(str(I1.compute_shape()), "tensor<1x1xsi32>")
        self.assertEqual(str(I2.compute_shape()), "tensor<1x1xsi32>")
        self.assertMultiLineEqual(str(program1), str(program2))

    def test_constant_add(self):
        a_buf = plaidml.Buffer(plaidml.TensorShape(plaidml.DType.INT32, [4]),
                               device=DEFAULT_DEVICE)
        b_buf = plaidml.Buffer(plaidml.TensorShape(plaidml.DType.INT32, [4]),
                               device=DEFAULT_DEVICE)
        a_buf.copy_from_ndarray(np.array([4, 3, 2, 1]))
        b_buf.copy_from_ndarray(np.array([1, 2, 3, 4]))
        A = Constant(LogicalShape(plaidml.DType.INT32, [4]), a_buf, name="A")
        B = Constant(LogicalShape(plaidml.DType.INT32, [4]), b_buf, name="B")
        O = A + B
        program = Program('const_add', [O])
        self.checkProgram(program, [], [
            [5, 5, 5, 5],
        ])

    def test_collect_passes(self):
        A = Input(plaidml.DType.FLOAT32, [10, 10])
        B = Input(plaidml.DType.FLOAT32, [10, 10])
        C = dot(A, B)
        program = Program('collect_passes', [C], debug=True)

        first_pass = program.passes[0]
        self.assertEqual('tile', first_pass[0])
        self.assertEqual(str(program), first_pass[1])


if __name__ == '__main__':
    plaidml.settings.set('PLAIDML_DEVICE', DEFAULT_DEVICE)
    plaidml.settings.set('PLAIDML_TARGET', DEFAULT_TARGET)
    unittest.main()
