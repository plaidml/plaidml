# Copyright 2020 Intel Corporation

import functools
import platform
import sys
import unittest

import numpy as np

import plaidml
import plaidml.exec
from plaidml import Program
from plaidml.edsl import *

DEFAULT_DEVICE = 'llvm_cpu.0'
DEFAULT_TARGET = 'llvm_cpu'


def dot(A, B):
    I, J, K = TensorDims(3)
    i, j, k = TensorIndexes(3)
    A.bind_dims(I, K)
    B.bind_dims(K, J)
    return Contraction().outShape(I, J).outAccess(i, j).sum(A[i, k] * B[k, j]).build()


def relu(I):
    zero = cast(0.0, I.dtype)
    return select(I < 0.0, zero, I)


def softmax(X):
    I, J = TensorDims(2)
    i, j = TensorIndexes(2)
    X.bind_dims(I, J)
    M = Contraction().outShape(I, 1).outAccess(i, 0).max(X[i, j]).build()
    E = exp(X - M)
    N = Contraction().outShape(I, 1).outAccess(i, 0).sum(E[i, j]).build()
    return E / N


def conv_1d(I, K):
    N, X, KX, CI, CO = TensorDims(5)
    n, x, k, ci, co = TensorIndexes(5)
    I.bind_dims(N, X, CI)
    K.bind_dims(KX, CI, CO)
    return Contraction() \
        .outShape(N, X - KX + 1, CO) \
        .outAccess(n, x, co) \
        .sum(I[n, x + k, ci] * K[k, ci, co]) \
        .build()


def conv_2d_dilated(I, K):
    N, X, Y, KX, KY, CI, CO = TensorDims(7)
    n, x, y, kx, ky, ci, co = TensorIndexes(7)
    I.bind_dims(N, X, Y, CI)
    K.bind_dims(KX, KY, CI, CO)
    return Contraction() \
        .outShape(N, X - 2 * (KX - 1), Y - 3 * (KY - 1), CO) \
        .outAccess(n, x, y, co) \
        .sum(I[n, x + 2 * kx, y + 3 * ky, ci] * K[kx, ky, ci, co]) \
        .build()


def conv_2d(I, K, I_layout='NHWC', K_layout='HWCK'):
    I_lens = TensorLens(I_layout, 'NHWC')
    K_lens = TensorLens(K_layout, 'HWCK')
    I = I.use(I_lens)
    K = K.use(K_lens)
    CI, CO, K0, K1, N, X0, X1 = TensorDims(7)
    n, x0, x1, k0, k1, ci, co = TensorIndexes(7)
    I.bind_dims(N, X0, X1, CI)
    K.bind_dims(K0, K1, CI, CO)
    return Contraction(I_lens) \
        .outShape(N, X0 - (K0 - 1), X1 - (K1 - 1), CO) \
        .outAccess(n, x0, x1, co) \
        .sum(I[n, x0 + k0 - (K0 // 2), x1 + k1 - (K1 // 2), ci] * K[k0, k1, ci, co]) \
        .build()


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

    # Compute the effective kernel size after dilation
    EK0, EK1 = TensorDims(2)
    EK0 = d0 * (K0 - 1) + 1
    EK1 = d1 * (K1 - 1) + 1

    # Compute the padding offset
    P0, P1 = TensorDims(2)
    P0 = ((Y0 - 1) * s0 + EK0 - X0) // 2
    P1 = ((Y1 - 1) * s1 + EK1 - X1) // 2

    # Compute the convolution
    return Contraction() \
        .outShape(N, Y0, Y1, G, GCO) \
        .outAccess(n, x0, x1, g, gco) \
        .sum(I[n, s0 * x1 + d0 * k0 - P0, s1 * x1 + d1 * k1 - P1, g, gci] * K[k0, k1, g, gci, gco]) \
        .build()


def max_pool_2d(I):
    N, X0, X1, C = TensorDims(4)
    n, x0, x1, i, j, c = TensorIndexes(6)
    I.bind_dims(N, X0, X1, C)
    return Contraction() \
        .outShape(N, (X0 + 1) // 2, (X1 + 1) // 2, C) \
        .outAccess(n, x0, x1, c) \
        .max(I[n, 2 * x0 + i, 2 * x1 + j, c]) \
        .add_constraint(i < 2) \
        .add_constraint(j < 2) \
        .build()


def flatten(X):
    X_dims = TensorDims(X.rank)
    X.bind_dims(*X_dims)
    product = functools.reduce(lambda x, y: x * y, X_dims[1:-1])
    return reshape(X, (1, product))


def normalize(X):
    idxs = TensorIndexes(X.rank)
    XSqr = X * X
    X_MS = Contraction().sum(XSqr[idxs]).build()
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
    Max = Contraction().outShape(X0, X2).outAccess(x0, x2).max(I[x0, x1, x2]).build()
    IX = index([X1], 0)
    O = Contraction() \
        .outShape(X0, X2) \
        .outAccess(x0, x2) \
        .max(cond(I[x0, x1, x2], Max[x0, x2], IX[x1])) \
        .build()
    return cast(O, DType.UINT32)


def global_min(I):
    i, j, k = TensorIndexes(3)
    Neg = -I
    O_Neg = Contraction().max(Neg[i, j, k]).build()
    O = -O_Neg
    return O


class TestEdsl(unittest.TestCase):
    maxDiff = None

    def runProgram(self, program):
        program.compile()
        if platform.system() == 'Windows':
            # the Orc JIT in LLVM is currently broken on windows.
            return
        input_buffers = [plaidml.Buffer(shape) for shape in program.inputs]
        output_buffers = [plaidml.Buffer(shape) for shape in program.outputs]
        executable = plaidml.exec.Executable(program)
        executable.run(input_buffers, output_buffers)

    def checkProgram(self, program, inputs, expected):
        if platform.system() == 'Windows':
            # the Orc JIT in LLVM is currently broken on windows.
            return
        outputs = plaidml.exec.run(program, inputs)
        for out, exp in zip(outputs, expected):
            self.assertEqual(out.tolist(), exp)

    def test_eltwise(self):
        A = Placeholder(plaidml.DType.FLOAT32, [3])
        B = Placeholder(plaidml.DType.FLOAT32, [3])
        O = A + B

        program = Program('eltwise_add', (A, B), [O])
        self.checkProgram(program, [
            np.array([1, 2, 3]),
            np.array([3, 2, 1]),
        ], [
            [4, 4, 4],
        ])

    def test_broadcast_cmp(self):
        A = Placeholder(plaidml.DType.INT64, [3, 4])
        B = Placeholder(plaidml.DType.INT64, [3, 1])
        O = cast(A >= B, DType.INT64)

        program = Program('broadcast_cmp', [A, B], [O])
        self.checkProgram(program, [
            np.array([
                [0, 1, 2, 3],
                [4, 5, 6, 7],
                [8, 9, 10, 11],
            ]),
            np.array([
                [0],
                [6],
                [12],
            ]),
        ], [
            [
                [1, 1, 1, 1],
                [0, 0, 1, 1],
                [0, 0, 0, 0],
            ],
        ])

    def test_higher_precision_constants(self):
        I = Placeholder(plaidml.DType.FLOAT32, [3, 3])
        O = I + 1 + 2.0

        program = Program('higher_precision_constants', [I], [O])
        self.checkProgram(program, [
            np.array([
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
            ]),
        ], [
            [
                [4, 5, 6],
                [7, 8, 9],
                [10, 11, 12],
            ],
        ])

    def test_dot(self):
        A = Placeholder(plaidml.DType.FLOAT32, [1, 784])
        B = Placeholder(plaidml.DType.FLOAT32, [784, 784])
        O = dot(A, B)
        program = Program('dot', [A, B], [O])
        self.runProgram(program)

    def test_cast_scalar(self):
        I = Placeholder(plaidml.DType.INT32)
        O = cast(I, plaidml.DType.FLOAT32)
        program = Program('cast', [I], [O])
        self.checkProgram(program, [np.array([(3)])], [3.0])

    def test_cast_folder(self):
        I = Placeholder(plaidml.DType.INT32, [3])
        O = cast(I, plaidml.DType.INT32)
        program = Program('cast', [I], [O])
        self.checkProgram(program, [np.array([(1, 2, 3)])], [[1, 2, 3]])

    def test_conv_1d(self):
        I = Placeholder(plaidml.DType.FLOAT32, [1, 224, 3])
        K = Placeholder(plaidml.DType.FLOAT32, [3, 3, 1])
        O = conv_1d(I, K)
        program = Program('conv_1d', [I, K], [O])
        self.runProgram(program)

    def test_conv_2d_dilated(self):
        I = Placeholder(plaidml.DType.FLOAT32, [1, 224, 224, 1])
        K = Placeholder(plaidml.DType.FLOAT32, [3, 3, 1, 32])
        O = conv_2d_dilated(I, K)
        program = Program('conv_2d_dilated', [I, K], [O])
        self.runProgram(program)

    def test_complex_conv_2d(self):
        I = Placeholder(plaidml.DType.FLOAT32, [1, 224, 224, 3, 3])
        K = Placeholder(plaidml.DType.FLOAT32, [3, 3, 3, 3, 32])
        O = complex_conv_2d(I, K, 1, 2, 1, 2)
        program = Program('complex_conv_2d', [I, K], [O])
        self.runProgram(program)

    def test_complex_conv_2d_dim_mismatch(self):
        I = Placeholder(plaidml.DType.FLOAT32, [1, 1, 1, 1, 1])
        K = Placeholder(plaidml.DType.FLOAT32, [1, 1, 1, 1, 1])
        O = complex_conv_2d(I, K, 1, 2, 1, 2)
        program = Program('complex_conv_2d', [I, K], [O])
        self.runProgram(program)

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
        program = Program('mnist_mlp', [I, K1, B1, K2, B2, K3, B3], [D3])
        self.runProgram(program)

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
        self.assertEqual(str(F.compute_shape()), '1x12100xf32')
        K3 = Placeholder(plaidml.DType.FLOAT32, [12100, 128])
        B3 = Placeholder(plaidml.DType.FLOAT32, [128])
        D1 = relu(dot(F, K3) + B3)
        # model.add(Dense(num_classes, activation='softmax'))
        K4 = Placeholder(plaidml.DType.FLOAT32, [128, 100])
        B4 = Placeholder(plaidml.DType.FLOAT32, [100])
        D2 = softmax(dot(D1, K4) + B4)
        program = Program('mnist_cnn', [I, K1, B1, K2, B2, K3, B3, K4, B4], [D2])
        self.runProgram(program)

    def test_arg_max(self):
        I = Placeholder(plaidml.DType.FLOAT32, [1, 10, 10])
        O = arg_max(I)
        program = Program('arg_max', [I], [O])
        self.assertEqual(str(O.compute_shape()), '1x10xui32')
        self.runProgram(program)

    def test_global_min(self):
        I = Placeholder(plaidml.DType.FLOAT32, [10, 10, 10], name='I')
        O = global_min(I)
        program = Program('global_min', [I], [O])
        self.runProgram(program)

    def test_unique_names(self):
        A = Placeholder(plaidml.DType.FLOAT32, name='A')
        B = Placeholder(plaidml.DType.FLOAT32, name='B')
        C0 = Placeholder(plaidml.DType.FLOAT32, name='C')
        C1 = Placeholder(plaidml.DType.FLOAT32, name='C')
        program = Program('unique_names', [A, B, C0, C1], [A + B + C0 + C1])
        self.runProgram(program)

    def test_lars_momentum4d(self):
        X_shape = TensorShape(plaidml.DType.FLOAT32, [4, 7, 3, 9])
        LR_Shape = TensorShape(plaidml.DType.FLOAT32)
        X = Placeholder(X_shape)
        Grad = Placeholder(X_shape)
        Veloc = Placeholder(X_shape)
        LR = Placeholder(LR_Shape)
        R = lars_momentum(X, Grad, Veloc, LR, 1. / 1024., 1. / 2048., 1. / 8.)
        program = Program('lars_momentum4d', [X, Grad, Veloc, LR], R)
        self.runProgram(program)

    def test_repeat_elts(self):
        I = Placeholder(plaidml.DType.FLOAT32, [10, 10, 10])
        N0, N1, N2 = TensorDims(3)
        n0, n1, n2, k = TensorIndexes(4)
        I.bind_dims(N0, N1, N2)
        O = Contraction() \
            .outShape(N0, 3 * N1, N2) \
            .outAccess(n0, 3 * n1 + k, n2) \
            .assign(I[n0, n1, n2]) \
            .add_constraint(k < 3) \
            .build()
        program = Program('repeat_elts', [I], [O])
        self.runProgram(program)

    def test_use_default(self):
        P = Placeholder(plaidml.DType.FLOAT32, [1, 7, 10, 10])
        I = Placeholder(plaidml.DType.FLOAT32, [1, 10, 10])
        B, N1, N2 = TensorDims(3)
        b, i1, i2 = TensorIndexes(3)
        I.bind_dims(B, N1, N2)
        O = Contraction() \
            .outShape(B, 7, N1, N2) \
            .outAccess(b, 3, i1, i2) \
            .init(P) \
            .assign(I[b, i1, i2]) \
            .build()
        program = Program('use_default', [I, P], [O])
        self.runProgram(program)

    def test_defract(self):
        I = Placeholder(plaidml.DType.FLOAT32, [3], name='I')
        K = Placeholder(plaidml.DType.FLOAT32, [3], name='K')
        i, j = TensorIndexes(2)
        O = Contraction().outShape(5).outAccess(i).sum(I[(i - j + 1) // 2] * K[j]).build()
        program = Program('defract_test', [I, K], [O])
        self.checkProgram(program, [
            np.array([1, 2, 3]),
            np.array([1, 2, 3]),
        ], [
            [2, 5, 4, 9, 6],
        ])

    def test_defract_short(self):
        I = Placeholder(plaidml.DType.FLOAT32, [3], name='I')
        i, j = TensorIndexes(2)
        O = Contraction().outShape(6).outAccess(i).sum(I[(i - 1) // 2]).build()
        program = Program('defract_short_test', [I], [O])
        self.checkProgram(program, [
            np.array([1, 2, 3]),
        ], [
            [0, 1, 0, 2, 0, 3],
        ])

    def test_defract_long(self):
        shape = [1, 3, 3, 1]
        I = Placeholder(plaidml.DType.FLOAT32, shape, name='I')
        K = Placeholder(plaidml.DType.FLOAT32, shape, name='K')
        n, x0, x1, c0, c1, co, ci, k0, k1 = TensorIndexes(9)
        O = Contraction() \
            .outShape(1, 5, 5, 1) \
            .outAccess(n, x0, x1, co) \
            .sum(I[n, (x0 + k0 - 1) // 2, (x1 + k1 - 1) // 2, ci] * K[2 - k0, 2 - k1, co, ci]) \
            .build()
        program = Program('defract_long', [I, K], [O])
        self.checkProgram(program, [
            np.array([[
                [[1], [3], [-1]],
                [[0], [2], [4]],
                [[1], [-1], [-2]],
            ]]),
            np.array([[
                [[2], [3], [4]],
                [[6], [-3], [-1]],
                [[-1], [-2], [1]],
            ]]),
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
        O = Contraction().outShape(5).outAccess(i).sum(I[(i - j + 1) // 2] * K[j]).build()
        program = Program('this-is-not an identifier', [I, K], [O])
        self.checkProgram(program, [
            np.array([1, 2, 3]),
            np.array([1, 2, 3]),
        ], [
            [2, 5, 4, 9, 6],
        ])

    def test_identity(self):
        I = Placeholder(plaidml.DType.FLOAT32, [3], name='I')
        program = Program('identity', [I], [I])
        self.checkProgram(program, [np.array([(1, 2, 3)])], [[1, 2, 3]])

    @unittest.skip('TODO: exception needs to be thrown')
    def test_assignment_exceptions(self):
        A = Placeholder(plaidml.DType.FLOAT32, [5, 1], name='A')
        B = Placeholder(plaidml.DType.FLOAT32, [1, 5], name='B')
        L, M, N = TensorDims(3)
        i, j, k = TensorIndexes(3)
        A.bind_dims(L, M)
        B.bind_dims(M, N)
        O = Contraction().outShape(L, N).outAccess(i, j).assign(A[i, k] * B[k, j]).build()
        program = Program('assignment_non_exception', [A, B], [O])
        self.checkProgram(program, [
            np.array([[1], [2], [3], [4], [5]]),
            np.array([1, 2, 3, 4, 5]),
        ], [[
            [1., 2., 3., 4., 5.],
            [2., 4., 6., 8., 10.],
            [3., 6., 9., 12., 15.],
            [4., 8., 12., 16., 20.],
            [5., 10., 15., 20., 25.],
        ]])

        O = Contraction().outShape(L, N).outAccess(i, j).assign(B[i, k] * A[k, j]).build()
        with self.assertRaises(plaidml.Error) as cm:
            program = Program('assignment_exception', [A, B], [O])
        self.assertTrue("illegal assignment aggregation" in str(cm.exception))

    def test_two_outputs(self):
        I = Placeholder(plaidml.DType.FLOAT32, [3], name='I')
        program1 = Program('two_outputs', [I], [I, I])
        self.checkProgram(program1, [
            np.array([1, 2, 3]),
        ], [
            [1, 2, 3],
            [1, 2, 3],
        ])

        O1 = I
        O2 = I
        program2 = Program('two_outputs', [I], [O1, O2])
        self.assertMultiLineEqual(str(program1), str(program2))
        self.checkProgram(program2, [
            np.array([1, 2, 3]),
        ], [
            [1, 2, 3],
            [1, 2, 3],
        ])

    def test_placeholder(self):
        I = Placeholder(plaidml.DType.INT32)
        program = Program('placeholder_noshape', [I], [I])
        self.assertEqual(str(I.compute_shape()), "si32")
        I = Placeholder(plaidml.DType.INT32, [1, 1])
        program = Program('placeholder_noname', [I], [I])
        self.assertEqual(str(I.compute_shape()), "1x1xsi32")
        I = Placeholder(plaidml.DType.INT32, [1, 1], name='I')
        program = Program('placeholder_with_name', [I], [I])
        self.assertEqual(str(I.compute_shape()), "1x1xsi32")

    def test_constant_add(self):

        def makeBuffer(shape, data):
            buf = plaidml.Buffer(shape)
            buf.copy_from_ndarray(data)
            return buf

        shape = TensorShape(plaidml.DType.INT32, [4])
        a_buf = makeBuffer(shape, np.array([4, 3, 2, 1]))
        b_buf = makeBuffer(shape, np.array([1, 2, 3, 4]))
        A = Constant(a_buf, name="A")
        B = Constant(b_buf, name="B")
        O = A + B
        program = Program('const_add', [], [O])
        self.checkProgram(program, [], [[5, 5, 5, 5]])
        self.assertEqual(len(program.inputs), 0)
        self.assertEqual(len(program.outputs), 1)

    def test_collect_passes(self):
        A = Placeholder(plaidml.DType.FLOAT32, [10, 10])
        B = Placeholder(plaidml.DType.FLOAT32, [10, 10])
        C = dot(A, B)
        program = Program('collect_passes', [A, B], [C])
        program.compile(debug=True)

        first_pass = program.passes[0]
        self.assertEqual('tile', first_pass[0])

    def test_trace(self):
        I = Placeholder(plaidml.DType.FLOAT32, [3, 3])
        O = trace(I, 'msg')
        program = Program('trace', [I], [O])

    def test_lens(self):
        I = Placeholder(plaidml.DType.FLOAT32, [1, 224, 224, 3])
        K = Placeholder(plaidml.DType.FLOAT32, [3, 3, 3, 32])
        O = conv_2d(I, K, I_layout='NHWC', K_layout='HWCK')
        program = Program('conv2d_nhwc', [I, K], [O])
        self.runProgram(program)

        I = Placeholder(plaidml.DType.FLOAT32, [1, 3, 224, 224])
        K = Placeholder(plaidml.DType.FLOAT32, [3, 32, 7, 7])
        O = conv_2d(I, K, I_layout='NCHW', K_layout='CKHW')
        program = Program('conv2d_nchw', [I, K], [O])
        self.runProgram(program)

        def transpose(I, layout='MN'):
            lens = TensorLens(layout, 'MN')
            I = I.use(lens)
            M, N = TensorDims(2)
            i, j = TensorIndexes(2)
            I.bind_dims(M, N)
            return Contraction(lens).outShape(N, M).outAccess(j, i).assign(I[i, j]).build()

        input = np.array([
            [1, 2, 3],
            [4, 5, 6],
        ])
        expected = [
            [1, 4],
            [2, 5],
            [3, 6],
        ]

        I = Placeholder(plaidml.DType.FLOAT32, [2, 3])
        O = transpose(I, 'MN')
        program = Program('transpose_mn', [I], [O])
        self.checkProgram(program, [input], [expected])

        I = Placeholder(plaidml.DType.FLOAT32, [2, 3])
        O = transpose(I, 'NM')
        program = Program('transpose_nm', [I], [O])
        self.checkProgram(program, [input], [expected])

    def test_argsort_1d(self):
        I = Placeholder(plaidml.DType.FLOAT32, [20])
        O = argsort(I, axis=0, mode=SortMode.ASC)
        program = Program('argsort_1d', [I], [O])

    def test_argsort_2d_axis_0(self):
        I = Placeholder(plaidml.DType.FLOAT32, [5, 4])
        O = argsort(I, axis=0, mode=SortMode.DESC)
        program = Program('argsort_2d_axis_0', [I], [O])

    def test_argsort_2d_axis_1(self):
        I = Placeholder(plaidml.DType.FLOAT32, [5, 4])
        O = argsort(I, axis=1)
        program = Program('argsort_2d_axis_1', [I], [O])

    def test_argsort_3d_axis_1_asc(self):
        I = Placeholder(plaidml.DType.FLOAT32, [5, 4, 3])
        O = argsort(I, axis=1, mode=SortMode.ASC)
        program = Program('argsort_3d_axis_1_asc', [I], [O])

    def test_argsort_3d_axis_1_desc(self):
        I = Placeholder(plaidml.DType.FLOAT32, [5, 4, 3])
        O = argsort(I, axis=1, mode=SortMode.DESC)
        program = Program('argsort_3d_axis_1_desc', [I], [O])

    def test_argsort_3d_last_axis(self):
        I = Placeholder(plaidml.DType.FLOAT32, [5, 4, 3])
        O = argsort(I, axis=-1)
        program = Program('argsort_3d_last_axis', [I], [O])


if __name__ == '__main__':
    plaidml.settings.set('PLAIDML_DEVICE', DEFAULT_DEVICE)
    plaidml.settings.set('PLAIDML_TARGET', DEFAULT_TARGET)
    unittest.main()
