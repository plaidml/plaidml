# Copyright 2019 Intel Corporation.

import argparse
import functools
import sys
import unittest

import plaidml2 as plaidml
from plaidml2.edsl import *
import plaidml2.settings as plaidml_settings
import plaidml2.exec as plaidml_exec

device = plaidml_settings.get('PLAIDML_DEVICE')
target = plaidml_settings.get('PLAIDML_TARGET')


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


def conv_2d(I, K):
    CI, CO, K0, K1, N, X0, X1 = TensorDims(7)
    n, x0, x1, k0, k1, ci, co = TensorIndexes(7)
    I.bind_dims(N, X0, X1, CI)
    K.bind_dims(K0, K1, CI, CO)
    R = TensorOutput(N, X0 - (K0 - 1), X1 - (K1 - 1), CO)
    R[n, x0, x1, co] += I[n, x0 + k0 - (K0 // 2), x1 + k1 - (K1 // 2), ci] * K[k0, k1, ci, co]
    return R


def max_pool_2d(I):
    N, X0, X1, C = TensorDims(4)
    n, x0, x1, i, j, c = TensorIndexes(6)
    I.bind_dims(N, X0, X1, C)
    R = TensorOutput(N, (X0 + 1) // 2, (X1 + 1) // 2, C)
    if i < 2 and j < 2:
        R[n, x0, x1, c] >= I[n, 2 * x0 + i, 2 * x1 + j, c]
    return R


def flatten(X):
    X_dims = TensorDims(X.shape.ndims)
    X.bind_dims(*X_dims)
    product = functools.reduce(lambda x, y: x * y, X_dims[1:-1])
    return reshape(X, (1, product))


def normalize(X):
    idxs = TensorIndexes(X.shape.ndims)
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
    One = Tensor(LogicalShape(plaidml.DType.FLOAT32))
    T = TensorOutput(X1)
    T[x1] = One[()]
    IX = index(T, 0)
    O = TensorOutput(X0, X2)
    O[x0, x2] >= cond(I[x0, x1, x2], Max[x0, x2], IX[x1])
    return as_uint(O, 32)


class TestEdsl(unittest.TestCase):
    maxDiff = None

    def test_mnist_mlp(self):
        # model.add(Dense(512, activation='relu', input_shape=(784,)))
        I = Tensor(LogicalShape(plaidml.DType.FLOAT32, [1, 784]))
        K1 = Tensor(LogicalShape(plaidml.DType.FLOAT32, [784, 512]))
        B1 = Tensor(LogicalShape(plaidml.DType.FLOAT32, [512]))
        D1 = relu(dot(I, K1) + B1)
        # model.add(Dense(512, activation='relu'))
        K2 = Tensor(LogicalShape(plaidml.DType.FLOAT32, [512, 512]))
        B2 = Tensor(LogicalShape(plaidml.DType.FLOAT32, [512]))
        D2 = relu(dot(D1, K2) + B2)
        # model.add(Dense(10, activation='softmax'))
        K3 = Tensor(LogicalShape(plaidml.DType.FLOAT32, [512, 10]))
        B3 = Tensor(LogicalShape(plaidml.DType.FLOAT32, [10]))
        D3 = softmax(dot(D2, K3) + B3)
        program = Program('mnist_mlp', [D3])
        self.assertMultiLineEqual(
            str(program), '''function (
  _X0[_X0_0, _X0_1],
  _X1[_X1_0, _X1_1],
  _X3[_X3_0],
  _X9[_X9_0, _X9_1],
  _X11[_X11_0],
  _X17[_X17_0, _X17_1],
  _X19[_X19_0]
) -> (
  _X25
) {
  _X2[x0, x2 : 1, 512] = +(_X0[x0, x1] * _X1[x1, x2]);
  _X4 = add(_X2, _X3);
  _X5 = 0.000000;
  _X6 = cmp_lt(_X4, _X5);
  _X7 = 0.000000;
  _X8 = cond(_X6, _X7, _X4);
  _X10[x0, x2 : 1, 512] = +(_X8[x0, x1] * _X9[x1, x2]);
  _X12 = add(_X10, _X11);
  _X13 = 0.000000;
  _X14 = cmp_lt(_X12, _X13);
  _X15 = 0.000000;
  _X16 = cond(_X14, _X15, _X12);
  _X18[x0, x2 : 1, 10] = +(_X16[x0, x1] * _X17[x1, x2]);
  _X20 = add(_X18, _X19);
  _X21[x0, 0 : 1, 1] = >(_X20[x0, x1]);
  _X22 = sub(_X20, _X21);
  _X23 = exp(_X22);
  _X24[x0, 0 : 1, 1] = +(_X23[x0, x1]);
  _X25 = div(_X23, _X24);
}
''')

    def test_mnist_cnn(self):
        # model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
        I = Tensor(LogicalShape(plaidml.DType.FLOAT32, [1, 224, 224, 1]))
        K1 = Tensor(LogicalShape(plaidml.DType.FLOAT32, [3, 3, 1, 32]))
        B1 = Tensor(LogicalShape(plaidml.DType.FLOAT32, [32]))
        C1 = relu(conv_2d(I, K1) + B1)
        # model.add(Conv2D(64, (3, 3), activation='relu'))
        K2 = Tensor(LogicalShape(plaidml.DType.FLOAT32, [3, 3, 32, 64]))
        B2 = Tensor(LogicalShape(plaidml.DType.FLOAT32, [64]))
        C2 = relu(conv_2d(C1, K2) + B2)
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        P1 = max_pool_2d(C2)
        # model.add(Flatten())
        F = flatten(P1)
        self.assertEqual(str(F.shape), 'fp32(1, 12100)')
        K3 = Tensor(LogicalShape(plaidml.DType.FLOAT32, [12100, 128]))
        B3 = Tensor(LogicalShape(plaidml.DType.FLOAT32, [128]))
        D1 = relu(dot(F, K3) + B3)
        # model.add(Dense(num_classes, activation='softmax'))
        K4 = Tensor(LogicalShape(plaidml.DType.FLOAT32, [128, 100]))
        B4 = Tensor(LogicalShape(plaidml.DType.FLOAT32, [100]))
        D2 = softmax(dot(D1, K4) + B4)
        program = Program('mnist_cnn', [D2])
        self.assertMultiLineEqual(
            str(program), '''function (
  _X0[_X0_0, _X0_1, _X0_2, _X0_3],
  _X1[_X1_0, _X1_1, _X1_2, _X1_3],
  _X3[_X3_0],
  _X9[_X9_0, _X9_1, _X9_2, _X9_3],
  _X11[_X11_0],
  _X21[_X21_0, _X21_1],
  _X23[_X23_0],
  _X29[_X29_0, _X29_1],
  _X31[_X31_0]
) -> (
  _X37
) {
  _X2[x0, x1, x3, x6 : 1, 222, 222, 32] = +(_X0[x0, -1 + x1 + x2, -1 + x3 + x4, x5] * _X1[x2, x4, x5, x6]);
  _X4 = add(_X2, _X3);
  _X5 = 0.000000;
  _X6 = cmp_lt(_X4, _X5);
  _X7 = 0.000000;
  _X8 = cond(_X6, _X7, _X4);
  _X10[x0, x1, x3, x6 : 1, 220, 220, 64] = +(_X8[x0, -1 + x1 + x2, -1 + x3 + x4, x5] * _X9[x2, x4, x5, x6]);
  _X12 = add(_X10, _X11);
  _X13 = 0.000000;
  _X14 = cmp_lt(_X12, _X13);
  _X15 = 0.000000;
  _X16 = cond(_X14, _X15, _X12);
  _X17[x0, x1, x3, x5 : 1, 110, 110, 64] = >(_X16[x0, 2*x1 + x2, 2*x3 + x4, x5]), x2 < 2, x4 < 2;
  _X18 = 1;
  _X19 = 12100;
  _X20 = reshape(_X17, _X18, _X19);
  _X22[x0, x2 : 1, 128] = +(_X20[x0, x1] * _X21[x1, x2]);
  _X24 = add(_X22, _X23);
  _X25 = 0.000000;
  _X26 = cmp_lt(_X24, _X25);
  _X27 = 0.000000;
  _X28 = cond(_X26, _X27, _X24);
  _X30[x0, x2 : 1, 100] = +(_X28[x0, x1] * _X29[x1, x2]);
  _X32 = add(_X30, _X31);
  _X33[x0, 0 : 1, 1] = >(_X32[x0, x1]);
  _X34 = sub(_X32, _X33);
  _X35 = exp(_X34);
  _X36[x0, 0 : 1, 1] = +(_X35[x0, x1]);
  _X37 = div(_X35, _X36);
}
''')

    def test_arg_max(self):
        I = Tensor(LogicalShape(plaidml.DType.FLOAT32, [1, 10, 10]))
        O = arg_max(I)
        self.assertEqual(str(O.shape), 'u32(1, 10)')
        program = Program('arg_max', [O])
        self.assertMultiLineEqual(
            str(program), '''function (
  _X0[_X0_0, _X0_1, _X0_2],
  _X2[]
) -> (
  _X8
) {
  _X1[x0, x2 : 1, 10] = >(_X0[x0, x1, x2]);
  _X3[x0 : 10] = =(_X2[]);
  _X4 = 0;
  _X5 = index(_X3, _X4);
  _X6[x0, x2 : 1, 10] = >(_X0[x0, x1, x2] == _X1[x0, x2] ? _X5[x1]);
  _X7 = 32;
  _X8 = as_uint(_X6, _X7);
}
''')

    def test_global_min(self):
        I = Tensor(LogicalShape(plaidml.DType.FLOAT32, [10, 10, 10]), name='I')
        i, j, k = TensorIndexes(3)
        O_Neg = TensorOutput()
        Neg = -I
        O_Neg[()] >= Neg[i, j, k]
        O = -O_Neg
        program = Program('global_min', [O])
        self.assertMultiLineEqual(
            str(program), '''function (
  I[I_0, I_1, I_2]
) -> (
  _X2
) {
  _X0 = neg(I);
  _X1[] = >(_X0[x0, x1, x2]);
  _X2 = neg(_X1);
}
''')

    def test_cum_sum(self):
        I = Tensor(LogicalShape(plaidml.DType.FLOAT32, [10]), name='I')
        N = TensorDim()
        i, k = TensorIndexes(2)
        I.bind_dims(N)
        O = TensorOutput(N)
        if i - k < N:
            O[i] += I[k]
        program = Program('cum_sum', [O])
        self.assertMultiLineEqual(
            str(program), '''function (
  I[I_0]
) -> (
  _X0
) {
  _X0[x1 : 10] = +(I[x0]), -x0 + x1 < 10;
}
''')

    def test_invalid_shape_error(self):
        O = TensorOutput(TensorDims(3))
        with self.assertRaises(plaidml.Error) as err:
            shape = O.shape
        self.assertTrue('Cannot compute shape' in str(err.exception))

    def test_unique_names(self):
        A = Tensor(LogicalShape(plaidml.DType.FLOAT32), name='A')
        B = Tensor(LogicalShape(plaidml.DType.FLOAT32), name='B')
        C0 = Tensor(LogicalShape(plaidml.DType.FLOAT32), name='C')
        C1 = Tensor(LogicalShape(plaidml.DType.FLOAT32), name='C')
        program = Program('unique_names', [A + B + C0 + C1])
        self.assertMultiLineEqual(
            str(program), '''function (
  A[],
  B[],
  C[],
  C0[]
) -> (
  _X2
) {
  _X0 = add(A, B);
  _X1 = add(_X0, C);
  _X2 = add(_X1, C0);
}
''')

    def test_lars_momentum_4d(self):
        X_shape = LogicalShape(plaidml.DType.FLOAT32, [4, 7, 3, 9])
        LR_Shape = LogicalShape(plaidml.DType.FLOAT32)
        X = Tensor(X_shape)
        Grad = Tensor(X_shape)
        Veloc = Tensor(X_shape)
        LR = Tensor(LR_Shape)
        R = lars_momentum(X, Grad, Veloc, LR, 1. / 1024., 1. / 2048., 1. / 8.)
        program = Program('lars_momentum_4d', R)
        self.assertMultiLineEqual(
            str(program), '''function (
  _X1[_X1_0, _X1_1, _X1_2, _X1_3],
  _X3[],
  _X6[_X6_0, _X6_1, _X6_2, _X6_3],
  _X11[_X11_0, _X11_1, _X11_2, _X11_3]
) -> (
  _X24,
  _X23
) {
  _X0 = 0.125000;
  _X2 = mul(_X0, _X1);
  _X4 = 0.000977;
  _X5 = mul(_X3, _X4);
  _X7 = mul(_X6, _X6);
  _X8[] = +(_X7[x0, x1, x2, x3]);
  _X9 = sqrt(_X8);
  _X10 = mul(_X5, _X9);
  _X12 = mul(_X11, _X11);
  _X13[] = +(_X12[x0, x1, x2, x3]);
  _X14 = sqrt(_X13);
  _X15 = 0.000488;
  _X16 = mul(_X15, _X9);
  _X17 = add(_X14, _X16);
  _X18 = div(_X10, _X17);
  _X19 = 0.000488;
  _X20 = mul(_X19, _X6);
  _X21 = add(_X11, _X20);
  _X22 = mul(_X18, _X21);
  _X23 = add(_X2, _X22);
  _X24 = sub(_X6, _X23);
}
''')

    def test_repeat_elts(self):
        I = Tensor(LogicalShape(plaidml.DType.FLOAT32, [10, 10, 10]))
        N0, N1, N2 = TensorDims(3)
        n0, n1, n2, k = TensorIndexes(4)
        I.bind_dims(N0, N1, N2)
        O = TensorOutput(N0, 3 * N1, N2)
        if k < 3:
            O[n0, 3 * n1 + k, n2] = I[n0, n1, n2]
        O.no_defract()
        program = Program('repeat_elts', [O])
        self.assertMultiLineEqual(
            str(program), '''function (
  _X0[_X0_0, _X0_1, _X0_2]
) -> (
  _X1
) {
  _X1[x0, 3*x1 + x3, x2 : 10, 30, 10] = =(_X0[x0, x1, x2]), x3 < 3 no_defract;
}
''')

    def test_use_default(self):
        P = Tensor(LogicalShape(plaidml.DType.FLOAT32, [1, 7, 10, 10]))
        I = Tensor(LogicalShape(plaidml.DType.FLOAT32, [1, 10, 10]))
        B, N1, N2 = TensorDims(3)
        b, i1, i2 = TensorIndexes(3)
        I.bind_dims(B, N1, N2)
        O = TensorOutput(B, 7, N1, N2)
        O[b, 3, i1, i2] = I[b, i1, i2]
        O.use_default(P)
        program = Program('use_default', [O])
        self.assertMultiLineEqual(
            str(program), '''function (
  _X0[_X0_0, _X0_1, _X0_2, _X0_3],
  _X1[_X1_0, _X1_1, _X1_2]
) -> (
  _X2
) {
  _X2[x0, 3, x1, x2 : 1, 7, 10, 10] = =(_X1[x0, x1, x2]) default _X0;
}
''')

    def testDefract(self):
        I = Tensor(LogicalShape(plaidml.DType.FLOAT32, [3]), name='I')
        K = Tensor(LogicalShape(plaidml.DType.FLOAT32, [3]), name='K')
        N = TensorDim()
        M = TensorDim()
        i = TensorIndex()
        j = TensorIndex()
        I.bind_dims(N)
        K.bind_dims(M)
        O = TensorOutput(5)

        O[i] += (I[(i - j + 1) // 2] * K[j])
        program = Program('defract_test', [O])
        self.assertMultiLineEqual(
            str(program), '''function (
  I[I_0],
  K[K_0]
) -> (
  _X0
) {
  _X0[x0 : 5] = +(I[1/2 + 1/2*x0 - 1/2*x1] * K[x1]);
}
''')
        outputs = run(program, [(I, np.array([1, 2, 3])), (K, np.array([1, 2, 3]))])
        self.assertEquals(outputs[0].tolist(), [2, 5, 4, 9, 6])

    def testDefractShort(self):

        I = Tensor(LogicalShape(plaidml.DType.FLOAT32, [3]), name='I')
        N = TensorDim()
        i = TensorIndex()
        j = TensorIndex()
        I.bind_dims(N)
        O = TensorOutput(6)
        O[i] += (I[(i - 1) // 2])
        program = Program('defract_short_test', [O])
        self.assertMultiLineEqual(
            str(program), '''function (
  I[I_0]
) -> (
  _X0
) {
  _X0[x0 : 6] = +(I[-1/2 + 1/2*x0]);
}
''')
        outputs = run(program, [(I, np.array([1, 2, 3]))])
        self.assertEquals(outputs[0].tolist(), [0, 1, 0, 2, 0, 3])

    def testDefractLong(self):

        shape = [1, 3, 3, 1]
        I = Tensor(LogicalShape(plaidml.DType.FLOAT32, shape), name='I')
        K = Tensor(LogicalShape(plaidml.DType.FLOAT32, shape), name='K')
        n, x0, x1, c0, c1, co, ci, k0, k1 = TensorIndexes(9)
        O = TensorOutput(1, 5, 5, 1)
        O[n, x0, x1, co] += (I[n, (x0 + k0 - 1) // 2,
                               (x1 + k1 - 1) // 2, ci] * K[2 - k0, 2 - k1, co, ci])
        program = Program('defract_long', [O])
        self.assertMultiLineEqual(
            str(program), '''function (
  I[I_0, I_1, I_2, I_3],
  K[K_0, K_1, K_2, K_3]
) -> (
  _X0
) {
  _X0[x0, x1, x3, x6 : 1, 5, 5, 1] = +(I[x0, -1/2 + 1/2*x1 + 1/2*x2, -1/2 + 1/2*x3 + 1/2*x4, x5] * K[2 - x2, 2 - x4, x6, x5]);
}
''')
        outputs = run(program, [(I, np.random.rand(1, 3, 3, 1)), (K, np.random.rand(1, 3, 3, 1))])

    def testFunkyLayerNames(self):
        '''Exercises fix for plaidml bug #241

        Now that we emit keras layer names as 'pid' attribute values, in order
        to help link tile code back to its origin while debugging, we must
        reformat those names as valid tile identifiers. If we are doing that,
        this test will pass, otherwise we'll get a syntax error.'''

        I = Tensor(LogicalShape(plaidml.DType.FLOAT32, [3]), name='I')
        K = Tensor(LogicalShape(plaidml.DType.FLOAT32, [3]), name='K')
        N = TensorDim()
        M = TensorDim()
        i = TensorIndex()
        j = TensorIndex()
        I.bind_dims(N)
        K.bind_dims(M)
        O = TensorOutput(5)

        O[i] += (I[(i - j + 1) // 2] * K[j])

        program = Program('this-is-not an identifier', [O])
        self.assertMultiLineEqual(
            str(program), '''function (
  I[I_0],
  K[K_0]
) -> (
  _X0
) {
  _X0[x0 : 5] = +(I[1/2 + 1/2*x0 - 1/2*x1] * K[x1]);
}
''')
        outputs = run(program, [(I, np.array([1, 2, 3])), (K, np.array([1, 2, 3]))])
        self.assertEquals(outputs[0].tolist(), [2, 5, 4, 9, 6])

    def testTileIdentity(self):
        I = Tensor(LogicalShape(plaidml.DType.FLOAT32, [3]), name='I')
        program = Program('tile_identity', [I])
        self.assertMultiLineEqual(str(program), '''function (
  I[I_0]
) -> (
  _X0
) {
  _X0 = ident(I);
}
''')
        outputs = run(program, [(I, np.array([(1, 2, 3)]))])
        self.assertEquals(outputs[0].tolist(), [1, 2, 3])

    def testAssignmentExceptions(self):
        A = Tensor(LogicalShape(plaidml.DType.FLOAT32, [5, 1]), name='A')
        B = Tensor(LogicalShape(plaidml.DType.FLOAT32, [1, 5]), name='B')
        L, M, N = TensorDims(3)
        i, j, k = TensorIndexes(3)
        A.bind_dims(L, M)
        B.bind_dims(M, N)
        O = TensorOutput(L, N)
        O[i, j] = A[i, k] * B[k, j]
        program = Program('assignment_non_exception', [O])
        self.assertMultiLineEqual(
            str(program), '''function (
  A[A_0, A_1],
  B[B_0, B_1]
) -> (
  _X0
) {
  _X0[x0, x2 : 5, 5] = =(A[x0, x1] * B[x1, x2]);
}
''')
        outputs = run(program, [(A, np.array([[1], [2], [3], [4], [5]])),
                                (B, np.array([1, 2, 3, 4, 5]))])
        self.assertEquals(outputs[0].tolist(),
                          [[1., 2., 3., 4., 5.], [2., 4., 6., 8., 10.], [3., 6., 9., 12., 15.],
                           [4., 8., 12., 16., 20.], [5., 10., 15., 20., 25.]])

        O = TensorOutput(L, N)
        O[i, j] = B[i, k] * A[k, j]
        program = Program('assignment_exception', [O])
        self.assertMultiLineEqual(
            str(program), '''function (
  B[B_0, B_1],
  A[A_0, A_1]
) -> (
  _X0
) {
  _X0[x0, x2 : 5, 5] = =(B[x0, x1] * A[x1, x2]);
}
''')
        outputs = run(program, [(A, np.array([[1], [2], [3], [4], [5]])),
                                (B, np.array([1, 2, 3, 4, 5]))])
        self.assertEquals(outputs[0].tolist(),
                          [[25., 0., 0., 0., 0.], [0., 0., 0., 0., 0.], [0., 0., 0., 0., 0.],
                           [0., 0., 0., 0., 0.], [0., 0., 0., 0., 0.]])

    def testTwoOutputs(self):
        I = Tensor(LogicalShape(plaidml.DType.FLOAT32, [3]), name='I')
        program1 = Program('two_outputs', [I, I])
        self.assertMultiLineEqual(
            str(program1), '''function (
  I[I_0]
) -> (
  _X1,
  _X0
) {
  _X0 = ident(I);
  _X1 = ident(I);
}
''')
        outputs = run(program1, [(I, np.array([(1, 2, 3)]))])
        self.assertEquals(outputs[0].tolist(), [1, 2, 3])
        self.assertEquals(outputs[1].tolist(), [1, 2, 3])

        N = TensorDim(3)
        O1 = TensorOutput(N)
        O2 = TensorOutput(N)
        O1 = I
        O2 = I
        program2 = Program('two_outputs', [O1, O2])
        self.assertMultiLineEqual(str(program1), str(program2))

        outputs = run(program2, [(I, np.array([(1, 2, 3)]))])
        self.assertEquals(outputs[0].tolist(), [1, 2, 3])
        self.assertEquals(outputs[1].tolist(), [1, 2, 3])


def run(program, inputs):

    def make_buffer(tensor):
        # convert LogicalShape into TensorShape
        shape = plaidml.TensorShape(tensor.shape.dtype, tensor.shape.int_dims)
        return plaidml.Buffer(device, shape)

    ibindings = [(x, make_buffer(x)) for x, y in inputs]
    obindings = [(x, make_buffer(x)) for x in program.outputs]

    exe = plaidml_exec.Executable(program, device, target, ibindings, obindings)
    return [x.as_ndarray() for x in exe([y for x, y in inputs])]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', type=int, default=0)
    args, remainder = parser.parse_known_args()
    unittest.main(argv=sys.argv[:1] + remainder, verbosity=args.verbose + 1)
