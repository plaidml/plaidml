# Copyright 2019 Intel Corporation.

import argparse
import functools
import sys
import unittest

import plaidml2 as plaidml
from plaidml2.edsl import *
import plaidml2.exec as plaidml_exec


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
    R[n, x0, x1, c] >= I[n, 2 * x0 + i, 2 * x1 + j, c]
    R.add_constraints([i < 2, j < 2])
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
            str(program), '''

!fp32 = type tensor<!eltwise.fp32>
module {
  func @mnist_mlp(%arg0: tensor<10x!eltwise.fp32>, %arg1: tensor<512x!eltwise.fp32>, %arg2: tensor<512x!eltwise.fp32>, %arg3: tensor<1x784x!eltwise.fp32>, %arg4: tensor<784x512x!eltwise.fp32>, %arg5: tensor<512x512x!eltwise.fp32>, %arg6: tensor<512x10x!eltwise.fp32>) -> tensor<1x10x!eltwise.fp32> {
    %c512 = "tile.affine_const"() {value = 512 : i64} : () -> index
    %cst = "eltwise.sconst"() {value = 0.000000e+00 : f32} : () -> !fp32
    %c10 = "tile.affine_const"() {value = 10 : i64} : () -> index
    %c0 = "tile.affine_const"() {value = 0 : i64} : () -> index
    %c1 = "tile.affine_const"() {value = 1 : i64} : () -> index
    %0 = "tile.domain"() ( {
    ^bb0(%arg7: index, %arg8: index, %arg9: index):	// no predecessors
      %15 = "tile.src_idx_map"(%arg3, %arg8, %arg7) : (tensor<1x784x!eltwise.fp32>, index, index) -> !tile.imap
      %16 = "tile.src_idx_map"(%arg4, %arg7, %arg9) : (tensor<784x512x!eltwise.fp32>, index, index) -> !tile.imap
      %17 = "tile.sink_idx_map"(%arg8, %arg9) : (index, index) -> !tile.imap
      %18 = "tile.size_map"(%c1, %c512) : (index, index) -> !tile.smap
      "tile.+(x*y)"(%18, %15, %16, %17) : (!tile.smap, !tile.imap, !tile.imap, !tile.imap) -> ()
    }) {idx_names = ["x0", "x1", "x2"]} : () -> tensor<1x512x!eltwise.fp32>
    %1 = "eltwise.add"(%0, %arg2) {type = !eltwise.fp32} : (tensor<1x512x!eltwise.fp32>, tensor<512x!eltwise.fp32>) -> tensor<1x512x!eltwise.fp32>
    %2 = "eltwise.cmp_lt"(%1, %cst) {type = !eltwise.fp32} : (tensor<1x512x!eltwise.fp32>, !fp32) -> tensor<1x512x!eltwise.bool>
    %3 = "eltwise.select"(%2, %cst, %1) {type = !eltwise.fp32} : (tensor<1x512x!eltwise.bool>, !fp32, tensor<1x512x!eltwise.fp32>) -> tensor<1x512x!eltwise.fp32>
    %4 = "tile.domain"() ( {
    ^bb0(%arg7: index, %arg8: index, %arg9: index):	// no predecessors
      %15 = "tile.src_idx_map"(%3, %arg8, %arg7) : (tensor<1x512x!eltwise.fp32>, index, index) -> !tile.imap
      %16 = "tile.src_idx_map"(%arg5, %arg7, %arg9) : (tensor<512x512x!eltwise.fp32>, index, index) -> !tile.imap
      %17 = "tile.sink_idx_map"(%arg8, %arg9) : (index, index) -> !tile.imap
      %18 = "tile.size_map"(%c1, %c512) : (index, index) -> !tile.smap
      "tile.+(x*y)"(%18, %15, %16, %17) : (!tile.smap, !tile.imap, !tile.imap, !tile.imap) -> ()
    }) {idx_names = ["x0", "x1", "x2"]} : () -> tensor<1x512x!eltwise.fp32>
    %5 = "eltwise.add"(%4, %arg1) {type = !eltwise.fp32} : (tensor<1x512x!eltwise.fp32>, tensor<512x!eltwise.fp32>) -> tensor<1x512x!eltwise.fp32>
    %6 = "eltwise.cmp_lt"(%5, %cst) {type = !eltwise.fp32} : (tensor<1x512x!eltwise.fp32>, !fp32) -> tensor<1x512x!eltwise.bool>
    %7 = "eltwise.select"(%6, %cst, %5) {type = !eltwise.fp32} : (tensor<1x512x!eltwise.bool>, !fp32, tensor<1x512x!eltwise.fp32>) -> tensor<1x512x!eltwise.fp32>
    %8 = "tile.domain"() ( {
    ^bb0(%arg7: index, %arg8: index, %arg9: index):	// no predecessors
      %15 = "tile.src_idx_map"(%7, %arg8, %arg7) : (tensor<1x512x!eltwise.fp32>, index, index) -> !tile.imap
      %16 = "tile.src_idx_map"(%arg6, %arg7, %arg9) : (tensor<512x10x!eltwise.fp32>, index, index) -> !tile.imap
      %17 = "tile.sink_idx_map"(%arg8, %arg9) : (index, index) -> !tile.imap
      %18 = "tile.size_map"(%c1, %c10) : (index, index) -> !tile.smap
      "tile.+(x*y)"(%18, %15, %16, %17) : (!tile.smap, !tile.imap, !tile.imap, !tile.imap) -> ()
    }) {idx_names = ["x0", "x1", "x2"]} : () -> tensor<1x10x!eltwise.fp32>
    %9 = "eltwise.add"(%8, %arg0) {type = !eltwise.fp32} : (tensor<1x10x!eltwise.fp32>, tensor<10x!eltwise.fp32>) -> tensor<1x10x!eltwise.fp32>
    %10 = "tile.domain"() ( {
    ^bb0(%arg7: index, %arg8: index):	// no predecessors
      %15 = "tile.src_idx_map"(%9, %arg8, %arg7) : (tensor<1x10x!eltwise.fp32>, index, index) -> !tile.imap
      %16 = "tile.sink_idx_map"(%arg8, %c0) : (index, index) -> !tile.imap
      %17 = "tile.size_map"(%c1, %c1) : (index, index) -> !tile.smap
      "tile.>(x)"(%17, %15, %16) : (!tile.smap, !tile.imap, !tile.imap) -> ()
    }) {idx_names = ["x0", "x1"]} : () -> tensor<1x1x!eltwise.fp32>
    %11 = "eltwise.sub"(%9, %10) {type = !eltwise.fp32} : (tensor<1x10x!eltwise.fp32>, tensor<1x1x!eltwise.fp32>) -> tensor<1x10x!eltwise.fp32>
    %12 = "eltwise.exp"(%11) {type = !eltwise.fp32} : (tensor<1x10x!eltwise.fp32>) -> tensor<1x10x!eltwise.fp32>
    %13 = "tile.domain"() ( {
    ^bb0(%arg7: index, %arg8: index):	// no predecessors
      %15 = "tile.src_idx_map"(%12, %arg8, %arg7) : (tensor<1x10x!eltwise.fp32>, index, index) -> !tile.imap
      %16 = "tile.sink_idx_map"(%arg8, %c0) : (index, index) -> !tile.imap
      %17 = "tile.size_map"(%c1, %c1) : (index, index) -> !tile.smap
      "tile.+(x)"(%17, %15, %16) : (!tile.smap, !tile.imap, !tile.imap) -> ()
    }) {idx_names = ["x0", "x1"]} : () -> tensor<1x1x!eltwise.fp32>
    %14 = "eltwise.div"(%12, %13) {type = !eltwise.fp32} : (tensor<1x10x!eltwise.fp32>, tensor<1x1x!eltwise.fp32>) -> tensor<1x10x!eltwise.fp32>
    return %14 : tensor<1x10x!eltwise.fp32>
  }
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
        self.assertEqual(str(F.shape), 'tensor<1x12100x!eltwise.fp32>')
        K3 = Tensor(LogicalShape(plaidml.DType.FLOAT32, [12100, 128]))
        B3 = Tensor(LogicalShape(plaidml.DType.FLOAT32, [128]))
        D1 = relu(dot(F, K3) + B3)
        # model.add(Dense(num_classes, activation='softmax'))
        K4 = Tensor(LogicalShape(plaidml.DType.FLOAT32, [128, 100]))
        B4 = Tensor(LogicalShape(plaidml.DType.FLOAT32, [100]))
        D2 = softmax(dot(D1, K4) + B4)
        program = Program('mnist_cnn', [D2])

    def test_arg_max(self):
        I = Tensor(LogicalShape(plaidml.DType.FLOAT32, [1, 10, 10]))
        O = arg_max(I)
        self.assertEqual(str(O.shape), 'tensor<1x10x!eltwise.u32>')
        program = Program('arg_max', [O])
        self.assertMultiLineEqual(
            str(program), '''

!i32 = type tensor<!eltwise.i32>
!fp32 = type tensor<!eltwise.fp32>
module {
  func @arg_max(%arg0: tensor<1x10x10x!eltwise.fp32>, %arg1: !fp32) -> tensor<1x10x!eltwise.u32> {
    %c1 = "tile.affine_const"() {value = 1 : i64} : () -> index
    %c0 = "eltwise.sconst"() {value = 0 : i64} : () -> !i32
    %c10 = "tile.affine_const"() {value = 10 : i64} : () -> index
    %0 = "tile.domain"() ( {
    ^bb0(%arg2: index, %arg3: index, %arg4: index):	// no predecessors
      %5 = "tile.src_idx_map"(%arg0, %arg4, %arg3, %arg2) : (tensor<1x10x10x!eltwise.fp32>, index, index, index) -> !tile.imap
      %6 = "tile.sink_idx_map"(%arg4, %arg2) : (index, index) -> !tile.imap
      %7 = "tile.size_map"(%c1, %c10) : (index, index) -> !tile.smap
      "tile.>(x)"(%7, %5, %6) : (!tile.smap, !tile.imap, !tile.imap) -> ()
    }) {idx_names = ["x0", "x1", "x2"]} : () -> tensor<1x10x!eltwise.fp32>
    %1 = "tile.domain"() ( {
    ^bb0(%arg2: index):	// no predecessors
      %5 = "tile.src_idx_map"(%arg1) : (!fp32) -> !tile.imap
      %6 = "tile.sink_idx_map"(%arg2) : (index) -> !tile.imap
      %7 = "tile.size_map"(%c10) : (index) -> !tile.smap
      "tile.=(x)"(%7, %5, %6) : (!tile.smap, !tile.imap, !tile.imap) -> ()
    }) {idx_names = ["x0"]} : () -> tensor<10x!eltwise.fp32>
    %2 = "tile.index"(%1, %c0) : (tensor<10x!eltwise.fp32>, !i32) -> tensor<10x!eltwise.i32>
    %3 = "tile.domain"() ( {
    ^bb0(%arg2: index, %arg3: index, %arg4: index):	// no predecessors
      %5 = "tile.src_idx_map"(%arg0, %arg4, %arg3, %arg2) : (tensor<1x10x10x!eltwise.fp32>, index, index, index) -> !tile.imap
      %6 = "tile.src_idx_map"(%0, %arg4, %arg2) : (tensor<1x10x!eltwise.fp32>, index, index) -> !tile.imap
      %7 = "tile.src_idx_map"(%2, %arg3) : (tensor<10x!eltwise.i32>, index) -> !tile.imap
      %8 = "tile.sink_idx_map"(%arg4, %arg2) : (index, index) -> !tile.imap
      %9 = "tile.size_map"(%c1, %c10) : (index, index) -> !tile.smap
      "tile.>(x==y?z)"(%9, %5, %6, %7, %8) : (!tile.smap, !tile.imap, !tile.imap, !tile.imap, !tile.imap) -> ()
    }) {idx_names = ["x0", "x1", "x2"]} : () -> tensor<1x10x!eltwise.fp32>
    %4 = "eltwise.as_uint"(%3) : (tensor<1x10x!eltwise.fp32>) -> tensor<1x10x!eltwise.u32>
    return %4 : tensor<1x10x!eltwise.u32>
  }
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
            str(program), '''

!fp32 = type tensor<!eltwise.fp32>
module {
  func @global_min(%arg0: tensor<10x10x10x!eltwise.fp32> {tile.name = "I"}) -> !fp32 {
    %0 = "eltwise.neg"(%arg0) {type = !eltwise.fp32} : (tensor<10x10x10x!eltwise.fp32>) -> tensor<10x10x10x!eltwise.fp32>
    %1 = "tile.domain"() ( {
    ^bb0(%arg1: index, %arg2: index, %arg3: index):	// no predecessors
      %3 = "tile.src_idx_map"(%0, %arg3, %arg2, %arg1) : (tensor<10x10x10x!eltwise.fp32>, index, index, index) -> !tile.imap
      %4 = "tile.sink_idx_map"() : () -> !tile.imap
      %5 = "tile.size_map"() : () -> !tile.smap
      "tile.>(x)"(%5, %3, %4) : (!tile.smap, !tile.imap, !tile.imap) -> ()
    }) {idx_names = ["x0", "x1", "x2"]} : () -> !fp32
    %2 = "eltwise.neg"(%1) {type = !eltwise.fp32} : (!fp32) -> !fp32
    return %2 : !fp32
  }
}
''')

    def test_cum_sum(self):
        I = Tensor(LogicalShape(plaidml.DType.FLOAT32, [10]), name='I')
        N = TensorDim()
        i, k = TensorIndexes(2)
        I.bind_dims(N)
        O = TensorOutput(N)
        O[i] += I[k]
        O.add_constraint(i - k < N)
        program = Program('cum_sum', [O])
        self.assertMultiLineEqual(
            str(program), '''

module {
  func @cum_sum(%arg0: tensor<10x!eltwise.fp32> {tile.name = "I"}) -> tensor<10x!eltwise.fp32> {
    %c10 = "tile.affine_const"() {value = 10 : i64} : () -> index
    %0 = "tile.domain"() ( {
    ^bb0(%arg1: index, %arg2: index):	// no predecessors
      %1 = "tile.src_idx_map"(%arg0, %arg1) : (tensor<10x!eltwise.fp32>, index) -> !tile.imap
      %2 = "tile.sink_idx_map"(%arg2) : (index) -> !tile.imap
      %3 = "tile.size_map"(%c10) : (index) -> !tile.smap
      %4 = "tile.affine_sub"(%arg2, %arg1) : (index, index) -> index
      "tile.constraint"(%4, %c10) ( {
        "tile.+(x)"(%3, %1, %2) : (!tile.smap, !tile.imap, !tile.imap) -> ()
      }) : (index, index) -> ()
    }) {idx_names = ["x0", "x1"]} : () -> tensor<10x!eltwise.fp32>
    return %0 : tensor<10x!eltwise.fp32>
  }
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
            str(program), '''

!fp32 = type tensor<!eltwise.fp32>
module {
  func @unique_names(%arg0: !fp32 {tile.name = "C"}, %arg1: !fp32 {tile.name = "C"}, %arg2: !fp32 {tile.name = "B"}, %arg3: !fp32 {tile.name = "A"}) -> !fp32 {
    %0 = "eltwise.add"(%arg3, %arg2) {type = !eltwise.fp32} : (!fp32, !fp32) -> !fp32
    %1 = "eltwise.add"(%0, %arg1) {type = !eltwise.fp32} : (!fp32, !fp32) -> !fp32
    %2 = "eltwise.add"(%1, %arg0) {type = !eltwise.fp32} : (!fp32, !fp32) -> !fp32
    return %2 : !fp32
  }
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
            str(program), '''

!fp32 = type tensor<!eltwise.fp32>
module {
  func @lars_momentum_4d(%arg0: tensor<4x7x3x9x!eltwise.fp32>, %arg1: tensor<4x7x3x9x!eltwise.fp32>, %arg2: !fp32, %arg3: tensor<4x7x3x9x!eltwise.fp32>) -> (tensor<4x7x3x9x!eltwise.fp32>, tensor<4x7x3x9x!eltwise.fp32>) {
    %cst = "eltwise.sconst"() {value = 4.8828125E-4 : f32} : () -> !fp32
    %cst_0 = "eltwise.sconst"() {value = 9.765625E-4 : f32} : () -> !fp32
    %cst_1 = "eltwise.sconst"() {value = 1.250000e-01 : f32} : () -> !fp32
    %0 = "eltwise.mul"(%arg0, %cst) {type = !eltwise.fp32} : (tensor<4x7x3x9x!eltwise.fp32>, !fp32) -> tensor<4x7x3x9x!eltwise.fp32>
    %1 = "eltwise.add"(%arg1, %0) {type = !eltwise.fp32} : (tensor<4x7x3x9x!eltwise.fp32>, tensor<4x7x3x9x!eltwise.fp32>) -> tensor<4x7x3x9x!eltwise.fp32>
    %2 = "eltwise.mul"(%arg0, %arg0) {type = !eltwise.fp32} : (tensor<4x7x3x9x!eltwise.fp32>, tensor<4x7x3x9x!eltwise.fp32>) -> tensor<4x7x3x9x!eltwise.fp32>
    %3 = "tile.domain"() ( {
    ^bb0(%arg4: index, %arg5: index, %arg6: index, %arg7: index):	// no predecessors
      %17 = "tile.src_idx_map"(%2, %arg7, %arg6, %arg5, %arg4) : (tensor<4x7x3x9x!eltwise.fp32>, index, index, index, index) -> !tile.imap
      %18 = "tile.sink_idx_map"() : () -> !tile.imap
      %19 = "tile.size_map"() : () -> !tile.smap
      "tile.+(x)"(%19, %17, %18) : (!tile.smap, !tile.imap, !tile.imap) -> ()
    }) {idx_names = ["x0", "x1", "x2", "x3"]} : () -> !fp32
    %4 = "eltwise.sqrt"(%3) {type = !eltwise.fp32} : (!fp32) -> !fp32
    %5 = "eltwise.mul"(%4, %cst) {type = !eltwise.fp32} : (!fp32, !fp32) -> !fp32
    %6 = "eltwise.mul"(%arg1, %arg1) {type = !eltwise.fp32} : (tensor<4x7x3x9x!eltwise.fp32>, tensor<4x7x3x9x!eltwise.fp32>) -> tensor<4x7x3x9x!eltwise.fp32>
    %7 = "tile.domain"() ( {
    ^bb0(%arg4: index, %arg5: index, %arg6: index, %arg7: index):	// no predecessors
      %17 = "tile.src_idx_map"(%6, %arg7, %arg6, %arg5, %arg4) : (tensor<4x7x3x9x!eltwise.fp32>, index, index, index, index) -> !tile.imap
      %18 = "tile.sink_idx_map"() : () -> !tile.imap
      %19 = "tile.size_map"() : () -> !tile.smap
      "tile.+(x)"(%19, %17, %18) : (!tile.smap, !tile.imap, !tile.imap) -> ()
    }) {idx_names = ["x0", "x1", "x2", "x3"]} : () -> !fp32
    %8 = "eltwise.sqrt"(%7) {type = !eltwise.fp32} : (!fp32) -> !fp32
    %9 = "eltwise.add"(%8, %5) {type = !eltwise.fp32} : (!fp32, !fp32) -> !fp32
    %10 = "eltwise.mul"(%arg2, %cst_0) {type = !eltwise.fp32} : (!fp32, !fp32) -> !fp32
    %11 = "eltwise.mul"(%10, %4) {type = !eltwise.fp32} : (!fp32, !fp32) -> !fp32
    %12 = "eltwise.div"(%11, %9) {type = !eltwise.fp32} : (!fp32, !fp32) -> !fp32
    %13 = "eltwise.mul"(%12, %1) {type = !eltwise.fp32} : (!fp32, tensor<4x7x3x9x!eltwise.fp32>) -> tensor<4x7x3x9x!eltwise.fp32>
    %14 = "eltwise.mul"(%arg3, %cst_1) {type = !eltwise.fp32} : (tensor<4x7x3x9x!eltwise.fp32>, !fp32) -> tensor<4x7x3x9x!eltwise.fp32>
    %15 = "eltwise.add"(%14, %13) {type = !eltwise.fp32} : (tensor<4x7x3x9x!eltwise.fp32>, tensor<4x7x3x9x!eltwise.fp32>) -> tensor<4x7x3x9x!eltwise.fp32>
    %16 = "eltwise.sub"(%arg0, %15) {type = !eltwise.fp32} : (tensor<4x7x3x9x!eltwise.fp32>, tensor<4x7x3x9x!eltwise.fp32>) -> tensor<4x7x3x9x!eltwise.fp32>
    return %16, %15 : tensor<4x7x3x9x!eltwise.fp32>, tensor<4x7x3x9x!eltwise.fp32>
  }
}
''')

    @unittest.skip('TODO: no_defract')
    def test_repeat_elts(self):
        I = Tensor(LogicalShape(plaidml.DType.FLOAT32, [10, 10, 10]))
        N0, N1, N2 = TensorDims(3)
        n0, n1, n2, k = TensorIndexes(4)
        I.bind_dims(N0, N1, N2)
        O = TensorOutput(N0, 3 * N1, N2)
        O[n0, 3 * n1 + k, n2] = I[n0, n1, n2]
        O.add_constraint(k < 3)
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
            str(program), '''

module {
  func @use_default(%arg0: tensor<1x10x10x!eltwise.fp32>, %arg1: tensor<1x7x10x10x!eltwise.fp32>) -> tensor<1x7x10x10x!eltwise.fp32> {
    %c3 = "tile.affine_const"() {value = 3 : i64} : () -> index
    %c10 = "tile.affine_const"() {value = 10 : i64} : () -> index
    %c7 = "tile.affine_const"() {value = 7 : i64} : () -> index
    %c1 = "tile.affine_const"() {value = 1 : i64} : () -> index
    %0 = "tile.domain"() ( {
    ^bb0(%arg2: index, %arg3: index, %arg4: index):	// no predecessors
      %1 = "tile.src_idx_map"(%arg0, %arg4, %arg3, %arg2) : (tensor<1x10x10x!eltwise.fp32>, index, index, index) -> !tile.imap
      %2 = "tile.sink_idx_map"(%arg4, %c3, %arg3, %arg2) : (index, index, index, index) -> !tile.imap
      %3 = "tile.size_map"(%c1, %c7, %c10, %c10) : (index, index, index, index) -> !tile.smap
      "tile.=(x)"(%3, %1, %2, %arg1) : (!tile.smap, !tile.imap, !tile.imap, tensor<1x7x10x10x!eltwise.fp32>) -> ()
    }) {idx_names = ["x0", "x1", "x2"]} : () -> tensor<1x7x10x10x!eltwise.fp32>
    return %0 : tensor<1x7x10x10x!eltwise.fp32>
  }
}
''')

    def test_defract(self):
        I = Tensor(LogicalShape(plaidml.DType.FLOAT32, [3]), name='I')
        K = Tensor(LogicalShape(plaidml.DType.FLOAT32, [3]), name='K')
        i, j = TensorIndexes(2)
        O = TensorOutput(5)
        O[i] += (I[(i - j + 1) // 2] * K[j])
        program = Program('defract_test', [O])
        self.assertMultiLineEqual(
            str(program), '''

module {
  func @defract_test(%arg0: tensor<3x!eltwise.fp32> {tile.name = "I"}, %arg1: tensor<3x!eltwise.fp32> {tile.name = "K"}) -> tensor<5x!eltwise.fp32> {
    %c1 = "tile.affine_const"() {value = 1 : i64} : () -> index
    %c2 = "tile.affine_const"() {value = 2 : i64} : () -> index
    %c5 = "tile.affine_const"() {value = 5 : i64} : () -> index
    %0 = "tile.domain"() ( {
    ^bb0(%arg2: index, %arg3: index):	// no predecessors
      %1 = "tile.affine_sub"(%arg3, %arg2) : (index, index) -> index
      %2 = "tile.affine_add"(%1, %c1) : (index, index) -> index
      %3 = "tile.affine_div"(%2, %c2) : (index, index) -> index
      %4 = "tile.src_idx_map"(%arg0, %3) : (tensor<3x!eltwise.fp32>, index) -> !tile.imap
      %5 = "tile.src_idx_map"(%arg1, %arg2) : (tensor<3x!eltwise.fp32>, index) -> !tile.imap
      %6 = "tile.sink_idx_map"(%arg3) : (index) -> !tile.imap
      %7 = "tile.size_map"(%c5) : (index) -> !tile.smap
      "tile.+(x*y)"(%7, %4, %5, %6) : (!tile.smap, !tile.imap, !tile.imap, !tile.imap) -> ()
    }) {idx_names = ["x0", "x1"]} : () -> tensor<5x!eltwise.fp32>
    return %0 : tensor<5x!eltwise.fp32>
  }
}
''')
        outputs = plaidml_exec.run(program, [(I, np.array([1, 2, 3])), (K, np.array([1, 2, 3]))])
        self.assertEqual(outputs[0].tolist(), [2, 5, 4, 9, 6])

    def test_defract_short(self):
        I = Tensor(LogicalShape(plaidml.DType.FLOAT32, [3]), name='I')
        i, j = TensorIndexes(2)
        O = TensorOutput(6)
        O[i] += (I[(i - 1) // 2])
        program = Program('defract_short_test', [O])
        self.assertMultiLineEqual(
            str(program), '''

module {
  func @defract_short_test(%arg0: tensor<3x!eltwise.fp32> {tile.name = "I"}) -> tensor<6x!eltwise.fp32> {
    %c1 = "tile.affine_const"() {value = 1 : i64} : () -> index
    %c2 = "tile.affine_const"() {value = 2 : i64} : () -> index
    %c6 = "tile.affine_const"() {value = 6 : i64} : () -> index
    %0 = "tile.domain"() ( {
    ^bb0(%arg1: index):	// no predecessors
      %1 = "tile.affine_sub"(%arg1, %c1) : (index, index) -> index
      %2 = "tile.affine_div"(%1, %c2) : (index, index) -> index
      %3 = "tile.src_idx_map"(%arg0, %2) : (tensor<3x!eltwise.fp32>, index) -> !tile.imap
      %4 = "tile.sink_idx_map"(%arg1) : (index) -> !tile.imap
      %5 = "tile.size_map"(%c6) : (index) -> !tile.smap
      "tile.+(x)"(%5, %3, %4) : (!tile.smap, !tile.imap, !tile.imap) -> ()
    }) {idx_names = ["x0"]} : () -> tensor<6x!eltwise.fp32>
    return %0 : tensor<6x!eltwise.fp32>
  }
}
''')
        outputs = plaidml_exec.run(program, [(I, np.array([1, 2, 3]))])
        self.assertEqual(outputs[0].tolist(), [0, 1, 0, 2, 0, 3])

    def test_defract_long(self):
        shape = [1, 3, 3, 1]
        I = Tensor(LogicalShape(plaidml.DType.FLOAT32, shape), name='I')
        K = Tensor(LogicalShape(plaidml.DType.FLOAT32, shape), name='K')
        n, x0, x1, c0, c1, co, ci, k0, k1 = TensorIndexes(9)
        O = TensorOutput(1, 5, 5, 1)
        O[n, x0, x1, co] += (I[n, (x0 + k0 - 1) // 2,
                               (x1 + k1 - 1) // 2, ci] * K[2 - k0, 2 - k1, co, ci])
        program = Program('defract_long', [O])
        self.assertMultiLineEqual(
            str(program), '''

module {
  func @defract_long(%arg0: tensor<1x3x3x1x!eltwise.fp32> {tile.name = "I"}, %arg1: tensor<1x3x3x1x!eltwise.fp32> {tile.name = "K"}) -> tensor<1x5x5x1x!eltwise.fp32> {
    %c2 = "tile.affine_const"() {value = 2 : i64} : () -> index
    %c5 = "tile.affine_const"() {value = 5 : i64} : () -> index
    %c1 = "tile.affine_const"() {value = 1 : i64} : () -> index
    %0 = "tile.domain"() ( {
    ^bb0(%arg2: index, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index):	// no predecessors
      %1 = "tile.affine_add"(%arg4, %arg3) : (index, index) -> index
      %2 = "tile.affine_sub"(%1, %c1) : (index, index) -> index
      %3 = "tile.affine_div"(%2, %c2) : (index, index) -> index
      %4 = "tile.affine_add"(%arg6, %arg5) : (index, index) -> index
      %5 = "tile.affine_sub"(%4, %c1) : (index, index) -> index
      %6 = "tile.affine_div"(%5, %c2) : (index, index) -> index
      %7 = "tile.src_idx_map"(%arg0, %arg7, %6, %3, %arg2) : (tensor<1x3x3x1x!eltwise.fp32>, index, index, index, index) -> !tile.imap
      %8 = "tile.affine_sub"(%c2, %arg3) : (index, index) -> index
      %9 = "tile.affine_sub"(%c2, %arg5) : (index, index) -> index
      %10 = "tile.src_idx_map"(%arg1, %9, %8, %arg8, %arg2) : (tensor<1x3x3x1x!eltwise.fp32>, index, index, index, index) -> !tile.imap
      %11 = "tile.sink_idx_map"(%arg7, %arg6, %arg4, %arg8) : (index, index, index, index) -> !tile.imap
      %12 = "tile.size_map"(%c1, %c5, %c5, %c1) : (index, index, index, index) -> !tile.smap
      "tile.+(x*y)"(%12, %7, %10, %11) : (!tile.smap, !tile.imap, !tile.imap, !tile.imap) -> ()
    }) {idx_names = ["x0", "x1", "x2", "x3", "x4", "x5", "x6"]} : () -> tensor<1x5x5x1x!eltwise.fp32>
    return %0 : tensor<1x5x5x1x!eltwise.fp32>
  }
}
''')
        outputs = plaidml_exec.run(program, [
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
        ])
        np.testing.assert_array_equal(
            outputs[0],
            np.array([[
                [[0], [0], [0], [0], [0]],
                [[0], [4], [12], [6], [24]],
                [[0], [0], [0], [0], [0]],
                [[6], [-3], [-6], [-3], [-12]],
                [[0], [0], [0], [0], [0]],
            ]]))

    def test_funky_names(self):
        '''Exercises fix for plaidml bug #241

        Now that we emit keras layer names as 'pid' attribute values, in order
        to help link tile code back to its origin while debugging, we must
        reformat those names as valid tile identifiers. If we are doing that,
        this test will pass, otherwise we'll get a syntax error.'''

        I = Tensor(LogicalShape(plaidml.DType.FLOAT32, [3]), name='I')
        K = Tensor(LogicalShape(plaidml.DType.FLOAT32, [3]), name='K')
        i, j = TensorIndexes(2)
        O = TensorOutput(5)
        O[i] += (I[(i - j + 1) // 2] * K[j])
        program = Program('this-is-not an identifier', [O])
        self.assertMultiLineEqual(
            str(program), '''

module {
  func @this-is-not an identifier(%arg0: tensor<3x!eltwise.fp32> {tile.name = "I"}, %arg1: tensor<3x!eltwise.fp32> {tile.name = "K"}) -> tensor<5x!eltwise.fp32> {
    %c1 = "tile.affine_const"() {value = 1 : i64} : () -> index
    %c2 = "tile.affine_const"() {value = 2 : i64} : () -> index
    %c5 = "tile.affine_const"() {value = 5 : i64} : () -> index
    %0 = "tile.domain"() ( {
    ^bb0(%arg2: index, %arg3: index):	// no predecessors
      %1 = "tile.affine_sub"(%arg3, %arg2) : (index, index) -> index
      %2 = "tile.affine_add"(%1, %c1) : (index, index) -> index
      %3 = "tile.affine_div"(%2, %c2) : (index, index) -> index
      %4 = "tile.src_idx_map"(%arg0, %3) : (tensor<3x!eltwise.fp32>, index) -> !tile.imap
      %5 = "tile.src_idx_map"(%arg1, %arg2) : (tensor<3x!eltwise.fp32>, index) -> !tile.imap
      %6 = "tile.sink_idx_map"(%arg3) : (index) -> !tile.imap
      %7 = "tile.size_map"(%c5) : (index) -> !tile.smap
      "tile.+(x*y)"(%7, %4, %5, %6) : (!tile.smap, !tile.imap, !tile.imap, !tile.imap) -> ()
    }) {idx_names = ["x0", "x1"]} : () -> tensor<5x!eltwise.fp32>
    return %0 : tensor<5x!eltwise.fp32>
  }
}
''')
        outputs = plaidml_exec.run(program, [(I, np.array([1, 2, 3])), (K, np.array([1, 2, 3]))])
        self.assertEqual(outputs[0].tolist(), [2, 5, 4, 9, 6])

    def test_identity(self):
        I = Tensor(LogicalShape(plaidml.DType.FLOAT32, [3]), name='I')
        program = Program('identity', [I])
        self.assertMultiLineEqual(
            str(program), '''

module {
  func @identity(%arg0: tensor<3x!eltwise.fp32> {tile.name = "I"}) -> tensor<3x!eltwise.fp32> {
    %0 = "eltwise.ident"(%arg0) {type = !eltwise.fp32} : (tensor<3x!eltwise.fp32>) -> tensor<3x!eltwise.fp32>
    return %0 : tensor<3x!eltwise.fp32>
  }
}
''')
        outputs = plaidml_exec.run(program, [(I, np.array([(1, 2, 3)]))])
        self.assertEqual(outputs[0].tolist(), [1, 2, 3])

    @unittest.skip('TODO: exception needs to be thrown')
    def test_assignment_exceptions(self):
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
        outputs = plaidml_exec.run(program, [(A, np.array([[1], [2], [3], [4], [5]])),
                                             (B, np.array([1, 2, 3, 4, 5]))])
        self.assertEqual(outputs[0].tolist(),
                         [[1., 2., 3., 4., 5.], [2., 4., 6., 8., 10.], [3., 6., 9., 12., 15.],
                          [4., 8., 12., 16., 20.], [5., 10., 15., 20., 25.]])

        O = TensorOutput(L, N)
        O[i, j] = B[i, k] * A[k, j]
        with self.assertRaises(plaidml.Error) as cm:
            program = Program('assignment_exception', [O])
        self.assertTrue("illegal assignment aggregation" in str(cm.exception))

    def test_two_outputs(self):
        I = Tensor(LogicalShape(plaidml.DType.FLOAT32, [3]), name='I')
        program1 = Program('two_outputs', [I, I])
        self.assertMultiLineEqual(
            str(program1), '''

module {
  func @two_outputs(%arg0: tensor<3x!eltwise.fp32> {tile.name = "I"}) -> (tensor<3x!eltwise.fp32>, tensor<3x!eltwise.fp32>) {
    %0 = "eltwise.ident"(%arg0) {type = !eltwise.fp32} : (tensor<3x!eltwise.fp32>) -> tensor<3x!eltwise.fp32>
    %1 = "eltwise.ident"(%arg0) {type = !eltwise.fp32} : (tensor<3x!eltwise.fp32>) -> tensor<3x!eltwise.fp32>
    return %0, %1 : tensor<3x!eltwise.fp32>, tensor<3x!eltwise.fp32>
  }
}
''')
        outputs = plaidml_exec.run(program1, [(I, np.array([(1, 2, 3)]))])
        self.assertEqual(outputs[0].tolist(), [1, 2, 3])
        self.assertEqual(outputs[1].tolist(), [1, 2, 3])

        O1 = I
        O2 = I
        program2 = Program('two_outputs', [O1, O2])
        self.assertMultiLineEqual(str(program1), str(program2))

        outputs = plaidml_exec.run(program2, [(I, np.array([(1, 2, 3)]))])
        self.assertEqual(outputs[0].tolist(), [1, 2, 3])
        self.assertEqual(outputs[1].tolist(), [1, 2, 3])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', type=int, default=0)
    args, remainder = parser.parse_known_args()
    unittest.main(argv=sys.argv[:1] + remainder, verbosity=args.verbose + 1)
