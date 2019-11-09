# Copyright 2019 Intel Corporation.

import argparse
import functools
import os
import sys
import unittest

import plaidml2 as plaidml
import plaidml2.exec as plaidml_exec
from plaidml2.edsl import *


def USE_MLIR():
    return os.getenv('PLAIDML_MLIR') == '1'


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
        if USE_MLIR():
            self.assertMultiLineEqual(
                str(program), '''

!float = type tensor<!eltwise.float>
module {
  func @mnist_mlp(%arg0: tensor<10x!eltwise.fp32>, %arg1: tensor<512x!eltwise.fp32>, %arg2: tensor<512x!eltwise.fp32>, %arg3: tensor<1x784x!eltwise.fp32>, %arg4: tensor<784x512x!eltwise.fp32>, %arg5: tensor<512x512x!eltwise.fp32>, %arg6: tensor<512x10x!eltwise.fp32>) -> tensor<1x10x!eltwise.fp32> {
    %c512 = "tile.affine_const"() {value = 512 : i64} : () -> !eltwise.int
    %cst = "eltwise.sconst"() {value = 0.000000e+00 : f64} : () -> !float
    %c10 = "tile.affine_const"() {value = 10 : i64} : () -> !eltwise.int
    %c0 = "tile.affine_const"() {value = 0 : i64} : () -> !eltwise.int
    %c1 = "tile.affine_const"() {value = 1 : i64} : () -> !eltwise.int
    %0 = "tile.domain"() ( {
    ^bb0(%arg7: !eltwise.int, %arg8: !eltwise.int, %arg9: !eltwise.int):	// no predecessors
      %15 = "tile.src_idx_map"(%arg3, %arg8, %arg7) : (tensor<1x784x!eltwise.fp32>, !eltwise.int, !eltwise.int) -> !tile.imap
      %16 = "tile.src_idx_map"(%arg4, %arg7, %arg9) : (tensor<784x512x!eltwise.fp32>, !eltwise.int, !eltwise.int) -> !tile.imap
      %17 = "tile.sink_idx_map"(%arg8, %arg9) : (!eltwise.int, !eltwise.int) -> !tile.imap
      %18 = "tile.size_map"(%c1, %c512) : (!eltwise.int, !eltwise.int) -> !tile.smap
      "tile.+(x*y)"(%18, %15, %16, %17) : (!tile.smap, !tile.imap, !tile.imap, !tile.imap) -> ()
    }) {idx_names = ["x0", "x1", "x2"]} : () -> tensor<1x512x!eltwise.fp32>
    %1 = "eltwise.add"(%0, %arg2) {type = !eltwise.fp32} : (tensor<1x512x!eltwise.fp32>, tensor<512x!eltwise.fp32>) -> tensor<1x512x!eltwise.fp32>
    %2 = "eltwise.cmp_lt"(%1, %cst) {type = !eltwise.fp32} : (tensor<1x512x!eltwise.fp32>, !float) -> tensor<1x512x!eltwise.bool>
    %3 = "eltwise.select"(%2, %cst, %1) {type = !eltwise.fp32} : (tensor<1x512x!eltwise.bool>, !float, tensor<1x512x!eltwise.fp32>) -> tensor<1x512x!eltwise.fp32>
    %4 = "tile.domain"() ( {
    ^bb0(%arg7: !eltwise.int, %arg8: !eltwise.int, %arg9: !eltwise.int):	// no predecessors
      %15 = "tile.src_idx_map"(%3, %arg8, %arg7) : (tensor<1x512x!eltwise.fp32>, !eltwise.int, !eltwise.int) -> !tile.imap
      %16 = "tile.src_idx_map"(%arg5, %arg7, %arg9) : (tensor<512x512x!eltwise.fp32>, !eltwise.int, !eltwise.int) -> !tile.imap
      %17 = "tile.sink_idx_map"(%arg8, %arg9) : (!eltwise.int, !eltwise.int) -> !tile.imap
      %18 = "tile.size_map"(%c1, %c512) : (!eltwise.int, !eltwise.int) -> !tile.smap
      "tile.+(x*y)"(%18, %15, %16, %17) : (!tile.smap, !tile.imap, !tile.imap, !tile.imap) -> ()
    }) {idx_names = ["x0", "x1", "x2"]} : () -> tensor<1x512x!eltwise.fp32>
    %5 = "eltwise.add"(%4, %arg1) {type = !eltwise.fp32} : (tensor<1x512x!eltwise.fp32>, tensor<512x!eltwise.fp32>) -> tensor<1x512x!eltwise.fp32>
    %6 = "eltwise.cmp_lt"(%5, %cst) {type = !eltwise.fp32} : (tensor<1x512x!eltwise.fp32>, !float) -> tensor<1x512x!eltwise.bool>
    %7 = "eltwise.select"(%6, %cst, %5) {type = !eltwise.fp32} : (tensor<1x512x!eltwise.bool>, !float, tensor<1x512x!eltwise.fp32>) -> tensor<1x512x!eltwise.fp32>
    %8 = "tile.domain"() ( {
    ^bb0(%arg7: !eltwise.int, %arg8: !eltwise.int, %arg9: !eltwise.int):	// no predecessors
      %15 = "tile.src_idx_map"(%7, %arg8, %arg7) : (tensor<1x512x!eltwise.fp32>, !eltwise.int, !eltwise.int) -> !tile.imap
      %16 = "tile.src_idx_map"(%arg6, %arg7, %arg9) : (tensor<512x10x!eltwise.fp32>, !eltwise.int, !eltwise.int) -> !tile.imap
      %17 = "tile.sink_idx_map"(%arg8, %arg9) : (!eltwise.int, !eltwise.int) -> !tile.imap
      %18 = "tile.size_map"(%c1, %c10) : (!eltwise.int, !eltwise.int) -> !tile.smap
      "tile.+(x*y)"(%18, %15, %16, %17) : (!tile.smap, !tile.imap, !tile.imap, !tile.imap) -> ()
    }) {idx_names = ["x0", "x1", "x2"]} : () -> tensor<1x10x!eltwise.fp32>
    %9 = "eltwise.add"(%8, %arg0) {type = !eltwise.fp32} : (tensor<1x10x!eltwise.fp32>, tensor<10x!eltwise.fp32>) -> tensor<1x10x!eltwise.fp32>
    %10 = "tile.domain"() ( {
    ^bb0(%arg7: !eltwise.int, %arg8: !eltwise.int):	// no predecessors
      %15 = "tile.src_idx_map"(%9, %arg8, %arg7) : (tensor<1x10x!eltwise.fp32>, !eltwise.int, !eltwise.int) -> !tile.imap
      %16 = "tile.sink_idx_map"(%arg8, %c0) : (!eltwise.int, !eltwise.int) -> !tile.imap
      %17 = "tile.size_map"(%c1, %c1) : (!eltwise.int, !eltwise.int) -> !tile.smap
      "tile.>(x)"(%17, %15, %16) : (!tile.smap, !tile.imap, !tile.imap) -> ()
    }) {idx_names = ["x0", "x1"]} : () -> tensor<1x1x!eltwise.fp32>
    %11 = "eltwise.sub"(%9, %10) {type = !eltwise.fp32} : (tensor<1x10x!eltwise.fp32>, tensor<1x1x!eltwise.fp32>) -> tensor<1x10x!eltwise.fp32>
    %12 = "eltwise.exp"(%11) {type = !eltwise.fp32} : (tensor<1x10x!eltwise.fp32>) -> tensor<1x10x!eltwise.fp32>
    %13 = "tile.domain"() ( {
    ^bb0(%arg7: !eltwise.int, %arg8: !eltwise.int):	// no predecessors
      %15 = "tile.src_idx_map"(%12, %arg8, %arg7) : (tensor<1x10x!eltwise.fp32>, !eltwise.int, !eltwise.int) -> !tile.imap
      %16 = "tile.sink_idx_map"(%arg8, %c0) : (!eltwise.int, !eltwise.int) -> !tile.imap
      %17 = "tile.size_map"(%c1, %c1) : (!eltwise.int, !eltwise.int) -> !tile.smap
      "tile.+(x)"(%17, %15, %16) : (!tile.smap, !tile.imap, !tile.imap) -> ()
    }) {idx_names = ["x0", "x1"]} : () -> tensor<1x1x!eltwise.fp32>
    %14 = "eltwise.div"(%12, %13) {type = !eltwise.fp32} : (tensor<1x10x!eltwise.fp32>, tensor<1x1x!eltwise.fp32>) -> tensor<1x10x!eltwise.fp32>
    return %14 : tensor<1x10x!eltwise.fp32>
  }
}
''')
        else:
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
        if USE_MLIR():
            self.assertEqual(str(F.shape), 'tensor<1x12100x!eltwise.fp32>')
        else:
            self.assertEqual(str(F.shape), 'fp32(1, 12100)')
        K3 = Tensor(LogicalShape(plaidml.DType.FLOAT32, [12100, 128]))
        B3 = Tensor(LogicalShape(plaidml.DType.FLOAT32, [128]))
        D1 = relu(dot(F, K3) + B3)
        # model.add(Dense(num_classes, activation='softmax'))
        K4 = Tensor(LogicalShape(plaidml.DType.FLOAT32, [128, 100]))
        B4 = Tensor(LogicalShape(plaidml.DType.FLOAT32, [100]))
        D2 = softmax(dot(D1, K4) + B4)
        program = Program('mnist_cnn', [D2])
        if not USE_MLIR():
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
        program = Program('arg_max', [O])
        if USE_MLIR():
            self.assertEqual(str(O.shape), 'tensor<1x10x!eltwise.u32>')
            self.assertMultiLineEqual(
                str(program), '''

!fp32 = type tensor<!eltwise.fp32>
module {
  func @arg_max(%arg0: tensor<1x10x10x!eltwise.fp32>, %arg1: !fp32) -> tensor<1x10x!eltwise.u32> {
    %c1 = "tile.affine_const"() {value = 1 : i64} : () -> !eltwise.int
    %c10 = "tile.affine_const"() {value = 10 : i64} : () -> !eltwise.int
    %0 = "tile.domain"() ( {
    ^bb0(%arg2: !eltwise.int, %arg3: !eltwise.int, %arg4: !eltwise.int):	// no predecessors
      %5 = "tile.src_idx_map"(%arg0, %arg4, %arg3, %arg2) : (tensor<1x10x10x!eltwise.fp32>, !eltwise.int, !eltwise.int, !eltwise.int) -> !tile.imap
      %6 = "tile.sink_idx_map"(%arg4, %arg2) : (!eltwise.int, !eltwise.int) -> !tile.imap
      %7 = "tile.size_map"(%c1, %c10) : (!eltwise.int, !eltwise.int) -> !tile.smap
      "tile.>(x)"(%7, %5, %6) : (!tile.smap, !tile.imap, !tile.imap) -> ()
    }) {idx_names = ["x0", "x1", "x2"]} : () -> tensor<1x10x!eltwise.fp32>
    %1 = "tile.domain"() ( {
    ^bb0(%arg2: !eltwise.int):	// no predecessors
      %5 = "tile.src_idx_map"(%arg1) : (!fp32) -> !tile.imap
      %6 = "tile.sink_idx_map"(%arg2) : (!eltwise.int) -> !tile.imap
      %7 = "tile.size_map"(%c10) : (!eltwise.int) -> !tile.smap
      "tile.=(x)"(%7, %5, %6) : (!tile.smap, !tile.imap, !tile.imap) -> ()
    }) {idx_names = ["x0"]} : () -> tensor<10x!eltwise.fp32>
    %2 = "tile.index"(%1) {dim = 0 : i64} : (tensor<10x!eltwise.fp32>) -> tensor<10x!eltwise.int>
    %3 = "tile.domain"() ( {
    ^bb0(%arg2: !eltwise.int, %arg3: !eltwise.int, %arg4: !eltwise.int):	// no predecessors
      %5 = "tile.src_idx_map"(%arg0, %arg4, %arg3, %arg2) : (tensor<1x10x10x!eltwise.fp32>, !eltwise.int, !eltwise.int, !eltwise.int) -> !tile.imap
      %6 = "tile.src_idx_map"(%0, %arg4, %arg2) : (tensor<1x10x!eltwise.fp32>, !eltwise.int, !eltwise.int) -> !tile.imap
      %7 = "tile.src_idx_map"(%2, %arg3) : (tensor<10x!eltwise.int>, !eltwise.int) -> !tile.imap
      %8 = "tile.sink_idx_map"(%arg4, %arg2) : (!eltwise.int, !eltwise.int) -> !tile.imap
      %9 = "tile.size_map"(%c1, %c10) : (!eltwise.int, !eltwise.int) -> !tile.smap
      "tile.>(x==y?z)"(%9, %5, %6, %7, %8) : (!tile.smap, !tile.imap, !tile.imap, !tile.imap, !tile.imap) -> ()
    }) {idx_names = ["x0", "x1", "x2"]} : () -> tensor<1x10x!eltwise.int>
    %4 = "eltwise.cast"(%3) : (tensor<1x10x!eltwise.int>) -> tensor<1x10x!eltwise.u32>
    return %4 : tensor<1x10x!eltwise.u32>
  }
}
''')
        else:
            self.assertEqual(str(O.shape), 'u32(1, 10)')
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
        if USE_MLIR():
            self.assertMultiLineEqual(
                str(program), '''

!fp32 = type tensor<!eltwise.fp32>
module {
  func @global_min(%arg0: tensor<10x10x10x!eltwise.fp32> {tile.name = "I"}) -> !fp32 {
    %0 = "eltwise.neg"(%arg0) {type = !eltwise.fp32} : (tensor<10x10x10x!eltwise.fp32>) -> tensor<10x10x10x!eltwise.fp32>
    %1 = "tile.domain"() ( {
    ^bb0(%arg1: !eltwise.int, %arg2: !eltwise.int, %arg3: !eltwise.int):	// no predecessors
      %3 = "tile.src_idx_map"(%0, %arg3, %arg2, %arg1) : (tensor<10x10x10x!eltwise.fp32>, !eltwise.int, !eltwise.int, !eltwise.int) -> !tile.imap
      %4 = "tile.sink_idx_map"() : () -> !tile.imap
      %5 = "tile.size_map"() : () -> !tile.smap
      "tile.>(x)"(%5, %3, %4) : (!tile.smap, !tile.imap, !tile.imap) -> ()
    }) {idx_names = ["x0", "x1", "x2"]} : () -> !fp32
    %2 = "eltwise.neg"(%1) {type = !eltwise.fp32} : (!fp32) -> !fp32
    return %2 : !fp32
  }
}
''')
        else:
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
        O[i] += I[k]
        O.add_constraint(i - k < N)
        program = Program('cum_sum', [O])
        if USE_MLIR():
            self.assertMultiLineEqual(
                str(program), '''

module {
  func @cum_sum(%arg0: tensor<10x!eltwise.fp32> {tile.name = "I"}) -> tensor<10x!eltwise.fp32> {
    %c10 = "tile.affine_const"() {value = 10 : i64} : () -> !eltwise.int
    %0 = "tile.domain"() ( {
    ^bb0(%arg1: !eltwise.int, %arg2: !eltwise.int):	// no predecessors
      %1 = "tile.src_idx_map"(%arg0, %arg1) : (tensor<10x!eltwise.fp32>, !eltwise.int) -> !tile.imap
      %2 = "tile.sink_idx_map"(%arg2) : (!eltwise.int) -> !tile.imap
      %3 = "tile.size_map"(%c10) : (!eltwise.int) -> !tile.smap
      %4 = "tile.affine_sub"(%arg2, %arg1) : (!eltwise.int, !eltwise.int) -> !eltwise.int
      "tile.constraint"(%4, %c10) ( {
        "tile.+(x)"(%3, %1, %2) : (!tile.smap, !tile.imap, !tile.imap) -> ()
      }) : (!eltwise.int, !eltwise.int) -> ()
    }) {idx_names = ["x0", "x1"]} : () -> tensor<10x!eltwise.fp32>
    return %0 : tensor<10x!eltwise.fp32>
  }
}
''')
        else:
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
        if USE_MLIR():
            self.assertMultiLineEqual(
                str(program), '''

!fp32 = type tensor<!eltwise.fp32>
module {
  func @unique_names(%arg0: !fp32 {tile.name = "C"}, %arg1: !fp32 {tile.name = "C_0"}, %arg2: !fp32 {tile.name = "B"}, %arg3: !fp32 {tile.name = "A"}) -> !fp32 {
    %0 = "eltwise.add"(%arg3, %arg2) {type = !eltwise.fp32} : (!fp32, !fp32) -> !fp32
    %1 = "eltwise.add"(%0, %arg1) {type = !eltwise.fp32} : (!fp32, !fp32) -> !fp32
    %2 = "eltwise.add"(%1, %arg0) {type = !eltwise.fp32} : (!fp32, !fp32) -> !fp32
    return %2 : !fp32
  }
}
''')
        else:
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
        if USE_MLIR():
            self.assertMultiLineEqual(
                str(program), '''

!float = type tensor<!eltwise.float>
!fp32 = type tensor<!eltwise.fp32>
module {
  func @lars_momentum_4d(%arg0: tensor<4x7x3x9x!eltwise.fp32>, %arg1: tensor<4x7x3x9x!eltwise.fp32>, %arg2: !fp32, %arg3: tensor<4x7x3x9x!eltwise.fp32>) -> (tensor<4x7x3x9x!eltwise.fp32>, tensor<4x7x3x9x!eltwise.fp32>) {
    %cst = "eltwise.sconst"() {value = 4.8828125E-4 : f64} : () -> !float
    %cst_0 = "eltwise.sconst"() {value = 9.765625E-4 : f64} : () -> !float
    %cst_1 = "eltwise.sconst"() {value = 1.250000e-01 : f64} : () -> !float
    %0 = "eltwise.mul"(%arg0, %cst) {type = !eltwise.fp32} : (tensor<4x7x3x9x!eltwise.fp32>, !float) -> tensor<4x7x3x9x!eltwise.fp32>
    %1 = "eltwise.add"(%arg1, %0) {type = !eltwise.fp32} : (tensor<4x7x3x9x!eltwise.fp32>, tensor<4x7x3x9x!eltwise.fp32>) -> tensor<4x7x3x9x!eltwise.fp32>
    %2 = "eltwise.mul"(%arg0, %arg0) {type = !eltwise.fp32} : (tensor<4x7x3x9x!eltwise.fp32>, tensor<4x7x3x9x!eltwise.fp32>) -> tensor<4x7x3x9x!eltwise.fp32>
    %3 = "tile.domain"() ( {
    ^bb0(%arg4: !eltwise.int, %arg5: !eltwise.int, %arg6: !eltwise.int, %arg7: !eltwise.int):	// no predecessors
      %17 = "tile.src_idx_map"(%2, %arg7, %arg6, %arg5, %arg4) : (tensor<4x7x3x9x!eltwise.fp32>, !eltwise.int, !eltwise.int, !eltwise.int, !eltwise.int) -> !tile.imap
      %18 = "tile.sink_idx_map"() : () -> !tile.imap
      %19 = "tile.size_map"() : () -> !tile.smap
      "tile.+(x)"(%19, %17, %18) : (!tile.smap, !tile.imap, !tile.imap) -> ()
    }) {idx_names = ["x0", "x1", "x2", "x3"]} : () -> !fp32
    %4 = "eltwise.sqrt"(%3) {type = !eltwise.fp32} : (!fp32) -> !fp32
    %5 = "eltwise.mul"(%4, %cst) {type = !eltwise.fp32} : (!fp32, !float) -> !fp32
    %6 = "eltwise.mul"(%arg1, %arg1) {type = !eltwise.fp32} : (tensor<4x7x3x9x!eltwise.fp32>, tensor<4x7x3x9x!eltwise.fp32>) -> tensor<4x7x3x9x!eltwise.fp32>
    %7 = "tile.domain"() ( {
    ^bb0(%arg4: !eltwise.int, %arg5: !eltwise.int, %arg6: !eltwise.int, %arg7: !eltwise.int):	// no predecessors
      %17 = "tile.src_idx_map"(%6, %arg7, %arg6, %arg5, %arg4) : (tensor<4x7x3x9x!eltwise.fp32>, !eltwise.int, !eltwise.int, !eltwise.int, !eltwise.int) -> !tile.imap
      %18 = "tile.sink_idx_map"() : () -> !tile.imap
      %19 = "tile.size_map"() : () -> !tile.smap
      "tile.+(x)"(%19, %17, %18) : (!tile.smap, !tile.imap, !tile.imap) -> ()
    }) {idx_names = ["x0", "x1", "x2", "x3"]} : () -> !fp32
    %8 = "eltwise.sqrt"(%7) {type = !eltwise.fp32} : (!fp32) -> !fp32
    %9 = "eltwise.add"(%8, %5) {type = !eltwise.fp32} : (!fp32, !fp32) -> !fp32
    %10 = "eltwise.mul"(%arg2, %cst_0) {type = !eltwise.fp32} : (!fp32, !float) -> !fp32
    %11 = "eltwise.mul"(%10, %4) {type = !eltwise.fp32} : (!fp32, !fp32) -> !fp32
    %12 = "eltwise.div"(%11, %9) {type = !eltwise.fp32} : (!fp32, !fp32) -> !fp32
    %13 = "eltwise.mul"(%12, %1) {type = !eltwise.fp32} : (!fp32, tensor<4x7x3x9x!eltwise.fp32>) -> tensor<4x7x3x9x!eltwise.fp32>
    %14 = "eltwise.mul"(%arg3, %cst_1) {type = !eltwise.fp32} : (tensor<4x7x3x9x!eltwise.fp32>, !float) -> tensor<4x7x3x9x!eltwise.fp32>
    %15 = "eltwise.add"(%14, %13) {type = !eltwise.fp32} : (tensor<4x7x3x9x!eltwise.fp32>, tensor<4x7x3x9x!eltwise.fp32>) -> tensor<4x7x3x9x!eltwise.fp32>
    %16 = "eltwise.sub"(%arg0, %15) {type = !eltwise.fp32} : (tensor<4x7x3x9x!eltwise.fp32>, tensor<4x7x3x9x!eltwise.fp32>) -> tensor<4x7x3x9x!eltwise.fp32>
    return %16, %15 : tensor<4x7x3x9x!eltwise.fp32>, tensor<4x7x3x9x!eltwise.fp32>
  }
}
''')
        else:
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
        O[n0, 3 * n1 + k, n2] = I[n0, n1, n2]
        O.add_constraint(k < 3)
        O.no_reduce()
        program = Program('repeat_elts', [O])
        if USE_MLIR():
            self.assertMultiLineEqual(
                str(program), '''

module {
  func @repeat_elts(%arg0: tensor<10x10x10x!eltwise.fp32>) -> tensor<10x30x10x!eltwise.fp32> {
    %c30 = "tile.affine_const"() {value = 30 : i64} : () -> !eltwise.int
    %c10 = "tile.affine_const"() {value = 10 : i64} : () -> !eltwise.int
    %c3 = "tile.affine_const"() {value = 3 : i64} : () -> !eltwise.int
    %0 = "tile.domain"() ( {
    ^bb0(%arg1: !eltwise.int, %arg2: !eltwise.int, %arg3: !eltwise.int, %arg4: !eltwise.int):	// no predecessors
      %1 = "tile.src_idx_map"(%arg0, %arg3, %arg2, %arg1) : (tensor<10x10x10x!eltwise.fp32>, !eltwise.int, !eltwise.int, !eltwise.int) -> !tile.imap
      %2 = "tile.affine_mul"(%arg2, %c3) : (!eltwise.int, !eltwise.int) -> !eltwise.int
      %3 = "tile.affine_add"(%2, %arg4) : (!eltwise.int, !eltwise.int) -> !eltwise.int
      %4 = "tile.sink_idx_map"(%arg3, %3, %arg1) : (!eltwise.int, !eltwise.int, !eltwise.int) -> !tile.imap
      %5 = "tile.size_map"(%c10, %c30, %c10) : (!eltwise.int, !eltwise.int, !eltwise.int) -> !tile.smap
      "tile.constraint"(%arg4, %c3) ( {
        "tile.=(x)"(%5, %1, %4) : (!tile.smap, !tile.imap, !tile.imap) -> ()
      }) : (!eltwise.int, !eltwise.int) -> ()
    }) {idx_names = ["x0", "x1", "x2", "x3"], no_reduce = true} : () -> tensor<10x30x10x!eltwise.fp32>
    return %0 : tensor<10x30x10x!eltwise.fp32>
  }
}
''')
        else:
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
        if USE_MLIR():
            self.assertMultiLineEqual(
                str(program), '''

module {
  func @use_default(%arg0: tensor<1x10x10x!eltwise.fp32>, %arg1: tensor<1x7x10x10x!eltwise.fp32>) -> tensor<1x7x10x10x!eltwise.fp32> {
    %c3 = "tile.affine_const"() {value = 3 : i64} : () -> !eltwise.int
    %c10 = "tile.affine_const"() {value = 10 : i64} : () -> !eltwise.int
    %c7 = "tile.affine_const"() {value = 7 : i64} : () -> !eltwise.int
    %c1 = "tile.affine_const"() {value = 1 : i64} : () -> !eltwise.int
    %0 = "tile.domain"() ( {
    ^bb0(%arg2: !eltwise.int, %arg3: !eltwise.int, %arg4: !eltwise.int):	// no predecessors
      %1 = "tile.src_idx_map"(%arg0, %arg4, %arg3, %arg2) : (tensor<1x10x10x!eltwise.fp32>, !eltwise.int, !eltwise.int, !eltwise.int) -> !tile.imap
      %2 = "tile.sink_idx_map"(%arg4, %c3, %arg3, %arg2) : (!eltwise.int, !eltwise.int, !eltwise.int, !eltwise.int) -> !tile.imap
      %3 = "tile.size_map"(%c1, %c7, %c10, %c10) : (!eltwise.int, !eltwise.int, !eltwise.int, !eltwise.int) -> !tile.smap
      "tile.=(x)"(%3, %1, %2, %arg1) : (!tile.smap, !tile.imap, !tile.imap, tensor<1x7x10x10x!eltwise.fp32>) -> ()
    }) {idx_names = ["x0", "x1", "x2"]} : () -> tensor<1x7x10x10x!eltwise.fp32>
    return %0 : tensor<1x7x10x10x!eltwise.fp32>
  }
}
''')
        else:
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

    def test_defract(self):
        I = Tensor(LogicalShape(plaidml.DType.FLOAT32, [3]), name='I')
        K = Tensor(LogicalShape(plaidml.DType.FLOAT32, [3]), name='K')
        i, j = TensorIndexes(2)
        O = TensorOutput(5)
        O[i] += (I[(i - j + 1) // 2] * K[j])
        program = Program('defract_test', [O])
        if USE_MLIR():
            self.assertMultiLineEqual(
                str(program), '''

module {
  func @defract_test(%arg0: tensor<3x!eltwise.fp32> {tile.name = "I"}, %arg1: tensor<3x!eltwise.fp32> {tile.name = "K"}) -> tensor<5x!eltwise.fp32> {
    %c1 = "tile.affine_const"() {value = 1 : i64} : () -> !eltwise.int
    %c2 = "tile.affine_const"() {value = 2 : i64} : () -> !eltwise.int
    %c5 = "tile.affine_const"() {value = 5 : i64} : () -> !eltwise.int
    %0 = "tile.domain"() ( {
    ^bb0(%arg2: !eltwise.int, %arg3: !eltwise.int):	// no predecessors
      %1 = "tile.affine_sub"(%arg3, %arg2) : (!eltwise.int, !eltwise.int) -> !eltwise.int
      %2 = "tile.affine_add"(%1, %c1) : (!eltwise.int, !eltwise.int) -> !eltwise.int
      %3 = "tile.affine_div"(%2, %c2) : (!eltwise.int, !eltwise.int) -> !eltwise.int
      %4 = "tile.src_idx_map"(%arg0, %3) : (tensor<3x!eltwise.fp32>, !eltwise.int) -> !tile.imap
      %5 = "tile.src_idx_map"(%arg1, %arg2) : (tensor<3x!eltwise.fp32>, !eltwise.int) -> !tile.imap
      %6 = "tile.sink_idx_map"(%arg3) : (!eltwise.int) -> !tile.imap
      %7 = "tile.size_map"(%c5) : (!eltwise.int) -> !tile.smap
      "tile.+(x*y)"(%7, %4, %5, %6) : (!tile.smap, !tile.imap, !tile.imap, !tile.imap) -> ()
    }) {idx_names = ["x0", "x1"]} : () -> tensor<5x!eltwise.fp32>
    return %0 : tensor<5x!eltwise.fp32>
  }
}
''')
        else:
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
        outputs = plaidml_exec.run(program, [(I, np.array([1, 2, 3])), (K, np.array([1, 2, 3]))])
        self.assertEqual(outputs[0].tolist(), [2, 5, 4, 9, 6])

    def test_defract_short(self):
        I = Tensor(LogicalShape(plaidml.DType.FLOAT32, [3]), name='I')
        i, j = TensorIndexes(2)
        O = TensorOutput(6)
        O[i] += (I[(i - 1) // 2])
        program = Program('defract_short_test', [O])
        if USE_MLIR():
            self.assertMultiLineEqual(
                str(program), '''

module {
  func @defract_short_test(%arg0: tensor<3x!eltwise.fp32> {tile.name = "I"}) -> tensor<6x!eltwise.fp32> {
    %c1 = "tile.affine_const"() {value = 1 : i64} : () -> !eltwise.int
    %c2 = "tile.affine_const"() {value = 2 : i64} : () -> !eltwise.int
    %c6 = "tile.affine_const"() {value = 6 : i64} : () -> !eltwise.int
    %0 = "tile.domain"() ( {
    ^bb0(%arg1: !eltwise.int):	// no predecessors
      %1 = "tile.affine_sub"(%arg1, %c1) : (!eltwise.int, !eltwise.int) -> !eltwise.int
      %2 = "tile.affine_div"(%1, %c2) : (!eltwise.int, !eltwise.int) -> !eltwise.int
      %3 = "tile.src_idx_map"(%arg0, %2) : (tensor<3x!eltwise.fp32>, !eltwise.int) -> !tile.imap
      %4 = "tile.sink_idx_map"(%arg1) : (!eltwise.int) -> !tile.imap
      %5 = "tile.size_map"(%c6) : (!eltwise.int) -> !tile.smap
      "tile.+(x)"(%5, %3, %4) : (!tile.smap, !tile.imap, !tile.imap) -> ()
    }) {idx_names = ["x0"]} : () -> tensor<6x!eltwise.fp32>
    return %0 : tensor<6x!eltwise.fp32>
  }
}
''')
        else:
            self.assertMultiLineEqual(
                str(program), '''function (
  I[I_0]
) -> (
  _X0
) {
  _X0[x0 : 6] = +(I[-1/2 + 1/2*x0]);
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
        if USE_MLIR():
            self.assertMultiLineEqual(
                str(program), '''

module {
  func @defract_long(%arg0: tensor<1x3x3x1x!eltwise.fp32> {tile.name = "I"}, %arg1: tensor<1x3x3x1x!eltwise.fp32> {tile.name = "K"}) -> tensor<1x5x5x1x!eltwise.fp32> {
    %c2 = "tile.affine_const"() {value = 2 : i64} : () -> !eltwise.int
    %c5 = "tile.affine_const"() {value = 5 : i64} : () -> !eltwise.int
    %c1 = "tile.affine_const"() {value = 1 : i64} : () -> !eltwise.int
    %0 = "tile.domain"() ( {
    ^bb0(%arg2: !eltwise.int, %arg3: !eltwise.int, %arg4: !eltwise.int, %arg5: !eltwise.int, %arg6: !eltwise.int, %arg7: !eltwise.int, %arg8: !eltwise.int):	// no predecessors
      %1 = "tile.affine_add"(%arg4, %arg3) : (!eltwise.int, !eltwise.int) -> !eltwise.int
      %2 = "tile.affine_sub"(%1, %c1) : (!eltwise.int, !eltwise.int) -> !eltwise.int
      %3 = "tile.affine_div"(%2, %c2) : (!eltwise.int, !eltwise.int) -> !eltwise.int
      %4 = "tile.affine_add"(%arg6, %arg5) : (!eltwise.int, !eltwise.int) -> !eltwise.int
      %5 = "tile.affine_sub"(%4, %c1) : (!eltwise.int, !eltwise.int) -> !eltwise.int
      %6 = "tile.affine_div"(%5, %c2) : (!eltwise.int, !eltwise.int) -> !eltwise.int
      %7 = "tile.src_idx_map"(%arg0, %arg7, %6, %3, %arg2) : (tensor<1x3x3x1x!eltwise.fp32>, !eltwise.int, !eltwise.int, !eltwise.int, !eltwise.int) -> !tile.imap
      %8 = "tile.affine_sub"(%c2, %arg3) : (!eltwise.int, !eltwise.int) -> !eltwise.int
      %9 = "tile.affine_sub"(%c2, %arg5) : (!eltwise.int, !eltwise.int) -> !eltwise.int
      %10 = "tile.src_idx_map"(%arg1, %9, %8, %arg8, %arg2) : (tensor<1x3x3x1x!eltwise.fp32>, !eltwise.int, !eltwise.int, !eltwise.int, !eltwise.int) -> !tile.imap
      %11 = "tile.sink_idx_map"(%arg7, %arg6, %arg4, %arg8) : (!eltwise.int, !eltwise.int, !eltwise.int, !eltwise.int) -> !tile.imap
      %12 = "tile.size_map"(%c1, %c5, %c5, %c1) : (!eltwise.int, !eltwise.int, !eltwise.int, !eltwise.int) -> !tile.smap
      "tile.+(x*y)"(%12, %7, %10, %11) : (!tile.smap, !tile.imap, !tile.imap, !tile.imap) -> ()
    }) {idx_names = ["x0", "x1", "x2", "x3", "x4", "x5", "x6"]} : () -> tensor<1x5x5x1x!eltwise.fp32>
    return %0 : tensor<1x5x5x1x!eltwise.fp32>
  }
}
''')
        else:
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
        if USE_MLIR():
            self.assertMultiLineEqual(
                str(program), '''

module {
  func @"this-is-not an identifier"(%arg0: tensor<3x!eltwise.fp32> {tile.name = "I"}, %arg1: tensor<3x!eltwise.fp32> {tile.name = "K"}) -> tensor<5x!eltwise.fp32> {
    %c1 = "tile.affine_const"() {value = 1 : i64} : () -> !eltwise.int
    %c2 = "tile.affine_const"() {value = 2 : i64} : () -> !eltwise.int
    %c5 = "tile.affine_const"() {value = 5 : i64} : () -> !eltwise.int
    %0 = "tile.domain"() ( {
    ^bb0(%arg2: !eltwise.int, %arg3: !eltwise.int):	// no predecessors
      %1 = "tile.affine_sub"(%arg3, %arg2) : (!eltwise.int, !eltwise.int) -> !eltwise.int
      %2 = "tile.affine_add"(%1, %c1) : (!eltwise.int, !eltwise.int) -> !eltwise.int
      %3 = "tile.affine_div"(%2, %c2) : (!eltwise.int, !eltwise.int) -> !eltwise.int
      %4 = "tile.src_idx_map"(%arg0, %3) : (tensor<3x!eltwise.fp32>, !eltwise.int) -> !tile.imap
      %5 = "tile.src_idx_map"(%arg1, %arg2) : (tensor<3x!eltwise.fp32>, !eltwise.int) -> !tile.imap
      %6 = "tile.sink_idx_map"(%arg3) : (!eltwise.int) -> !tile.imap
      %7 = "tile.size_map"(%c5) : (!eltwise.int) -> !tile.smap
      "tile.+(x*y)"(%7, %4, %5, %6) : (!tile.smap, !tile.imap, !tile.imap, !tile.imap) -> ()
    }) {idx_names = ["x0", "x1"]} : () -> tensor<5x!eltwise.fp32>
    return %0 : tensor<5x!eltwise.fp32>
  }
}
''')
        else:
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
        outputs = plaidml_exec.run(program, [(I, np.array([1, 2, 3])), (K, np.array([1, 2, 3]))])
        self.assertEqual(outputs[0].tolist(), [2, 5, 4, 9, 6])

    def test_identity(self):
        I = Tensor(LogicalShape(plaidml.DType.FLOAT32, [3]), name='I')
        program = Program('identity', [I])
        if USE_MLIR():
            self.assertMultiLineEqual(
                str(program), '''

module {
  func @identity(%arg0: tensor<3x!eltwise.fp32> {tile.name = "I"}) -> tensor<3x!eltwise.fp32> {
    %0 = "eltwise.ident"(%arg0) {type = !eltwise.fp32} : (tensor<3x!eltwise.fp32>) -> tensor<3x!eltwise.fp32>
    return %0 : tensor<3x!eltwise.fp32>
  }
}
''')
        else:
            self.assertMultiLineEqual(
                str(program), '''function (
  I[I_0]
) -> (
  _X0
) {
  _X0 = ident(I);
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
        if USE_MLIR():
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
        if USE_MLIR():
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
        else:
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
