import argparse
import sys
import unittest

import plaidml
from plaidml.edsl import *


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
    product = 1
    X_shape = X.shape()
    for i in range(1, len(X_shape.dims) - 1):
        product *= X_shape.dims[i].size
    shape = TensorShape(X_shape.type, (1, product))
    return reshape(X, shape)


def normalize(X):
    idxs = TensorIndexes(X.shape().dims.size())
    XSqr = X * X
    X_MS = TensorOutput()
    X_MS[()] += XSqr[idxs]
    return sqrt(X_MS)


class TestEdsl(unittest.TestCase):

    def testMnistMlp(self):
        # model.add(Dense(512, activation='relu', input_shape=(784,)))
        I = Tensor(TensorShape(plaidml.DType.FLOAT32, [1, 784]))
        K1 = Tensor(TensorShape(plaidml.DType.FLOAT32, [784, 512]))
        B1 = Tensor(TensorShape(plaidml.DType.FLOAT32, [512]))
        D1 = relu(dot(I, K1) + B1)
        # model.add(Dense(512, activation='relu'))
        K2 = Tensor(TensorShape(plaidml.DType.FLOAT32, [512, 512]))
        B2 = Tensor(TensorShape(plaidml.DType.FLOAT32, [512]))
        D2 = relu(dot(D1, K2) + B2)
        # model.add(Dense(10, activation='softmax'))
        K3 = Tensor(TensorShape(plaidml.DType.FLOAT32, [512, 10]))
        B3 = Tensor(TensorShape(plaidml.DType.FLOAT32, [10]))
        D3 = softmax(dot(D2, K3) + B3)

    def testMnistCnn(self):
        # model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
        I = Tensor(TensorShape(plaidml.DType.FLOAT32, [1, 224, 224, 1]))
        K1 = Tensor(TensorShape(plaidml.DType.FLOAT32, [3, 3, 1, 32]))
        B1 = Tensor(TensorShape(plaidml.DType.FLOAT32, [32]))
        C1 = relu(conv_2d(I, K1) + B1)
        # model.add(Conv2D(64, (3, 3), activation='relu'))
        K2 = Tensor(TensorShape(plaidml.DType.FLOAT32, [3, 3, 32, 64]))
        B2 = Tensor(TensorShape(plaidml.DType.FLOAT32, [64]))
        C2 = relu(conv_2d(C1, K2) + B2)
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        P1 = max_pool_2d(C2)
        # model.add(Flatten())
        # F = flatten(P1)
        # K3 = Tensor()
        # B3 = Tensor()
        # D1 = relu(dot(F, K3) + B3)
        # # model.add(Dense(num_classes, activation='softmax'))
        # K4 = Tensor()
        # B4 = Tensor()
        # D2 = softmax(dot(D1, K4) + B4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', type=int, default=0)
    args, remainder = parser.parse_known_args()
    plaidml._internal_set_vlog(args.verbose)
    unittest.main(argv=sys.argv[:1] + remainder, verbosity=args.verbose + 1)
