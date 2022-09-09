# Copyright 2021 Intel Corporation.
# Note:
#    This file is being used by sphinx docs to pull in code blocks.
#    Code blocks are pulled into docs/usage/*.rst
#    Any changes made here may upset the docs.

import plaidml
from plaidml.edsl import *


def gemv(A, x, y):
    I, J = TensorDims(2)
    i, j = TensorIndexes(2)
    A.bind_dims(I, J)
    x.bind_dims(J)
    return Contraction().outShape(J).outAccess(i).sum(A[i, j] * x[j]).build() + y


def gemv2(A, x, y, alpha, beta):
    I, J = TensorDims(2)
    i, j = TensorIndexes(2)
    A_alpha = A * alpha
    A_alpha.bind_dims(I, J)
    x.bind_dims(J)
    y_beta = y * beta
    y_beta.bind_dims(J)
    return Contraction().outShape(I).outAccess(i).sum(A[i, j] * x[j]).build() + y
