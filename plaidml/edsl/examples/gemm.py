# Copyright 2021 Intel Corporation.
# Note:
#    This file is being used by sphinx docs to pull in code blocks.
#    Code blocks are pulled into docs/usage/*.rst
#    Any changes made here may upset the docs.

import plaidml
from plaidml.edsl import *


def gemm(A, B, C):
    I, J, K = TensorDims(3)
    i, j, k = TensorIndexes(3)
    A.bind_dims(I, K)
    B.bind_dims(K, J)
    O = Contraction().outShape(I, J).outAccess(i, j).sum(A[i, k] * B[k, j]).build()
    return O + C
