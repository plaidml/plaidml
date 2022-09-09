# Copyright 2021 Intel Corporation.
# Note:
#    This file is being used by sphinx docs to pull in code blocks.
#    Code blocks are pulled into docs/usage/*.rst
#    Any changes made here may upset the docs.

import plaidml
from plaidml.edsl import *


def conv_1d(I, K):
    N, X, KX, CI, CO = TensorDims(5)
    n, x, k, ci, co = TensorIndexes(5)
    I.bind_dims(N, X, CI)
    K.bind_dims(KX, CI, CO)
    return Contraction()\
            .outShape(N, X - KX + 1, CO)\
            .outAccess(n, x, co).sum(I[n, x + k, ci] * K[k, ci, co]).build()
