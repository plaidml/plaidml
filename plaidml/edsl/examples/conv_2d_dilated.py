# Copyright 2021 Intel Corporation.
# Note:
#    This file is being used by sphinx docs to pull in code blocks.
#    Code blocks are pulled into docs/usage/*.rst
#    Any changes made here may upset the docs.

import plaidml
from plaidml.edsl import *


def conv_2d_dilated(I, K):
    N, X, Y, KX, KY, CI, CO = TensorDims(7)
    n, x, y, kx, ky, ci, co = TensorIndexes(7)
    I.bind_dims(N, X, Y, CI)
    K.bind_dims(KX, KY, CI, CO)
    return Contraction()\
            .outShape(N, X - 2 * (KX - 1), Y - 3 * (KY - 1), CO)\
            .outAccess(n, x, y, co)\
            .sum(I[n, x + 2 * kx, y + 3 * ky, ci] * K[kx, ky, ci, co]).build()