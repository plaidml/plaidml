# Copyright 2021 Intel Corporation.
# Note:
#    This file is being used by sphinx docs to pull in code blocks.
#    Code blocks are pulled into docs/usage/*.rst
#    Any changes made here may upset the docs.

import unittest

import plaidml
from plaidml.edsl import *

# The example below illustrates a maxpool operation on a one dimensional tensor.


def max_pool_1d(I):
    N = TensorDim()
    i, j = TensorIndexes(2)
    I.bind_dims(N)
    return Contraction().outShape(N // 2).outAccess(i).max(I[2 * i +
                                                             j]).add_constraint(j < 2).build()


# the code below is used in documentation for further explanation of the maxpool operation


def for_loop_max_pool_1d():
    # for_loop_max_pool_1d_start
    for i in range(1, N // 2):
        curr_max = numpy.finfo(float).eps
        for j in range(1, 2):
            if I[2 * i * j] > curr_max:
                curr_max = I[2 * i + j]
        O[i] = curr_max
    # for_loop_max_pool_1d_end


def wrong_max_pool_1d(I):
    N = TensorDim()
    i, j = TensorIndexes(2)
    I.bind_dims(N)
    return Contraction().outShape(N // 2).outAccess(i).max(I[2 * i + j]).build()
