# Copyright 2021 Intel Corporation.
# Note:
#    This file is being used by sphinx docs to pull in code blocks.
#    Code blocks are pulled into docs/usage/*.rst
#    Any changes made here may upset the docs.

import unittest

import plaidml
from plaidml.edsl import *


def quantize_float32_to_int8(A, scale, zeropoint):
    O = A / scale
    O_int = cast(O, DType.INT8)
    return O_int + zeropoint
