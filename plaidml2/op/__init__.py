# Copyright 2019 Intel Corporation.

import logging

import numpy as np
import plaidml2.edsl as edsl
from plaidml2.ffi import ForeignObject, decode_str, ffi, ffi_call, lib

logger = logging.getLogger(__name__)


def __init():
    ffi_call(lib.plaidml_op_init)


ffi.init_once(__init, 'plaidml_op_init')


def op(op_name, args):
    value = edsl.Value(args)
    return edsl.Value(ffi_call(lib.plaidml_op_make, op_name.encode(), value.as_ptr()))


def mean(I, axis=None, keepdims=False):
    if isinstance(axis, np.ndarray):
        axis = axis.tolist()
    return op("mean", [I, axis, keepdims]).as_tensor()


def square(I):
    return I * I


def sum(I, axis=None, keepdims=False):
    if isinstance(axis, np.ndarray):
        axis = axis.tolist()
    return op("sum", [I, axis, keepdims]).as_tensor()
