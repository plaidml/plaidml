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


def argmax(x, axis=-1):
    return op('argmax', [x, axis]).as_tensor()


def dot(x, y):
    return op('dot', [x, y]).as_tensor()


def mean(I, axis=None, keepdims=False):
    if isinstance(axis, np.ndarray):
        axis = axis.tolist()
    return op('mean', [I, axis, keepdims]).as_tensor()


def relu(x, alpha=None, max_value=None, threshold=0.):
    return op('relu', [x, alpha, max_value, threshold]).as_tensor()


def softmax(x, axis=None):
    return op('softmax', [x, axis]).as_tensor()


def pool(I, pool_mode, pool_size, strides, autopadding, manual_padding, data_layout, use_ceil,
         include_pad_in_avg):
    if isinstance(pool_size, np.ndarray):
        pool_size = pool_size.tolist()
    if isinstance(strides, np.ndarray):
        strides = strides.tolist()
    if isinstance(manual_padding, np.ndarray):
        manual_padding = manual_padding.to_list()
    return op("pool", [
        I, pool_mode, pool_size, strides, autopadding, manual_padding, data_layout, use_ceil,
        include_pad_in_avg
    ]).as_tensor()


def square(I):
    return I * I


def sum(I, axis=None, keepdims=False):
    if isinstance(axis, np.ndarray):
        axis = axis.tolist()
    return op('sum', [I, axis, keepdims]).as_tensor()
