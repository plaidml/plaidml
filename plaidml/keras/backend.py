# Copyright Vertex.AI.
"""Implements a Keras backend using PlaidML.

This module implements the Keras backend interface, using PlaidML for computation.

As of this writing, Keras expects the backend module to be located at .backend,
and hard-codes the list of available modules; there's no straightforward way to
load a new backend.  So the PlaidML backend needs to be monkey-patched in place,
using the following code:

    import plaidml.keras
    plaidml.keras.install_backend()

This should be done in the main program module, after __future__ imports
(if any) and before importing any Keras modules.  Calling install_backend()
replaces the standard keras.backend module with the PlaidML backend, causing
subsequently loaded Keras modules to use PlaidML.
"""

from __future__ import print_function, division

import atexit
import functools
import math
import numpy as np
import operator
import os
import plaidml
import scipy.stats
import six
import sys
import threading
import traceback
import types

from collections import OrderedDict
from contextlib import contextmanager
from six import iteritems
from six.moves import builtins

from keras.backend.common import cast_to_floatx
from keras.backend.common import epsilon
from keras.backend.common import floatx
from keras.backend.common import image_data_format
from keras.backend.common import set_floatx as keras_set_floatx
from keras.backend.common import set_image_data_format


class PlaidMLKerasException(Exception):
    pass


_dtypes = {
    'float16': plaidml.DType.FLOAT16,
    'float32': plaidml.DType.FLOAT32,
    'float64': plaidml.DType.FLOAT64,
    'bool': plaidml.DType.BOOLEAN,
    'int32': plaidml.DType.INT32,
    'int64': plaidml.DType.INT64,
    'uint32': plaidml.DType.UINT32,
    'uint64': plaidml.DType.UINT64,
}

_in_train_phase = None  #Will be initialized on first use
_app_stack = []

_ctx = plaidml.Context()

PLAIDML_EVENTLOG_FILENAME = os.getenv('PLAIDML_EVENTLOG_FILENAME')
if PLAIDML_EVENTLOG_FILENAME:
    print("Logging events to", PLAIDML_EVENTLOG_FILENAME)
    _ctx.set_eventlog_filename(PLAIDML_EVENTLOG_FILENAME)

    @atexit.register
    def close_eventlog():
        _ctx.shutdown()


_device_lock = threading.Lock()
_dev = None


def _device():
    global _ctx, _dev, _device_lock
    with _device_lock:
        if not _dev:
            devices = plaidml.devices(_ctx)
            _dev = plaidml.Device(_ctx, devices[0])
    return _dev


# Keras needs us to keep track of unique IDs for prefix strings
# (for use with get_uid and reset_uids)
_UID_PREFIX_DICT = dict()


def _broadcast_shape(x, y):
    """Returns the shape of the result of a broadcasted element-wise operation.

    Note: We assume the first parameter is an actual PlaidMLKeras Variable object.
    """
    x_shape = list(x.shape)
    if not isinstance(y, _Var):
        return x_shape
    y_shape = list(y.shape)
    if len(y_shape) < len(x_shape):
        (x_shape, y_shape) = (y_shape, x_shape)
    if len(x_shape) < len(y_shape):
        x_shape = ([1] * (len(y_shape) - len(x_shape))) + x_shape

    def broad_dim(xd, yd):
        if xd == 1 or xd == None:
            return yd
        if yd == 1 or yd == None:
            return xd
        if xd == yd:
            return xd
        e = PlaidMLKerasException('Broadcast failure: (%s) and (%s) cannot be unified' %
                                  (', '.join([str(d) for d in x_shape]),
                                   ', '.join([str(d) for d in y_shape])))
        raise e  # Incompatible shapes

    return tuple([broad_dim(x, y) for (x, y) in zip(x_shape, y_shape)])


def _dump_val(x, indent=2):
    if not isinstance(x, _Var):
        return '[val=%s, type=%s]' % (str(x), type(x).__name__)
    result = str(x)
    if isinstance(x, _Op):
        for (name, val) in iteritems(x._inputs):
            result = result + '\n' + (' ' * indent) + 'where ' + name + ' = ' + _dump_val(
                val, indent + 2)
    elif isinstance(x, _Var) and x._src:
        result = result + '\n' + (' ' * indent) + 'from ' + _dump_val(x._src, indent + 2)
    return result


def _plaidml_val(x, indent=0):
    """Gets the PlaidML value for some input x (unwrapping wrapped values).

    This function also implements composite operation recognition, making it
    possible to represent operations in a Keras-specific computation tree
    that have no simple / efficient representation in PlaidML.
    """
    if not isinstance(x, _Var):
        return x

    try:
        if (x.ident == 'equal' and x._inputs['X'].ident == 'argmax' and
                x._inputs['Y'].ident == 'argmax'):
            # Basically: we want a bool tensor that's true iff the index of
            # the maximum element along some axis for both tensors is the same.
            # Since we don't have argmax itself, what we do is, for each tensor,
            # extract the maximum element along the axis, broadcast it and test
            # for equality (creating a bool tensor of the same shape as the original),
            # logically and the two bool tensors together, and then contract them
            # again along the requested axis (with a logical or).
            #
            # TODO: It would be lovely to have a suite of operators, combiners,
            # and aggregators for boolean logic.
            #
            # TODO: It would also be nice to be able to express operations like
            # argmax directly.
            b = x._inputs['X']._inputs['I'].ismax(x._inputs['X']._inputs['axis'])
            c = x._inputs['Y']._inputs['I'].ismax(x._inputs['Y']._inputs['axis'])
            and_f = """function (B, C) -> (A) {
                           A = B ? (C ? 1 : 0) : 0;
                       }"""
            and_op = _Op('max_arg_match', 'int32', b.shape, and_f,
                         OrderedDict([('B', b), ('C', c)]), ['A'])
            sum_op = and_op.sum(axis=x._inputs['X']._inputs['axis'], keepdims=True)
            eq_f = """function (I) -> (O) {
                          O = 0 < I;
                      }"""
            x = _Op('eq_argmax', 'bool', sum_op.shape, eq_f, {'I': sum_op}, ['O'])
        if x.ident == 'slice' and x._inputs['I'].ident == 'shape' and isinstance(x.original_key, int):
            in_tensor = x._inputs['I']._inputs['T']
            key = x.original_key
            if not isinstance(key, int):
                raise ValueError("The output of shape can only be sliced using integers; received type '{}'".format(type(key)))
            if key < -in_tensor.ndim or key >= in_tensor.ndim:
                raise ValueError("Asked to get dimension {} of a tensor with only {} dimensions".format(key, in_tensor.ndim))
            if key < 0:
                key = key % in_tensor.ndim
            name = 'extract_shape{}'.format(key)
            f = """ function (I[{dims}]) -> (O) {{
                        O = N{key};
                    }}""".format(dims=", ".join("N{}".format(i) for i in range(in_tensor.ndim)), key=key)
            x = _Op(name, 'int32', tuple(), f, {'I': in_tensor}, ['O'])

    except:
        pass

    return x._plaidml_val(indent)


def _report_unimplemented(name):
    report = ("The Keras backend function '{}' is not yet implemented in " +
              "Plaid. You can help us prioritize by letting us know if this " +
              "function is important to you, and as always, contributions are " +
              "welcome!").format(name)
    raise NotImplementedError(report)


class _Var(object):
    """A PlaidML variable.

    Variables may be directly allocated by the caller, or returned to Keras by
    the PlaidML backend."""

    def __init__(self,
                 ident,
                 dtype,
                 shape,
                 name=None,
                 plaidml_val=None,
                 src=None,
                 is_keras_tensor=False):
        self._ident = ident
        self._keras_dtype = dtype
        if isinstance(shape, int):
            shape = (shape,)
        self._keras_shape = tuple(shape)
        self._name = name
        self._value = plaidml_val
        self._src = src
        self._is_keras_tensor = is_keras_tensor

    @property
    def is_keras_tensor(self):
        return self._is_keras_tensor

    @property
    def ident(self):
        return self._ident

    @property
    def dtype(self):
        return self._keras_dtype

    @property
    def shape(self):
        return self._keras_shape

    @property
    def ndim(self):
        return len(self._keras_shape)

    @property
    def name(self):
        if self._name:
            return self._name
        return self.ident

    def _plaidml_val(self, indent=0):
        return self._value

    def _side_effects(self):
        return {}

    def eval(self):
        exn = PlaidMLKerasException('unable to evaluate \'%s\' (%s)' % (self.name, self.ident))
        raise exn

    def _parse_slice(self, key, idx):
        if isinstance(key[idx], int):
            return 1, None, key[idx]
        if ((not isinstance(key[idx].start, int) and not isinstance(key[idx].start, type(None))) or
            (not isinstance(key[idx].stop, int) and not isinstance(key[idx].stop, type(None))) or
            (not isinstance(key[idx].step, int) and not isinstance(key[idx].step, type(None)))):
            raise ValueError("Must use ints when slicing _Op; received {}".format(key[idx]))
        step = key[idx].step or 1
        if step == 0:
            raise ValueError("Cannot slice with step size 0")

        start = key[idx].start
        if start == None:
            if step > 0:
                start = 0
            else:
                start = -1
        if start < 0:
            if self.shape[idx] == None:  #Replace condition w/ 'True' for some tests
                start = 'N{} + {}'.format(idx, start)
            else:
                start = self.shape[idx] + start
        if step > 0:
            if self.shape[idx] == None:  #Replace condition w/ 'True' for some tests
                start = 'max({}, 0)'.format(start)
            else:
                start = getattr(builtins, "max")(start, 0)
        else:
            if self.shape[idx] == None:  #Replace condition w/ 'True' for some tests
                start = 'min({}, N{} - 1)'.format(start, idx)
            else:
                start = getattr(builtins, "min")(start, self.shape[idx] - 1)

        stop = key[idx].stop
        if stop == None:
            if step > 0:
                if self.shape[idx] == None:  #Replace condition w/ 'True' for some tests
                    stop = 'N{}'.format(idx)
                else:
                    stop = self.shape[idx]
            else:
                stop = -1
            # Can return now and skip unneeded max/min
            if self.shape[idx] == None:  #Replace condition w/ 'True' for some tests
                return '({} - ({}))'.format(stop, start), step, start
            return stop - start, step, start
        elif stop < 0:
            if self.shape[idx] == None:  #Replace condition w/ 'True' for some tests
                stop = 'N{} + {}'.format(idx, stop)
            else:
                stop = self.shape[idx] + stop
        if step > 0:
            if self.shape[idx] == None:  #Replace condition w/ 'True' for some tests
                stop = 'min({}, N{})'.format(stop, idx)
            else:
                stop = getattr(builtins, "min")(stop, self.shape[idx])
        else:
            if self.shape[idx] == None:  #Replace condition w/ 'True' for some tests
                stop = 'max({}, -1)'.format(stop)
            else:
                stop = getattr(builtins, "max")(stop, -1)
        if self.shape[idx] == None:  #Replace condition w/ 'True' for some tests
            length_numerator = '({} - ({}))'.format(stop, start)
        else:
            length_numerator = stop - start
        return length_numerator, step, start

    def _gen_slice(self, key):
        if isinstance(key, slice) or isinstance(key, int):
            key = (key,)
        if not isinstance(key, tuple):
            raise ValueError("Cannot index _Var using type {}".format(type(key)))

        var_list = list()
        dim_list = list()
        formula_list = list()
        offset_list = list()
        shape = list()
        inner_idx = 0
        for idx in range(len(key)):
            length_numerator, step, offset = self._parse_slice(key, idx)
            if step == None:
                # In this case offset is an int
                if offset >= 0:
                    formula_list.append('{}'.format(offset))
                else:
                    offset_list.append('Offset{} = N{}+{};'.format(idx, idx, offset))
                    formula_list.append('{}'.format('Offset{}'.format(idx)))
            else:
                var_list.append('i{}'.format(inner_idx))
                dim_subs = {'numer': length_numerator, 'step': step}
                if step > 0:
                    dim_list.append('({numer} + {step} - 1)/{step}'.format(**dim_subs))
                else:
                    dim_list.append('({numer} + {step} + 1)/{step}'.format(**dim_subs))
                if isinstance(length_numerator, str):
                    shape.append(None)
                    offset_list.append('Offset{} = {};'.format(idx, offset))
                    formula_list.append('{}*i{}+{}'.format(step, inner_idx,
                                                           'Offset{}'.format(idx)))
                else:
                    shape.append(int(math.ceil(float(length_numerator) / step)))
                    formula_list.append('{}*i{}+{}'.format(step, inner_idx, offset))
                inner_idx += 1
        # Separately handle extra indices not sliced over
        for idx in range(len(key), len(self.shape)):
            var_list.append('i{}'.format(inner_idx))
            dim_list.append('N{}'.format(idx))
            shape.append(None)
            formula_list.append('i{}'.format(inner_idx))
            inner_idx += 1
        shape = tuple(shape)
        return (var_list, dim_list, formula_list, offset_list, shape)

    def __getitem__(self, key):
        (var_list, dim_list, formula_list, offset_list, shape) = self._gen_slice(key)

        if len(shape) == 0:
            body = "  O[] = =(I[" + ', '.join(formula_list) + "]);"
        else:
            body = "  O[{}: {}] = =(I[{}]);".format(', '.join(var_list), ', '.join(dim_list),
                                                    ', '.join(formula_list))

        # TODO: Example below is out of date, although it shows the spirit of the op
        # Example 'code' (slicing X[5:10,3,:,2:6:2]):
        #   function (I[N0, N1, N2, N3]) -> (O) {
        #     O[i0, i1, i2: 5, N2, 2] = +(I[i0+5, 3, i1, 2*i2+2]);
        #   }
        subs = {
            'indims': ', '.join(['N{}'.format(i) for i in range(len(self.shape))]),
            'offsets': '  \n' + '  \n'.join(offset_list),
            'body': body
        }
        code = ('function (I[{indims}]) -> (O) {{{offsets}\n{body}\n}}').format(**subs)

        op = _Op('slice', self.dtype, shape, code, {'I': self}, ['O'])
        op.original_key = key   # Make key info available for special slice-of-shape parsing
        return op

    def __str__(self):
        return self.name + '[' + ', '.join([str(dim) for dim in self.shape]) + ']'

    def __repr__(self):
        return self.name + '[' + ', '.join([str(dim) for dim in self.shape]) + ']'

    def _compute_agg_axes(self, axis=None, keepdims=False):
        if axis is None:
            axis = self.ndim - 1
        if isinstance(axis, list):
            axis = [(self.ndim + i if i < 0 else i) for i in axis]
        else:
            if axis < 0:
                axis = self.ndim + axis
            axis = [axis]
        axis.sort(reverse=True)
        src_indices = ['x' + str(i) for i in range(self.ndim)]
        src_ranges = ['X' + str(i) for i in range(self.ndim)]
        dest_indices = src_indices[:]
        dest_ranges = src_ranges[:]
        shape = list(self.shape)
        if keepdims:
            for ax in axis:
                dest_indices[ax] = 's' + dest_indices[ax]
                dest_ranges[ax] = '1'
                shape[ax] = 1
        else:
            for ax in axis:
                del dest_indices[ax]
                del dest_ranges[ax]
                shape = list(shape)
                del shape[ax]
        shape = tuple(shape)

        return shape, axis, {
            'src_indices': ', '.join(src_indices),
            'src_ranges': ', '.join(src_ranges),
            'src_sep': ' : ' if len(src_indices) else '',
            'dest_indices': ', '.join(dest_indices),
            'dest_ranges': ', '.join(dest_ranges),
            'dest_sep': ' : ' if len(dest_indices) else '',
        }

    def __add__(self, other):
        return _Op('+', self.dtype,
                   _broadcast_shape(self, other), 'function (B, C) -> (A) { A = B + C; }',
                   OrderedDict([('B', self), ('C', other)]), ['A'])

    def __radd__(self, other):
        return _Op('+', self.dtype,
                   _broadcast_shape(self, other), 'function (B, C) -> (A) { A = B + C; }',
                   OrderedDict([('B', other), ('C', self)]), ['A'])

    def __sub__(self, other):
        return _Op('-', self.dtype,
                   _broadcast_shape(self, other), 'function (B, C) -> (A) { A = B - C; }',
                   OrderedDict([('B', self), ('C', other)]), ['A'])

    def __rsub__(self, other):
        return _Op('-', self.dtype,
                   _broadcast_shape(self, other), 'function (B, C) -> (A) { A = B - C; }',
                   OrderedDict([('B', other), ('C', self)]), ['A'])

    def __neg__(self):
        return _Op('neg', self.dtype, self.shape, 'function (B) -> (A) { A = -B; }',
                   OrderedDict([('B', self)]), ['A'])

    def __mul__(self, other):
        return _Op('*', self.dtype,
                   _broadcast_shape(self, other), 'function (B, C) -> (A) { A = B * C; }',
                   OrderedDict([('B', self), ('C', other)]), ['A'])

    def __rmul__(self, other):
        return _Op('*', self.dtype,
                   _broadcast_shape(self, other), 'function (B, C) -> (A) { A = B * C; }',
                   OrderedDict([('B', other), ('C', self)]), ['A'])

    def __div__(self, other):
        # TODO: Consider implementing truncating div, instead of truediv.
        return _Op('/', self.dtype,
                   _broadcast_shape(self, other), 'function (B, C) -> (A) { A = B / C; }',
                   OrderedDict([('B', self), ('C', other)]), ['A'])

    def __rdiv__(self, other):
        # TODO: Consider implementing truncating div, instead of truediv.
        return _Op('/', self.dtype,
                   _broadcast_shape(self, other), 'function (B, C) -> (A) { A = B / C; }',
                   OrderedDict([('B', other), ('C', self)]), ['A'])

    def __truediv__(self, other):
        return _Op('/', self.dtype,
                   _broadcast_shape(self, other), 'function (B, C) -> (A) { A = B / C; }',
                   OrderedDict([('B', self), ('C', other)]), ['A'])

    def __rtruediv__(self, other):
        return _Op('/', self.dtype,
                   _broadcast_shape(self, other), 'function (B, C) -> (A) { A = B / C; }',
                   OrderedDict([('B', other), ('C', self)]), ['A'])

    def __lt__(self, other):
        return _Op('<', self.dtype,
                   _broadcast_shape(self, other), 'function (B, C) -> (A) { A = cmp_lt(B, C); }',
                   OrderedDict([('B', self), ('C', other)]), ['A'])

    def __le__(self, other):
        return _Op('<=', self.dtype,
                   _broadcast_shape(self, other), 'function (B, C) -> (A) { A = cmp_le(B, C); }',
                   OrderedDict([('B', self), ('C', other)]), ['A'])

    def __gt__(self, other):
        return _Op('>', self.dtype,
                   _broadcast_shape(self, other), 'function (B, C) -> (A) { A = cmp_gt(B, C); }',
                   OrderedDict([('B', self), ('C', other)]), ['A'])

    def __ge__(self, other):
        return _Op('>=', self.dtype,
                   _broadcast_shape(self, other), 'function (B, C) -> (A) { A = cmp_ge(B, C); }',
                   OrderedDict([('B', self), ('C', other)]), ['A'])

    def batch_flatten(self):
        # Flatten all but first dimension to a single dimension; leave 1st dimension unchanged
        # Note this is a specific kind of reshape that serves a special role in Keras (for Flatten layers)
        if self.ndim < 2:
            raise Exception("batch_flatten called on tensor with ndim < 2")

        in_dim_list = ["N{}".format(i) for i in range(self.ndim)]
        out_dim_list = ["N0", "*".join(["N{}".format(i) for i in range(1, self.ndim)])]
        rest_shape = 1
        for i in range(1, self.ndim):
            if rest_shape is not None:
                if self.shape[i] is not None:
                    rest_shape *= self.shape[i]
                else:
                    rest_shape = None

        py_shape = [self.shape[0], rest_shape]

        code = ('function (I[{idims}]) -> (O) {{\n' + '  O = reshape(I, {odims});\n'
                '}}').format(
                    idims=", ".join(in_dim_list), odims=", ".join(out_dim_list))
        return _Op('batch_flatten', self.dtype, py_shape, code, {'I': self}, ['O'])

    def flatten(self):
        in_dim_list = ["N{}".format(i) for i in range(self.ndim)]
        out_dim_list = ["*".join(["N{}".format(i) for i in range(self.ndim)])]
        py_out_dim = 1
        for i in range(self.ndim):
            if self.shape[i] is not None:
                py_out_dim *= self.shape[i]
            else:
                py_out_dim = None
                break
        code = 'function (I[{idims}]) -> (O) {{\n  O = reshape(I, {odims});\n}}'.format(
                idims=", ".join(in_dim_list), odims=", ".join(out_dim_list))
        return _Op('flatten', self.dtype, [py_out_dim], code, {'I': self}, ['O'])

    def reshape(self, shape):
        in_dim_list = ["N{}".format(i) for i in range(self.ndim)]

        # Fill in Nones in target shape
        o_shape = [self.shape[i] if shape[i] is None else shape[i] for i in range(len(shape))]
        o_shape = ["N{}".format(i) if shape[i] is None else shape[i] for i in range(len(shape))]

        # Replace -1 in target shape
        Neg1Idx = None
        for i in range(len(o_shape)):
            if o_shape[i] == -1:
                if Neg1Idx is not None:
                    raise ValueError(("Cannot infer multiple dimensions in reshape. " +
                                      "Requested {} -> {}").format(self.shape, shape))
                else:
                    Neg1Idx = i
        if Neg1Idx is None:
            o_shape_sz = "*".join([str(x) for x in o_shape])
        else:
            o_shape_sz = "*".join([str(x) for x in o_shape[:Neg1Idx] + o_shape[Neg1Idx + 1:]])
            if o_shape_sz == '':  #Handle case shape == (-1,)
                o_shape_sz = '1'
            o_shape[Neg1Idx] = "{ishsz}/({oshsz})".format(
                ishsz="*".join(in_dim_list), oshsz=o_shape_sz)

        # Write python shape with dims computed at bind time replaced with 'None'
        py_shape = tuple(x if isinstance(x, int) else None for x in o_shape)
        o_shape = [str(x) for x in o_shape]

        code = ('function (I[{idims}]) -> (O) {{\n' + '  O = reshape(I, {odims});\n'
                '}}').format(
                    idims=", ".join(in_dim_list), odims=", ".join(o_shape))
        return _Op('reshape', self.dtype, py_shape, code, {'I': self}, ['O'])

    def prod(self, axis=None, dtype=None, keepdims=False, acc_dtype=None):
        if self.dtype == 'bool':
            self = cast(self, floatx())

        if not len(self.shape):
            return self

        if isinstance(axis, tuple):
            axis = list(axis)

        if axis == None:
            axis = list(range(self.ndim))

        if isinstance(axis, list) and not len(axis):
            # We're taking the product across an empty axis list.
            return self

        shape, _, subs = self._compute_agg_axes(axis, keepdims)

        f = """function (I[%(src_ranges)s]) -> (O) {
                   O[%(dest_indices)s%(dest_sep)s%(dest_ranges)s] = *(I[%(src_indices)s]);
               }""" % subs

        return _Op('prod', self.dtype, shape, f, {'I': self}, ['O'])

    def sum(self, axis=None, dtype=None, keepdims=False, acc_dtype=None):
        if self.dtype == 'bool':
            self = cast(self, floatx())

        if not len(self.shape):
            return self

        if isinstance(axis, tuple):
            axis = list(axis)

        if axis == None:
            axis = list(range(self.ndim))

        if isinstance(axis, list) and not len(axis):
            # We're taking the sum across an empty axis list.
            return self

        shape, _, subs = self._compute_agg_axes(axis, keepdims)

        f = """function (I[%(src_ranges)s]) -> (O) {
                   O[%(dest_indices)s%(dest_sep)s%(dest_ranges)s] = +(I[%(src_indices)s]);
               }""" % subs

        return _Op('sum', self.dtype, shape, f, {'I': self}, ['O'])

    def cumsum(self, axis=0):
        ranges = ", ".join(["N{}".format(n) for n in range(self.ndim)])
        dest_idxs = ", ".join(["i{}".format(n) for n in range(self.ndim)])
        src_idxs = ["i{}".format(n) for n in range(self.ndim)]
        src_idxs[axis] += " - k"
        src_idxs = ", ".join(src_idxs)
        f = ("""function (I[{src_ranges}]) -> (O) {{
                    O[{dest_idxs}: {dest_ranges}] = +(I[{src_idxs}]), k < N{ax};
                }}""").format(
            src_ranges=ranges, dest_idxs=dest_idxs, dest_ranges=ranges, src_idxs=src_idxs, ax=axis)
        return _Op('cumsum', self.dtype, self.shape, f, {'I': self}, ['O'])

    def max(self, axis=None, keepdims=False):
        if not len(self.shape):
            return self

        if axis == None:
            axis = list(range(self.ndim))

        if isinstance(axis, list) and not len(axis):
            # Do nothing if max'ing over empty axis list
            return self

        shape, axis, subs = self._compute_agg_axes(axis, keepdims)

        f = """function (I[{src_ranges}]) -> (O) {{
                   O[{dest_indices}{dest_sep}{dest_ranges}] = >(I[{src_indices}]);
               }}""".format(**subs)

        return _Op('max', self.dtype, shape, f, {'I': self}, ['O'])

    def mean(self, axis=None, keepdims=False):
        if self.dtype == 'bool':
            self = cast(self, floatx())

        if not len(self.shape):
            return self

        if axis == None:
            axis = list(range(self.ndim))

        if isinstance(axis, list) and not len(axis):
            # We're taking the mean across an empty axis list.
            # Keras sometimes does this when squeezing a matrix that doesn't need
            # to be squeezed.
            return self

        shape, axis, subs = self._compute_agg_axes(axis, keepdims)

        subs['mean_ranges'] = '*'.join(['X' + str(i) for i in axis])

        f = """function (I[%(src_ranges)s]) -> (O) {
                   SO[%(dest_indices)s%(dest_sep)s%(dest_ranges)s] = +(I[%(src_indices)s]);
                   O = SO / (%(mean_ranges)s);
               }""" % subs

        return _Op('mean', self.dtype, shape, f, {'I': self}, ['O'])

    def min(self, axis=None, keepdims=False):
        if not len(self.shape):
            return self

        if axis == None:
            axis = list(range(self.ndim))

        if isinstance(axis, list) and not len(axis):
            # Do nothing if min'ing over empty axis list
            return self

        shape, axis, subs = self._compute_agg_axes(axis, keepdims)

        f = """function (I[{src_ranges}]) -> (O) {{
                   O[{dest_indices}{dest_sep}{dest_ranges}] = <(I[{src_indices}]);
               }}""".format(**subs)

        return _Op('min', self.dtype, shape, f, {'I': self}, ['O'])

    def var(self, axis=None, keepdims=False):
        # This closely follows the implementation of the mean method
        # This computes the *uncorrected* sample variance (i.e. denominator
        # = n rather than = n-1) to match tensorflow
        if self.dtype == 'bool':
            self = cast(self, floatx())

        if not len(self.shape):
            return self

        if axis == None:
            axis = list(range(self.ndim))

        shape, axis, subs = self._compute_agg_axes(axis, keepdims)

        subs['prod_src_ranges'] = '*'.join(['X' + str(i) for i in axis])
        subs['mean_ranges'] = ', '.join(['Y' + str(i) for i in range(len(self._keras_shape))])

        mean = self.mean(axis, True)

        # TODO: Might be possible to write this more efficiently
        f = """function (I[%(src_ranges)s], M[%(mean_ranges)s]) -> (O) {
                   DIFF_SQ = (I - M) * (I - M);
                   SUM[%(dest_indices)s%(dest_sep)s%(dest_ranges)s] = +(DIFF_SQ[%(src_indices)s]);
                   O = SUM / (%(prod_src_ranges)s);
               }""" % subs

        return _Op('var', self.dtype, shape, f, {'I': self, 'M': mean}, ['O'])

    def ismax(self, axis=None):
        if not len(self.shape):
            return self

        shape, axis, subs = self._compute_agg_axes(axis, True)

        f = """function (I[%(src_ranges)s]) -> (O) {
                   MAX[%(dest_indices)s%(dest_sep)s%(dest_ranges)s] = >(I[%(src_indices)s]);
                   O = (MAX == I);
               }""" % subs

        return _Op('ismax', 'bool', shape, f, {'I': self}, ['O'])

    def clip(self, min_val, max_val):
        # TODO: Use clamp(), once we can take the derivative of clamp.
        f = """function (I, MIN_VAL, MAX_VAL) -> (O) {
                   M = (I < MAX_VAL ? I : MAX_VAL);
                   O = (MIN_VAL < M ? M : MIN_VAL);
               }"""

        return _Op('clip', self.dtype, self.shape, f,
                   OrderedDict([('I', self), ('MIN_VAL', min_val), ('MAX_VAL', max_val)]), ['O'])

    def set_value(self, value):
        raise PlaidMLKerasException("Can only set values of tensors")


class _Tensor(_Var):

    def __init__(self, ident, dtype, shape, name, value, src=None):
        super(_Tensor, self).__init__(
            ident, dtype, shape, name, value, src=src, is_keras_tensor=True)

    def set_value(self, value):
        try:
            if tuple(self.shape) != value.shape:
                raise NotImplementedError(
                    "The PlaidML backend for Keras does not support changing tensor shapes with set_value.\n"
                    + "existing.shape = " + str(self.shape) + ", value.shape = " + str(
                        value.shape))
        except AttributeError:
            if tuple(self.shape) != () and tuple(self.shape) != (1,):
                raise NotImplementedError(
                    "The PlaidML backend for Keras does not support changing tensor shapes with set_value.\n"
                    + "existing.shape = " + str(
                        self.shape) + ", value is a non-array object of type: " + str(type(value)))
        with self._value.mmap_discard(_ctx) as view:
            view.copy_from_ndarray(np.asarray(value))
            view.writeback()

    def eval(self):
        out = np.ndarray(tuple(dim for dim in self.shape), dtype=self.dtype)
        with self._value.mmap_current() as view:
            view.copy_to_ndarray(out)
        return out


class _Op(_Var):
    """A composite variable returned to Keras by the PlaidML backend.

    Keras invokes a series of backend APIs to describe a computation, building up
    values bottom-up.  The PlaidML backend implements these by returning _Op,
    describing a tree of the operations to be performed.  When Keras requests a
    concrete value, the PlaidML backend translates this
    operation tree into PlaidML code, and returns the bound PlaidML function; when Keras
    requests evaluation, the bound PlaidML function is evaluated.

    The reason for building up a tree of operations (vs. calling PlaidML for each
    individual operation as it's being built) is to allow the backend to coalesce
    operations whose individual components are costly to describe directly in
    PlaidML.  As an analogy, imagine having a pluggable arithmetic backend that only
    supports addition, and imagine a framework that invokes the addition operator
    some number of times in order to implement a multiplication operation;
    recording the tree of operations would allow the backend to pattern-match
    against the requested operations and coalesce them into a single
    multiplication step.

    _Op objects have an identifier, dtype, shape, a dict of inputs, and a list
    of output identifiers (typically only one).

    Since most operations translate directly to Tile code, _Op objects also
    typically include the Tile code for their particular operations.  This
    is done to keep per-operation logic together when possible; the only bits that
    are handled seperately are the multiple-operation-coalescing optimizations.
    """

    def __init__(self, ident, dtype, shape, code, inputs, outputs, side_effects=None):
        super(_Op, self).__init__(ident, dtype, shape)
        self._code = code
        self._inputs = inputs
        self._outputs = outputs
        self._value = None
        self._dtype = dtype
        self._self_side_effects = side_effects
        self._cached_side_effects = None
        self._backtrace = "".join(traceback.format_stack()[:-1])
        if not self._code:
            self._trace = traceback.extract_stack()[:-2]

    def traceback(self, indent=0, depth=5):
        ret = ""
        ret += "{}{}\n".format('  ' * indent, 'outputs: {}'.format(','.join(self._outputs)))
        ret += "{}{}\n".format('  ' * indent, 'inputs:')
        indent += 1
        if indent < depth:
            for n, inp in self._inputs.items():
                ret += "{}{}\n".format('  ' * indent, "{}: {}".format(n, inp))
                if isinstance(inp, _Op):
                    ret += inp.traceback(indent, depth)
                else:
                    ret += "{}{}\n".format('  ' * indent, inp)
        else:
            ret += "{}{}\n".format('  ' * indent, '...truncated...')
        return ret

    def __repr__(self):
        return "_Op({})".format(self._ident)

    def __str__(self):
        return "_Op({})".format(self._ident)

    def _plaidml_val(self, indent=0, path=''):
        if self._value is None:
            try:
                if not self._code:
                    exn = PlaidMLKerasException(
                        'unable to construct value for operation \'%s\' at:\n%s' %
                        (self.ident, ''.join(traceback.format_list(self._trace))))
                    raise exn
                a = plaidml.Applier(_ctx, plaidml.Function(self._code, self._backtrace))
                for k, v in iteritems(self._inputs):
                    a.add_input(k, _plaidml_val(v, indent + 1))
                self._value = a.add_output(self._outputs[0])
            except plaidml.exceptions.PlaidMLError as e:
                raise PlaidMLKerasException("{}\nTraceback:\n{}\n{}".format(
                    str(e), self, self.traceback()))
        return self._value

    def _side_effects(self):
        if self._cached_side_effects is not None:
            return self._cached_side_effects
        self._cached_side_effects = {}
        if self._self_side_effects is not None:
            self._cached_side_effects = self._self_side_effects
        for ki, vi in iteritems(self._inputs):
            if not isinstance(vi, _Var):
                continue
            inner_effects = vi._side_effects()
            for k, v in iteritems(inner_effects):
                self._cached_side_effects[k] = v
        return self._cached_side_effects

    def eval(self):
        # Get my value:
        sv = _plaidml_val(self)
        # Flatten to a 0 input, 1 output function
        c = plaidml.Composer()
        c.add_output("eval_out", sv)
        func = c.build()
        # Add any side effects
        for (var, newval) in iteritems(self._side_effects()):
            c.add_update(var, newval)

        # Now build an invoker for that function
        i = plaidml.Invoker(_ctx, func)

        # Now request the output shape
        shape = i.get_output_shape("eval_out")

        # Allocate room for the output
        tensor = plaidml.Tensor(_device(), shape)

        # Now, add the output
        i.set_output("eval_out", tensor)

        # Invoke the function, updating the output tensor
        i.invoke()

        # Make ndarrays for the output
        out = np.ndarray(tuple(dim.size for dim in tensor.shape.dimensions), dtype=self._dtype)

        # Copy the data
        with tensor.mmap_current() as view:
            view.copy_to_ndarray(out)

        # Return a result
        return out


class _Function(object):

    def __init__(self, inputs, outputs, updates, name):
        # Inputs: a list of placeholders
        # Outputs: a list of ops
        # Updates: a list of (var, newval) tuples
        # Name: a string, which we ignore
        self._input_names = ['I' + str(n) for n in range(len(inputs))]
        self._output_names = ['O' + str(n) for n in range(len(outputs))]
        self._name = name
        self._input_types = dict()  # Will be filled with dtype of each input

        c = plaidml.Composer()
        for (name, val) in zip(self._input_names, inputs):
            if isinstance(_plaidml_val(val), plaidml._Var):
                if isinstance(_plaidml_val(val), plaidml.Placeholder):
                    c.add_input(name, _plaidml_val(val))
                    self._input_types[name] = val.dtype
                else:
                    raise RuntimeError("_Function given unexpected input type {}".format(
                        type(val)))
            else:
                raise RuntimeError("_Function given unexpected input type {}".format(type(val)))
        for (name, val) in zip(self._output_names, outputs):
            c.add_output(name, _plaidml_val(val))
        for (var, newval) in updates:
            c.add_update(_plaidml_val(var), _plaidml_val(newval))
        side_effects = {}
        for o in outputs:
            inner_effects = o._side_effects()
            for k, v in iteritems(inner_effects):
                side_effects[k] = v
        for (var, newval) in iteritems(side_effects):
            c.add_update(var, newval)

        self._func = c.build()
        self._invoker = plaidml.Invoker(_ctx, self._func)

    def __call__(self, inputs):
        # Inputs: a list of bindings for the placeholders.

        for (name, val) in zip(self._input_names, inputs):
            if isinstance(val, six.integer_types):
                val = plaidml.Integer(val)
            elif isinstance(val, float):
                val = plaidml.Real(val)
            else:
                val = variable(val, dtype=self._input_types[name])._plaidml_val()
            self._invoker.set_input(name, val)

        tensors = [
            plaidml.Tensor(_device(), self._invoker.get_output_shape(name))
            for name in self._output_names
        ]

        for (name, t) in zip(self._output_names, tensors):
            self._invoker.set_output(name, t)

        self._invoker.invoke()

        return [t.as_ndarray(_ctx) for t in tensors]


_k_rng_size = 2048


def _make_rng_state(seed=None):
    if seed:
        np.random.seed(seed)

    rng_init = np.empty((3, _k_rng_size), dtype=np.uint32)
    rng_init[0] = np.random.randint(1, 2**32, (_k_rng_size,), dtype=np.uint32)
    rng_init[1] = np.random.randint(7, 2**32, (_k_rng_size,), dtype=np.uint32)
    rng_init[2] = np.random.randint(15, 2**32, (_k_rng_size,), dtype=np.uint32)
    rng_state = variable(rng_init, dtype='uint32')

    return rng_state


def abs(x):
    f = """function (I) -> (O) { O = abs(I); }"""
    return _Op('abs', x.dtype, x.shape, f, {'I': x}, ['O'])


def all(x, axis=None, keepdims=False):
    _report_unimplemented('all')


def any(x, axis=None, keepdims=False):
    _report_unimplemented('any')


def arange(start, stop=None, step=1, dtype='int32'):
    _report_unimplemented('arange')


def argmax(x, axis=-1):
    return _Op('argmax', x.dtype, x.shape, None, OrderedDict([('I', x), ('axis', axis)]), ['O'])


def argmin(x, axis=-1):
    # Do argmin(x) by computing argmax(-x)
    return _Op('argmax', x.dtype, x.shape, None, OrderedDict([('I', -x), ('axis', axis)]), ['O'])


def backend():
    return 'plaidml'


def batch_dot(x, y, axes=None):
    if isinstance(axes, int):
        axes = (axes, axes)
    if axes is None:
        axes = (x.ndim - 1, y.ndim - 2)
    out_shape = x.shape[:axes[0]] + x.shape[axes[0] + 1:] + y.shape[1:axes[1]] + y.shape[axes[1] +
                                                                                         1:]
    if out_shape[0] is None:  # can infer batch size from either x or y
        out_shape = (y.shape[0],) + out_shape[1:]

    xdim_list = ["M{}".format(i) for i in range(x.ndim)]
    xdim_list[0] = "B"
    xdim_list[axes[0]] = "D"
    ydim_list = ["N{}".format(i) for i in range(y.ndim)]
    ydim_list[0] = "B"
    ydim_list[axes[1]] = "D"
    odim_list = [N for N in xdim_list if N != "D"] + [N for N in ydim_list[1:] if N != "D"]
    xidx_list = [N.lower() for N in xdim_list]
    yidx_list = [N.lower() for N in ydim_list]
    oidx_list = [N.lower() for N in odim_list]
    # Example
    # function (X[B, M1, M2, M3, D], Y[B, N1, D, N3]) -> (O) {
    #   O[b, m1, m2, m3, n1, n3: B, M1, M2, M3, N1, N3] = +(X[b, m1, m2, m3, d] * Y[b, n1, d, n3]);
    # }
    f = ("function (X[{xdims}], Y[{ydims}]) -> (O) {{\n" +
         "  O[{oidxs}: {odims}] = +(X[{xidxs}] * Y[{yidxs}]);\n" + "}}").format(
             xdims=", ".join(xdim_list),
             ydims=", ".join(ydim_list),
             odims=", ".join(odim_list),
             xidxs=", ".join(xidx_list),
             yidxs=", ".join(yidx_list),
             oidxs=", ".join(oidx_list))
    ret = _Op('batch_dot', x.dtype, out_shape, f, OrderedDict([('X', x), ('Y', y)]), ['O'])
    if len(out_shape) == 1:
        ret = expand_dims(ret, 1)
    return ret


def batch_flatten(x):
    return x.batch_flatten()


def batch_set_value(tuples):
    for pair in tuples:
        set_value(pair[0], pair[1])


def batch_get_value(xs):
    return [get_value(x) for x in xs]


def batch_normalization(x, mean, var, beta, gamma, epsilon=1e-3):
    # gamma == scale
    # beta == offset
    denom = sqrt(var + epsilon)
    if gamma is not None and beta is not None:
        return ((x - mean) * gamma / denom) + beta
    elif gamma is not None:
        return ((x - mean) * gamma / denom)
    elif beta is not None:
        return ((x - mean) / denom) + beta
    else:
        return ((x - mean) / denom)


def bias_add(x, bias, data_format=None):
    if data_format is None:
        data_format = image_data_format()
    if data_format not in {'channels_first', 'channels_last'}:
        raise PlaidMLKerasException("Unrecognized data_format given to bias_add: '" + str(
            data_format) + "'; only 'channels_first' and 'channels_last' recognized.")
    if ndim(x) > 2:
        if data_format == 'channels_first':
            x += reshape(bias, (1, bias.shape[0]) + (1,) * (ndim(x) - 2))
        elif data_format == 'channels_last':
            x += reshape(bias, (1,) * (ndim(x) - 1) + (bias.shape[0],))
    else:
        x += bias
    return x


def binary_crossentropy(target, output, from_logits=False):
    if from_logits:
        output = sigmoid(output)
    output = output.clip(epsilon(), 1.0 - epsilon())
    input_sizes = ",".join(["I" + str(i) for i in range(ndim(output))])
    input_sizes_prod = "*".join(["I" + str(i) for i in range(ndim(output))])
    f = """function (O[{dims}], T[{dims}]) -> (R) {{
               R = builtin_binary_crossentropy(O,T,{prod});
           }}""".format(
        dims=input_sizes, prod=input_sizes_prod)
    return _Op('binary_crossentropy', output.dtype, output.shape, f,
               OrderedDict([('O', output), ('T', target)]), ['R'])


def cast(x, dtype):
    # Not clear what datatypes Keras supports.
    # Each backend appears to implement support for its own subset of some assumed
    # but undocumented pool of possible numeric types. Perhaps this pool may be
    # the array element scalar type names defined by Numpy?
    # Tensorflow supports:
    #  float16, float32, float64, int16, int32, int64, uint8, uint16
    # Scipy offers
    # Not sure where "bool" comes from; scipy uses "bool_" and "bool8".

    if x.dtype == dtype:
        return x

    basetype = 'None'
    bitwidth = '0'
    if dtype == 'float16':
        basetype = 'float'
        bitwidth = '16'
    elif dtype == 'float32':
        basetype = 'float'
        bitwidth = '32'
    elif dtype == 'float64':
        basetype = 'float'
        bitwidth = '64'
    elif dtype == 'int8':
        basetype = 'int'
        bitwidth = '8'
    elif dtype == 'int16':
        basetype = 'int'
        bitwidth = '16'
    elif dtype == 'int32':
        basetype = 'int'
        bitwidth = '32'
    elif dtype == 'int64':
        basetype = 'int'
        bitwidth = '64'
    elif dtype == 'uint8':
        basetype = 'uint'
        bitwidth = '8'
    elif dtype == 'uint16':
        basetype = 'uint'
        bitwidth = '16'
    elif dtype == 'uint32':
        basetype = 'uint'
        bitwidth = '32'
    elif dtype == 'uint64':
        basetype = 'uint'
        bitwidth = '64'
    else:
        raise PlaidMLKerasException('Unsupported cast (%s -> %s)' % (x.dtype, dtype))

    return _Op('cast_as_' + dtype, dtype, x.shape,
               'function (I) -> (O) { O = as_' + basetype + '(I, ' + bitwidth + '); }', {'I': x},
               ['O'])


def categorical_crossentropy(target, output, from_logits=False):
    if from_logits:
        output = softmax(output)
    elif not isinstance(output, _Op) or output._ident != "softmax":
        output /= output.sum(axis=-1, keepdims=True)
        output = output.clip(epsilon(), 1.0 - epsilon())
    if output.ndim == 1:
        f = """function (O[Y], T[Y]) -> (R) {
                   LO = log(O);
                   TR[] = +(T[y] * LO[y]);
                   R = -TR;
               }"""
    else:
        fixed_dims = ",".join("X{}".format(i) for i in range(output.ndim - 1))
        fixed_idxs = ",".join("x{}".format(i) for i in range(output.ndim - 1))
        f = """function (O[{fixed_dims},Y], T[{fixed_dims},Y]) -> (R) {{
                   LO = log(O);
                   TR[{fixed_idxs}:{fixed_dims}] = +(T[{fixed_idxs},y] * LO[{fixed_idxs},y]);
                   R = -TR;
               }}""".format(
            fixed_dims=fixed_dims, fixed_idxs=fixed_idxs)
    return _Op('categorical_crossentropy', output.dtype, output.shape[:-1], f,
               OrderedDict([('O', output), ('T', target)]), ['R'])


def ceil(x):
    f = """function (I) -> (O) { O = ceil(I); }"""
    return _Op('ceil', x.dtype, x.shape, f, {'I': x}, ['O'])


def clear_session():
    _report_unimplemented('clear_session')


def clip(x, min_value, max_value):
    return x.clip(min_value, max_value)


def concatenate(tensors, axis=-1):
    rank = ndim(tensors[0])
    if axis >= rank or axis < -rank:
        raise ValueError("Cannot concatenate tensors with {} dimensions along axis {}".format(
            rank, axis))
    elif axis < 0:
        axis = axis % rank

    def __clear_axis(shape):
        return [shape[i] for i in range(len(shape)) if i != axis]

    shape_template = __clear_axis(tensors[0].shape)
    for t in tensors:
        if __clear_axis(t.shape) != shape_template:
            raise ValueError(
                "Incompatible shapes: cannot concatenate along axis {}\n{} v {}".format(
                    axis, tensors[0].shape, t.shape))

    offsets = [0]
    for i in range(len(tensors)):
        offsets.append(offsets[i] + tensors[i].shape[axis])
    out_shape = tuple(
        tensors[0].shape[i] if i != axis else offsets[len(tensors)] for i in range(rank))

    output_dims_list = ["N{}".format(i) for i in range(rank)]
    output_dims_list[axis] = offsets[len(tensors)]
    output_dims_str = ', '.join([str(i) for i in output_dims_list])
    # output_dims_list also serves as a base for input dims,
    # with `axis` index to be overwritten by "Ai" (i = input index)
    inputs_list = list()
    for i in range(len(tensors)):
        curr_input_dims = list(output_dims_list)  # using 'list' here to make a copy
        curr_input_dims[axis] = "A{}".format(i)
        inputs_list.append("I{}[{}]".format(i, ', '.join(curr_input_dims)))
    inputs_str = ', '.join(inputs_list)

    if axis == 0:
        indices_begin = "a"
    else:
        indices_begin = ', '.join(["n{}".format(i) for i in range(axis)]) + ", a"
    if axis == rank - 1:
        indices_end = ""
    else:
        indices_end = ", " + ', '.join(["n{}".format(i) for i in range(axis + 1, rank)])

    body_str = ""
    line_subs = {'beg': indices_begin, 'end': indices_end, 'odims': output_dims_str}
    for i in range(len(tensors)):
        line_subs['off'] = "+{}".format(offsets[i])
        line_subs['i'] = i
        curr_line = "  T{i}[{beg}{off}{end}: {odims}] = =(I{i}[{beg}{end}]);\n".format(**line_subs)
        body_str += curr_line
    body_str += "O = "
    body_str += " + ".join(["T{}".format(i) for i in range(len(tensors))])
    body_str += ";"

    # Example 'code' (concatenating (4,3,2), (4,5,2), (4,1,2)):
    #   function (I0[N0, A0, N2], I1[N0, A1, N2], I2[N0, A2, N2]) -> (O) {
    #     T0[n0, a, n2: N0, 9, N2] = =(I0[n0, a, n2]);
    #     T1[n0, a+3, n2: N0, 9, N2] = =(I1[n0, a, n2]);
    #     T2[n0, a+8, n2: N0, 9, N2] = =(I2[n0, a, n2]);
    #     O = T0 + T1 + T2;
    #   }
    code = ('function ({inputs}) -> (O) {{\n{body}\n}}').format(
        inputs=inputs_str,
        body=body_str,
    )
    inputs_dict = dict()
    for i in range(len(tensors)):
        inputs_dict['I{}'.format(i)] = tensors[i]
    return _Op('concatenate', tensors[0].dtype, out_shape, code, inputs_dict, ['O'])


def constant(value, dtype=None, shape=None, name=None):
    # Enforce sensible defaults if given None
    dtype = dtype or floatx()
    shape = shape or (1,)
    np_value = value * np.ones(shape)
    return variable(np_value, dtype=dtype, name=_prepend_name_scope(name, "constant"))


def cos(x):
    f = "function (I) -> (O) { O = cos(I); }"
    return _Op('cos', x.dtype, x.shape, f, {'I': x}, ['O'])


# Logic comes for apeing TF, which I think is a poor choice but good for compatibility
def pad_compute(sym, input_size, filter_size, stride, padding):
    if padding == 'valid':
        if input_size is None or isinstance(input_size, _Op):
            num_out_size = None
        else:
            num_out_size = int((input_size - filter_size + stride) // stride)
        sym_output_size = "({sym} - {fs} + {s}) / {s}".format(sym=sym, fs=filter_size, s=stride)
        sym_padding_before = 0
    elif padding == 'same':
        if input_size is None or isinstance(input_size, _Op):
            num_out_size = None
        else:
            num_out_size = int((input_size + stride - 1) // stride)
        sym_output_size = "({sym} + {s} - 1) / {s}".format(sym=sym, s=stride)
        sym_padding_before = "(max(0, ({symout} - 1) * {s} + {fs} - {syminp})) / 2".format(
            symout=sym_output_size, s=stride, fs=filter_size, syminp=sym)
    else:
        raise Exception("Invalid padding")
    if num_out_size != None and num_out_size < 0:
        raise Exception(
            "Invalid output size computed for convolution: num_out_size={}".format(num_out_size))
    return (sym_output_size, sym_padding_before, num_out_size)


def spatial_2d_padding(x, padding=((1, 1), (1, 1)), data_format=None):
    if data_format is None:
        data_format = image_data_format()
    if data_format != 'channels_last':
        raise "Not supported"

    yFront = padding[0][0]
    yTotal = padding[0][0] + padding[0][1]
    xFront = padding[1][0]
    xTotal = padding[1][0] + padding[1][1]
    f = ("""
        function (I[N, H, W, C]) -> (O) {{
            O[n, y, x, c : N, H + {yTotal}, W + {xTotal}, C] = =(I[n, y - {yFront}, x - {xFront}, c]);
        }}
    """).format(
        yFront=yFront, yTotal=yTotal, xFront=xFront, xTotal=xTotal)

    # TODO: reorder output dimensions in theano ordering case
    outshape = (x.shape[0], x.shape[1] + yTotal, x.shape[2] + xTotal, x.shape[3])

    return _Op('spatial_2d_padding', x.dtype, outshape, f, OrderedDict([('I', x)]), ['O'])


def _format_conv_strings(rank,
                         in_shape,
                         kernel_shape,
                         strides,
                         padding,
                         data_format,
                         dilation_rate,
                         channelwise,
                         forward=True,
                         expected_output_shape=None,
                         ):
    # Variable meanings:
    # N: Number of items in the batch
    # L<i>: Spatial dimension i of each (input) item
    # CI: Number of channels (aka filters) of each input item
    # LK<i>: Spatial dimension i of kernel
    # CO: Number of channels (aka filters) of each output item
    # C: Number of input channels in channelwise convolutions
    # M: Channel multiplier in channelwise convolutions (each input channel yields
    #     M output channels for such convolutions)
    #
    # n: Which element of the batch we're on
    # x<i>: The ith coordinate in the output/image
    # k<i>: The ith coordinate in the kernel
    # ci: The input channel we're on
    # co: The output channel we're on
    # c: The input channel we're on for channelwise convolutions
    # m: The output channel multiplier we're on for output convolutions
    if data_format == 'channels_first':
        n = 0
        c = 1
        l = [i + 2 for i in range(rank)]
    elif data_format == 'channels_last':
        n = 0
        l = [i + 1 for i in range(rank)]
        c = rank + 1
    else:
        raise ValueError("Unrecognized data format '{}'".format(data_format))
    if channelwise == True and in_shape[c] != kernel_shape[-2]:
        raise ValueError(
            "Channelwise convolution must have same number of channels in both input and kernel:\n"
            + "{} (from shape {}) v {} (from shape {})".format(in_shape[c], in_shape,
                                                               kernel_shape[-2], kernel_shape))
    sym_out_shape = list()
    pad_amount = list()
    num_out_shape = list()
    for i in range(rank):
        if forward:
            sym_out, sym_pad, num_out = pad_compute("L{}".format(i), in_shape[l[i]],
                                                    dilation_rate[i] * (kernel_shape[i] - 1) + 1,
                                                    strides[i], padding)
        else:
            sym_out, sym_pad, num_out = pad_compute("D{}".format(i), in_shape[l[i]],
                                                    dilation_rate[i] * (kernel_shape[i] - 1) + 1,
                                                    strides[i], padding)
        sym_out_shape.append(sym_out)
        pad_amount.append(sym_pad)
        num_out_shape.append(num_out)
    if expected_output_shape is not None:
        # Confirm that the output shape is consistent with the rest of the convolution
        computed_output_shape = [0] * (rank + 2)
        computed_output_shape[n] = in_shape[n]
        computed_output_shape[c] = kernel_shape[-1]
        for i in range(rank):
            computed_output_shape[l[i]] = num_out_shape[i]
        for i in range(rank + 2):
            if computed_output_shape[i] is not None and not isinstance(computed_output_shape[i], _Var) and \
                    expected_output_shape[i] is not None and computed_output_shape[i] != expected_output_shape[i]:
                raise ValueError("Expected convolution output of shape {}, received {}".format(
                        expected_output_shape, computed_output_shape))
    padding_list = ["  Pad{} = {};".format(i, pad_amount[i]) for i in range(rank)]
    padding_str = "\n".join(padding_list)
    input_idx_list = [
        "{s}*{x} + {d}*{k} - {p}".format(
            s=strides[i],
            x="x{}".format(i),
            d="{}".format(dilation_rate[i]),
            k="k{}".format(i),
            p="Pad{}".format(i)) for i in range(rank)
    ]
    if data_format == 'channels_first' and not channelwise:
        if forward:
            input_dims_str = "N, CI, " + ", ".join(["L{}".format(i) for i in range(rank)])
            out_dims_str = "N, CO, " + ", ".join(["{}".format(sym_out_shape[i]) for i in range(rank)])
            outshape = [in_shape[0]] + [kernel_shape[-1]] + num_out_shape
        else:
            input_dims_str = "N, CI, " + ", ".join("D{}".format(i) for i in range(rank))
            out_dims_str = "N, CO, " + ", ".join(["L{}".format(i) for i in range(rank)])
        out_idx_str = "n, co, " + ", ".join(["x{}".format(i) for i in range(rank)])
        input_idx_str = "n, ci, " + ", ".join(input_idx_list)
    elif data_format == 'channels_last' and not channelwise:
        if forward:
            input_dims_str = "N, " + ", ".join(["L{}".format(i) for i in range(rank)]) + ", CI"
            out_dims_str = "N, " + ", ".join(["{}".format(sym_out_shape[i]) for i in range(rank)]) + ", CO"
            outshape = [in_shape[0]] + num_out_shape + [kernel_shape[-1]]
        else:
            input_dims_str = "N, " + ", ".join("D{}".format(i) for i in range(rank)) + ", CI"
            out_dims_str = "N, " + ", ".join(["L{}".format(i) for i in range(rank)]) + ", CO"
        out_idx_str = "n, " + ", ".join(["x{}".format(i) for i in range(rank)]) + ", co"
        input_idx_str = "n, " + ", ".join(input_idx_list) + ", ci"
    elif data_format == 'channels_first' and channelwise:
        if not forward:
            raise NotImplementedError("Channelwise transposed convolutions not implemented.")
        input_dims_str = "N, C, " + ", ".join(["L{}".format(i) for i in range(rank)])
        out_idx_str = "n, c*M + m, " + ", ".join(["x{}".format(i) for i in range(rank)])
        out_dims_str = "N, C*M, " + ", ".join(["{}".format(sym_out_shape[i]) for i in range(rank)])
        input_idx_str = "n, c, " + ", ".join(input_idx_list)
        outshape = [in_shape[0]] + [kernel_shape[-2] * kernel_shape[-1]] + num_out_shape
    elif data_format == 'channels_last' and channelwise:
        if not forward:
            raise NotImplementedError("Channelwise transposed convolutions not implemented.")
        input_dims_str = "N, " + ", ".join(["L{}".format(i) for i in range(rank)]) + ", C"
        out_idx_str = "n, " + ", ".join(["x{}".format(i) for i in range(rank)]) + ", c*M + m"
        out_dims_str = "N, " + ", ".join(["{}".format(sym_out_shape[i]) for i in range(rank)]) + ", C*M"
        input_idx_str = "n, " + ", ".join(input_idx_list) + ", c"
        outshape = [in_shape[0]] + num_out_shape + [kernel_shape[-2] * kernel_shape[-1]]
    else:
        raise ValueError("Unrecognized data format '{}'".format(data_format))
    if channelwise:
        ker_dims_str = ", ".join(["LK{}".format(i) for i in range(rank)]) + ", C, M"
        ker_idx_str = ", ".join(["k{}".format(i) for i in range(rank)]) + ", c, m"
    else:
        ker_dims_str = ", ".join(["LK{}".format(i) for i in range(rank)]) + ", CI, CO"
        ker_idx_str = ", ".join(["k{}".format(i) for i in range(rank)]) + ", ci, co"
    ret = {'input_dims_str': input_dims_str,
            'ker_dims_str': ker_dims_str,
            'out_idx_str': out_idx_str,
            'out_dims_str': out_dims_str,
            'input_idx_str': input_idx_str,
            'ker_idx_str': ker_idx_str,
            'padding_str': padding_str}
    if forward:
        ret['outshape_tuple'] = outshape
    else:
        ret['dim_input'] = ', ' + ', '.join(['D{}'.format(i) for i in range(rank)])
    return ret


def conv(x,
         kernel,
         strides=None,
         padding='valid',
         data_format=None,
         dilation_rate=None,
         channelwise=False):
    rank = len(x.shape) - 2
    if strides is None:
        strides = tuple(1 for _ in range(rank))
    if data_format is None:
        data_format = image_data_format()
    if dilation_rate is None:
        dilation_rate = tuple(1 for _ in range(rank))

    for entry in dilation_rate:
        if not isinstance(entry, int) or entry <= 0:
            raise ValueError("Invalid dilation_rate: {}".format(dilation_rate))
    if kernel.ndim != rank + 2:
        raise ValueError("Convolution kernel shape inconsistent with input shape: " +
                         "{} (rank {}) v {} (rank {})".format(kernel.shape, kernel.ndim - 2,
                                                              x.shape, x.ndim - 2))
    if len(strides) != rank:
        raise ValueError("Convolution strides length inconsistent with input shape: " +
                         "{} (rank {}) v {} (rank {})".format(strides,
                                                              len(strides), x.shape, x.ndim - 2))
    if len(dilation_rate) != rank:
        raise ValueError("Convolution dilation_rate length inconsistent with input shape: " +
                         "{} (rank {}) v {} (rank {})".format(
                             dilation_rate, len(dilation_rate), x.shape, x.ndim - 2))

    conv_strs = _format_conv_strings(rank,
                                     x.shape,
                                     kernel.shape,
                                     strides,
                                     padding,
                                     data_format,
                                     dilation_rate,
                                     channelwise)
    outshape = conv_strs['outshape_tuple']

    f = ('function (I[{input_dims_str}], K[{ker_dims_str}]) ' + '-> (O) {{\n{padding_str}\n' +
         '  O[{out_idx_str} : {out_dims_str}]' +
         '= +(I[{input_idx_str}]*K[{ker_idx_str}]);\n}}').format(**conv_strs)
    name = "conv{}d".format(rank)
    return _Op(name, x.dtype, tuple(outshape), f, OrderedDict([('I', x), ('K', kernel)]), ['O'])


def conv_transpose(x, kernel, output_shape, strides, padding, data_format):
    rank = x.ndim - 2
    if data_format is None:
        data_format = image_data_format()
    if kernel.ndim != rank + 2:
        raise ValueError("Transpose convolution kernel shape inconsistent with input shape: " +
                         "{} (rank {}) v {} (rank {})".format(kernel.shape, kernel.ndim - 2,
                                                              x.shape, x.ndim - 2))
    if len(output_shape) != rank + 2:
        raise ValueError("Transpose convolution output_shape inconsistent with input shape: " +
                         "{} (rank {}) v {} (rank {})".format(output_shape, len(output_shape) - 2,
                                                              x.shape, x.ndim - 2))
    if len(strides) != rank:
        raise ValueError("Transpose convolution strides inconsistent with input shape: " +
                         "{} (rank {}) v {} (rank {})".format(strides, len(strides),
                                                              x.shape, x.ndim - 2))
    if x.shape[0] != output_shape[0] and x.shape[0] is not None \
            and output_shape[0] is not None and not isinstance(output_shape[0], _Var):
        raise ValueError("Transpose convolution batch size inconsistent between input " +
                         "and output: {} v {}".format(x.shape[0], output_shape[0]))

    conv_strs = _format_conv_strings(rank,
                                     output_shape,
                                     kernel.shape,
                                     strides,
                                     padding,
                                     data_format,
                                     (1,) * rank,
                                     False,
                                     False,
                                     x.shape)

    f = ('function (O[{out_dims_str}], K[{ker_dims_str}]{dim_input}) ' + '-> (I) {{\n{padding_str}\n' +
         '  I[{input_idx_str} : {input_dims_str}]' +
         '= +(O[{out_idx_str}]*K[{ker_idx_str}]);\n}}').format(**conv_strs)
    name = "conv{}d".format(rank)
    # Output shape may be dynamic, so pass its sizes as inputs to Tile
    if data_format == 'channels_first':
        l = [i + 2 for i in range(rank)]
    elif data_format == 'channels_last':
        l = [i + 1 for i in range(rank)]
    else:
        raise ValueError("Unrecognized data format '{}'".format(data_format))
    input_tensors = OrderedDict([('O', x), ('K', kernel)])
    for i in range(rank):
        input_tensors['D{}'.format(i)] = output_shape[l[i]]
    # If output shape was dynamically calculated, pass through Keras size as None
    output_shape = list(output_shape)
    for i in range(rank + 2):
        if isinstance(output_shape[i], _Var):
            output_shape[i] = None
    output_shape = tuple(output_shape)
    return _Op(name, x.dtype, tuple(output_shape), f, input_tensors, ['I'])


def _func_once(func):
    "A decorator that runs a function only once."

    def decorated(*args, **kwargs):
        try:
            return decorated._once_result
        except AttributeError:
            decorated._once_result = func(*args, **kwargs)
            return decorated._once_result

    return decorated


@_func_once
def _compute_transforms(block, conv):
    # Returns (A, B, G)
    out = block - conv + 1
    if (out == 2 and conv == 3):
        A = constant([[1, 0], [1, 1], [1, -1], [0, -1]])
        B = constant([[1, 0, 0, 0], [0, 1, -1, 1], [-1, 1, 1, 0], [0, 0, 0, -1]])
        G = constant([[1, 0, 0], [.5, .5, .5], [.5, -.5, .5], [0, 0, 1]])
        return (A, B, G)
    if (out == 4 and conv == 3):
        #s2 = np.sqrt(2.0)
        #A = constant([[1., 0., 0., 0.], [1., s2/2., 1./2., s2/4.], [1, -s2/2., 1./2., -s2/4.],
        #              [1., s2, 2., 2.*s2], [1., -s2, 2., -2.*s2], [0., 0., 0., 1.]])
        #B = constant([[1., 0., 0., 0., 0., 0.], [0., -s2, s2, -s2/2., s2/2., 1], [-5./2., -2., -2., -1./2., -1./2., 0],
        #              [0., s2/2., -s2/2., s2, -s2, -5./2], [1., 1., 1., 1., 1., 0.], [0., 0., 0., 0., 0., 1.]])
        #G = constant([[1., 0., 0.], [-2./3., -s2/3., -1./3.], [-2./3., s2/3., -1./3.],
        #              [1./6., s2/6., 1./3.], [1./6., -s2/6., 1./3.], [0., 0., 1.]])
        #return (A, B, G)
        # yapf: disable
        A = np.array([
            [ 1.13777777777778,   0,                  0,                 0,                ],
            [-0.688403361344538, -0.430252100840336, -0.26890756302521, -0.168067226890756 ],
            [-0.688403361344538,  0.430252100840336, -0.26890756302521,  0.168067226890756 ],
            [ 0.119514472455649,  0.179271708683473,  0.26890756302521,  0.403361344537815 ],
            [ 0.119514472455649, -0.179271708683473,  0.26890756302521, -0.403361344537815 ],
            [ 0,                  0,                  0,                 1,                ]])
        B = np.array([
            [ 0.87890625,  0,          -2.640625,  0,        1, 0 ],
            [ 0,          -1.40625,    -2.25,      0.625,    1, 0 ],
            [ 0,           1.40625,    -2.25,     -0.625,    1, 0 ],
            [ 0,          -0.5859375,  -0.390625,  1.5,      1, 0 ],
            [ 0,           0.5859375,  -0.390625, -1.5,      1, 0 ],
            [ 0,           0.87890625,  0,        -2.640625, 0, 1 ]]).T
        G = np.array([
            [ 1, 1,         1 ,       1,     1,     0 ],
            [ 0, 0.625,    -0.625,    1.5,  -1.5,   0 ],
            [ 0, 0.390625,  0.390625, 2.25,  2.25,  1 ]]).T
        # yapf: enable

        return (constant(A), constant(B), constant(G))

    raise Exception("Only support L(2, 3) and L(4, 3) right now")


def _winograd(x, kernel, padding='valid', block=6):
    (A, B, G) = _compute_transforms(block, kernel.shape[0])
    s = kernel.shape[0]
    (XO, XP, NXO) = pad_compute("X", x.shape[1], s, 1, padding)
    (YO, YP, NYO) = pad_compute("Y", x.shape[2], s, 1, padding)
    outshape = (x.shape[0], NXO, NYO, kernel.shape[3])
    f = """
        function (I[N, X, Y, CI], K[S, S, CI, CO], A[BI, BO], B[BI, BI], G[BI, S] ) -> (O) {{
            Assert = assert_winograd_valid(BI - CI + 1 == BO);
            XO = {XO};
            YO = {YO};
            XB = (XO + BO - 1) / BO;
            YB = (YO + BO - 1) / BO;
            XP = {XP};
            YP = {YP};
            U1[i, j, ci, co : BI, S, CI, CO] = +(G[i, k] * K[k, j, ci, co]);
            U[i, j, ci, co : BI, BI, CI, CO] = +(U1[i, k, ci, co] * G[j, k]);
            V1[n, i, j, x, y, ci : N, BI, BI, XB, YB, CI] = +(B[k, i] * I[n, BO*x + k - XP, BO*y + j - YP, ci]);
            V[n, i, j, x, y, ci : N, BI, BI, XB, YB, CI] = +(V1[n, i, k, x, y, ci] * B[k, j]);
            M[n, i, j, x, y, co : N, BI, BI, XB, YB, CO] = +(V[n, i, j, x, y, ci] * U[i, j, ci, co]);
            O1[n, i, j, x, y, co : N, BO, BI, XB, YB, CO] = +(A[k, i] * M[n, k, j, x, y, co]);
            O[n, BO*x + i, BO*y + j, co : N, XO, YO, CO] = +(O1[n, i, k, x, y, co] * A[k, j]) no_defract;
        }}""".format(
        XO=XO, YO=YO, XP=XP, YP=YP)
    return _Op("winograd", x.dtype, outshape, f,
               OrderedDict([('I', x), ('K', kernel), ('A', A), ('B', B), ('G', G)]), ['O'])


def conv1d(x, kernel, strides=1, padding='valid', data_format=None, dilation_rate=1):
    return conv(x, kernel, (strides,), padding, data_format, (dilation_rate,))


def conv2d(x,
           kernel,
           strides=(1, 1),
           padding='valid',
           dilation_rate=(1, 1),
           data_format=None,
           force_winograd=False):
    if data_format is None:
        data_format = image_data_format()
    if (force_winograd or
        (data_format == 'channels_last' and kernel.shape[0] == 3 and kernel.shape[1] == 3 and
         strides == (1, 1) and dilation_rate == (1, 1) and kernel.shape[2] > 4 and
         kernel.shape[3] > 4)):
        return _winograd(x, kernel, padding=padding)
    return conv(x, kernel, strides, padding, data_format, dilation_rate)


def conv2d_transpose(x, kernel, output_shape, strides=(1, 1), padding='valid', data_format=None):
    return conv_transpose(x, kernel, output_shape, strides, padding, data_format)


def conv3d(x,
           kernel,
           strides=(1, 1, 1),
           padding='valid',
           dilation_rate=(1, 1, 1),
           data_format=None):
    return conv(x, kernel, strides, padding, data_format, dilation_rate)


def conv3d_transpose(x, kernel, output_shape, strides=(1, 1, 1), padding='valid',
                     data_format=None):
    return conv_transpose(x, kernel, output_shape, strides, padding, data_format)


def count_params(x):
    result = 1
    for dim in x.shape:
        result *= dim
    return result


def ctc_batch_cost(y_true, y_pred, input_length, label_length):
    _report_unimplemented('ctc_batch_cost')


def ctc_decode(y_pred, input_length, greedy=True, beam_width=100, top_paths=1):
    _report_unimplemented('ctc_decode')


def ctc_label_dense_to_sparse(labels, label_lengths):
    _report_unimplemented('ctc_label_dense_to_sparse')


def cumprod(x, axis=0):
    _report_unimplemented('cumprod')


def cumsum(x, axis=0):
    return x.cumsum(axis=axis)


def depthwise_conv2d(x,
                     kernel,
                     strides=(1, 1),
                     padding='valid',
                     data_format=None,
                     dilation_rate=(1, 1)):
    if data_format is None:
        data_format = image_data_format()
    return conv(x, kernel, strides, padding, data_format, dilation_rate, channelwise=True)


def dot(x, y):
    if x.dtype != y.dtype:
        raise PlaidMLKerasException(
            'Invalid dtype in multiplication: x.dtype=\'%s\', y.dtype=\'%s\'' % (x.dtype, y.dtype))

    if x.ndim == 1 and y.ndim == 1:
        f = 'function (X[I], Y[I]) -> (R) { R[i:I] = +(X[i] * Y[i]); }'
        shape = x.shape
    elif 1 <= x.ndim and 2 <= y.ndim:
        f = """function(X[%(x_ranges)s], Y[%(y_ranges)s]) -> (R) {
                   R[%(dest_indices)s : %(dest_ranges)s] = +(X[%(x_indices)s] * Y[%(y_indices)s]);
               }""" % {
            'x_ranges':
                ', '.join(['X{}'.format(i) for i in range(x.ndim)]),
            'y_ranges':
                ', '.join(['Y{}'.format(i) for i in range(y.ndim)]),
            'dest_indices':
                ', '.join(['x{}'.format(i) for i in range(x.ndim - 1)] +
                          ['y{}'.format(i) for i in list(range(y.ndim - 2)) + [y.ndim - 1]]),
            'dest_ranges':
                ', '.join(['X{}'.format(i) for i in range(x.ndim - 1)] +
                          ['Y{}'.format(i) for i in list(range(y.ndim - 2)) + [y.ndim - 1]]),
            'x_indices':
                ', '.join(['x{}'.format(i) for i in range(x.ndim - 1)] + ['z']),
            'y_indices':
                ', '.join(['y{}'.format(i)
                           for i in range(y.ndim - 2)] + ['z'] + ['y{}'.format(y.ndim - 1)]),
        }
        shape = list(x.shape[:-1]) + list(y.shape[:-2]) + [y.shape[-1]]

    else:
        raise PlaidMLKerasException('TODO: Implement dot when x.dim=' + str(x.dim) + ' and y.dim='
                                    + str(y.dim))

    return _Op('dot', x.dtype, shape, f, OrderedDict([('X', x), ('Y', y)]), ['R'])


def dropout(x, level, noise_shape=None, seed=None):
    if noise_shape is not None:
        raise PlaidMLKerasException('Unimplemented noise shape in dropout')

    rng_state = _make_rng_state(seed)

    szs = ", ".join(['S' + str(i) for i in range(x.ndim)])
    args = ", ".join(['I'] + ['S' + str(i) for i in range(x.ndim)])
    rng_step = "function (I, X[{szs}]) -> (O) {{ O = prng_step({args}); }}".format(
        szs=szs, args=args)
    rng_update = "function (I) -> (O) { O = prng_state(I); }"
    rng_value = """function (I, X, L) -> (O) {
        R = 1.0 - L;
        M = 1.0 / R;
        O = (prng_value(I) < R ? X * M : 0.0);
    }"""
    t = _Op('random_uniform_step', 'uint32', (), rng_step,
            OrderedDict([('I', rng_state), ('X', x)]), ['O'])
    n = _Op('random_uniform_state', 'uint32', (3, _k_rng_size), rng_update,
            OrderedDict([('I', t)]), ['O'])
    side_effects = {_plaidml_val(rng_state): _plaidml_val(n)}
    o = _Op(
        'random_uniform_value',
        'float32',
        x.shape,
        rng_value,
        OrderedDict([('I', t), ('X', x), ('L', level)]), ['O'],
        side_effects=side_effects)
    return o


def dtype(x):
    return x.dtype


def elu(x, alpha=1.0):
    _report_unimplemented('elu')


def eval(x):
    return x.eval()


def equal(x, y):
    return _Op('equal', 'bool',
               _broadcast_shape(x, y), 'function (X, Y) -> (R) { R = (X == Y); }',
               OrderedDict([('X', x), ('Y', y)]), ['R'])


def exp(x):
    f = """function (I) -> (O) { O = exp(I); }"""
    return _Op('exp', x.dtype, x.shape, f, {'I': x}, ['O'])


def eye(size, dtype=None, name=None):
    _report_unimplemented('eye')


def pow(x, p):
    f = """function (I, P) -> (O) { O = pow(I, P); }"""
    return _Op('pow', x.dtype, x.shape, f, {'I': x, 'P': p}, ['O'])


def expand_dims(x, axis=-1):
    if axis < 0:
        axis = x.ndim + 1 + axis
    slist_in = ["S" + str(i) for i in range(x.ndim)]
    ilist_in = ["i" + str(i) for i in range(x.ndim)]
    slist_out = slist_in[0:axis] + ["1"] + slist_in[axis:]
    ilist_out = ilist_in[0:axis] + ["0"] + ilist_in[axis:]
    newshape = tuple(list(x.shape[0:axis]) + [
        1,
    ] + list(x.shape[axis:]))
    f = """function (IN[{slist_in}]) -> (OUT) {{
               OUT[{ilist_out} : {slist_out}] = =(IN[{ilist_in}]);
           }}""".format(
        slist_in=", ".join(slist_in),
        slist_out=", ".join(slist_out),
        ilist_in=", ".join(ilist_in),
        ilist_out=", ".join(ilist_out))
    return _Op('expand_dims', x.dtype, newshape, f, {'IN': x}, ['OUT'])


def flatten(x):
    return x.flatten()


def floor(x):
    f = """function (I) -> (O) { O = floor(I); }"""
    return _Op('floor', x.dtype, x.shape, f, {'I': x}, ['O'])


def foldl(fn, elems, initializer=None, name=None):
    _report_unimplemented('foldl')


def foldr(fn, elems, initializer=None, name=None):
    _report_unimplemented('foldr')


def function(inputs, outputs, updates=None, name=None):
    if updates == None:
        updates = []
    if name == None:
        name = ''
    return _Function(inputs, outputs, updates, name)


def gather(v, i):
    i = cast(i, 'int32')
    f = """function (V, I) -> (OUT) {
               OUT = gather(V, I);
           }"""
    return _Op('gather', v.dtype, i.shape + v.shape[1:], f, {'V': v, 'I': i}, ['OUT'])


def shape(t):
    f = """function (T) -> (OUT) {
               OUT = shape(T);
           }"""
    return _Op('shape', 'int32', np.array([len(t.shape)]), f, {'T': t}, ['OUT'])


def get_uid(prefix=''):
    if not prefix in _UID_PREFIX_DICT:
        _UID_PREFIX_DICT[prefix] = 1
    else:
        _UID_PREFIX_DICT[prefix] += 1
    return _UID_PREFIX_DICT[prefix]


def get_value(x):
    return x.eval()


def get_variable_shape(x):
    return x.shape


def gradients(loss, variables):
    if isinstance(variables, _Var):
        variables = [variables]
    return [
        _Var('grad', var.dtype, var.shape, plaidml_val=grad, src=loss)
        for var, grad in zip(variables,
                             plaidml.gradients(
                                 _plaidml_val(loss), [_plaidml_val(var) for var in variables]))
    ]


def greater(x, y):
    return x > y


def greater_equal(x, y):
    return x >= y


def hard_sigmoid(x):
    _report_unimplemented('hard_sigmoid')


def identity(x):
    # Return a tensor with the same content as the input tensor.
    f = """function (I) -> (O) { O = I; }"""
    return _Op('ident', x.dtype, x.shape, f, {'I': x}, ['O'])


def in_test_phase(x, alt, training=None):
    # Note that this flips 'alt' and 'x'
    return in_train_phase(alt, x, training=training)


def in_top_k(predictions, targets, k):
    _report_unimplemented('in_top_k')


def in_train_phase(x, alt, training=None):
    if training is None:
        training = learning_phase()
        uses_learning_phase = True
    else:
        uses_learning_phase = False

    if callable(x):
        cx = x()
    else:
        cx = x
    if callable(alt):
        calt = alt()
    else:
        calt = alt

    if training is 1 or training is True:
        return cx
    elif training is 0 or training is False:
        return calt
    else:
        o = switch(training, cx, calt)
        if uses_learning_phase:
            o._uses_learning_phase = True
        return o


def int_shape(x):
    if hasattr(x, '_keras_shape'):
        return x._keras_shape
    else:
        raise InvalidArgument('Cannot find int_shape of ' + str(x))


def is_keras_tensor(x):
    return isinstance(x, _Var) and x.is_keras_tensor


def is_placeholder(x):
    _report_unimplemented('is_placeholder')


def is_sparse(x):
    return False


def l2_normalize(x, axis):
    norm = sqrt(sum(square(x), axis=axis, keepdims=True))
    return x / norm


def learning_phase():
    # Initialize _in_train_phase if this is the first use
    global _in_train_phase
    if _in_train_phase is None:
        _in_train_phase = placeholder(ndim=0)
    return _in_train_phase


def less(x, y):
    return x < y


def less_equal(x, y):
    return x <= y


def local_conv1d(inputs, kernel, kernel_size, strides, data_format=None):
    _report_unimplemented('local_conv1d')


def local_conv2d(inputs, kernel, kernel_size, strides, output_shape, data_format=None):
    _report_unimplemented('local_conv2d')


def log(x):
    f = """function (I) -> (O) { O = log(I); }"""
    return _Op('log', x.dtype, x.shape, f, {'I': x}, ['O'])


def logsumexp(x, axis=None, keepdims=False):
    _report_unimplemented('logsumexp')


def manual_variable_initialization(value):
    _report_unimplemented('manual_variable_initialization')


def map_fn(fn, elems, name=None, dtype=None):
    _report_unimplemented('map_fn')


# WARNING: You can't use python's builtin function 'max()' directly in this
# file. Use builtins("max")() instead.
# Unfortunately, this name is required by Keras.
def max(x, axis=None, keepdims=False):
    return x.max(axis, keepdims)


def maximum(x, y):
    return _Op('maximum', x.dtype,
               _broadcast_shape(x, y), 'function (X, Y) -> (R) { R = (X < Y ? Y : X); }',
               OrderedDict([('X', x), ('Y', y)]), ['R'])


def mean(x, axis=None, keepdims=False):
    return x.mean(axis, keepdims)


# WARNING: You can't use python's builtin function 'min()' directly in this
# file. Use getattr(__builtin__, "min")() instead.
# Unfortunately, this name is required by Keras.
def min(x, axis=None, keepdims=False):
    return x.min(axis, keepdims)


def minimum(x, y):
    return _Op('minimum', x.dtype,
               _broadcast_shape(x, y), 'function (X, Y) -> (R) { R = (X < Y ? X : Y); }',
               OrderedDict([('X', x), ('Y', y)]), ['R'])


def moving_average_update(x, value, momentum):
    return (x, x * momentum + value * (1. - momentum))


_NAME_SCOPE_STACK = []


def _prepend_name_scope(name, default):
    global _NAME_SCOPE_STACK
    if name is None:
        r = "/".join(_NAME_SCOPE_STACK + [default])
        r += "_" + str(get_uid(r))
    else:
        r = "/".join(_NAME_SCOPE_STACK + [name])
    return r


@contextmanager
def name_scope(name):
    global _NAME_SCOPE_STACK
    _NAME_SCOPE_STACK.append(name)
    yield
    _NAME_SCOPE_STACK.pop()


def ndim(x):
    return x.ndim


def not_equal(x, y):
    return _Op('not_equal', 'bool',
               _broadcast_shape(x, y), 'function (X, Y) -> (R) { R = (X != Y); }',
               OrderedDict([('X', x), ('Y', y)]), ['R'])


def normalize_batch_in_training(x, gamma, beta, reduction_axes, epsilon=1e-3):
    rank = x.ndim
    if reduction_axes == None:
        axes = [rank - 1]
    else:
        axes = reduction_axes

    # Will need to squeeze axes in order, so make sure none are negative and sort
    axes = [i + rank if i < 0 else i for i in axes]
    for i in axes:
        if i < 0:
            raise ValueError((
                "Unexpected axis '{}' in normalize_batch_in training " + "(tensor dim {})").format(
                    i - rank, rank))
        if i >= rank:
            raise ValueError((
                "Unexpected axis '{}' in normalize_batch_in training " + "(tensor dim {})").format(
                    i, rank))
    axes.sort()

    # Mean and var need to keepdims for computing normalized_tensor, but their
    # returned values need to not keepdims. So keepdims for now, then squeeze.
    mean = x.mean(axis=axes, keepdims=True)
    var = x.var(axis=axes, keepdims=True)

    # TODO: Tensorflow's code implies using anything other than the single
    # final axis as the sole element of axis requires broadcasting,
    # but I don't see it ...
    # Indeed, this passes unit tests with a non-final axis selected
    normalized_tensor = batch_normalization(
        x=x, mean=mean, var=var, beta=beta, gamma=gamma, epsilon=epsilon)

    # Tensorflow and Theano disagree on whether mean and var should be squeezed
    # here. For now, going with Theano for simplicity.
    #  for ax in reversed(axes):
    #    mean = squeeze(mean, ax)
    #    var = squeeze(var, ax)

    return normalized_tensor, mean, var


def one_hot(indices, num_classes):
    #Note: does not error check for entries in indices that are >= num_classes

    count = variable(np.array(range(num_classes)), dtype='int32')
    f = ('function (Idx[{idim}], Count[C]) -> (O) {{\n' +
         '  O[{iidx}, c : {idim}, C] = =(Idx[{iidx}] == Count[c]);\n' + '}}').format(
             idim=", ".join(["I{}".format(k) for k in range(indices.ndim)]),
             iidx=", ".join(["i{}".format(k) for k in range(indices.ndim)]))
    return _Op('one_hot', 'bool', indices.shape + (num_classes,), f,
               OrderedDict([('Idx', indices), ('Count', count)]), ['O'])


def ones(shape, dtype=None, name=None):
    dtype = dtype or floatx()
    return constant(1.0, shape=shape, dtype=dtype, name=_prepend_name_scope(name, "ones"))


def ones_like(x, dtype=None, name=None):
    dtype = dtype or floatx()
    a_one = constant(1.0, shape=(1), dtype=dtype, name=_prepend_name_scope(name, "a_one"))
    ndims = len(x.shape)
    sizes = ", ".join(["S" + str(i) for i in range(ndims)])
    dims = ", ".join(["i" + str(i) for i in range(ndims)])
    f = """function (IN[{sizes}], ONE[SZ]) -> (OUT) {{
               OUT[{dims} : {sizes}] = =(ONE[0]);
           }}""".format(
        sizes=sizes, dims=dims)
    return _Op('ones_like', dtype, x.shape, f, {'IN': x, 'ONE': a_one}, ['OUT'])


def permute_dimensions(x, pattern):
    return _Op('permute', x.dtype,
               tuple([x.shape[idx] for idx in range(x.ndim)]),
               """function (X[%(src_ranges)s]) -> (R) {
                      R[%(dest_indices)s : %(dest_ranges)s] = =(X[%(src_indices)s]);
                  }""" % {
                   'src_ranges': ', '.join(['X{}'.format(i) for i in range(x.ndim)]),
                   'src_indices': ', '.join(['x{}'.format(i) for i in range(x.ndim)]),
                   'dest_ranges': ', '.join(['X{}'.format(pattern[i]) for i in range(x.ndim)]),
                   'dest_indices': ', '.join(['x{}'.format(pattern[i]) for i in range(x.ndim)]),
               }, OrderedDict([('X', x)]), ['R'])


def placeholder(shape=None, ndim=None, dtype=None, sparse=False, name=None):
    dtype = dtype or floatx()
    if shape is None and ndim is None:
        raise PlaidMLKerasException('Specify either a shape or ndim value for placeholder.')

    if shape:
        ndim = len(shape)
    else:
        shape = tuple([None for _ in range(ndim)])

    ret = _Var(
        'placeholder',
        dtype,
        shape,
        _prepend_name_scope(name, "placeholder"),
        plaidml.Placeholder(len(shape)),
        is_keras_tensor=True)
    ret._uses_learning_phase = False
    return ret


def pool(x, pool_size, strides=None, padding='valid', data_format=None, pool_mode='max'):
    # TODO: There are major similarities between pool and conv. I think keeping
    # them separate makes sense, but we could consider merging them.
    rank = len(x.shape) - 2
    if strides is None:
        strides = tuple(1 for _ in range(rank))
    if data_format is None:
        data_format = image_data_format()

    if len(pool_size) != rank:
        raise ValueError("Pool size inconsistent with input shape: " +
                         "{} (rank {}) v {} (rank {})".format(pool_size,
                                                              len(pool_size), x.shape, x.ndim - 2))
    if len(strides) != rank:
        raise ValueError("Pool strides length inconsistent with input shape: " +
                         "{} (rank {}) v {} (rank {})".format(strides,
                                                              len(strides), x.shape, x.ndim - 2))

    if data_format == 'channels_first':
        n = 0
        c = 1
        l = [i + 2 for i in range(rank)]
    elif data_format == 'channels_last':
        n = 0
        l = [i + 1 for i in range(rank)]
        c = rank + 1
    else:
        raise ValueError("Unrecognized data format '{}'".format(data_format))

    out_size = list()
    pad_amount = list()
    num_out_size = list()
    for i in range(rank):
        sym_out, sym_pad, num_out = pad_compute("L{}".format(i), x.shape[l[i]], pool_size[i],
                                                strides[i], padding)
        out_size.append(sym_out)
        pad_amount.append(sym_pad)
        num_out_size.append(num_out)
    padding_list = ["  Pad{} = {};".format(i, pad_amount[i]) for i in range(rank)]
    padding_str = "\n".join(padding_list)
    input_idx_list = [
        "{}*{} + {} - {}".format(strides[i], "x{}".format(i), "k{}".format(i), "Pad{}".format(i))
        for i in range(rank)
    ]
    pool_bounds = ", " + ", ".join(["k{} < {}".format(i, pool_size[i]) for i in range(rank)])
    if data_format == 'channels_first':
        input_dims_str = "N, C, " + ", ".join(["L{}".format(i) for i in range(rank)])
        out_idx_str = "n, c, " + ", ".join(["x{}".format(i) for i in range(rank)])
        out_dims_str = "N, C, " + ", ".join(["{}".format(out_size[i]) for i in range(rank)])
        input_idx_str = "n, c, " + ", ".join(input_idx_list)
        outshape = list(x.shape[:2]) + num_out_size
    elif data_format == 'channels_last':
        input_dims_str = "N, " + ", ".join(["L{}".format(i) for i in range(rank)]) + ", C"
        out_idx_str = "n, " + ", ".join(["x{}".format(i) for i in range(rank)]) + ", c"
        out_dims_str = "N, " + ", ".join(["{}".format(out_size[i]) for i in range(rank)]) + ", C"
        input_idx_str = "n, " + ", ".join(input_idx_list) + ", c"
        outshape = [x.shape[0]] + num_out_size + [x.shape[-1]]
    else:
        raise ValueError("Unrecognized data format '{}'".format(data_format))
    if pool_mode == 'max':
        pool_sym = '>'
        internal_name = 'O'
        scale_expr = ''
        extra_input = ''
    elif pool_mode == 'avg':
        pool_sym = '+'
        internal_name = 'OT'
        # Want average pooling not sum pooling, so divide by number of elements in a pool
        # However, the number of elements in the pool should only count true elements,
        # not zero padding. Thus, we build a tensor that is 1 everywhere the original
        # tensor is defined, and we sum that tensor over the pool area to find the
        # number of elements in the pool for the corresponding output entry.
        ones = ones_like(x)
        scale_expr = (
            '  C[{out_idx_str}: {out_dims_str}] = +(Ones[{input_idx_str}]){pool_bounds};\n' +
            '  O = OT / C;').format(**{
                'out_idx_str': out_idx_str,
                'out_dims_str': out_dims_str,
                'input_idx_str': input_idx_str,
                'pool_bounds': pool_bounds
            })
        extra_input = ', Ones[{}]'.format(input_dims_str)
    else:
        raise ValueError("Unrecognized pool mode '{}'".format(pool_mode))

    f = ('function (I[{input_dims_str}]{extra_input}) -> (O) {{\n' + '{padding_str}\n' +
         '  {internal_name}[{out_idx_str}: {out_dims_str}]' +
         '= {pool_sym}(I[{input_idx_str}]){pool_bounds};\n'
         '{scale_expr}\n}}').format(**{
             'input_dims_str': input_dims_str,
             'out_idx_str': out_idx_str,
             'out_dims_str': out_dims_str,
             'pool_sym': pool_sym,
             'input_idx_str': input_idx_str,
             'pool_bounds': pool_bounds,
             'scale_expr': scale_expr,
             'internal_name': internal_name,
             'padding_str': padding_str,
             'extra_input': extra_input
         })

    name = "pool{}d".format(rank)
    if pool_mode == 'max':
        input_dict = {'I': x}
    elif pool_mode == 'avg':
        input_dict = OrderedDict([('I', x), ('Ones', ones)])
    else:
        raise ValueError("Unrecognized pool mode '{}'".format(pool_mode))
    return _Op(name, x.dtype, outshape, f, input_dict, ['O'])


def pool2d(x, pool_size, strides=(1, 1), padding='valid', data_format=None, pool_mode='max'):
    return pool(
        x,
        pool_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        pool_mode=pool_mode)


def pool3d(x, pool_size, strides=(1, 1, 1), padding='valid', data_format=None, pool_mode='max'):
    return pool(
        x,
        pool_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        pool_mode=pool_mode)


def print_tensor(x, message=''):
    _report_unimplemented('print_tensor')


def prod(x, axis=None, keepdims=False):
    return x.prod(axis=axis, keepdims=keepdims)


def random_binomial(shape, p=0.0, dtype=None, see=None):
    _report_unimplemented('random_binomial')


def random_normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
    if dtype is None:
        dtype = floatx()
    if seed:
        np.random.seed(seed)
    # TODO: We only use half of the Box-Muller here
    u1 = random_uniform(shape, dtype='float32')
    u2 = random_uniform(shape, dtype='float32')
    z0 = sqrt(-2.0 * log(u1 + (1.0 / (2**33)))) * cos(2.0 * math.pi * u2)
    if stddev != 1.0:
        z0 = stddev * z0
    if mean != 0.0:
        z0 = z0 + mean
    if dtype != 'float32':
        z0 = cast(z0, dtype)
    return z0


def random_normal_variable(shape, mean, scale, dtype=None, name=None, seed=None):
    _report_unimplemented('random_normal_variable')


def random_uniform(shape, minval=0.0, maxval=1.0, dtype=None, seed=None):
    dtype = dtype or floatx()
    rng_state = _make_rng_state(seed)

    args = ", ".join(["I"] + [str(i) for i in shape])
    func_step = "function (I) -> (O) {{ O = prng_step({args}); }}".format(args=args)
    func_state = "function (I) -> (O) { O = prng_state(I); }"
    func_value = "function (I) -> (O) { O = prng_value(I); }"
    t = _Op('random_uniform_step', 'uint32', (), func_step, OrderedDict([('I', rng_state)]), ['O'])
    n = _Op('random_uniform_state', 'uint32', (3, _k_rng_size), func_state,
            OrderedDict([('I', t)]), ['O'])
    side_effects = {_plaidml_val(rng_state): _plaidml_val(n)}
    o = _Op(
        'random_uniform_value',
        'float32',
        shape,
        func_value,
        OrderedDict([('I', t)]), ['O'],
        side_effects=side_effects)

    if dtype != 'float32':
        o = cast(o, dtype)
    if maxval - minval != 1.0:
        o = (maxval - minval) * o
    if minval != 0.0:
        o = o + minval
    return o


def random_uniform_variable(shape, low, high, dtype=None, name=None):
    if seed:
        np.random.seed(seed)
    val = np.random.uniform(low=low, high=high, size=size)
    return variable(val, dtype=dtype)


def relu(x, alpha=0.0, max_value=None):
    inputs = OrderedDict([('X', x), ('Alpha', alpha)])
    if max_value:
        inputs['MaxValue'] = max_value
        f = """function (X, Alpha, MaxValue) -> (R) {
                   M = (X < 0 ? Alpha*X : X);
                   R = (M < MaxValue ? M : MaxValue);
               }"""
    else:
        f = 'function (X, Alpha) -> (R) { R = (X < 0 ? Alpha*X : X); }'

    return _Op('relu', x.dtype, x.shape, f, inputs, ['R'])


def repeat(x, n):
    assert x.ndim == 2
    f = ("function (I[N0, N1]) -> (O) {{" + "  O[i0, r, i1: N0, {reps}, N1] = =(I[i0, i1]);" +
         "}}").format(reps=n)
    return _Op('repeat', x.dtype, (x.shape[0], n, x.shape[1]), f, {'I': x}, ['O'])


def repeat_elements(x, rep, axis):
    if x.shape[axis] is None:
        # Note: other backends just raise exception in this case
        out_shape = x.shape[:axis] + (None,) + x.shape[axis + 1:]
    else:
        out_shape = x.shape[:axis] + (rep * x.shape[axis],) + x.shape[axis + 1:]
    idim_list = ["N{}".format(i) for i in range(x.ndim)]
    iidx_list = [N.lower() for N in idim_list]
    odim_list = ["{}*N{}".format(rep, i) if i == axis else "N{}".format(i) for i in range(x.ndim)]
    oidx_list = [
        "{}*n{} + k".format(rep, i) if i == axis else "n{}".format(i) for i in range(x.ndim)
    ]

    # Example
    # function(I[N0, N1, N2]) -> (O) {
    #   O[n0, 3*n1 + k, n2 : N0, 3*N1, N2] = =(I[n0, n1, n2]), k < 3 no_defract;
    # }
    f = (
        "function (I[{idims}]) -> (O) {{\n" +
        "  O[{oidxs} : {odims}] = =(I[{iidxs}]), k < {rep} no_defract;\n" +  #;\n" + #
        "}}").format(
            idims=", ".join(idim_list),
            iidxs=", ".join(iidx_list),
            odims=", ".join(odim_list),
            oidxs=", ".join(oidx_list),
            rep=str(rep))
    return _Op('repeat_elements', x.dtype, out_shape, f, {'I': x}, ['O'])


def reset_uids():
    global _UID_PREFIX_DICT
    _UID_PREFIX_DICT = dict()


def reshape(x, shape):
    return x.reshape(shape)


def resize_images(x, height_factor, width_factor, data_format):
    if data_format == 'channels_first':
        ret = repeat_elements(x, height_factor, axis=2)
        ret = repeat_elements(ret, width_factor, axis=3)
    elif data_format == 'channels_last':
        ret = repeat_elements(x, height_factor, axis=1)
        ret = repeat_elements(ret, width_factor, axis=2)
    else:
        raise ValueError("Invalid data_format {}".format(data_format))
    return ret


def resize_volumes(x, depth_factor, height_factor, width_factor, data_format):
    _report_unimplemented('resize_volumes')


def reverse(x, axes):
    if isinstance(axes, int):
        axes = [axes]
    for axis in axes:
        if not isinstance(axis, int):
            raise ValueError("The axes parameter of reverse only accepts an integer or a list of integers, received {}".format(type(axis)))
        if axis >= x.ndim or axis < -x.ndim:
            raise ValueError("Invalid axis {} in reverse: target {} too short (ndim={})".format(axis, x, x.ndim))
    axes = [a % x.ndim for a in axes]
    dims = ", ".join("N{}".format(j) for j in range(x.ndim))
    in_idxs = ", ".join("i{}".format(j) for j in range(x.ndim))
    out_idxs = ", ".join(("N{j} - 1 - i{j}" if j in axes else "i{j}").format(j=j) for j in range(x.ndim))
    f = """function (I[{dims}]) -> (O) {{\nO[{out_idxs}: {dims}] = =(I[{in_idxs}]);\n}}""".format(
            dims=dims, out_idxs=out_idxs, in_idxs=in_idxs)
    return _Op('reverse', x.dtype, x.shape, f, {'I': x}, ['O'])


def reverse_gradient(x, coeff=1.0):
    f = """function (I, C) -> (O) { O = reverse_grad(I, C); }"""
    return _Op('reverse_grad', x.dtype, x.shape, f, {'I': x, 'C': coeff}, ['O'])


def rnn(step_function,
        inputs,
        initial_states,
        go_backwards=False,
        mask=None,
        constants=None,
        unroll=False,
        input_length=None):
    _report_unimplemented('rnn')


def round(x):
    f = """function (I) -> (O) { O = round(I); }"""
    return _Op('round', x.dtype, x.shape, f, {'I': x}, ['O'])


def separable_conv(x,
                   depthwise_kernel,
                   pointwise_kernel,
                   strides=None,
                   padding='valid',
                   data_format=None,
                   dilation_rate=None):
    if data_format is None:
        data_format = image_data_format()
    if pointwise_kernel.shape[-2] != depthwise_kernel.shape[-1] * depthwise_kernel.shape[-2]:
        raise ValueError(
            ("Shape mismatch in separable convolution. Depthwise kernel input " +
             "channel count must match pointwise kernel channel count times channel " +
             "multiplier.\nReceived {} v {} * {} (from full shapes {} and " + "{})").format(
                 pointwise_kernel.shape[-2], depthwise_kernel.shape[-2],
                 depthwise_kernel.shape[-1], pointwise_kernel.shape, depthwise_kernel.shape))
    intermediate = conv(
        x,
        depthwise_kernel,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        channelwise=True)
    rank = x.ndim - 2
    ones = tuple(1 for _ in range(rank))
    return conv(
        intermediate,
        pointwise_kernel,
        strides=ones,
        padding='valid',
        data_format=data_format,
        dilation_rate=ones)


def separable_conv2d(x,
                     depthwise_kernel,
                     pointwise_kernel,
                     strides=(1, 1),
                     padding='valid',
                     data_format=None,
                     dilation_rate=(1, 1)):
    return separable_conv(x, depthwise_kernel, pointwise_kernel, strides, padding, data_format,
                          dilation_rate)


def set_floatx(dtype):
    keras_set_floatx(dtype)
    plaidml.set_floatx(_dtypes[dtype])


def set_learning_phase(value):
    if value != 0 and value != 1:
        raise ValueError("May only set_learning_phase to 0 or 1")
    global _in_train_phase
    if _in_train_phase is None:
        # Initialize _in_train_phase if this is the first use
        _in_train_phase = variable(value)
    else:
        _in_train_phase.set_value(value)


def set_value(x, value):
    x.set_value(value)


def sigmoid(x):
    f = "function (I) -> (O) { O = sigmoid(I); }"
    return _Op('sigmoid', x.dtype, x.shape, f, {'I': x}, ['O'])


def sign(x):
    _report_unimplemented('sign')


def sin(x):
    f = "function (I) -> (O) { O = sin(I); }"
    return _Op('sin', x.dtype, x.shape, f, {'I': x}, ['O'])


def softmax(x):
    f = """function (IN[X, Y]) -> (OUT) {
               OUT = builtin_softmax(IN, X, Y);
           }"""
    if x.ndim == 2:
        return _Op('softmax', x.dtype, x.shape, f, {'IN': x}, ['OUT'])
    else:
        full_shape = x.shape
        flat_shape = (np.prod(x.shape[:-1]), x.shape[-1])
        flat_x = reshape(x, flat_shape)
        softmaxed = _Op('softmax', x.dtype, flat_shape, f, {'IN': flat_x}, ['OUT'])
        return reshape(softmaxed, full_shape)


def softplus(x):
    _report_unimplemented('softplus')


def softsign(x):
    _report_unimplemented('softsign')


def sparse_categorical_crossentropy(target, output, from_logits=False):
    return categorical_crossentropy(reshape(one_hot(target, output.shape[-1]), output.shape), output, from_logits)


def spatial_3d_padding(x, padding=((1, 1), (1, 1), (1, 1)), data_format=None):
    _report_unimplemented('spatial_3d_padding')


def square(x):
    return x * x


def squeeze(x, axis):
    if x.shape[axis] != 1:
        raise ValueError("Can only squeeze length 1 axis")
    if axis == -1:
        result = x.reshape(x.shape[:axis])
    else:
        result = x.reshape(x.shape[:axis] + x.shape[axis + 1:])
    return result


def sqrt(x):
    f = """function (I) -> (O) {
               IC = (I < 0 ? 0 : I);
               O = sqrt(IC);
           }"""
    return _Op('sqrt', x.dtype, x.shape, f, {'I': x}, ['O'])


def stack(x, axis=0):
    _report_unimplemented('stack')


def std(x, axis=None, keepdims=False):
    return sqrt(var(x, axis=axis, keepdims=keepdims))


def stop_gradient(variables):
    _report_unimplemented('stop_gradient')


def sum(x, axis=None, keepdims=False):
    return x.sum(axis=axis, keepdims=keepdims)


def switch(condition, then_expression, else_expression):
    f = """function (C, T, E) -> (O) { O = (C ? T : E); }"""
    return _Op('switch', then_expression.dtype, then_expression.shape, f,
               OrderedDict([
                   ('C', condition),
                   ('T', then_expression),
                   ('E', else_expression),
               ]), ['O'])


def tanh(x):
    f = """function (I) -> (O) { O = tanh(I); }"""
    return _Op('tanh', x.dtype, x.shape, f, {'I': x}, ['O'])


def temporal_padding(x, padding=(1, 1)):
    _report_unimplemented('temporal_padding')


def tile(x, n):
    if len(n) != x.ndim:
        raise Exception("Tile size dimensions doesn't match ndims")
    sizes = ", ".join(["S" + str(i) for i in range(x.ndim)])
    out_idx = ", ".join(["t" + str(i) + " * S" + str(i) + " + i" + str(i) for i in range(x.ndim)])
    out_sizes = ", ".join(["S" + str(i) + " * " + str(n[i]) for i in range(x.ndim)])
    in_idx = ", ".join(["i" + str(i) for i in range(x.ndim)])
    cons = ", ".join(["t" + str(i) + " < " + str(n[i]) for i in range(x.ndim)])
    f = """function (I[{sizes}]) -> (O) {{
               O[{out_idx} : {out_sizes}] = =(I[{in_idx}]), {cons} no_defract;
           }}""".format(
        sizes=sizes, out_idx=out_idx, out_sizes=out_sizes, in_idx=in_idx, cons=cons)
    out_shape = tuple([None if x.shape[i] is None else x.shape[i] * n[i] for i in range(x.ndim)])
    return _Op('tile', x.dtype, out_shape, f, {'I': x}, ['O'])


def to_dense(tensor):
    _report_unimplemented('to_dense')


def transpose(x):
    return permute_dimensions(x, range(x.ndim - 1, -1, -1))


def truncated_normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
    if dtype is None:
        dtype = floatx()
    if seed:
        np.random.seed(seed)
    return stddev * scipy.stats.truncnorm.rvs(-2.0, 2.0, size=shape) + mean


def update(x, new_x):
    return (x, new_x)


def update_add(x, increment):
    return (x, x + increment)


def update_sub(x, decrement):
    return (x, x - decrement)


def var(x, axis=None, keepdims=False):
    return x.var(axis, keepdims)


def variable(value, dtype=None, name=None, constraint=None):
    dtype = dtype or floatx()
    if constraint:
        raise PlaidMLKerasException('Unsupported variable constraint')
    if isinstance(value, six.integer_types):
        tensor = plaidml.Tensor(_device(), plaidml.Shape(_ctx, _dtypes[dtype]))
        with tensor.mmap_discard(_ctx) as view:
            view.copy_from_ndarray(np.array(value))
            view.writeback()
        ret = _Tensor('integer', dtype, (), _prepend_name_scope(name, "int_variable"), tensor)
    elif isinstance(value, float):
        tensor = plaidml.Tensor(_device(), plaidml.Shape(_ctx, _dtypes[dtype]))
        with tensor.mmap_discard(_ctx) as view:
            view.copy_from_ndarray(np.array(value))
            view.writeback()
        ret = _Tensor('float', dtype, (), _prepend_name_scope(name, "float_variable"), tensor)
    elif isinstance(value, _Tensor):
        ret = value
    else:
        if isinstance(value, _Op):
            value = value.eval()
        tensor = plaidml.Tensor(_device(), plaidml.Shape(_ctx, _dtypes[dtype], *value.shape))
        with tensor.mmap_discard(_ctx) as view:
            view.copy_from_ndarray(value)
            view.writeback()
        ret = _Tensor('tensor', dtype, value.shape, _prepend_name_scope(name, "variable"), tensor)
    if constraint is not None:
        ret.constraint = constraint
    ret._uses_learning_phase = False
    return ret


def zeros(shape, dtype=floatx(), name=None):
    return constant(0.0, shape=shape, dtype=dtype, name=_prepend_name_scope(name, "zeros"))


def zeros_like(x, dtype=floatx(), name=None):
    a_zero = constant(0.0, shape=(1), dtype=dtype, name=_prepend_name_scope(name, "a_zero"))
    ndims = len(x.shape)
    sizes = ", ".join(["S" + str(i) for i in range(ndims)])
    dims = ", ".join(["i" + str(i) for i in range(ndims)])
    f = """function (IN[{sizes}], ZERO[SZ]) -> (OUT) {{
               OUT[{dims} : {sizes}] = =(ZERO[0]);
           }}""".format(
        sizes=sizes, dims=dims)
    return _Op('zeros_like', dtype, x.shape, f, {'IN': x, 'ZERO': a_zero}, ['OUT'])
