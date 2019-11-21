# Copyright 2018 Intel Corporation.
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
import inspect
import logging
import math
import numpy as np
import os
import plaidml
import plaidml.op as op
import plaidml.settings
import plaidml.tile as ptile
import scipy.stats
import six
import sys
import threading
import types

from collections import defaultdict, OrderedDict
from contextlib import contextmanager
from six import iteritems
from six.moves import builtins

from keras.backend.common import cast_to_floatx
from keras.backend.common import epsilon
from keras.backend.common import floatx
from keras.backend.common import image_data_format
from keras.backend.common import set_floatx as keras_set_floatx
from keras.backend.common import set_image_data_format

logger = logging.getLogger(__name__)


def _log_call(func):
    '''A decorator that logs the call of the wrapped function'''

    def wrapper(*args, **kwargs):
        # Call the requested function regardless
        return func(*args, **kwargs)

    return wrapper


def _normalize_data_format_unavailable(*args, **kwargs):
    raise RuntimeError('Did not find function "normalize_data_format" in Keras common ' +
                       'backend. Are you running Keras 2.2.1+ code with Keras 2.2.0-?')


try:
    from keras.backend.common import normalize_data_format
except ImportError:
    normalize_data_format = _normalize_data_format_unavailable


class PlaidMLKerasException(Exception):
    pass


_in_train_phase = None  # Will be initialized on first use
_app_stack = []

_ctx = plaidml.Context()

PLAIDML_EVENTLOG_FILENAME = os.getenv('PLAIDML_EVENTLOG_FILENAME')
if PLAIDML_EVENTLOG_FILENAME:
    print('Logging events to', PLAIDML_EVENTLOG_FILENAME)
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
_UID_PREFIX_DICT = defaultdict(int)


def _report_unimplemented(name):
    report = (
        'The Keras backend function \'{}\' is not yet implemented in ' +
        'Plaid. You can help us prioritize by letting us know if this ' +
        'function is important to you, and as always, contributions are welcome!').format(name)
    raise NotImplementedError(report)


_AUTO_PAD = {
    'valid': op.AutoPadding.VALID,
    'same': op.AutoPadding.SAME_UPPER,
}

_CONV_DATA_FORMAT = {
    'channels_first': op.ConvolutionDataFormat.CHANNELS_FIRST,
    'channels_last': op.ConvolutionDataFormat.CHANNELS_LAST,
}

_POOL_DATA_FORMAT = {
    'channels_first': op.PoolDataFormat.NCX,
    'channels_last': op.PoolDataFormat.NXC,
}

_POOL_MODE = {
    'max': op.PoolMode.MAX,
    'avg': op.PoolMode.AVG,
}


class _Function(object):
    """Represents a composed function object."""

    def __init__(self, inputs, outputs, updates, name):
        """Initializes a composed function object.

        Args:
            inputs ([ptile.Value]): A list of placeholder values.
            outputs ([ptile.Value]): A list of operation outputs.
            updates ([(ptile.Value, ptile.Value)]): A list of (var, newval) tuples.
            name (str): A name for the function (ignored).
        """
        self._name = name
        self._input_names = ['I' + str(n) for n in range(len(inputs))]
        self._output_names = ['O' + str(n) for n in range(len(outputs))]
        self._func = ptile.compose(_ctx,
                                   _device(),
                                   list(zip(self._input_names, inputs)),
                                   list(zip(self._output_names, outputs)),
                                   updates,
                                   name=name)
        self._invoker = plaidml.Invoker(_ctx, self._func)

        self._input_types = {}
        for name, val in zip(self._input_names, inputs):
            if is_placeholder(val):
                self._input_types[name] = ptile.convert_pml_dtype_to_np(val.shape.dtype)

    def __call__(self, inputs):
        # Inputs: a list of bindings for the placeholders.

        for (name, val) in zip(self._input_names, inputs):
            if isinstance(val, six.integer_types):
                val = plaidml.Integer(val)
            elif isinstance(val, float):
                val = plaidml.Real(val)
            else:
                val = variable(val, dtype=self._input_types[name]).var
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


@_log_call
def abs(x):
    return builtins.abs(x)


all = op.all

any = op.any


@_log_call
def arange(start, stop=None, step=1, dtype='int32'):
    if isinstance(dtype, plaidml.DType):
        dtype = ptile.convert_pml_dtype_to_np(dtype)
    return variable(np.arange(start, stop, step, dtype), dtype=dtype)


argmax = op.argmax


@_log_call
def argmin(x, axis=-1):
    return argmax(-x, axis=axis)


@_log_call
def backend():
    return 'plaidml'


class BatchDot(ptile.Operation):

    def __init__(self, x, y, axes=None, name=None):
        # axes match
        if isinstance(axes, int):
            axes = (axes, axes)
        if axes is None:
            axes = (x.shape.ndims - 1, y.shape.ndims - 2)
        PLAIDML_BATCHDOT_TF_BEHAVIOR = os.getenv('PLAIDML_BATCHDOT_TF_BEHAVIOR')
        if PLAIDML_BATCHDOT_TF_BEHAVIOR:
            # replicate tf behavior
            # TensorFlow has output dimensions defined as:
            # A tensor with shape equal to the concatenation of x's shape
            # (less the dimension that was summed over) and y's shape
            # (less the batch dimension and the dimension that was summed over).
            # If the final rank is 1, we reshape it to (batch_size, 1).
            x_excl = (x.shape.dims[:axes[0]] + x.shape.dims[axes[0] + 1:])
            y_excl = (y.shape.dims[:axes[1]] + y.shape.dims[axes[1] + 1:])
            xdim_list = ['M{}'.format(i) for i in range(x.shape.ndims)]
            xdim_list[0] = 'B'
            xdim_list[axes[0]] = 'D'
            ydim_list = ['N{}'.format(i) for i in range(y.shape.ndims)]
            ydim_list[0] = 'B'
            ydim_list[axes[1]] = 'D'
            m_only = [m for m in xdim_list if m.startswith('M')]
            n_only = [n for n in ydim_list if n.startswith('N')]
            bcast_pairs = OrderedDict()
            if len(y_excl) > len(x_excl):
                for i in range(len(m_only)):
                    bcast_pairs[n_only[i]] = m_only[i]
            elif len(y_excl) == len(x_excl):
                for i in range(len(m_only) - 1):
                    bcast_pairs[n_only[i]] = m_only[i]
            else:
                for i in range(len(n_only)):
                    bcast_pairs[n_only[i]] = m_only[i]
            # for the case where x.shape.dims[N] == None, the PlaidML backend will always evaluate the comparison x.shape.dims[N] > 1 to True. This behavior is necessary for the correctness of xidx_list.
            xidx_list = [
                xdim_list[N].lower() if x.shape.dims[N] >= 1 else '0'
                for N in range(len(xdim_list))
            ]
            # for the case where y.shape.dims[N] == None, the PlaidML backend will always evaluate the comparison y.shape.dims[N] > 1 to True. This behavior is necessary for the correctness of yidx_list.
            yidx_list = [
                ydim_list[N].lower() if y.shape.dims[N] >= 1 else '0'
                for N in range(len(ydim_list))
            ]
            if len(bcast_pairs):
                yidx_list = [
                    bcast_pairs[str(yidx_list[i]).upper()].lower()
                    if str(yidx_list[i]).upper() in bcast_pairs.keys() else yidx_list[i]
                    for i in range(len(yidx_list))
                ]
            if len(y_excl) > len(x_excl):
                out_dims = (x_excl + y_excl[len(x_excl):])
            elif len(y_excl) == len(x_excl) != 1:
                out_dims = (x_excl + y_excl[len(x_excl) - 1:])
            else:
                out_dims = x_excl
            odim_list = ['B'] + ['O{}'.format(N + 1) for N in range(len(bcast_pairs))]
            oidx_list = ['b']
            if len(m_only) > len(n_only):
                odim_list += m_only[len(n_only):]
                oidx_list += [i.lower() for i in m_only[len(n_only):]]
            elif len(n_only) > len(m_only):
                odim_list += n_only[len(m_only):]
                oidx_list += [i.lower() for i in n_only[len(m_only):]]
            else:
                odim_list += [i for i in m_only if i not in bcast_pairs.values()]
                odim_list += [i for i in n_only if i not in bcast_pairs.keys()]
                oidx_list += [i.lower() for i in m_only]
                oidx_list += [i.lower() for i in n_only if i not in bcast_pairs.keys()]
            bcast_cmd_list = [
                'O{} '.format(i + 1) + '= broadcast({},{});'.format(
                    list(bcast_pairs.values())[i],
                    list(bcast_pairs.keys())[i]) for i in range(len(bcast_pairs.items()))
            ]
            if out_dims[0] is None:  # can infer batch size from either x or y
                out_dims = (y.shape.dims[0],) + out_dims[1:]
            f = """
                function (X[{xdims}], Y[{ydims}]) -> (O) {{
                    {bcast_cmd_str}
                    O[{oidxs}: {odims}] = +(X[{xidxs}] * Y[{yidxs}]);
                }}""".format(bcast_cmd_str=' '.join(bcast_cmd_list),
                             xdims=', '.join(xdim_list),
                             ydims=', '.join(ydim_list),
                             odims=', '.join(odim_list),
                             xidxs=', '.join(xidx_list),
                             yidxs=', '.join(yidx_list),
                             oidxs=', '.join(oidx_list))
            super(BatchDot, self).__init__(f, [('X', x), ('Y', y)],
                                           [('O', ptile.Shape(x.shape.dtype, out_dims))],
                                           name=name)
        else:
            # replicate theano behavior
            out_dims = (x.shape.dims[:axes[0]] + x.shape.dims[axes[0] + 1:] +
                        y.shape.dims[1:axes[1]] + y.shape.dims[axes[1] + 1:])
            if out_dims[0] is None:  # can infer batch size from either x or y
                out_dims = (y.shape.dims[0],) + out_dims[1:]
            xdim_list = ['M{}'.format(i) for i in range(x.shape.ndims)]
            xdim_list[0] = 'B'
            xdim_list[axes[0]] = 'D'
            ydim_list = ['N{}'.format(i) for i in range(y.shape.ndims)]
            ydim_list[0] = 'B'
            ydim_list[axes[1]] = 'D'
            odim_list = [N for N in xdim_list if N != 'D'] + [N for N in ydim_list[1:] if N != 'D']
            xidx_list = [N.lower() for N in xdim_list]
            yidx_list = [N.lower() for N in ydim_list]
            oidx_list = [N.lower() for N in odim_list]
            # Example
            # function (X[B, M1, M2, M3, D], Y[B, N1, D, N3]) -> (O) {
            #   O[b, m1, m2, m3, n1, n3: B, M1, M2, M3, N1, N3] = +(X[b, m1, m2, m3, d] * Y[b, n1, d, n3]);
            # }
            f = """
                function (X[{xdims}], Y[{ydims}]) -> (O) {{
                    O[{oidxs}: {odims}] = +(X[{xidxs}] * Y[{yidxs}]);
                }}""".format(xdims=', '.join(xdim_list),
                             ydims=', '.join(ydim_list),
                             odims=', '.join(odim_list),
                             xidxs=', '.join(xidx_list),
                             yidxs=', '.join(yidx_list),
                             oidxs=', '.join(oidx_list))
            super(BatchDot, self).__init__(f, [('X', x), ('Y', y)],
                                           [('O', ptile.Shape(x.shape.dtype, out_dims))],
                                           name=name)


@_log_call
def batch_dot(x, y, axes=None, name=None):
    ret = BatchDot.function(x, y, axes=axes, name=name)
    if ret.shape.ndims == 1:
        ret = expand_dims(ret, 1)
    return ret


class BatchFlatten(ptile.Operation):

    def __init__(self, x):
        # Flatten all but first dimension to a single dimension; leave 1st dimension unchanged
        # Note this is a specific kind of reshape that serves a special role in Keras (for Flatten layers)
        if x.shape.ndims < 2:
            raise PlaidMLKerasException('BatchFlatten called on tensor with ndim < 2')

        in_dim_list = ['N{}'.format(i) for i in range(x.shape.ndims)]
        out_dim_list = ['N0', '*'.join(['N{}'.format(i) for i in range(1, x.shape.ndims)])]
        rest_shape = functools.reduce(lambda x, y: x * y, x.shape.dims[1:])

        outshape = ptile.Shape(x.shape.dtype, (x.shape.dims[0], rest_shape))

        code = ('function (I[{idims}]) -> (O) {{\n' + '  O = reshape(I, {odims});\n'
                '}}').format(idims=', '.join(in_dim_list), odims=', '.join(out_dim_list))
        super(BatchFlatten, self).__init__(code, [('I', x)], [('O', outshape)])


batch_flatten = BatchFlatten.function


@_log_call
def batch_set_value(tuples):
    for pair in tuples:
        set_value(pair[0], pair[1])


@_log_call
def batch_get_value(xs):
    return [get_value(x) for x in xs]


@_log_call
def batch_normalization(x, mean, var, beta, gamma, axis=-1, epsilon=1e-3):
    # gamma == scale
    # beta == offset
    # The `axis` parameter is only used to tell TF the format of a fused batchnorm,
    # so we ignore it.
    denom = sqrt(var + epsilon)
    if gamma is not None and beta is not None:
        return ((x - mean) * gamma / denom) + beta
    elif gamma is not None:
        return ((x - mean) * gamma / denom)
    elif beta is not None:
        return ((x - mean) / denom) + beta
    else:
        return ((x - mean) / denom)


@_log_call
def bias_add(x, bias, data_format=None):
    if data_format is None:
        data_format = image_data_format()
    if data_format not in _CONV_DATA_FORMAT:
        raise PlaidMLKerasException(
            'Unrecognized data_format given to bias_add: \'' + str(data_format) +
            '\'; only \'channels_first\' and \'channels_last\' recognized.')
    try:
        bias_dims = bias.shape.dims
    except AttributeError:
        bias_dims = bias.shape

    if ndim(x) > 2:
        if data_format == 'channels_first':
            x += reshape(bias, (1, bias_dims[0]) + (1,) * (ndim(x) - 2))
        elif data_format == 'channels_last':
            x += bias
    else:
        x += bias
    return x


@_log_call
def binary_crossentropy(target, output, from_logits=False):
    return op.binary_crossentropy(target, output, epsilon(), from_logits)


@_log_call
def cast(x, dtype):
    # Not clear what datatypes Keras supports.
    # Each backend appears to implement support for its own subset of some assumed
    # but undocumented pool of possible numeric types. Perhaps this pool may be
    # the array element scalar type names defined by Numpy?
    # Tensorflow supports:
    #  float16, float32, float64, int16, int32, int64, uint8, uint16
    # Scipy offers
    # Not sure where 'bool' comes from; scipy uses 'bool_' and 'bool8'.

    x = ptile.Value.from_python_value(x)

    try:
        dtype = ptile.convert_np_dtype_to_pml(dtype)
    except ValueError:
        raise PlaidMLKerasException('Unsupported cast (%s -> %s)' % (x.shape.dtype, dtype))

    if x.shape.dtype == dtype:
        return x

    return op.cast(x, dtype)


class CategoricalCrossentropy(ptile.Operation):
    """
    Computes the categorical crossentropy of a value relative to a target.
    """

    def __init__(self, target, output, from_logits=False):
        if from_logits:
            output = softmax(output)
        elif (not output.source) or not (isinstance(output.source.op, op.Softmax)):
            output /= op.summation(output, axes=(-1,), keepdims=True)
            output = op.clip(output, epsilon(), 1.0 - epsilon())
        if output.shape.ndims == 1:
            code = """
                function (O[Y], T[Y]) -> (R) {
                    LO = log(O);
                    TR[] = +(T[y] * LO[y]);
                    R = -TR;
                }"""
        else:
            fixed_dims = ','.join('X{}'.format(i) for i in range(output.shape.ndims - 1))
            fixed_idxs = ','.join('x{}'.format(i) for i in range(output.shape.ndims - 1))
            code = """
                function (O[{fixed_dims},Y], T[{fixed_dims},Y]) -> (R) {{
                    LO = log(O);
                    TR[{fixed_idxs}:{fixed_dims}] = +(T[{fixed_idxs},y] * LO[{fixed_idxs},y]);
                    R = -TR;
                }}""".format(fixed_dims=fixed_dims, fixed_idxs=fixed_idxs)

        super(CategoricalCrossentropy,
              self).__init__(code, [('O', output), ('T', target)],
                             [('R', ptile.Shape(output.shape.dtype, output.shape.dims[:-1]))])


categorical_crossentropy = CategoricalCrossentropy.function

ceil = op.ceiling


@_log_call
def clear_session():
    global _in_train_phase, _ctx, _dev, PLAIDML_EVENTLOG_FILENAME
    _in_train_phase = None
    _ctx = plaidml.Context()
    if _dev:
        _dev.close()
    _dev = None
    if PLAIDML_EVENTLOG_FILENAME:
        _ctx.set_eventlog_filename(PLAIDML_EVENTLOG_FILENAME)


clip = op.clip

concatenate = op.concatenate


@_log_call
def constant(value, dtype=None, shape=None, name=None):
    # Enforce sensible defaults if given None
    dtype = dtype or floatx()
    if shape is None:
        if isinstance(value, np.ndarray):
            shape = value.shape
        elif isinstance(value, list) or isinstance(value, tuple):
            shape = (len(value),)
        else:
            shape = (1,)
    np_value = np.full(shape, value)
    return variable(np_value, dtype=dtype, name=_prepend_name_scope(name, 'constant'))


cos = op.cos


@_log_call
def conv(x,
         kernel,
         strides=None,
         padding='valid',
         data_format=None,
         dilation_rate=None,
         channelwise=False):
    try:
        padding = _AUTO_PAD[padding]
    except KeyError:
        six.raise_from(ValueError('Unrecognized padding: {}'.format(padding)), None)

    if data_format is None:
        data_format = image_data_format()
    try:
        data_format = _CONV_DATA_FORMAT[data_format]
    except KeyError:
        six.raise_from(ValueError('Unrecognized data format: {}'.format(data_format)), None)

    if channelwise:
        grouping = op.ConvolutionGrouping.MAX
    else:
        grouping = op.ConvolutionGrouping.NONE
    return op.convolution(
        x,
        kernel,
        padding=padding,
        dilation_rate=dilation_rate,
        strides=strides,
        data_format=data_format,
        grouping=grouping,
        kernel_format=op.ConvolutionKernelFormat.CHANNELS_LAST,
        group_format=op.GroupedChannelFormat.GroupGroupOut,
        winograd_allowed=plaidml.settings.enable_winograd,
        name=cur_name(),
    )


@_log_call
def conv_transpose(x, kernel, output_shape, strides, padding, data_format, dilation_rate):
    try:
        padding = _AUTO_PAD[padding]
    except KeyError:
        six.raise_from(ValueError('Unrecognized padding: {}'.format(padding)), None)

    if data_format is None:
        data_format = image_data_format()

    try:
        data_format = _CONV_DATA_FORMAT[data_format]
    except KeyError:
        six.raise_from(ValueError('Unrecognized data format: {}'.format(data_format)), None)

    return op.convolution_transpose(
        x,
        kernel,
        output_shape,
        strides,
        padding,
        data_format,
        kernel_format=op.ConvolutionKernelFormat.CHANNELS_LAST,
        dilation_rate=dilation_rate,
        name=cur_name(),
    )


@_log_call
def conv1d(x, kernel, strides=1, padding='valid', data_format=None, dilation_rate=1):
    if padding == 'causal':
        left_pad = dilation_rate * (kernel.shape.dims[0] - 1)
        x = temporal_padding(x, (left_pad, 0))
        padding = 'valid'
    return conv(x, kernel, (strides,), padding, data_format, (dilation_rate,))


@_log_call
def conv2d(x, kernel, strides=(1, 1), padding='valid', dilation_rate=(1, 1), data_format=None):
    if data_format is None:
        data_format = image_data_format()
    return conv(x, kernel, strides, padding, data_format, dilation_rate)


@_log_call
def conv2d_transpose(x,
                     kernel,
                     output_shape,
                     strides=(1, 1),
                     padding='valid',
                     data_format=None,
                     dilation_rate=(1, 1)):
    return conv_transpose(x, kernel, output_shape, strides, padding, data_format, dilation_rate)


@_log_call
def conv3d(x,
           kernel,
           strides=(1, 1, 1),
           padding='valid',
           dilation_rate=(1, 1, 1),
           data_format=None):
    return conv(x, kernel, strides, padding, data_format, dilation_rate)


@_log_call
def conv3d_transpose(x,
                     kernel,
                     output_shape,
                     strides=(1, 1, 1),
                     padding='valid',
                     data_format=None,
                     dilation_rate=(1, 1, 1)):
    return conv_transpose(x, kernel, output_shape, strides, padding, data_format, dilation_rate)


@_log_call
def count_params(x):
    result = 1
    for dim in x.shape.dims:
        result *= dim
    return result


@_log_call
def ctc_batch_cost(y_true, y_pred, input_length, label_length):
    _report_unimplemented('ctc_batch_cost')


@_log_call
def ctc_decode(y_pred, input_length, greedy=True, beam_width=100, top_paths=1):
    _report_unimplemented('ctc_decode')


@_log_call
def ctc_label_dense_to_sparse(labels, label_lengths):
    _report_unimplemented('ctc_label_dense_to_sparse')


cumprod = op.cumulative_prod

cumsum = op.cumulative_sum


@_log_call
def depthwise_conv2d(x,
                     kernel,
                     strides=(1, 1),
                     padding='valid',
                     data_format=None,
                     dilation_rate=(1, 1)):
    if data_format is None:
        data_format = image_data_format()
    return conv(x, kernel, strides, padding, data_format, dilation_rate, channelwise=True)


dot = op.dot


@_log_call
def dropout(x, level, noise_shape=None, seed=None):
    if noise_shape is not None and len(noise_shape) != x.shape.ndims:
        raise ValueError("Length of noise_shape doesn't match input ndims")

    rng_state = _make_rng_state(seed)
    szs = ', '.join(['S' + str(i) for i in range(x.shape.ndims)])
    if noise_shape is None:
        args = 'I, ' + szs
    else:
        ishape = x.shape.dims
        args = ', '.join(['I'] + [
            "S{}".format(i) if v == ishape[i] or v in (None, -1) else "1"
            for i, v in enumerate(noise_shape)
        ])
    rng_step = 'function (I, X[{szs}]) -> (O) {{ O = prng_step({args}); }}'.format(szs=szs,
                                                                                   args=args)
    rng_value = """function (I, X, L) -> (O) {
        R = 1.0 - L;
        M = 1.0 / R;
        O = (prng_value(I) < R ? X * M : 0.0);
    }"""

    t = ptile.Operation(rng_step, [('I', rng_state), ('X', x)],
                        [('O', ptile.Shape(plaidml.DType.UINT32, tuple()))],
                        name='PrngStep').sole_output()
    n = ptile.Operation('function (I) -> (O) { O = prng_state(I); }', [('I', t)],
                        [('O', ptile.Shape(plaidml.DType.UINT32, (3, _k_rng_size)))],
                        name='PrngState').sole_output()
    o = ptile.Operation(rng_value, [('I', t), ('X', x), ('L', level)],
                        [('O', ptile.Shape(plaidml.DType.FLOAT32, x.shape.dims))],
                        side_effects=[(rng_state, n)],
                        name='PrngValue').sole_output()

    return o


@_log_call
def dtype(x):
    return ptile.convert_pml_dtype_to_np(x.shape.dtype)


@_log_call
def elu(x, alpha=1.0):
    return op.elu(x, alpha)


@_log_call
def eval(x):
    return get_value(x)


@_log_call
def equal(x, y):
    return op.equal(x, y)


exp = op.exp


@_log_call
def eye(size, dtype=None, name=None):
    if dtype is None:
        dtype = floatx()
    elif isinstance(dtype, plaidml.DType):
        dtype = ptile.convert_pml_dtype_to_np(dtype)
    return variable(np.eye(size, dtype=dtype), name=name, dtype=dtype)


class ExpandDims(ptile.Operation):

    def __init__(self, x, axis=-1, name=None):
        if axis < 0:
            axis = x.shape.ndims + 1 + axis
        slist_in = ['S' + str(i) for i in range(x.shape.ndims)]
        ilist_in = ['i' + str(i) for i in range(x.shape.ndims)]
        slist_out = slist_in[0:axis] + ['1'] + slist_in[axis:]
        ilist_out = ilist_in[0:axis] + ['0'] + ilist_in[axis:]
        newdims = tuple(list(x.shape.dims[0:axis]) + [
            1,
        ] + list(x.shape.dims[axis:]))
        f = """
            function (IN[{slist_in}]) -> (OUT) {{
                OUT[{ilist_out} : {slist_out}] = =(IN[{ilist_in}]);
            }}""".format(slist_in=', '.join(slist_in),
                         slist_out=', '.join(slist_out),
                         ilist_in=', '.join(ilist_in),
                         ilist_out=', '.join(ilist_out))
        super(ExpandDims, self).__init__(f, [('IN', x)],
                                         [('OUT', ptile.Shape(x.shape.dtype, newdims))],
                                         name=name)


expand_dims = ExpandDims.function

flatten = op.flatten

floor = op.floor


@_log_call
def foldl(fn, elems, initializer=None, name=None):
    _report_unimplemented('foldl')


@_log_call
def foldr(fn, elems, initializer=None, name=None):
    _report_unimplemented('foldr')


@_log_call
def function(inputs, outputs, updates=None, name=None):
    if updates == None:
        updates = []
    if name == None:
        name = ''
    return _Function(inputs, outputs, updates, name)


gather = op.gather


@_log_call
def get_variable_shape(x):
    return x._keras_shape


shape = op.shape_of


@_log_call
def get_uid(prefix=''):
    _UID_PREFIX_DICT[prefix] += 1
    return _UID_PREFIX_DICT[prefix]


@_log_call
def get_value(x):
    func = ptile.compose(_ctx, _device(), [], [('out', x)], name='get_value')
    invoker = plaidml.Invoker(_ctx, func)
    shape = invoker.get_output_shape('out')
    tensor = plaidml.Tensor(_device(), shape)
    invoker.set_output('out', tensor)
    invoker.invoke()
    out_shape = tuple(x.size for x in shape.dimensions)
    array = np.ndarray(out_shape, dtype=ptile.convert_pml_dtype_to_np(x.shape.dtype))
    with tensor.mmap_current() as view:
        view.copy_to_ndarray(array)
    return array


gradients = op.gradients


@_log_call
def greater(x, y):
    return x > y


@_log_call
def greater_equal(x, y):
    return x >= y


@_log_call
def hard_sigmoid(x):
    f = 'function (X) -> (R) { R = (X < -2.5 ? 0 : (X > 2.5 ? 1 : 0.2 * X + 0.5)); }'
    return ptile.Operation(f, [('X', x)], [('R', x.shape)], name='HardSigmoid').sole_output()


identity = op.identity


@_log_call
def in_test_phase(x, alt, training=None):
    # Note that this flips 'alt' and 'x'
    return in_train_phase(alt, x, training=training)


@_log_call
def in_top_k(predictions, targets, k):
    _report_unimplemented('in_top_k')


@_log_call
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


@_log_call
def int_shape(x):
    return tuple(None if isinstance(dim, ptile.Value) else dim for dim in x.shape.dims)


@_log_call
def is_keras_tensor(x):
    if not is_tensor(x):
        raise ValueError('Unexpectedly found an instance of type `' + str(type(x)) + '`. '
                         'Expected a symbolic tensor instance.')
    return hasattr(x, '_keras_history')


@_log_call
def is_placeholder(x):
    if isinstance(x, ptile.Value) and x.var and isinstance(x.var, plaidml.Placeholder):
        return True
    return False


@_log_call
def is_sparse(x):
    return False


@_log_call
def is_tensor(x):
    return isinstance(x, ptile.Value)


@_log_call
def l2_normalize(x, axis):
    norm = sqrt(sum(square(x), axis=axis, keepdims=True))
    return x / norm


@_log_call
def learning_phase():
    # Initialize _in_train_phase if this is the first use
    global _in_train_phase
    if _in_train_phase is None:
        _in_train_phase = placeholder(ndim=0, dtype='bool')
    return _in_train_phase


@_log_call
def less(x, y):
    return x < y


@_log_call
def less_equal(x, y):
    return x <= y


@_log_call
def local_conv1d(inputs, kernel, kernel_size, strides, data_format=None):
    _report_unimplemented('local_conv1d')


@_log_call
def local_conv2d(inputs, kernel, kernel_size, strides, output_shape, data_format=None):
    _report_unimplemented('local_conv2d')


log = op.log


@_log_call
def logsumexp(x, axis=None, keepdims=False):
    return log(sum(exp(x), axis=axis, keepdims=keepdims))


@_log_call
def manual_variable_initialization(value):
    _report_unimplemented('manual_variable_initialization')


@_log_call
def map_fn(fn, elems, name=None, dtype=None):
    _report_unimplemented('map_fn')


@_log_call
def max(x, axis=None, keepdims=False):
    return op.max_reduce(x, axes=axis, keepdims=keepdims)


maximum = op.maximum


@_log_call
def mean(x, axis=None, keepdims=False):
    return op.mean(x, axes=axis, keepdims=keepdims, floatx=ptile.convert_np_dtype_to_pml(floatx()))


@_log_call
def min(x, axis=None, keepdims=False):
    return op.min_reduce(x, axes=axis, keepdims=keepdims)


minimum = op.minimum


@_log_call
def moving_average_update(x, value, momentum):
    return (x, x * momentum + value * (1. - momentum))


_NAME_SCOPE_STACK = []


@_log_call
def _prepend_name_scope(name, default):
    global _NAME_SCOPE_STACK
    if name is None:
        r = '/'.join(_NAME_SCOPE_STACK + [default])
        r += '_' + str(get_uid(r))
    else:
        r = '/'.join(_NAME_SCOPE_STACK + [name])
    return r


@contextmanager
def name_scope(name):
    global _NAME_SCOPE_STACK
    _NAME_SCOPE_STACK.append(name)
    yield
    _NAME_SCOPE_STACK.pop()


@_log_call
def cur_name():
    if len(_NAME_SCOPE_STACK):
        return _NAME_SCOPE_STACK[0]
    return ''


@_log_call
def ndim(x):
    return len(x._keras_shape)


not_equal = op.not_equal


@_log_call
def normalize_batch_in_training(x, gamma, beta, reduction_axes, epsilon=1e-3):
    rank = x.shape.ndims
    if rank == 4 and reduction_axes in [[0, 1, 2], [0, 2, 3]]:
        # NOTE: Tensorflow's code is explicitly checking for reduction axes in
        # the case where there are 4 dims in x. If the reduction axes are passed
        # in explicitly when a layer is created, the broadcasting behavior
        # differs.
        xdims = x.shape.dims
        target_shape = [1 if i in reduction_axes else xdims[i] for i in range(4)]
        m = mean(x, axis=reduction_axes, keepdims=True)
        m = reshape(m, target_shape)
        v = var(x, axis=reduction_axes, keepdims=True)
        v = reshape(v, target_shape)
        if beta is not None:
            beta = reshape(beta, target_shape)
        if gamma is not None:
            gamma = reshape(gamma, target_shape)
    else:
        if reduction_axes == None:
            axes = [rank - 1]
        else:
            axes = reduction_axes

        # Will need to squeeze axes in order, so make sure none are negative and
        # sort
        axes = [i + rank if i < 0 else i for i in axes]
        for i in axes:
            if i < 0:
                raise ValueError(('Unexpected axis \'{}\' in normalize_batch_in' +
                                  ' training (tensor dim {})').format(i - rank, rank))
            if i >= rank:
                raise ValueError(('Unexpected axis \'{}\' in normalize_batch_in' +
                                  ' training (tensor dim {})').format(i, rank))
        axes.sort()

        # Mean and var need to keepdims for computing normalized_tensor, but
        # their returned values need to not keepdims. So keepdims for now, then
        # squeeze.
        m = mean(x, axis=axes, keepdims=True)
        v = var(x, axis=axes, keepdims=True)

    # TODO: Tensorflow's code implies using anything other than the single
    # final axis as the sole element of axis requires broadcasting,
    # but I don't see it ...
    # Indeed, this passes unit tests with a non-final axis selected
    normalized_tensor = batch_normalization(x=x,
                                            mean=m,
                                            var=v,
                                            beta=beta,
                                            gamma=gamma,
                                            epsilon=epsilon)

    m = squeeze(m)
    v = squeeze(v)

    return normalized_tensor, m, v


class OneHot(ptile.Operation):

    def __init__(self, indices, num_classes):
        #Note: does not error check for entries in indices that are >= num_classes

        count = variable(np.array(range(num_classes)), dtype='int32')
        f = """
            function (Idx[{idim}], Count[C]) -> (O) {{
                O[{iidx}, c : {idim}, C] = =(Idx[{iidx}] == Count[c]);
            }}""".format(idim=', '.join(['I{}'.format(k) for k in range(indices.shape.ndims)]),
                         iidx=', '.join(['i{}'.format(k) for k in range(indices.shape.ndims)]))

        outshape = ptile.Shape(ptile.convert_np_dtype_to_pml(floatx()),
                               tuple(list(indices.shape.dims) + [num_classes]))

        super(OneHot, self).__init__(f, [('Idx', indices), ('Count', count)], [('O', outshape)])


one_hot = OneHot.function


@_log_call
def ones(shape, dtype=None, name=None):
    dtype = dtype or floatx()
    return constant(1.0, shape=shape, dtype=dtype, name=_prepend_name_scope(name, 'ones'))


@_log_call
def ones_like(x, dtype=None, name=None):
    dtype = dtype or floatx()
    a_one = constant(1.0, shape=(1), dtype=dtype, name=_prepend_name_scope(name, 'a_one'))
    ndims = x.shape.ndims
    sizes = ', '.join(['S' + str(i) for i in range(ndims)])
    dims = ', '.join(['i' + str(i) for i in range(ndims)])
    f = """
        function (IN[{sizes}], ONE[SZ]) -> (OUT) {{
            OUT[{dims} : {sizes}] = =(ONE[0]);
        }}""".format(sizes=sizes, dims=dims)
    return ptile.Operation(f, [('IN', x), ('ONE', a_one)],
                           [('OUT', ptile.Shape(ptile.convert_np_dtype_to_pml(dtype), x.shape.dims))],
                           name='OnesLike') \
                .sole_output()


@_log_call
def permute_dimensions(x, pattern):
    return ptile.Operation(
        """function (X[{src_ranges}]) -> (R) {{
               R[{dest_indices} : {dest_ranges}] = =(X[{src_indices}]);
           }}""".format(
            src_ranges=', '.join(['X{}'.format(i) for i in range(x.shape.ndims)]),
            src_indices=', '.join(['x{}'.format(i) for i in range(x.shape.ndims)]),
            dest_ranges=', '.join(['X{}'.format(pattern[i]) for i in range(x.shape.ndims)]),
            dest_indices=', '.join(['x{}'.format(pattern[i]) for i in range(x.shape.ndims)])),
        [('X', x)], [
            ('R',
             ptile.Shape(x.shape.dtype,
                         tuple(x.shape.dims[pattern[idx]] for idx in range(x.shape.ndims))))
        ]).sole_output()


@_log_call
def placeholder(shape=None, ndim=None, dtype=None, sparse=False, name=None):
    dtype = ptile.convert_np_dtype_to_pml(dtype or floatx())
    if shape is not None:
        return ptile.Value.from_dimensions(shape,
                                           dtype,
                                           name=_prepend_name_scope(name, 'placeholder'))
    elif ndim is not None:
        return ptile.Value.from_ndims(ndim, dtype, name=_prepend_name_scope(name, 'placeholder'))
    else:
        raise PlaidMLKerasException('Specify either a shape or ndim value for placeholder.')


@_log_call
def pool(x, pool_size, strides=None, padding='valid', data_format=None, pool_mode='max'):
    if strides is None:
        strides = tuple(1 for _ in range(rank))
    if data_format is None:
        data_format = image_data_format()
    try:
        data_format = _POOL_DATA_FORMAT[data_format]
    except KeyError:
        six.raise_from(ValueError('Unrecognized data format: {}'.format(data_format)), None)
    try:
        pool_mode = _POOL_MODE[pool_mode]
    except KeyError:
        six.raise_from(ValueError('Unrecognized pool mode: {}'.format(pool_mode)), None)
    try:
        padding = _AUTO_PAD[padding]
    except KeyError:
        six.raise_from(ValueError('Unrecognized padding: {}'.format(padding)), None)
    return op.pool(data=x,
                   mode=pool_mode,
                   kernel_shape=pool_size,
                   strides=strides,
                   padding=padding,
                   data_format=data_format,
                   name=cur_name())


@_log_call
def pool2d(x, pool_size, strides=(1, 1), padding='valid', data_format=None, pool_mode='max'):
    return pool(x,
                pool_size,
                strides=strides,
                padding=padding,
                data_format=data_format,
                pool_mode=pool_mode)


@_log_call
def pool3d(x, pool_size, strides=(1, 1, 1), padding='valid', data_format=None, pool_mode='max'):
    return pool(x,
                pool_size,
                strides=strides,
                padding=padding,
                data_format=data_format,
                pool_mode=pool_mode)


@_log_call
def pow(x, a):
    if not isinstance(x, ptile.Value):
        x = variable(x)
    return op.pow(x, a)


@_log_call
def print_tensor(x, message=''):
    _report_unimplemented('print_tensor')


@_log_call
def prod(value, axis=None, keepdims=False):
    return op.prod(value,
                   axes=axis,
                   keepdims=keepdims,
                   floatx=ptile.convert_np_dtype_to_pml(floatx()))


@_log_call
def random_binomial(shape, p=0.0, dtype=None, see=None):
    _report_unimplemented('random_binomial')


@_log_call
def random_normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
    if dtype is None:
        dtype = floatx()
    if seed:
        np.random.seed(seed)
    # TODO: We only use half of the Box-Muller here
    u1 = random_uniform(shape, dtype='float32')
    u2 = random_uniform(shape, dtype='float32')
    z0 = op.sqrt(-2.0 * op.log(u1 + (1.0 / (2**33)))) * op.cos(2.0 * math.pi * u2)
    z0 = stddev * z0
    z0 = z0 + mean
    if dtype != 'float32':
        z0 = cast(z0, dtype)
    return z0


@_log_call
def random_normal_variable(shape, mean, scale, dtype=None, name=None, seed=None):
    if dtype is None:
        dtype = floatx()
    elif isinstance(dtype, plaidml.DType):
        dtype = ptile.convert_pml_dtype_to_np(dtype)
    if seed:
        np.random.seed(seed)
    data = np.random.normal(mean, scale, shape).astype(dtype)
    return variable(data, dtype=dtype, name=name)


@_log_call
def random_uniform(shape, minval=0.0, maxval=1.0, dtype=None, seed=None):
    if minval == maxval:
        return constant(minval, dtype, shape)

    dtype = dtype or floatx()
    rng_state = _make_rng_state(seed)
    shape_inputs = []
    shape_vars = []
    shape_args = []
    for idx, value in enumerate(shape):
        if isinstance(value, ptile.Value):
            shape_var = 'S{}'.format(idx)
            shape_vars.append(shape_var)
            shape_args.append(shape_var)
            shape_inputs.append((shape_var, value))
        else:
            shape_args.append(str(value))
    t = ptile.Operation('function ({inputs}) -> (O) {{ O = prng_step({args}); }}'.format(
        inputs=', '.join(['I'] + shape_vars), args=', '.join(['I'] + shape_args)),
                        [('I', rng_state)] + shape_inputs,
                        [('O', ptile.Shape(plaidml.DType.UINT32, tuple()))],
                        name='PrngStep').sole_output()
    n = ptile.Operation('function (I) -> (O) { O = prng_state(I); }', [('I', t)],
                        [('O', ptile.Shape(plaidml.DType.UINT32, (3, _k_rng_size)))],
                        name='PrngState').sole_output()
    o = ptile.Operation('function (I) -> (O) { O = prng_value(I); }', [('I', t)],
                        [('O', ptile.Shape(plaidml.DType.FLOAT32, shape))],
                        side_effects=[(rng_state, n)],
                        name='PrngValue').sole_output()

    if dtype != 'float32':
        o = cast(o, dtype)

    o = (maxval - minval) * o
    o = o + minval

    return o


@_log_call
def random_uniform_variable(shape, low, high, dtype=None, name=None, seed=None):
    if seed:
        np.random.seed(seed)
    val = np.random.uniform(low=low, high=high, size=shape)
    return variable(val, dtype=dtype)


relu = op.relu


@_log_call
def repeat(x, n):
    assert x.shape.ndims == 2
    code = """
           function (I[N0, N1]) -> (O) {{
               O[i0, r, i1: N0, {reps}, N1] = =(I[i0, i1]);
           }}""".format(reps=n)
    return ptile.Operation(code, [('I', x)],
                           [('O', ptile.Shape(x.shape.dtype,
                                              (x.shape.dims[0], n, x.shape.dims[1])))],
                           name='Repeat').sole_output()


@_log_call
def repeat_elements(x, rep, axis):
    if x.shape.dims[axis] is None:
        # Note: other backends just raise exception in this case
        out_shape = x.shape.dims[:axis] + (None,) + x.shape.dims[axis + 1:]
    else:
        out_shape = x.shape.dims[:axis] + (rep * x.shape.dims[axis],) + x.shape.dims[axis + 1:]
    idim_list = ['N{}'.format(i) for i in range(x.shape.ndims)]
    iidx_list = [N.lower() for N in idim_list]
    odim_list = [
        '{}*N{}'.format(rep, i) if i == axis else 'N{}'.format(i) for i in range(x.shape.ndims)
    ]
    oidx_list = [
        '{}*n{} + k'.format(rep, i) if i == axis else 'n{}'.format(i) for i in range(x.shape.ndims)
    ]

    # Example
    # function(I[N0, N1, N2]) -> (O) {
    #   O[n0, 3*n1 + k, n2 : N0, 3*N1, N2] = =(I[n0, n1, n2]), k < 3 no_defract;
    # }
    f = """
        function (I[{idims}]) -> (O) {{
            O[{oidxs} : {odims}] = =(I[{iidxs}]), k < {rep} no_defract;
        }}""".format(idims=', '.join(idim_list),
                     iidxs=', '.join(iidx_list),
                     odims=', '.join(odim_list),
                     oidxs=', '.join(oidx_list),
                     rep=str(rep))
    return ptile.Operation(f, [('I', x)], [('O', ptile.Shape(x.shape.dtype, out_shape))],
                           name='RepeatElements') \
                           .sole_output()


@_log_call
def reset_uids():
    global _UID_PREFIX_DICT
    _UID_PREFIX_DICT.clear()


reshape = op.reshape


@_log_call
def resize_images(x, height_factor, width_factor, data_format, interpolation='nearest'):
    if not isinstance(height_factor, int) or not isinstance(width_factor, int):
        raise ValueError(
            'height_factor and width_factor must be integers, received types {} and {}'.format(
                type(height_factor), type(width_factor)))
    if height_factor <= 0 or width_factor <= 0:
        raise ValueError(
            'height_factor and width_factor must be positive, received {} and {}'.format(
                height_factor, width_factor))
    if interpolation == 'nearest':
        if data_format == 'channels_first':
            ret = repeat_elements(x, height_factor, axis=2)
            ret = repeat_elements(ret, width_factor, axis=3)
        elif data_format == 'channels_last':
            ret = repeat_elements(x, height_factor, axis=1)
            ret = repeat_elements(ret, width_factor, axis=2)
        else:
            raise ValueError('Invalid data_format {}'.format(data_format))
    elif interpolation == 'bilinear':
        # This aligns the corners to (0, 0) and ({hf}*(H-1),{wf}*(H-1)), and assumes zero-padding beyond the top,
        # which is a bit weird, but it's easy to code and the weirdness is probably mostly irrelevant for ML.
        # Could eke out a tiny bit more perf by precomputing K instead of doing it in Tile
        if data_format == 'channels_first':
            idims = 'N, C, H, W'
            odims = 'N, C, HFactor*H, WFactor*W'
            iidxs = 'n, c, h, w'
            oidxs = 'n, c, HFactor*h + j - HFactor + 1, WFactor*w + i - WFactor + 1'
            outshape = ptile.Shape(x.shape.dtype, [
                x.shape.dims[0], x.shape.dims[1], height_factor * x.shape.dims[2],
                width_factor * x.shape.dims[3]
            ])
        elif data_format == 'channels_last':
            idims = 'N, H, W, C'
            odims = 'N, HFactor*H, WFactor*W, C'
            iidxs = 'n, h, w, c'
            oidxs = 'n, HFactor*h + j - HFactor + 1, WFactor*w + i - WFactor + 1, c'
            outshape = ptile.Shape(x.shape.dtype, [
                x.shape.dims[0], height_factor * x.shape.dims[1], width_factor * x.shape.dims[2],
                x.shape.dims[3]
            ])
        else:
            raise ValueError('Invalid data_format {}'.format(data_format))
        HBase = constant(1. / height_factor, shape=(height_factor,))
        WBase = constant(1. / width_factor, shape=(width_factor,))
        code = '''
        function (I[{idims}], HBase[HFactor], WBase[WFactor]) -> (O) {{
            HK[y : 2*HFactor - 1] = +(HBase[y + j - HFactor + 1]), j < HFactor;
            WK[x : 2*WFactor - 1] = +(WBase[x + i - WFactor + 1]), i < WFactor;
            K[y, x: 2*HFactor - 1, 2*WFactor - 1] = =(HK[y] * WK[x]);
            O[{oidxs} : {odims}] = +(I[{iidxs}] * K[j, i]);
        }}
        '''.format(
            idims=idims,
            odims=odims,
            iidxs=iidxs,
            oidxs=oidxs,
        )
        ret = ptile.Operation(code, [('I', x), ('HBase', HBase), ('WBase', WBase)],
                              [('O', outshape)]).sole_output()
    else:
        raise ValueError('Invalid interpolation mode {}'.format(interpolation))
    return ret


@_log_call
def resize_volumes(x, depth_factor, height_factor, width_factor, data_format):
    if data_format == 'channels_first':
        ret = repeat_elements(x, depth_factor, axis=2)
        ret = repeat_elements(ret, height_factor, axis=3)
        ret = repeat_elements(ret, width_factor, axis=4)
    elif data_format == 'channels_last':
        ret = repeat_elements(x, depth_factor, axis=1)
        ret = repeat_elements(ret, height_factor, axis=2)
        ret = repeat_elements(ret, width_factor, axis=3)
    else:
        raise ValueError('Invalid data_format {}'.format(data_format))
    return ret


@_log_call
def reverse(x, axes):
    if isinstance(axes, int):
        axes = [axes]
    for axis in axes:
        if not isinstance(axis, int):
            raise ValueError(
                'The axes parameter of reverse only accepts an integer or a list of integers, received {}'
                .format(type(axis)))
        if axis >= x.shape.ndims or axis < -x.shape.ndims:
            raise ValueError('Invalid axis {} in reverse: target {} too short (ndim={})'.format(
                axis, x, x.shape.ndims))
    axes = [a % x.shape.ndims for a in axes]
    dims = ', '.join('N{}'.format(j) for j in range(x.shape.ndims))
    in_idxs = ', '.join('i{}'.format(j) for j in range(x.shape.ndims))
    out_idxs = ', '.join(
        ('N{j} - 1 - i{j}' if j in axes else 'i{j}').format(j=j) for j in range(x.shape.ndims))
    f = """
        function (I[{dims}]) -> (O) {{
            O[{out_idxs}: {dims}] = =(I[{in_idxs}]);
        }}""".format(dims=dims, out_idxs=out_idxs, in_idxs=in_idxs)

    return ptile.Operation(f, [('I', x)], [('O', x.shape)], name='Reverse').sole_output()


@_log_call
def reverse_gradient(x, coeff=1.0):
    return ptile.binary_op(x, coeff, 'reverse_grad(L, R)', name='ReverseGradient')


@_log_call
def rnn(step_function,
        inputs,
        initial_states,
        go_backwards=False,
        mask=None,
        constants=None,
        unroll=False,
        input_length=None):
    if input_length is None:
        input_length = inputs.shape.dims[1]
    if isinstance(input_length, ptile.Value):
        raise NotImplementedError('rnn is not implemented for variable sized inputs')
    if mask is not None:
        raise NotImplementedError('rnn is not implemented with mask support')
    if constants is None:
        constants = list()

    def time_expand(val, ii, t, prev):
        if (len(val.shape.dims) < 1):
            raise PlaidMLKerasException('output values must have a batch size dimension')
        ndmo = len(val.shape.dims) - 1
        sizes = ', '.join(['N' + str(i) for i in range(ndmo)])
        idxs = ', '.join(['i' + str(i) for i in range(ndmo)])
        newshape = ptile.Shape(val.shape.dtype, (val.shape.dims[0], t) + val.shape.dims[1:])
        if prev is None:
            if ii != 0:
                raise RuntimeError(
                    "Generating RNN at time step {} with no previous time step".format(ii))
            f = "function (I[B, {sizes}]) -> (O) {{ O[b, 0, {idxs} : B, {T}, {sizes}] = =(I[b, {idxs}]); }}"
            f = f.format(sizes=sizes, idxs=idxs, T=t)
            return ptile.Operation(f, [('I', val)], [('O', newshape)],
                                   name='TimeExpand').sole_output()
        else:
            f = "function (I[B, {sizes}], P) -> (O) {{ O[b, {ii}, {idxs} : B, {T}, {sizes}] = =(I[b, {idxs}]) default P; }}"
            f = f.format(sizes=sizes, idxs=idxs, ii=ii, T=t)
            return ptile.Operation(f, [('I', val), ('P', prev)], [('O', newshape)],
                                   name='TimeExpand').sole_output()

    states = initial_states
    output = None
    for i in range(input_length):
        if go_backwards:
            input_val = inputs[:, input_length - 1 - i]
        else:
            input_val = inputs[:, i]
        output_val, new_states = step_function(input_val, states + constants)
        output = time_expand(output_val, i, input_length, output)
        states = new_states

    return (output_val, output, states)


@_log_call
def round(x):
    return ptile.unary_op(x, 'round(I)', 'Round')


@_log_call
def separable_conv(x,
                   depthwise_kernel,
                   pointwise_kernel,
                   strides=None,
                   padding='valid',
                   data_format=None,
                   dilation_rate=None):
    if data_format is None:
        data_format = image_data_format()
    if pointwise_kernel.shape.dims[
            -2] != depthwise_kernel.shape.dims[-1] * depthwise_kernel.shape.dims[-2]:
        raise ValueError(
            ('Shape mismatch in separable convolution. Depthwise kernel input ' +
             'channel count must match pointwise kernel channel count times channel ' +
             'multiplier.\nReceived {} v {} * {} (from full shapes {} and ' + '{})').format(
                 pointwise_kernel.shape.dims[-2], depthwise_kernel.shape.dims[-2],
                 depthwise_kernel.shape.dims[-1], pointwise_kernel.shape, depthwise_kernel.shape))
    intermediate = conv(x,
                        depthwise_kernel,
                        strides=strides,
                        padding=padding,
                        data_format=data_format,
                        dilation_rate=dilation_rate,
                        channelwise=True)
    rank = x.shape.ndims - 2
    ones = tuple(1 for _ in range(rank))
    return conv(intermediate,
                pointwise_kernel,
                strides=ones,
                padding='valid',
                data_format=data_format,
                dilation_rate=ones)


@_log_call
def separable_conv2d(x,
                     depthwise_kernel,
                     pointwise_kernel,
                     strides=(1, 1),
                     padding='valid',
                     data_format=None,
                     dilation_rate=(1, 1)):
    return separable_conv(x, depthwise_kernel, pointwise_kernel, strides, padding, data_format,
                          dilation_rate)


@_log_call
def set_floatx(dtype):
    keras_set_floatx(dtype)
    plaidml.set_floatx(ptile.convert_np_dtype_to_pml(dtype))


@_log_call
def set_learning_phase(value):
    if value != 0 and value != 1:
        raise ValueError("May only set_learning_phase to 0 or 1")
    value = int(value)
    global _in_train_phase
    _in_train_phase = value


@_log_call
def set_value(x, value):
    if not isinstance(x.var, plaidml.Tensor):
        raise PlaidMLKerasException('Can only set values of tensors')

    try:
        if x.shape.dims != value.shape:
            raise NotImplementedError(
                'The PlaidML backend for Keras does not support changing tensor shapes with set_value.\n'
                + 'existing.shape = ' + str(x.shape) + ', value.shape = ' + str(value.shape))
    except AttributeError:
        if x.shape.dims != () and x.shape.dims != (1,):
            raise NotImplementedError(
                'The PlaidML backend for Keras does not support changing tensor shapes with set_value.\n'
                + 'existing.shape = ' + str(x.shape) + ', value is a non-array object of type: ' +
                str(type(value)))
    with x.var.mmap_discard(_ctx) as view:
        view.copy_from_ndarray(np.asarray(value))
        view.writeback()


sigmoid = op.sigmoid


@_log_call
def sign(x):
    return ptile.unary_op(x, "I == 0 ? 0 : (I > 0 ? 1 : -1)", name="Sign")


sin = op.sin


@_log_call
def softmax(x):
    return op.softmax(x, axis=x.shape.ndims - 1)


@_log_call
def softplus(x):
    return log(1. + exp(x))


@_log_call
def softsign(x):
    return x / (1 + abs(x))


@_log_call
def sparse_categorical_crossentropy(target, output, from_logits=False):
    return categorical_crossentropy(
        reshape(one_hot(target, output.shape.dims[-1]), output.shape.dims), output, from_logits)


class SpatialPadding(ptile.Operation):

    def __init__(self, x, padding, data_format=None):
        if data_format is None:
            data_format = image_data_format()
        rank = x.shape.ndims - 2
        if rank < 1:
            raise ValueError('Can only perform spatial padding on a tensor with at least one '
                             'spatial dimension.')
        if len(padding) != rank:
            raise ValueError('Failed to pad {} spatial dimensions; received padding '
                             'amounts for {} spatial dimensions'.format(rank, len(padding)))
        for pad_amount in padding:
            if len(pad_amount) != 2:
                raise ValueError('Expected padding to be tuple of {} length 2 tuples; '
                                 'received {}'.format(rank, padding))

        in_spatial_dims = []
        out_spatial_dims = []
        total_padding = []
        in_spatial_idxs = []
        out_spatial_idxs = []
        for i in range(rank):
            front_padding = padding[i][0]
            total_padding.append(padding[i][0] + padding[i][1])
            in_spatial_dims.append('D{}'.format(i))
            out_spatial_dims.append('D{} + {}'.format(i, total_padding[i]))
            in_spatial_idxs.append('d{} - {}'.format(i, front_padding))
            out_spatial_idxs.append('d{}'.format(i))
        if data_format == 'channels_last':
            in_dims = 'N, {}, C'.format(', '.join(in_spatial_dims))
            out_dims = 'N, {}, C'.format(', '.join(out_spatial_dims))
            in_idxs = 'n, {}, c'.format(', '.join(in_spatial_idxs))
            out_idxs = 'n, {}, c'.format(', '.join(out_spatial_idxs))
            numeric_spatial_out_dims = [
                x.shape.dims[i + 1] + total_padding[i] for i in range(rank)
            ]
            numeric_out_dims = tuple([x.shape.dims[0]] + numeric_spatial_out_dims +
                                     [x.shape.dims[-1]])
        elif data_format == 'channels_first':
            in_dims = 'N, C, {}'.format(', '.join(in_spatial_dims))
            out_dims = 'N, C, {}'.format(', '.join(out_spatial_dims))
            in_idxs = 'n, c, {}'.format(', '.join(in_spatial_idxs))
            out_idxs = 'n, c, {}'.format(', '.join(out_spatial_idxs))
            numeric_spatial_out_dims = [
                x.shape.dims[i + 2] + total_padding[i] for i in range(rank)
            ]
            numeric_out_dims = tuple([x.shape.dims[0], x.shape.dims[1]] + numeric_spatial_out_dims)
        else:
            raise ValueError('Unrecognized data_format {}'.format(data_format))
        f = ("""
            function (I[{in_dims}]) -> (O) {{
                O[{out_idxs} : {out_dims}] = =(I[{in_idxs}]);
            }}
        """).format(in_dims=in_dims, in_idxs=in_idxs, out_dims=out_dims, out_idxs=out_idxs)

        super(SpatialPadding, self).__init__(f, [('I', x)],
                                             [('O', ptile.Shape(x.shape.dtype, numeric_out_dims))])


@_log_call
def spatial_2d_padding(x, padding=((1, 1), (1, 1)), data_format=None):
    return SpatialPadding.function(x, padding, data_format)


@_log_call
def spatial_3d_padding(x, padding=((1, 1), (1, 1), (1, 1)), data_format=None):
    return SpatialPadding.function(x, padding, data_format)


@_log_call
def square(x):
    return x * x


@_log_call
def squeeze(x, axis=None):
    if axis is None:
        # define axis
        axis = []
        for s in range(len(x.shape.dims)):
            if x.shape.dims[s] == 1:
                axis.append(s)
    if isinstance(axis, list) and len(axis):
        x_squeezed = x
        for i in range(len(axis)):
            ax = axis[i] - i
            x_squeezed = squeeze_one(x_squeezed, ax)
        return x_squeezed
    else:
        return squeeze_one(x, axis)


@_log_call
def squeeze_one(x, axis):
    if x.shape.dims[axis] != 1:
        raise ValueError('Can only squeeze length 1 axis')
    if axis == -1:
        result = reshape(x, x.shape.dims[:axis])
    else:
        result = reshape(x, x.shape.dims[:axis] + x.shape.dims[axis + 1:])
    return result


sqrt = op.sqrt


@_log_call
def stack(x, axis=0):
    tshape = x[0].shape
    for item in x:
        if tshape != item.shape:
            raise ValueError("All inputs must have the same shape and type")
    nshape = list(tshape.dims)
    nshape.insert(axis if axis >= 0 else len(nshape), 1)
    return concatenate([reshape(item, nshape) for item in x], axis=axis)


@_log_call
def std(x, axis=None, keepdims=False):
    return sqrt(var(x, axis=axis, keepdims=keepdims))


@_log_call
def stop_gradient(variables):
    _report_unimplemented('stop_gradient')


@_log_call
def sum(x, axis=None, keepdims=False):
    return op.summation(x,
                        axes=axis,
                        keepdims=keepdims,
                        floatx=ptile.convert_np_dtype_to_pml(floatx()))


class Switch(ptile.Operation):

    def __init__(self, condition, then_expression, else_expression):
        super(Switch, self).__init__('function (C, T, E) -> (O) { O = (C ? T : E); }',
                                     [('C', condition), ('T', then_expression),
                                      ('E', else_expression)], [('O', then_expression.shape)])


switch = Switch.function

tanh = op.tanh


@_log_call
def temporal_padding(x, padding=(1, 1)):
    if x.shape.ndims != 3:
        raise ValueError('Can only perform temporal_padding on 3D tensor')
    # Temporal padding is channels_last 1D spatial padding
    return SpatialPadding.function(x, padding=(padding,), data_format='channels_last')


@_log_call
def tile(x, n):
    if len(n) != x.shape.ndims:
        raise PlaidMLKerasException('Tile size dimensions doesn\'t match ndims')
    sizes = ', '.join(['S' + str(i) for i in range(x.shape.ndims)])
    out_idx = ', '.join(
        ['t' + str(i) + ' * S' + str(i) + ' + i' + str(i) for i in range(x.shape.ndims)])
    out_sizes = ', '.join(['S' + str(i) + ' * ' + str(n[i]) for i in range(x.shape.ndims)])
    in_idx = ', '.join(['i' + str(i) for i in range(x.shape.ndims)])
    cons = ', '.join(['t' + str(i) + ' < ' + str(n[i]) for i in range(x.shape.ndims)])
    f = """
        function (I[{sizes}]) -> (O) {{
            O[{out_idx} : {out_sizes}] = =(I[{in_idx}]), {cons} no_defract;
        }}""".format(sizes=sizes, out_idx=out_idx, out_sizes=out_sizes, in_idx=in_idx, cons=cons)
    out_dims = tuple(x.shape.dims[i] * n[i] for i in range(x.shape.ndims))
    return ptile.Operation(f, [('I', x)], [('O', ptile.Shape(x.shape.dtype, out_dims))],
                           name='Tile').sole_output()


@_log_call
def to_dense(tensor):
    _report_unimplemented('to_dense')


@_log_call
def transpose(x):
    return permute_dimensions(x, range(x.shape.ndims - 1, -1, -1))


@_log_call
def truncated_normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
    if dtype is None:
        dtype = floatx()
    if seed:
        np.random.seed(seed)
    return variable(stddev * scipy.stats.truncnorm.rvs(-2.0, 2.0, size=shape) + mean, dtype)


@_log_call
def update(x, new_x):
    return (x, new_x)


@_log_call
def update_add(x, increment):
    return (x, x + increment)


@_log_call
def update_sub(x, decrement):
    return (x, x - decrement)


@_log_call
def var(x, axis=None, keepdims=False):
    return op.variance(x,
                       axes=axis,
                       keepdims=keepdims,
                       floatx=ptile.convert_np_dtype_to_pml(floatx()))


@_log_call
def variable(value, dtype=None, name=None, constraint=None):
    dtype = dtype or floatx()
    if constraint:
        raise PlaidMLKerasException('Unsupported variable constraint')
    if isinstance(value, float) or isinstance(value, six.integer_types):
        tensor = plaidml.Tensor(_device(), plaidml.Shape(_ctx,
                                                         ptile.convert_np_dtype_to_pml(dtype)))
        with tensor.mmap_discard(_ctx) as view:
            view.copy_from_ndarray(np.array(value))
            view.writeback()
        return ptile.Value.from_var(
            tensor, tuple(), ptile.convert_np_dtype_to_pml(dtype),
            _prepend_name_scope(name,
                                'float_variable' if isinstance(value, float) else 'int_variable'))
    elif isinstance(value, ptile.Value):
        func = ptile.compose(_ctx, _device(), [], [('out', value)], name='variable')
        invoker = plaidml.Invoker(_ctx, func)
        shape = invoker.get_output_shape('out')
        tensor = plaidml.Tensor(_device(), shape)
        invoker.set_output('out', tensor)
        invoker.invoke()
        return ptile.Value.from_var(tensor, [d.size for d in shape.dimensions], shape.dtype,
                                    _prepend_name_scope(name, 'variable'))
    elif isinstance(value, list) or isinstance(value, tuple):
        value = np.array(value)
        # Fallthrough
    # Default to treating the value as an ndarray.
    tensor = plaidml.Tensor(
        _device(), plaidml.Shape(_ctx, ptile.convert_np_dtype_to_pml(dtype), *value.shape))
    with tensor.mmap_discard(_ctx) as view:
        view.copy_from_ndarray(value)
        view.writeback()
    return ptile.Value.from_var(tensor, value.shape, ptile.convert_np_dtype_to_pml(dtype),
                                _prepend_name_scope(name, 'tensor_variable'))


@_log_call
def zeros(shape, dtype=floatx(), name=None):
    return constant(0.0, shape=shape, dtype=dtype, name=_prepend_name_scope(name, 'zeros'))


@_log_call
def zeros_like(x, dtype=floatx(), name=None):
    dtype = dtype or floatx()
    a_zero = constant(0.0, shape=(1), dtype=dtype, name=_prepend_name_scope(name, 'a_zero'))
    ndims = x.shape.ndims
    sizes = ', '.join(['S' + str(i) for i in range(ndims)])
    dims = ', '.join(['i' + str(i) for i in range(ndims)])
    f = """
        function (IN[{sizes}], ZERO[SZ]) -> (OUT) {{
            OUT[{dims} : {sizes}] = =(ZERO[0]);
        }}""".format(sizes=sizes, dims=dims)
    return ptile.Operation(f, [('IN', x), ('ZERO', a_zero)],
                           [('OUT', ptile.Shape(ptile.convert_np_dtype_to_pml(dtype), x.shape.dims))],
                           name='ZerosLike') \
                .sole_output()


# Dynamically add Keras functionality to the underlying tile.Value class.
# This allows us to transparently use Value as the tensor type exposed by
# the Keras backend; it's a little squirrelly in this one place, but it
# greatly simplifies the rest of this module.
@_log_call
def _get_keras_shape(x):
    try:
        return x.__keras_shape
    except AttributeError:
        return int_shape(x)


@_log_call
def _set_keras_shape(x, shape):
    x.__keras_shape = shape


ptile.Value._keras_shape = property(_get_keras_shape, _set_keras_shape)

ptile.Value.eval = get_value
