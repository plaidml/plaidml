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
import os
import plaidml
import plaidml.op as op
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
                                   list(zip(self._output_names, outputs)), updates)
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


def abs(x):
    return builtins.abs(x)


def all(x, axis=None, keepdims=False):
    _report_unimplemented('all')


def any(x, axis=None, keepdims=False):
    _report_unimplemented('any')


def arange(start, stop=None, step=1, dtype='int32'):
    _report_unimplemented('arange')


argmax = op.argmax


def argmin(x, axis=-1):
    return argmax(-x, axis=axis)


def backend():
    return 'plaidml'


class BatchDot(ptile.Operation):

    def __init__(self, x, y, axes=None, name=None):
        if isinstance(axes, int):
            axes = (axes, axes)
        if axes is None:
            axes = (x.shape.ndims - 1, y.shape.ndims - 2)
        out_dims = (x.shape.dims[:axes[0]] + x.shape.dims[axes[0] + 1:] + y.shape.dims[1:axes[1]] +
                    y.shape.dims[axes[1] + 1:])
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
            }}""".format(
            xdims=', '.join(xdim_list),
            ydims=', '.join(ydim_list),
            odims=', '.join(odim_list),
            xidxs=', '.join(xidx_list),
            yidxs=', '.join(yidx_list),
            oidxs=', '.join(oidx_list))

        super(BatchDot, self).__init__(
            f, [('X', x), ('Y', y)], [('O', ptile.Shape(x.shape.dtype, out_dims))], name=name)


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
                '}}').format(
                    idims=', '.join(in_dim_list), odims=', '.join(out_dim_list))
        super(BatchFlatten, self).__init__(code, [('I', x)], [('O', outshape)])


batch_flatten = BatchFlatten.function


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
    if data_format not in _CONV_DATA_FORMAT:
        raise PlaidMLKerasException('Unrecognized data_format given to bias_add: \'' + str(
            data_format) + '\'; only \'channels_first\' and \'channels_last\' recognized.')
    try:
        bias_dims = bias.shape.dims
    except AttributeError:
        bias_dims = bias.shape

    if ndim(x) > 2:
        if data_format == 'channels_first':
            x += reshape(bias, (1, bias_dims[0]) + (1,) * (ndim(x) - 2))
        elif data_format == 'channels_last':
            x += reshape(bias, (1,) * (ndim(x) - 1) + (bias_dims[0],))
    else:
        x += bias
    return x


def binary_crossentropy(target, output, from_logits=False):
    return op.binary_crossentropy(target, output, epsilon(), from_logits)


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
                }}""".format(
                fixed_dims=fixed_dims, fixed_idxs=fixed_idxs)

        super(CategoricalCrossentropy,
              self).__init__(code, [('O', output), ('T', target)],
                             [('R', ptile.Shape(output.shape.dtype, output.shape.dims[:-1]))])


categorical_crossentropy = CategoricalCrossentropy.function

ceil = op.ceiling


def clear_session():
    _report_unimplemented('clear_session')


clip = op.clip

concatenate = op.concatenate


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

    return op.convolution(
        x,
        kernel,
        padding=padding,
        dilation_rate=dilation_rate,
        strides=strides,
        data_format=data_format,
        channelwise=channelwise)


def conv_transpose(x, kernel, output_shape, strides, padding, data_format):
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

    return op.convolution_transpose(x, kernel, output_shape, strides, padding, data_format)


def _func_once(func):
    """A decorator that runs a function only once."""

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

    raise PlaidMLKerasException('Only support L(2, 3) and L(4, 3) right now')


def _winograd(x, kernel, padding='valid', block=6):
    (A, B, G) = _compute_transforms(block, kernel.shape.dims[0])
    s = kernel.shape.dims[0]
    (XO, XP, NXO) = op.pad_compute('X', x.shape.dims[1], s, 1, _AUTO_PAD[padding])
    (YO, YP, NYO) = op.pad_compute('Y', x.shape.dims[2], s, 1, _AUTO_PAD[padding])
    outdims = (x.shape.dims[0], NXO, NYO, kernel.shape.dims[3])
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

    return ptile.Operation(
        f, [('I', x), ('K', kernel), ('A', A), ('B', B), ('G', G)],
        [('O', ptile.Shape(x.shape.dtype, outdims))],
        name='Winograd').sole_output()


def conv1d(x, kernel, strides=1, padding='valid', data_format=None, dilation_rate=1):
    if padding == 'causal':
        left_pad = dilation_rate * (kernel.shape.dims[0] - 1)
        x = temporal_padding(x, (left_pad, 0))
        padding = 'valid'
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
        (data_format == 'channels_last' and kernel.shape.dims[0] == 3 and
         kernel.shape.dims[1] == 3 and strides == (1, 1) and dilation_rate == (1, 1) and
         kernel.shape.dims[2] > 4 and kernel.shape.dims[3] > 4)):
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
    for dim in x.shape.dims:
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


cumsum = op.cumulative_sum


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


def dropout(x, level, noise_shape=None, seed=None):
    if noise_shape is not None:
        raise PlaidMLKerasException('Unimplemented noise shape in dropout')

    rng_state = _make_rng_state(seed)

    szs = ', '.join(['S' + str(i) for i in range(x.shape.ndims)])
    args = ', '.join(['I'] + ['S' + str(i) for i in range(x.shape.ndims)])
    rng_step = 'function (I, X[{szs}]) -> (O) {{ O = prng_step({args}); }}'.format(
        szs=szs, args=args)
    rng_value = """function (I, X, L) -> (O) {
        R = 1.0 - L;
        M = 1.0 / R;
        O = (prng_value(I) < R ? X * M : 0.0);
    }"""

    t = ptile.Operation(
        rng_step, [('I', rng_state), ('X', x)],
        [('O', ptile.Shape(plaidml.DType.UINT32, tuple()))],
        name='PrngStep').sole_output()
    n = ptile.Operation(
        'function (I) -> (O) { O = prng_state(I); }', [('I', t)],
        [('O', ptile.Shape(plaidml.DType.UINT32, (3, _k_rng_size)))],
        name='PrngState').sole_output()
    o = ptile.Operation(
        rng_value, [('I', t), ('X', x), ('L', level)],
        [('O', ptile.Shape(plaidml.DType.FLOAT32, x.shape.dims))],
        side_effects=[(rng_state, n)],
        name='PrngValue').sole_output()

    return o


def dtype(x):
    return ptile.convert_pml_dtype_to_np(x.shape.dtype)


def elu(x, alpha=1.0):
    return op.elu(x, alpha)


def eval(x):
    return get_value(x)


def equal(x, y):
    return op.equal(x, y)


exp = op.exp


def eye(size, dtype=None, name=None):
    _report_unimplemented('eye')


pow = op.pow


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
            }}""".format(
            slist_in=', '.join(slist_in),
            slist_out=', '.join(slist_out),
            ilist_in=', '.join(ilist_in),
            ilist_out=', '.join(ilist_out))
        super(ExpandDims, self).__init__(
            f, [('IN', x)], [('OUT', ptile.Shape(x.shape.dtype, newdims))], name=name)


expand_dims = ExpandDims.function

flatten = op.flatten

floor = op.floor


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


gather = op.gather


def get_variable_shape(x):
    return x._keras_shape


shape = op.shape_of


def get_uid(prefix=''):
    _UID_PREFIX_DICT[prefix] += 1
    return _UID_PREFIX_DICT[prefix]


def get_value(x):
    func = ptile.compose(_ctx, _device(), [], [('out', x)])
    invoker = plaidml.Invoker(_ctx, func)
    shape = invoker.get_output_shape('out')
    tensor = plaidml.Tensor(_device(), shape)
    invoker.set_output('out', tensor)
    invoker.invoke()
    array = np.ndarray(x.shape.dims, dtype=ptile.convert_pml_dtype_to_np(x.shape.dtype))
    with tensor.mmap_current() as view:
        view.copy_to_ndarray(array)
    return array


gradients = op.gradients


def greater(x, y):
    return x > y


def greater_equal(x, y):
    return x >= y


def hard_sigmoid(x):
    f = 'function (X) -> (R) { R = (X < -2.5 ? 0 : (X > 2.5 ? 1 : 0.2 * X + 0.5)); }'
    return ptile.Operation(f, [('X', x)], [('R', x.shape)], name='HardSigmoid').sole_output()


identity = op.identity


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
    return tuple(None if isinstance(dim, ptile.Value) else dim for dim in x.shape.dims)


def is_keras_tensor(x):
    if not isinstance(x, ptile.Value):
        return False
    return hasattr(x, '_keras_history')


def is_placeholder(x):
    if isinstance(x, ptile.Value) and x.var and isinstance(x.var, plaidml.Placeholder):
        return True
    return False


def is_sparse(x):
    return False


def l2_normalize(x, axis):
    norm = sqrt(sum(square(x), axis=axis, keepdims=True))
    return x / norm


def learning_phase():
    # Initialize _in_train_phase if this is the first use
    global _in_train_phase
    if _in_train_phase is None:
        _in_train_phase = placeholder(ndim=0, dtype='bool')
    return _in_train_phase


def less(x, y):
    return x < y


def less_equal(x, y):
    return x <= y


def local_conv1d(inputs, kernel, kernel_size, strides, data_format=None):
    _report_unimplemented('local_conv1d')


def local_conv2d(inputs, kernel, kernel_size, strides, output_shape, data_format=None):
    _report_unimplemented('local_conv2d')


log = op.log


def logsumexp(x, axis=None, keepdims=False):
    _report_unimplemented('logsumexp')


def manual_variable_initialization(value):
    _report_unimplemented('manual_variable_initialization')


def map_fn(fn, elems, name=None, dtype=None):
    _report_unimplemented('map_fn')


def max(x, axis=None, keepdims=False):
    return op.max_reduce(x, axes=axis, keepdims=keepdims)


maximum = op.maximum


def mean(x, axis=None, keepdims=False):
    return op.mean(x, axes=axis, keepdims=keepdims, floatx=ptile.convert_np_dtype_to_pml(floatx()))


def min(x, axis=None, keepdims=False):
    return op.min_reduce(x, axes=axis, keepdims=keepdims)


minimum = op.minimum


def moving_average_update(x, value, momentum):
    return (x, x * momentum + value * (1. - momentum))


_NAME_SCOPE_STACK = []


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


def ndim(x):
    return len(x._keras_shape)


not_equal = op.not_equal


def normalize_batch_in_training(x, gamma, beta, reduction_axes, epsilon=1e-3):
    rank = x.shape.ndims
    if reduction_axes == None:
        axes = [rank - 1]
    else:
        axes = reduction_axes

    # Will need to squeeze axes in order, so make sure none are negative and sort
    axes = [i + rank if i < 0 else i for i in axes]
    for i in axes:
        if i < 0:
            raise ValueError(('Unexpected axis \'{}\' in normalize_batch_in training ' +
                              '(tensor dim {})').format(i - rank, rank))
        if i >= rank:
            raise ValueError(('Unexpected axis \'{}\' in normalize_batch_in training ' +
                              '(tensor dim {})').format(i, rank))
    axes.sort()

    # Mean and var need to keepdims for computing normalized_tensor, but their
    # returned values need to not keepdims. So keepdims for now, then squeeze.
    m = mean(x, axis=axes, keepdims=True)
    v = var(x, axis=axes, keepdims=True)

    # TODO: Tensorflow's code implies using anything other than the single
    # final axis as the sole element of axis requires broadcasting,
    # but I don't see it ...
    # Indeed, this passes unit tests with a non-final axis selected
    normalized_tensor = batch_normalization(
        x=x, mean=m, var=v, beta=beta, gamma=gamma, epsilon=epsilon)

    # Tensorflow and Theano disagree on whether mean and var should be squeezed
    # here. For now, going with Theano for simplicity.
    #  for ax in reversed(axes):
    #    m = squeeze(m, ax)
    #    v = squeeze(v, ax)

    return normalized_tensor, m, v


class OneHot(ptile.Operation):

    def __init__(self, indices, num_classes):
        #Note: does not error check for entries in indices that are >= num_classes

        count = variable(np.array(range(num_classes)), dtype='int32')
        f = """
            function (Idx[{idim}], Count[C]) -> (O) {{
                O[{iidx}, c : {idim}, C] = =(Idx[{iidx}] == Count[c]);
            }}""".format(
            idim=', '.join(['I{}'.format(k) for k in range(indices.shape.ndims)]),
            iidx=', '.join(['i{}'.format(k) for k in range(indices.shape.ndims)]))

        outshape = ptile.Shape(plaidml.DType.BOOLEAN,
                               tuple(list(indices.shape.dims) + [num_classes]))

        super(OneHot, self).__init__(f, [('Idx', indices), ('Count', count)], [('O', outshape)])


one_hot = OneHot.function


def ones(shape, dtype=None, name=None):
    dtype = dtype or floatx()
    return constant(1.0, shape=shape, dtype=dtype, name=_prepend_name_scope(name, 'ones'))


def ones_like(x, dtype=None, name=None):
    dtype = dtype or floatx()
    a_one = constant(1.0, shape=(1), dtype=dtype, name=_prepend_name_scope(name, 'a_one'))
    ndims = x.shape.ndims
    sizes = ', '.join(['S' + str(i) for i in range(ndims)])
    dims = ', '.join(['i' + str(i) for i in range(ndims)])
    f = """
        function (IN[{sizes}], ONE[SZ]) -> (OUT) {{
            OUT[{dims} : {sizes}] = =(ONE[0]);
        }}""".format(
        sizes=sizes, dims=dims)
    return ptile.Operation(f, [('IN', x), ('ONE', a_one)],
                           [('OUT', ptile.Shape(ptile.convert_np_dtype_to_pml(dtype), x.shape.dims))],
                           name='OnesLike') \
                .sole_output()


def permute_dimensions(x, pattern):
    return ptile.Operation("""function (X[{src_ranges}]) -> (R) {{
               R[{dest_indices} : {dest_ranges}] = =(X[{src_indices}]);
           }}""".format(
        src_ranges=', '.join(['X{}'.format(i) for i in range(x.shape.ndims)]),
        src_indices=', '.join(['x{}'.format(i) for i in range(x.shape.ndims)]),
        dest_ranges=', '.join(['X{}'.format(pattern[i]) for i in range(x.shape.ndims)]),
        dest_indices=', '.join(['x{}'.format(
            pattern[i]) for i in range(x.shape.ndims)])), [('X', x)], [('R', ptile.Shape(
                x.shape.dtype, tuple(x.shape.dims[pattern[idx]] for idx in range(x.shape.ndims))))
                                                                      ]).sole_output()


def placeholder(shape=None, ndim=None, dtype=None, sparse=False, name=None):
    dtype = ptile.convert_np_dtype_to_pml(dtype or floatx())
    if shape is not None:
        return ptile.Value.from_dimensions(shape, dtype, name=name)
    elif ndim is not None:
        return ptile.Value.from_ndims(ndim, dtype, name=name)
    else:
        raise PlaidMLKerasException('Specify either a shape or ndim value for placeholder.')


class Pool(ptile.Operation):

    def __init__(self,
                 x,
                 pool_size,
                 strides=None,
                 padding='valid',
                 data_format=None,
                 pool_mode='max'):
        try:
            padding = _AUTO_PAD[padding]
        except KeyError:
            six.raise_from(ValueError('Unrecognized padding: {}'.format(padding)), None)

        # TODO: There are major similarities between pool and conv. I think keeping
        # them separate makes sense, but we could consider merging them.
        rank = x.shape.ndims - 2
        if strides is None:
            strides = tuple(1 for _ in range(rank))
        if data_format is None:
            data_format = image_data_format()

        if len(pool_size) != rank:
            raise ValueError(
                'Pool size inconsistent with input shape: ' + '{} (rank {}) v {} (rank {})'.format(
                    pool_size, len(pool_size), x.shape, x.shape.ndims - 2))
        if len(strides) != rank:
            raise ValueError('Pool strides length inconsistent with input shape: ' +
                             '{} (rank {}) v {} (rank {})'.format(
                                 strides, len(strides), x.shape.dims, x.shape.ndims - 2))

        if data_format == 'channels_first':
            n = 0
            c = 1
            l = [i + 2 for i in range(rank)]
        elif data_format == 'channels_last':
            n = 0
            l = [i + 1 for i in range(rank)]
            c = rank + 1
        else:
            raise ValueError('Unrecognized data format \'{}\''.format(data_format))

        out_size = list()
        pad_amount = list()
        num_out_size = list()
        for i in range(rank):
            sym_out, sym_pad, num_out = op.pad_compute('L{}'.format(i), x.shape.dims[l[i]],
                                                       pool_size[i], strides[i], padding)
            out_size.append(sym_out)
            pad_amount.append(sym_pad)
            num_out_size.append(num_out)
        padding_list = ['  Pad{} = {};'.format(i, pad_amount[i]) for i in range(rank)]
        padding_str = '\n'.join(padding_list)
        input_idx_list = [
            '{}*{} + {} - {}'.format(strides[i], 'x{}'.format(i), 'k{}'.format(i),
                                     'Pad{}'.format(i)) for i in range(rank)
        ]
        pool_bounds = ', ' + ', '.join(['k{} < {}'.format(i, pool_size[i]) for i in range(rank)])
        if data_format == 'channels_first':
            input_dims_str = 'N, C, ' + ', '.join(['L{}'.format(i) for i in range(rank)])
            out_idx_str = 'n, c, ' + ', '.join(['x{}'.format(i) for i in range(rank)])
            out_dims_str = 'N, C, ' + ', '.join(['{}'.format(out_size[i]) for i in range(rank)])
            input_idx_str = 'n, c, ' + ', '.join(input_idx_list)
            outshape = list(x.shape.dims[:2]) + num_out_size
        elif data_format == 'channels_last':
            input_dims_str = 'N, ' + ', '.join(['L{}'.format(i) for i in range(rank)]) + ', C'
            out_idx_str = 'n, ' + ', '.join(['x{}'.format(i) for i in range(rank)]) + ', c'
            out_dims_str = 'N, ' + ', '.join(['{}'.format(out_size[i])
                                              for i in range(rank)]) + ', C'
            input_idx_str = 'n, ' + ', '.join(input_idx_list) + ', c'
            outshape = [x.shape.dims[0]] + num_out_size + [x.shape.dims[-1]]
        else:
            raise ValueError('Unrecognized data format \'{}\''.format(data_format))
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
            raise ValueError('Unrecognized pool mode \'{}\''.format(pool_mode))

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

        name = 'pool{}d'.format(rank)
        if pool_mode == 'max':
            inputs = [('I', x)]
        elif pool_mode == 'avg':
            inputs = [('I', x), ('Ones', ones)]
        else:
            raise ValueError('Unrecognized pool mode \'{}\''.format(pool_mode))

        super(Pool, self).__init__(
            f, inputs, [('O', ptile.Shape(x.shape.dtype, outshape))], name=name)


pool = Pool.function


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


def prod(value, axis=None, keepdims=False):
    return op.prod(
        value, axes=axis, keepdims=keepdims, floatx=ptile.convert_np_dtype_to_pml(floatx()))


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
    z0 = op.sqrt(-2.0 * op.log(u1 + (1.0 / (2**33)))) * op.cos(2.0 * math.pi * u2)
    z0 = stddev * z0
    z0 = z0 + mean
    if dtype != 'float32':
        z0 = cast(z0, dtype)
    return z0


def random_normal_variable(shape, mean, scale, dtype=None, name=None, seed=None):
    _report_unimplemented('random_normal_variable')


def random_uniform(shape, minval=0.0, maxval=1.0, dtype=None, seed=None):
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
    t = ptile.Operation(
        'function ({inputs}) -> (O) {{ O = prng_step({args}); }}'.format(
            inputs=', '.join(['I'] + shape_vars), args=', '.join(['I'] + shape_args)),
        [('I', rng_state)] + shape_inputs, [('O', ptile.Shape(plaidml.DType.UINT32, tuple()))],
        name='PrngStep').sole_output()
    n = ptile.Operation(
        'function (I) -> (O) { O = prng_state(I); }', [('I', t)],
        [('O', ptile.Shape(plaidml.DType.UINT32, (3, _k_rng_size)))],
        name='PrngState').sole_output()
    o = ptile.Operation(
        'function (I) -> (O) { O = prng_value(I); }', [('I', t)],
        [('O', ptile.Shape(plaidml.DType.FLOAT32, shape))],
        side_effects=[(rng_state, n)],
        name='PrngValue').sole_output()

    if dtype != 'float32':
        o = cast(o, dtype)

    o = (maxval - minval) * o
    o = o + minval

    return o


def random_uniform_variable(shape, low, high, dtype=None, name=None, seed=None):
    if seed:
        np.random.seed(seed)
    val = np.random.uniform(low=low, high=high, size=shape)
    return variable(val, dtype=dtype)


relu = op.relu


def repeat(x, n):
    assert x.shape.ndims == 2
    code = """
           function (I[N0, N1]) -> (O) {{
               O[i0, r, i1: N0, {reps}, N1] = =(I[i0, i1]);
           }}""".format(reps=n)
    return ptile.Operation(
        code, [('I', x)], [('O', ptile.Shape(x.shape.dtype,
                                             (x.shape.dims[0], n, x.shape.dims[1])))],
        name='Repeat').sole_output()


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
        '{}*n{} + k'.format(rep, i) if i == axis else 'n{}'.format(i)
        for i in range(x.shape.ndims)
    ]

    # Example
    # function(I[N0, N1, N2]) -> (O) {
    #   O[n0, 3*n1 + k, n2 : N0, 3*N1, N2] = =(I[n0, n1, n2]), k < 3 no_defract;
    # }
    f = """
        function (I[{idims}]) -> (O) {{
            O[{oidxs} : {odims}] = =(I[{iidxs}]), k < {rep} no_defract;
        }}""".format(
        idims=', '.join(idim_list),
        iidxs=', '.join(iidx_list),
        odims=', '.join(odim_list),
        oidxs=', '.join(oidx_list),
        rep=str(rep))
    return ptile.Operation(f, [('I', x)], [('O', ptile.Shape(x.shape.dtype, out_shape))],
                           name='RepeatElements') \
                           .sole_output()


def reset_uids():
    global _UID_PREFIX_DICT
    _UID_PREFIX_DICT.clear()


reshape = op.reshape


def resize_images(x, height_factor, width_factor, data_format):
    if data_format == 'channels_first':
        ret = repeat_elements(x, height_factor, axis=2)
        ret = repeat_elements(ret, width_factor, axis=3)
    elif data_format == 'channels_last':
        ret = repeat_elements(x, height_factor, axis=1)
        ret = repeat_elements(ret, width_factor, axis=2)
    else:
        raise ValueError('Invalid data_format {}'.format(data_format))
    return ret


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


def reverse(x, axes):
    if isinstance(axes, int):
        axes = [axes]
    for axis in axes:
        if not isinstance(axis, int):
            raise ValueError(
                'The axes parameter of reverse only accepts an integer or a list of integers, received {}'.
                format(type(axis)))
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
        }}""".format(
        dims=dims, out_idxs=out_idxs, in_idxs=in_idxs)

    return ptile.Operation(f, [('I', x)], [('O', x.shape)], name='Reverse').sole_output()


def reverse_gradient(x, coeff=1.0):
    return ptile.binary_op(x, coeff, 'reverse_grad(L, R)', name='ReverseGradient')


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
            return ptile.Operation(
                f, [('I', val)], [('O', newshape)], name='TimeExpand').sole_output()
        else:
            f = "function (I[B, {sizes}], P) -> (O) {{ O[b, {ii}, {idxs} : B, {T}, {sizes}] = =(I[b, {idxs}]) default P; }}"
            f = f.format(sizes=sizes, idxs=idxs, ii=ii, T=t)
            return ptile.Operation(
                f, [('I', val), ('P', prev)], [('O', newshape)], name='TimeExpand').sole_output()

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


def round(x):
    return ptile.unary_op(x, 'round(I)', 'Round')


def separable_conv(x,
                   depthwise_kernel,
                   pointwise_kernel,
                   strides=None,
                   padding='valid',
                   data_format=None,
                   dilation_rate=None):
    if data_format is None:
        data_format = image_data_format()
    if pointwise_kernel.shape.dims[-2] != depthwise_kernel.shape.dims[-1] * depthwise_kernel.shape.dims[-2]:
        raise ValueError(
            ('Shape mismatch in separable convolution. Depthwise kernel input ' +
             'channel count must match pointwise kernel channel count times channel ' +
             'multiplier.\nReceived {} v {} * {} (from full shapes {} and ' + '{})').format(
                 pointwise_kernel.shape.dims[-2], depthwise_kernel.shape.dims[-2],
                 depthwise_kernel.shape.dims[-1], pointwise_kernel.shape, depthwise_kernel.shape))
    intermediate = conv(
        x,
        depthwise_kernel,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        channelwise=True)
    rank = x.shape.ndims - 2
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
    plaidml.set_floatx(ptile.convert_np_dtype_to_pml(dtype))


def set_learning_phase(value):
    if value != 0 and value != 1:
        raise ValueError("May only set_learning_phase to 0 or 1")
    value = int(value)
    global _in_train_phase
    _in_train_phase = value


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
                + 'existing.shape = ' + str(
                    x.shape) + ', value is a non-array object of type: ' + str(type(value)))
    with x.var.mmap_discard(_ctx) as view:
        view.copy_from_ndarray(np.asarray(value))
        view.writeback()


sigmoid = op.sigmoid


def sign(x):
    _report_unimplemented('sign')


sin = op.sin


def softmax(x):
    return op.softmax(x, axis=x.shape.ndims - 1)


def softplus(x):
    _report_unimplemented('softplus')


def softsign(x):
    _report_unimplemented('softsign')


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
        """).format(
            in_dims=in_dims, in_idxs=in_idxs, out_dims=out_dims, out_idxs=out_idxs)

        super(SpatialPadding, self).__init__(f, [('I', x)],
                                             [('O', ptile.Shape(x.shape.dtype, numeric_out_dims))])


def spatial_2d_padding(x, padding=((1, 1), (1, 1)), data_format=None):
    return SpatialPadding.function(x, padding, data_format)


def spatial_3d_padding(x, padding=((1, 1), (1, 1), (1, 1)), data_format=None):
    return SpatialPadding.function(x, padding, data_format)


def square(x):
    return x * x


def squeeze(x, axis):
    if x.shape.dims[axis] != 1:
        raise ValueError('Can only squeeze length 1 axis')
    if axis == -1:
        result = reshape(x, x.shape.dims[:axis])
    else:
        result = reshape(x, x.shape.dims[:axis] + x.shape.dims[axis + 1:])
    return result


sqrt = op.sqrt


def stack(x, axis=0):
    _report_unimplemented('stack')


def std(x, axis=None, keepdims=False):
    return sqrt(var(x, axis=axis, keepdims=keepdims))


def stop_gradient(variables):
    _report_unimplemented('stop_gradient')


def sum(x, axis=None, keepdims=False):
    return op.summation(
        x, axes=axis, keepdims=keepdims, floatx=ptile.convert_np_dtype_to_pml(floatx()))


class Switch(ptile.Operation):

    def __init__(self, condition, then_expression, else_expression):
        super(Switch, self).__init__('function (C, T, E) -> (O) { O = (C ? T : E); }',
                                     [('C', condition), ('T', then_expression),
                                      ('E', else_expression)], [('O', then_expression.shape)])


switch = Switch.function

tanh = op.tanh


def temporal_padding(x, padding=(1, 1)):
    if x.shape.ndims != 3:
        raise ValueError('Can only perform temporal_padding on 3D tensor')
    # Temporal padding is channels_last 1D spatial padding
    return SpatialPadding.function(x, padding=(padding,), data_format='channels_last')


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
        }}""".format(
        sizes=sizes, out_idx=out_idx, out_sizes=out_sizes, in_idx=in_idx, cons=cons)
    out_dims = tuple(x.shape.dims[i] * n[i] for i in range(x.shape.ndims))
    return ptile.Operation(
        f, [('I', x)], [('O', ptile.Shape(x.shape.dtype, out_dims))], name='Tile').sole_output()


def to_dense(tensor):
    _report_unimplemented('to_dense')


def transpose(x):
    return permute_dimensions(x, range(x.shape.ndims - 1, -1, -1))


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


def uses_correlation():
    return True


def var(x, axis=None, keepdims=False):
    return op.variance(
        x, axes=axis, keepdims=keepdims, floatx=ptile.convert_np_dtype_to_pml(floatx()))


def variable(value, dtype=None, name=None, constraint=None):
    dtype = dtype or floatx()
    if constraint:
        raise PlaidMLKerasException('Unsupported variable constraint')
    if isinstance(value, float) or isinstance(value, six.integer_types):
        tensor = plaidml.Tensor(_device(),
                                plaidml.Shape(_ctx, ptile.convert_np_dtype_to_pml(dtype)))
        with tensor.mmap_discard(_ctx) as view:
            view.copy_from_ndarray(np.array(value))
            view.writeback()
        return ptile.Value.from_var(tensor,
                                    tuple(),
                                    ptile.convert_np_dtype_to_pml(dtype),
                                    _prepend_name_scope(name, 'float_variable' if isinstance(
                                        value, float) else 'int_variable'))
    elif isinstance(value, ptile.Value):
        func = ptile.compose(_ctx, _device(), [], [('out', value)])
        invoker = plaidml.Invoker(_ctx, func)
        shape = invoker.get_output_shape('out')
        tensor = plaidml.Tensor(_device(), shape)
        invoker.set_output('out', tensor)
        invoker.invoke()
        return ptile.Value.from_var(tensor, [d.size for d in shape.dimensions], shape.dtype, name)
    elif isinstance(value, list) or isinstance(value, tuple):
        value = np.array(value)
        # Fallthrough
    # Default to treating the value as an ndarray.
    tensor = plaidml.Tensor(_device(),
                            plaidml.Shape(_ctx, ptile.convert_np_dtype_to_pml(dtype),
                                          *value.shape))
    with tensor.mmap_discard(_ctx) as view:
        view.copy_from_ndarray(value)
        view.writeback()
    return ptile.Value.from_var(tensor, value.shape,
                                ptile.convert_np_dtype_to_pml(dtype),
                                _prepend_name_scope(name, 'tensor_variable'))


def zeros(shape, dtype=floatx(), name=None):
    return constant(0.0, shape=shape, dtype=dtype, name=_prepend_name_scope(name, 'zeros'))


def zeros_like(x, dtype=floatx(), name=None):
    dtype = dtype or floatx()
    a_zero = constant(0.0, shape=(1), dtype=dtype, name=_prepend_name_scope(name, 'a_zero'))
    ndims = x.shape.ndims
    sizes = ', '.join(['S' + str(i) for i in range(ndims)])
    dims = ', '.join(['i' + str(i) for i in range(ndims)])
    f = """
        function (IN[{sizes}], ZERO[SZ]) -> (OUT) {{
            OUT[{dims} : {sizes}] = =(ZERO[0]);
        }}""".format(
        sizes=sizes, dims=dims)
    return ptile.Operation(f, [('IN', x), ('ZERO', a_zero)],
                           [('OUT', ptile.Shape(ptile.convert_np_dtype_to_pml(dtype), x.shape.dims))],
                           name='ZerosLike') \
                .sole_output()


# Dynamically add Keras functionality to the underlying tile.Value class.
# This allows us to transparently use Value as the tensor type exposed by
# the Keras backend; it's a little squirrelly in this one place, but it
# greatly simplifies the rest of this module.
def _get_keras_shape(x):
    try:
        return x.__keras_shape
    except AttributeError:
        return int_shape(x)


def _set_keras_shape(x, shape):
    x.__keras_shape = shape


ptile.Value._keras_shape = property(_get_keras_shape, _set_keras_shape)

ptile.Value.eval = get_value
