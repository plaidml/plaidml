# Copyright 2019 Intel Corporation.

import functools
import inspect
import logging
import math
import os
from collections import defaultdict
from contextlib import contextmanager

import six

import numpy as np
import scipy.stats
import plaidml
import plaidml.edsl as edsl
import plaidml.exec
import plaidml.op as plaidml_op
import plaidml.settings
from keras.backend.common import epsilon, floatx, image_data_format
from keras.backend.common import set_floatx as keras_set_floatx

logger = logging.getLogger(__name__)

# Keras needs us to keep track of unique IDs for prefix strings
# (for use with get_uid and reset_uids)
_UID_PREFIX_DICT = defaultdict(int)

_NAME_SCOPE_STACK = []

_CONV_DATA_FORMAT = ['channels_first', 'channels_last']

_in_train_phase = None  # Will be initialized on first use

_device = plaidml.settings.get('PLAIDML_DEVICE')


def _prepend_name_scope(name, default):
    if name:
        r = '/'.join(_NAME_SCOPE_STACK + [name])
    else:
        r = '/'.join(_NAME_SCOPE_STACK + [default])
        r += '/' + str(get_uid(r))
    return r


def _normalize_axis(axis, ndims, name=''):
    negative_axis_given = False
    normalized_axis = axis + ndims if axis < 0 else axis
    if normalized_axis < 0 or ndims <= normalized_axis:
        name_str = 'for {} op '.format(name) if name else ''
        raise RuntimeError(
            'Axis out of range {}(axis {} requested for tensors with {} dimensions)'.format(
                name_str, axis, ndims))
    return normalized_axis


def _normalize_data_format(data_format):
    if data_format is None:
        data_format = image_data_format()
    if data_format == 'channels_last':
        return 'nxc'
    if data_format == 'channels_first':
        return 'ncx'
    if data_format in ['nxc', 'ncx']:
        return data_format
    raise ValueError('Unrecognized data_format "{}"'.format(data_format))


def _normalize_padding(padding):
    if padding == 'same':
        return 'same_upper'
    if padding in ['same_lower', 'same_upper', 'valid', 'full']:
        return padding
    raise ValueError('Unrecognized padding type "{}"'.format(padding))


def _log_call(func):
    '''A decorator that logs the call of the wrapped function'''

    def wrapper(*args, **kwargs):
        # Call the requested function regardless
        return func(*args, **kwargs)

    return wrapper


class _Executable(object):

    def __init__(self, name, inputs, outputs, updates):
        self._inputs = inputs
        self.program = edsl.Program(name, outputs, updates=updates)
        self.binder = plaidml.exec.Binder(self.program)
        self.executable = self.binder.compile()

    def __call__(self, inputs):
        for tensor, data in zip(self._inputs, inputs):
            buffer = self.binder.input(tensor)
            if buffer:
                data = np.array(data, dtype=buffer.shape.dtype.into_numpy())
                buffer.copy_from_ndarray(data)
        self.executable.run()
        return [self.binder.output(x.ref).as_ndarray() for x in self.program.outputs]


class _Function(object):

    def __init__(self, inputs, outputs, updates, name):
        self._name = name
        self._inputs = inputs
        self._outputs = outputs
        self._updates = updates
        self._cache = {}

    def __call__(self, inputs):
        inputs = [np.array(x) if isinstance(x, (six.integer_types, float)) else x for x in inputs]
        input_shapes = tuple([x.shape for x in inputs])
        logger.debug('_Function: {}({})'.format(self._name, input_shapes))
        exe = self._cache.get(input_shapes)
        if not exe:
            exe = self._compile(inputs)
            self._cache[input_shapes] = exe
        return exe(inputs)

    def _compile(self, inputs):
        for node, data in zip(self._inputs, inputs):
            dtype = node.tensor.dtype
            shape = edsl.LogicalShape(dtype, data.shape)
            node.tensor.bind(shape)
        inputs = [x.tensor for x in self._inputs]
        outputs = [x.tensor for x in self._outputs]
        updates = [(x[0].tensor, x[1].tensor) for x in self._updates]
        return _Executable(self._name, inputs, outputs, updates)


def _create_var(name, value):
    dtype = plaidml.DType.from_numpy(value.dtype)
    shape = edsl.LogicalShape(dtype, value.shape)
    tensor_shape = plaidml.TensorShape(dtype, value.shape)
    buffer = plaidml.Buffer(tensor_shape, device=_device)
    buffer.copy_from_ndarray(value)
    return edsl.Tensor(shape=shape, name=name, buffer=buffer)


class _KerasNode(object):

    def __init__(self, opname, name=None, shape=None, tensor=None, value=None):
        self.opname = opname
        self.name = _prepend_name_scope(name, opname)
        if value is not None:
            tensor = _create_var(self.name, value)
        elif tensor is None:
            tensor = edsl.Tensor(shape=shape, name=self.name)
        # logger.debug('_KerasNode({})'.format(tensor))
        self.tensor = tensor

    def __repr__(self):
        return str(self.tensor)

    def __str__(self):
        return self.name

    def eval(self):
        return get_value(self)

    @property
    def _keras_shape(self):
        try:
            return self.__keras_shape
        except AttributeError:
            return int_shape(self)

    @_keras_shape.setter
    def _keras_shape(self, shape):
        self.__keras_shape = shape

    def __getitem__(self, key):
        logger.debug('__getitem__(self: {}, key: {})'.format(self, key))
        # Any _RawTensorDims are to be forwarded
        raw_tensor_dims = getattr(self, '_RawTensorDims', None)
        if raw_tensor_dims is not None:
            raw_tensor_dims = raw_tensor_dims[key]
        if isinstance(key, slice) or isinstance(key, six.integer_types) or isinstance(
                key, type(Ellipsis)):
            key = (key,)
        if not isinstance(key, tuple):
            raise ValueError('Cannot index PlaidML tensors using type {}'.format(type(key)))
        if key.count(Ellipsis) > 1:
            raise ValueError('Cannot use multiple ellipses in a slice (given {})'.format(key))
        # Fill in ellipsis
        try:
            ellipsis_idx = key.index(Ellipsis)
        except ValueError:
            ellipsis_idx = None
        I = self.tensor
        ndims = I.rank
        extension_length = ndims - len(key)
        if ellipsis_idx is not None:
            # The ellipsis is counted in the length of the key, but does not persist as an axis for slicing, so undo that count
            extension_length += 1
            if extension_length < 0:
                raise ValueError('Slice key too long. Tensor has {} dimensions, key is {}'.format(
                    ndims, key))
            key = tuple(
                list(key[:ellipsis_idx]) + [slice(None, None, None)] * extension_length +
                list(key[ellipsis_idx + 1:]))
        else:
            key = tuple(list(key) + [slice(None, None, None)] * extension_length)
        ret = _KerasNode('slice', tensor=plaidml_op.slice_of(I, key))
        if raw_tensor_dims is not None:
            ret._RawTensorDims = raw_tensor_dims
        return ret

    def __neg__(self):
        return _KerasNode('neg', tensor=-self.tensor)

    def __add__(self, other):
        return self.__binary_op('add', other, lambda x, y: x + y)

    def __radd__(self, other):
        return self.__binary_op('add', other, lambda x, y: y + x)

    def __sub__(self, other):
        return self.__binary_op('sub', other, lambda x, y: x - y)

    def __rsub__(self, other):
        return self.__binary_op('sub', other, lambda x, y: y - x)

    def __mul__(self, other):
        return self.__binary_op('mul', other, lambda x, y: x * y)

    def __rmul__(self, other):
        return self.__binary_op('mul', other, lambda x, y: y * x)

    def __div__(self, other):
        return self.__binary_op('div', other, lambda x, y: x / y)

    def __rdiv__(self, other):
        return self.__binary_op('div', other, lambda x, y: y / x)

    def __truediv__(self, other):
        return self.__binary_op('div', other, lambda x, y: x / y)

    def __rtruediv__(self, other):
        return self.__binary_op('div', other, lambda x, y: y / x)

    def __ge__(self, other):
        return self.__binary_op('cmp_ge', other, lambda x, y: x >= y)

    def __gt__(self, other):
        return self.__binary_op('cmp_gt', other, lambda x, y: x > y)

    def __le__(self, other):
        return self.__binary_op('cmp_le', other, lambda x, y: x <= y)

    def __lt__(self, other):
        return self.__binary_op('cmp_lt', other, lambda x, y: x < y)

    def __binary_op(self, op, other, fn):
        logger.debug('{}(self: {}, other: {})'.format(op, self, other))
        if isinstance(other, _KerasNode):
            other = other.tensor
        if isinstance(other, np.ndarray):
            other = variable(other).tensor
        return _KerasNode(op, tensor=fn(self.tensor, other))


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


def _report_unimplemented(name):
    report = (
        'The Keras backend function \'{}\' is not yet implemented in ' +
        'plaidml. You can help us prioritize by letting us know if this ' +
        'function is important to you, and as always, contributions are welcome!').format(name)
    raise NotImplementedError(report)


class PlaidMLKerasException(Exception):
    pass


@_log_call
def abs(x):
    return _KerasNode('abs', tensor=plaidml_op.abs(x.tensor))


@_log_call
def all(x, axis=None, keepdims=False):
    return _KerasNode('all', tensor=plaidml_op.all(x.tensor, axis, keepdims))


@_log_call
def any(x, axis=None, keepdims=False):
    return _KerasNode('any', tensor=plaidml_op.any(x.tensor, axis, keepdims))


@_log_call
def arange(start, stop=None, step=1, dtype='int32'):
    if isinstance(dtype, plaidml.DType):
        dtype = dtype.into_numpy()
    return variable(np.arange(start, stop, step, dtype), dtype=dtype)


@_log_call
def argmax(x, axis=-1):
    return _KerasNode('argmax', tensor=plaidml_op.argmax(x.tensor, axis))


@_log_call
def argmin(x, axis=-1):
    return argmax(-x, axis=axis)


@_log_call
def backend():
    return 'plaidml'


@_log_call
def batch_dot(x, y, axes=None, name=None):
    X = x.tensor
    Y = y.tensor
    if isinstance(axes, six.integer_types):
        axes = (axes, axes)
    if axes is None:
        axes = (X.rank - 1, Y.rank - 2)
    PLAIDML_BATCHDOT_TF_BEHAVIOR = os.getenv('PLAIDML_BATCHDOT_TF_BEHAVIOR')
    if PLAIDML_BATCHDOT_TF_BEHAVIOR:
        _report_unimplemented('batch_dot')
    else:
        # replicate theano/documentation-specified behavior
        first_dim = edsl.TensorDim()
        first_idx = edsl.TensorIndex()
        batch_dim = edsl.TensorDim()
        batch_idx = edsl.TensorIndex()
        xdims = edsl.TensorDims(X.rank)
        xdims[0] = first_dim
        xdims[axes[0]] = batch_dim
        xidxs = edsl.TensorIndexes(X.rank)
        xidxs[0] = first_idx
        xidxs[axes[0]] = batch_idx
        ydims = edsl.TensorDims(Y.rank)
        ydims[0] = first_dim
        ydims[axes[1]] = batch_dim
        yidxs = edsl.TensorIndexes(Y.rank)
        yidxs[0] = first_idx
        yidxs[axes[1]] = batch_idx
        odims = [xdims[N] for N in range(len(xdims)) if N != axes[0]
                ] + [ydims[N] for N in range(1, len(ydims)) if N != axes[1]]
        oidxs = [xidxs[N] for N in range(len(xidxs)) if N != axes[0]
                ] + [yidxs[N] for N in range(1, len(yidxs)) if N != axes[1]]
        X.bind_dims(*xdims)
        Y.bind_dims(*ydims)
        O = edsl.TensorOutput(*odims)
        O[oidxs] += X[xidxs] * Y[yidxs]
    if len(odims) == 1:
        O = plaidml_op.expand_dims(O, 1)
    return _KerasNode('batch_dot', tensor=O)


@_log_call
def batch_flatten(x):
    I = x.tensor
    I_dims = edsl.TensorDims(I.rank)
    I.bind_dims(*I_dims)
    if len(I_dims) == 1:
        return reshape(x, [I_dims[0], 1])
    if len(I_dims) == 2:
        return x
    return reshape(x, [I_dims[0]] + [functools.reduce((lambda x, y: x * y), I_dims[1:])])


@_log_call
def batch_get_value(xs):
    return [get_value(x) for x in xs]


@_log_call
def batch_set_value(tuples):
    for pair in tuples:
        set_value(pair[0], pair[1])


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
            'Unrecognized data_format given to bias_add: "{}"; '.format(data_format) +
            'only "channels_first" and "channels_last" recognized.')
    if ndim(x) > 2:
        if data_format == 'channels_first':
            # FIXME: tensor.shape is expensive
            try:
                bias_dims = bias.tensor.shape.dims
            except AttributeError:
                bias_dims = bias.shape
            x += reshape(bias, (1, bias_dims[0]) + (1,) * (ndim(x) - 2))
        elif data_format == 'channels_last':
            x += bias
    else:
        x += bias
    return x


@_log_call
def binary_crossentropy(target, output, from_logits=False):
    if from_logits:
        output = sigmoid(output)
    return _KerasNode('binary_crossentropy',
                      tensor=plaidml_op.binary_crossentropy(target.tensor, output.tensor,
                                                            epsilon()))


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

    # TODO: deal with aribtrary python values
    # x = ptile.Value.from_python_value(x)

    try:
        dtype = plaidml.DType.from_numpy(dtype)
    except ValueError:
        raise PlaidMLKerasException('Unsupported cast (%s -> %s)' % (x.shape.dtype, dtype))

    if x.tensor.dtype == dtype:
        return x

    return _KerasNode('cast', tensor=edsl.cast(x.tensor, dtype))


@_log_call
def categorical_crossentropy(target, output, from_logits=False):
    if from_logits:
        output = softmax(output)
    elif output.opname != 'softmax':
        output /= sum(output, axis=(-1,), keepdims=True)
        output = clip(output, epsilon(), 1.0 - epsilon())
    T = target.tensor
    O = output.tensor
    ndims = O.rank
    fixed_dims = edsl.TensorDims(ndims - 1)
    fixed_idxs = edsl.TensorIndexes(ndims - 1)
    Y = edsl.TensorDim()
    y = edsl.TensorIndex()
    input_dims = fixed_dims + [Y]
    O.bind_dims(*input_dims)
    T.bind_dims(*input_dims)
    LO = edsl.log(O)
    TR = edsl.TensorOutput(*fixed_dims)
    TR[fixed_idxs] += T[fixed_idxs + [y]] * LO[fixed_idxs + [y]]
    R = -TR
    return _KerasNode('categorical_crossentropy', tensor=R)


@_log_call
def ceil(x):
    return _KerasNode('ceil', tensor=edsl.ceil(x.tensor))


@_log_call
def clear_session():
    global _in_train_phase
    _in_train_phase = None


@_log_call
def clip(x, min_val, max_val):
    return _KerasNode('clip',
                      tensor=plaidml_op.clip(x.tensor,
                                             variable(min_val).tensor,
                                             variable(max_val).tensor))


@_log_call
def concatenate(tensors, axis=-1):
    tensor_vals = [x.tensor for x in tensors]
    return _KerasNode('concatenate', tensor=plaidml_op.concatenate(tensor_vals, axis))


@_log_call
def constant(value, dtype=None, shape=None, name=None):
    if shape is None:
        if isinstance(value, np.ndarray):
            shape = value.shape
        elif isinstance(value, list) or isinstance(value, tuple):
            shape = (len(value),)
        else:
            shape = (1,)
    np_value = np.full(shape, value, dtype=dtype or floatx())
    return _KerasNode('constant', name=name, value=np_value)


@_log_call
def cos(x):
    return _KerasNode('cos', tensor=edsl.cos(x.tensor))


@_log_call
def conv(x,
         kernel,
         strides=None,
         padding='valid',
         data_format=None,
         dilation_rate=None,
         channelwise=False):
    if channelwise:
        group_layout = 'in_C'
        autogroup_mode = 'max'
    else:
        group_layout = 'none'
        autogroup_mode = 'ungrouped'
    rank = x.tensor.rank - 2
    if strides is None:
        strides = tuple(1 for _ in range(rank))
    if dilation_rate is None:
        dilation_rate = tuple(1 for _ in range(rank))
    return _KerasNode(
        'conv',
        tensor=plaidml_op.convolution(
            x.tensor,
            kernel.tensor,
            strides,
            dilation_rate,
            [1] * len(strides),
            [],
            1,
            _normalize_padding(padding),
            [],
            _normalize_data_format(data_format),
            'xck',
            group_layout,
            False,  # winograd_allowed
            cur_name(),
            autogroup_mode,
            'none',
            []))


@_log_call
def conv_transpose(x, kernel, output_shape, strides, padding, data_format, dilation_rate):
    # Keras gives every dim on the output_shape, but PlaidML expects to infer the channel dims; so restrict to spatial dims
    data_format = _normalize_data_format(data_format)
    if data_format == 'nxc':
        output_shape = output_shape[1:-1]
    elif data_format == 'ncx':
        output_shape = output_shape[2:]
    else:
        raise ValueError('Could not parse data_format "{}"'.format(data_format))
    rank = x.tensor.rank - 2
    if strides is None:
        strides = tuple(1 for _ in range(rank))
    if dilation_rate is None:
        dilation_rate = tuple(1 for _ in range(rank))
    return _KerasNode(
        'conv',
        tensor=plaidml_op.convolution(
            x.tensor,
            kernel.tensor,
            strides,
            dilation_rate,
            [1] * len(strides),
            [],
            1,
            _normalize_padding(padding),
            [],
            data_format,
            'xck',
            'none',
            False,  # winograd_allowed
            cur_name(),
            'ungrouped',
            'data',
            output_shape))


@_log_call
def conv1d(x, kernel, strides=1, padding='valid', data_format=None, dilation_rate=1):
    if padding == 'causal':
        left_pad = dilation_rate * (int_shape(kernel)[0] - 1)
        x = temporal_padding(x, (left_pad, 0))
        padding = 'valid'
    return conv(x, kernel, (strides,), padding, data_format, (dilation_rate,))


@_log_call
def conv2d(x, kernel, strides=(1, 1), padding='valid', dilation_rate=(1, 1), data_format=None):
    if isinstance(strides, six.integer_types):
        strides = (strides,) * 2
    if isinstance(dilation_rate, six.integer_types):
        dilation_rate = (dilation_rate,) * 2
    return conv(x, kernel, strides, padding, data_format, dilation_rate)


@_log_call
def conv2d_transpose(x,
                     kernel,
                     output_shape,
                     strides=(1, 1),
                     padding='valid',
                     data_format=None,
                     dilation_rate=(1, 1)):
    if isinstance(strides, six.integer_types):
        strides = (strides,) * 2
    if isinstance(dilation_rate, six.integer_types):
        dilation_rate = (dilation_rate,) * 2
    return conv_transpose(x, kernel, output_shape, strides, padding, data_format, dilation_rate)


@_log_call
def conv3d(x,
           kernel,
           strides=(1, 1, 1),
           padding='valid',
           dilation_rate=(1, 1, 1),
           data_format=None):
    if isinstance(strides, six.integer_types):
        strides = (strides,) * 3
    if isinstance(dilation_rate, six.integer_types):
        dilation_rate = (dilation_rate,) * 3
    return conv(x, kernel, strides, padding, data_format, dilation_rate)


@_log_call
def conv3d_transpose(x,
                     kernel,
                     output_shape,
                     strides=(1, 1, 1),
                     padding='valid',
                     data_format=None,
                     dilation_rate=(1, 1, 1)):
    if isinstance(strides, six.integer_types):
        strides = (strides,) * 3
    if isinstance(dilation_rate, six.integer_types):
        dilation_rate = (dilation_rate,) * 3
    return conv_transpose(x, kernel, output_shape, strides, padding, data_format, dilation_rate)


@_log_call
def count_params(x):
    result = 1
    for dim in x.tensor.compute_shape().into_TensorShape().sizes:
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


@_log_call
def cumprod(x, axis=0):
    return _KerasNode('cumprod', tensor=plaidml_op.cumprod(x.tensor, axis))


@_log_call
def cumsum(x, axis=0):
    return _KerasNode('cumsum', tensor=plaidml_op.cumsum(x.tensor, axis))


@_log_call
def cur_name():
    if len(_NAME_SCOPE_STACK):
        return _NAME_SCOPE_STACK[0]
    return ''


@_log_call
def depthwise_conv2d(x,
                     kernel,
                     strides=(1, 1),
                     padding='valid',
                     data_format=None,
                     dilation_rate=(1, 1)):
    return conv(x, kernel, strides, padding, data_format, dilation_rate, channelwise=True)


@_log_call
def dot(x, y, name=None):
    return _KerasNode('dot', tensor=plaidml_op.dot(x.tensor, y.tensor), name=name)


@_log_call
def dropout(x, level, noise_shape=None, seed=None):
    I = x.tensor
    ndims = I.rank
    if noise_shape is not None and len(noise_shape) != ndims:
        raise ValueError('noise_shape ndims doesn\'t match input ndims')
    if noise_shape is None:
        shape = edsl.TensorDims(ndims)
        I.bind_dims(*shape)
    else:
        shape = noise_shape
    rng_state = _make_rng_state(seed)
    R = 1.0 - level
    M = 1.0 / R
    T = edsl.prng(rng_state.tensor, shape)
    O = edsl.select(T < R, I * M, 0.0)
    return _KerasNode('dropout', tensor=O)


@_log_call
def dtype(x):
    return x.tensor.dtype.into_numpy()


@_log_call
def elu(x, alpha=1.0):
    return _KerasNode('elu', name='elu', tensor=plaidml_op.elu(x.tensor, alpha))


@_log_call
def equal(x, y):
    if isinstance(x, _KerasNode):
        x = x.tensor
    if isinstance(x, np.ndarray):
        x = variable(x).tensor
    if isinstance(y, _KerasNode):
        y = y.tensor
    if isinstance(y, np.ndarray):
        y = variable(y).tensor
    return _KerasNode('equal', tensor=(x == y))


@_log_call
def exp(x):
    return _KerasNode('exp', tensor=edsl.exp(x.tensor))


@_log_call
def eval(x):
    return get_value(x)


@_log_call
def expand_dims(x, axis=-1, name=None):
    return _KerasNode('expand_dims', name=name, tensor=plaidml_op.expand_dims(x.tensor, axis))


@_log_call
def eye(size, dtype=None, name=None):
    if dtype is None:
        dtype = floatx()
    elif isinstance(dtype, plaidml.DType):
        dtype = dtype.into_numpy()
    return variable(np.eye(size, dtype=dtype), name=name, dtype=dtype)


@_log_call
def flatten(x):
    I = x.tensor
    I_dims = edsl.TensorDims(I.rank)
    I.bind_dims(*I_dims)
    O_dim = functools.reduce(lambda x, y: x * y, I_dims)
    return reshape(x, [O_dim])


@_log_call
def floor(x):
    return _KerasNode('floor', tensor=edsl.floor(x.tensor))


@_log_call
def foldl(fn, elems, initializer=None, name=None):
    _report_unimplemented('foldl')


@_log_call
def foldr(fn, elems, initializer=None, name=None):
    _report_unimplemented('foldr')


# No _log_call as this does specialized logging
def function(inputs, outputs, updates=None, name=None):
    logger.debug('function(name: {})'.format(name))
    logger.debug('  inputs:')
    for input in inputs:
        logger.debug('    {}'.format(input))
    logger.debug('  outputs:')
    for output in outputs:
        logger.debug('    {}'.format(output))
    if updates:
        logger.debug('  updates:')
        for update in updates:
            logger.debug('    {}'.format(update))
    if updates is None:
        updates = []
    if name is None:
        name = ''
    return _Function(inputs, outputs, updates, name)


@_log_call
def gather(x, indicies):
    return _KerasNode('gather', tensor=edsl.gather(x.tensor, indicies.tensor))


@_log_call
def get_uid(prefix=''):
    _UID_PREFIX_DICT[prefix] += 1
    return _UID_PREFIX_DICT[prefix]


@_log_call
def get_value(x):
    if x.tensor._buffer:
        return x.tensor._buffer.as_ndarray()
    inputs = []
    fn = _Function(inputs, [x], [], name='get_value')
    outputs = fn(inputs)
    return outputs[0]


@_log_call
def get_variable_shape(x):
    return x._keras_shape


@_log_call
def gradients(loss, variables):
    grads = edsl.gradients(loss.tensor, [x.tensor for x in variables])
    return [_KerasNode('gradients', tensor=x) for x in grads]


@_log_call
def greater(x, y):
    return x > y


@_log_call
def greater_equal(x, y):
    return x >= y


@_log_call
def hard_sigmoid(x):
    return _KerasNode('hard_sigmoid',
                      name='hard_sigmoid',
                      tensor=plaidml_op.hard_sigmoid(x.tensor, 0.2))


@_log_call
def identity(x):
    return _KerasNode('identity', tensor=edsl.ident(x.tensor))


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

    cx = x() if callable(x) else x
    calt = alt() if callable(alt) else alt

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
    shape = x.tensor.compute_shape().into_TensorShape()
    return tuple(None if x == 0 else x for x in shape.sizes)


@_log_call
def is_keras_tensor(x):
    if not is_tensor(x):
        raise ValueError()
    return hasattr(x, '_keras_history')


@_log_call
def is_placeholder(x):
    if not is_tensor(x):
        return False
    return x.opname == 'placeholder'


@_log_call
def is_sparse(x):
    return False


@_log_call
def is_tensor(x):
    return isinstance(x, _KerasNode)


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


@_log_call
def log(x):
    return _KerasNode('log', tensor=edsl.log(x.tensor))


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
    return _KerasNode('max', tensor=plaidml_op.max(x.tensor, axis, keepdims))


@_log_call
def maximum(x, y):
    return _KerasNode('maximum', tensor=plaidml_op.maximum(x.tensor, y.tensor))


@_log_call
def mean(x, axis=None, keepdims=False):
    return _KerasNode('mean', tensor=plaidml_op.mean(x.tensor, axis, keepdims))


@_log_call
def min(x, axis=None, keepdims=False):
    return _KerasNode('min', tensor=plaidml_op.min(x.tensor, axis, keepdims))


@_log_call
def minimum(x, y):
    return _KerasNode('minimum', tensor=plaidml_op.minimum(x.tensor, y.tensor))


@_log_call
def moving_average_update(x, value, momentum):
    return (x, x * momentum + value * (1. - momentum))


# No _log_call as this manages logging specially
@contextmanager
def name_scope(name):
    _NAME_SCOPE_STACK.append(name)
    logger.debug('name_scope({}), push: {}'.format(name, _NAME_SCOPE_STACK))
    yield
    _NAME_SCOPE_STACK.pop()
    logger.debug('name_scope({}), pop: {}'.format(name, _NAME_SCOPE_STACK))


@_log_call
def ndim(x):
    return len(x._keras_shape)


@_log_call
def not_equal(lhs, rhs):
    if isinstance(lhs, _KerasNode):
        lhs = lhs.tensor
    if isinstance(lhs, np.ndarray):
        lhs = variable(lhs).tensor
    if isinstance(rhs, _KerasNode):
        rhs = rhs.tensor
    if isinstance(rhs, np.ndarray):
        rhs = variable(rhs).tensor
    return _KerasNode('not_equal', tensor=(lhs != rhs))


@_log_call
def normalize_batch_in_training(x, gamma, beta, reduction_axes, epsilon=1e-3):
    I = x.tensor
    ndims = I.rank
    if reduction_axes == None:
        raw_axes = [ndims - 1]
    else:
        raw_axes = reduction_axes
    axes = [_normalize_axis(x, ndims, 'normalize_batch_in_training') for x in raw_axes]
    m = mean(x, axis=axes, keepdims=True)
    v = var(x, axis=axes, keepdims=True)

    # We reshape beta & gamma to the target shape; this discards shape information on beta & gamma but matches the behavior with the TF backend
    dims = edsl.TensorDims(ndims)
    I.bind_dims(*dims)
    for ax in axes:
        dims[ax] = 1
    if beta is not None:
        beta = reshape(beta, dims)
    if gamma is not None:
        gamma = reshape(gamma, dims)

    normalized_tensor = batch_normalization(x=x,
                                            mean=m,
                                            var=v,
                                            beta=beta,
                                            gamma=gamma,
                                            epsilon=epsilon)

    # squeeze the mean and variance in all cases, that's what TF does
    m = squeeze(m)
    v = squeeze(v)

    return normalized_tensor, m, v


@_log_call
def one_hot(indices, num_classes):
    #Note: does not error check for entries in indices that are >= num_classes
    count = variable(np.array(range(num_classes)), dtype='int32').tensor
    I = indices.tensor
    I_ndims = I.rank
    I_dims = edsl.TensorDims(I_ndims)
    I_idxs = edsl.TensorIndexes(I_ndims)
    C = edsl.TensorDim()
    c = edsl.TensorIndex()
    O_dims = I_dims + [C]
    O_idxs = I_idxs + [c]
    I.bind_dims(*I_dims)
    count.bind_dims(C)
    O = edsl.TensorOutput(*O_dims)
    O[O_idxs] = I[I_idxs] == count[c]
    return _KerasNode('one_hot', name='one_hot', tensor=O)


@_log_call
def ones(shape, dtype=None, name=None):
    value = np.full(shape, 1, dtype=dtype or floatx())
    return _KerasNode('ones', name=name, value=value)


@_log_call
def ones_like(x, dtype=None, name=None):
    value = np.full((1), 1, dtype=dtype or floatx())
    one = _create_var('a_one', value)
    I = x.tensor
    ndim = I.rank
    dims = edsl.TensorDims(ndim)
    idxs = edsl.TensorIndexes(ndim)
    I.bind_dims(*dims)
    O = edsl.TensorOutput(*dims)
    O[idxs] = one[0]
    return _KerasNode('ones_like', name=name, tensor=O)


@_log_call
def permute_dimensions(x, pattern=None):
    return _KerasNode('permute_dimensions', tensor=plaidml_op.transpose(x.tensor, pattern))


@_log_call
def placeholder(shape=None, ndim=None, dtype=None, sparse=False, name=None):
    dtype = plaidml.DType.from_numpy(dtype or floatx())
    # TODO: Need to support empty shapes; once supported, convert below to `if _ is not None`
    if shape is not None:
        return _KerasNode('placeholder', shape=edsl.LogicalShape(dtype, shape), name=name)
    if ndim is not None:
        return _KerasNode('placeholder', shape=edsl.LogicalShape(dtype, [0] * ndim), name=name)
    raise ValueError()


@_log_call
def pool(x, pool_size, strides=None, padding='valid', data_format=None, pool_mode='max'):
    return _KerasNode('pool',
                      tensor=plaidml_op.pool(
                          x.tensor,
                          pool_mode,
                          pool_size,
                          strides,
                          _normalize_padding(padding),
                          tuple(),
                          _normalize_data_format(data_format),
                          False,
                          False,
                      ))


@_log_call
def pool2d(x, pool_size, strides=(1, 1), padding='valid', data_format=None, pool_mode='max'):
    return pool(x=x,
                pool_size=pool_size,
                strides=strides,
                padding=padding,
                data_format=data_format,
                pool_mode=pool_mode)


@_log_call
def pool3d(x, pool_size, strides=(1, 1, 1), padding='valid', data_format=None, pool_mode='max'):
    return pool(x=x,
                pool_size=pool_size,
                strides=strides,
                padding=padding,
                data_format=data_format,
                pool_mode=pool_mode)


@_log_call
def pow(x, a):
    return _KerasNode('pow', tensor=edsl.pow(x.tensor, a))


@_log_call
def print_tensor(x, message=''):
    _report_unimplemented('print_tensor')


@_log_call
def prod(value, axis=None, keepdims=False):
    if isinstance(value, (tuple, list)):
        # In this case, a product of the elements of the tuple/list is being requested,
        # rather than a within-tensor product
        return functools.reduce(lambda x, y: x * y, value)
    return _KerasNode('prod', tensor=plaidml_op.prod(value.tensor, axis, keepdims))


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
    z0 = sqrt(-2.0 * log(u1 + (1.0 / (2**33)))) * cos(2.0 * math.pi * u2)
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
    rng_state = _make_rng_state(seed)
    R = edsl.prng(rng_state.tensor, shape)
    dtype = dtype or floatx()
    if dtype != 'float32':
        R = edsl.cast(R, plaidml.DType.from_numpy(dtype))
    O = (maxval - minval) * R + minval
    return _KerasNode('random_uniform', tensor=O)


@_log_call
def random_uniform_variable(shape, low, high, dtype=None, name=None, seed=None):
    if seed:
        np.random.seed(seed)
    val = np.random.uniform(low=low, high=high, size=shape)
    return variable(val, dtype=dtype)


@_log_call
def relu(x, alpha=None, max_value=None, threshold=0.):
    return _KerasNode('relu', tensor=plaidml_op.relu(x.tensor, alpha, max_value, threshold))


@_log_call
def repeat(x, n):
    y = expand_dims(x, 1, name='repeat')
    return repeat_elements(y, n, 1)


@_log_call
def repeat_elements(x, rep, axis):
    return _KerasNode('repeat_elements',
                      name='repeat_elements',
                      tensor=plaidml_op.repeat(x.tensor, rep, axis))


@_log_call
def reset_uids():
    global _UID_PREFIX_DICT
    _UID_PREFIX_DICT.clear()


@_log_call
def reshape(x, dims):
    # TODO: This needs to be more thoroughly tested with symbolic shapes
    dims = list(dims)
    I = x.tensor
    for i, s in enumerate(dims):
        if isinstance(s, _KerasNode):
            # If using dims from a call to `shape`, they are saved directly in the _RawTensorDims attribute
            raw_dim = getattr(s, '_RawTensorDims', None)
            if isinstance(raw_dim, edsl.TensorDim):
                dims[i] = raw_dim
            else:
                raise RuntimeError('Cannot parse dimension from {} for reshape'.format(s))
        if s == -1:
            dims[i] = 'fill'
            continue
        if s == 0:
            dims[i] = 'match'
    return _KerasNode('reshape', tensor=plaidml_op.reshape(I, dims))


@_log_call
def resize_images(x, height_factor, width_factor, data_format, interpolation='nearest'):
    return _KerasNode('resize_images',
                      tensor=plaidml_op.image_resize(x.tensor, (height_factor, width_factor),
                                                     interpolation,
                                                     _normalize_data_format(data_format)))


@_log_call
def resize_volumes(x, depth_factor, height_factor, width_factor, data_format):
    data_format = _normalize_data_format(data_format)
    if data_format == 'ncx':
        ret = repeat_elements(x, depth_factor, axis=2)
        ret = repeat_elements(ret, height_factor, axis=3)
        ret = repeat_elements(ret, width_factor, axis=4)
    elif data_format == 'nxc':
        ret = repeat_elements(x, depth_factor, axis=1)
        ret = repeat_elements(ret, height_factor, axis=2)
        ret = repeat_elements(ret, width_factor, axis=3)
    else:
        raise ValueError('Invalid data_format {}'.format(data_format))
    return ret


@_log_call
def reverse(x, axes):
    return _KerasNode('reverse', name='reverse', tensor=plaidml_op.flip(x.tensor, axes))


@_log_call
def reverse_gradient(x, coeff=1.0):
    return _KerasNode('reverse_gradient', tensor=plaidml_op.scale_gradient(x.tensor, -coeff))


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
        input_length = inputs.tensor.compute_shape().into_TensorShape().sizes[1]
    if not isinstance(input_length, six.integer_types):
        raise NotImplementedError('rnn is not implemented for variable sized inputs')
    if mask is not None:
        raise NotImplementedError('rnn is not implemented with mask support')
    if constants is None:
        constants = list()

    def time_expand(val, ii, t, prev):
        I = val.tensor
        ndmo = I.rank - 1
        if (ndmo < 0):
            raise PlaidMLKerasException('output values must have a batch size dimension')
        dims = edsl.TensorDims(ndmo)
        idxs = edsl.TensorIndexes(ndmo)
        batch_dim = edsl.TensorDim()
        batch_idx = edsl.TensorIndex()
        I_dims = [batch_dim] + dims
        I_idxs = [batch_idx] + idxs
        I.bind_dims(*I_dims)
        O_dims = [batch_dim] + [t] + dims
        O = edsl.TensorOutput(*O_dims)
        O_idxs = [batch_idx] + [ii] + idxs
        O[O_idxs] = I[I_idxs]
        if prev is None:
            if ii != 0:
                raise RuntimeError(
                    'Generating RNN at time step {} with no previous time step'.format(ii))
        else:
            O.use_default(prev.tensor)
        return _KerasNode('time_expand', name='time_expand', tensor=O)

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
    return _KerasNode('round', tensor=edsl.round(x.tensor))


@_log_call
def separable_conv(x,
                   depthwise_kernel,
                   pointwise_kernel,
                   strides=None,
                   padding='valid',
                   data_format=None,
                   dilation_rate=None):
    data_format = _normalize_data_format(data_format)
    if int_shape(pointwise_kernel
                )[-2] != int_shape(depthwise_kernel)[-1] * int_shape(depthwise_kernel)[-2]:
        # FIXME: tensor.shape is expensive
        raise ValueError(
            ('Shape mismatch in separable convolution. Depthwise kernel input ' +
             'channel count must match pointwise kernel channel count times channel ' +
             'multiplier.\nReceived {} v {} * {} (from full shapes {} and ' + '{})').format(
                 pointwise_kernel.tensor.shape.dims[-2], depthwise_kernel.tensor.shape.dims[-2],
                 depthwise_kernel.tensor.shape.dims[-1], pointwise_kernel.tensor.shape,
                 depthwise_kernel.tensor.shape))
    intermediate = conv(x,
                        depthwise_kernel,
                        strides=strides,
                        padding=padding,
                        data_format=data_format,
                        dilation_rate=dilation_rate,
                        channelwise=True)
    rank = x.tensor.rank - 2
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
    return separable_conv(
        x,
        depthwise_kernel,
        pointwise_kernel,
        strides,
        padding,
        data_format,
        dilation_rate,
    )


@_log_call
def set_floatx(dtype):
    keras_set_floatx(dtype)
    # plaidml.set_floatx(ptile.convert_np_dtype_to_pml(dtype))


@_log_call
def set_learning_phase(value):
    if value != 0 and value != 1:
        raise ValueError('May only set_learning_phase to 0 or 1')
    value = int(value)
    global _in_train_phase
    _in_train_phase = value


@_log_call
def set_value(x, value):
    dtype = plaidml.DType.from_numpy(value.dtype)
    tensor_shape = plaidml.TensorShape(dtype, value.shape)
    buffer = plaidml.Buffer(tensor_shape, device=_device)
    buffer.copy_from_ndarray(value)
    x.tensor.set_param_value(buffer)


@_log_call
def shape(x):
    ret = _KerasNode('shape', tensor=edsl.shape(x.tensor))
    # Save the TensorDims directly on the _KerasNode, where they can be extracted if needed
    ret._RawTensorDims = edsl.TensorDims(x.tensor.rank)
    x.tensor.bind_dims(*ret._RawTensorDims)
    return ret


@_log_call
def sigmoid(x):
    return _KerasNode('sigmoid', tensor=plaidml_op.sigmoid(x.tensor))


@_log_call
def sign(x):
    intermediate = _KerasNode('sign_intermediate', tensor=edsl.select((x > 0).tensor, 1., -1.))
    return _KerasNode('sign', tensor=edsl.select((x.tensor == 0.), 0., intermediate.tensor))


@_log_call
def sin(x):
    return _KerasNode('sin', tensor=edsl.sin(x.tensor))


@_log_call
def softmax(x, axis=None, name=None):
    I = x.tensor
    if name is None:
        name = 'softmax'
    if axis is None:
        axis = I.rank - 1
    y = plaidml_op.softmax(I, axis=axis)
    return _KerasNode('softmax', name=name, tensor=y)


@_log_call
def softplus(x):
    return log(1. + exp(x))


@_log_call
def softsign(x):
    return x / (1. + abs(x))


@_log_call
def sparse_categorical_crossentropy(target, output, from_logits=False):
    shape = output.tensor.compute_shape()
    dims = edsl.TensorDims(shape.rank)
    output.tensor.bind_dims(*dims)
    # FIXME: tensor.shape is expensive
    return categorical_crossentropy(
        reshape(one_hot(target,
                        shape.into_TensorShape().sizes[-1]), dims), output, from_logits)


@_log_call
def spatial_2d_padding(x, padding=((1, 1), (1, 1)), data_format=None):
    data_format = _normalize_data_format(data_format)
    lo_pads = [padding[i][0] for i in range(2)]
    hi_pads = [padding[i][1] for i in range(2)]
    return _KerasNode('spatial_2d_padding',
                      tensor=plaidml_op.spatial_padding(x.tensor,
                                                        lo_pads=lo_pads,
                                                        hi_pads=hi_pads,
                                                        data_layout=data_format))


@_log_call
def spatial_3d_padding(x, padding=((1, 1), (1, 1), (1, 1)), data_format=None):
    data_format = _normalize_data_format(data_format)
    lo_pads = [padding[i][0] for i in range(3)]
    hi_pads = [padding[i][1] for i in range(3)]
    return _KerasNode('spatial_2d_padding',
                      tensor=plaidml_op.spatial_padding(x.tensor,
                                                        lo_pads=lo_pads,
                                                        hi_pads=hi_pads,
                                                        data_layout=data_format))


@_log_call
def sqrt(x):
    return _KerasNode('sqrt', tensor=edsl.sqrt(x.tensor))


@_log_call
def square(x):
    return _KerasNode('square', tensor=plaidml_op.square(x.tensor))


@_log_call
def squeeze(x, axis=None):
    if axis is None:
        # Auto-squeeze the size 1 dims. Note that this never squeezes symbolic dims
        axis = []
        x_shape = int_shape(x)
        for s in range(len(x_shape)):
            if x_shape[s] == 1:
                axis.append(s)
    return _KerasNode('squeeze', tensor=plaidml_op.squeeze(x.tensor, axis))


@_log_call
def stack(x, axis=0):
    return concatenate([expand_dims(item, axis) for item in x], axis=axis)


@_log_call
def std(x, axis=None, keepdims=False):
    return sqrt(var(x, axis=axis, keepdims=keepdims))


def stop_gradient(variables):
    _report_unimplemented('stop_gradient')


@_log_call
def sum(x, axis=None, keepdims=False):
    if isinstance(x, (tuple, list)):
        # In this case, a sum of the elements of the tuple/list is being requested,
        # rather than a within-tensor sum
        return functools.reduce(lambda a, b: a + b, x)
    return _KerasNode('sum', tensor=plaidml_op.sum(x.tensor, axis, keepdims))


@_log_call
def switch(condition, then_expression, else_expression):
    return _KerasNode('switch',
                      tensor=edsl.select(condition.tensor, then_expression.tensor,
                                         else_expression.tensor))


@_log_call
def tanh(x):
    return _KerasNode('tanh', tensor=edsl.tanh(x.tensor))


@_log_call
def temporal_padding(x, padding=(1, 1)):
    data_format = _normalize_data_format(None)  # uses image_data_format()
    lo_pads = [padding[0]]
    hi_pads = [padding[1]]
    return _KerasNode('temporal_padding',
                      tensor=plaidml_op.spatial_padding(x.tensor,
                                                        lo_pads=lo_pads,
                                                        hi_pads=hi_pads,
                                                        data_layout=data_format))


@_log_call
def tile(x, n):
    return _KerasNode('tile', tensor=plaidml_op.tile(x.tensor, n))


@_log_call
def to_dense(tensor):
    _report_unimplemented('to_dense')


@_log_call
def transpose(x):
    return _KerasNode('transpose', tensor=plaidml_op.transpose(x.tensor))


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
    return _KerasNode('var', tensor=plaidml_op.variance(x.tensor, axis, keepdims))


@_log_call
def variable(value, dtype=None, name=None, constraint=None):
    if name is None:
        name = 'anon'
    dtype = dtype or floatx()
    if isinstance(value, _KerasNode):
        value = value.eval()
    if isinstance(value, float) or isinstance(value, six.integer_types):
        value = np.array(value, dtype=dtype)
    if isinstance(value, list) or isinstance(value, tuple):
        value = np.array(value, dtype=dtype)
    if isinstance(value, np.ndarray):
        if dtype != value.dtype:
            logger.debug(
                'Casting to requested dtype in variable, received {} and requested {}'.format(
                    value.dtype, dtype))
            value = value.astype(dtype)
        return _KerasNode('variable', name=name, value=value)
    raise TypeError('Unknown type for variable: {}'.format(type(value)))


@_log_call
def zeros(shape, dtype=None, name=None):
    value = np.full(shape, 0, dtype=dtype or floatx())
    return _KerasNode('zeros', name=name, value=value)


@_log_call
def zeros_like(x, dtype=None, name=None):
    value = np.full((1), 0, dtype=dtype or floatx())
    zero = _create_var('a_zero', value)
    I = x.tensor
    ndim = I.rank
    dims = edsl.TensorDims(ndim)
    idxs = edsl.TensorIndexes(ndim)
    I.bind_dims(*dims)
    O = edsl.TensorOutput(*dims)
    O[idxs] = zero[0]
    return _KerasNode('zeros_like', name=name, tensor=O)
