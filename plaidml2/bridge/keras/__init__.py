# Copyright 2019 Intel Corporation.

import functools
import logging
import os
from collections import defaultdict
from contextlib import contextmanager

import six

import numpy as np
import scipy.stats
import plaidml2 as plaidml
import plaidml2.edsl as edsl
import plaidml2.exec as plaidml_exec
import plaidml2.op as plaidml_op
import plaidml2.settings as plaidml_settings
from keras.backend.common import epsilon, floatx, image_data_format
from keras.backend.common import set_floatx as keras_set_floatx

logger = logging.getLogger(__name__)

# Keras needs us to keep track of unique IDs for prefix strings
# (for use with get_uid and reset_uids)
_UID_PREFIX_DICT = defaultdict(int)

_NAME_SCOPE_STACK = []

_CONV_DATA_FORMAT = ['channels_first', 'channels_last']

_in_train_phase = None  # Will be initialized on first use

_device = plaidml_settings.get('PLAIDML_DEVICE')
_target = plaidml_settings.get('PLAIDML_TARGET')


def _prepend_name_scope(name, default):
    if name:
        r = '_'.join(_NAME_SCOPE_STACK + [name])
    else:
        r = '_'.join(_NAME_SCOPE_STACK + [default])
        r += '_' + str(get_uid(r))
    return r


def _normalize_axis(axis, ndims, name=""):
    negative_axis_given = False
    if axis < 0:
        axis += ndims
        negative_axis_given = True
    if axis < 0 or ndims <= axis:
        if negative_axis_given:
            axis -= ndims
        if name:
            name_str = "for {} op ".format(name)
        else:
            name_str = ""
        raise RuntimeError(
            "Axis out of range {}(axis {} requested for tensors with {} dimensions)".format(
                name_str, axis, ndims))
    return axis


def _normalize_data_format(data_format):
    if data_format is None:
        data_format = image_data_format()
    if data_format == 'channels_last':
        return 'nxc'
    if data_format == 'channels_first':
        return 'ncx'
    if data_format in ['nxc', 'ncx']:
        return data_format
    raise ValueError("Unrecognized data_format '{}'".format(data_format))


def _normalize_padding(padding):
    if padding == 'same':
        return 'same_upper'
    if padding in ['same_lower', 'same_upper', 'valid', 'full']:
        return padding
    raise ValueError("Unrecognized padding type '{}'".format(padding))


class _Function(object):

    def __init__(self, inputs, outputs, updates, name):
        self._name = name
        self._inputs = inputs
        self._outputs = outputs
        self._updates = updates
        self._cache = {}

    def __call__(self, inputs):
        input_shapes = tuple([x.shape for x in inputs])
        # logger.debug('_Function: {}({})'.format(self._name, input_shapes))
        exe = self._cache.get(input_shapes)
        if not exe:
            exe = self._compile(inputs)
            self._cache[input_shapes] = exe
        return [x.as_ndarray() for x in exe(inputs)]

    def _compile(self, inputs):
        for node, data in zip(self._inputs, inputs):
            dtype = node.tensor.shape.dtype
            shape = edsl.LogicalShape(dtype, data.shape)
            node.tensor.bind(shape)
        outputs = [x.tensor for x in self._outputs]
        updates = [(x[0].tensor, x[1].tensor) for x in self._updates]
        program = edsl.Program(self._name, outputs, updates)

        def make_buffer(tensor):
            # convert LogicalShape into TensorShape
            shape = plaidml.TensorShape(tensor.shape.dtype, tensor.shape.int_dims)
            return plaidml.Buffer(_device, shape)

        input_bindings = [(x.tensor, make_buffer(x.tensor)) for x in self._inputs]
        output_bindings = [(x, make_buffer(x)) for x in program.outputs]
        return plaidml_exec.Executable(program, _device, _target, input_bindings, output_bindings)


def _create_var(name, value):
    dtype = plaidml.DType.from_numpy(value.dtype)
    shape = edsl.LogicalShape(dtype, value.shape)
    tensor_shape = plaidml.TensorShape(dtype, value.shape)
    buffer = plaidml.Buffer(_device, tensor_shape)
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
        return '{}|{}'.format(self.name, self.tensor.shape)

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
        raise NotImplementedError('TODO: slice_of')

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
        'plaidml2. You can help us prioritize by letting us know if this ' +
        'function is important to you, and as always, contributions are welcome!').format(name)
    raise NotImplementedError(report)


class PlaidMLKerasException(Exception):
    pass


def abs(x):
    logger.debug('abs(x: {})'.format(x))
    return _KerasNode('abs', tensor=plaidml_op.abs(x.tensor))


def all(x, axis=None, keepdims=False):
    _report_unimplemented('all')


def any(x, axis=None, keepdims=False):
    _report_unimplemented('any')


def arange(start, stop=None, step=1, dtype='int32'):
    _report_unimplemented('arange')


def argmax(x, axis=-1):
    logger.debug('argmax(x: {}, axis: {})'.format(x, axis))
    return _KerasNode('argmax', tensor=plaidml_op.argmax(x.tensor, axis))


def argmin(x, axis=-1):
    return argmax(-x, axis=axis)


def backend():
    return 'plaidml2'


def batch_dot(x, y, axes=None, name=None):
    _report_unimplemented('batch_dot')


def batch_flatten(x):
    _report_unimplemented('batch_flatten')


def batch_set_value(tuples):
    _report_unimplemented('batch_set_value')


def batch_get_value(xs):
    _report_unimplemented('batch_get_value')


def batch_normalization(x, mean, var, beta, gamma, axis=-1, epsilon=1e-3):
    _report_unimplemented('batch_normalization')


def bias_add(x, bias, data_format=None):
    logger.debug('bias_add(x: {}, bias: {}, data_format: {})'.format(x, bias, data_format))
    if data_format is None:
        data_format = image_data_format()
    if data_format not in _CONV_DATA_FORMAT:
        raise PlaidMLKerasException(
            "Unrecognized data_format given to bias_add: '{}'; ".format(data_format) +
            "only 'channels_first' and 'channels_last' recognized.")
    if ndim(x) > 2:
        if data_format == 'channels_first':
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


def binary_crossentropy(target, output, from_logits=False):
    _report_unimplemented('binary_crossentropy')


def cast(x, dtype):
    logger.debug('cast(x: {}, dtype: {})'.format(x, dtype))
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

    if x.tensor.shape.dtype == dtype:
        return x

    return _KerasNode('cast', tensor=edsl.cast(x.tensor, dtype))


def categorical_crossentropy(target, output, from_logits=False):
    logger.debug('categorical_crossentropy(target: {}, output: {}, from_logits: {})'.format(
        target, output, from_logits))
    if from_logits:
        output = softmax(output)
    elif output.opname != 'softmax':
        output /= sum(output, axis=(-1,), keepdims=True)
        output = clip(output, epsilon(), 1.0 - epsilon())
    T = target.tensor
    O = output.tensor
    ndims = O.shape.ndims
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


def ceil(x):
    _report_unimplemented('ceil')


def clear_session():
    _report_unimplemented('clear_session')


def clip(x, min_val, max_val):
    logger.debug('clip(x: {}, min_val: {}, max_val: {}'.format(x, min_val, max_val))
    return _KerasNode('clip',
                      tensor=plaidml_op.clip(x.tensor,
                                             variable(min_val).tensor,
                                             variable(max_val).tensor))


def concatenate(tensors, axis=-1):
    logger.debug('concatenate(tensors: {}, axis: {})'.format(tensors, axis))
    tensor_vals = [x.tensor for x in tensors]
    return _KerasNode('concatenate', tensor=plaidml_op.concatenate(tensor_vals, axis))


def constant(value, dtype=None, shape=None, name=None):
    logger.debug('constant(value: {}, dtype: {}, shape: {}, name: {})'.format(
        value, dtype, shape, name))
    if shape is None:
        if isinstance(value, np.ndarray):
            shape = value.shape
        elif isinstance(value, list) or isinstance(value, tuple):
            shape = (len(value),)
        else:
            shape = (1,)
    np_value = np.full(shape, value, dtype=dtype or floatx())
    return _KerasNode('constant', name=name, value=np_value)


def cos(x):
    logger.debug('cos(x: {})'.format(x))
    return _KerasNode('cos', tensor=edsl.cos(x.tensor))


def conv(x,
         kernel,
         strides=None,
         padding='valid',
         data_format=None,
         dilation_rate=None,
         channelwise=False):
    logger.debug(
        'conv(x: {}, kernel: {}, strides: {}, padding: {}, data_format: {}, dilation_rate: {}, channelwise: {}'
        .format(x, kernel, strides, padding, data_format, dilation_rate, channelwise))
    if channelwise:
        group_layout = 'in_C'
        autogroup_mode = 'max'
    else:
        group_layout = 'none'
        autogroup_mode = 'ungrouped'
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


def conv_transpose(x, kernel, output_shape, strides, padding, data_format, dilation_rate):
    logger.debug(
        'conv_transpose(x: {}, kernel: {}, output_shape: {}, strides: {}, padding: {}, data_format: {}, dilation_rate: {}'
        .format(x, kernel, output_shape, strides, padding, data_format, dilation_rate))
    # Keras gives every dim on the output_shape, but PlaidML expects to infer the channel dims; so restrict to spatial dims
    data_format = _normalize_data_format(data_format)
    if data_format == 'nxc':
        output_shape = output_shape[1:-1]
    elif data_format == 'ncx':
        output_shape = output_shape[2:]
    else:
        raise ValueError("Could not parse data_format '{}'".format(data_format))
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


def conv1d(x, kernel, strides=1, padding='valid', data_format=None, dilation_rate=1):
    if padding == 'causal':
        left_pad = dilation_rate * (int_shape(kernel)[0] - 1)
        x = temporal_padding(x, (left_pad, 0))
        padding = 'valid'
    return conv(x, kernel, (strides,), padding, data_format, (dilation_rate,))


def conv2d(x, kernel, strides=(1, 1), padding='valid', dilation_rate=(1, 1), data_format=None):
    if isinstance(strides, six.integer_types):
        strides = (strides,) * 2
    if isinstance(dilation_rate, six.integer_types):
        dilation_rate = (dilation_rate,) * 2
    return conv(x, kernel, strides, padding, data_format, dilation_rate)


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


def count_params(x):
    logger.debug('count_params(x: {})'.format(x))
    result = 1
    for dim in x.tensor.shape.int_dims:
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
    _report_unimplemented('cumsum')


def cur_name():
    if len(_NAME_SCOPE_STACK):
        return _NAME_SCOPE_STACK[0]
    return ''


def depthwise_conv2d(x,
                     kernel,
                     strides=(1, 1),
                     padding='valid',
                     data_format=None,
                     dilation_rate=(1, 1)):
    return conv(x, kernel, strides, padding, data_format, dilation_rate, channelwise=True)


def dot(x, y, name=None):
    logger.debug('dot(x: {}, y: {}, name: {})'.format(x, y, name))
    return _KerasNode('dot', tensor=plaidml_op.dot(x.tensor, y.tensor), name=name)


def dropout(x, level, noise_shape=None, seed=None):
    logger.debug('dropout(x: {}, level: {}, noise_shape: {}, seed: {})'.format(
        x, level, noise_shape, seed))
    I = x.tensor
    if noise_shape is not None and len(noise_shape) != I.shape.ndims:
        raise ValueError("noise_shape ndims doesn't match input ndims")
    if noise_shape is None:
        shape = I.shape.dims
    else:
        shape = noise_shape
    rng_state = _make_rng_state(seed)
    R = 1.0 - level
    M = 1.0 / R
    T = edsl.prng(rng_state.tensor, shape)
    O = edsl.select(T < R, I * M, 0.0)
    return _KerasNode('dropout', tensor=O)


def dtype(x):
    return x.tensor.shape.dtype.into_numpy()


def elu(x, alpha=1.0):
    _report_unimplemented('elu')


def equal(x, y):
    logger.debug('equal(x: {}, y: {})'.format(x, y))
    if isinstance(x, _KerasNode):
        x = x.tensor
    if isinstance(x, np.ndarray):
        x = variable(x).tensor
    if isinstance(y, _KerasNode):
        y = y.tensor
    if isinstance(y, np.ndarray):
        y = variable(y).tensor
    return _KerasNode('equal', tensor=(x == y))


def exp(x):
    logger.debug('exp(x: {})'.format(x))
    return _KerasNode('exp', tensor=edsl.exp(x.tensor))


def eval(x):
    return get_value(x)


def expand_dims(x, axis=-1, name=None):
    logger.debug('expand_dims(x: {}, axis: {}, name={})'.format(x, axis, name))
    return _KerasNode('expand_dims', name=name, tensor=plaidml_op.expand_dims(x.tensor, axis))


def eye(size, dtype=None, name=None):
    _report_unimplemented('eye')


def flatten(x):
    logger.debug('flatten(x: {})'.format(x))
    _report_unimplemented('flatten')


def floor(x):
    logger.debug('floor(x: {})'.format(x))
    return _KerasNode('floor', tensor=edsl.floor(x.tensor))


def foldl(fn, elems, initializer=None, name=None):
    _report_unimplemented('foldl')


def foldr(fn, elems, initializer=None, name=None):
    _report_unimplemented('foldr')


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


def gather(x, indicies):
    logger.debug('gather(x: {}, indicies: {})'.format(x, indicies))
    return _KerasNode('gather', tensor=edsl.gather(x.tensor, indicies.tensor))


def get_uid(prefix=''):
    _UID_PREFIX_DICT[prefix] += 1
    return _UID_PREFIX_DICT[prefix]


def get_value(x):
    logger.debug('get_value(x: {})'.format(x))
    inputs = []
    fn = _Function(inputs, [x], [], name='get_value')
    outputs = fn(inputs)
    return outputs[0]


def get_variable_shape(x):
    return x._keras_shape


def gradients(loss, variables):
    logger.debug('gradients(loss: {}, variables: {})'.format(loss, variables))
    grads = edsl.gradients(loss.tensor, [x.tensor for x in variables])
    return [_KerasNode('gradients', tensor=x) for x in grads]


def greater(x, y):
    return x > y


def greater_equal(x, y):
    return x >= y


def hard_sigmoid(x):
    _report_unimplemented('hard_sigmoid')


def identity(x):
    logger.debug('identity(x: {})'.format(x))
    return _KerasNode('identity', tensor=edsl.ident(x.tensor))


def in_test_phase(x, alt, training=None):
    _report_unimplemented('in_test_phase')


def in_top_k(predictions, targets, k):
    _report_unimplemented('in_top_k')


def in_train_phase(x, alt, training=None):
    _report_unimplemented('in_train_phase')


def int_shape(x):
    return tuple(None if x == 0 else x for x in x.tensor.shape.int_dims)


def is_keras_tensor(x):
    # logger.debug('>>is_keras_tensor({})'.format(x))
    if not is_tensor(x):
        raise ValueError()
    return hasattr(x, '_keras_history')


def is_placeholder(x):
    _report_unimplemented('is_placeholder')


def is_sparse(x):
    return False


def is_tensor(x):
    # logger.debug('>>is_tensor({})'.format(x))
    return isinstance(x, _KerasNode)


def l2_normalize(x, axis):
    _report_unimplemented('l2_normalize')


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


def log(x):
    logger.debug('log(x: {})'.format(x))
    return _KerasNode('log', tensor=edsl.log(x.tensor))


def logsumexp(x, axis=None, keepdims=False):
    _report_unimplemented('logsumexp')


def manual_variable_initialization(value):
    _report_unimplemented('manual_variable_initialization')


def map_fn(fn, elems, name=None, dtype=None):
    _report_unimplemented('map_fn')


def max(x, axis=None, keepdims=False):
    logger.debug('max(x: {}, axis: {}, keepdims: {})'.format(x, axis, keepdims))
    return _KerasNode('max', tensor=plaidml_op.max(x.tensor, axis, keepdims))


def maximum(x, y):
    logger.debug('maximum(x: {}, y: {})'.format(x, y))
    return _KerasNode('maximum', tensor=edsl.max(x.tensor, y.tensor))


def mean(x, axis=None, keepdims=False):
    logger.debug('mean(x: {}, axis: {}, keepdims: {})'.format(x, axis, keepdims))
    return _KerasNode('mean', tensor=plaidml_op.mean(x.tensor, axis, keepdims))


def min(x, axis=None, keepdims=False):
    logger.debug('min(x: {}, axis: {}, keepdims: {})'.format(x, axis, keepdims))
    return _KerasNode('min', tensor=plaidml_op.min(x.tensor, axis, keepdims))


def minimum(x, y):
    logger.debug('minimum(x: {}, y: {})'.format(x, y))
    return _KerasNode('minimum', tensor=edsl.min(x.tensor, y.tensor))


def moving_average_update(x, value, momentum):
    _report_unimplemented('moving_average_update')


@contextmanager
def name_scope(name):
    _NAME_SCOPE_STACK.append(name)
    logger.debug('name_scope({}), push: {}'.format(name, _NAME_SCOPE_STACK))
    yield
    _NAME_SCOPE_STACK.pop()
    logger.debug('name_scope({}), pop: {}'.format(name, _NAME_SCOPE_STACK))


def ndim(x):
    logger.debug('ndim({})'.format(x))
    return len(x._keras_shape)


def not_equal(lhs, rhs):
    logger.debug('not_equal(lhs: {}, rhs: {})'.format(lhs, rhs))
    if isinstance(lhs, _KerasNode):
        lhs = lhs.tensor
    if isinstance(lhs, np.ndarray):
        lhs = variable(lhs).tensor
    if isinstance(rhs, _KerasNode):
        rhs = rhs.tensor
    if isinstance(rhs, np.ndarray):
        rhs = variable(rhs).tensor
    return _KerasNode('not_equal', tensor=(lhs != rhs))


def normalize_batch_in_training(x, gamma, beta, reduction_axes, epsilon=1e-3):
    _report_unimplemented('normalize_batch_in_training')


def one_hot(indices, num_classes):
    #Note: does not error check for entries in indices that are >= num_classes
    logger.debug('one_hot(indices: {}, num_classes: {})'.format(indices, num_classes))
    count = variable(np.array(range(num_classes)), dtype='int32').tensor
    I = indices.tensor
    I_ndims = I.shape.ndims
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


def ones(shape, dtype=None, name=None):
    _report_unimplemented('ones')


def ones_like(x, dtype=None, name=None):
    _report_unimplemented('ones_like')


def permute_dimensions(x, pattern=None):
    logger.debug('permute_dimensions(x: {}, pattern: {})'.format(x, pattern))
    return _KerasNode('permute_dimensions', tensor=plaidml_op.transpose(x.tensor, pattern))


def placeholder(shape=None, ndim=None, dtype=None, sparse=False, name=None):
    logger.debug('placeholder(shape: {}, ndim: {}, dtype: {}, sparse: {}, name: {})'.format(
        shape, ndim, dtype, sparse, name))
    dtype = plaidml.DType.from_numpy(dtype or floatx())
    # TODO: Need to support empty shapes; once supported, convert below to `if _ is not None`
    if shape:
        return _KerasNode('placeholder', shape=edsl.LogicalShape(dtype, shape), name=name)
    if ndim:
        return _KerasNode('placeholder', shape=edsl.LogicalShape(dtype, [0] * ndim), name=name)
    raise ValueError()


def pool(x, pool_size, strides=None, padding='valid', data_format=None, pool_mode='max'):
    logger.debug(
        'pool(x: {}, pool_size: {}, strides: {}, padding: {}, data_format: {}, pool_mode: {})'.
        format(x, pool_size, strides, padding, data_format, pool_mode))
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


def pool2d(x, pool_size, strides=(1, 1), padding='valid', data_format=None, pool_mode='max'):
    return pool(x=x,
                pool_size=pool_size,
                strides=strides,
                padding=padding,
                data_format=data_format,
                pool_mode=pool_mode)


def pool3d(x, pool_size, strides=(1, 1, 1), padding='valid', data_format=None, pool_mode='max'):
    return pool(x=x,
                pool_size=pool_size,
                strides=strides,
                padding=padding,
                data_format=data_format,
                pool_mode=pool_mode)


def pow(x, a):
    logger.debug('pow(x: {}, a: {})'.format(x, a))
    return _KerasNode('pow', tensor=edsl.pow(x.tensor, a))


def print_tensor(x, message=''):
    _report_unimplemented('print_tensor')


def prod(value, axis=None, keepdims=False):
    _report_unimplemented('prod')


def random_binomial(shape, p=0.0, dtype=None, see=None):
    _report_unimplemented('random_binomial')


def random_normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
    _report_unimplemented('random_normal')


def random_normal_variable(shape, mean, scale, dtype=None, name=None, seed=None):
    _report_unimplemented('random_normal_variable')


def random_uniform(shape, minval=0.0, maxval=1.0, dtype=None, seed=None):
    logger.debug('random_uniform(shape: {}, minval: {}, maxval: {}, dtype: {}, seed: {})'.format(
        shape, minval, maxval, dtype, seed))
    rng_state = _make_rng_state(seed)
    R = edsl.prng(rng_state.tensor, shape)
    dtype = dtype or floatx()
    if dtype != 'float32':
        R = edsl.cast(R, plaidml.DType.from_numpy(dtype))
    O = (maxval - minval) * R + minval
    return _KerasNode('random_uniform', tensor=O)


def random_uniform_variable(shape, low, high, dtype=None, name=None, seed=None):
    _report_unimplemented('random_uniform_variable')


def relu(x, alpha=None, max_value=None, threshold=0.):
    logger.debug('relu(x: {}, alpha: {}, max_value: {}, threshold: {})'.format(
        x, alpha, max_value, threshold))
    return _KerasNode('relu', tensor=plaidml_op.relu(x.tensor, alpha, max_value, threshold))


def repeat(x, n):
    _report_unimplemented('repeat')


def repeat_elements(x, rep, axis):
    _report_unimplemented('repeat_elements')


def reset_uids():
    global _UID_PREFIX_DICT
    _UID_PREFIX_DICT.clear()


def reshape(x, dims):
    # TODO: This needs to be more thoroughly tested with symbolic shapes
    logger.debug('reshape(x: {}, dims: {})'.format(x, dims))
    dims = list(dims)
    I = x.tensor
    I_dims = edsl.TensorDims(I.shape.ndims)
    I.bind_dims(*I_dims)
    neg_idx = None
    for idx, dim in enumerate(dims):
        if isinstance(dim, edsl.TensorDim):
            continue
        if dim == 0 or dim is None:
            dims[idx] = I_dims[idx]  # TODO: Fix how we manage shape
        elif dim == -1:
            if neg_idx:
                raise RuntimeError('At most one dimension of size -1 may be provided in Reshape')
            neg_idx = idx
            dims[idx] = 1  # Just to simplify the size computation later
    if neg_idx is not None:
        # Compute the value to use for the -1 dimension in the
        # output shape, by making it what it needs to be in order
        # to preserve the correct number of elements in the
        # tensor.
        #
        # This code is a little tricky because symbolic values
        # (e.g. the batch size in a typical neural network) may
        # appear in both the original shape and the target shape.
        # Naively multiplying the original shape's dimensions and
        # dividing by the target shape's dimensions (excluding the
        # -1 dimension) would produce a symbolic value.
        #
        # So:
        #
        # We scan the input dimensions, counting the number of
        # instances of each symbolic size encountered and
        # multiplying together the non-symbolic sizes into the
        # numerator.
        #
        # We then scan the output dimensions.  Where there's a
        # symbolic size, we check and see if we have a count for
        # it, and decrement the count if we do.  Otherwise -- if
        # we don't have a count for it, or if it's not symbolic --
        # we multiply it into the denominator.
        #
        # We then take the remaining symbolic input dimensions,
        # and multiply them into the numerator -- these are the
        # dimensions that haven't been cancelled out.
        #
        # And then the size of the -1 dimension is just numerator
        # / denominator; if there are any remaining uncancelled
        # symbolic dimension sizes, the output will be symbolic,
        # but otherwise we'll come out with a concrete dimension
        # size.

        num = 1
        syms = defaultdict(int)
        for idx, dim in enumerate(I.shape.int_dims):
            if dim is None:
                syms[I_dims[idx]] += 1
            else:
                num *= dim
        den = 1
        for dim in dims:
            if isinstance(dim, edsl.TensorDim) and syms[dim] > 0:
                syms[dim] -= 1
            else:
                den *= dim
        for sym, count in syms.items():
            for _ in range(count):
                num *= sym
        dims[neg_idx] = num // den
    return _KerasNode('reshape', tensor=edsl.reshape(I, dims))


def resize_images(x, height_factor, width_factor, data_format, interpolation='nearest'):
    _report_unimplemented('resize_images')


def resize_volumes(x, depth_factor, height_factor, width_factor, data_format):
    _report_unimplemented('resize_volumes')


def reverse(x, axes):
    logger.debug('reverse(x: {}, axes: {})'.format(x, axes))
    I = x.tensor
    ndims = I.shape.ndims
    if isinstance(axes, six.integer_types):
        axes = [axes]
    normalized_axes = list()
    for axis in axes:
        if not isinstance(axis, six.integer_types):
            raise ValueError(
                'The axes parameter of reverse only accepts an integer or a list of integers, received {}'
                .format(type(axis)))
        normalized_axes.append(_normalize_axis(axis, ndims, 'reverse'))

    dims = edsl.TensorDims(ndims)
    I.bind_dims(*dims)
    I_idxs = edsl.TensorIndexes(ndims)
    O_idxs = I_idxs.copy()
    for axis in axes:
        O_idxs[axis] = -I_idxs[axis] + dims[axis] - 1
    O = edsl.TensorOutput(*dims)
    O[O_idxs] = I[I_idxs]
    return _KerasNode('reverse', name='reverse', tensor=O)


def reverse_gradient(x, coeff=1.0):
    _report_unimplemented('reverse_gradient')


def rnn(step_function,
        inputs,
        initial_states,
        go_backwards=False,
        mask=None,
        constants=None,
        unroll=False,
        input_length=None):
    logger.debug(
        'rnn(step_function: {}, inputs: {}, initial_states: {}, mask: {}, constants: {}, unroll: {}, input_length: {})'
        .format(step_function, inputs, initial_states, mask, constants, unroll, input_length))
    _report_unimplemented('rnn')
    # if input_length is None:
    #     input_length = inputs.tensor.shape.dims[1]
    # states = initial_states
    # for i in range(input_length):
    #     input_val = inputs[:, i]
    #     output_val, new_states = step_function(input_val, states + constants)
    # return (output_val, output, states)


def round(x):
    logger.debug('round(x: {})'.format(x))
    return _KerasNode('round', tensor=edsl.round(x.tensor))


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
    rank = x.tensor.shape.ndims - 2
    ones = tuple(1 for _ in range(rank))
    return conv(intermediate,
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
    logger.debug('set_floatx(dtype: {})'.format(dtype))
    keras_set_floatx(dtype)
    # plaidml.set_floatx(ptile.convert_np_dtype_to_pml(dtype))


def set_learning_phase(value):
    if value != 0 and value != 1:
        raise ValueError("May only set_learning_phase to 0 or 1")
    value = int(value)
    global _in_train_phase
    _in_train_phase = value


def set_value(x, value):
    _report_unimplemented('set_value')


def shape(x):
    logger.debug('shape(x: {})'.format(x))
    return _KerasNode('shape', tensor=edsl.shape(x.tensor))


def sigmoid(x):
    logger.debug('sigmoid(x: {})'.format(x))
    return _KerasNode('sigmoid', tensor=edsl.sigmoid(x.tensor))


def sign(x):
    logger.debug('sign(x: {})'.format(x))
    intermediate = _KerasNode('sign_intermediate', tensor=edsl.select((x > 0).tensor, 1., -1.))
    return _KerasNode('sign', tensor=edsl.select((x.tensor == 0.), 0., intermediate.tensor))


def sin(x):
    logger.debug('sin(x: {})'.format(x))
    return _KerasNode('sin', tensor=edsl.sin(x.tensor))


def softmax(x):
    logger.debug('softmax(x: {})'.format(x))
    y = plaidml_op.softmax(x.tensor, axis=x.tensor.shape.ndims - 1)
    return _KerasNode('softmax', tensor=y)


def softmax(x, axis=None, name=None):
    logger.debug('softmax(x: {}, axis: {}, name: {})'.format(x, axis, name))
    if name is None:
        name = 'softmax'
    I = x.tensor
    ndims = I.shape.ndims
    I_dims = edsl.TensorDims(ndims)
    I.bind_dims(*I_dims)
    if axis is None:
        axis = ndims - 1
    axis = _normalize_axis(axis=axis, ndims=ndims, name=name + ' (softmax)')
    if ndims == 2 and axis == 1:
        return _KerasNode(name, tensor=plaidml_op.softmax(I, axis=1))

    if axis == 0:
        group = 1
    else:
        group = functools.reduce(lambda x, y: x * y, I_dims[:axis])
    values = functools.reduce(lambda x, y: x * y, I_dims[axis:])
    flat_x = reshape(x, (group, values))
    result = _KerasNode(name, tensor=plaidml_op.softmax(flat_x.tensor, axis=1))
    return reshape(result, I_dims)


def softplus(x):
    logger.debug('softplus(x: {})'.format(x))
    return log(1. + exp(x))


def softsign(x):
    logger.debug('softsign(x: {})'.format(x))
    return x / (1. + abs(x))


def sparse_categorical_crossentropy(target, output, from_logits=False):
    dims = edsl.TensorDims(output.tensor.shape.ndims)
    output.tensor.bind_dims(*dims)
    return categorical_crossentropy(
        reshape(one_hot(target, output.tensor.shape.int_dims[-1]), dims), output, from_logits)


def spatial_2d_padding(x, padding=((1, 1), (1, 1)), data_format=None):
    data_format = _normalize_data_format(data_format)
    lo_pads = [padding[i][0] for i in range(2)]
    hi_pads = [padding[i][1] for i in range(2)]
    return _KerasNode('spatial_2d_padding',
                      tensor=plaidml_op.spatial_padding(x.tensor,
                                                        lo_pads=lo_pads,
                                                        hi_pads=hi_pads,
                                                        data_layout=data_format))


def spatial_3d_padding(x, padding=((1, 1), (1, 1), (1, 1)), data_format=None):
    data_format = _normalize_data_format(data_format)
    lo_pads = [padding[i][0] for i in range(3)]
    hi_pads = [padding[i][1] for i in range(3)]
    return _KerasNode('spatial_2d_padding',
                      tensor=plaidml_op.spatial_padding(x.tensor,
                                                        lo_pads=lo_pads,
                                                        hi_pads=hi_pads,
                                                        data_layout=data_format))


def sqrt(x):
    logger.debug('sqrt(x: {})'.format(x))
    return _KerasNode('sqrt', tensor=edsl.sqrt(x.tensor))


def square(x):
    logger.debug('square(x: {})'.format(x))
    return _KerasNode('square', tensor=plaidml_op.square(x.tensor))


def squeeze(x, axis):
    logger.debug('squeeze(x: {}, axis: {})'.format(x, axis))
    return _KerasNode('squeeze', tensor=plaidml_op.squeeze(x.tensor, axis))


def stack(x, axis=0):
    logger.debug('expand_dims(x: {}, axis: {})'.format(x, axis))
    return concatenate([expand_dims(item, axis) for item in x], axis=axis)


def std(x, axis=None, keepdims=False):
    logger.debug('std(x: {}, axis: {}, keepdims: {})'.format(x, axis, keepdims))
    return sqrt(var(x, axis=axis, keepdims=keepdims))


def stop_gradient(variables):
    _report_unimplemented('stop_gradient')


def sum(x, axis=None, keepdims=False):
    logger.debug('sum(x: {}, axis: {}, keepdims: {})'.format(x, axis, keepdims))
    return _KerasNode('sum', tensor=plaidml_op.sum(x.tensor, axis, keepdims))


def switch(condition, then_expression, else_expression):
    logger.debug('switch(condition: {}, then_expression: {}, else_expression: {})'.format(
        condition, then_expression, else_expression))
    return _KerasNode('switch',
                      tensor=edsl.select(condition.tensor, then_expression.tensor,
                                         else_expression.tensor))


def tanh(x):
    logger.debug('tanh(x: {})'.format(x))
    return _KerasNode('tanh', tensor=edsl.tanh(x.tensor))


def temporal_padding(x, padding=(1, 1)):
    data_format = _normalize_data_format(None)  # uses image_data_format()
    lo_pads = [padding[0]]
    hi_pads = [padding[1]]
    return _KerasNode('temporal_padding',
                      tensor=plaidml_op.spatial_padding(x.tensor,
                                                        lo_pads=lo_pads,
                                                        hi_pads=hi_pads,
                                                        data_layout=data_format))


def tile(x, n):
    logger.debug('tile(x: {}, n: {})'.format(x, n))
    return _KerasNode('tile', tensor=plaidml_op.tile(x.tensor, n))


def to_dense(tensor):
    _report_unimplemented('to_dense')


def transpose(x):
    logger.debug('transpose(x: {})'.format(x))
    return _KerasNode('transpose', tensor=plaidml_op.transpose(x.tensor))


def truncated_normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
    logger.debug('truncated_normal(shape: {}, mean: {}, stddev: {}, dtype: {}, seed: {})'.format(
        shape, mean, stddev, dtype, seed))
    if dtype is None:
        dtype = floatx()
    if seed:
        np.random.seed(seed)
    return variable(stddev * scipy.stats.truncnorm.rvs(-2.0, 2.0, size=shape) + mean, dtype)


def update(x, new_x):
    logger.debug('update(x: {}, new_x: {})'.format(x, new_x))
    return (x, new_x)


def update_add(x, increment):
    logger.debug('update_add(x: {}, increment: {})'.format(x, increment))
    return (x, x + increment)


def update_sub(x, decrement):
    logger.debug('update_sub(x: {}, decrement: {})'.format(x, decrement))
    return (x, x - decrement)


def var(x, axis=None, keepdims=False):
    logger.debug('var(x: {}, axis: {}, keepdims: {})'.format(x, axis, keepdims))
    return _KerasNode('var', tensor=plaidml_op.variance(x.tensor, axis, keepdims))


def variable(value, dtype=None, name=None, constraint=None):
    logger.debug('variable(value: {}, dtype: {}, name: {}, constraint: {})'.format(
        value, dtype, name, constraint))
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


def zeros(shape, dtype=None, name=None):
    logger.debug('zeros(shape: {}, dtype: {}, name: {})'.format(shape, dtype, name))
    value = np.full(shape, 0, dtype=dtype or floatx())
    return _KerasNode('zeros', name=name, value=value)


def zeros_like(x, dtype=None, name=None):
    logger.debug('zeros_like(x: {}, dtype: {}, name: {})'.format(x, dtype, name))
    value = np.full((1), 0, dtype=dtype or floatx())
    zero = _create_var('a_zero', value)
    I = x.tensor
    ndim = I.shape.ndims
    dims = edsl.TensorDims(ndim)
    idxs = edsl.TensorIndexes(ndim)
    I.bind_dims(*dims)
    O = edsl.TensorOutput(*dims)
    O[idxs] = zero[0]
    return _KerasNode('zeros_like', name=name, tensor=O)
