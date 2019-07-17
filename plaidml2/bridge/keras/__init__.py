# Copyright 2019 Intel Corporation.

import logging
import os
from collections import defaultdict
from contextlib import contextmanager

import six

import numpy as np
import plaidml2 as plaidml
import plaidml2.edsl as edsl
import plaidml2.exec as plaidml_exec
import plaidml2.op as plaidml_op
import plaidml2.settings as plaidml_settings
from keras.backend.common import floatx
from keras.backend.common import image_data_format
from keras.backend.common import set_floatx as keras_set_floatx

logger = logging.getLogger(__name__)

# Keras needs us to keep track of unique IDs for prefix strings
# (for use with get_uid and reset_uids)
_UID_PREFIX_DICT = defaultdict(int)

_NAME_SCOPE_STACK = []

_device = plaidml_settings.get('PLAIDML_DEVICE')
_target = plaidml_settings.get('PLAIDML_TARGET')


def _prepend_name_scope(name, default):
    if name:
        r = '_'.join(_NAME_SCOPE_STACK + [name])
    else:
        r = '_'.join(_NAME_SCOPE_STACK + [default])
        r += '_' + str(get_uid(r))
    return r


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
        return 'same_lower'
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
        program = edsl.Program(self._name, [x.tensor for x in self._outputs])

        def make_buffer(tensor):
            # convert LogicalShape into TensorShape
            shape = plaidml.TensorShape(tensor.shape.dtype, tensor.shape.int_dims)
            return plaidml.Buffer(_device, shape)

        input_bindings = [(x.tensor, make_buffer(x.tensor)) for x in self._inputs]
        output_bindings = [(x, make_buffer(x)) for x in program.outputs]
        return plaidml_exec.Executable(program, _device, _target, input_bindings, output_bindings)


class _KerasNode(object):

    def __init__(self, opname, name=None, shape=None, tensor=None):
        self.name = _prepend_name_scope(name, opname)
        if tensor is None:
            tensor = edsl.Tensor(shape=shape, name=self.name)
        # logger.debug('_KerasNode({})'.format(tensor))
        self.tensor = tensor

    def __repr__(self):
        return str(self.tensor)

    def __str__(self):
        return '{}: {}'.format(self.name, self.tensor.shape)

    def eval(self):
        return get_value(self)

    @property
    def _keras_shape(self):
        return int_shape(self)

    @_keras_shape.setter
    def _keras_shape(self, value):
        raise NotImplementedError()

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
        if isinstance(other, _KerasNode):
            other = other.tensor
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


def abs(x):
    _report_unimplemented('abs')


def all(x, axis=None, keepdims=False):
    _report_unimplemented('all')


def any(x, axis=None, keepdims=False):
    _report_unimplemented('any')


def arange(start, stop=None, step=1, dtype='int32'):
    _report_unimplemented('arange')


def argmax(x, axis=-1):
    _report_unimplemented('argmax')


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
    _report_unimplemented('bias_add')


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
    _report_unimplemented('categorical_crossentropy')


def ceil(x):
    _report_unimplemented('ceil')


def clear_session():
    _report_unimplemented('clear_session')


def clip(x, min_val, max_val):
    _report_unimplemented('clip')


def concatenate(tensors, axis=-1):
    _report_unimplemented('concatenate')


def constant(value, dtype=None, shape=None, name=None):
    logger.debug('constant(value: {}, dtype: {}, shape: {}, name: {})'.format(
        value, dtype, shape, name))
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
    _report_unimplemented('conv')


def conv_transpose(x, kernel, output_shape, strides, padding, data_format, dilation_rate):
    _report_unimplemented('conv_transpose')


def conv1d(x, kernel, strides=1, padding='valid', data_format=None, dilation_rate=1):
    _report_unimplemented('conv1d')


def conv2d(x, kernel, strides=(1, 1), padding='valid', dilation_rate=(1, 1), data_format=None):
    _report_unimplemented('conv2d')


def conv2d_transpose(x,
                     kernel,
                     output_shape,
                     strides=(1, 1),
                     padding='valid',
                     data_format=None,
                     dilation_rate=(1, 1)):
    _report_unimplemented('conv2d_transpose')


def conv3d(x,
           kernel,
           strides=(1, 1, 1),
           padding='valid',
           dilation_rate=(1, 1, 1),
           data_format=None):
    _report_unimplemented('conv3d')


def conv3d_transpose(x,
                     kernel,
                     output_shape,
                     strides=(1, 1, 1),
                     padding='valid',
                     data_format=None,
                     dilation_rate=(1, 1, 1)):
    _report_unimplemented('conv3d_transpose')


def count_params(x):
    _report_unimplemented('count_params')


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
    _report_unimplemented('depthwise_conv2d')


def dot(x, y, name=None):
    _report_unimplemented('dot')


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
    _report_unimplemented('equal')


def exp(x):
    logger.debug('exp(x: {})'.format(x))
    return _KerasNode('exp', tensor=edsl.exp(x.tensor))


def eval(x):
    return get_value(x)


def expand_dims(x, axis=-1, name=None):
    logger.debug('expand_dims(x: {}, axis: {}, name={})'.format(x, axis, name))
    I = x.tensor
    ndims = I.shape.ndims
    if axis < 0:
        axis = ndims + 1 + axis
    dims_in = edsl.TensorDims(ndims)
    idxs_in = edsl.TensorIndexes(ndims)
    dims_out = dims_in[0:axis] + [1] + dims_in[axis:]
    idxs_out = idxs_in[0:axis] + [0] + idxs_in[axis:]
    I.bind_dims(*dims_in)
    O = edsl.TensorOutput(*dims_out)
    O[idxs_out] = I[idxs_in]
    return _KerasNode('expand_dims', name=name, tensor=O)


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
    logger.debug('function(inputs: {}, outputs: {}, updates: {}, name: {})'.format(
        inputs, outputs, updates, name))
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
    _report_unimplemented('learning_phase')


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
    _report_unimplemented('max')


def maximum(x, y):
    logger.debug('maximum(x: {}, y: {})'.format(x, y))
    return _KerasNode('maximum', tensor=edsl.max(x.tensor, y.tensor))


def mean(x, axis=None, keepdims=False):
    logger.debug('mean(x: {}, axis: {}, keepdims: {})'.format(x, axis, keepdims))
    return _KerasNode('mean', tensor=plaidml_op.mean(x.tensor, axis, keepdims))


def min(x, axis=None, keepdims=False):
    _report_unimplemented('min')


def minimum(x, y):
    logger.debug('minimum(x: {}, y: {})'.format(x, y))
    return _KerasNode('minimum', tensor=edsl.min(x.tensor, y.tensor))


def moving_average_update(x, value, momentum):
    _report_unimplemented('moving_average_update')


@contextmanager
def name_scope(name):
    # logger.debug('name_scope({})'.format(name))
    _NAME_SCOPE_STACK.append(name)
    yield
    _NAME_SCOPE_STACK.pop()


def ndim(x):
    logger.debug('ndim({})'.format(x))
    return len(x._keras_shape)


def not_equal(lhs, rhs):
    logger.debug('not_equal(lhs: {}, rhs: {})'.format(lhs, rhs))
    if isinstance(rhs, _KerasNode):
        O = lhs.tensor != rhs.tensor
        return _KerasNode('not_equal', tensor=O)
    O = lhs.tensor != rhs
    return _KerasNode('not_equal', tensor=O)


def normalize_batch_in_training(x, gamma, beta, reduction_axes, epsilon=1e-3):
    _report_unimplemented('normalize_batch_in_training')


def one_hot(indices, num_classes):
    _report_unimplemented('one_hot')


def ones(shape, dtype=None, name=None):
    _report_unimplemented('ones')


def ones_like(x, dtype=None, name=None):
    _report_unimplemented('ones_like')


def permute_dimensions(x, pattern):
    _report_unimplemented('permute_dimensions')


def placeholder(shape=None, ndim=None, dtype=None, sparse=False, name=None):
    logger.debug('placeholder(shape: {}, ndim: {}, dtype: {}, sparse: {}, name: {})'.format(
        shape, ndim, dtype, sparse, name))
    dtype = plaidml.DType.from_numpy(dtype or floatx())
    if shape:
        return _KerasNode('placeholder', shape=edsl.LogicalShape(dtype, shape), name=name)
    if ndim:
        return _KerasNode('placeholder', shape=edsl.LogicalShape(dtype, [0] * ndim), name=name)
    raise ValueError()


def pool(x, pool_size, strides=None, padding='valid', data_format=None, pool_mode='max'):
    logger.debug(
        'pool(x: {}, pool_size: {}, strides: {}, padding: {}, data_format: {}, pool_mode: {})'.
        format(x, pool_size, strides, padding, data_format, pool_mode))
    return _KerasNode(
        'pool',
        tensor=plaidml_op.pool(
            x.tensor,  #
            pool_mode,  #
            pool_size,  #
            strides,  #
            _normalize_padding(padding),  #
            tuple(),  #
            _normalize_data_format(data_format),  #
            False,  #
            False))


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
    dtype = dtype or floatx()
    rng_state = _make_rng_state(seed)
    R = edsl.prng(rng_state.tensor, shape)
    if dtype != 'float32':
        R = edsl.cast(R, dtype)
    O = (maxval - minval) * R + minval
    return _KerasNode('random_uniform', tensor=O)


def random_uniform_variable(shape, low, high, dtype=None, name=None, seed=None):
    _report_unimplemented('random_uniform_variable')


def relu(x, alpha=None, max_value=None, threshold=0.):
    logger.debug('relu(x: {}, alpha: {}, max_value: {}, threshold: {})'.format(
        x, alpha, max_value, threshold))
    _report_unimplemented('relu')


def repeat(x, n):
    _report_unimplemented('repeat')


def repeat_elements(x, rep, axis):
    _report_unimplemented('repeat_elements')


def reset_uids():
    global _UID_PREFIX_DICT
    _UID_PREFIX_DICT.clear()


def reshape(x, dims):
    logger.debug('reshape(x: {}, dims: {})'.format(x, dims))
    _report_unimplemented('reshape')


def resize_images(x, height_factor, width_factor, data_format, interpolation='nearest'):
    _report_unimplemented('resize_images')


def resize_volumes(x, depth_factor, height_factor, width_factor, data_format):
    _report_unimplemented('resize_volumes')


def reverse(x, axes):
    _report_unimplemented('reverse')


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
    _report_unimplemented('round')


def separable_conv(x,
                   depthwise_kernel,
                   pointwise_kernel,
                   strides=None,
                   padding='valid',
                   data_format=None,
                   dilation_rate=None):
    _report_unimplemented('separable_conv')


def separable_conv2d(x,
                     depthwise_kernel,
                     pointwise_kernel,
                     strides=(1, 1),
                     padding='valid',
                     data_format=None,
                     dilation_rate=(1, 1)):
    _report_unimplemented('separable_conv2d')


def set_floatx(dtype):
    logger.debug('set_floatx(dtype: {})'.format(dtype))
    keras_set_floatx(dtype)
    # plaidml.set_floatx(ptile.convert_np_dtype_to_pml(dtype))


def set_learning_phase(value):
    _report_unimplemented('set_learning_phase')


def set_value(x, value):
    _report_unimplemented('set_value')


def shape(x):
    logger.debug('shape(x: {})'.format(x))
    return _KerasNode('shape', tensor=edsl.shape(x.tensor))


def sigmoid(x):
    logger.debug('sigmoid(x: {})'.format(x))
    return _KerasNode('sigmoid', tensor=edsl.sigmoid(x.tensor))


def sign(x):
    _report_unimplemented('sign')


def sin(x):
    logger.debug('sin(x: {})'.format(x))
    return _KerasNode('sin', tensor=edsl.sin(x.tensor))


def softmax(x):
    _report_unimplemented('softmax')


def softplus(x):
    _report_unimplemented('softplus')


def softsign(x):
    _report_unimplemented('softsign')


def sparse_categorical_crossentropy(target, output, from_logits=False):
    _report_unimplemented('sparse_categorical_crossentropy')


def spatial_2d_padding(x, padding=((1, 1), (1, 1)), data_format=None):
    _report_unimplemented('spatial_2d_padding')


def spatial_3d_padding(x, padding=((1, 1), (1, 1), (1, 1)), data_format=None):
    _report_unimplemented('spatial_3d_padding')


def sqrt(x):
    logger.debug('sqrt(x: {})'.format(x))
    return _KerasNode('sqrt', tensor=edsl.sqrt(x.tensor))


def square(x):
    logger.debug('square(x: {})'.format(x))
    return _KerasNode('square', tensor=plaidml_op.square(x.tensor))


def squeeze(x, axis):
    _report_unimplemented('squeeze')


def stack(x, axis=0):
    _report_unimplemented('stack')


def std(x, axis=None, keepdims=False):
    _report_unimplemented('std')


def stop_gradient(variables):
    _report_unimplemented('stop_gradient')


def sum(x, axis=None, keepdims=False):
    logger.debug('sum(x: {}, axis: {}, keepdims: {})'.format(x, axis, keepdims))
    return _KerasNode('sum', tensor=plaidml_op.sum(x.tensor, axis, keepdims))


def switch(condition, then_expression, else_expression):
    _report_unimplemented('switch')


def tanh(x):
    logger.debug('tanh(x: {})'.format(x))
    return _KerasNode('tanh', tensor=edsl.tanh(x.tensor))


def temporal_padding(x, padding=(1, 1)):
    _report_unimplemented('temporal_padding')


def tile(x, n):
    logger.debug('tile(x: {}, n: {})'.format(x, n))
    I = x.tensor
    ndims = I.shape.ndims
    if len(n) != ndims:
        raise PlaidMLKerasException('Tile size dimensions doesn\'t match ndims')
    dims = edsl.TensorDims(ndims)
    idxs = edsl.TensorIndexes(ndims)
    I.bind_dims(*dims)
    out_idxs = [edsl.TensorIndex() * dims[i] + idxs[i] for i in range(ndims)]
    out_dims = [dims[i] * n[i] for i in range(ndims)]
    O = edsl.TensorOutput(*out_dims)
    O[out_idxs] = I[idxs]
    O.no_defract()
    return _KerasNode('tile', tensor=O)


def to_dense(tensor):
    _report_unimplemented('to_dense')


def transpose(x):
    _report_unimplemented('transpose')


def truncated_normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
    _report_unimplemented('truncated_normal')


def update(x, new_x):
    _report_unimplemented('update')


def update_add(x, increment):
    return (x, x + increment)


def update_sub(x, decrement):
    return (x, x - decrement)


def var(x, axis=None, keepdims=False):
    _report_unimplemented('var')


def variable(value, dtype=None, name=None, constraint=None):
    logger.debug('variable(value: {}, dtype: {}, name: {}, constraint: {})'.format(
        value, dtype, name, constraint))
    if name is None:
        name = ''
    if isinstance(value, _KerasNode):
        return value
    if isinstance(value, float) or isinstance(value, six.integer_types):
        tensor = edsl.Tensor(value=value, name=name)
        return _KerasNode('variable', name=name, tensor=tensor)
    if isinstance(value, list) or isinstance(value, tuple):
        value = np.array(value)
    if isinstance(value, np.ndarray):
        # print(value.shape)
        dtype = plaidml.DType.from_numpy(dtype or floatx())
        shape = edsl.LogicalShape(dtype, value.shape)
        tensor_shape = plaidml.TensorShape(dtype, value.shape)
        buffer = plaidml.Buffer(_device, tensor_shape)
        buffer.copy_from_ndarray(value)
        tensor = edsl.Tensor(shape=shape, name=name, buffer=buffer)
        return _KerasNode('variable', name=name, tensor=tensor)
    raise TypeError('Unknown type for variable: {}'.format(type(value)))


def zeros(shape, dtype=floatx(), name=None):
    return constant(0.0, shape=shape, dtype=dtype, name=_prepend_name_scope(name, 'zeros'))


def zeros_like(x, dtype=floatx(), name=None):
    logger.debug('zeros_like(x: {}, dtype: {}, name: {})'.format(x, dtype, name))
    I = x.tensor
    dtype = dtype or floatx()
    a_zero = constant(0.0, shape=(1), dtype=dtype, name=_prepend_name_scope(name, 'a_zero'))
    ndim = I.shape.ndims
    dims = edsl.TensorDims(ndim)
    idxs = edsl.TensorIndexes(ndim)
    I.bind_dims(*dims)
    O = edsl.TensorOutput(*dims)
    O[idxs] = a_zero.tensor[0]
    return _KerasNode('zeros_like', name=name, tensor=O)
