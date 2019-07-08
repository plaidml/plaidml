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
        output_bindings = [(x.tensor, make_buffer(x.tensor)) for x in self._outputs]
        return plaidml_exec.Executable(program, _device, _target, input_bindings, output_bindings)


class _KerasNode(object):

    def __init__(self, opname, name=None, shape=None, tensor=None):
        name = _prepend_name_scope(name, opname)
        if tensor is None:
            tensor = edsl.Tensor(shape=shape, name=name)
        # logger.debug('_KerasNode({})'.format(tensor))
        self.tensor = tensor

    def __repr__(self):
        return str(self.tensor)

    def __str__(self):
        return str(self.tensor)

    def eval(self):
        return get_value(self)

    @property
    def _keras_shape(self):
        return int_shape(self)

    @_keras_shape.setter
    def _keras_shape(self, value):
        raise NotImplementedError()

    def __getitem__(self, key):
        # logger.debug('__getitem__(self: {}, key: {})'.format(self, key))
        # TODO: slice_of
        pass

    def __neg__(self):
        return _KerasNode('neg', -self.tensor)

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


def backend():
    return 'edsl_plaidml'


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


def dtype(x):
    return x.tensor.shape.dtype.into_numpy()


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


def eval(x):
    return get_value(x)


def function(inputs, outputs, updates=None, name=None):
    logger.debug('function(inputs: {}, outputs: {}, updates: {}, name: {})'.format(
        inputs, outputs, updates, name))
    if updates is None:
        updates = []
    if name is None:
        name = ''
    return _Function(inputs, outputs, updates, name)


def get_uid(prefix=''):
    _UID_PREFIX_DICT[prefix] += 1
    return _UID_PREFIX_DICT[prefix]


def get_value(x):
    inputs = []
    fn = _Function(inputs, [x], [], name='get_value')
    outputs = fn(inputs)
    return outputs[0]


def int_shape(x):
    return tuple(None if x == 0 else x for x in x.tensor.shape.int_dims)


def is_keras_tensor(x):
    # logger.debug('>>is_keras_tensor({})'.format(x))
    if not is_tensor(x):
        raise ValueError()
    return hasattr(x, '_keras_history')


def is_sparse(x):
    return False


def is_tensor(x):
    # logger.debug('>>is_tensor({})'.format(x))
    return isinstance(x, _KerasNode)


def mean(x, axis=None, keepdims=False):
    logger.debug('mean(x: {}, axis: {}, keepdims: {})'.format(x, axis, keepdims))
    return _KerasNode('mean', tensor=plaidml_op.mean(x.tensor, axis, keepdims))


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


def placeholder(shape=None, ndim=None, dtype=None, sparse=False, name=None):
    logger.debug('placeholder(shape: {}, ndim: {}, dtype: {}, sparse: {}, name: {})'.format(
        shape, ndim, dtype, sparse, name))
    dtype = plaidml.DType.from_numpy(dtype or floatx())
    if shape:
        return _KerasNode('placeholder', shape=edsl.LogicalShape(dtype, shape), name=name)
    if ndim:
        return _KerasNode('placeholder', shape=edsl.LogicalShape(dtype, [0] * ndim), name=name)
    raise ValueError()


def random_uniform(shape, minval=0.0, maxval=1.0, dtype=None, seed=None):
    logger.debug('random_uniform(shape: {}, minval: {}, maxval: {}, dtype: {}, seed: {})'.format(
        shape, minval, maxval, dtype, seed))
    dtype = dtype or floatx()
    rng_state = _make_rng_state(seed)
    # for x in shape:
    #     print('  {}: {}'.format(type(x), x))
    T = edsl.prng_step(rng_state.tensor, shape)
    # n = edsl.prng_state(T)
    O = edsl.prng_value(T)
    # if dtype != 'float32':
    #     O = edsl.cast()
    O = (maxval - minval) * O + minval
    return _KerasNode('random_uniform', tensor=O)


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
    if input_length is None:
        input_length = inputs.shape.dims[1]
    states = initial_states
    for i in range(input_length):
        input_val = inputs[:, i]
        output_val, new_states = step_function(input_val, states + constants)
    return (output_val, output, states)


def set_floatx(dtype):
    logger.debug('set_floatx(dtype: {})'.format(dtype))
    keras_set_floatx(dtype)
    # plaidml.set_floatx(ptile.convert_np_dtype_to_pml(dtype))


def square(x):
    logger.debug('square(x: {})'.format(x))
    return _KerasNode('square', tensor=plaidml_op.square(x.tensor))


def sum(x, axis=None, keepdims=False):
    logger.debug('sum(x: {}, axis: {}, keepdims: {})'.format(x, axis, keepdims))
    return _KerasNode('sum', tensor=plaidml_op.sum(x.tensor, axis, keepdims))


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


def zeros_like(x, dtype=floatx(), name=None):
    logger.debug('zeros_like(z: {}, dtype: {}, name: {})'.format(x, dtype, name))
    I = x.tensor
    dtype = dtype or floatx()
    a_zero = constant(0.0, shape=(1), dtype=dtype, name=_prepend_name_scope(name, 'a_zero'))
    ndim = I.shape.ndims
    dims = edsl.TensorDims(ndim)
    idxs = edsl.TensorIndexes(ndim)
    I.bind_dims(*dims)
    O = edsl.TensorOutput(*dims)
    O[idxs] = a_zero.tensor[()]
    return _KerasNode('zeros_like', name=name, tensor=O)
