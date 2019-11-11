# Copyright 2019 Intel Corporation.

import logging
from collections import namedtuple

import numpy as np
import six

from plaidml2 import DType
from plaidml2.core import TensorShape
from plaidml2.ffi import ForeignObject, ffi, ffi_call, lib

logger = logging.getLogger(__name__)


def __init():
    ffi_call(lib.plaidml_edsl_init)


ffi.init_once(__init, 'plaidml_edsl_init')


class LogicalShape(ForeignObject):
    __ffi_del__ = lib.plaidml_logical_shape_free
    __ffi_repr__ = lib.plaidml_logical_shape_repr

    def __init__(self, dtype=None, dims=[], ptr=None):
        if ptr:
            ffi_obj = ptr
        elif dtype is not None:
            raw_dims = ffi.new('int64_t[]', [0 if x is None else x for x in dims])
            ffi_obj = ffi_call(lib.plaidml_logical_shape_alloc, dtype, len(dims), raw_dims)
        else:
            raise ValueError('One of dtype= or ptr= must be specified.')
        super(LogicalShape, self).__init__(ffi_obj)

    @property
    def dtype(self):
        return DType(ffi_call(lib.plaidml_logical_shape_get_dtype, self.as_ptr()))

    @property
    def ndims(self):
        return ffi_call(lib.plaidml_logical_shape_get_ndims, self.as_ptr())

    @property
    def int_dims(self):
        return [
            ffi_call(lib.plaidml_logical_shape_get_dim_int, self.as_ptr(), i)
            for i in range(self.ndims)
        ]

    # @property
    # def dims(self):
    #     return [
    #         TensorDim(expr=ffi_call(lib.plaidml_logical_shape_get_dim_expr, self.as_ptr(), i))
    #         for i in range(self.ndims)
    #     ]

    def into_TensorShape(self):
        return TensorShape(
            ptr=ffi_call(lib.plaidml_logical_shape_into_tensor_shape, self.as_ptr()))


Constraint = namedtuple('Constraint', ['lhs', 'rhs'])


def wrap_dim(x):
    if isinstance(x, six.integer_types):
        return TensorDim(expr=ffi_call(lib.plaidml_dim_expr_int, x))
    return x


def dim_op(op, *args):
    args = [wrap_dim(x) for x in args]
    raw_args = [x.as_ptr() for x in args]
    return ffi_call(lib.plaidml_dim_expr_op, op, len(args), raw_args)


class TensorDim(ForeignObject):
    __ffi_del__ = lib.plaidml_dim_expr_free
    __ffi_repr__ = lib.plaidml_dim_expr_repr

    def __init__(self, expr=None):
        if expr is None:
            expr = ffi_call(lib.plaidml_dim_expr_none)
        super(TensorDim, self).__init__(expr)

    def _bind(self, expr):
        self.take_ptr(expr)

    def __neg__(self):
        return TensorDim(dim_op(lib.PLAIDML_INT_OP_NEG, self))

    def __add__(self, other):
        return TensorDim(dim_op(lib.PLAIDML_INT_OP_ADD, self, other))

    def __radd__(self, other):
        return TensorDim(dim_op(lib.PLAIDML_INT_OP_ADD, other, self))

    def __sub__(self, other):
        return TensorDim(dim_op(lib.PLAIDML_INT_OP_SUB, self, other))

    def __rsub__(self, other):
        return TensorDim(dim_op(lib.PLAIDML_INT_OP_SUB, other, self))

    def __mul__(self, other):
        return TensorDim(dim_op(lib.PLAIDML_INT_OP_MUL, self, other))

    def __rmul__(self, other):
        return TensorDim(dim_op(lib.PLAIDML_INT_OP_MUL, other, self))

    def __floordiv__(self, other):
        return TensorDim(dim_op(lib.PLAIDML_INT_OP_DIV, self, other))

    def __rfloordiv__(self, other):
        return TensorDim(dim_op(lib.PLAIDML_INT_OP_DIV, other, self))


def wrap_poly(x):
    if isinstance(x, six.integer_types):
        return TensorIndex(expr=ffi_call(lib.plaidml_poly_expr_literal, x))
    if isinstance(x, TensorDim):
        return TensorIndex(expr=ffi_call(lib.plaidml_poly_expr_dim, x.as_ptr()))
    return x


def poly_op(op, *args):
    args = [wrap_poly(x) for x in args]
    raw_args = [x.as_ptr() for x in args]
    return ffi_call(lib.plaidml_poly_expr_op, op, len(args), raw_args)


class TensorIndex(ForeignObject):
    __ffi_del__ = lib.plaidml_poly_expr_free
    __ffi_repr__ = lib.plaidml_poly_expr_repr

    def __init__(self, expr=None, name=''):
        if expr is None:
            expr = ffi_call(lib.plaidml_poly_expr_index, name.encode())
        super(TensorIndex, self).__init__(expr)

    def __lt__(self, rhs):
        return Constraint(self, wrap_dim(rhs))

    def __neg__(self):
        return TensorIndex(poly_op(lib.PLAIDML_INT_OP_NEG, self))

    def __add__(self, rhs):
        return TensorIndex(poly_op(lib.PLAIDML_INT_OP_ADD, self, rhs))

    def __radd__(self, lhs):
        return TensorIndex(poly_op(lib.PLAIDML_INT_OP_ADD, lhs, self))

    def __sub__(self, rhs):
        return TensorIndex(poly_op(lib.PLAIDML_INT_OP_SUB, self, rhs))

    def __rsub__(self, lhs):
        return TensorIndex(poly_op(lib.PLAIDML_INT_OP_SUB, lhs, self))

    def __mul__(self, rhs):
        return TensorIndex(poly_op(lib.PLAIDML_INT_OP_MUL, self, rhs))

    def __rmul__(self, lhs):
        return TensorIndex(poly_op(lib.PLAIDML_INT_OP_MUL, lhs, self))

    def __floordiv__(self, rhs):
        return TensorIndex(poly_op(lib.PLAIDML_INT_OP_DIV, self, rhs))

    def __rfloordiv__(self, lhs):
        return TensorIndex(poly_op(lib.PLAIDML_INT_OP_DIV, lhs, self))


class _IndexMap(ForeignObject):
    __ffi_del__ = lib.plaidml_expr_free
    __ffi_repr__ = lib.plaidml_expr_repr

    def __init__(self, ref, key):
        if isinstance(key, tuple) or isinstance(key, list):
            idxs = key
        else:
            idxs = [key]
        idxs = [wrap_poly(x) for x in idxs]
        raw_idxs = [x.as_ptr() for x in idxs]
        expr = ffi_call(lib.plaidml_expr_index_map, ref.as_ptr(), len(idxs), raw_idxs)
        super(_IndexMap, self).__init__(expr)


class _SizeMap(ForeignObject):
    __ffi_del__ = lib.plaidml_expr_free
    __ffi_repr__ = lib.plaidml_expr_repr

    def __init__(self, dims):
        dims = [wrap_dim(x) for x in dims]
        raw_dims = [x.as_ptr() for x in dims]
        expr = ffi_call(lib.plaidml_expr_size_map, len(dims), raw_dims)
        super(_SizeMap, self).__init__(expr)


class _Contraction(ForeignObject):
    __ffi_del__ = lib.plaidml_expr_free
    __ffi_repr__ = lib.plaidml_expr_repr

    def __init__(self, agg_op, combo_op, src_idxs, sink_idxs, sink_sizes, name):
        src_idxs = [x.as_ptr() for x in src_idxs]
        expr = ffi_call(
            lib.plaidml_expr_contraction,
            agg_op,
            combo_op,
            sink_idxs.as_ptr(),
            sink_sizes.as_ptr(),
            len(src_idxs),
            src_idxs,
            name.encode(),
        )
        super(_Contraction, self).__init__(expr)


_ContractionPart = namedtuple('_ContractionPart', ['op', 'args'])


class IndexedTensor(object):

    def __init__(self, impl, tensor=None):
        self._impl = impl
        self._tensor = tensor

    def __repr__(self):
        return repr(self._impl)

    # Represents an aggregation_op of SUM in a contraction
    def __iadd__(self, rhs):
        return IndexedTensor(self._make_contraction(lib.PLAIDML_AGG_OP_SUM, rhs))

    # Represents an aggregation_op of PROD in a contraction
    def __imul__(self, rhs):
        return IndexedTensor(self._make_contraction(lib.PLAIDML_AGG_OP_PROD, rhs))

    # Represents an aggregation_op of MAX in a contraction
    def __ge__(self, rhs):
        self._tensor._set_contraction(self._make_contraction(lib.PLAIDML_AGG_OP_MAX, rhs))

    # Represents an aggregation_op of MIN in a contraction
    def __le__(self, rhs):
        self._tensor._set_contraction(self._make_contraction(lib.PLAIDML_AGG_OP_MIN, rhs))

    # Represents a combo_op of PLUS in a contraction
    def __add__(self, rhs):
        return IndexedTensor(_ContractionPart(lib.PLAIDML_COMBO_OP_ADD, (self, rhs)))

    # Represents a combo_op of MULTIPLY in a contraction
    def __mul__(self, rhs):
        return IndexedTensor(_ContractionPart(lib.PLAIDML_COMBO_OP_MUL, (self, rhs)))

    # Represents a combo_op of EQ in a contraction
    def __eq__(self, rhs):
        return IndexedTensor(_ContractionPart(lib.PLAIDML_COMBO_OP_EQ, (self, rhs)))

    def _make_contraction(self, agg_op, rhs):
        # Extract combo_op and inputs
        if isinstance(rhs._impl, _IndexMap):
            # Unary op
            combo_op = lib.PLAIDML_COMBO_OP_NONE
            inputs = [rhs._impl]
        elif isinstance(rhs._impl, _ContractionPart):
            # Binary/Ternary op
            combo_op = rhs._impl.op
            inputs = [x._impl for x in rhs._impl.args]
        else:
            raise ValueError('Invalid impl')
        return _Contraction(
            agg_op,
            combo_op,
            inputs,
            self._impl,
            _SizeMap(self._tensor._dims),
            self._tensor._name,
        )


class Tensor(ForeignObject):
    __ffi_del__ = lib.plaidml_expr_free
    __ffi_repr__ = lib.plaidml_expr_repr

    _dims = None
    _is_contraction = False

    def __init__(self, shape=None, dims=None, expr=None, value=None, name='', buffer=None):
        self._name = name
        if shape:
            if buffer is None:
                raw_buffer = ffi.NULL
            else:
                raw_buffer = buffer.as_ptr()
            expr = ffi_call(lib.plaidml_expr_placeholder, shape.as_ptr(), raw_buffer,
                            name.encode())
        elif dims is not None:
            self._dims = dims
            expr = None
        elif value is not None:
            if isinstance(value, six.integer_types):
                expr = ffi_call(lib.plaidml_expr_int, value)
            elif isinstance(value, float):
                expr = ffi_call(lib.plaidml_expr_float, value)
            else:
                raise TypeError('Invalid type for value={}'.format(value))
        elif expr is None:
            raise ValueError('One of dims=, shape=, or expr= must be specified.')
        super(Tensor, self).__init__(expr)

    def set_param_value(self, buffer):
        # Changes the value of a parameter tensor (i.e. one explicitly set to a buffer value)
        # Illegal on other tensors
        ffi_call(lib.plaidml_expr_param_reset, self.__ffi_obj__, buffer.as_ptr())

    def __hash__(self):
        return hash((self.as_ptr(), self._dims, self._is_contraction))

    def __getitem__(self, key):
        return IndexedTensor(_IndexMap(self, key), tensor=self)

    def __setitem__(self, key, value):
        if isinstance(value._impl, _Contraction):
            # standard contraction
            self._set_contraction(value._impl)
        elif isinstance(value, Tensor):
            pass
        elif isinstance(value._impl, _IndexMap):
            # Unary ASSIGN contraction
            self._set_contraction(
                _Contraction(
                    lib.PLAIDML_AGG_OP_ASSIGN,
                    lib.PLAIDML_COMBO_OP_NONE,
                    [value._impl],
                    _IndexMap(self, key),
                    _SizeMap(self._dims),
                    self._name,
                ))
        elif isinstance(value._impl, _ContractionPart):
            # Binary or ternary ASSIGN contraction
            self._set_contraction(
                _Contraction(
                    lib.PLAIDML_AGG_OP_ASSIGN,
                    value._impl.op,
                    [x._impl for x in value._impl.args],
                    _IndexMap(self, key),
                    _SizeMap(self._dims),
                    self._name,
                ))
        else:
            raise ValueError('Invalid impl when assigning to a Tensor (Type: {})'.format(
                type(value._impl)))

    def _set_contraction(self, cion):
        self._is_contraction = True
        self.take_ptr(cion)

    # Represents an eltwise negation
    def __neg__(self):
        return call('neg', self)

    # Represents an eltwise bit_not
    def __invert__(self):
        return call('bit_not', self)

    # Represents an eltwise addition
    def __add__(self, rhs):
        return call('add', self, rhs)

    def __radd__(self, lhs):
        return call('add', lhs, self)

    # Represents an eltwise subtraction
    def __sub__(self, rhs):
        return call('sub', self, rhs)

    def __rsub__(self, lhs):
        return call('sub', lhs, self)

    # Represents an eltwise multiplication
    def __mul__(self, rhs):
        return call('mul', self, rhs)

    def __rmul__(self, lhs):
        return call('mul', lhs, self)

    # Represents an eltwise division
    def __div__(self, rhs):
        return call('div', self, rhs)

    def __rdiv__(self, lhs):
        return call('div', lhs, self)

    # Represents an eltwise division
    def __truediv__(self, rhs):
        return call('div', self, rhs)

    def __rtruediv__(self, lhs):
        return call('div', lhs, self)

    # Represents an eltwise cmp_eq
    def __eq__(self, rhs):
        return call('cmp_eq', self, rhs)

    # Represents an eltwise cmp_ne
    def __ne__(self, rhs):
        return call('cmp_ne', self, rhs)

    # Represents an eltwise cmp_lt
    def __lt__(self, rhs):
        return call('cmp_lt', self, rhs)

    # Represents an eltwise cmp_gt
    def __gt__(self, rhs):
        return call('cmp_gt', self, rhs)

    # Represents an eltwise cmp_le
    def __le__(self, rhs):
        return call('cmp_le', self, rhs)

    # Represents an eltwise cmp_ge
    def __ge__(self, rhs):
        return call('cmp_ge', self, rhs)

    # Represents an eltwise bit_left
    def __lshift__(self, rhs):
        return call('bit_left', self, rhs)

    def __rlshift__(self, lhs):
        return call('bit_left', lhs, self)

    # Represents an eltwise bit_right
    def __rshift__(self, rhs):
        return call('bit_right', self, rhs)

    def __rrshift__(self, lhs):
        return call('bit_right', lhs, self)

    # Represents an eltwise bit_and
    def __and__(self, rhs):
        return call('bit_and', self, rhs)

    def __rand__(self, lhs):
        return call('bit_and', lhs, self)

    # Represents an eltwise bit_or
    def __or__(self, rhs):
        return call('bit_or', self, rhs)

    def __ror__(self, lhs):
        return call('bit_or', lhs, self)

    # Represents an eltwise bit_xor
    def __xor__(self, rhs):
        return call('bit_xor', self, rhs)

    def __rxor__(self, lhs):
        return call('bit_xor', lhs, self)

    # Enable no_reduce on a contraction
    def no_reduce(self):
        if not self._is_contraction:
            raise TypeError('no_reduce can only be specified on a contraction.')
        ffi_call(lib.plaidml_expr_contraction_set_no_reduce, self.as_ptr(), True)
        return self

    # Set use_default on a contraction
    def use_default(self, rhs):
        if not self._is_contraction:
            raise TypeError('use_default can only be specified on a contraction.')
        ffi_call(lib.plaidml_expr_contraction_set_use_default, self.as_ptr(), rhs.as_ptr())
        return self

    def add_constraint(self, constraint):
        ffi_call(
            lib.plaidml_expr_contraction_add_constraint,
            self.as_ptr(),
            constraint.lhs.as_ptr(),
            constraint.rhs.as_ptr(),
        )

    def add_constraints(self, constraints):
        for constraint in constraints:
            self.add_constraint(constraint)

    # Return the tensor's shape
    @property
    def shape(self):
        return LogicalShape(ptr=ffi_call(lib.plaidml_expr_get_shape, self.as_ptr()))

    # Verify that the specified dims match the dims of this tensor.
    def bind_dims(self, *dims):
        raw_dims = [x.as_ptr() for x in dims]
        ffi_call(lib.plaidml_expr_bind_dims, self.as_ptr(), len(raw_dims), raw_dims)

    # bind a concrete shape to this tensor
    def bind(self, shape):
        ffi_call(lib.plaidml_expr_bind_shape, self.as_ptr(), shape.as_ptr())


class Value(ForeignObject):
    __ffi_del__ = lib.plaidml_expr_free
    __ffi_repr__ = lib.plaidml_expr_repr

    def __init__(self, value):
        # logger.debug('Value({})'.format(value))
        if isinstance(value, np.ndarray):
            if value.ndim == 0:
                value = value.item()
            else:
                value = value.tolist()
        if value is None:
            ffi_obj = ffi_call(lib.plaidml_expr_none)
        elif isinstance(value, (six.integer_types, bool)):
            ffi_obj = ffi_call(lib.plaidml_expr_int, value)
        elif isinstance(value, float):
            ffi_obj = ffi_call(lib.plaidml_expr_float, value)
        elif isinstance(value, TensorDim):
            ffi_obj = ffi_call(lib.plaidml_expr_dim, value.as_ptr())
        elif isinstance(value, Tensor):
            ffi_obj = ffi_call(lib.plaidml_expr_clone, value.as_ptr())
        elif isinstance(value, (list, tuple)):
            self._elts = [Value(x) for x in value]
            raw_elts = [x.as_ptr() for x in self._elts]
            ffi_obj = ffi_call(lib.plaidml_expr_tuple, len(raw_elts), raw_elts)
        elif isinstance(value, six.string_types):
            ffi_obj = ffi_call(lib.plaidml_expr_str, value.encode('utf-8'))
        elif isinstance(value, ffi.CData) and ffi.typeof(value) is ffi.typeof('plaidml_expr*'):
            ffi_obj = value
        else:
            raise TypeError('Unsupported type {} for value={}'.format(type(value), value))
        super(Value, self).__init__(ffi_obj)

    def as_tensor(self):
        return Tensor(expr=ffi_call(lib.plaidml_expr_clone, self.as_ptr()))


def TensorOutput(*args):
    return Tensor(dims=args)


def TensorDims(count):
    return [TensorDim() for i in range(count)]


def TensorIndexes(count):
    return [TensorIndex() for i in range(count)]


class Program(ForeignObject):
    __ffi_del__ = lib.plaidml_program_free
    __ffi_repr__ = lib.plaidml_program_repr

    def __init__(self, name, outputs, updates=[]):
        raw_outputs = [x.as_ptr() for x in outputs]
        new_outputs = ffi.new('plaidml_expr*[]', len(outputs))
        dst_updates = [x[0].as_ptr() for x in updates]
        src_updates = [x[1].as_ptr() for x in updates]
        ffi_obj = ffi_call(
            lib.plaidml_program_evaluate,
            name.encode(),
            len(raw_outputs),
            raw_outputs,
            new_outputs,
            len(updates),
            src_updates,
            dst_updates,
        )
        self.outputs = [Tensor(expr=x) for x in new_outputs]
        super(Program, self).__init__(ffi_obj)


def wrap_tensor(x):
    if isinstance(x, six.integer_types):
        return Tensor(expr=ffi_call(lib.plaidml_expr_int, x))
    if np.issubdtype(type(x), np.integer):
        return Tensor(expr=ffi_call(lib.plaidml_expr_int, x.item()))
    if isinstance(x, float):
        return Tensor(expr=ffi_call(lib.plaidml_expr_float, x))
    if isinstance(x, TensorDim):
        return Tensor(expr=ffi_call(lib.plaidml_expr_dim, x.as_ptr()))
    if isinstance(x, Tensor):
        return x
    raise TypeError('Unexpected type for call argument: {}. fn: {}, args: {}, bad arg: {}'.format(
        type(x), fn, args, x))


def call(fn, *args):
    args = [wrap_tensor(x) for x in args]
    raw_args = [x.as_ptr() for x in args]
    return Tensor(expr=ffi_call(lib.plaidml_expr_call, fn.encode(), len(args), raw_args))


def cast(x, dtype):
    return Tensor(expr=ffi_call(lib.plaidml_expr_cast, wrap_tensor(x).as_ptr(), dtype))


def as_bool(x):
    return cast(x, DType.BOOLEAN)


def as_float(x, bit_size):
    map = {
        16: DType.FLOAT16,
        32: DType.FLOAT32,
        64: DType.FLOAT64,
    }
    dtype = map.get(bit_size)
    if not dtype:
        raise 'Unsupport bit_size for as_float'
    return cast(x, dtype)


def as_int(x, bit_size):
    map = {
        8: DType.INT8,
        16: DType.INT16,
        32: DType.INT32,
        64: DType.INT64,
    }
    dtype = map.get(bit_size)
    if not dtype:
        raise 'Unsupport bit_size for as_int'
    return cast(x, dtype)


def as_uint(x, bit_size):
    map = {
        8: DType.UINT8,
        16: DType.UINT16,
        32: DType.UINT32,
        64: DType.UINT64,
    }
    dtype = map.get(bit_size)
    if not dtype:
        raise 'Unsupport bit_size for as_uint'
    return cast(x, dtype)


def ceil(x):
    return call('ceil', x)


def cond(lhs, rhs, true_case):
    return IndexedTensor(_ContractionPart(lib.PLAIDML_COMBO_OP_COND, (lhs, rhs, true_case)))


def cos(x):
    return call('cos', x)


def exp(x):
    return call('exp', x)


def floor(x):
    return call('floor', x)


def gather(x, y):
    return call('gather', x, y)


def gradients(loss, variables):
    wrts = [x.as_ptr() for x in variables]
    raw_grads = ffi.new('plaidml_expr*[]', len(wrts))
    ffi_call(
        lib.plaidml_expr_gradient,
        len(wrts),
        wrts,
        loss.as_ptr(),
        raw_grads,
    )
    return [Tensor(expr=x) for x in raw_grads]


def ident(x):
    return call('ident', x)


def index(x, axis):
    return call('index', x, axis)


def log(x):
    return call('log', x)


def max(x, y):
    return call('max', x, y)


def min(x, y):
    return call('min', x, y)


def pow(x, y):
    return call('pow', x, y)


def prng(state, shape):
    return call('prng', state, *shape)


def reshape(x, dims):
    return call('reshape', x, *dims)


def round(x):
    return call('round', x)


def scatter(x, y, z):
    return call('scatter', x, y, z)


def select(cond, true_case, false_case):
    return call('cond', cond, true_case, false_case)


def shape(x):
    return call('shape', x)


def sin(x):
    return call('sin', x)


def sqrt(x):
    return call('sqrt', x)


def tan(x):
    return call('tan', x)


def tanh(x):
    return call('tanh', x)
