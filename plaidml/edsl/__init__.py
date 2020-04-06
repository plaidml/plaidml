# Copyright 2020 Intel Corporation

import enum
import logging
from collections import namedtuple

import numpy as np
import six

import plaidml.settings
from plaidml import DType
from plaidml.core import Buffer, TensorShape
from plaidml.ffi import ForeignObject, ffi, ffi_call, lib

logger = logging.getLogger(__name__)


def __init():
    """Initializes the PlaidML EDSL API."""
    ffi_call(lib.plaidml_edsl_init)


ffi.init_once(__init, 'plaidml_edsl_init')


class LogicalShape(ForeignObject):
    """Represents the logical shape of a Tensor.

    Args:
        dtype (:py:class:`~plaidml.core.DType`): The element DType.
        dims (:obj:`list` of :obj:`int`, optional): The dimensions for this
            LogicalShape.
    """

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
        """:py:class:`~plaidml.core.DType`: Returns the element DType of this
        LogicalShape.
        """
        return DType(self._methodcall(lib.plaidml_logical_shape_get_dtype))

    @property
    def rank(self):
        """:obj:`int`: Returns the rank (i.e. number of dimensions) of this
        LogicalShape.
        """
        return self._methodcall(lib.plaidml_logical_shape_get_rank)

    @property
    def sizes(self):
        """:obj:`list` of :obj:`int`: Returns the sizes of this LogicalShape."""
        return get_integers(lib.plaidml_logical_shape_get_sizes, self.as_ptr())

    def into_TensorShape(self):
        """Converts a ``LogicalShape`` into a ``TensorShape``.

        Returns:
            :py:class:`~plaidml.core.TensorShape`: The resultant TensorShape.
        """
        return TensorShape(ptr=self._methodcall(lib.plaidml_logical_shape_into_tensor_shape))


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
    """Represents a symbolic dimension size in a polynomial expression."""

    __ffi_del__ = lib.plaidml_dim_expr_free
    __ffi_repr__ = lib.plaidml_dim_expr_repr

    def __init__(self, expr=None):
        """TensorDim constructor."""
        if expr is None:
            expr = ffi_call(lib.plaidml_dim_expr_none)
        super(TensorDim, self).__init__(expr)

    def _bind(self, expr):
        self.take_ptr(expr)

    def __neg__(self):
        """Negates a TensorDim in a polynomial expression.

        Example:
            >>> N, M = TensorDims(2)
            >>> A = Placeholder(DType.FLOAT32, [3, 3])
            >>> A.bind_dims(N, M)
            >>> R = TensorOutput(-N)
        """
        return TensorDim(dim_op(lib.PLAIDML_INT_OP_NEG, self))

    def __add__(self, other):
        """Performs an addition between a TensorDim and another operand in a
        polynomial expression.

        Example:
            >>> N, M = TensorDims(2)
            >>> A = Placeholder(DType.FLOAT32, [3, 3])
            >>> A.bind_dims(N, M)
            >>> R = TensorOutput(N + 5)
        """
        return TensorDim(dim_op(lib.PLAIDML_INT_OP_ADD, self, other))

    def __radd__(self, other):
        """Performs an addition between a TensorDim and another operand in a
        polynomial expression.

        Example:
            >>> N, M = TensorDims(2)
            >>> A = Placeholder(DType.FLOAT32, [3, 3])
            >>> A.bind_dims(N, M)
            >>> R = TensorOutput(5 + N)
        """
        return TensorDim(dim_op(lib.PLAIDML_INT_OP_ADD, other, self))

    def __sub__(self, other):
        """Performs a subtraction between a TensorDim and another operand in a
        polynomial expression.

        Example:
            >>> N, M = TensorDims(2)
            >>> A = Placeholder(DType.FLOAT32, [3, 3])
            >>> A.bind_dims(N, M)
            >>> R = TensorOutput(N - 5)
        """
        return TensorDim(dim_op(lib.PLAIDML_INT_OP_SUB, self, other))

    def __rsub__(self, other):
        """Performs a subtraction between a TensorDim and another operand in a
        polynomial expression.

        Example:
            >>> N, M = TensorDims(2)
            >>> A = Placeholder(DType.FLOAT32, [3, 3])
            >>> A.bind_dims(N, M)
            >>> R = TensorOutput(5 - N)
        """
        return TensorDim(dim_op(lib.PLAIDML_INT_OP_SUB, other, self))

    def __mul__(self, other):
        """Performs a multiplication between a TensorDim and another operand in
        a polynomial expression.

        Example:
            >>> N, M = TensorDims(2)
            >>> A = Placeholder(DType.FLOAT32, [3, 3])
            >>> A.bind_dims(N, M)
            >>> R = TensorOutput(N * 5)
        """
        return TensorDim(dim_op(lib.PLAIDML_INT_OP_MUL, self, other))

    def __rmul__(self, other):
        """Performs a multiplication between a TensorDim and another operand in
        a polynomial expression.

        Example:
            >>> N, M = TensorDims(2)
            >>> A = Placeholder(DType.FLOAT32, [3, 3])
            >>> A.bind_dims(N, M)
            >>> R = TensorOutput(5 * N)
        """
        return TensorDim(dim_op(lib.PLAIDML_INT_OP_MUL, other, self))

    def __floordiv__(self, other):
        """Performs a floor division between a TensorDim and another operand in
        a polynomial expression.

        Example:
            >>> N, M = TensorDims(2)
            >>> A = Placeholder(DType.FLOAT32, [3, 3])
            >>> A.bind_dims(N, M)
            >>> R = TensorOutput(N // 5)
        """
        return TensorDim(dim_op(lib.PLAIDML_INT_OP_DIV, self, other))

    def __rfloordiv__(self, other):
        """Performs a floor division between a TensorDim and another operand in
        a polynomial expression.

        Example:
            >>> N, M = TensorDims(2)
            >>> A = Placeholder(DType.FLOAT32, [3, 3])
            >>> A.bind_dims(N, M)
            >>> R = TensorOutput(5 // N)
        """
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
    """Represents an index in a polynomial expression.
    
    Args:
        name (str, optional): The name to give this TensorIndex.
    """

    __ffi_del__ = lib.plaidml_poly_expr_free
    __ffi_repr__ = lib.plaidml_poly_expr_repr

    def __init__(self, expr=None, name=''):
        """TensorIndex constructor."""
        if expr is None:
            expr = ffi_call(lib.plaidml_poly_expr_index, name.encode())
        super(TensorIndex, self).__init__(expr)

    def __lt__(self, rhs):
        """Represents a constraint that can be applied to a contraction.

        Example:
            >>> i, j = TensorIndexes(2)
            >>> A = Placeholder(DType.FLOAT32, [3, 3])
            >>> R = TensorOutput()
            >>> R[()] += A[i, j]
            >>> R.add_constraint(i < 5)
        """
        return Constraint(self, wrap_dim(rhs))

    def __neg__(self):
        """Negates a TensorIndex in a polynomial expression.

        Example:
            >>> i, j = TensorIndexes(2)
            >>> A = Placeholder(DType.FLOAT32, [3, 3])
            >>> R = TensorOutput()
            >>> R[()] += A[-i, j]
        """
        return TensorIndex(poly_op(lib.PLAIDML_INT_OP_NEG, self))

    def __add__(self, rhs):
        """Performs an addition between a TensorIndex and another operand in a
        polynomial expression.

        Example:
            >>> i, j = TensorIndexes(2)
            >>> A = Placeholder(DType.FLOAT32, [3, 3])
            >>> R = TensorOutput()
            >>> R[()] += A[i + 5, j]
        """
        return TensorIndex(poly_op(lib.PLAIDML_INT_OP_ADD, self, rhs))

    def __radd__(self, lhs):
        """Performs an addition between a TensorIndex and another operand in a
        polynomial expression.

        Example:
            >>> i, j = TensorIndexes(2)
            >>> A = Placeholder(DType.FLOAT32, [3, 3])
            >>> R = TensorOutput()
            >>> R[()] += A[5 + i, j]
        """
        return TensorIndex(poly_op(lib.PLAIDML_INT_OP_ADD, lhs, self))

    def __sub__(self, rhs):
        """Performs a subtraction between a TensorIndex and another operand in a
        polynomial expression.

        Example:
            >>> i, j = TensorIndexes(2)
            >>> A = Placeholder(DType.FLOAT32, [3, 3])
            >>> R = TensorOutput()
            >>> R[()] += A[i - 5, j]
        """
        return TensorIndex(poly_op(lib.PLAIDML_INT_OP_SUB, self, rhs))

    def __rsub__(self, lhs):
        """Performs a subtraction between a TensorIndex and another operand in a
        polynomial expression.

        Example:
            >>> i, j = TensorIndexes(2)
            >>> A = Placeholder(DType.FLOAT32, [3, 3])
            >>> R = TensorOutput()
            >>> R[()] += A[5 - i, j]
        """
        return TensorIndex(poly_op(lib.PLAIDML_INT_OP_SUB, lhs, self))

    def __mul__(self, rhs):
        """Performs a multiplication between a TensorIndex and another operand
        in a polynomial expression.

        Example:
            >>> i, j = TensorIndexes(2)
            >>> A = Placeholder(DType.FLOAT32, [3, 3])
            >>> R = TensorOutput()
            >>> R[()] += A[i * 5, j]
        """
        return TensorIndex(poly_op(lib.PLAIDML_INT_OP_MUL, self, rhs))

    def __rmul__(self, lhs):
        """Performs a multiplication between a TensorIndex and another operand
        in a polynomial expression.

        Example:
            >>> i, j = TensorIndexes(2)
            >>> A = Placeholder(DType.FLOAT32, [3, 3])
            >>> R = TensorOutput()
            >>> R[()] += A[5 * i, j]
        """
        return TensorIndex(poly_op(lib.PLAIDML_INT_OP_MUL, lhs, self))

    def __floordiv__(self, rhs):
        """Performs a floor division between a TensorIndex and another operand
        in a polynomial expression.

        Example:
            >>> i, j = TensorIndexes(2)
            >>> A = Placeholder(DType.FLOAT32, [3, 3])
            >>> R = TensorOutput()
            >>> R[()] += A[i // 5, j]
        """
        return TensorIndex(poly_op(lib.PLAIDML_INT_OP_DIV, self, rhs))

    def __rfloordiv__(self, lhs):
        """Performs a floor division between a TensorIndex and another operand
        in a polynomial expression.

        Example:
            >>> i, j = TensorIndexes(2)
            >>> A = Placeholder(DType.FLOAT32, [3, 3])
            >>> R = TensorOutput()
            >>> R[()] += A[5 // i, j]
        """
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

    def __iadd__(self, rhs):
        """Represents a `summation` reduction within a contraction.

        Example:
            >>> i, j = TensorIndexes(2)
            >>> A = Placeholder(DType.FLOAT32, [3, 3])
            >>> R = TensorOutput()
            >>> R[()] += A[i, j]
        """
        return IndexedTensor(self._make_contraction(lib.PLAIDML_AGG_OP_SUM, rhs))

    def __imul__(self, rhs):
        """Represents a `product` reduction within a contraction.

        Example:
            >>> i, j = TensorIndexes(2)
            >>> A = Placeholder(DType.FLOAT32, [3, 3])
            >>> R = TensorOutput()
            >>> R[()] *= A[i, j]
        """
        return IndexedTensor(self._make_contraction(lib.PLAIDML_AGG_OP_PROD, rhs))

    def __ge__(self, rhs):
        """Represents a `maximum` reduction within a contraction.

        Example:
            >>> i, j = TensorIndexes(2)
            >>> A = Placeholder(DType.FLOAT32, [3, 3])
            >>> R = TensorOutput()
            >>> R[()] >= A[i, j]
        """
        self._tensor._set_contraction(self._make_contraction(lib.PLAIDML_AGG_OP_MAX, rhs))

    def __le__(self, rhs):
        """Represents a `minimum` reduction within a contraction.

        Example:
            >>> i, j = TensorIndexes(2)
            >>> A = Placeholder(DType.FLOAT32, [3, 3])
            >>> R = TensorOutput()
            >>> R[()] <= A[i, j]
        """
        self._tensor._set_contraction(self._make_contraction(lib.PLAIDML_AGG_OP_MIN, rhs))

    def __add__(self, rhs):
        """Represents an `addition` combination within a contraction.

        Example:
            >>> i, j, k = TensorIndexes(3)
            >>> A = Placeholder(DType.FLOAT32, [3, 3])
            >>> B = Placeholder(DType.FLOAT32, [3, 3])
            >>> A[i, j] + B[j, k]
        """
        return IndexedTensor(_ContractionPart(lib.PLAIDML_COMBO_OP_ADD, (self, rhs)))

    def __mul__(self, rhs):
        """Represents a `multiply` combination within in a contraction.

        Example:
            >>> i, j, k = TensorIndexes(3)
            >>> A = Placeholder(DType.FLOAT32, [3, 3])
            >>> B = Placeholder(DType.FLOAT32, [3, 3])
            >>> A[i, j] * B[j, k]
        """
        return IndexedTensor(_ContractionPart(lib.PLAIDML_COMBO_OP_MUL, (self, rhs)))

    def __eq__(self, rhs):
        """Represents an `equality comparison` combination within a contraction.

        Example:
            >>> i, j, k = TensorIndexes(3)
            >>> A = Placeholder(DType.FLOAT32, [3, 3])
            >>> A[i, j] == B[j, k]
        """
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
    """Represents a multi-dimensional result of an Operation."""

    __ffi_del__ = lib.plaidml_expr_free
    __ffi_repr__ = lib.plaidml_expr_repr

    _dims = None
    _is_contraction = False

    def __init__(self, shape=None, dims=None, expr=None, value=None, name='', buffer=None):
        self._name = name
        self._buffer = buffer
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

    # Represents an eltwise bit_shl
    def __lshift__(self, rhs):
        return call('bit_shl', self, rhs)

    def __rlshift__(self, lhs):
        return call('bit_shl', lhs, self)

    # Represents an eltwise bit_shr
    def __rshift__(self, rhs):
        return call('bit_shr', self, rhs)

    def __rrshift__(self, lhs):
        return call('bit_shr', lhs, self)

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
        self._methodcall(lib.plaidml_expr_contraction_set_no_reduce, True)
        return self

    # Set use_default on a contraction
    def use_default(self, rhs):
        if not self._is_contraction:
            raise TypeError('use_default can only be specified on a contraction.')
        self._methodcall(lib.plaidml_expr_contraction_set_use_default, rhs.as_ptr())
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
    def compute_shape(self):
        return LogicalShape(ptr=self._methodcall(lib.plaidml_expr_get_shape))

    @property
    def dtype(self):
        return DType(self._methodcall(lib.plaidml_expr_get_dtype))

    @property
    def rank(self):
        return self._methodcall(lib.plaidml_expr_get_rank)

    # Verify that the specified dims match the dims of this tensor.
    def bind_dims(self, *dims):
        raw_dims = [x.as_ptr() for x in dims]
        self._methodcall(lib.plaidml_expr_bind_dims, len(raw_dims), raw_dims)

    # bind a concrete shape to this tensor
    def bind(self, shape):
        self._methodcall(lib.plaidml_expr_bind_shape, shape.as_ptr())


class TensorRef:

    def __init__(self, tensor):
        self.tensor = tensor

    def __hash__(self):
        return hash(ffi_call(lib.plaidml_expr_ptr, self.tensor.as_ptr()))

    def __eq__(self, other):
        if isinstance(other, Tensor):
            return self.__hash__() == TensorRef(other).__hash__()
        return self.__hash__() == other.__hash__()


class Value(ForeignObject):
    __ffi_del__ = lib.plaidml_value_free
    __ffi_repr__ = lib.plaidml_value_repr

    def __init__(self, value):
        logger.debug('Value({})'.format(value))
        if isinstance(value, np.ndarray):
            if value.ndim == 0:
                value = value.item()
            else:
                value = value.tolist()
        if value is None:
            ffi_obj = ffi_call(lib.plaidml_value_none)
        elif isinstance(value, enum.IntEnum):
            print('enum:', value, value.value)
            ffi_obj = ffi_call(lib.plaidml_value_int, value.value)
        elif isinstance(value, (six.integer_types, bool)):
            ffi_obj = ffi_call(lib.plaidml_value_int, value)
        elif isinstance(value, float):
            ffi_obj = ffi_call(lib.plaidml_value_float, value)
        elif isinstance(value, TensorDim):
            ffi_obj = ffi_call(lib.plaidml_value_dim, value.as_ptr())
        elif isinstance(value, Tensor):
            ffi_obj = ffi_call(lib.plaidml_value_expr, value.as_ptr())
        elif isinstance(value, (list, tuple)):
            self._elts = [Value(x) for x in value]
            raw_elts = [x.as_ptr() for x in self._elts]
            ffi_obj = ffi_call(lib.plaidml_value_tuple, len(raw_elts), raw_elts)
        elif isinstance(value, six.string_types):
            ffi_obj = ffi_call(lib.plaidml_value_str, value.encode('utf-8'))
        elif isinstance(value, ffi.CData) and ffi.typeof(value) is ffi.typeof('plaidml_value*'):
            ffi_obj = value
        else:
            raise TypeError('Unsupported type {} for value={}'.format(type(value), value))
        super(Value, self).__init__(ffi_obj)

    def as_tensor(self):
        return Tensor(expr=self._methodcall(lib.plaidml_value_expr_get))


def TensorOutput(*args):
    """Declares a ``Tensor`` and specifies its output shape.

    This must be used before building a contraction.
    """
    return Tensor(dims=args)


def TensorDims(count):
    """Creates multiple ``TensorDim`` objects based on ``count``."""
    return [TensorDim() for i in range(count)]


def TensorIndexes(count):
    """Creates multiple ``TensorIndex`` objects based on ``count``."""
    return [TensorIndex() for i in range(count)]


def Placeholder(dtype_or_shape, dims=[], name=''):
    """Creates a placeholder tensor.

    Args:
        dtype_or_shape (DType | LogicalShape): A data type or a shape can be
            specified. If a shape is specified, the `dims` parameter is ignored.
        dims (list, optional): Specifies the dimensions of the ``Placeholder``.
        name (string, optional): A name to be assigned to the ``Tensor``.

    Returns:
        Tensor: The placeholder ``Tensor``.
    """
    if isinstance(dtype_or_shape, LogicalShape):
        shape = dtype_or_shape
    elif isinstance(dtype_or_shape, DType):
        shape = LogicalShape(dtype=dtype_or_shape, dims=dims)
    else:
        raise TypeError('Unsupported type {} for dtype_or_shape={}'.format(
            type(dtype_or_shape), dtype_or_shape))
    return Tensor(shape=shape, name=name)


class ProgramArgument:

    def __init__(self, arg):
        self.is_input = arg.is_input
        self.ref = TensorRef(Tensor(expr=ffi_call(lib.plaidml_expr_clone, arg.tensor)))
        self.shape = LogicalShape(ptr=ffi_call(lib.plaidml_logical_shape_clone, arg.shape))
        if arg.buffer:
            tensor_shape = self.shape.into_TensorShape()
            self.buffer = Buffer(tensor_shape, ptr=ffi_call(lib.plaidml_buffer_clone, arg.buffer))
        else:
            self.buffer = None


class Program(ForeignObject):
    __ffi_del__ = lib.plaidml_program_free
    __ffi_repr__ = lib.plaidml_program_repr

    def __init__(self,
                 name,
                 outputs,
                 updates=[],
                 floatx=DType.FLOAT32,
                 intx=DType.INT32,
                 debug=False,
                 target=None):
        if target is None:
            target = plaidml.settings.get('PLAIDML_TARGET')
        raw_outputs = [x.as_ptr() for x in outputs]
        dst_updates = [x[0].as_ptr() for x in updates]
        src_updates = [x[1].as_ptr() for x in updates]
        raw_args = ffi.new('plaidml_program_args**')
        ffi_obj = ffi_call(
            lib.plaidml_compile,
            name.encode(),
            target.encode(),
            len(raw_outputs),
            raw_outputs,
            len(updates),
            src_updates,
            dst_updates,
            floatx,
            intx,
            debug,
            raw_args,
        )
        self.args = [ProgramArgument(raw_args[0].elts[i]) for i in range(raw_args[0].size)]
        ffi_call(lib.plaidml_program_args_free, raw_args[0])
        super(Program, self).__init__(ffi_obj)

    @property
    def inputs(self):
        return [x for x in self.args if x.is_input]

    @property
    def outputs(self):
        return [x for x in self.args if not x.is_input]

    @property
    def passes(self):
        """Returns a list of passes.

        Each pass in the list is a tuple of ``(name, ir)``, where ``ir`` means
        `intermediate representation`.

        Note that ``debug`` must be enabled when compiling the program.

        Returns:
            :obj:`list` of :obj:`tuple` of :obj:`str`: The passes.

        """
        return plaidml.kvps_to_list(self._methodcall(lib.plaidml_program_get_passes))


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


def abs(x):
    """Computes the elementwise absolute value of ``x``.

    Args:
        x (Tensor): The tensor used to peform the elementwise ``abs``.

    Returns:
        Tensor: The result of the elementwise ``abs`` operation.
    """
    return call('abs', x)


def cast(x, dtype):
    """Casts the element type of a tensor ``x`` to the type specified by ``dtype``.

    Args:
        x (Tensor): The tensor used to peform the elementwise ``cast``.
        dtype (DType): The datatype to ``cast`` to

    Returns:
        Tensor: The result of the elementwise ``cast`` operation.
    """
    return Tensor(expr=ffi_call(lib.plaidml_expr_cast, wrap_tensor(x).as_ptr(), dtype))


def ceil(x):
    """Computes the elementwise ceiling of ``x``.

    Args:
        x (Tensor): The tensor used to peform the elementwise ``ceil``.

    Returns:
        Tensor: The result of the elementwise ``ceil`` operation.
    """
    return call('ceil', x)


def cond(lhs, rhs, true_case):
    return IndexedTensor(_ContractionPart(lib.PLAIDML_COMBO_OP_COND, (lhs, rhs, true_case)))


def cos(x):
    """Computes the elementwise cosine of ``x``.

    Args:
        x (Tensor): The tensor used to peform the elementwise ``cos``.

    Returns:
        Tensor: The result of the elementwise ``cos`` operation.
    """
    return call('cos', x)


def cosh(x):
    """Computes the elementwise hyperbolic cosine of ``x``.

    Args:
        x (Tensor): The tensor used to peform the elementwise ``cosh``.

    Returns:
        Tensor: The result of the elementwise ``cosh`` operation.
    """
    return call('cosh', x)


def exp(x):
    """Computes the elementwise natural exponential function of ``x``.

    Args:
        x (Tensor): The tensor used to peform the elementwise ``exp``.

    Returns:
        Tensor: The result of the elementwise ``exp`` operation.
    """
    return call('exp', x)


def floor(x):
    """Computes the elementwise floor of ``x``.

    Args:
        x (Tensor): The tensor used to peform the elementwise ``floor``.

    Returns:
        Tensor: The result of the elementwise ``floor`` operation.
    """
    return call('floor', x)


def gather(x, y):
    """Takes an input tensor (``x``) and a set of indices to gather over
    (``y``), and returns an output tensor that gathers the input tensor from the
    indices specified.

    Args:
        x (Tensor): The tensor to peform ``gather`` on.
        y (Tensor): The set of indices to ``gather`` over.

    Returns:
        Tensor: The result of the ``gather`` operation.
    """
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
    """Returns the identity of ``x``.

    Args:
        x (Tensor): The input Tensor.

    Returns:
        Tensor: The resultant tensor.
    """
    return call('ident', x)


def index(x, axis):
    """Returns the index of ``x`` at the specified ``axis``.

    Args:
        x (Tensor): The Tensor to index.
        axis (Tensor): The axis used for indexing.

    Returns:
        Tensor: The indexed tensor.
    """
    return call('index', x, axis)


def log(x):
    """Computes the elementwise natural logarithm of ``x``.

    Args:
        x (Tensor): The input Tensor.

    Returns:
        Tensor: The resultant tensor.
    """
    return call('log', x)


def max(x, y):
    """Computes the elementwise maximum of ``x`` and ``y``.

    Args:
        x (Tensor): The first input Tensor.
        y (Tensor): The second input Tensor.

    Returns:
        Tensor: The resultant tensor.
    """
    return call('max', x, y)


def min(x, y):
    """Computes the elementwise minimum of ``x`` and ``y``.

    Args:
        x (Tensor): The first input Tensor.
        y (Tensor): The second input Tensor.

    Returns:
        Tensor: The resultant tensor.
    """
    return call('min', x, y)


def pow(x, y):
    """Computes the elementwise ``y``th power of ``x``.

    Args:
        x (Tensor): The base Tensor.
        y (Tensor): The exponent Tensor.

    Returns:
        Tensor: The resultant tensor.
    """
    return call('pow', x, y)


def prng(state, shape):
    """Generates a Tensor of pseudorandom numbers seeded with values specified
    by ``state``.

    Args:
        state (Tensor): The seed values for the ``prng`` operation.
        shape (Tensor): The desired shape of the tensor of pseudorandom numbers.

    Returns:
        Tensor: The tensor of pseudorandom numbers.
    """
    return call('prng', state, *shape)


def reshape(x, dims):
    """Takes a tensor ``x`` and reshapes it according to ``dims``.

    Args:
        x (Tensor): The tensor to reshape.
        dims (list): The desired shape of the tensor.

    Returns:
        Tensor: The reshaped tensor.
    """
    return call('reshape', x, *dims)


def round(x):
    """Rounds ``x`` elementwise.

    Args:
        x (Tensor): The tensor to round.

    Returns:
        Tensor: The rounded tensor.
    """
    return call('round', x)


def scatter(x, y, z):
    """Takes an input tensor (``x``), a set of indices to scatter over (``y``),
    and the number of elements in the scattered tensor (``z``), and returns an
    output tensor that scatters the input tensor across the number of elements
    specified.

    Args:
        x (Tensor): The tensor to perform ``scatter`` on.
        y (Tensor): The tensor containing the indices to scatter over.
        z (Tensor): The number of elements in the scattered tensor.

    Returns:
        Tensor: The scattered tensor.
    """
    return call('scatter', x, y, z)


def select(cond, true_case, false_case):
    """Performs an elementwise conditional which returns the corresponding
    element in ``true_case`` if the condition is evaluated to be true or the
    corresponding element in ``false_case`` if the condition is evaluated to be
    false.

    Args:
        cond (Tensor): The tensor used to perform the conditional.
        true_case (Tensor): The tensor whose elements are selected if the
            condition evaluates to be true.
        false_case (Tensor): The tensor whose elements are selected if the
            condition evaluates to be false.

    Returns:
        Tensor: The tensor with the conditionally selected elements.
    """
    return call('cond', cond, true_case, false_case)


def shape(x):
    """Returns the shape of ``x`` as a Tensor.

    Args:
        x (Tensor): The tensor used to calculate the shape.

    Returns:
        Tensor: The shape of the tensor.
    """
    return call('shape', x)


def sin(x):
    """Computes the elementwise sine of ``x``.

    Args:
        x (Tensor): The tensor used to peform the elementwise ``sin``.

    Returns:
        Tensor: The result of the elementwise ``sin`` operation.
    """
    return call('sin', x)


def sinh(x):
    """Computes the elementwise hyperbolic sine of ``x``.

    Args:
        x (Tensor): The tensor used to peform the elementwise ``sinh``.

    Returns:
        Tensor: The result of the elementwise ``sinh`` operation.

    """
    return call('sin', x)


def sqrt(x):
    """Computes the elementwise square root of ``x``.

    Args:
        x (Tensor): The tensor used to peform the elementwise ``sqrt``.

    Returns:
        Tensor: The result of the elementwise ``sqrt`` operation.
    """
    return call('sqrt', x)


def tan(x):
    """Computes the elementwise tangent of ``x``.

    Args:
        x (Tensor): The tensor used to peform the elementwise ``tan``.

    Returns:
        Tensor: The result of the elementwise ``tan`` operation.
    """
    return call('tan', x)


def tanh(x):
    """Computes the elementwise hyperbolic tangent of ``x``.

    Args:
        x (Tensor): The tensor used to peform the elementwise ``tanh``.

    Returns:
        Tensor: The result of the elementwise ``tanh`` operation.
    """
    return call('tanh', x)
