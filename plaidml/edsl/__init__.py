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

Constraint = namedtuple('Constraint', ['lhs', 'rhs'])


def _wrap_dim(x):
    if isinstance(x, six.integer_types):
        return TensorDim(expr=ffi_call(lib.plaidml_dim_expr_int, x))
    return x


def _dim_op(op, *args):
    args = [_wrap_dim(x) for x in args]
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

    def __neg__(self):
        """Negates a TensorDim in a polynomial expression.

        Example:
            >>> N, M = TensorDims(2)
            >>> A = Placeholder(DType.FLOAT32, [3, 3])
            >>> A.bind_dims(N, M)
            >>> R = Contraction().outShape(-N)
        """
        return TensorDim(_dim_op(lib.PLAIDML_INT_OP_NEG, self))

    def __add__(self, other):
        """Performs an addition between a TensorDim and another operand in a
        polynomial expression.

        Example:
            >>> N, M = TensorDims(2)
            >>> A = Placeholder(DType.FLOAT32, [3, 3])
            >>> A.bind_dims(N, M)
            >>> R = Contraction().outShape(N + 5)
        """
        return TensorDim(_dim_op(lib.PLAIDML_INT_OP_ADD, self, other))

    def __radd__(self, other):
        """Performs an addition between a TensorDim and another operand in a
        polynomial expression.

        Example:
            >>> N, M = TensorDims(2)
            >>> A = Placeholder(DType.FLOAT32, [3, 3])
            >>> A.bind_dims(N, M)
            >>> R = Contraction().outShape(5 + N)
        """
        return TensorDim(_dim_op(lib.PLAIDML_INT_OP_ADD, other, self))

    def __sub__(self, other):
        """Performs a subtraction between a TensorDim and another operand in a
        polynomial expression.

        Example:
            >>> N, M = TensorDims(2)
            >>> A = Placeholder(DType.FLOAT32, [3, 3])
            >>> A.bind_dims(N, M)
            >>> R = Contraction().outShape(N - 5)
        """
        return TensorDim(_dim_op(lib.PLAIDML_INT_OP_SUB, self, other))

    def __rsub__(self, other):
        """Performs a subtraction between a TensorDim and another operand in a
        polynomial expression.

        Example:
            >>> N, M = TensorDims(2)
            >>> A = Placeholder(DType.FLOAT32, [3, 3])
            >>> A.bind_dims(N, M)
            >>> R = Contraction().outShape(5 - N)
        """
        return TensorDim(_dim_op(lib.PLAIDML_INT_OP_SUB, other, self))

    def __mul__(self, other):
        """Performs a multiplication between a TensorDim and another operand in
        a polynomial expression.

        Example:
            >>> N, M = TensorDims(2)
            >>> A = Placeholder(DType.FLOAT32, [3, 3])
            >>> A.bind_dims(N, M)
            >>> R = Contraction().outShape(N * 5)
        """
        return TensorDim(_dim_op(lib.PLAIDML_INT_OP_MUL, self, other))

    def __rmul__(self, other):
        """Performs a multiplication between a TensorDim and another operand in
        a polynomial expression.

        Example:
            >>> N, M = TensorDims(2)
            >>> A = Placeholder(DType.FLOAT32, [3, 3])
            >>> A.bind_dims(N, M)
            >>> R = Contraction().outShape(5 * N)
        """
        return TensorDim(_dim_op(lib.PLAIDML_INT_OP_MUL, other, self))

    def __floordiv__(self, other):
        """Performs a floor division between a TensorDim and another operand in
        a polynomial expression.

        Example:
            >>> N, M = TensorDims(2)
            >>> A = Placeholder(DType.FLOAT32, [3, 3])
            >>> A.bind_dims(N, M)
            >>> R = Contraction().outShape(N // 5)
        """
        return TensorDim(_dim_op(lib.PLAIDML_INT_OP_DIV, self, other))

    def __rfloordiv__(self, other):
        """Performs a floor division between a TensorDim and another operand in
        a polynomial expression.

        Example:
            >>> N, M = TensorDims(2)
            >>> A = Placeholder(DType.FLOAT32, [3, 3])
            >>> A.bind_dims(N, M)
            >>> R = Contraction().outShape(5 // N)
        """
        return TensorDim(_dim_op(lib.PLAIDML_INT_OP_DIV, other, self))


def _wrap_poly(x):
    if isinstance(x, six.integer_types):
        return TensorIndex(expr=ffi_call(lib.plaidml_poly_expr_literal, x))
    if isinstance(x, TensorDim):
        return TensorIndex(expr=ffi_call(lib.plaidml_poly_expr_dim, x.as_ptr()))
    return x


def _poly_op(op, *args):
    args = [_wrap_poly(x) for x in args]
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
            >>> R = Contraction().sum(A[i, j]).add_constraint(i < 5).build()
        """
        return Constraint(self, _wrap_dim(rhs))

    def __neg__(self):
        """Negates a TensorIndex in a polynomial expression.

        Example:
            >>> i, j = TensorIndexes(2)
            >>> A = Placeholder(DType.FLOAT32, [3, 3])
            >>> R = Contraction().sum(A[-i, j]).build()
        """
        return TensorIndex(_poly_op(lib.PLAIDML_INT_OP_NEG, self))

    def __add__(self, rhs):
        """Performs an addition between a TensorIndex and another operand in a
        polynomial expression.

        Example:
            >>> i, j = TensorIndexes(2)
            >>> A = Placeholder(DType.FLOAT32, [3, 3])
            >>> R = Contraction().sum(A[i + 5, j]).build()
        """
        return TensorIndex(_poly_op(lib.PLAIDML_INT_OP_ADD, self, rhs))

    def __radd__(self, lhs):
        """Performs an addition between a TensorIndex and another operand in a
        polynomial expression.

        Example:
            >>> i, j = TensorIndexes(2)
            >>> A = Placeholder(DType.FLOAT32, [3, 3])
            >>> R = Contraction().sum(A[5 + i, j]).build()
        """
        return TensorIndex(_poly_op(lib.PLAIDML_INT_OP_ADD, lhs, self))

    def __sub__(self, rhs):
        """Performs a subtraction between a TensorIndex and another operand in a
        polynomial expression.

        Example:
            >>> i, j = TensorIndexes(2)
            >>> A = Placeholder(DType.FLOAT32, [3, 3])
            >>> R = Contraction().sum(A[i - 5, j]).build()
        """
        return TensorIndex(_poly_op(lib.PLAIDML_INT_OP_SUB, self, rhs))

    def __rsub__(self, lhs):
        """Performs a subtraction between a TensorIndex and another operand in a
        polynomial expression.

        Example:
            >>> i, j = TensorIndexes(2)
            >>> A = Placeholder(DType.FLOAT32, [3, 3])
            >>> R = Contraction().sum(A[5 - i, j]).build()
        """
        return TensorIndex(_poly_op(lib.PLAIDML_INT_OP_SUB, lhs, self))

    def __mul__(self, rhs):
        """Performs a multiplication between a TensorIndex and another operand
        in a polynomial expression.

        Example:
            >>> i, j = TensorIndexes(2)
            >>> A = Placeholder(DType.FLOAT32, [3, 3])
            >>> R = Contraction().sum()
            >>> R = Contraction().sum(A[i * 5, j]).build()
        """
        return TensorIndex(_poly_op(lib.PLAIDML_INT_OP_MUL, self, rhs))

    def __rmul__(self, lhs):
        """Performs a multiplication between a TensorIndex and another operand
        in a polynomial expression.

        Example:
            >>> i, j = TensorIndexes(2)
            >>> A = Placeholder(DType.FLOAT32, [3, 3])
            >>> R = Contraction().sum(A[5 * i, j]).build()
        """
        return TensorIndex(_poly_op(lib.PLAIDML_INT_OP_MUL, lhs, self))

    def __floordiv__(self, rhs):
        """Performs a floor division between a TensorIndex and another operand
        in a polynomial expression.

        Example:
            >>> i, j = TensorIndexes(2)
            >>> A = Placeholder(DType.FLOAT32, [3, 3])
            >>> R = Contraction().sum(A[i // 5, j]).build()
        """
        return TensorIndex(_poly_op(lib.PLAIDML_INT_OP_DIV, self, rhs))

    def __rfloordiv__(self, lhs):
        """Performs a floor division between a TensorIndex and another operand
        in a polynomial expression.

        Example:
            >>> i, j = TensorIndexes(2)
            >>> A = Placeholder(DType.FLOAT32, [3, 3])
            >>> R = Contraction().sum(A[5 // i, j]).build()
        """
        return TensorIndex(_poly_op(lib.PLAIDML_INT_OP_DIV, lhs, self))


class TensorLens(object):

    def __init__(self, source='', target=''):
        self.map = []
        if len(source) != len(target):
            raise ValueError('source and target rank mismatch')
        for i in range(len(source)):
            pos = target.find(source[i])
            if pos == -1:
                raise ValueError('source and target dims mismatch')
            self.map.append(pos)

    def apply(self, dims):
        if len(self.map) == 0:
            return dims
        if len(dims) != len(self.map):
            raise ValueError('rank mismatch in TensorLens apply')
        return [dims[x] for x in self.map]


class Contraction(object):

    def __init__(self, lens=TensorLens(), name=''):
        self.__lens = lens
        self.__name = name
        self.__outDims = []
        self.__outIdxs = []
        self.__constraints = []
        self.__rhs = None
        self.__agg_op = None
        self.__init = None

    def outShape(self, *args):
        self.__outDims = args
        return self

    def outAccess(self, *args):
        self.__outIdxs = args
        return self

    def assign(self, rhs):
        self.__agg_op = lib.PLAIDML_AGG_OP_ASSIGN
        self.__rhs = rhs
        return self

    def max(self, rhs):
        """Performs a `maximum` reduction within a contraction.

        Example:
            >>> i, j = TensorIndexes(2)
            >>> A = Placeholder(DType.FLOAT32, [3, 3])
            >>> R = Contraction().max(A[i, j]).build()
        """
        self.__agg_op = lib.PLAIDML_AGG_OP_MAX
        self.__rhs = rhs
        return self

    def min(self, rhs):
        """Performs a `minimum` reduction within a contraction.

        Example:
            >>> i, j = TensorIndexes(2)
            >>> A = Placeholder(DType.FLOAT32, [3, 3])
            >>> R = Contraction().min(A[i, j]).build()
        """
        self.__agg_op = lib.PLAIDML_AGG_OP_MIN
        self.__rhs = rhs
        return self

    def product(self, rhs):
        """Performs a `product` reduction within a contraction.

        Example:
            >>> i, j = TensorIndexes(2)
            >>> A = Placeholder(DType.FLOAT32, [3, 3])
            >>> R = Contraction().product(A[i, j]).build()
        """
        self.__agg_op = lib.PLAIDML_AGG_OP_PROD
        return self

    def sum(self, rhs):
        """Performs a `summation` reduction within a contraction.

        Example:
            >>> i, j = TensorIndexes(2)
            >>> A = Placeholder(DType.FLOAT32, [3, 3])
            >>> R = Contraction().sum(A[i, j]).build()
        """
        self.__agg_op = lib.PLAIDML_AGG_OP_SUM
        self.__rhs = rhs
        return self

    def init(self, rhs):
        self.__init = rhs
        return self

    def add_constraint(self, constraint):
        self.__constraints.append(constraint)
        return self

    def add_constraints(self, constraints):
        self.__constraints.extend(constraints)
        return self

    def build(self):

        if isinstance(self.__rhs, IndexedTensor):
            rhs = self.__rhs
        elif isinstance(self.__rhs, Tensor):
            rhs = IndexedTensor(lib.PLAIDML_COMBO_OP_NONE, ref=self.__rhs, idxs=())
        else:
            tensor = Tensor(value=self.__rhs)
            rhs = IndexedTensor(lib.PLAIDML_COMBO_OP_NONE, ref=tensor, idxs=())

        def make_list(idxs):
            if isinstance(idxs, tuple) or isinstance(idxs, list):
                return idxs
            return [idxs]

        dims = [_wrap_dim(x) for x in self.__outDims]
        raw_dims = [x.as_ptr() for x in self.__lens.apply(dims)]

        idxs = [_wrap_poly(x) for x in make_list(self.__outIdxs)]
        raw_idxs = [x.as_ptr() for x in self.__lens.apply(idxs)]

        init = ffi.NULL
        if self.__init:
            init = self.__init.as_ptr()

        tensor = Tensor(expr=ffi_call(
            lib.plaidml_expr_contraction,
            self.__agg_op,
            rhs._op,
            len(raw_idxs),
            raw_idxs,
            raw_dims,
            init,
            self.__name.encode(),
        ))

        if rhs._op == lib.PLAIDML_COMBO_OP_NONE:
            operands = [rhs]
        else:
            operands = rhs._args

        for operand in operands:
            idxs = [_wrap_poly(x) for x in make_list(operand._idxs)]
            raw_idxs = [x.as_ptr() for x in operand._ref._lens.apply(idxs)]
            ffi_call(
                lib.plaidml_contraction_add_operand,
                tensor.as_ptr(),
                operand._ref.as_ptr(),
                len(raw_idxs),
                raw_idxs,
            )

        for constraint in self.__constraints:
            ffi_call(
                lib.plaidml_contraction_add_constraint,
                tensor.as_ptr(),
                constraint.lhs.as_ptr(),
                constraint.rhs.as_ptr(),
            )

        ffi_call(lib.plaidml_contraction_build, tensor.as_ptr())
        return tensor


class IndexedTensor(object):

    def __init__(self, op, ref=None, idxs=None, args=None):
        self._op = op
        self._ref = ref
        self._idxs = idxs
        self._args = args

    def __add__(self, rhs):
        """Represents an `addition` combination within a contraction.

        Example:
            >>> i, j, k = TensorIndexes(3)
            >>> A = Placeholder(DType.FLOAT32, [3, 3])
            >>> B = Placeholder(DType.FLOAT32, [3, 3])
            >>> A[i, j] + B[j, k]
        """
        return IndexedTensor(lib.PLAIDML_COMBO_OP_ADD, args=(self, rhs))

    def __mul__(self, rhs):
        """Represents a `multiply` combination within in a contraction.

        Example:
            >>> i, j, k = TensorIndexes(3)
            >>> A = Placeholder(DType.FLOAT32, [3, 3])
            >>> B = Placeholder(DType.FLOAT32, [3, 3])
            >>> A[i, j] * B[j, k]
        """
        return IndexedTensor(lib.PLAIDML_COMBO_OP_MUL, args=(self, rhs))

    def __eq__(self, rhs):
        """Represents an `equality comparison` combination within a contraction.

        Example:
            >>> i, j, k = TensorIndexes(3)
            >>> A = Placeholder(DType.FLOAT32, [3, 3])
            >>> A[i, j] == B[j, k]
        """
        return IndexedTensor(lib.PLAIDML_COMBO_OP_EQ, args=(self, rhs))


class Tensor(ForeignObject):
    """Represents a multi-dimensional result of an Operation."""

    __ffi_del__ = lib.plaidml_expr_free
    __ffi_repr__ = lib.plaidml_expr_repr

    def __init__(self, expr=None, value=None, lens=TensorLens()):
        self._lens = lens
        if value is not None:
            if isinstance(value, six.integer_types):
                expr = ffi_call(lib.plaidml_expr_int, value)
            elif isinstance(value, float):
                expr = ffi_call(lib.plaidml_expr_float, value)
            else:
                raise TypeError('Invalid type for value={}'.format(value))
        elif expr is None:
            raise ValueError('One of expr= or value= must be specified.')
        super(Tensor, self).__init__(expr)

    def __getitem__(self, key):
        return IndexedTensor(lib.PLAIDML_COMBO_OP_NONE, ref=self, idxs=key)

    # Represents an eltwise negation
    def __neg__(self):
        return intrinsic('neg', self)

    # Represents an eltwise bit_not
    def __invert__(self):
        return intrinsic('bit_not', self)

    # Represents an eltwise addition
    def __add__(self, rhs):
        return intrinsic('add', self, rhs)

    def __radd__(self, lhs):
        return intrinsic('add', lhs, self)

    # Represents an eltwise subtraction
    def __sub__(self, rhs):
        return intrinsic('sub', self, rhs)

    def __rsub__(self, lhs):
        return intrinsic('sub', lhs, self)

    # Represents an eltwise multiplication
    def __mul__(self, rhs):
        return intrinsic('mul', self, rhs)

    def __rmul__(self, lhs):
        return intrinsic('mul', lhs, self)

    # Represents an eltwise division
    def __div__(self, rhs):
        return intrinsic('div', self, rhs)

    def __rdiv__(self, lhs):
        return intrinsic('div', lhs, self)

    # Represents an eltwise division
    def __truediv__(self, rhs):
        return intrinsic('div', self, rhs)

    def __rtruediv__(self, lhs):
        return intrinsic('div', lhs, self)

    # Represents an eltwise cmp_eq
    def __eq__(self, rhs):
        return intrinsic('cmp_eq', self, rhs)

    # Represents an eltwise cmp_ne
    def __ne__(self, rhs):
        return intrinsic('cmp_ne', self, rhs)

    # Represents an eltwise cmp_lt
    def __lt__(self, rhs):
        return intrinsic('cmp_lt', self, rhs)

    # Represents an eltwise cmp_gt
    def __gt__(self, rhs):
        return intrinsic('cmp_gt', self, rhs)

    # Represents an eltwise cmp_le
    def __le__(self, rhs):
        return intrinsic('cmp_le', self, rhs)

    # Represents an eltwise cmp_ge
    def __ge__(self, rhs):
        return intrinsic('cmp_ge', self, rhs)

    # Represents an eltwise bit_shl
    def __lshift__(self, rhs):
        return intrinsic('bit_shl', self, rhs)

    def __rlshift__(self, lhs):
        return intrinsic('bit_shl', lhs, self)

    # Represents an eltwise bit_shr
    def __rshift__(self, rhs):
        return intrinsic('bit_shr', self, rhs)

    def __rrshift__(self, lhs):
        return intrinsic('bit_shr', lhs, self)

    # Represents an eltwise bit_and
    def __and__(self, rhs):
        return intrinsic('bit_and', self, rhs)

    def __rand__(self, lhs):
        return intrinsic('bit_and', lhs, self)

    # Represents an eltwise bit_or
    def __or__(self, rhs):
        return intrinsic('bit_or', self, rhs)

    def __ror__(self, lhs):
        return intrinsic('bit_or', lhs, self)

    # Represents an eltwise bit_xor
    def __xor__(self, rhs):
        return intrinsic('bit_xor', self, rhs)

    def __rxor__(self, lhs):
        return intrinsic('bit_xor', lhs, self)

    # Return the tensor's shape
    def compute_shape(self):
        return TensorShape(ptr=self._methodcall(lib.plaidml_expr_get_shape))

    @property
    def dtype(self):
        return DType(self._methodcall(lib.plaidml_expr_get_dtype))

    @property
    def rank(self):
        return self._methodcall(lib.plaidml_expr_get_rank)

    # Verify that the specified dims match the dims of this tensor.
    def bind_dims(self, *dims):
        raw_dims = [x.as_ptr() for x in self._lens.apply(dims)]
        self._methodcall(lib.plaidml_expr_bind_dims, len(raw_dims), raw_dims)

    def element(self, ordinal):
        return Tensor(expr=self._methodcall(lib.plaidml_expr_element, ordinal))

    def use(self, lens):
        return Tensor(expr=self._methodcall(lib.plaidml_expr_clone), lens=lens)


class Value(ForeignObject):
    __ffi_del__ = lib.plaidml_value_free
    __ffi_repr__ = lib.plaidml_value_repr

    def __init__(self, value):
        # logger.debug('Value({})'.format(value))
        if isinstance(value, np.ndarray):
            if value.ndim == 0:
                value = value.item()
            else:
                value = value.tolist()
        if value is None:
            ffi_obj = ffi_call(lib.plaidml_value_none)
        elif isinstance(value, enum.IntEnum):
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
            ffi_obj = ffi_call(lib.plaidml_value_str, value.encode())
        elif isinstance(value, ffi.CData) and ffi.typeof(value) is ffi.typeof('plaidml_value*'):
            ffi_obj = value
        else:
            raise TypeError('Unsupported type {} for value={}'.format(type(value), value))
        super(Value, self).__init__(ffi_obj)

    def as_tensor(self):
        return Tensor(expr=self._methodcall(lib.plaidml_value_expr_get))


def TensorDims(count):
    """Creates multiple ``TensorDim`` objects based on ``count``."""
    return [TensorDim() for i in range(count)]


def TensorIndexes(count):
    """Creates multiple ``TensorIndex`` objects based on ``count``."""
    return [TensorIndex() for i in range(count)]


def Constant(buffer, dims=[], name=''):
    """Creates a tensor with constant values.

    Args:
        buffer (Buffer): A Buffer that stores the values of the ``Constant``.
        dims (list, optional): Specifies the dimensions of the ``Constant``.
        name (string, optional): A name to be assigned to the ``Tensor``.

    Returns:
        Tensor: The constant ``Tensor``.
    """
    return Tensor(expr=ffi_call(lib.plaidml_expr_constant, buffer.as_ptr(), name.encode()))


def Placeholder(dtype_or_shape, dims=[], name=''):
    """Creates a placeholder tensor.

    Args:
        dtype_or_shape (DType | TensorShape): A data type or a shape can be
            specified. If a shape is specified, the `dims` parameter is ignored.
        dims (list, optional): Specifies the dimensions of the ``Placeholder``.
        name (string, optional): A name to be assigned to the ``Tensor``.

    Returns:
        Tensor: The placeholder ``Tensor``.
    """
    if isinstance(dtype_or_shape, TensorShape):
        shape = dtype_or_shape
    elif isinstance(dtype_or_shape, DType):
        shape = TensorShape(dtype=dtype_or_shape, sizes=dims)
    else:
        raise TypeError('Unsupported type {} for dtype_or_shape={}'.format(
            type(dtype_or_shape), dtype_or_shape))
    return Tensor(expr=ffi_call(lib.plaidml_expr_input, shape.as_ptr(), name.encode()))


def _wrap_tensor(x):
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


def intrinsic(fn, *args):
    args = [_wrap_tensor(x) for x in args]
    raw_args = [x.as_ptr() for x in args]
    return Tensor(expr=ffi_call(lib.plaidml_expr_intrinsic, fn.encode(), len(args), raw_args))


def abs(x):
    """Computes the elementwise absolute value of ``x``.

    Args:
        x (Tensor): The tensor used to peform the elementwise ``abs``.

    Returns:
        Tensor: The result of the elementwise ``abs`` operation.
    """
    return intrinsic('abs', x)


def cast(x, dtype):
    """Casts the element type of a tensor ``x`` to the type specified by ``dtype``.

    Args:
        x (Tensor): The tensor used to peform the elementwise ``cast``.
        dtype (DType): The datatype to ``cast`` to

    Returns:
        Tensor: The result of the elementwise ``cast`` operation.
    """
    tensor = _wrap_tensor(x)
    return Tensor(expr=ffi_call(lib.plaidml_expr_cast, tensor.as_ptr(), dtype))


def ceil(x):
    """Computes the elementwise ceiling of ``x``.

    Args:
        x (Tensor): The tensor used to peform the elementwise ``ceil``.

    Returns:
        Tensor: The result of the elementwise ``ceil`` operation.
    """
    return intrinsic('ceil', x)


def cond(lhs, rhs, true_case):
    return IndexedTensor(lib.PLAIDML_COMBO_OP_COND, args=(lhs, rhs, true_case))


def cos(x):
    """Computes the elementwise cosine of ``x``.

    Args:
        x (Tensor): The tensor used to peform the elementwise ``cos``.

    Returns:
        Tensor: The result of the elementwise ``cos`` operation.
    """
    return intrinsic('cos', x)


def cosh(x):
    """Computes the elementwise hyperbolic cosine of ``x``.

    Args:
        x (Tensor): The tensor used to peform the elementwise ``cosh``.

    Returns:
        Tensor: The result of the elementwise ``cosh`` operation.
    """
    return intrinsic('cosh', x)


def exp(x):
    """Computes the elementwise natural exponential function of ``x``.

    Args:
        x (Tensor): The tensor used to peform the elementwise ``exp``.

    Returns:
        Tensor: The result of the elementwise ``exp`` operation.
    """
    return intrinsic('exp', x)


def floor(x):
    """Computes the elementwise floor of ``x``.

    Args:
        x (Tensor): The tensor used to peform the elementwise ``floor``.

    Returns:
        Tensor: The result of the elementwise ``floor`` operation.
    """
    return intrinsic('floor', x)


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
    return intrinsic('gather', x, y)


def ident(x):
    """Returns the identity of ``x``.

    Args:
        x (Tensor): The input Tensor.

    Returns:
        Tensor: The resultant tensor.
    """
    return intrinsic('ident', x)


def index(dims, axis):
    """Returns a tensor populated with the index value of the shape and axis specified.

    Args:
        dims (list): The shape of the tensor to base indexing on.
        axis (Tensor): The axis used for indexing.

    Returns:
        Tensor: The resultant tensor.
    """
    return intrinsic('index', axis, *dims)


def log(x):
    """Computes the elementwise natural logarithm of ``x``.

    Args:
        x (Tensor): The input Tensor.

    Returns:
        Tensor: The resultant tensor.
    """
    return intrinsic('log', x)


def max(x, y):
    """Computes the elementwise maximum of ``x`` and ``y``.

    Args:
        x (Tensor): The first input Tensor.
        y (Tensor): The second input Tensor.

    Returns:
        Tensor: The resultant tensor.
    """
    return intrinsic('max', x, y)


def min(x, y):
    """Computes the elementwise minimum of ``x`` and ``y``.

    Args:
        x (Tensor): The first input Tensor.
        y (Tensor): The second input Tensor.

    Returns:
        Tensor: The resultant tensor.
    """
    return intrinsic('min', x, y)


def pow(x, y):
    """Computes the elementwise ``y``th power of ``x``.

    Args:
        x (Tensor): The base Tensor.
        y (Tensor): The exponent Tensor.

    Returns:
        Tensor: The resultant tensor.
    """
    return intrinsic('pow', x, y)


def prng(state, shape):
    """Generates a Tensor of pseudorandom numbers seeded with values specified
    by ``state``.

    Args:
        state (Tensor): The state of the pseudorandom number generator.
        shape (Tensor): The desired shape of the tensor of pseudorandom numbers.

    Returns:
        Tensor: The tensor of pseudorandom numbers.
        Tensor: The updated state of the pseudorandom number generator.
    """
    x = intrinsic('prng', state, *shape)
    return x.element(0), x.element(1)


def reshape(x, dims):
    """Takes a tensor ``x`` and reshapes it according to ``dims``.

    Args:
        x (Tensor): The tensor to reshape.
        dims (list): The desired shape of the tensor.

    Returns:
        Tensor: The reshaped tensor.
    """
    return intrinsic('reshape', x, *dims)


def round(x):
    """Rounds ``x`` elementwise.

    Args:
        x (Tensor): The tensor to round.

    Returns:
        Tensor: The rounded tensor.
    """
    return intrinsic('round', x)


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
    return intrinsic('scatter', x, y, z)


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
    return intrinsic('select', cond, true_case, false_case)


def shape(x):
    """Returns the shape of ``x`` as a Tensor.

    Args:
        x (Tensor): The tensor used to calculate the shape.

    Returns:
        Tensor: The shape of the tensor.
    """
    return intrinsic('shape', x)


def sin(x):
    """Computes the elementwise sine of ``x``.

    Args:
        x (Tensor): The tensor used to peform the elementwise ``sin``.

    Returns:
        Tensor: The result of the elementwise ``sin`` operation.
    """
    return intrinsic('sin', x)


def sinh(x):
    """Computes the elementwise hyperbolic sine of ``x``.

    Args:
        x (Tensor): The tensor used to peform the elementwise ``sinh``.

    Returns:
        Tensor: The result of the elementwise ``sinh`` operation.

    """
    return intrinsic('sin', x)


def sqrt(x):
    """Computes the elementwise square root of ``x``.

    Args:
        x (Tensor): The tensor used to peform the elementwise ``sqrt``.

    Returns:
        Tensor: The result of the elementwise ``sqrt`` operation.
    """
    return intrinsic('sqrt', x)


def tan(x):
    """Computes the elementwise tangent of ``x``.

    Args:
        x (Tensor): The tensor used to peform the elementwise ``tan``.

    Returns:
        Tensor: The result of the elementwise ``tan`` operation.
    """
    return intrinsic('tan', x)


def tanh(x):
    """Computes the elementwise hyperbolic tangent of ``x``.

    Args:
        x (Tensor): The tensor used to peform the elementwise ``tanh``.

    Returns:
        Tensor: The result of the elementwise ``tanh`` operation.
    """
    return intrinsic('tanh', x)


def _wrap_value(x):
    if isinstance(x, Value):
        return x
    return Value(x)


def pragma(tensor, op, attrs):
    tensor = _wrap_tensor(tensor)
    keys = []
    values = []
    raw_attrs = []
    for key, value in attrs.items():
        key = ffi.new('char[]', key.encode())
        keys.append(key)
        value = _wrap_value(value)
        values.append(value)
        raw_attrs.append(ffi.new('plaidml_attr*', {'key': key, 'value': value.as_ptr()}))
    return Tensor(expr=ffi_call(lib.plaidml_expr_pragma, tensor.as_ptr(), op.encode(),
                                len(raw_attrs), raw_attrs))


def trace(x, msg):
    return pragma(x, 'trace', dict(msg=msg))
