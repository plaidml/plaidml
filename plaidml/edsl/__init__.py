# Copyright 2019 Intel Corporation.

import logging
from collections import namedtuple

import numpy as np
import six

from plaidml import DType
from plaidml.core import TensorShape, Buffer
from plaidml.ffi import ForeignObject, ffi, ffi_call, lib

logger = logging.getLogger(__name__)


def __init():
    """
    Initializes PlaidML's EDSL API.
    """
    ffi_call(lib.plaidml_edsl_init)


ffi.init_once(__init, 'plaidml_edsl_init')


class LogicalShape(ForeignObject):
    """Docstring for class LogicalShape"""
    __ffi_del__ = lib.plaidml_logical_shape_free
    __ffi_repr__ = lib.plaidml_logical_shape_repr

    def __init__(self, dtype=None, dims=[], ptr=None):
        """Initializes the LogicalShape.

        Args:
            self (pointer): The object pointer for a LogicalShape
            dtype (DType): Description of dtype
            dims (list): The dimensions of the LogicalShape
            ptr (pointer): Description of ptr

        """
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
        """Returns the dimensions of a LogicalShape as a list.

        Args:
            self (pointer): The object pointer for a LogicalShape

        Returns:
            list (int): Integer dimensions of the LogicalShape.

        """
        return [
            ffi_call(lib.plaidml_logical_shape_get_dim_int, self.as_ptr(), i)
            for i in range(self.ndims)
        ]

    def into_TensorShape(self):
        """Converts a LogicalShape into a TensorShape.

        Args:
            self (pointer): The object pointer for a LogicalShape

        Returns:
            TensorShape: The resultant TensorShape.

        """
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
    """Docstring for class TensorDim"""
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
    """Docstring for class TensorIndex"""
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
    """Docstring for class IndexedTensor"""

    def __init__(self, impl, tensor=None):
        self._impl = impl
        self._tensor = tensor

    def __repr__(self):
        return repr(self._impl)

    def __iadd__(self, rhs):
        """Represents an aggregation_op of SUM in a contraction.

        Args:
            self (pointer): The object pointer for an IndexedTensor
            rhs (IndexedTensor): The rhs

        Returns:
            y (IndexedTensor): The result

        """
        return IndexedTensor(self._make_contraction(lib.PLAIDML_AGG_OP_SUM, rhs))

    def __imul__(self, rhs):
        """Represents an aggregation_op of PROD in a contraction"""
        return IndexedTensor(self._make_contraction(lib.PLAIDML_AGG_OP_PROD, rhs))

    def __ge__(self, rhs):
        """Represents an aggregation_op of MAX in a contraction"""
        self._tensor._set_contraction(self._make_contraction(lib.PLAIDML_AGG_OP_MAX, rhs))

    def __le__(self, rhs):
        """Represents an aggregation_op of MIN in a contraction"""
        self._tensor._set_contraction(self._make_contraction(lib.PLAIDML_AGG_OP_MIN, rhs))

    def __add__(self, rhs):
        """Represents a combo_op of PLUS in a contraction"""
        return IndexedTensor(_ContractionPart(lib.PLAIDML_COMBO_OP_ADD, (self, rhs)))

    def __mul__(self, rhs):
        """Represents a combo_op of MULTIPLY in a contraction"""
        return IndexedTensor(_ContractionPart(lib.PLAIDML_COMBO_OP_MUL, (self, rhs)))

    def __eq__(self, rhs):
        """Represents a combo_op of EQ in a contraction"""
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
    """Docstring for class Tensor"""
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


class TensorRef:
    """Docstring for class TensorRef"""

    def __init__(self, tensor):
        self.tensor = tensor

    def __hash__(self):
        return hash(ffi_call(lib.plaidml_expr_ptr, self.tensor.as_ptr()))

    def __eq__(self, other):
        if isinstance(other, Tensor):
            return self.__hash__() == TensorRef(other).__hash__()
        return self.__hash__() == other.__hash__()


class Value(ForeignObject):
    """Docstring for class Value"""
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
        return Tensor(expr=ffi_call(lib.plaidml_value_expr_get, self.as_ptr()))


def TensorOutput(*args):
    return Tensor(dims=args)


def TensorDims(count):
    return [TensorDim() for i in range(count)]


def TensorIndexes(count):
    return [TensorIndex() for i in range(count)]


class ProgramArgument:
    """Docstring for class ProgramArgument"""

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
    """Docstring for class Program"""
    __ffi_del__ = lib.plaidml_program_free
    __ffi_repr__ = lib.plaidml_program_repr

    def __init__(self, name, outputs, updates=[], floatx=DType.FLOAT32, intx=DType.INT32):
        raw_outputs = [x.as_ptr() for x in outputs]
        dst_updates = [x[0].as_ptr() for x in updates]
        src_updates = [x[1].as_ptr() for x in updates]
        raw_args = ffi.new('plaidml_program_args**')
        ffi_obj = ffi_call(
            lib.plaidml_program_evaluate,
            name.encode(),
            len(raw_outputs),
            raw_outputs,
            len(updates),
            src_updates,
            dst_updates,
            floatx,
            intx,
            raw_args,
        )
        self.args = [ProgramArgument(raw_args[0].args[i]) for i in range(raw_args[0].nargs)]
        ffi_call(lib.plaidml_program_args_free, raw_args[0])
        super(Program, self).__init__(ffi_obj)

    @property
    def inputs(self):
        return [x for x in self.args if x.is_input]

    @property
    def outputs(self):
        return [x for x in self.args if not x.is_input]


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
            y (Tensor): The result of the elementwise ``abs`` operation.

    """
    return call('abs', x)


def cast(x, dtype):
    """Casts the element type of a tensor ``x`` to the type specified by ``dtype``.

        Args:
            x (Tensor): The tensor used to peform the elementwise ``cast``.
            dtype (DType): The datatype to ``cast`` to

        Returns:
            y (Tensor): The result of the elementwise ``cast`` operation.

    """
    return Tensor(expr=ffi_call(lib.plaidml_expr_cast, wrap_tensor(x).as_ptr(), dtype))


def ceil(x):
    """Computes the elementwise ceiling of ``x``.

        Args:
            x (Tensor): The tensor used to peform the elementwise ``ceil``.

        Returns:
            y (Tensor): The result of the elementwise ``ceil`` operation.

    """
    return call('ceil', x)


def cond(lhs, rhs, true_case):
    """A placeholder docstring."""
    return IndexedTensor(_ContractionPart(lib.PLAIDML_COMBO_OP_COND, (lhs, rhs, true_case)))


def cos(x):
    """Computes the elementwise cosine of ``x``.

        Args:
            x (Tensor): The tensor used to peform the elementwise ``cos``.

        Returns:
            y (Tensor): The result of the elementwise ``cos`` operation.

    """
    return call('cos', x)


def cosh(x):
    """Computes the elementwise hyperbolic cosine of ``x``.

        Args:
            x (Tensor): The tensor used to peform the elementwise ``cosh``.

        Returns:
            y (Tensor): The result of the elementwise ``cosh`` operation.

    """
    return call('cosh', x)


def exp(x):
    """Computes the elementwise natural exponential function of ``x``.

        Args:
            x (Tensor): The tensor used to peform the elementwise ``exp``.

        Returns:
            y (Tensor): The result of the elementwise ``exp`` operation.

    """
    return call('exp', x)


def floor(x):
    """Computes the elementwise floor of ``x``.

        Args:
            x (Tensor): The tensor used to peform the elementwise ``floor``.

        Returns:
            y (Tensor): The result of the elementwise ``floor`` operation.

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
            r (Tensor): The result of the ``gather`` operation.

    """
    return call('gather', x, y)


def gradients(loss, variables):
    """A placeholder docstring."""
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
            r (Tensor): The resultant tensor.

    """
    return call('ident', x)


def index(x, axis):
    """Returns the index of ``x`` at the specified ``axis``.

        Args:
            x (Tensor): The Tensor to index.
            axis (Tensor): The axis used for indexing.

        Returns:
            r (Tensor): The indexed tensor.

    """
    return call('index', x, axis)


def log(x):
    """Computes the elementwise natural logarithm of ``x``.

        Args:
            x (Tensor): The input Tensor.

        Returns:
            r (Tensor): The resultant tensor.

    """
    return call('log', x)


def max(x, y):
    """Computes the elementwise maximum of ``x`` and ``y``.

        Args:
            x (Tensor): The first input Tensor.
            y (Tensor): The second input Tensor.

        Returns:
            r (Tensor): The resultant tensor.

    """
    return call('max', x, y)


def min(x, y):
    """Computes the elementwise minimum of ``x`` and ``y``.

        Args:
            x (Tensor): The first input Tensor.
            y (Tensor): The second input Tensor.

        Returns:
            r (Tensor): The resultant tensor.

    """
    return call('min', x, y)


def pow(x, y):
    """Computes the elementwise ``y``th power of ``x``.

        Args:
            x (Tensor): The base Tensor.
            y (Tensor): The exponent Tensor.

        Returns:
            r (Tensor): The resultant tensor.

    """
    return call('pow', x, y)


def prng(state, shape):
    """Generates a Tensor of elementwise pseudorandom numbers using the seed values specified in ``state``.

        Args:
            state (Tensor): The seed values for the ``prng`` operation.
            shape (Tensor): The desired shape of the tensor of pseudorandom numbers.

        Returns:
            y (Tensor): The tensor of pseudorandom numbers.

    """
    return call('prng', state, *shape)


def reshape(x, dims):
    """Takes a tensor ``x`` and reshapes it according to ``dims``.

        Args:
            x (Tensor): The tensor to reshape.
            dims (list): The desired shape of the tensor.

        Returns:
            y (Tensor): The reshaped tensor.

    """
    return call('reshape', x, *dims)


def round(x):
    """Rounds ``x`` elementwise.

        Args:
            x (Tensor): The tensor to round.

        Returns:
            y (Tensor): The rounded tensor.

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
           r (Tensor): The scattered tensor.

    """
    return call('scatter', x, y, z)


def select(cond, true_case, false_case):
    """Performs an elementwise conditional which returns the corresponding
       element in ``true_case`` if the condition is evaluated to be true or the
       corresponding element in ``false_case`` if the condition is evaluated to be
       false.

       Args:
           cond (Tensor): The tensor used to perform the conditional.
           true_case (Tensor): The tensor whose elements are selected if the condition evaluates to be true.
           false_case (Tensor): The tensor whose elements are selected if the condition evaluates to be false.

       Returns:
           y (Tensor): The tensor with the conditionally selected elements.

    """
    return call('cond', cond, true_case, false_case)


def shape(x):
    """Returns the shape of ``x`` as a Tensor.

        Args:
            x (Tensor): The tensor used to calculate the shape.

        Returns:
            y (Tensor): The shape of the tensor.

    """
    return call('shape', x)


def sin(x):
    """Computes the elementwise sine of ``x``.

        Args:
            x (Tensor): The tensor used to peform the elementwise ``sin``.

        Returns:
            y (Tensor): The result of the elementwise ``sin`` operation.

    """
    return call('sin', x)


def sinh(x):
    """Computes the elementwise hyperbolic sine of ``x``.

        Args:
            x (Tensor): The tensor used to peform the elementwise ``sinh``.

        Returns:
            y (Tensor): The result of the elementwise ``sinh`` operation.

    """
    return call('sin', x)


def sqrt(x):
    """Computes the elementwise square root of ``x``.

        Args:
            x (Tensor): The tensor used to peform the elementwise ``sqrt``.

        Returns:
            y (Tensor): The result of the elementwise ``sqrt`` operation.

    """
    return call('sqrt', x)


def tan(x):
    """Computes the elementwise tangent of ``x``.

        Args:
            x (Tensor): The tensor used to peform the elementwise ``tan``.

        Returns:
            y (Tensor): The result of the elementwise ``tan`` operation.

    """
    return call('tan', x)


def tanh(x):
    """Computes the elementwise hyperbolic tangent of ``x``.

        Args:
            x (Tensor): The tensor used to peform the elementwise ``tanh``.

        Returns:
            y (Tensor): The result of the elementwise ``tanh`` operation.

    """
    return call('tanh', x)
