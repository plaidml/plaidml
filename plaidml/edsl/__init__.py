from collections import namedtuple

import six

from plaidml.edsl._ffi import NativeObject, decode_str, ffi, ffi_call, lib


class TensorShape(NativeObject):
    __ffi_del__ = lib.tile_shape_free
    __ffi_repr__ = lib.tile_shape_repr

    def __init__(self, dtype=None, sizes=[], strides=None, ptr=None, layout=''):
        if ptr:
            ffi_obj = ptr
        elif dtype is not None:
            ffi_obj = ffi_call(lib.tile_shape_alloc, dtype, layout.encode())
            if strides is None:
                strides = []
            if len(strides) != len(sizes):
                stride = 1
                for i in range(len(sizes) - 1, -1, -1):
                    strides.insert(0, stride)
                    stride *= sizes[i]
            for (size, stride) in zip(sizes, strides):
                ffi_call(lib.tile_shape_add_dimension, ffi_obj, size, stride)
        else:
            raise ValueError('One of dtype= or ptr= must be specified.')
        super(TensorShape, self).__init__(ffi_obj)

    @property
    def type(self):
        return ffi_call(lib.tile_shape_get_type, self.as_ptr())

    @property
    def rank(self):
        return ffi_call(lib.tile_shape_get_rank, self.as_ptr())

    @property
    def sizes(self):
        return [
            ffi_call(lib.tile_shape_get_dimension_size, self.as_ptr(), i) for i in range(self.rank)
        ]

    @property
    def strides(self):
        return [
            ffi_call(lib.tile_shape_get_dimension_stride, self.as_ptr(), i)
            for i in range(self.rank)
        ]


class Constraint(object):

    def __bool__(self):
        return True


class TensorDim(object):

    def __init__(self, size=None):
        self.size = size

    def __repr__(self):
        if self.size is None:
            return 'None'
        return str(self.size)

    def __neg__(self):
        if self.size is None:
            raise ValueError('Undefined dimension')
        return TensorDim(-self.size)

    def __add__(self, other):
        return self.__binary_op(other, lambda x, y: x + y)

    def __radd__(self, other):
        return self.__binary_op(other, lambda x, y: y + x)

    def __sub__(self, other):
        return self.__binary_op(other, lambda x, y: x - y)

    def __rsub__(self, other):
        return self.__binary_op(other, lambda x, y: y - x)

    def __mul__(self, other):
        return self.__binary_op(other, lambda x, y: x * y)

    def __rmul__(self, other):
        return self.__binary_op(other, lambda x, y: y * x)

    def __div__(self, other):
        return self.__binary_op(other, lambda x, y: x // y)

    def __rdiv__(self, other):
        return self.__binary_op(other, lambda x, y: y // x)

    def __floordiv__(self, other):
        return self.__binary_op(other, lambda x, y: x // y)

    def __rfloordiv__(self, other):
        return self.__binary_op(other, lambda x, y: y // x)

    def __binary_op(self, other, fn):
        if self.size is None:
            raise ValueError('Undefined dimension')
        if isinstance(other, TensorDim):
            if other.size is None:
                raise ValueError('Undefined dimension')
            size = other.size
        else:
            size = other
        return TensorDim(fn(self.size, size))


def poly_op(op, *args):

    def wrap(x):
        if isinstance(x, six.integer_types):
            return ffi_call(lib.tile_poly_expr_literal, x)
        if isinstance(x, TensorDim):
            if x.size is None:
                raise ValueError('Undefined dimension')
            return ffi_call(lib.tile_poly_expr_literal, x.size)
        return x.as_ptr()

    args = [wrap(x) for x in args]
    return ffi_call(lib.tile_poly_expr_op, op, len(args), args)


class TensorIndex(NativeObject):
    __ffi_del__ = lib.tile_poly_expr_free
    __ffi_repr__ = lib.tile_poly_expr_repr

    def __init__(self, expr=None, name=''):
        if expr is None:
            expr = ffi_call(lib.tile_poly_expr_index, name.encode())
        super(TensorIndex, self).__init__(expr)

    def __lt__(self, rhs):
        if isinstance(rhs, TensorDim):
            rhs = rhs.size
        ffi_call(lib.tile_poly_expr_add_constraint, self.as_ptr(), rhs)
        return Constraint()

    def __neg__(self):
        return TensorIndex(poly_op(lib.TILE_POLY_OP_NEG, self))

    def __add__(self, rhs):
        return TensorIndex(poly_op(lib.TILE_POLY_OP_ADD, self, rhs))

    def __radd__(self, lhs):
        return TensorIndex(poly_op(lib.TILE_POLY_OP_ADD, lhs, self))

    def __sub__(self, rhs):
        return TensorIndex(poly_op(lib.TILE_POLY_OP_SUB, self, rhs))

    def __rsub__(self, lhs):
        return TensorIndex(poly_op(lib.TILE_POLY_OP_SUB, lhs, self))

    def __mul__(self, rhs):
        return TensorIndex(poly_op(lib.TILE_POLY_OP_MUL, self, rhs))

    def __rmul__(self, lhs):
        return TensorIndex(poly_op(lib.TILE_POLY_OP_MUL, lhs, self))

    def __div__(self, rhs):
        return TensorIndex(poly_op(lib.TILE_POLY_OP_DIV, self, rhs))

    def __rdiv__(self, lhs):
        return TensorIndex(poly_op(lib.TILE_POLY_OP_DIV, lhs, self))

    def __floordiv__(self, rhs):
        return TensorIndex(poly_op(lib.TILE_POLY_OP_DIV, self, rhs))

    def __rfloordiv__(self, lhs):
        return TensorIndex(poly_op(lib.TILE_POLY_OP_DIV, lhs, self))


def _wrap_dim(x):
    if isinstance(x, six.integer_types):
        return x
    return x.size


class _TensorSpec(NativeObject):
    __ffi_del__ = lib.tile_expr_free
    __ffi_repr__ = lib.tile_expr_repr

    def __init__(self, ref, key, dims):
        if isinstance(key, tuple) or isinstance(key, list):
            idxs = key
        else:
            idxs = [key]

        def wrap_idx(x):
            if isinstance(x, six.integer_types):
                return ffi_call(lib.tile_poly_expr_literal, x)
            return x.as_ptr()

        idxs = [wrap_idx(x) for x in idxs]
        if dims is None:
            dims = ffi.NULL
        else:
            dims = [_wrap_dim(x) for x in dims]
        expr = ffi_call(lib.tile_expr_tensor_spec, ref.as_ptr(), len(idxs), idxs, dims)
        super(_TensorSpec, self).__init__(expr)


class _Contraction(NativeObject):
    __ffi_del__ = lib.tile_expr_free
    __ffi_repr__ = lib.tile_expr_repr

    def __init__(self, agg_op, combo_op, output, inputs, name):
        inputs = [x.as_ptr() for x in inputs]
        expr = ffi_call(
            lib.tile_expr_contraction,
            agg_op,
            combo_op,
            output.as_ptr(),
            len(inputs),
            inputs,
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
        return IndexedTensor(self._make_contraction(lib.TILE_AGG_OP_SUM, rhs))

    # Represents an aggregation_op of PROD in a contraction
    def __imul__(self, rhs):
        return IndexedTensor(self._make_contraction(lib.TILE_AGG_OP_PROD, rhs))

    # Represents an aggregation_op of MAX in a contraction
    def __ge__(self, rhs):
        self._tensor._set_contraction(self._make_contraction(lib.TILE_AGG_OP_MAX, rhs))

    # Represents an aggregation_op of MIN in a contraction
    def __le__(self, rhs):
        self._tensor._set_contraction(self._make_contraction(lib.TILE_AGG_OP_MIN, rhs))

    # Represents a combo_op of PLUS in a contraction
    def __add__(self, rhs):
        return IndexedTensor(_ContractionPart(lib.TILE_COMBO_OP_ADD, (self, rhs)))

    # Represents a combo_op of MULTIPLY in a contraction
    def __mul__(self, rhs):
        return IndexedTensor(_ContractionPart(lib.TILE_COMBO_OP_MUL, (self, rhs)))

    # Represents a combo_op of EQ in a contraction
    def __eq__(self, rhs):
        return IndexedTensor(_ContractionPart(lib.TILE_COMBO_OP_EQ, (self, rhs)))

    def _make_contraction(self, agg_op, rhs):
        # Extract combo_op and inputs
        if isinstance(rhs._impl, _TensorSpec):
            # Unary op
            combo_op = lib.TILE_COMBO_OP_NONE
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
            self._impl,
            inputs,
            self._tensor._name,
        )


class Tensor(NativeObject):
    __ffi_del__ = lib.tile_expr_free
    __ffi_repr__ = lib.tile_expr_repr

    _dims = None
    _shape = None
    _is_contraction = False

    def __init__(self, shape=None, dims=None, expr=None, value=None, name=''):
        self._name = name
        if shape:
            self._shape = shape
            expr = ffi_call(lib.tile_expr_param, shape.as_ptr(), name.encode())
        elif dims is not None:
            self._dims = dims
            expr = None
        elif value is not None:
            if isinstance(value, six.integer_types):
                expr = ffi_call(lib.tile_expr_int, value)
            elif isinstance(value, float):
                expr = ffi_call(lib.tile_expr_float, value)
            else:
                raise TypeError('Invalid type for value={}'.format(value))
        elif expr is None:
            raise ValueError('One of dims=, shape=, or expr= must be specified.')
        super(Tensor, self).__init__(expr)

    def __hash__(self):
        return hash((self.as_ptr(), self._dims, self._shape, self._is_contraction))

    def __getitem__(self, key):
        return IndexedTensor(_TensorSpec(self, key, self._dims), tensor=self)

    def __setitem__(self, key, value):
        if isinstance(value._impl, _Contraction):
            # standard contraction
            self._set_contraction(value._impl)
        elif isinstance(value, Tensor):
            pass
        elif isinstance(value._impl, _TensorSpec):
            # ASSIGN contraction
            self._set_contraction(
                _Contraction(
                    lib.TILE_AGG_OP_ASSIGN,
                    lib.TILE_COMBO_OP_NONE,
                    _TensorSpec(self, key, self._dims),
                    [value._impl],
                    self._name,
                ))
        else:
            raise ValueError('Invalid impl')

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

    # Enable no_defract on a contraction
    def no_defract(self):
        if not self._is_contraction:
            raise TypeError('no_defract can only be specified on a contraction.')
        ffi_call(lib.tile_expr_contraction_set_no_defract, self.as_ptr(), True)
        return self

    # Set use_default on a contraction
    def use_default(self, rhs):
        if not self._is_contraction:
            raise TypeError('use_default can only be specified on a contraction.')
        ffi_call(lib.tile_expr_contraction_set_use_default, self.as_ptr(), rhs.as_ptr())
        return self

    # Return the tensor's shape
    def shape(self):
        if self._shape is None:
            self._shape = TensorShape(ptr=ffi_call(lib.tile_expr_evaluate_shape, self.as_ptr()))
        return self._shape

    # Return the size of the tensor's shape at the specified dimension.
    # def dims(self, dim):
    #     return 0

    # Verify that the specified dims match the dims of this tensor.
    def bind_dims(self, *dims):
        if self._dims is not None:
            # this handles intermediate temporaries (results of previous outputs)
            sizes = [_wrap_dim(x) for x in self._dims]
        else:
            # this is the fallback which handles user inputs and any other case
            sizes = self.shape().sizes
        if len(dims) != len(sizes):
            raise RuntimeError('bind_dims() mismatch. Tensor shape: {}, dims: {}'.format(
                len(sizes), len(dims)))
        for i in range(len(dims)):
            if dims[i].size is None:
                dims[i].size = sizes[i]
            elif dims[i].size != sizes[i]:
                raise RuntimeError(
                    'bind_dims() mismatch on dim {}. Required: {}, Actual: {}'.format(
                        i, dims[i].size, sizes[i]))


def TensorOutput(*args):
    return Tensor(dims=args)


def TensorDims(count):
    return [TensorDim() for i in range(count)]


def TensorIndexes(count):
    return [TensorIndex() for i in range(count)]


class Program(NativeObject):
    __ffi_del__ = lib.tile_program_free
    __ffi_repr__ = lib.tile_program_repr

    def __init__(self, name, *vars):
        exprs = [x.as_ptr() for x in vars]
        ffi_obj = ffi_call(lib.tile_program_evaluate, name.encode(), len(exprs), exprs)
        super(Program, self).__init__(ffi_obj)


def call(fn, *args):

    def wrap(x):
        if isinstance(x, six.integer_types):
            return ffi_call(lib.tile_expr_int, x)
        if isinstance(x, float):
            return ffi_call(lib.tile_expr_float, x)
        if isinstance(x, Tensor):
            return x.as_ptr()
        raise TypeError('Unexpected type for call argument: {}. args: {}'.format(x, args))

    args = [wrap(x) for x in args]
    return Tensor(expr=ffi_call(lib.tile_expr_call, fn.encode(), len(args), args))


def as_float(x, bit_size):
    return call("as_float", x, bit_size)


def as_int(x, bit_size):
    return call("as_int", x, bit_size)


def as_uint(x, bit_size):
    return call("as_uint", x, bit_size)


# def element(x) : return call("element", {x}) # TODO: tuple


def cond(lhs, rhs, true_case):
    return IndexedTensor(_ContractionPart(lib.TILE_COMBO_OP_COND, (lhs, rhs, true_case)))


def cos(x):
    return call("cos", x)


def exp(x):
    return call("exp", x)


def gather(x, y):
    return call("gather", x, y)


def index(x, axis):
    return call("index", x, axis)


def log(x):
    return call("log", x)


def pow(x, y):
    return call("pow", x, y)


def prng_state(x):
    return call("prng_state", x)


def prng_step(x, sizes):
    return call("prng_step", x, *sizes)


def prng_value(x):
    return call("prng_value", x)


def reshape(x, shape):
    return call("reshape", x, *shape.sizes)


def scatter(x, y, z):
    return call("scatter", x, y, z)


def select(cond, true_case, false_case):
    return call("cond", cond, true_case, false_case)


def shape(x):
    return call("shape", x)


def sigmoid(x):
    return call("sigmoid", x)


def sin(x):
    return call("sin", x)


def sqrt(x):
    return call("sqrt", x)


def tan(x):
    return call("tan", x)


def tanh(x):
    return call("tanh", x)
