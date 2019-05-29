from collections import namedtuple

from plaidml._edsl import NativeObject, decode_str, ffi, ffi_call, lib


class TensorShape(NativeObject):
    __ffi_del__ = lib.tile_shape_free
    __ffi_repr__ = lib.tile_shape_repr

    def __init__(self, dtype_or_ptr, sizes=[], strides=None):
        if isinstance(dtype_or_ptr,
                      ffi.CData) and ffi.typeof(dtype_or_ptr) is ffi.typeof('tile_shape*'):
            ffi_obj = dtype_or_ptr
        else:
            ffi_obj = ffi_call(lib.tile_shape_alloc, dtype_or_ptr)
            if strides is None:
                strides = []
            if len(strides) != len(sizes):
                stride = 1
                for i in range(len(sizes) - 1, -1, -1):
                    strides.insert(0, stride)
                    stride *= sizes[i]
            for (size, stride) in zip(sizes, strides):
                ffi_call(lib.tile_shape_add_dimension, ffi_obj, size, stride)
        super(TensorShape, self).__init__(ffi_obj)

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

    def __neg__(self):
        if self.size is None:
            raise ValueError('Undefined dimension')
        return TensorDim(-self.size)

    def __add__(self, rhs):
        return self.__binary_op(rhs, lambda x, y: x + y)

    def __radd__(self, lhs):
        return self.__binary_op(rhs, lambda x, y: y + x)

    def __sub__(self, rhs):
        return self.__binary_op(rhs, lambda x, y: x - y)

    def __rsub__(self, lhs):
        return self.__binary_op(rhs, lambda x, y: y - x)

    def __mul__(self, rhs):
        return self.__binary_op(rhs, lambda x, y: x * y)

    def __rmul__(self, lhs):
        return self.__binary_op(rhs, lambda x, y: y * x)

    def __div__(self, rhs):
        return self.__binary_op(rhs, lambda x, y: x // y)

    def __rdiv__(self, lhs):
        return self.__binary_op(rhs, lambda x, y: y // x)

    def __floordiv__(self, rhs):
        return self.__binary_op(rhs, lambda x, y: x // y)

    def __rfloordiv__(self, lhs):
        return self.__binary_op(rhs, lambda x, y: y // x)

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
    print('poly_op: ', args, flush=True)

    def wrap(x):
        if isinstance(x, int):
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
            expr = ffi_call(lib.tile_poly_expr_index, ffi.new_handle(self), name.encode())
        super(TensorIndex, self).__init__(expr)

    def __lt__(self, rhs):
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


class _TensorSpec(NativeObject):
    __ffi_del__ = lib.tile_expr_free
    __ffi_repr__ = lib.tile_expr_repr

    def __init__(self, ref, key, dims):
        if isinstance(key, tuple) or isinstance(key, list):
            idxs = key
        else:
            idxs = [key]

        def wrap_idx(x):
            if isinstance(x, int):
                return ffi_call(lib.tile_poly_expr_literal, x)
            return x.as_ptr()

        def wrap_dim(x):
            if isinstance(x, int):
                return x
            return x.size

        idxs = [wrap_idx(x) for x in idxs]
        if dims is None:
            dims = ffi.NULL
        else:
            dims = [wrap_dim(x) for x in dims]
        expr = ffi_call(lib.tile_expr_tensor_spec, ref.as_ptr(), len(idxs), idxs, dims)
        super(_TensorSpec, self).__init__(expr)


class _Contraction(NativeObject):
    __ffi_del__ = lib.tile_expr_free
    __ffi_repr__ = lib.tile_expr_repr

    def __init__(self,
                 agg_op,
                 combo_op,
                 output,
                 inputs,
                 constraints,
                 no_defract=False,
                 use_default=None):
        if use_default is None:
            use_default = ffi.NULL
        inputs = [x.as_ptr() for x in inputs]
        expr = ffi_call(
            lib.tile_expr_contraction,
            agg_op,
            combo_op,
            output.as_ptr(),
            len(inputs),
            inputs,
            len(constraints),
            constraints,
            no_defract,
            use_default,
        )
        super(_Contraction, self).__init__(expr)


_ContractionPart = namedtuple('_ContractionPart', ['op', 'args'])


class IndexedTensor(object):

    def __init__(self, impl, tensor=None):
        self._impl = impl
        self._tensor = tensor

    # Represents an aggregation_op of SUM in a contraction
    def __iadd__(self, rhs):
        print('__iadd__()', flush=True)
        return IndexedTensor(self._make_contraction(lib.TILE_AGG_OP_SUM, rhs))

    # Represents an aggregation_op of PROD in a contraction
    def __imul__(self, rhs):
        print('__imul__()', flush=True)
        return IndexedTensor(self._make_contraction(lib.TILE_AGG_OP_PROD, rhs))

    # Represents an aggregation_op of MAX in a contraction
    def __ge__(self, rhs):
        print('__ge__()', flush=True)
        self._tensor._set_contraction(self._make_contraction(lib.TILE_AGG_OP_MAX, rhs))

    # Represents an aggregation_op of MIN in a contraction
    def __le__(self, rhs):
        print('__le__()', flush=True)
        self._tensor._set_contraction(self._make_contraction(lib.TILE_AGG_OP_MIN, rhs))

    # Represents a combo_op of PLUS in a contraction
    def __add__(self, rhs):
        print('__add__()', flush=True)
        return IndexedTensor(_ContractionPart(lib.TILE_COMBO_OP_ADD, (self, rhs)))

    # Represents a combo_op of MULTIPLY in a contraction
    def __mul__(self, rhs):
        print('__mul__()', flush=True)
        return IndexedTensor(_ContractionPart(lib.TILE_COMBO_OP_MUL, (self, rhs)))

    # Represents a combo_op of EQ in a contraction
    def __eq__(self, rhs):
        print('__eq__()', flush=True)
        return IndexedTensor(_ContractionPart(lib.TILE_COMBO_OP_EQ, (self, rhs)))

    def _make_contraction(self, agg_op, rhs):
        print('make_contraction({}, {}, {})'.format(agg_op, self, rhs), flush=True)
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
        return _Contraction(agg_op, combo_op, self._impl, inputs, [])


class Tensor(NativeObject):
    __ffi_del__ = lib.tile_expr_free
    __ffi_repr__ = lib.tile_expr_repr

    _dims = None
    _is_contraction = False
    _shape = None

    def __init__(self, value, name=''):
        print('Tensor:', value, flush=True)
        self._name = name
        if isinstance(value, ffi.CData) and ffi.typeof(value) is ffi.typeof('tile_expr*'):
            expr = value
        elif isinstance(value, TensorShape):
            self._shape = value
            expr = ffi_call(lib.tile_expr_param, value.as_ptr(), name.encode())
        elif isinstance(value, tuple) or isinstance(value, list):
            self._dims = value
            expr = None
        else:
            raise ValueError('Unknown type')
        super(Tensor, self).__init__(expr)

    def __getitem__(self, key):
        print('__getitem__({})'.format(key), flush=True)
        return IndexedTensor(_TensorSpec(self, key, self._dims), tensor=self)

    def __setitem__(self, key, value):
        print('__setitem__({}, {})'.format(key, value), flush=True)
        if isinstance(value._impl, _Contraction):
            # standard contraction
            self._set_contraction(value._impl)
        elif isinstance(value._impl, _TensorSpec):
            # ASSIGN contraction
            self._set_contraction(
                _Contraction(
                    lib.TILE_AGG_OP_ASSIGN,
                    lib.TILE_COMBO_OP_NONE,
                    _TensorSpec(self, key, self._dims),
                    [value._impl],
                    [],
                ))
        else:
            raise ValueError('Invalid impl')

    def _set_contraction(self, cion):
        self._is_contraction = True
        self.take_ptr(cion)

    # Represents an eltwise negation
    def __neg__(self):
        return Tensor(call('neg', self))

    # Represents an eltwise bit_not
    def __invert__(self):
        return Tensor(call('bit_not', self))

    # Represents an eltwise addition
    def __add__(self, rhs):
        return Tensor(call('add', self, rhs))

    # Represents an eltwise subtraction
    def __sub__(self, rhs):
        return Tensor(call('sub', self, rhs))

    # Represents an eltwise multiplication
    def __mul__(self, rhs):
        return Tensor(call('mul', self, rhs))

    # Represents an eltwise division
    def __div__(self, rhs):
        return Tensor(call('div', self, rhs))

    # Represents an eltwise division
    def __truediv__(self, rhs):
        return Tensor(call('div', self, rhs))

    # Represents an eltwise cmp_eq
    def __eq__(self, rhs):
        return Tensor(call('cmp_eq', self, rhs))

    # Represents an eltwise cmp_ne
    def __ne__(self, rhs):
        return Tensor(call('cmp_ne', self, rhs))

    # Represents an eltwise cmp_lt
    def __lt__(self, rhs):
        return Tensor(call('cmp_lt', self, rhs))

    # Represents an eltwise cmp_gt
    def __gt__(self, rhs):
        return Tensor(call('cmp_gt', self, rhs))

    # Represents an eltwise cmp_le
    def __le__(self, rhs):
        return Tensor(call('cmp_le', self, rhs))

    # Represents an eltwise cmp_ge
    def __ge__(self, rhs):
        return Tensor(call('cmp_ge', self, rhs))

    # Represents an eltwise bit_left
    def __lshift__(self, rhs):
        return Tensor(call('bit_left', self, rhs))

    # Represents an eltwise bit_right
    def __rshift__(self, rhs):
        return Tensor(call('bit_right', self, rhs))

    # Represents an eltwise bit_and
    def __and__(self, rhs):
        return Tensor(call('bit_and', self, rhs))

    # Represents an eltwise bit_or
    def __or__(self, rhs):
        return Tensor(call('bit_or', self, rhs))

    # Represents an eltwise bit_xor
    def __xor__(self, rhs):
        return Tensor(call('bit_xor', self, rhs))

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
            self._shape = TensorShape(ffi_call(lib.tile_expr_evaluate_shape, self.as_ptr()))
        return self._shape

    # Return the size of the tensor's shape at the specified dimension.
    # def dims(self, dim):
    #     return 0

    # Verify that the specified dims match the dims of this tensor.
    def bind_dims(self, *dims):
        print('bind_dims:', dims)
        if self._dims is not None:
            # this handles intermediate temporaries (results of previous outputs)
            sizes = [x.size for x in self._dims]
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
    return Tensor(args)


def TensorDims(count):
    return [TensorDim() for i in range(count)]


def TensorIndexes(count):
    return [TensorIndex() for i in range(count)]


def call(fn, *args):
    print('call: {}({})'.format(fn, args), flush=True)

    def wrap(x):
        if isinstance(x, int):
            return ffi_call(lib.tile_expr_int, x)
        if isinstance(x, float):
            return ffi_call(lib.tile_expr_float, x)
        return x.as_ptr()

    args = [wrap(x) for x in args]
    return ffi_call(lib.tile_expr_call, fn.encode(), len(args), args)


def select(cond, true_case, false_case):
    return Tensor(call('cond', cond, true_case, false_case))


def exp(x):
    return Tensor(call('exp', x))
