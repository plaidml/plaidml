# Copyright Vertex.AI.
"""
TILE program construction utilities.

When writing a program in TILE, you typically specify a graph of operations,
where each operation is defined by a single TILE function.  You then use
composition to connect the individual TILE functions into a single composite
TILE function, which can then be compiled, scheduled, and executed as a single
unit.  The PlaidML API provides functions for building up and using these
composite TILE programs.

Defining the individual per-operation TILE functions is sometimes trivial,
and sometimes not.  For example: although TILE makes it very easy to write
a matrix multiply operation, the details of how that operations is expressed
in TILE vary depending on the number of dimensions involved, whether
broadcasting is required, &c.  Higher-level frameworks tend to expect their
backends to have a single "Do a matrix multiply" operation that's supposed
to internally figure out all of these details; that's not really something
that can be done in TILE directly.

It's wasteful and error-prone to implement these sometimes-tricky conversions
from high-level semantics to low-level TILE code for each framework.  And
the frameworks tend to have similar semantics, thanks to the influence of
Numpy.  So PlaidML provides:

  * A standard high-level operation library for constructing the per-operation
    TILE functions and composing them together (this module)

  * Utilities to assist in constructing operations, such as broadcasting logic
    (this module)

  * A suite of operations that we've found to be useful across a variety of
    frameworks (the `plaidml.tile.op` module).

This library uses two passes for building up composite functions.  The first
pass constructs Python objects representing the operation graph; the second
pass translates the operation graph to the composite TILE function.  This is
done for two reasons: it allows for higher-level optimizations (e.g.
translating particular subtrees to more efficient TILE operations) and for
expressing operations that cannot be efficiently implemented in the current
version of TILE (e.g. it's very expensive to implement ArgMax in the initial
released version of TILE, but ArgMax is typically used in composite
expressions like Equal(ArgMax(X), ArgMax(Y)), which is trivial to efficiently
implement in TILE).

More precisely, this library builds up a bipartite directed acyclic graph of
`Operation` and `Value` objects.  `Operation` is the base class of each
operation; `Value` represents an operation input or output.  `compose`
translates an operation graph into a `plaidml.Function`.

See [The Tile Tutorial](https://github.com/plaidml/plaidml/wiki/Tile-Tutorial)
for more information about how the TILE language works and details for writing
your own operations.
"""
from collections import namedtuple
import functools
import six

import plaidml


class Error(Exception):
    """Errors raised during TILE function composition."""
    pass


class LogicError(Error):
    """Logic errors on the part of the caller."""
    pass


Source = namedtuple('Source', ['op', 'name'])


class Shape(namedtuple('Shape', ['dtype', 'dims'])):
    """Represents a symbolic tensor shape."""
    __slots__ = ()

    @property
    def ndims(self):
        """The shape's dimensionality.

        Returns:
            int: The number of dimensions in the shape.
        """
        return len(self.dims)


class _OpBindings(object):
    """A mapping of bindings discovered during an operation binding traversal."""
    def __init__(self, ctx, dev):
        """Initialize the `_OpBindings`.

        Args:
            ctx (plaidml.Context): The context for the binding traversal.
        """
        self.ctx = ctx
        self.dev = dev
        self._bindings = dict()

    def lookup(self, operation):
        """Looks up an operation's output bindings within this traversal.

        Args:
            operation (Operation): The operation producing the outputs.

        Returns:
            str -> plaidml.Var: The operation's bound outputs.
        """
        try:
            return self._bindings[operation]
        except KeyError:
            six.raise_from(
                LogicError('Unbound operation encountered during operation composition'), None)

    def insert(self, operation, outputs):
        """Adds an operation's output bindings.

        Args:
            operation (Operation): The operation that's been bound.
            outputs (str -> plaidml.Var): The operation's output bindings.
        """
        self._bindings[operation] = outputs

    def is_bound(self, operation):
        """Indicates whether an operation's already been bound.

        Args:
            operation (Operation): The operation to look up.

        Returns:
            bool: True iff the operation's already been bound.
        """
        return operation in self._bindings


class Operation(object):
    """Operation base class."""
    def __init__(self, code, inputs, outputs):
        """`Operation` constructor.

        Most `Operation`s are defined via subclasses whose constructors
        build up the TILE code to implement the requested operation and
        then invoke this constructor to initialize the underlying
        `Operation` functionality.  This makes it simpler for binding-time
        code generation to recognize and transform chunks of the operation
        graph (since they can just check that operations are subclasses
        of known derived classes).

        Args:
            code (string): The code implementing the operation, or None.
            inputs ([(string, Value)]): Operation inputs.
            outputs ([(string, Shape)]): Operation outputs.
        """
        self.code = code
        self.inputs = dict(inputs)
        output_list = [(name, Value.for_op(shape, self, name)) for name, shape in outputs]
        self.output_tuple = tuple([val for _, val in output_list])
        self.outputs = dict(output_list)

    @classmethod
    def function(cls, *args, **kwargs):
        """Invokes an `Operation` as a function.

        When processing the operation graph, it's useful for each `Operation`
        type to be its own class, enabling operation-specific behaviors.  But
        it's awkward to use a class as a function.  This classmethod
        instantiates an `Operation` subclass using the supplied inputs, and
        returns the operation's outputs as a tuple of `Value`s (or as a single
        `Value` for single-output operations).

        Args:
            args: The operation constructor positional arguments.
            kwargs: The operation constructor keyword arguments.

        Raises:
            LogicError: If invoked on Operation instead of on an Operation subclass.

        Returns:
            tuple(Value): The operation outputs, in operation-defined order.
        """
        if cls is Operation:
            raise LogicError(
                'Operation.function is defined on subclasses of Operation, not Operation itself.')
        operation = cls(*args, **kwargs)
        if len(operation.output_tuple) == 1:
            return operation.output_tuple[0]
        return operation.output_tuple

    def bind(self, bindings):
        """Builds an output variable dictionary for the operation.

        N.B. Subclasses may override this method in order to implement optimizations and
        composite operations that aren't easily described in TILE.

        Args:
            bindings (_OpBindings): The previously-computed output bindings.

        Returns:
            str -> plaidml.Var: The bound outputs for this operation.  The caller is responsible
                                for adding this to the known output bindings; this is typically
                                called by _OpBindings.__missing__, which does this automatically.
        """
        if not self.code:
            raise NotImplementedError(
                '{} is not directly implemented.'.format(self.__class__.__name__))
        applier = plaidml.Applier(bindings.ctx, plaidml.Function(self.code))
        for name, input_value in self.inputs.items():
            applier.add_input(name, input_value.bind(bindings))
        outputs = {}
        for name in self.outputs.keys():
            outputs[name] = applier.add_output(name)
        return outputs


class Value(object):
    """A PlaidML variable and associated metadata."""
    def __init__(self, shape, var, source):
        """Constructs a `Value`.

        This isn't typically used directly; instead, use one of the `Value`
        classmethods.

        Args:
            shape (Shape): The `Value`'s symbolic shape.
            var (plaidml.Var): The PlaidML variable backing this `Value`, or None.
            source (Source): The source of the `Value`, or None.
        """
        if (var is None) and (not source):
            raise LogicError('Either a variable or a variable source must be supplied')
        if not isinstance(shape.dims, tuple):
            shape = Shape(shape.dtype, tuple(shape.dims))
        self.shape = shape
        self.var = var
        self.source = source

    def __repr__(self):
        return '<tile.Value {}{}>'.format(str(self.shape.dtype).split('.')[-1], self.shape.dims)

    @staticmethod
    def for_op(shape, operation, name):
        """Builds an operation output value.

        Args:
            shape (Shape): The symbolic shape of the operation output.
            operation (Operation): The operation producing the output.
            name (str): The name of the operation output.

        Returns:
            Value: The operation output.
        """
        return Value(shape, None, Source(operation, name))

    @staticmethod
    def from_ndims(ndims, dtype=plaidml.DType.FLOAT32):
        """Builds an N-dimensional placeholder value.

        The resulting `Value`'s shape will contain `Value` instances that will
        be computed at binding time from the actual dimensions of the bound
        tensor.

        Args:
            ndims (int): The number of dimensions.
            dtype (plaidml.DType): The element datatype.

        Returns:
            Value: The placeholder value.
        """
        return Value.from_var(plaidml.Placeholder(ndims), [None] * ndims, dtype)

    @staticmethod
    def from_dimensions(dimensions, dtype=plaidml.DType.FLOAT32):
        """Builds an N-dimensional placeholder value from a list of dimension sizes.

        `None` elements in the dimension list will be replaced by `Value` instances
        that will be computed at binding time from the actual dimensions of the bound
        tensor.

        Args:
            dimensions (tuple or list): The size of each dimension.
            dtype (plaidml.DType): The element datatype.

        Returns:
            Value: The placeholder value.
        """
        return Value.from_var(plaidml.Placeholder(len(dimensions)), dimensions, dtype)

    @staticmethod
    def from_var(var, dimensions, dtype=plaidml.DType.FLOAT32):
        """Builds a value from a PlaidML variable.

        `None` elements in the dimension list will be replaced by `Value` instances
        that will be computed at binding time from the actual dimensions of the bound
        tensor.

        Args:
            var (plaidml.Var): The variable to be wrapped by the Value.
            dimensions (tuple or list): The size of each dimension.
            dtype (plaidml.DType): The element datatype.

        Returns:
            Value: The wrapped value.
        """
        ndims = len(dimensions)
        dims = [Value._filldim(var, ndims, idx, dim) for idx, dim in enumerate(dimensions)]
        return Value(Shape(dtype, tuple(dims)), var, None)

    @staticmethod
    def _filldim(var, ndims, idx, dim):
        if dim is not None:
            return dim
        return Value._dim(var, ndims, idx)

    @staticmethod
    def _dim(var, ndims, idx):
        """The symbolic size a dimension of the supplied variable.

        Args:
            var (plaidml.Var): The var whose dimension we're getting.
            ndims (int): The total number of dimensions.
            idx (int): The 0-based index of the dimension to get.

        Returns:
            Value: The size of dimension `idx` of `var`.
        """
        code = 'function (I[{dims}]) -> (O) {{ O = D{idx}; }}'.format(
            dims=','.join(['D{}'.format(idx) for idx in range(ndims)]),
            idx=str(idx)
        )
        shape = Shape(plaidml.DType.UINT64, tuple())
        operation = Operation(code, [('I', var)], [('O', shape)])
        return operation.outputs['O']

    def bind(self, bindings):
        """Translates the `Value` to a PlaidML variable.

        Args:
            bindings (_OpBindings): The previously-computed output bindings.

        Returns:
            plaidml.Var: The variable representing this `Value`.
        """
        if self.var:
            return self.var
        outputs = bindings.lookup(self.source.op)
        return outputs[self.source.name]

    def is_bound(self, bindings):
        """Indicates whether the `Value` has been bound.

        Args:
            bindings (_OpBindings): The bindings to check.

        Returns:
            bool: True iff the `Value` is a concrete PlaidML variable, or if its source operation
                  is already bound in `bindings`.
        """
        return self.var or (self.source and bindings.is_bound(self.source.op))

    # Python numeric type methods.  These allow Value objects to be used in
    # ordinary expressions, returning derived Values.

    @staticmethod
    def _binary_op(lhs, rhs, op_format_str):
        """Builds a Value for an elementwise binary operation.

        Args:
            lhs (Value or int): The left-hand side of the operation.
            rhs (Value or int): The right-hand side of the operation.
            op_format_str (str): The format string to use for the operation.
                                 The format string should have two substitutions:
                                 'L' and 'R', for the LHS and RHS.

        Returns:
            Value: A Value representing the result of the operation.
        """
        if isinstance(lhs, Value):
            if isinstance(rhs, Value):
                input_str = 'L, R'
                op_str = op_format_str.format(L='L', R='R')
                inputs = [('L', lhs), ('R', rhs)]
                shape = Shape(common_dtype(lhs.shape.dtype, rhs.shape.dtype),
                              broadcast_dims(lhs.shape.dims, rhs.shape.dims))
            else:
                input_str = 'L'
                op_str = op_format_str.format(L='L', R=rhs)
                inputs = [('L', lhs)]
                shape = lhs.shape
        else:
            # rhs must be a Value, and this must be a reflected operation.
            input_str = 'R'
            op_str = op_format_str.format(L=lhs, R='R')
            inputs = [('R', rhs)]
            shape = rhs.shape

        operation = Operation(
            'function ({input_str}) -> (O) {{ O = {op_str}; }}'.format(
                input_str=input_str, op_str=op_str),
            inputs,
            [('O', shape)])

        return operation.output_tuple[0]

    @staticmethod
    def _unary_op(value, op_str):
        """Builds a Value for an elementwise unary operation.

        Args:
            value (Value): The operation input.
            op_str (str): The format string to use for the operation.
                          The format string should be an expression in terms of 'I'.

        Returns:
            Value: A Value representing the result of the operation.
        """
        operation = Operation(
            'function (I) -> (O) {{ O = {op_str}; }}'.format(op_str=op_str),
            [('I', value)], [('O', value.shape)])

        return operation.output_tuple[0]

    def __abs__(self):
        return Value._unary_op(self, 'I < 0 ? -I : I')

    def __add__(self, other):
        return Value._binary_op(self, other, '{L} + {R}')

    def __radd__(self, other):
        return Value._binary_op(other, self, '{L} + {R}')

    def __and__(self, other):
        return Value._binary_op(self, other, '{L} & {R}')

    def __rand__(self, other):
        return Value._binary_op(other, self, '{L} & {R}')

    def __div__(self, other):
        return Value._binary_op(self, other, '{L} / {R}')

    def __rdiv__(self, other):
        return Value._binary_op(other, self, '{L} / {R}')

    def __floordiv__(self, other):
        return Value._binary_op(self, other, 'floor({L} / {R})')

    def __rfloordiv__(self, other):
        return Value._binary_op(other, self, 'floor({L} / {R})')

    def __invert__(self):
        return Value._unary_op(self, '~I')

    def __lshift__(self, other):
        return Value._binary_op(self, other, '{L} << {R}')

    def __rlshift__(self, other):
        return Value._binary_op(other, self, '{L} << {R}')

    def __mul__(self, other):
        return Value._binary_op(self, other, '{L} * {R}')

    def __rmul__(self, other):
        return Value._binary_op(other, self, '{L} * {R}')

    def __neg__(self):
        return Value._unary_op(self, '-I')

    def __or__(self, other):
        return Value._binary_op(self, other, '{L} | {R}')

    def __ror__(self, other):
        return Value._binary_op(other, self, '{L} | {R}')

    def __pos__(self):
        return Value._unary_op(self, 'I')

    def __rshift__(self, other):
        return Value._binary_op(self, other, '{L} >> {R}')

    def __rrshift__(self, other):
        return Value._binary_op(other, self, '{L} >> {R}')

    def __sub__(self, other):
        return Value._binary_op(self, other, '{L} - {R}')

    def __rsub__(self, other):
        return Value._binary_op(other, self, '{L} - {R}')

    def __truediv__(self, other):
        return Value._binary_op(self, other, '{L} / {R}')

    def __rtruediv__(self, other):
        return Value._binary_op(other, self, '{L} / {R}')

    def __xor__(self, other):
        return Value._binary_op(self, other, '{L} ^ {R}')

    def __rxor__(self, other):
        return Value._binary_op(other, self, '{L} ^ {R}')


def compose(ctx, dev, inputs, outputs, updates=None):
    """Builds a TILE Function that computes the indicated values.

    Args:
        ctx (plaidml.Context): The context to use for building the function.
        dev (plaidml.Device): The device used to build the function (where constants will live)
        inputs ([(name, Value)]): A list of named input placeholders.
        outputs ([(name, Value)]): A list of named output values.
        updates ([(original, updated)]): A list of updates to perform (side-effects).

    Returns:
        plaidml.Invoker: The composed TILE function.
    """
    bindings = _OpBindings(ctx, dev)
    to_be_bound = [val for _, val in outputs]
    if updates:
        to_be_bound.extend((val for val in update for update in updates))
    to_be_bound.extend((val for _, val in inputs))
    while to_be_bound:
        current = to_be_bound.pop()
        if current.is_bound(bindings):
            continue
        op = current.source.op
        reqs = [v for v in op.inputs.values() if not v.is_bound(bindings)]
        if reqs:
            to_be_bound.append(current)
            to_be_bound.extend(reqs)
            continue
        bindings.insert(op, op.bind(bindings))

    composer = plaidml.Composer()
    for (name, val) in inputs:
        composer.add_input(name, val.bind(bindings))
    for (name, val) in outputs:
        binding = val.bind(bindings)
        composer.add_output(name, binding)
    if updates:
        for (original, updated) in updates:
            composer.add_update(original.bind(bindings), updated.bind(bindings))

    return composer.build()


_DTypeInfo = namedtuple('_DTypeInfo', ['base', 'width'])


_DTYPE_INFOS = {
    plaidml.DType.BOOLEAN: _DTypeInfo(base='b', width=1),
    plaidml.DType.INT8: _DTypeInfo(base='i', width=1),
    plaidml.DType.INT16: _DTypeInfo(base='i', width=2),
    plaidml.DType.INT32: _DTypeInfo(base='i', width=4),
    plaidml.DType.INT64: _DTypeInfo(base='i', width=8),
    plaidml.DType.UINT8: _DTypeInfo(base='u', width=1),
    plaidml.DType.UINT16: _DTypeInfo(base='u', width=2),
    plaidml.DType.UINT32: _DTypeInfo(base='u', width=4),
    plaidml.DType.UINT64: _DTypeInfo(base='u', width=8),
    plaidml.DType.FLOAT16: _DTypeInfo(base='f', width=2),
    plaidml.DType.FLOAT32: _DTypeInfo(base='f', width=4),
    plaidml.DType.FLOAT64: _DTypeInfo(base='f', width=8),
}

_INFO_DTYPES = dict([[v, k] for k, v in _DTYPE_INFOS.items()])


def common_dtype(*args):
    """Finds the common dtype of a set of dtypes.

    Args:
        args ([plaidml.DType]): The list of dtypes to be considered.

    Returns:
        plaidml.DType: The smallest dtype whose range encompasses the ranges of the supplied dtypes.
    """
    best = _DTypeInfo(base='b', width=1)
    for dtype in args:
        current = _DTYPE_INFOS[dtype]
        if best.base != current.base:
            if best.base == 'b':
                best = current
            elif current.base == 'b':
                # Just use whatever we have so far; booleans can be coerced to anything.
                pass
            elif best.base == 'f' or current.base == 'f':
                # We're unifying some integer type with a float.  The float needs to be
                # at least twice the size of the integer type, clamped to float64.
                best_width = best.width if best.base == 'f' else best.width * 2
                current_width = current.width if current.base == 'f' else current.width * 2
                width = max(best_width, current_width)
                if width > 8:
                    width = 8
                best = _DTypeInfo(base='f', width=width)
            else:
                # We're unifying an 'i' with a 'u'.  The 'u' can be held in an 'i' twice
                # the width; if that pushes us up to width=16, use a float64.
                best_width = best.width if best.base == 'i' else best.width * 2
                current_width = current.width if current.base == 'i' else current.width * 2
                width = max(best_width, current_width)
                if width > 8:
                    best = _DTypeInfo(base='f', width=8)
                else:
                    best = _DTypeInfo(base='i', width=width)
    return _INFO_DTYPES[best]


def broadcast_dims(*args):
    """Computes the broadcast dimensions of the supplied dimensions.

    Args:
        args ([dims]): The list of dimension tuples to be broadcast.

    Returns:
        tuple(Value): The broadcasted dims tuple.
    """
    dtuples = args
    result_dcount = max((len(dtuple) for dtuple in dtuples))

    def make_binding_broadcast(sizes):
        """Builds a bind-time Value for the broadcast of the supplied sizes.

        Args:
            sizes ([int or Value]): The sizes being broadcast.

        Returns:
            Value: The broadcasted size.
        """
        vsizes = [sz for sz in sizes if isinstance(sz, Value)]
        vsize_strs = ['I{}'.format(idx) for idx in range(len(vsizes))]
        isizes = [sz for sz in sizes if not isinstance(sz, Value)]
        code = 'function ({var_sizes}) -> (O) {{ O = broadcast({sizes}); }}'.format(
            var_sizes=', '.join(vsize_strs),
            sizes=', '.join(vsize_strs + isizes))
        shape = Shape(plaidml.DType.UINT64, tuple())
        operation = Operation(code, zip(vsize_strs, vsizes), [('O', shape)])
        return operation.outputs['O']

    def make_axis(rdim_idx):
        """Builds a single axis of a broadcast output.

        Args:
            rdim_idx (int): The reversed index of the dimension within the broadcast output
                            dimensions: zero corresponds to the last (greatest index) element
                            of the tuple, result_dcount-1 corresponds to the first (0-index)
                            element of the tuple.  (This is used because for dimension tuples,
                            the 0-index corresponds to the greatest-stride element, while
                            broadcasting aligns the smallest-stride elements).

        Raises:
            LogicError: If the dimensions are incompatible on this dimension.

        Returns:
            int or Value: The output dimension size.
        """
        sizes = []
        for dtuple in dtuples:
            if len(dtuple) <= rdim_idx:
                size = 1
            else:
                size = dtuple[len(dtuple) - rdim_idx - 1]
            sizes.append(size)
        sizes = [sz for sz in sizes if isinstance(sz, Value) or sz != 1]
        if not sizes:
            return 1
        if len(sizes) == 1:
            return sizes[0]
        size = None
        for this_size in sizes:
            if isinstance(this_size, Value):
                # This broadcast can only be computed at binding time.
                return make_binding_broadcast(sizes)
            if size and size != this_size:
                raise LogicError(
                    'Broadcast mismatch: {} and {} are incompatible; inputs were: {}'.format(
                        this_size, size,
                        ', '.join(['({})'.format(
                            ', '.join(str(dim) for dim in dtuple)) for dtuple in dtuples])))
            size = this_size
        return size

    return tuple([make_axis(result_dcount - dim_idx - 1) for dim_idx in range(result_dcount)])


def compute_aggregation_axes(dims, axes=None, keepdims=False):
    """Computes parameters for an aggregation-over-axes operation.

    Args:
        dims ([int or Value]): The dimensions of the value being aggregated.
        axes ([int], optional): Defaults to None. The indices of the axes to aggregate over.
        keepdims (bool, optional): Defaults to False. Iff true, keep the aggregated axes in the
                                   output.

    Returns:
        (dims, axes, dict(string->string)): The resulting dimensions and axes, and a dictionary of
                                            formatting replacements to use when building the TILE
                                            operation.
    """
    if axes is None:
        axes = len(dims) - 1
    if isinstance(axes, list) or isinstance(axes, tuple):
        axes = [(len(dims) + i if i < 0 else i) for i in axes]
    else:
        if axes < 0:
            axes = len(dims) + axes
        axes = [axes]
    axes.sort(reverse=True)
    src_indices = ['x' + str(i) for i in range(len(dims))]
    src_ranges = ['X' + str(i) for i in range(len(dims))]
    dest_indices = src_indices[:]
    dest_ranges = src_ranges[:]
    dims = list(dims)
    if keepdims:
        for axis in axes:
            dest_indices[axis] = 's' + dest_indices[axis]
            dest_ranges[axis] = '1'
            dims[axis] = 1
    else:
        for axis in axes:
            del dest_indices[axis]
            del dest_ranges[axis]
            del dims[axis]

    return tuple(dims), axes, {
        'src_indices': ', '.join(src_indices),
        'src_ranges': ', '.join(src_ranges),
        'src_sep': ' : ' if src_indices else '',
        'dest_indices': ', '.join(dest_indices),
        'dest_ranges': ', '.join(dest_ranges),
        'dest_sep': ' : ' if dest_indices else '',
    }
