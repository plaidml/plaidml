# Copyright 2018 Intel Corporation.
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
    frameworks (the :doc:`plaidml.op` module).

This library uses two passes for building up composite functions.  The first
pass constructs Python objects representing the operation graph; the second
pass translates the operation graph to the composite TILE function.  This is
done for two reasons: it allows for higher-level optimizations (e.g.
translating particular subtrees to more efficient TILE operations) and for
expressing operations that cannot be efficiently implemented in the current
version of TILE (e.g. it's very expensive to implement ArgMax in the initial
released version of TILE, but ArgMax is typically used in composite
expressions like ``Equal(ArgMax(X), ArgMax(Y))``, which is trivial to
efficiently implement in TILE).

More precisely, this library builds up a bipartite directed acyclic graph of
``Operation`` and ``Value`` objects.  ``Operation`` is the base class of each
operation; ``Value`` represents an operation input or output.  ``compose``
translates an operation graph into a ``plaidml.Function``.

See the `Tile Tutorial <https://github.com/plaidml/plaidml/wiki/Tile-Tutorial>`_
for more information about how the TILE language works, or check out the
`PlaidML Op Tutorial <https://github.com/plaidml/plaidml/wiki/PlaidML-Op-Tutorial>`_
for the details of writing your own operations.
"""
import functools
import logging
import math
import re
import sys
import traceback
from collections import namedtuple

import numpy as np
import six

import plaidml


class Error(Exception):
    """Errors raised during TILE function composition."""
    pass


class LogicError(Error):
    """Logic errors on the part of the caller."""
    pass


NUMPY_DTYPE_TO_PLAIDML = {
    'bool': plaidml.DType.BOOLEAN,
    'float16': plaidml.DType.FLOAT16,
    'float32': plaidml.DType.FLOAT32,
    'float64': plaidml.DType.FLOAT64,
    'int8': plaidml.DType.INT8,
    'int16': plaidml.DType.INT16,
    'int32': plaidml.DType.INT32,
    'int64': plaidml.DType.INT64,
    'uint8': plaidml.DType.UINT8,
    'uint16': plaidml.DType.UINT16,
    'uint32': plaidml.DType.UINT32,
    'uint64': plaidml.DType.UINT64,
}

PLAIDML_DTYPE_TO_NUMPY = dict([[v, k] for k, v in NUMPY_DTYPE_TO_PLAIDML.items()])


def convert_np_dtype_to_pml(dtype):
    if isinstance(dtype, np.dtype):
        dtype = str(dtype)
    if isinstance(dtype, plaidml.DType):
        return dtype
    try:
        return NUMPY_DTYPE_TO_PLAIDML[dtype]
    except KeyError:
        raise ValueError("Unrecognized Numpy dtype {}".format(dtype))


def convert_pml_dtype_to_np(dtype):
    try:
        return PLAIDML_DTYPE_TO_NUMPY[dtype]
    except KeyError:
        raise ValueError("Unrecognized PlaidML dtype {}".format(dtype))


Source = namedtuple('Source', ['op', 'output_name'])


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

    def __init__(self, code, inputs, outputs, name=None, side_effects=None):
        """`Operation` constructor.

        Most `Operation`s are defined via subclasses whose constructors
        build up the TILE code to implement the requested operation and
        then invoke this constructor to initialize the underlying
        `Operation` functionality.  This makes it simpler for binding-time
        code generation to recognize and transform chunks of the operation
        graph (since they can just check that operations are subclasses
        of known derived classes).

        Operations may have side-effects -- tensors that are to be updated
        as a side-effect of evaluating the operation.  These are supplied as
        a list of `(variable, new_value)` tuples, where both `variable` and
        `new_value` are `Value` objects.  These will be wrapped up in the
        function returned by `compose()`.

        Args:
            code (str): The code implementing the operation, or None.
            inputs ([(str, Value)]): Operation inputs.
            outputs ([(str, Shape)]): Operation outputs.
            name (str): A name for this operation, or None.
            side_effects ([Value, Value]): A dict of side-effects of this operation.
        """
        self.code = code
        self.inputs = dict([
            (k, _ShapelessValue.from_value(Value.from_python_value(v))) for k, v in inputs
        ])
        output_list = [
            (output_name, Value.for_op(shape, self, output_name)) for output_name, shape in outputs
        ]
        self.output_tuple = tuple([val for _, val in output_list])
        self.outputs = dict(output_list)
        self.name = name or self.__class__.__name__
        self.side_effects = side_effects or []
        if plaidml.is_backtrace_enabled():
            self.backtrace = ''.join(traceback.format_stack()[:-1])
        else:
            self.backtrace = None

    def sole_output(self):
        if len(self.output_tuple) != 1:
            raise LogicError(
                'Sole output requested from operation {}; operation has multiple outputs ({})' \
                .format(self.name, ', '.join(self.outputs.keys())))
        return self.output_tuple[0]

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
            raise NotImplementedError('{} is not directly implemented.'.format(
                self.__class__.__name__))
        safe_name = self.name.replace('/', '_')
        func = plaidml.Function(self.code, backtrace=self.backtrace, fid=safe_name)
        applier = plaidml.Applier(bindings.ctx, func)
        for input_name, input_value in self.inputs.items():
            applier.add_input(input_name, input_value.bind(bindings))
        outputs = {}
        for output_name in self.outputs.keys():
            try:
                outputs[output_name] = applier.add_output(output_name)
            except BaseException as e:
                raise Exception('Failed to add output \'{}\' in op {}: {}; code={}'.format(
                    output_name, self.name, e.message, self.code))
        return outputs


def unary_op(value, op_str, name=None):
    """Builds a Value for an elementwise unary operation.

    Args:
        value (Value): The operation input.
        op_str (str): The string to use for the operation.
                      The string should be an expression in terms of 'I'.
        name (str): The name of the operation, or None.

    Returns:
        Value: A Value representing the result of the operation.
    """
    operation = Operation('function (I) -> (O) {{ O = {}; }}'.format(op_str), [('I', value)],
                          [('O', value.shape)],
                          name=name)

    return operation.sole_output()


def binary_op(lhs, rhs, op_str, dtype=None, name=None):
    """Builds a Value for an elementwise binary operation.

    Args:
        lhs (Value or numeric): The left-hand side of the operation.
        rhs (Value or numeric): The right-hand side of the operation.
        op_str (str): The string to use for the operation.
                      The string should be an expression in terms of 'L' and 'R'.
        dtype (plaidml.DType): If not None, supplies the operation dtype;
                               otherwise, the common dtype of lhs and rhs is used.
        name (str): The name of the operation, or None.

    Returns:
        Value: A Value representing the result of the operation.
    """
    lhs = Value.from_python_value(lhs)
    rhs = Value.from_python_value(rhs)

    shape = Shape(common_dtype(lhs.shape.dtype, rhs.shape.dtype),
                  broadcast_dims(lhs.shape.dims, rhs.shape.dims))

    if dtype:
        shape = Shape(dtype, shape.dims)

    operation = Operation('function (L, R) -> (O) {{ O = {}; }}'.format(op_str), [('L', lhs),
                                                                                  ('R', rhs)],
                          [('O', shape)],
                          name=name)

    return operation.sole_output()


def maximum(x, y):
    if isinstance(x, Value) or isinstance(y, Value):
        return binary_op(x, y, 'max(L, R)', name='Maximum')
    else:
        return max(x, y)


def minimum(x, y):
    if isinstance(x, Value) or isinstance(y, Value):
        return binary_op(x, y, 'min(L, R)', name='Minimum')
    else:
        return min(x, y)


class ShapeOf(Operation):
    """
    Computes the shape of a supplied tensor.

    (N.B. This is in tile.py instead of in op.py solely because it's useful
          to be able to reference it from Value.__getitem__().  For future-proofing,
          frameworks should reference this class as op.ShapeOf.)
    """

    def __init__(self, x):
        self.source = x
        super(ShapeOf, self).__init__('function (I) -> (O) { O = shape(I); }', [('I', x)],
                                      [('O', Shape(plaidml.DType.INT32, (x.shape.ndims,)))])


shape_of = ShapeOf.function


class _SliceOf(Operation):
    """
    Computes a subslice of a supplied tensor.
    """

    def __init__(self, value, key):

        if isinstance(key, slice) or isinstance(key, int) or isinstance(key, type(Ellipsis)):
            key = (key,)
        if not isinstance(key, tuple):
            raise ValueError('Cannot index Values using type {}'.format(type(key)))
        if key.count(Ellipsis) > 1:
            raise ValueError('Cannot use multiple ellipses in a slice (given {})'.format(key))

        var_list = list()
        dim_list = list()
        formula_list = list()
        offset_list = list()
        dims = list()
        inner_idx = 0
        extra_vars = []
        try:
            ellipsis_idx = key.index(Ellipsis)
        except ValueError:
            ellipsis_idx = None
        if ellipsis_idx is not None:
            ellipsis_length = value.shape.ndims - len(key) + 1
            if ellipsis_length < 0:
                raise ValueError('Slice key too long. Tensor has {} dimensions, key is {}'.format(
                    value.shape.ndims, key))
            key = tuple(
                list(key[:ellipsis_idx]) + [slice(None, None, None)] * ellipsis_length +
                list(key[ellipsis_idx + 1:]))
        for idx in range(len(key)):
            length_numerator, length_numerator_value, step, offset, idx_extra_vars = self._parse_slice(
                value.shape.dims, key, idx)
            extra_vars.extend(idx_extra_vars)
            if step == None:
                # In this case offset is an int
                if offset >= 0:
                    formula_list.append('{}'.format(offset))
                else:
                    offset_list.append('Offset{} = N{}+{};'.format(idx, idx, offset))
                    formula_list.append('{}'.format('Offset{}'.format(idx)))
            else:
                var_list.append('i{}'.format(inner_idx))
                dim_subs = {'numer': length_numerator, 'step': step}
                if step > 0:
                    dim_list.append('({numer} + {step} - 1)/{step}'.format(**dim_subs))
                else:
                    dim_list.append('({numer} + {step} + 1)/{step}'.format(**dim_subs))
                if isinstance(length_numerator, str):
                    dims.append(unary_op(length_numerator_value / step, 'ceil(I)', 'Ceiling'))
                    offset_list.append('Offset{} = {};'.format(idx, offset))
                    formula_list.append('{}*i{}+{}'.format(step, inner_idx,
                                                           'Offset{}'.format(idx)))
                else:
                    dims.append(int(math.ceil(float(length_numerator) / step)))
                    formula_list.append('{}*i{}+{}'.format(step, inner_idx, offset))
                inner_idx += 1

        # Separately handle extra indices not sliced over
        for idx in range(len(key), value.shape.ndims):
            var_list.append('i{}'.format(inner_idx))
            dim_list.append('N{}'.format(idx))
            dims.append(value.shape.dims[idx])
            formula_list.append('i{}'.format(inner_idx))
            inner_idx += 1
        dims = tuple(dims)

        if len(dims) == 0:
            body = 'O[] = =(I[' + ', '.join(formula_list) + ']);'
        else:
            body = 'O[{}: {}] = =(I[{}]);'.format(', '.join(var_list), ', '.join(dim_list),
                                                  ', '.join(formula_list))

        # TODO: Example below is out of date, although it shows the spirit of the op
        # Example 'code' (slicing X[5:10,3,:,2:6:2]):
        #   function (I[N0, N1, N2, N3]) -> (O) {
        #     O[i0, i1, i2: 5, N2, 2] = +(I[i0+5, 3, i1, 2*i2+2]);
        #   }
        prefix = '\n                   '
        code = """
               function (I[{indims}]) -> (O) {{
                   {extra_vars}{offsets}{body}
               }}""".format(indims=', '.join(['N{}'.format(i) for i in range(value.shape.ndims)]),
                            extra_vars=''.join(v + prefix for v in extra_vars),
                            offsets=''.join(o + prefix for o in offset_list),
                            body=body)

        super(_SliceOf, self).__init__(code, [('I', value)],
                                       [('O', Shape(value.shape.dtype, dims))],
                                       name='SliceOf')
        self.key = key

    @staticmethod
    def _parse_slice(dims, key, idx):
        extra_vars = []
        if isinstance(key[idx], int):
            return 1, 1, None, key[idx], extra_vars

        def check(val):
            if isinstance(val, six.integer_types):
                return
            if val is None:
                return
            raise ValueError('Must use ints when slicing; received {} of type {}'.format(
                val, val.__class__.__name__))

        check(key[idx].start)
        check(key[idx].stop)
        check(key[idx].step)

        step = 1 if key[idx].step is None else key[idx].step
        if step == 0:
            raise ValueError('Cannot slice with step size 0')

        start = key[idx].start
        if start == None:
            if step > 0:
                start = 0
            else:
                start = -1

        start_value = start

        if start < 0:
            if isinstance(dims[idx], Value):
                start = 'N{} + {}'.format(idx, start)
            else:
                start = dims[idx] + start
            start_value = dims[idx] + start_value

        if step > 0:
            if isinstance(dims[idx], Value):
                extra_vars.append('Start{idx} = max({start}, 0);'.format(start=start, idx=idx))
                start = 'Start{}'.format(idx)
            else:
                start = maximum(start, 0)
            start_value = maximum(start_value, 0)
        else:
            if isinstance(dims[idx], Value):
                extra_vars.append('Start{idx} = min({start}, N{idx} - 1);'.format(start=start,
                                                                                  idx=idx))
                start = 'Start{}'.format(idx)
            else:
                start = minimum(start, dims[idx] - 1)
            start_value = minimum(start_value, dims[idx] - 1)

        stop = key[idx].stop
        if stop == None:
            if step > 0:
                if isinstance(dims[idx], Value):
                    stop = 'N{}'.format(idx)
                else:
                    stop = dims[idx]
                stop_value = dims[idx]
            else:
                stop = -1
                stop_value = -1
            # Can return now and skip unneeded max/min
            if isinstance(dims[idx], Value):
                return '({} - ({}))'.format(
                    stop, start), stop_value - start_value, step, start, extra_vars
            return stop - start, stop_value - start_value, step, start, extra_vars

        stop_value = stop

        if stop < 0:
            if isinstance(dims[idx], Value):
                stop = 'N{} + {}'.format(idx, stop)
            else:
                stop = dims[idx] + stop
            stop_value = dims[idx] + stop_value

        if step > 0:
            if isinstance(dims[idx], Value):
                extra_vars.append('Stop{idx} = min({stop}, N{idx});'.format(stop=stop, idx=idx))
                stop = 'Stop{}'.format(idx)
            else:
                stop = minimum(stop, dims[idx])
            stop_value = minimum(stop_value, dims[idx])
        else:
            if isinstance(dims[idx], Value):
                extra_vars.append('Stop{idx} = max({stop}, -1);'.format(stop=stop, idx=idx))
                stop = 'Stop{}'.format(idx)
            else:
                stop = maximum(stop, -1)
            stop_value = maximum(stop_value, -1)

        if isinstance(dims[idx], Value):
            length_numerator = '({} - ({}))'.format(stop, start)
        else:
            length_numerator = stop - start
        return length_numerator, stop_value - start_value, step, start, extra_vars


class _NDArray(Operation):
    """
    An operation that builds a value from a Numpy ndarray.
    """

    def __init__(self, value):
        # TODO: Consider copying the value if it's writeable.
        self._value = value
        shape = Shape(convert_np_dtype_to_pml(value.dtype.name), tuple(value.shape))
        super(_NDArray, self).__init__(None, [], [('O', shape)], name='NDArray')

    def bind(self, bindings):
        tensor = plaidml.Tensor(
            bindings.dev,
            plaidml.Shape(bindings.ctx, convert_np_dtype_to_pml(self._value.dtype.name),
                          *self._value.shape))
        with tensor.mmap_discard(bindings.ctx) as view:
            view.copy_from_ndarray(self._value)
            view.writeback()

        return {'O': tensor}


class _ShapelessValue(object):
    """
    Wraps a PlaidML variable, without shape information.

    Symbolic shape information requires a reference back to the underlying shaped
    variable.  To support this without creating cycles in the operation graph,
    `Operation` objects reference `_ShapelessValue` instances as their inputs.
    `Value` then inherits from `_ShapelessValue` in order to augment the value
    with shape information.
    """

    def __init__(self, var, source, name=None):
        if (var is None) and (not source):
            raise LogicError('Either a variable or a variable source must be supplied')
        self.var = var
        self.source = source
        self._name = name

    @staticmethod
    def from_value(value):
        return _ShapelessValue(value.var, value.source, value._name)

    def __str__(self):
        vcls = ' ' + self.var.__class__.__name__ if self.var else ''
        return '{}{}'.format(self.name, vcls)

    def __repr__(self):
        return '<tile._ShapelessValue {}>'.format(self)

    @property
    def name(self):
        if self._name:
            return self._name
        if self.source:
            if len(self.source.op.outputs) == 1:
                return self.source.op.name
            return self.source.op.name + '/' + self.source.output_name
        return '<unnamed>'

    @property
    def key(self):
        """A dictionary key that unifies a Value with its shapeless slices."""
        return (self.var, self.source)

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
        return outputs[self.source.output_name]

    def is_bound(self, bindings):
        """Indicates whether the `Value` has been bound.

        Args:
            bindings (_OpBindings): The bindings to check.

        Returns:
            bool: True iff the `Value` is a concrete PlaidML variable, or if its source operation
                  is already bound in `bindings`.
        """
        return self.var or (self.source and bindings.is_bound(self.source.op))


class Value(_ShapelessValue):
    """A PlaidML variable and associated metadata."""

    def __init__(self, shape, var, source, name=None):
        """Constructs a `Value`.

        This isn't typically used directly; instead, use one of the `Value`
        classmethods.

        Args:
            shape (Shape): The `Value`'s symbolic shape.
            var (plaidml.Var): The PlaidML variable backing this `Value`, or None.
            source (Source): The source of the `Value`, or None.
            name (str): A mnemonic name for the `Value`, or None.
        """
        if not isinstance(shape.dims, tuple):
            shape = Shape(shape.dtype, tuple(shape.dims))
        self.shape = shape
        # created to replicate the _uses_learning_phase attribute in Keras, defaulting to False
        self._uses_learning_phase = False
        super(Value, self).__init__(var, source, name)

    def __str__(self):
        vcls = ' ' + self.var.__class__.__name__ if self.var else ''
        return '{}{} {}{}'.format(self.name, vcls,
                                  str(self.shape.dtype).split('.')[-1], self.shape.dims)

    def __repr__(self):
        return '<tile.Value {}>'.format(self)

    def __getitem__(self, key):
        if self.source and isinstance(self.source.op, ShapeOf):
            return self.source.op.source.shape.dims.__getitem__(key)

        # Otherwise, we need to perform the slice as a TILE operation.
        return _SliceOf.function(self, key)

    @staticmethod
    def for_op(shape, operation, output, name=None):
        """Builds an operation output Value.

        Args:
            shape (Shape): The symbolic shape of the operation output.
            operation (Operation): The operation producing the output.
            output (str): The name of the operation output.
            name (str): A mnemonic name for the `Value`, or None.

        Returns:
            Value: The operation output.
        """
        return Value(shape, None, Source(operation, output), name)

    @staticmethod
    def from_ndims(ndims, dtype=plaidml.DType.FLOAT32, name=None):
        """Builds an N-dimensional placeholder Value.

        The resulting `Value`'s shape will contain `Value` instances that will
        be computed at binding time from the actual dimensions of the bound
        tensor.

        Args:
            ndims (int): The number of dimensions.
            dtype (plaidml.DType): The element datatype.
            name (str): A mnemonic name for the `Value`, or None.

        Returns:
            Value: The placeholder value.
        """
        return Value.from_var(plaidml.Placeholder(ndims), [None] * ndims, dtype, name)

    @staticmethod
    def from_dimensions(dimensions, dtype=plaidml.DType.FLOAT32, name=None):
        """Builds an N-dimensional placeholder Value from a list of dimension sizes.

        `None` elements in the dimension list will be replaced by `Value` instances
        that will be computed at binding time from the actual dimensions of the bound
        tensor.

        Args:
            dimensions (tuple or list): The size of each dimension.
            dtype (plaidml.DType): The element datatype.
            name (str): A mnemonic name for the `Value`, or None.

        Returns:
            Value: The placeholder value.
        """
        return Value.from_var(plaidml.Placeholder(len(dimensions)), dimensions, dtype, name)

    @staticmethod
    def from_var(var, dimensions, dtype=plaidml.DType.FLOAT32, name=None):
        """Builds a Value from a PlaidML variable.

        `None` elements in the dimension list will be replaced by `Value` instances
        that will be computed at binding time from the actual dimensions of the bound
        tensor.

        Args:
            var (plaidml.Var): The variable to be wrapped by the Value.
            dimensions (tuple or list): The size of each dimension.
            dtype (plaidml.DType): The element datatype.
            name (str): A mnemonic name for the `Value`, or None.

        Returns:
            Value: The wrapped value.
        """
        ndims = len(dimensions)

        # Create the value with a temporary zero-dimensional shape, so that it can
        # be supplied to Operation instances that calculate its dimensions.
        val = Value(Shape(dtype, tuple()), var, None, name)

        # Create the dimensions list.
        dims = [val._filldim(ndims, idx, dim) for idx, dim in enumerate(dimensions)]

        # Update the Value to have the new shape.
        val.shape = Shape(dtype, tuple(dims))
        return val

    @staticmethod
    def from_python_value(py_val, dtype=None, name=None, ctx=None, dev=None):
        """Builds a Value from a Python value.

        Note: if the context and device are present, the returned value will always be a concrete
        `Value` (wrapping a PlaidML variable, not an `Operation` output).  Otherwise, the returned
        `Value` may be an `Operation` output.

        Args:
            var: A value of a standard Python type.
            dtype (plaidml.DType): The element datatype, or None.
            name (str): A mnemonic name for the `Value`, or None.
            ctx (plaidml.context.Context): The context to use for the variable, or None.
            dev (plaidml.Device): The device to use for the variable, or None.

        Returns:
            Value: The wrapped value.
        """
        if isinstance(py_val, Value):
            return py_val
        elif isinstance(py_val, plaidml.Var):
            return py_val
        elif isinstance(py_val, six.integer_types):
            if dtype is None:
                dtype = plaidml.DType.INT32
            return Value.from_var(plaidml.Integer(py_val), tuple(), dtype, name=name)
        elif isinstance(py_val, float):
            if dtype is None:
                dtype = plaidml.DType.FLOAT32
            return Value.from_var(plaidml.Real(py_val), tuple(), dtype, name=name)
        elif hasattr(py_val, 'shape') and hasattr(py_val, 'dtype'):
            # Assume it's an ndarray.
            if len(py_val.shape) == 0:
                # Handle 0-dimensional numpy arrays as scalars
                return Value.from_python_value(py_val.item())
            if ctx and dev:
                # We have the device; we can return a value immediately.
                tensor = plaidml.Tensor(
                    dev,
                    plaidml.Shape(ctx, convert_np_dtype_to_pml(py_val.dtype.name), *py_val.shape))
                with tensor.mmap_discard(ctx) as view:
                    view.copy_from_ndarray(py_val)
                    view.writeback()
                return Value.from_var(tensor,
                                      py_val.shape,
                                      convert_np_dtype_to_pml(py_val.dtype.name),
                                      name='NDArray')
            # Otherwise, defer the value creation.
            return _NDArray(py_val).sole_output()
        else:
            raise NotImplementedError('Unable to build a Value from a \'{}\' instance'.format(
                py_val.__class__.__name__))

    def _filldim(self, ndims, idx, dim):
        if dim is not None:
            return dim
        return self._dim(ndims, idx)

    def _dim(self, ndims, idx):
        """The symbolic size a dimension of the supplied variable.

        Args:
            ndims (int): The total number of dimensions.
            idx (int): The 0-based index of the dimension to get.

        Returns:
            Value: The size of dimension `idx` of `var`.
        """
        code = 'function (I[{dims}]) -> (O) {{ O = D{idx}; }}'.format(dims=','.join(
            ['D{}'.format(i) for i in range(ndims)]),
                                                                      idx=str(idx))
        shape = Shape(plaidml.DType.UINT64, tuple())
        operation = Operation(code, [('I', self)], [('O', shape)], name='SymbolicDim')
        return operation.outputs['O']

    # Python numeric type methods.  These allow Value objects to be used in
    # ordinary expressions, returning derived Values.

    # Logical operations
    #
    # N.B. We neither define __eq__ nor __ne__, because Value objects are compared for
    #      equality and inequality in a number of contexts, such as "value in some_list".
    #      So we use standard Python object definitions for equality/inequality; callers
    #      that want TILE operations for these should use the operation library's
    #      `equal()` and `not_equal()` functions.

    def __ge__(self, other):
        return binary_op(self, other, 'L >= R', dtype=plaidml.DType.BOOLEAN, name='Ge')

    def __gt__(self, other):
        return binary_op(self, other, 'L > R', dtype=plaidml.DType.BOOLEAN, name='Gt')

    def __le__(self, other):
        return binary_op(self, other, 'L <= R', dtype=plaidml.DType.BOOLEAN, name='Le')

    def __lt__(self, other):
        return binary_op(self, other, 'L < R', dtype=plaidml.DType.BOOLEAN, name='Lt')

    # Arithmetic operations

    def __abs__(self):
        return unary_op(self, 'abs(I)', 'Abs')

    def __add__(self, other):
        if isinstance(other, six.integer_types) and other == 0:
            return self
        if isinstance(other, float) and other == 0.0:
            return self
        return binary_op(self, other, 'L + R', name='Add')

    def __radd__(self, other):
        if isinstance(other, six.integer_types) and other == 0:
            return self
        if isinstance(other, float) and other == 0.0:
            return self
        return binary_op(other, self, 'L + R', name='RevAdd')

    def __and__(self, other):
        return binary_op(self, other, 'L & R', name='And')

    def __rand__(self, other):
        return binary_op(other, self, 'L & R', name='RevAnd')

    def __div__(self, other):
        if isinstance(other, six.integer_types) and other == 1:
            return self
        if isinstance(other, float) and other == 1.0:
            return self
        return binary_op(self, other, 'L / R', name='Div')

    def __rdiv__(self, other):
        return binary_op(other, self, 'L / R', name='RevDiv')

    def __floordiv__(self, other):
        if isinstance(other, six.integer_types) and other == 1:
            return self
        if isinstance(other, float) and other == 1.0:
            return self
        return binary_op(self, other, 'floor(L / R)', name='FloorDiv')

    def __rfloordiv__(self, other):
        return binary_op(other, self, 'floor(L / R)', name='RevFloorDiv')

    def __invert__(self):
        return unary_op(self, '~I', 'Invert')

    def __lshift__(self, other):
        if isinstance(other, six.integer_types) and other == 0:
            return self
        return binary_op(self, other, 'L << R', name='LShift')

    def __rlshift__(self, other):
        return binary_op(other, self, 'L << R', name='RevLShift')

    def __mul__(self, other):
        if isinstance(other, six.integer_types) and other == 1:
            return self
        if isinstance(other, float) and other == 1.0:
            return self
        return binary_op(self, other, 'L * R', name='Mul')

    def __rmul__(self, other):
        if isinstance(other, six.integer_types) and other == 1:
            return self
        if isinstance(other, float) and other == 1.0:
            return self
        return binary_op(other, self, 'L * R', name='RevMul')

    def __neg__(self):
        return unary_op(self, '-I', 'Negate')

    def __or__(self, other):
        return binary_op(self, other, 'L | R', name='Or')

    def __ror__(self, other):
        return binary_op(other, self, 'L | R', name='RevOr')

    def __pos__(self):
        return unary_op(self, 'I', 'Identity')

    def __rshift__(self, other):
        if isinstance(other, six.integer_types) and other == 0:
            return self
        return binary_op(self, other, 'L >> R', name='RShift')

    def __rrshift__(self, other):
        return binary_op(other, self, 'L >> R', name='RevRShift')

    def __sub__(self, other):
        if isinstance(other, six.integer_types) and other == 0:
            return self
        if isinstance(other, float) and other == 0.0:
            return self
        return binary_op(self, other, 'L - R', name='Sub')

    def __rsub__(self, other):
        if isinstance(other, six.integer_types) and other == 0:
            return self
        if isinstance(other, float) and other == 0.0:
            return self
        return binary_op(other, self, 'L - R', name='RevSub')

    def __truediv__(self, other):
        if isinstance(other, six.integer_types) and other == 1:
            return self
        if isinstance(other, float) and other == 1.0:
            return self
        return binary_op(self, other, 'L / R', name='TrueDiv')

    def __rtruediv__(self, other):
        return binary_op(other, self, 'L / R', name='RevTrueDiv')

    def __xor__(self, other):
        return binary_op(self, other, 'L ^ R', name='Xor')

    def __rxor__(self, other):
        return binary_op(other, self, 'L ^ R', name='RevXor')


def compose(ctx, dev, inputs, outputs, updates=None, name='unnamed_function'):
    """Builds a TILE Function that computes the indicated values.

    Args:
        ctx (plaidml.Context): The context to use for building the function.
        dev (plaidml.Device): The device used to build the function (where constants will live)
        inputs ([(name, Value)]): A list of named input placeholders.
        outputs ([(name, Value)]): A list of named output values.
        updates ([(original, updated)]): A list of updates to perform (side-effects).

    Returns:
        plaidml._Function: The composed TILE function.
    """
    logger = logging.getLogger('plaidml')
    logger.debug('compose: {}'.format(name))
    logger.debug('  Inputs:')
    for input in inputs:
        logger.debug('    {}'.format(input))
    logger.debug('  Outputs:')
    for output in outputs:
        logger.debug('    {}'.format(output))
    if updates:
        logger.debug('  Updates:')
        for update in updates:
            logger.debug('    {}'.format(update))
    bindings = _OpBindings(ctx, dev)
    to_be_bound = [val for _, val in outputs]
    if updates is None:
        updates = []
    else:
        for original, updated in updates:
            to_be_bound.append(original)
            to_be_bound.append(updated)
    to_be_bound.extend((val for _, val in inputs))
    while to_be_bound:
        current = to_be_bound.pop()
        if current.is_bound(bindings):
            continue
        op = current.source.op
        vals = list(op.inputs.values())
        for v, nv in op.side_effects:
            vals.append(v)
            vals.append(nv)
        if any(not isinstance(val, _ShapelessValue) for val in vals):
            raise LogicError('Operation {} bound a non-Value; inputs={}, side_effects={}'.format(
                op.name, op.inputs, op.side_effects))
        reqs = [v for v in vals if not v.is_bound(bindings)]
        if reqs:
            to_be_bound.append(current)
            to_be_bound.extend(reqs)
            continue
        bindings.insert(op, op.bind(bindings))
        updates.extend(op.side_effects)

    try:
        composer = plaidml.Composer()
        for (input_name, val) in inputs:
            composer.add_input(input_name, val.bind(bindings))
        for (output_name, val) in outputs:
            binding = val.bind(bindings)
            composer.add_output(output_name, binding)
        for (original, updated) in updates:
            composer.add_update(original.bind(bindings), updated.bind(bindings))

        return composer.build()

    except:
        sys.stderr.writelines(to_dot(inputs, outputs, updates))
        raise


def to_dot(inputs, outputs, updates=None, name='Tile'):
    """Translates a chain of tensor computations to a DOT graph.

    Args:
        inputs ([(name, Value)]): Provides names for computation inputs.
        outputs ([(name, Value)]): The outputs to use for deriving the graph.
        updates ([(original, updates)]): Side-effect updates that are part of the computations.
        name (str): The name to use for the graph, or None.

    Yields:
        The strings comprising the lines of the DOT graph.
    """
    yield 'digraph {} {{\n'.format(name)

    # The general idea:
    #   We're building a bipartite digraph.
    #   Each Value becomes a Box graph node containings its type info.
    #   Each Operation becomes an Oval containing its name
    #   Each connection between Operations and Values is an edge, labeled with
    #     the Value's name relative to the Operation.
    #   We start by building a mapping from the initial graph inputs to their names,
    #     adding all of the output values to a set, and initializing an object-to-name
    #     map to the empty set.
    #   At each step, we remove a value from the set:
    #     If its producing operation has a name, we can continue.
    #     Otherwise, we add:
    #       A node for the operation
    #       Nodes for all unnamed input and output values (adding them to the queue),
    #       Graph edges for all of the operation inputs and outputs.
    #   When the set is empty, we're done.
    #
    #   Since order of values doesn't really matter, we just use a list to maintain the set.

    def value_label(val):
        return re.escape(str(val))

    def op_label(op):
        return re.escape(op.name)

    to_be_processed = []
    names = {}

    def name_generator():
        next_idx = 1
        while True:
            yield 'n' + str(next_idx)
            next_idx += 1

    namegen = name_generator()

    for name, val in inputs:
        dot_name = next(namegen)
        names[val.key] = dot_name
        yield '  {} [label="{}\\n{}" shape=circle];\n'.format(dot_name, name, value_label(val))

    for name, val in outputs:
        dot_name = next(namegen)
        names[val.key] = dot_name
        yield '  {} [label="{}\\n{}" shape=doublecircle];\n'.format(dot_name, name,
                                                                    value_label(val))
        to_be_processed.append(val)

    if updates:
        for original, update in updates:
            original_dot_name = next(namegen)
            names[original.key] = original_dot_name
            yield '  {} [label="{}" shape=circle];\n'.format(original_dot_name,
                                                             value_label(original))
            to_be_processed.append(original)

            update_dot_name = next(namegen)
            names[update.key] = update_dot_name
            yield '  {} [label="{}" shape=circle];\n'.format(update_dot_name, value_label(update))
            to_be_processed.append(update)

            yield '  {} -> {} [style=dotted];\n'.format(update_dot_name, original_dot_name)

    while to_be_processed:
        val = to_be_processed.pop()
        if not val.source:
            continue
        op = val.source.op
        if op in names:
            continue
        op_dot_name = next(namegen)
        names[op] = op_dot_name
        yield '  {} [label="{}" shape=oval];\n'.format(op_dot_name, op_label(op))
        for name, inval in op.inputs.items():
            if inval.key not in names:
                dot_name = next(namegen)
                names[inval.key] = dot_name
                to_be_processed.append(inval)
                yield '  {} [label="{}" shape=box];\n'.format(dot_name, value_label(inval))
            yield '  {} -> {} [label="{}"];\n'.format(names[inval.key], op_dot_name, name)
        for name, outval in op.outputs.items():
            if outval.key not in names:
                dot_name = next(namegen)
                names[outval.key] = dot_name
                to_be_processed.append(outval)
                yield '  {} [label="{}" shape=box];\n'.format(dot_name, value_label(outval))
            yield '  {} -> {} [label="{}"];\n'.format(op_dot_name, names[outval.key], name)

    yield '}\n'


class DTypeInfo(namedtuple('DTypeInfo', ['base', 'width'])):
    """Describes a PlaidML datatype."""
    __slots__ = ()

    @property
    def bitwidth(self):
        """The number of bits occupied by an instance of the type."""
        if self.base == 'bool':
            return 1
        return self.width * 8


DTYPE_INFOS = {
    plaidml.DType.BOOLEAN: DTypeInfo(base='bool', width=1),
    plaidml.DType.INT8: DTypeInfo(base='int', width=1),
    plaidml.DType.INT16: DTypeInfo(base='int', width=2),
    plaidml.DType.INT32: DTypeInfo(base='int', width=4),
    plaidml.DType.INT64: DTypeInfo(base='int', width=8),
    plaidml.DType.UINT8: DTypeInfo(base='uint', width=1),
    plaidml.DType.UINT16: DTypeInfo(base='uint', width=2),
    plaidml.DType.UINT32: DTypeInfo(base='uint', width=4),
    plaidml.DType.UINT64: DTypeInfo(base='uint', width=8),
    plaidml.DType.FLOAT16: DTypeInfo(base='float', width=2),
    plaidml.DType.FLOAT32: DTypeInfo(base='float', width=4),
    plaidml.DType.FLOAT64: DTypeInfo(base='float', width=8),
}

INFO_DTYPES = dict([[v, k] for k, v in DTYPE_INFOS.items()])


def common_dtype(*args):
    """Finds the common dtype of a set of dtypes.

    Args:
        args ([plaidml.DType]): The list of dtypes to be considered.

    Returns:
        plaidml.DType: The smallest dtype whose range encompasses the ranges of the supplied dtypes.
    """
    best = DTypeInfo(base='bool', width=1)
    for dtype in args:
        current = DTYPE_INFOS[dtype]
        if best.base != current.base:
            if best.base == 'bool':
                best = current
            elif current.base == 'bool':
                # Just use whatever we have so far; booleans can be coerced to anything.
                pass
            elif best.base == 'float' or current.base == 'float':
                # We're unifying some integer type with a float.  The float needs to be
                # at least twice the size of the integer type, clamped to float64.
                best_width = best.width if best.base == 'float' else best.width * 2
                current_width = current.width if current.base == 'float' else current.width * 2
                width = max(best_width, current_width)
                if width > 8:
                    width = 8
                best = DTypeInfo(base='float', width=width)
            else:
                # We're unifying an 'int' with a 'uint'.  The 'uint' can be held in an 'int' twice
                # the width; if that pushes us up to width=16, use a float64.
                best_width = best.width if best.base == 'int' else best.width * 2
                current_width = current.width if current.base == 'int' else current.width * 2
                width = max(best_width, current_width)
                if width > 8:
                    best = DTypeInfo(base='float', width=8)
                else:
                    best = DTypeInfo(base='int', width=width)
    return INFO_DTYPES[best]


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
        isize_strs = [str(sz) for sz in sizes if not isinstance(sz, Value)]
        code = 'function ({var_sizes}) -> (O) {{ O = broadcast({sizes}); }}'.format(
            var_sizes=', '.join(vsize_strs), sizes=', '.join(vsize_strs + isize_strs))
        shape = Shape(plaidml.DType.UINT64, tuple())
        operation = Operation(code, list(zip(vsize_strs, vsizes)), [('O', shape)])
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
                        this_size, size, ', '.join([
                            '({})'.format(', '.join(str(dim)
                                                    for dim in dtuple))
                            for dtuple in dtuples
                        ])))
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
    elif type(axes) == np.ndarray:
        axes = axes.tolist()
    else:
        if axes < 0:
            axes = len(dims) + axes
        axes = [axes]
    axes.sort(reverse=True)
    src_indices = ['x' + str(i) for i in range(len(dims))]
    src_ranges = ['X' + str(i) for i in range(len(dims))]
    dest_indices = src_indices[:]
    dest_ranges = src_ranges[:]
    reduce_indices = [dest_indices[i] for i in axes]
    reduce_ranges = [dest_ranges[i] for i in axes]
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
        'reduce_indices': ', '.join(reduce_indices),
        'reduce_ranges': ', '.join(reduce_ranges),
        'reduce_sep': ' : ' if reduce_indices else '',
    }
