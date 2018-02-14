# Copyright Vertex.AI.
"""
The TILE standard operation library.

These operations have been shown to be useful across a variety of frameworks.
(Frameworks are of course free to define their own operations in addition to
these, although it'll be easier to use them with these if a framework's own
operations are defined using the standard `plaidml.tile` base classes.)

Each operation is defined as a tile.Operation subclass, allowing it to be
used in pattern matching.  Additionally, each operation is provided via a
top-level function that wraps the class, allowing composite operations to
be built up using a functional programming style.
"""

# pylint: disable=invalid-name

import plaidml
from plaidml import tile


class ArgMax(tile.Operation):
    """Maximum of elements along an axis.

    Builds a tensor whose elements are the maximum value on some axis of an input tensor.
    """

    def __init__(self, value, axis=-1):
        self.axis = axis
        self.value = value
        super(ArgMax, self).__init__(None, [('I', value)], [('O', value.shape)])


argmax = ArgMax.function


class Equal(tile.Operation):
    """Elementwise tensor equality.

    Builds a boolean tensor whose values are true where the corresponding elements of the inputs
    are equal.
    """

    def __init__(self, lhs, rhs):
        shape = tile.Shape(plaidml.DType.BOOLEAN,
                           tile.broadcast_dims(lhs.shape.dims, rhs.shape.dims))
        self.lhs = lhs
        self.rhs = rhs
        super(Equal, self).__init__('function (L, R) -> (O) { O = (L == R); }',
                                    [('L', lhs), ('R', rhs)], [('O', shape)])


class Equal_ArgMax(tile.Operation):

    def __init__(self, lhs, rhs):
        lmax = ismax(lhs.source.op.value, axes=(lhs.source.op.axis,))
        rmax = ismax(rhs.source.op.value, axes=(rhs.source.op.axis,))

        and_shape = tile.Shape(plaidml.DType.INT32,
                               tile.broadcast_dims(lmax.shape.dims, rmax.shape.dims))
        and_op = tile.Operation('function (L, R) -> (O) { O = L ? (R ? 1 : 0) : 0; }',
                                [('L', lmax), ('R', rmax)], [('O', and_shape)])
        sum_val = summation(and_op.output_tuple[0], axes=(lhs.source.op.axis,), keepdims=True)
        eq_shape = tile.Shape(plaidml.DType.BOOLEAN, sum_val.shape.dims)
        super(Equal_ArgMax, self).__init__('function (I) -> (O) { O = 0 < I; }', [('I', sum_val)],
                                           [('O', eq_shape)])


def equal(lhs, rhs):
    """Elementwise tensor equality.

    Builds a boolean tensor whose values are true when the corresponding elements of the inputs
    are equal.

    Args:
        lhs (tile.Value): The left-hand side
        rhs (tile.Value): The right-hand side

    Returns:
        tile.Value: The output value
    """
    # TODO: Separate function builders from optimization/composition logic.
    #
    # Putting the composition logic in functions like this makes it a little hard for
    # higher-layer modules to add their own compositions -- think eq(MySpecialOp, MySpecialOp),
    # when some completely unrelated module is invoking the eq.  It would be better to have
    # something like a rewriter registry that could be consulted to match patterns during binding.
    if (lhs.source and isinstance(lhs.source.op, ArgMax) and rhs.source and
            isinstance(rhs.source.op, ArgMax)):
        return Equal_ArgMax.function(lhs, rhs)
    return Equal.function(lhs, rhs)


class Gradients(tile.Operation):
    """
    Compute the gradients of a loss with respect to a set of values
    """

    def __init__(self, loss, variables):
        super(Gradients, self).__init__(
            None, [('Loss', loss)] + [('I' + str(i), variables[i]) for i in range(len(variables))],
            [('O' + str(i), variables[i].shape) for i in range(len(variables))])
        self.num_vars = len(variables)

    def bind(self, bindings):
        loss_var = self.inputs['Loss'].bind(bindings)
        input_vars = [self.inputs['I' + str(i)].bind(bindings) for i in range(self.num_vars)]
        output_vars = plaidml.gradients(loss_var, input_vars)
        outputs = {}
        for i in range(self.num_vars):
            outputs['O' + str(i)] = output_vars[i]
        return outputs


def gradients(loss, variables):
    op = Gradients(loss, variables)
    outs = []
    for i in range(len(op.outputs)):
        outs.append(op.outputs['O' + str(i)])
    return outs


class IsMax(tile.Operation):
    """
    True iff an input's value is the maximum along some set of axes.
    """

    def __init__(self, value, axes):
        dims, _, subs = tile.compute_aggregation_axes(value.shape.dims, axes, True)

        code = """function (I[{src_ranges}]) -> (O) {{
                    MAX[{dest_indices}{dest_sep}{dest_ranges}] = >(I[{src_indices}]);
                    O = (MAX == I);
                }}""".format(**subs)

        super(IsMax, self).__init__(code, [('I', value)],
                                    [('O', tile.Shape(plaidml.DType.BOOLEAN, dims))])


ismax = IsMax.function


class MatMul(tile.Operation):
    """
    A matrix multiplication, using numpy semantics.

    See https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.matmul.html for details.
    """

    def __init__(self, a, b):
        # So, for matmul, we have identity dimensions (which remain the same
        # in the output tensor), and summation dimensions (which are
        # eliminated in the output tensor).  We call these I{1,2,...} and S.
        #
        # The matrix multiplication and summation takes place on the low two dimensions.
        # If either input is one-dimensional, that's its summation dimension.
        # Otherwise, A's summation dimension is the lowest dimension, and B's summation
        # dimension is its second-to-lowest.
        #
        # Naturally, there can be broadcasting involved; corresponding dimensions
        # must be broadcast-compatible.
        a_ndims = a.shape.ndims
        b_ndims = b.shape.ndims

        if a_ndims == 0 or b_ndims == 0:
            raise NotImplementedError('MatMul isn\'t defined over scalar values')

        if a_ndims == 1:
            if b_ndims == 1:
                # Both A and B are one dimensional; C is a scalar.
                #   A's dims are [S]
                #   B's dims are [S]
                #   C's dims are []
                c_dims = tuple()
                a_ranges = ['S']
                a_indicies = ['s']
                b_ranges = ['S']
                b_indicies = ['s']
                c_ranges = []
                c_indicies = []
            else:
                # A is one-dimensional, but B is not:
                #   A's dims are [S]
                #   B's dims are [I0, I1... IN-3, S, IN-1]
                #   C's dims are [I0, I1... IN-3, IN-1]
                c_shape = tuple(b.dims[:-2] + b.dims[-1])
                a_ranges = ['S']
                a_indicies = ['s']
                b_ranges = (['I{}'.format(n) for n in range(b_ndims - 2)] +
                            ['S', 'I{}'.format(b_ndims - 1)])
                b_indicies = (['i{}'.format(n) for n in range(b_ndims - 2)] +
                              ['s', 'i{}'.format(b_ndims - 1)])
                c_ranges = ['I{}'.format(n) for n in range(b_ndims - 2) + [b_ndims - 1]]
                c_indicies = ['i{}'.format(n) for n in range(b_ndims - 2) + [b_ndims - 1]]
        else:
            if b_ndims == 1:
                # B is one-dimensional, but A is not:
                #   A's dims are [I0, I1... IN-3, IN-2, S]
                #   B's dims are [S]
                #   C's dims are [I0, I1... IN-3, IN-2]
                c_dims = tuple(a.shape.dims[:-1])
                a_ranges = ['I{}'.format(n) for n in range(a_ndims - 1)] + ['S']
                a_indicies = ['i{}'.format(n) for n in range(a_ndims - 1)] + ['s']
                b_ranges = ['S']
                b_indicies = ['s']
                c_ranges = ['I{}'.format(n) for n in range(a_ndims - 1)]
                c_indicies = ['i{}'.format(n) for n in range(a_ndims - 1)]
            else:
                # Both tensors have more than one dimension.
                #   A's dims are [I0, I1... IN-3, IN-2, S]
                #   B's dims are [I0, I1... IN-3, S, IN-1]
                #   C's dims are [I0, I1... IN-3, IN-2, IN-1].
                c_dims = tuple(
                    list(tile.broadcast_dims(a.shape.dims[:-2], b.shape.dims[:-2])) +
                    [a.shape.dims[-2], b.shape.dims[-1]])
                a_ranges = ['I{}'.format(n) for n in range(a_ndims - 1)] + ['S']
                a_indicies = ['i{}'.format(n) for n in range(a_ndims - 1)] + ['s']
                b_ranges = (['I{}'.format(n) for n in range(b_ndims - 2)] +
                            ['S', 'I{}'.format(b_ndims - 1)])
                b_indicies = (['i{}'.format(n) for n in range(b_ndims - 2)] +
                              ['s', 'i{}'.format(b_ndims - 1)])
                c_ranges = ['I{}'.format(n) for n in range(len(c_dims))]
                c_indicies = ['i{}'.format(n) for n in range(len(c_dims))]

        func = """function(A[{a_ranges}], B[{b_ranges}]) -> (C) {{
                        C[{c_indicies} : {c_ranges}] = +(A[{a_indicies}] * B[{b_indicies}]);
                    }}""".format(
            a_ranges=', '.join(a_ranges),
            a_indicies=', '.join(a_indicies),
            b_ranges=', '.join(b_ranges),
            b_indicies=', '.join(b_indicies),
            c_ranges=', '.join(c_ranges),
            c_indicies=', '.join(c_indicies))

        c_shape = tile.Shape(tile.common_dtype(a.shape.dtype, b.shape.dtype), c_dims)

        super(MatMul, self).__init__(func, [('A', a), ('B', b)], [('C', c_shape)])


matmul = MatMul.function


class Relu(tile.Operation):
    """A Rectified Linear Unit."""

    def __init__(self, x):
        super(Relu, self).__init__('function (X) -> (Y) { Y = relu(X); }', [('X', x)],
                                   [('Y', x.shape)])


relu = Relu.function


class Reshape(tile.Operation):
    """
    Reshapes a tensor, without changing the type or number of elements.
    """

    def __init__(self, x, dims):
        super(Reshape, self).__init__('function (I) -> (O) {{ O = reshape(I, {}); }}'.format(
            ', '.join([str(d) for d in dims])), [('I', x)],
                                      [('O', tile.Shape(x.shape.dtype, dims))])


reshape = Reshape.function


class Sqrt(tile.Operation):
    """
    Computes the elementwise square root of a value.
    """

    def __init__(self, x):
        super(Sqrt, self).__init__("""function (I) -> (O) {
                   IC = (I < 0 ? 0 : I);
                   O = sqrt(IC);
               }""", [('I', x)], [('O', x.shape)])


sqrt = Sqrt.function


class Summation(tile.Operation):
    """
    Sums an input value along some set of axes.
    """

    def __init__(self, value, axes, keepdims=False):
        dims, _, subs = tile.compute_aggregation_axes(value.shape.dims, axes, keepdims)

        code = """function (I[{src_ranges}]) -> (O) {{
                    O[{dest_indices}{dest_sep}{dest_ranges}] = +(I[{src_indices}]);
                }}""".format(**subs)

        super(Summation, self).__init__(code, [('I', value)],
                                        [('O', tile.Shape(value.shape.dtype, dims))])


summation = Summation.function
