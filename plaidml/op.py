# Copyright Vertex.AI.
"""
The TILE standard operation library.

These operations have been shown to be useful across a variety of frameworks.
(Frameworks are of course free to define their own operations in addition to
these, although it'll be easier to use them with these if a framework's own
operations are defined using the standard :doc:`plaidml.tile` base classes.)

Each operation is defined as a ``tile.Operation`` subclass, allowing it to be
used in pattern matching.  Additionally, each operation is provided via a
top-level function that wraps the class, allowing composite operations to
be built up using a functional programming style.

See the `PlaidML Op Tutorial <https://github.com/plaidml/plaidml/wiki/PlaidML-Op-Tutorial>`_
for information about writing your own custom operations.
"""

# pylint: disable=invalid-name

from collections import defaultdict
import functools

from enum import Enum
import plaidml
from plaidml import tile
import six


class AutoPadding(Enum):
    EXPLICIT = 1
    VALID = 2
    SAME_UPPER = 3
    SAME_LOWER = 4


class ConvolutionDataFormat(Enum):
    CHANNELS_FIRST = 1
    CHANNELS_LAST = 2


def _extend_pads(pads, rank):
    """Extends a padding list to match the necessary rank.
    
    Args:
        pads ([int] or None): The explicitly-provided padding list.
        rank (int): The rank of the operation.

    Returns:
        None: If pads is None
        [int]: The extended padding list.
    """
    if pads is None:
        return pads
    pads = list(pads)
    if len(pads) < rank:
        pads.extend([0] * (rank - len(pads)))
    if len(pads) < (2 * rank):
        pads.extend(pads[len(pads) - rank:rank])
    return pads


def pad_compute(sym, input_size, filter_size, stride, padding, pads=None):
    """Computes info for an axis of a padded filter.

    Args:
        sym (str): The symbol for the input axis.
        input_size (tile.Value or int): The size of the input axis (possibly symbolic).
        filter_size (int): The size of the filter along this axis.
        stride (int): The stride of the filter along this axis.
        padding (AutoPadding): The padding style to use.
        pads ((int, int) or None): Explicit pre- and post-padding for this axis.

    Returns:
        tuple(A string representing the output size as TILE code,
              The pre-padding to use when building input accessor expressions,
              A tile.Value representing the computed output size)
    """
    if pads:
        num_out_size = (input_size + pads[0] + pads[1] - filter_size + stride) // stride
        sym_output_size = '({sym} + {pre} + {post} - {fs} + {s}) / {s}'.format(
            sym=sym, pre=pads[0], post=pads[1], fs=filter_size, s=stride)
        sym_padding_before = pads[0]
    elif padding == AutoPadding.VALID:
        num_out_size = (input_size - filter_size + stride) // stride
        sym_output_size = '({sym} - {fs} + {s}) / {s}'.format(sym=sym, fs=filter_size, s=stride)
        sym_padding_before = 0
    elif padding == AutoPadding.SAME_UPPER or padding == AutoPadding.SAME_LOWER:
        num_out_size = (input_size + stride - 1) // stride
        sym_output_size = '({sym} + {s} - 1) / {s}'.format(sym=sym, s=stride)

        if padding == AutoPadding.SAME_UPPER:
            expr = '(max(0, ({symout} - 1) * {s} + {fs} - {syminp})) / 2'
        else:
            expr = '((max(0, ({symout} - 1) * {s} + {fs} - {syminp})) + 1) / 2'
        sym_padding_before = expr.format(
            symout=sym_output_size, s=stride, fs=filter_size, syminp=sym)
    else:
        raise Exception('Invalid padding: ' + str(padding))
    if not isinstance(num_out_size, tile.Value) and num_out_size < 0:
        raise Exception(
            'Invalid output size computed for convolution: num_out_size={}'.format(num_out_size))
    return (sym_output_size, sym_padding_before, num_out_size)


def _format_conv_strings(
        rank,
        in_shape,
        kernel_shape,
        strides,
        padding,
        data_format,
        dilation_rate,
        channelwise,
        forward=True,
        expected_output_shape=None,
):
    # Variable meanings:
    # N: Number of items in the batch
    # L<i>: Spatial dimension i of each (input) item
    # CI: Number of channels (aka filters) of each input item
    # LK<i>: Spatial dimension i of kernel
    # CO: Number of channels (aka filters) of each output item
    # C: Number of input channels in channelwise convolutions
    # M: Channel multiplier in channelwise convolutions (each input channel yields
    #     M output channels for such convolutions)
    #
    # n: Which element of the batch we're on
    # x<i>: The ith coordinate in the output/image
    # k<i>: The ith coordinate in the kernel
    # ci: The input channel we're on
    # co: The output channel we're on
    # c: The input channel we're on for channelwise convolutions
    # m: The output channel multiplier we're on for output convolutions
    if data_format == ConvolutionDataFormat.CHANNELS_FIRST:
        n = 0
        c = 1
        l = [i + 2 for i in range(rank)]
    elif data_format == ConvolutionDataFormat.CHANNELS_LAST:
        n = 0
        l = [i + 1 for i in range(rank)]
        c = rank + 1
    else:
        raise ValueError('Unrecognized data format \'{}\''.format(data_format))
    if channelwise == True and in_shape[c] != kernel_shape[-2]:
        raise ValueError(
            'Channelwise convolution must have same number of channels in both input and kernel:\n'
            + '{} (from shape {}) v {} (from shape {})'.format(in_shape[c], in_shape,
                                                               kernel_shape[-2], kernel_shape))
    sym_out_shape = list()
    pad_amount = list()
    num_out_shape = list()
    for i in range(rank):
        if forward:
            sym_out, sym_pad, num_out = pad_compute('L{}'.format(i), in_shape[l[i]],
                                                    dilation_rate[i] * (kernel_shape[i] - 1) + 1,
                                                    strides[i], padding, None)
        else:
            sym_out, sym_pad, num_out = pad_compute('D{}'.format(i), in_shape[l[i]],
                                                    dilation_rate[i] * (kernel_shape[i] - 1) + 1,
                                                    strides[i], padding, None)
        sym_out_shape.append(sym_out)
        pad_amount.append(sym_pad)
        num_out_shape.append(num_out)
    if expected_output_shape is not None:
        # Confirm that the output shape is consistent with the rest of the convolution
        computed_output_shape = [0] * (rank + 2)
        computed_output_shape[n] = in_shape[n]
        computed_output_shape[c] = kernel_shape[-1]
        for i in range(rank):
            computed_output_shape[l[i]] = num_out_shape[i]
        for i in range(rank + 2):
            if (not isinstance(computed_output_shape[i], tile.Value) and
                    not isinstance(expected_output_shape[i], tile.Value) and
                    computed_output_shape[i] != expected_output_shape[i]):
                raise ValueError('Expected convolution output of shape {}, received {}'.format(
                    expected_output_shape, computed_output_shape))
    padding_list = ['Pad{} = {};'.format(i, pad_amount[i]) for i in range(rank)]
    padding_str = ''.join(p + '\n                   ' for p in padding_list)
    input_idx_list = [
        '{s}*{x} + {d}*{k} - {p}'.format(
            s=strides[i],
            x='x{}'.format(i),
            d='{}'.format(dilation_rate[i]),
            k='k{}'.format(i),
            p='Pad{}'.format(i)) for i in range(rank)
    ]
    if data_format == ConvolutionDataFormat.CHANNELS_FIRST and not channelwise:
        if forward:
            input_dims_str = 'N, CI, ' + ', '.join(['L{}'.format(i) for i in range(rank)])
            out_dims_str = 'N, CO, ' + ', '.join(
                ['{}'.format(sym_out_shape[i]) for i in range(rank)])
            outshape = [in_shape[0]] + [kernel_shape[-1]] + num_out_shape
        else:
            input_dims_str = 'N, CI, ' + ', '.join('D{}'.format(i) for i in range(rank))
            out_dims_str = 'N, CO, ' + ', '.join(['L{}'.format(i) for i in range(rank)])
        out_idx_str = 'n, co, ' + ', '.join(['x{}'.format(i) for i in range(rank)])
        input_idx_str = 'n, ci, ' + ', '.join(input_idx_list)
    elif data_format == ConvolutionDataFormat.CHANNELS_LAST and not channelwise:
        if forward:
            input_dims_str = 'N, ' + ', '.join(['L{}'.format(i) for i in range(rank)]) + ', CI'
            out_dims_str = 'N, ' + ', '.join(['{}'.format(sym_out_shape[i])
                                              for i in range(rank)]) + ', CO'
            outshape = [in_shape[0]] + num_out_shape + [kernel_shape[-1]]
        else:
            input_dims_str = 'N, ' + ', '.join('D{}'.format(i) for i in range(rank)) + ', CI'
            out_dims_str = 'N, ' + ', '.join(['L{}'.format(i) for i in range(rank)]) + ', CO'
        out_idx_str = 'n, ' + ', '.join(['x{}'.format(i) for i in range(rank)]) + ', co'
        input_idx_str = 'n, ' + ', '.join(input_idx_list) + ', ci'
    elif data_format == ConvolutionDataFormat.CHANNELS_FIRST and channelwise:
        if not forward:
            raise NotImplementedError('Channelwise transposed convolutions not implemented.')
        input_dims_str = 'N, C, ' + ', '.join(['L{}'.format(i) for i in range(rank)])
        out_idx_str = 'n, c*M + m, ' + ', '.join(['x{}'.format(i) for i in range(rank)])
        out_dims_str = 'N, C*M, ' + ', '.join(['{}'.format(sym_out_shape[i]) for i in range(rank)])
        input_idx_str = 'n, c, ' + ', '.join(input_idx_list)
        outshape = [in_shape[0]] + [kernel_shape[-2] * kernel_shape[-1]] + num_out_shape
    elif data_format == ConvolutionDataFormat.CHANNELS_LAST and channelwise:
        if not forward:
            raise NotImplementedError('Channelwise transposed convolutions not implemented.')
        input_dims_str = 'N, ' + ', '.join(['L{}'.format(i) for i in range(rank)]) + ', C'
        out_idx_str = 'n, ' + ', '.join(['x{}'.format(i) for i in range(rank)]) + ', c*M + m'
        out_dims_str = 'N, ' + ', '.join(['{}'.format(sym_out_shape[i])
                                          for i in range(rank)]) + ', C*M'
        input_idx_str = 'n, ' + ', '.join(input_idx_list) + ', c'
        outshape = [in_shape[0]] + num_out_shape + [kernel_shape[-2] * kernel_shape[-1]]
    else:
        raise ValueError('Unrecognized data format \'{}\''.format(data_format))
    if channelwise:
        ker_dims_str = ', '.join(['LK{}'.format(i) for i in range(rank)]) + ', C, M'
        ker_idx_str = ', '.join(['k{}'.format(i) for i in range(rank)]) + ', c, m'
    else:
        ker_dims_str = ', '.join(['LK{}'.format(i) for i in range(rank)]) + ', CI, CO'
        ker_idx_str = ', '.join(['k{}'.format(i) for i in range(rank)]) + ', ci, co'
    ret = {
        'input_dims_str': input_dims_str,
        'ker_dims_str': ker_dims_str,
        'out_idx_str': out_idx_str,
        'out_dims_str': out_dims_str,
        'input_idx_str': input_idx_str,
        'ker_idx_str': ker_idx_str,
        'padding_str': padding_str
    }
    if forward:
        ret['outshape_tuple'] = outshape
    else:
        ret['dim_input'] = ', ' + ', '.join(['D{}'.format(i) for i in range(rank)])
    return ret


class ArgMax(tile.Operation):
    """Maximum of elements along an axis.

    Builds a tensor whose elements are the maximum value on some axis of an input tensor.
    """

    def __init__(self, value, axis=-1):
        self.axis = axis
        self.value = value
        super(ArgMax, self).__init__(None, [('I', value)], [('O', value.shape)])


argmax = ArgMax.function


class AveragePool(tile.Operation):
    """
    A standard ML average pooling operator.
    """

    def __init__(self, data, kernel_shape, pads, strides, padding=AutoPadding.EXPLICIT):
        rank = data.shape.ndims - 2
        pads = _extend_pads(pads, rank)
        if not strides:
            strides = tuple(1 for _ in range(rank))
        elif len(strides) != rank:
            raise ValueError('Pool strides length inconsistent with input shape: ' +
                             '{} (rank {}) v {} (rank {})'.format(strides,
                                                                  len(strides), data.shape, rank))
        out_dims = ['N', 'C']
        num_out_shape = list()
        in_idxs = list()
        for i in range(rank):
            sym_out, sym_pad, num_out = pad_compute('L{}'.format(i), data.shape.dims[i + 2],
                                                    kernel_shape[i], strides[i], padding,
                                                    (pads[i], pads[i + rank]) if pads else None)
            out_dims.append(sym_out)
            num_out_shape.append(num_out)
            in_idxs.append('{stride}*x{idx} + a{idx} - {pad}'.format(
                stride=strides[i], idx=i, pad=sym_pad))
        out_idxs = ['n', 'c'] + ['x{}'.format(i) for i in range(rank)]

        code = """
        function (I[N, C, {in_dims}], One[]) -> (O) {{
            Ones[{one_idxs} : {in_dims}] = =(One[]);
            Count[{cout_idxs}{cout_sep}{cout_dims}] = +(Ones[{in_idxs}]), {pool_bounds};
            S[{out_idxs} : {out_dims}] = +(I[n, c, {in_idxs}]), {pool_bounds};
            O = S / Count;
        }}""".format(
            out_idxs=', '.join(out_idxs),
            out_dims=', '.join(out_dims),
            cout_idxs=', '.join(out_idxs[2:]),
            cout_dims=', '.join(out_dims[2:]),
            cout_sep=' : ' if len(out_idxs) > 2 else '',
            one_idxs=', '.join(['o{}'.format(i) for i in range(rank)]),
            in_idxs=', '.join(in_idxs),
            in_dims=', '.join(['L{}'.format(i) for i in range(rank)]),
            pool_bounds=', '.join(['a{} < {}'.format(i, kernel_shape[i]) for i in range(rank)]))

        outshape = tile.Shape(data.shape.dtype, list(data.shape.dims[0:2]) + num_out_shape)

        super(AveragePool, self).__init__(
            code, [('I', data), ('One', tile.Value.from_var(1., tuple()))], [('O', outshape)])


average_pool = AveragePool.function


class BinaryCrossentropy(tile.Operation):
    """
    Computes the binary crossentropy of a value relative to a target.
    """

    def __init__(self, target, output, epsilon, from_logits=False):
        if epsilon is None:
            epsilon = 0.0
        if from_logits:
            output = sigmoid(output)
        output = clip(output, epsilon, 1.0 - epsilon)
        input_sizes = ','.join(['I' + str(i) for i in range(output.shape.ndims)])
        input_sizes_prod = '*'.join(['I' + str(i) for i in range(output.shape.ndims)])
        f = """
            function (O[{dims}], T[{dims}]) -> (R) {{
                R = builtin_binary_crossentropy(O,T,{prod});
            }}""".format(
            dims=input_sizes, prod=input_sizes_prod)
        super(BinaryCrossentropy, self).__init__(f, [('O', output), ('T', target)],
                                                 [('R', output.shape)])


binary_crossentropy = BinaryCrossentropy.function


class Cast(tile.Operation):

    def __init__(self, x, dtype):
        info = tile.DTYPE_INFOS[dtype]
        super(Cast, self).__init__('function (I) -> (O) {{ O = as_{}(I, {}); }}'.format(
            info.base, info.bitwidth), [('I', x)], [('O', tile.Shape(dtype, x.shape.dims))])


cast = Cast.function


def ceiling(data):
    """Elementwise ceiling."""
    return tile.unary_op(data, 'ceil(I)', 'Ceiling')


class ClipMin(tile.Operation):
    """Clips a Value to a minimum bound."""

    def __init__(self, value, min_val):
        code = """
               function (I, MIN_VAL) -> (O) {
                   O = (MIN_VAL < I ? I : MIN_VAL);
               }"""
        super(ClipMin, self).__init__(code, [('I', value), ('MIN_VAL', min_val)],
                                      [('O', value.shape)])


class ClipMax(tile.Operation):
    """Clips a Value to a maximum bound."""

    def __init__(self, value, max_val):
        code = """
               function (I, MAX_VAL) -> (O) {
                   O = (I < MAX_VAL ? I : MAX_VAL);
               }"""
        super(ClipMax, self).__init__(code, [('I', value), ('MAX_VAL', max_val)],
                                      [('O', value.shape)])


def clip(value, min_val, max_val):
    if min_val is not None:
        value = ClipMin.function(value, min_val)
    if max_val is not None:
        value = ClipMax.function(value, max_val)
    return value


class Concatenate(tile.Operation):
    """Concatenates tensors to make a single larger tensor."""

    def __init__(self, tensors, axis=-1):
        rank = tensors[0].shape.ndims
        if axis >= rank or axis < -rank:
            raise ValueError('Cannot concatenate tensors with {} dimensions along axis {}'.format(
                rank, axis))
        elif axis < 0:
            axis = axis % rank

        def __clear_axis(dims):
            return [
                None if isinstance(dims[i], tile.Value) else dims[i] for i in range(len(dims))
                if i != axis
            ]

        shape_template = __clear_axis(tensors[0].shape.dims)
        for t in tensors:
            if __clear_axis(t.shape.dims) != shape_template:
                raise ValueError(
                    'Incompatible shapes: cannot concatenate along axis {}\n{} v {}'.format(
                        axis, tensors[0].shape, t.shape))

        offsets = [0]
        for i in range(len(tensors)):
            offsets.append(offsets[i] + tensors[i].shape.dims[axis])
        out_dims = tuple(
            tensors[0].shape.dims[i] if i != axis else offsets[len(tensors)] for i in range(rank))

        output_dims_list = ['N{}'.format(i) for i in range(rank)]
        output_dims_list[axis] = offsets[len(tensors)]
        output_dims_str = ', '.join([str(i) for i in output_dims_list])
        # output_dims_list also serves as a base for input dims,
        # with `axis` index to be overwritten by 'Ai' (i = input index)
        inputs_list = list()
        for i in range(len(tensors)):
            curr_input_dims = list(output_dims_list)  # using 'list' here to make a copy
            curr_input_dims[axis] = 'A{}'.format(i)
            inputs_list.append('I{}[{}]'.format(i, ', '.join(curr_input_dims)))
        inputs_str = ', '.join(inputs_list)

        if axis == 0:
            indices_begin = 'a'
        else:
            indices_begin = ', '.join(['n{}'.format(i) for i in range(axis)]) + ', a'
        if axis == rank - 1:
            indices_end = ''
        else:
            indices_end = ', ' + ', '.join(['n{}'.format(i) for i in range(axis + 1, rank)])

        body_str = ''
        line_subs = {'beg': indices_begin, 'end': indices_end, 'odims': output_dims_str}
        for i in range(len(tensors)):
            # TODO: If offsets[i] is symbolic, add it to the function
            # inputs and use it symbolically.
            line_subs['off'] = '+{}'.format(offsets[i])
            line_subs['i'] = i
            curr_line = '  T{i}[{beg}{off}{end}: {odims}] = =(I{i}[{beg}{end}]);\n'.format(
                **line_subs)
            body_str += curr_line
        body_str += '  O = '
        body_str += ' + '.join(['T{}'.format(i) for i in range(len(tensors))])
        body_str += ';'

        # Example 'code' (concatenating (4,3,2), (4,5,2), (4,1,2)):
        #   function (I0[N0, A0, N2], I1[N0, A1, N2], I2[N0, A2, N2]) -> (O) {
        #     T0[n0, a, n2: N0, 9, N2] = =(I0[n0, a, n2]);
        #     T1[n0, a+3, n2: N0, 9, N2] = =(I1[n0, a, n2]);
        #     T2[n0, a+8, n2: N0, 9, N2] = =(I2[n0, a, n2]);
        #     O = T0 + T1 + T2;
        #   }
        code = ('function ({inputs}) -> (O) {{\n{body}\n}}').format(
            inputs=inputs_str,
            body=body_str,
        )
        inputs_list = []
        inputs_list.extend([('I{}'.format(i), tensors[i]) for i in range(len(tensors))])

        super(Concatenate, self).__init__(code, inputs_list,
                                          [('O', tile.Shape(tensors[0].shape.dtype, out_dims))])


concatenate = Concatenate.function


class Convolution(tile.Operation):
    """
    A standard ML convolution operator.
    """

    def __init__(self,
                 data,
                 kernel,
                 strides=None,
                 padding=AutoPadding.EXPLICIT,
                 pads=None,
                 group=1,
                 kernel_shape=None,
                 data_format=None,
                 dilation_rate=None,
                 channelwise=False):
        if group != 1:
            raise NotImplementedError('Grouped convolutions are not currently implemented')
        rank = data.shape.ndims - 2
        if strides is None:
            strides = tuple(1 for _ in range(rank))
        if dilation_rate is None:
            dilation_rate = tuple(1 for _ in range(rank))
        if not kernel_shape:
            kernel_shape = kernel.shape.dims
        else:
            kernel_shape = tuple([kernel.shape.dims[0], kernel.shape.dims[1]] + list(kernel_shape))

        for entry in dilation_rate:
            if not isinstance(entry, int) or entry <= 0:
                raise ValueError('Invalid dilation_rate: {}'.format(dilation_rate))
        if len(kernel_shape) != rank + 2:
            raise ValueError('Convolution kernel shape inconsistent with input shape: ' +
                             '{} (rank {}) v {} (rank {})'.format(
                                 kernel_shape,
                                 len(kernel_shape) - 2, data.shape, data.shape.ndims - 2))
        if len(strides) != rank:
            raise ValueError('Convolution strides length inconsistent with input shape: ' +
                             '{} (rank {}) v {} (rank {})'.format(
                                 strides, len(strides), data.shape, data.shape.ndims - 2))
        if len(dilation_rate) != rank:
            raise ValueError('Convolution dilation_rate length inconsistent with input shape: ' +
                             '{} (rank {}) v {} (rank {})'.format(dilation_rate,
                                                                  len(dilation_rate), data.shape,
                                                                  data.shape.ndims - 2))

        conv_strs = _format_conv_strings(rank, data.shape.dims, kernel_shape, strides, padding,
                                         data_format, dilation_rate, channelwise)
        code = """
               function (I[{input_dims_str}], K[{ker_dims_str}]) -> (O) {{
                   {padding_str}O[{out_idx_str} : {out_dims_str}] = +(I[{input_idx_str}]*K[{ker_idx_str}]);
               }}""".format(**conv_strs)

        outshape = tile.Shape(data.shape.dtype, conv_strs['outshape_tuple'])

        super(Convolution, self).__init__(
            code, [('I', data), ('K', kernel)], [('O', outshape)],
            name='Convolution-{}d'.format(rank))


convolution = Convolution.function


class ConvolutionTranspose(tile.Operation):
    """
    A transposed convolution operator.
    """

    def __init__(self, x, kernel, output_shape, strides, padding, data_format):
        rank = x.shape.ndims - 2

        if kernel.shape.ndims != rank + 2:
            raise ValueError('Transpose convolution kernel shape inconsistent with input shape: ' +
                             '{} (rank {}) v {} (rank {})'.format(
                                 kernel.shape, kernel.shape.ndims - 2, x.shape, x.shape.ndims - 2))
        if len(output_shape) != rank + 2:
            raise ValueError('Transpose convolution output_shape inconsistent with input shape: ' +
                             '{} (rank {}) v {} (rank {})'.format(
                                 output_shape, len(output_shape) - 2, x.shape, x.shape.ndims - 2))
        if len(strides) != rank:
            raise ValueError('Transpose convolution strides inconsistent with input shape: ' +
                             '{} (rank {}) v {} (rank {})'.format(
                                 strides, len(strides), x.shape, x.shape.ndims - 2))
        if (x.shape.dims[0] != output_shape[0] and
                isinstance(x.shape.dims[0], six.integer_types) and
                isinstance(output_shape[0], six.integer_types)):
            raise ValueError('Transpose convolution batch size inconsistent between input ' +
                             'and output: {} v {}'.format(x.shape.dims[0], output_shape[0]))

        conv_strs = _format_conv_strings(rank, output_shape, kernel.shape.dims, strides, padding,
                                         data_format, (1,) * rank, False, False, x.shape.dims)

        f = """
            function (O[{out_dims_str}], K[{ker_dims_str}]{dim_input}) -> (I) {{
                {padding_str}
                I[{input_idx_str} : {input_dims_str}] = +(O[{out_idx_str}]*K[{ker_idx_str}]);
            }}""".format(**conv_strs)

        # Output shape may be dynamic, so pass its sizes as inputs to Tile
        if data_format == ConvolutionDataFormat.CHANNELS_FIRST:
            l = [i + 2 for i in range(rank)]
        elif data_format == ConvolutionDataFormat.CHANNELS_LAST:
            l = [i + 1 for i in range(rank)]
        else:
            raise ValueError('Unrecognized data format \'{}\''.format(data_format))

        input_tensors = [('O', x), ('K', kernel)] + \
                        [('D{}'.format(i), output_shape[l[i]]) for i in range(rank)]

        super(ConvolutionTranspose, self).__init__(
            f,
            input_tensors, [('I', tile.Shape(x.shape.dtype, tuple(output_shape)))],
            name='ConvolutionTranspose-{}d'.format(rank))


convolution_transpose = ConvolutionTranspose.function


def cos(data):
    """Elementwise cosine."""
    return tile.unary_op(data, 'cos(I)', 'Cosine')


class CumulativeSum(tile.Operation):
    """Cumulative sum of a tensor"""

    def __init__(self, x, axis=0):
        ranges = ', '.join(['N{}'.format(n) for n in range(x.shape.ndims)])
        dest_idxs = ', '.join(['i{}'.format(n) for n in range(x.shape.ndims)])
        src_idxs = ['i{}'.format(n) for n in range(x.shape.ndims)]
        src_idxs[axis] += ' - k'
        src_idxs = ', '.join(src_idxs)
        f = """
            function (I[{src_ranges}]) -> (O) {{
                O[{dest_idxs}: {dest_ranges}] = +(I[{src_idxs}]), k < N{ax};
            }}""".format(
            src_ranges=ranges, dest_idxs=dest_idxs, dest_ranges=ranges, src_idxs=src_idxs, ax=axis)
        super(CumulativeSum, self).__init__(f, [('I', x)], [('O', x.shape)])


cumulative_sum = CumulativeSum.function


class Dot(tile.Operation):
    """Dot-product of two tensors."""

    def __init__(self, x, y):
        if x.shape.dtype != y.shape.dtype:
            raise ValueError(
                'Invalid dtype in multiplication: x.dtype=\'{}\', y.dtype=\'{}\''.format(
                    x.shape.dtype, y.shape.dtype))

        if x.shape.ndims == 1 and y.shape.ndims == 1:
            f = 'function (X[I], Y[I]) -> (R) { R[i:I] = +(X[i] * Y[i]); }'
            shape = x.shape
        elif 1 <= x.shape.ndims and 2 <= y.shape.ndims:
            f = """function(X[{x_ranges}], Y[{y_ranges}]) -> (R) {{
                       R[{dest_indices} : {dest_ranges}] = +(X[{x_indices}] * Y[{y_indices}]);
                   }}""".format(
                x_ranges=', '.join(['X{}'.format(i) for i in range(x.shape.ndims)]),
                y_ranges=', '.join(['Y{}'.format(i) for i in range(y.shape.ndims)]),
                dest_indices=', '.join(['x{}'.format(i) for i in range(x.shape.ndims - 1)] + [
                    'y{}'.format(i) for i in (list(range(y.shape.ndims - 2)) + [y.shape.ndims - 1])
                ]),
                dest_ranges=', '.join(['X{}'.format(i) for i in range(x.shape.ndims - 1)] + [
                    'Y{}'.format(i) for i in (list(range(y.shape.ndims - 2)) + [y.shape.ndims - 1])
                ]),
                x_indices=', '.join(['x{}'.format(i) for i in range(x.shape.ndims - 1)] + ['z']),
                y_indices=', '.join(['y{}'.format(i) for i in range(y.shape.ndims - 2)] + ['z'] +
                                    ['y{}'.format(y.shape.ndims - 1)]))
            shape = tile.Shape(
                x.shape.dtype,
                (list(x.shape.dims[:-1]) + list(y.shape.dims[:-2]) + [y.shape.dims[-1]]))
        else:
            raise NotImplementedError('Implement dot when x.dims={} and y.dims={}'.format(
                x.shape.dims, y.shape.dims))

        super(Dot, self).__init__(f, [('X', x), ('Y', y)], [('R', shape)])


dot = Dot.function


class Elu(tile.Operation):
    """Exponential linear unit."""

    def __init__(self, x, alpha=1.0):
        if alpha == 1:
            code = """
                   function (X) -> (R) {
                       A = exp(X)-1;
                       R = (X < 0 ? A : X);
                   }"""
        else:
            code = """
                   function (X) -> (R) {{
                       A = {alpha}*exp(X) - {alpha};
                       R = X < 0 ? A : X;
                   }}""".format(alpha=alpha)

        super(Elu, self).__init__(code, [('X', x)], [('R', x.shape)])


elu = Elu.function


class Equal(tile.Operation):
    """Elementwise tensor equality.

    Builds a boolean tensor whose values are true where the corresponding elements of the inputs
    are equal.
    """

    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs

        if isinstance(rhs, tile.Value):
            shape = tile.Shape(plaidml.DType.BOOLEAN,
                               tile.broadcast_dims(lhs.shape.dims, rhs.shape.dims))
            super(Equal, self).__init__('function (L, R) -> (O) { O = (L == R); }',
                                        [('L', lhs), ('R', rhs)], [('O', shape)])
        else:
            shape = tile.Shape(plaidml.DType.BOOLEAN, lhs.shape.dims)
            super(Equal, self).__init__('function (L) -> (O) {{ O = (L == {}); }}'.format(rhs),
                                        [('L', lhs)], [('O', shape)])


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


def exp(data):
    """Elementwise exponential."""
    return tile.unary_op(data, 'exp(I)', 'Exp')


class Flatten(tile.Operation):
    """
    Flattens a tensor to a one-dimensional value.
    """

    def __init__(self, data):
        in_dim_list = ['N{}'.format(i) for i in range(data.shape.ndims)]
        out_dim_list = ['*'.join(['N{}'.format(i) for i in range(data.shape.ndims)])]
        new_size = functools.reduce(lambda x, y: x * y, data.shape.dims)

        code = 'function (I[{idims}]) -> (O) {{ O = reshape(I, {odims}); }}'.format(
            idims=', '.join(in_dim_list), odims=', '.join(out_dim_list))
        super(Flatten, self).__init__(code, [('I', data)],
                                      [('O', tile.Shape(data.shape.dtype, (new_size,)))])


flatten = Flatten.function


def floor(data):
    """Elementwise floor."""
    return tile.unary_op(data, 'floor(I)', 'Floor')


class Gather(tile.Operation):
    """
    Gathers elements of a tensor.
    """

    def __init__(self, value, indicies):
        outshape = tile.Shape(value.shape.dtype,
                              list(indicies.shape.dims) + list(value.shape.dims[1:]))

        super(Gather, self).__init__('function (V, I) -> (O) { O = gather(V, I); }',
                                     [('V', value), ('I', indicies)], [('O', outshape)])


gather = Gather.function


class Gemm(tile.Operation):
    """
    Implements a general matrix multiplication.
    """

    def __init__(self, a, b, c, alpha=None, beta=None, broadcast=True, transA=False, transB=False):
        if broadcast:
            if c.shape.ndims != 1:
                raise NotImplementedError(
                    'Gemm with multiplier broadcast requires a one-dimensional scalar multiplier; multiplier rank={}'.
                    format(c.shape.ndims))
        elif c.shape.ndims != 2:
            raise NotImplementedError(
                'Gemm without multiplier broadcast requires a two-dimensional scalar multiplier; multiplier rank={}'.
                format(c.shape.ndims))

        def gemm_reshape(value):
            if value.shape.ndims < 2:
                raise tile.LogicError(
                    'Invalid Gemm input; two-dimensions required, got: {}'.format(value.shape))
            if value.shape.ndims == 2:
                return value
            newdims = (value.shape.dims[0], functools.reduce(lambda x, y: x * y,
                                                             value.shape.dims[1:]))
            return reshape(value, newdims)

        a = gemm_reshape(a)
        b = gemm_reshape(b)

        code = """
        function (A[{a_dims}], B[{b_dims}], C[{c_dims}]) -> (O) {{
          OM[row, col : ROW, COL] = +(A[{a_idxs}] * B[{b_idxs}]);
          OA = {alpha_expr};
          CB = {beta_expr};
          O = OA + CB;
        }}""".format(
            a_dims='MID, ROW' if transA else 'ROW, MID',
            b_dims='COL, MID' if transB else 'COL, MID',
            c_dims='ROW, COL' if c.shape.ndims == 2 else 'COL',
            a_idxs='mid, row' if transA else 'row, mid',
            b_idxs='col, mid' if transB else 'col, mid',
            alpha_expr='OM * {}'.format(alpha) if alpha else 'OM',
            beta_expr='C * {}'.format(beta) if beta else 'C',
        )

        outshape = tile.Shape(
            tile.common_dtype(a.shape.dtype, b.shape.dtype, c.shape.dtype),
            tile.broadcast_dims((
                a.shape.dims[1] if transA else a.shape.dims[0],
                b.shape.dims[0] if transB else b.shape.dims[1],
            ), c.shape.dims))

        super(Gemm, self).__init__(code, [('A', a), ('B', b), ('C', c)], [('O', outshape)])


gemm = Gemm.function


class Gradients(tile.Operation):
    """
    Compute the gradients of a loss with respect to a set of values
    """

    def __init__(self, loss, variables):
        super(Gradients, self).__init__(None, [('Loss', loss)] + [('I' + str(i), variables[i])
                                                                  for i in range(len(variables))],
                                        [('O' + str(i), variables[i].shape)
                                         for i in range(len(variables))])
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
    if isinstance(variables, tile.Value):
        variables = [variables]
    op = Gradients(loss, variables)
    outs = []
    for i in range(len(op.outputs)):
        outs.append(op.outputs['O' + str(i)])
    return outs


class Hardmax(tile.Operation):
    """
    Implements a standard ML hardmax.
    """

    def __init__(self, data):
        if data.shape.ndims != 2:
            raise NotImplementedError(
                'Hardmax with a non-two-dimensional tensor is not currently implemented')

        code = """
        function (I[X, Y]) -> (O) {
            MAXX[x : X] = >(I[x, y]);
            MAX[x, y : X, Y] = =(MAXX[x]);
            O = (MAX == I ? 1.0 : 0.0);
        }"""

        super(Hardmax, self).__init__(code, [('I', data)], [('O', data.shape)])


def hardmax(x, axis=None):
    if x.shape.ndims == 2:
        return Hardmax.function(x)
    if axis is None:
        axis = 1
    full_dims = x.shape.dims
    if axis == 0:
        group = 1
    else:
        group = functools.reduce(lambda x, y: x * y, x.shape.dims[:axis])
    if axis == len(x.shape.dims):
        values = 1
    else:
        values = functools.reduce(lambda x, y: x * y, x.shape.dims[axis:])
    flat_x = reshape(x, (group, values))
    result = Hardmax.function(flat_x)
    return reshape(result, full_dims)


class Identity(tile.Operation):
    """A simple identity operation."""

    def __init__(self, x):
        super(Identity, self).__init__('function (X) -> (Y) { Y = X; }', [('X', x)],
                                       [('Y', x.shape)])


identity = Identity.function


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


def log(data):
    """Elementwise logarithm."""
    return tile.unary_op(data, 'log(I)', 'Log')


class LogSoftmax(tile.Operation):
    """
    Implements the log() of a standard ML softmax.
    """

    def __init__(self, data):
        if data.shape.ndims != 2:
            raise NotImplementedError(
                'LogSoftmax with a non-two-dimensional tensor is not currently implemented')

        code = """
        function (I[X, Y]) -> (O) {
            O = builtin_logsoftmax(I, X, Y);
        }"""

        super(LogSoftmax, self).__init__(code, [('I', data)], [('O', data.shape)])


def log_softmax(x, axis=None):
    if x.shape.ndims == 2:
        return LogSoftmax.function(x)
    if axis is None:
        axis = 1
    full_dims = x.shape.dims
    if axis == 0:
        group = 1
    else:
        group = functools.reduce(lambda x, y: x * y, x.shape.dims[:axis])
    if axis == len(x.shape.dims):
        values = 1
    else:
        values = functools.reduce(lambda x, y: x * y, x.shape.dims[axis:])
    flat_x = reshape(x, (group, values))
    result = LogSoftmax.function(flat_x)
    return reshape(result, full_dims)


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
                b_ranges = (['I{}'.format(n)
                             for n in range(b_ndims - 2)] + ['S', 'I{}'.format(b_ndims - 1)])
                b_indicies = (['i{}'.format(n)
                               for n in range(b_ndims - 2)] + ['s', 'i{}'.format(b_ndims - 1)])
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
                b_ranges = (['I{}'.format(n)
                             for n in range(b_ndims - 2)] + ['S', 'I{}'.format(b_ndims - 1)])
                b_indicies = (['i{}'.format(n)
                               for n in range(b_ndims - 2)] + ['s', 'i{}'.format(b_ndims - 1)])
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


class MaxReduce(tile.Operation):
    """Computes the maximum value along some set of axes."""

    def __init__(self, x, axes=None, keepdims=False):
        if axes == None:
            axes = list(range(x.shape.ndims))

        shape, axes, subs = tile.compute_aggregation_axes(x.shape.dims, axes, keepdims)

        f = """function (I[{src_ranges}]) -> (O) {{
                   O[{dest_indices}{dest_sep}{dest_ranges}] = >(I[{src_indices}]);
               }}""".format(**subs)

        super(MaxReduce, self).__init__(f, [('I', x)], [('O', tile.Shape(x.shape.dtype, shape))])


def max_reduce(x, axes=None, keepdims=False):
    if not x.shape.ndims:
        return x

    if isinstance(axes, (tuple, list)) and not len(axes):
        # Do nothing if max'ing over an empty axis list
        return x

    return MaxReduce.function(x, axes=axes, keepdims=keepdims)


maximum = tile.maximum


class MaxPool(tile.Operation):
    """
    A standard ML max pooling operator.
    """

    def __init__(self, data, padding, kernel_shape, pads, strides):
        rank = data.shape.ndims - 2
        pads = _extend_pads(pads, rank)
        if not strides:
            strides = tuple(1 for _ in range(rank))
        elif len(strides) != rank:
            raise ValueError('Pool strides length inconsistent with input shape: ' +
                             '{} (rank {}) v {} (rank {})'.format(strides,
                                                                  len(strides), data.shape, rank))
        sym_out_shape = list()
        num_out_shape = list()
        in_idxs = list()
        for i in range(rank):
            sym_out, sym_pad, num_out = pad_compute('L{}'.format(i), data.shape.dims[i + 2],
                                                    kernel_shape[i], strides[i], padding,
                                                    (pads[i], pads[i + rank]) if pads else None)
            sym_out_shape.append(sym_out)
            num_out_shape.append(num_out)
            in_idxs.append('{stride}*x{idx} + k{idx} - {pad}'.format(
                stride=strides[i], idx=i, pad=sym_pad))
        code = """
        function (I[N, C, {in_dims}]) -> (O) {{
            O[n, c, {out_idxs} : N, C, {out_dims}] = >(I[n, c, {in_idxs}]), {pool_bounds};
        }}""".format(
            out_idxs=', '.join(['x{}'.format(i) for i in range(rank)]),
            out_dims=', '.join(sym_out_shape),
            in_idxs=', '.join(in_idxs),
            in_dims=', '.join(['L{}'.format(i) for i in range(rank)]),
            pool_bounds=', '.join(['k{} < {}'.format(i, kernel_shape[i]) for i in range(rank)]))

        outshape = tile.Shape(data.shape.dtype, list(data.shape.dims[0:2]) + num_out_shape)

        super(MaxPool, self).__init__(code, [('I', data)], [('O', outshape)])


max_pool = MaxPool.function


class Mean(tile.Operation):
    """Computes the mean value along some set of axes."""

    def __init__(self, x, axes=None, keepdims=False, floatx=plaidml.DType.FLOAT32):
        if x.shape.dtype == plaidml.DType.BOOLEAN:
            x = cast(x, floatx)

        if axes == None:
            axes = list(range(x.shape.ndims))

        shape, axes, subs = tile.compute_aggregation_axes(x.shape.dims, axes, keepdims)

        subs['mean_ranges'] = '*'.join(['X' + str(i) for i in axes])

        f = """
            function (I[{src_ranges}]) -> (O) {{
                SO[{dest_indices}{dest_sep}{dest_ranges}] = +(I[{src_indices}]);
                 O = SO / ({mean_ranges});
            }}""".format(**subs)

        super(Mean, self).__init__(f, [('I', x)], [('O', tile.Shape(x.shape.dtype, shape))])


def mean(x, axes=None, keepdims=False, floatx=plaidml.DType.FLOAT32):
    if not x.shape.ndims:
        return x

    if isinstance(axes, (tuple, list)) and not len(axes):
        # We're taking the mean across an empty axis list.
        # Keras sometimes does this when squeezing a matrix that doesn't need
        # to be squeezed.
        return x

    return Mean.function(x, axes=axes, keepdims=keepdims, floatx=floatx)


class MinReduce(tile.Operation):
    """Computes the minimum value along some set of axes."""

    def __init__(self, x, axes=None, keepdims=False):
        if axes == None:
            axes = list(range(x.shape.ndims))

        shape, axes, subs = tile.compute_aggregation_axes(x.shape.dims, axes, keepdims)

        f = """function (I[{src_ranges}]) -> (O) {{
                   O[{dest_indices}{dest_sep}{dest_ranges}] = <(I[{src_indices}]);
               }}""".format(**subs)

        super(MinReduce, self).__init__(f, [('I', x)], [('O', tile.Shape(x.shape.dtype, shape))])


def min_reduce(x, axes=None, keepdims=False):
    if not x.shape.ndims:
        return x

    if isinstance(axes, (tuple, list)) and not len(axes):
        # Do nothing if min'ing over an empty axis list
        return x

    return MinReduce.function(x, axes=axes, keepdims=keepdims)


minimum = tile.minimum


class NotEqual(tile.Operation):
    """Elementwise tensor inequality.

    Builds a boolean tensor whose values are true where the corresponding elements of the inputs
    are not equal.
    """

    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs

        if isinstance(rhs, tile.Value):
            shape = tile.Shape(plaidml.DType.BOOLEAN,
                               tile.broadcast_dims(lhs.shape.dims, rhs.shape.dims))
            super(NotEqual, self).__init__('function (L, R) -> (O) { O = (L != R); }',
                                           [('L', lhs), ('R', rhs)], [('O', shape)])
        else:
            shape = tile.Shape(plaidml.DType.BOOLEAN, lhs.shape.dims)
            super(NotEqual, self).__init__('function (L) -> (O) {{ O = (L != {}); }}'.format(rhs),
                                           [('L', lhs)], [('O', shape)])


not_equal = NotEqual.function


class Pow(tile.Operation):
    """An elementwise pow() function."""

    def __init__(self, x, p):
        super(Pow, self).__init__('function (I, P) -> (O) { O = pow(I, P); }',
                                  [('I', x), ('P', p)], [('O', x.shape)])


pow = Pow.function


class Prod(tile.Operation):

    def __init__(self, value, axes=None, keepdims=False, floatx=plaidml.DType.FLOAT32):
        if value.shape.dtype == plaidml.DType.BOOLEAN:
            value = cast(value, floatx)

        if axes is None:
            axes = list(range(value.shape.ndims))

        dims, _, subs = tile.compute_aggregation_axes(value.shape.dims, axes, keepdims)

        code = """
               function (I[{src_ranges}]) -> (O) {{
                   O[{dest_indices}{dest_sep}{dest_ranges}] = *(I[{src_indices}]);
               }}""".format(**subs)

        super(Prod, self).__init__(code, [('I', value)],
                                   [('O', tile.Shape(value.shape.dtype, dims))])


def prod(value, axes=None, keepdims=False, floatx=plaidml.DType.FLOAT32):
    if isinstance(value, (tuple, list)):
        return functools.reduce(lambda x, y: x * y, value)
    if not value.shape.ndims:
        return value
    if isinstance(axes, (tuple, list)) and not len(axes):
        # We're taking the product across an empty axis list.
        return value
    return Prod.function(value, axes=axes, keepdims=keepdims, floatx=floatx)


class Relu(tile.Operation):
    """A Rectified Linear Unit."""

    def __init__(self, x, alpha=None, max_value=None):
        if (alpha is not None) and (max_value is not None):
            # Alpha with a max_value; cap a hand-coded relu.
            code = """
                   function (X, Alpha, MaxValue) -> (Y) {
                       M = (X < 0.0 ? Alpha*X : X);
                       Y = (M < MaxValue ? M : MaxValue);
                   }"""
        elif alpha is not None:
            # Alpha with no max_value; use a hand-coded relu.
            code = 'function (X, Alpha) -> (Y) { Y = (X < 0 ? Alpha*X : X); }'
        elif max_value is not None:
            # No alpha, but a max_value; cap the builtin relu.
            code = """
                   function (X, MaxValue) -> (Y) {
                       M = (X < 0.0 ? 0.0 : X);
                       Y = (M < MaxValue ? M : MaxValue);
                   }"""
        else:
            # Neither alpha nor max_value; use the builtin relu.
            code = 'function (X) -> (Y) { Y = relu(X); }'

        inputs = [('X', x)]

        if alpha is not None:
            inputs.append(('Alpha', alpha))

        if max_value is not None:
            inputs.append(('MaxValue', max_value))

        super(Relu, self).__init__(code, inputs, [('Y', x.shape)])


relu = Relu.function


class Reshape(tile.Operation):
    """
    Reshapes a tensor, without changing the type or number of elements.
    """

    def __init__(self, x, dims):
        dims = list(dims)
        neg_idx = None
        for idx, dim in enumerate(dims):
            if isinstance(dim, tile.Value):
                continue
            if dim == 0 or dim is None:
                dims[idx] = x.shape.dims[idx]
            elif dim == -1:
                if neg_idx:
                    raise tile.LogicError(
                        'At most one dimension of size -1 may be provided in Reshape')
                neg_idx = idx
                dims[idx] = 1  # Just to simplify the size computation later
        if neg_idx is not None:
            # Compute the value to use for the -1 dimension in the
            # output shape, by making it what it needs to be in order
            # to preserve the correct number of elements in the
            # tensor.
            #
            # This code is a little tricky because symbolic values
            # (e.g. the batch size in a typical neural network) may
            # appear in both the original shape and the target shape.
            # Naively multiplying the original shape's dimensions and
            # dividing by the target shape's dimensions (excluding the
            # -1 dimension) would produce a symbolic value.
            #
            # So:
            #
            # We scan the input dimensions, counting the number of
            # instances of each symbolic size encountered and
            # multiplying together the non-symbolic sizes into the
            # numerator.
            #
            # We then scan the output dimensions.  Where there's a
            # symbolic size, we check and see if we have a count for
            # it, and decrement the count if we do.  Otherwise -- if
            # we don't have a count for it, or if it's not symbolic --
            # we multiply it into the denominator.
            #
            # We then take the remaining symbolic input dimensions,
            # and multiply them into the numerator -- these are the
            # dimensions that haven't been cancelled out.
            #
            # And then the size of the -1 dimension is just numerator
            # / denominator; if there are any remaining uncancelled
            # symbolic dimension sizes, the output will be symbolic,
            # but otherwise we'll come out with a concrete dimension
            # size.

            num = 1
            syms = defaultdict(int)
            for dim in x.shape.dims:
                if isinstance(dim, tile.Value):
                    syms[dim] += 1
                else:
                    num *= dim
            den = 1
            for dim in dims:
                if isinstance(dim, tile.Value) and syms[dim] > 0:
                    syms[dim] -= 1
                else:
                    den *= dim
            for sym, count in syms.items():
                for _ in range(count):
                    num *= sym
            dims[neg_idx] = num // den

        inputs = [('I', x)]
        dstrs = list(dims)
        for idx, dim in enumerate(dstrs):
            if isinstance(dim, tile.Value):
                dname = 'D{}'.format(idx)
                inputs.append((dname, dim))
                dstrs[idx] = dname

        super(Reshape, self).__init__('function ({}) -> (O) {{ O = reshape(I, {}); }}'.format(
            ', '.join(inp[0] for inp in inputs), ', '.join([str(d) for d in dstrs])), inputs,
                                      [('O', tile.Shape(x.shape.dtype, dims))])


reshape = Reshape.function

ShapeOf = tile.ShapeOf

shape_of = tile.shape_of


def sigmoid(data):
    """Elementwise sigmoid."""
    return tile.unary_op(data, 'sigmoid(I)', 'Sigmoid')


def sin(data):
    """Elementwise sine."""
    return tile.unary_op(data, 'sin(I)', 'Sine')


class SliceTensor(tile.Operation):
    """
    Implements tensor slicing.
    """

    def __init__(self, data, axes=None, ends=None, starts=None):
        if not ends or not starts:
            raise tile.LogicError('Slice requires starts and ends to be set')
        if len(starts) != len(ends):
            raise tile.LogicError('Slice requires starts and ends for all sliced axes')
        if not axes:
            axes = range(len(starts))

        in_dims = ['D{}'.format(d) for d in range(data.shape.ndims)]
        out_dims = list(in_dims)
        in_idxs = ['d{}'.format(d) for d in range(data.shape.ndims)]
        out_idxs = list(in_idxs)
        shape_dims = list(data.shape.dims)

        for axis, start, end in zip(axes, starts, ends):
            clamped_end = tile.minimum(end, data.shape.dims[axis])
            clamped_start = tile.minimum(start, data.shape.dims[axis])
            if isinstance(clamped_start, tile.Value):
                clamped_start_str = 'min({}, D{})'.format(start, axis)
            else:
                clamped_start_str = str(clamped_start)
            if isinstance(clamped_end, tile.Value):
                clamped_end_str = 'min({}, D{})'.format(end, axis)
            else:
                clamped_end_str = str(clamped_end)
            delta = clamped_end - clamped_start
            if isinstance(clamped_end, tile.Value) or isinstance(clamped_start, tile.Value):
                delta_str = '{}-{}'.format(clamped_end_str, clamped_start_str)
            else:
                delta_str = str(clamped_end - clamped_start)

            if end > 0:
                out_dims[axis] = delta_str
                shape_dims[axis] = delta
            elif start - end > 0:
                out_dims[axis] = 'D{}+({})'.format(axis, delta_str)
                shape_dims[axis] += delta
            if start:
                in_idxs[axis] = 'd{}+{}'.format(axis, clamped_start_str)

        code = """
        function (I[{in_dims}]) -> (O) {{
            O[{out_idxs} : {out_dims}] = =(I[{in_idxs}]);
        }}""".format(
            in_dims=', '.join(in_dims),
            out_dims=', '.join(out_dims),
            in_idxs=', '.join(in_idxs),
            out_idxs=', '.join(out_idxs))

        outshape = tile.Shape(data.shape.dtype, shape_dims)

        super(SliceTensor, self).__init__(code, [('I', data)], [('O', outshape)])


slice_tensor = SliceTensor.function


class Softmax(tile.Operation):
    """
    Implements a standard ML softmax.
    """

    def __init__(self, data):
        if data.shape.ndims != 2:
            raise NotImplementedError(
                'Softmax with a non-two-dimensional tensor is not currently implemented')

        code = """
        function (I[X, Y]) -> (O) {
            O = builtin_softmax(I, X, Y);
        }"""

        super(Softmax, self).__init__(code, [('I', data)], [('O', data.shape)])


def softmax(x, axis=None):
    if x.shape.ndims == 2:
        return Softmax.function(x)
    if axis is None:
        axis = 1
    full_dims = x.shape.dims
    if axis == 0:
        group = 1
    else:
        group = functools.reduce(lambda x, y: x * y, x.shape.dims[:axis])
    if axis == len(x.shape.dims):
        values = 1
    else:
        values = functools.reduce(lambda x, y: x * y, x.shape.dims[axis:])
    flat_x = reshape(x, (group, values))
    result = Softmax.function(flat_x)
    return reshape(result, full_dims)


class Sqrt(tile.Operation):
    """
    Computes the elementwise square root of a value.
    """

    def __init__(self, x):
        super(Sqrt, self).__init__("""
            function (I) -> (O) {
                IC = (I < 0 ? 0 : I);
                O = sqrt(IC);
            }""", [('I', x)], [('O', x.shape)])


sqrt = Sqrt.function


def squeeze(x, axes):
    dims = [x.shape.dims[axis] for axis in range(x.shape.ndims) if axis not in axes]
    return reshape(x, dims)


class Summation(tile.Operation):
    """
    Sums an input value along some set of axes.
    """

    def __init__(self, value, axes=None, keepdims=False, floatx=plaidml.DType.FLOAT32):
        if value.shape.dtype == plaidml.DType.BOOLEAN:
            value = cast(value, floatx)

        if axes is None:
            axes = list(range(value.shape.ndims))

        dims, _, subs = tile.compute_aggregation_axes(value.shape.dims, axes, keepdims)

        code = """
               function (I[{src_ranges}]) -> (O) {{
                   O[{dest_indices}{dest_sep}{dest_ranges}] = +(I[{src_indices}]);
               }}""".format(**subs)

        super(Summation, self).__init__(code, [('I', value)],
                                        [('O', tile.Shape(value.shape.dtype, dims))])


def summation(value, axes=None, keepdims=False, floatx=plaidml.DType.FLOAT32):
    if not value.shape.ndims:
        return value
    if isinstance(axes, (tuple, list)) and not len(axes):
        # We're taking the sum across an empty axis list.
        return value
    return Summation.function(value, axes=axes, keepdims=keepdims, floatx=floatx)


def tanh(data):
    """Elementwise hyperbolic tangent."""
    return tile.unary_op(data, 'tanh(I)', 'Tanh')


def unsqueeze(x, axes):
    src_idx = 0
    dims = []
    for axis in range(len(x.shape.dims) + len(axes)):
        if axis in axes:
            dims.append(1)
        else:
            dims.append(x.shape.dims[src_idx])
            src_idx += 1
    return reshape(x, dims)


class Variance(tile.Operation):

    def __init__(self, x, axes=None, keepdims=False, floatx=plaidml.DType.FLOAT32):
        # This closely follows the implementation of the mean method
        # This computes the *uncorrected* sample variance (i.e. denominator
        # = n rather than = n-1) to match tensorflow
        if x.shape.dtype == plaidml.DType.BOOLEAN:
            x = cast(x, floatx)

        if not x.shape.ndims:
            return x

        if axes == None:
            axes = list(range(x.shape.ndims))

        shape, axes, subs = tile.compute_aggregation_axes(x.shape.dims, axes, keepdims)

        subs['prod_src_ranges'] = '*'.join(['X' + str(i) for i in axes])
        subs['mean_ranges'] = ', '.join(['Y' + str(i) for i in range(x.shape.ndims)])

        m = mean(x, axes, True, floatx)

        # TODO: Might be possible to write this more efficiently
        f = """
            function (I[{src_ranges}], M[{mean_ranges}]) -> (O) {{
                DIFF_SQ = (I - M) * (I - M);
                SUM[{dest_indices}{dest_sep}{dest_ranges}] = +(DIFF_SQ[{src_indices}]);
                O = SUM / ({prod_src_ranges});
            }}""".format(**subs)

        super(Variance, self).__init__(f, [('I', x), ('M', m)],
                                       [('O', tile.Shape(x.shape.dtype, shape))])


variance = Variance.function
