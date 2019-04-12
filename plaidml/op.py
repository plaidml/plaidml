# Copyright 2018 Intel Corporation.
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
import numpy as np
import plaidml
from plaidml import tile
import six
import math


class AutoPadding(Enum):
    EXPLICIT = 1
    VALID = 2
    SAME_UPPER = 3
    SAME_LOWER = 4


class ConvolutionDataFormat(Enum):
    CHANNELS_FIRST = 1
    CHANNELS_LAST = 2


class ConvolutionKernelFormat(Enum):
    CHANNELS_FIRST = 1
    CHANNELS_LAST = 2


class ConvolutionGrouping(Enum):
    NONE = 1
    MAX = 2
    EXPLICIT = 3
    AUTO = 4


class GroupedChannelFormat(Enum):
    FullOutGroupIn = 1
    GroupGroupOut = 2
    GroupGroupOutGroupIn = 3


class ConvIndex(Enum):
    ci = 1  # input channel (overall)
    co = 2  # output channel (overall
    g = 3  # group
    gci = 4  # input channel (within-group)
    gco = 5  # output channel (within-group)
    k = 6  # spatial location (kernel)
    n = 7  # batch
    x = 8  # spatial location (data)


class PoolDataFormat(Enum):
    NXC = 1
    NCX = 2


class PoolMode(Enum):
    MAX = 1
    AVG = 2


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
        sym_output_size = '({sym} + {pre} + {post} - {fs} + {s}) / {s}'.format(sym=sym,
                                                                               pre=pads[0],
                                                                               post=pads[1],
                                                                               fs=filter_size,
                                                                               s=stride)
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
        sym_padding_before = expr.format(symout=sym_output_size,
                                         s=stride,
                                         fs=filter_size,
                                         syminp=sym)
    else:
        raise Exception('Invalid padding: ' + str(padding))
    if not isinstance(num_out_size, tile.Value) and num_out_size < 0:
        raise Exception(
            'Invalid output size computed for convolution: num_out_size={}'.format(num_out_size))
    return (sym_output_size, sym_padding_before, num_out_size)


class _ConvolutionStringFormatter:
    """Produces the strings needed to write Tile code for a convolution.

    This class provides strings for the dimensions, indices, and supporting code
    (i.e. padding and reshapes) for a convolution. The convolution operation
    then only provides skeleton code specifying how the various tensors interact
    and uses this class's functions when it needs to specify indices or dims for
    a tensor or when it needs supporting code.

    This class manages the complex interactions between the various tensor
    formats and the various types of convolutions in constructing these strings.

    Member function categories
    --------------------------
     * Index lookup (get_I_axis, get_K_axis, get_O_axis): Given the variable
    name of an index (as a ConvIndex), returns the axis number (or a list of
    axis numbers for k and x) of that index
    for the indicated tensor.
     * Needs reshape (kernel_needs_reshape, output_needs_reshape): Return a
    boolean: Is a reshape required for this tensor? (i.e. because the shape most
    useful for the Tile contraction is not the same as the expected format of
    the tensor as specified by the format parameters.
     * Reshape renames (Kitrn, Oitrn): Return the appropriate internal name for
    the tensor (e.g. either 'K' or 'Kitrn' -- 'K' if no reshape is needed for
    the kernel and 'Kitrn' if a reshape is needed)
     * Padding (pad_amount, padding_str): Values and code blocks related to the
    spatial padding needed to align the kernel to the data.
     * Parameter lists ([I/K/Ki/O/Oi]_[batch/channel/spatial]_[dim/dims/
    idx/idxs][_numeric]): Returns a list of strings (or, in the `_numeric` case,
    a list of ints/SymbolicDims) giving individual parameters.
     * Parameter code blocks ([I/K/Ki/O/Oi]_dims, [I/Ki/Oi]_idxs): Return fully
    formatted strings ready to plug in to the convolution Tile code to specify
    dimension or index parameters for the specified tensor.
     * Outshape (O_shape_tuple_numeric): Computes dims used to construct the
    tile.Shape of the output tensor.

    Tile tensor variable names
    --------------------------
     * 'I': input data
     * 'K': kernel (as input by caller)
     * 'Kitrn': internal kernel (i.e. kernel reshaped for main contraction)
     * 'O': output (as returned to caller -- possibly reshaped from Oitrn)
     * 'Oitrn': internal output (i.e. output as produced by main contraction)

    Tile dimension meanings
    -----------------------
     * 'CI': input channels (total)
     * 'CO': output channels (total)
     * 'G': groups
     * 'GCI': input channels per group
     * 'GCO': output channels per group
     * 'L#': data spatial dimension (number #)
     * 'LK#': kernel spatial dimension (number #)
     * 'N': batch size
     * 'Pad#': padding amount in spatial dimension number #

    Tile index meanings
    -------------------
     * 'ci': input channel index (overall)
     * 'co': output channel index (overall)
     * 'g': group index
     * 'gci': within-group input channel index
     * 'gco': within-group output channel index
     * 'k#': kernel spatial index (number #)
     * 'n': batch index
     * 'x#': data spatial index (number #)

    How grouped convolutions work
    -----------------------------
    In a standard convolution, at fixed spatial locations in both the input data
    and the kernel and at a fixed batch element, input channels are densely
    mapped to output channels: each input channel affects every output channel:

    IN CHANNELS:        o  o
                        |\/|
                        |/\|
    OUT CHANNELS:       o  o

    In a grouped convolution, channels are split into groups, and input channels
    affect output channels if and only if they're in the same group:

    IN CHANNELS:        o  o    o  o    o  o
                        |\/|    |\/|    |\/|
                        |/\|    |/\|    |/\|
    OUT CHANNELS:       o  o    o  o    o  o
    (This is ONE convolution with 6 input and 6 output channels.)

    Grouped convolutions with only one input per group are often called
    depthwise or channel-wise convolutions:

    IN CHANNELS:           o       o       o
                          /|\     /|\     /|\
                         / | \   / | \   / | \
    OUT CHANNELS:       o  o  o o  o  o o  o  o
    (A depthwise convolution with multiplicity 3, i.e. 3 output channels per
    input channel.)
    """

    def __init__(
            self,
            rank,
            in_shape,
            kernel_shape,
            strides,
            padding,
            dilation_rate,
            data_format,
            kernel_format,
            pads=None,
            grouping=ConvolutionGrouping.NONE,
            groups=None,
            group_format=None,
            transposed=False,
            expected_output_shape=None,
    ):
        """
        Constructs a string formatter for a specific convolution.

        Args:
            rank (int): The number of spatial dimensions of the convolution
            in_shape (tuple of ints): All the dimensions of 'I' in the order used by the caller
            kernel_shape (tuple of ints): All the dimensions of 'K' in the order used by the caller
            strides (tuple of ints): The stride for each spatial dimension
            padding (AutoPadding): The padding style to use
            dilation_rate (tuple of ints): The kernel spacing for each spatial dimension
            data_format (ConvolutionDataFormat): The parameter order style of 'I'
            kernel_format (ConvolutionKernelFormat): The parameter order & semantics style of 'K'
            pads (tuple of (int, int)s or None): For explicit padding, the pre- and post-padding
            grouping (ConvolutionGrouping): Whether this convolution is grouped and if so how. NONE
                means standard ungrouped convolution, MAX means channelwise convolution, EXPLICIT
                means the number of groups will be provided in the `groups` parameter, AUTO means
                the number of groups will be inferred from in_shape, kernel_shape, and group_format.
                Note that ONNX's kernel format is consistent between ungrouped and grouped
                convolutions and so AUTO always works; but Keras' kernel format is not consistent
                between ungrouped and channelwise convolutions. Keras therefore must explicitly
                pass NONE or MAX. In Keras EXPLICIT is only possible with custom code for
                non-standard (to Keras) kernel shapes.
            groups (int or None): The number of groups for explicit grouping
            group_format (GroupedChannelFormat): The channel order & semantics 'K' (if grouping
                isn't NONE)
            transposed (Boolean): Is this a transposed convolution? (i.e., one that takes 'O' and
                'K' as inputs and produces 'I' as output)
            expected_output_shape (tuple of ints/SymbolicDims or None): The shape of the output
                tensor the caller expects. This shape is also computed from the other parameters,
                and if this is not None, it is verified that these two versions of the shape match
        """
        self.rank = rank
        self.in_shape = in_shape
        self.kernel_shape = kernel_shape
        self.strides = strides
        self.padding = padding
        self.dilation_rate = dilation_rate
        self.data_format = data_format
        self.kernel_format = kernel_format
        self.pads = pads
        self.grouping = grouping
        self.groups = groups
        self.group_format = group_format
        self.transposed = transposed
        self.expected_output_shape = expected_output_shape

        self._O_spatial_dims = None
        self._pad_amount = None
        self._O_spatial_dims_numeric = None
        self._assertion = ''

        if self.data_format not in [
                ConvolutionDataFormat.CHANNELS_FIRST, ConvolutionDataFormat.CHANNELS_LAST
        ]:
            raise ValueError('Unknown data_format \'{}\''.format(self.data_format))
        if self.kernel_format not in [
                ConvolutionKernelFormat.CHANNELS_FIRST, ConvolutionKernelFormat.CHANNELS_LAST
        ]:
            raise ValueError('Unknown kernel_format \'{}\''.format(self.kernel_format))
        if self.grouping not in [
                ConvolutionGrouping.NONE, ConvolutionGrouping.MAX, ConvolutionGrouping.EXPLICIT,
                ConvolutionGrouping.AUTO
        ]:
            raise ValueError('Unknown grouping \'{}\''.format(self.grouping))

        self._convert_grouping_type()
        self._verify_expected_output_shape()
        if self.grouping != ConvolutionGrouping.NONE:
            if self.transposed:
                raise NotImplementedError('Grouped transposed convolutions not implemented.')
            if self.group_format not in [
                    GroupedChannelFormat.FullOutGroupIn, GroupedChannelFormat.GroupGroupOut,
                    GroupedChannelFormat.GroupGroupOutGroupIn
            ]:
                raise ValueError('Unknown group_format \'{}\''.format(self.group_format))

    def get_I_axis(self, sem_var):
        """Get the axis of I corresponding to the variable sem_var.

        Args:
            sem_var (ConvIndex): Variable to get the axis of

        Returns:
            int or list: The index(es) of the dimension(s) containing this
                         variable (i.e. axis) in the initial input tensor I. A
                         list for spatial dims, otherwise an int.
        """
        if self.data_format == ConvolutionDataFormat.CHANNELS_FIRST:
            if sem_var == ConvIndex.n:
                return 0
            if sem_var == ConvIndex.ci:
                return 1
            if sem_var == ConvIndex.x:
                return [i + 2 for i in range(self.rank)]
        elif self.data_format == ConvolutionDataFormat.CHANNELS_LAST:
            if sem_var == ConvIndex.n:
                return 0
            if sem_var == ConvIndex.x:
                return [i + 1 for i in range(self.rank)]
            if sem_var == ConvIndex.ci:
                return self.rank + 1
        else:
            raise ValueError('Unknown data format \'{}\''.format(self.data_format))
        raise ValueError('Unknown input variable name \'{}\''.format(sem_var))

    def get_K_axis(self, sem_var):
        """Get the axis of K corresponding to the variable sem_var.

        Note that which variables are available will depend on the grouping type
        and grouped channel format. Will raise an exception if sem_var is
        unavailable.

        Args:
            sem_var (ConvIndex): Variable to get the axis of

        Returns:
            int or list: The index(es) of the dimension(s) containing this
                         variable (i.e. axis) in the initial kernel tensor K. A
                         list for spatial dims, otherwise an int.
        """
        wrong_group_msg = 'Variable {} not a kernel dim for this grouping (type {}, format {})'
        pos_among_channels = None
        if self.grouping == ConvolutionGrouping.NONE:
            channel_count = 2
            # For this case channel order depends on kernel format and will be
            # calculated later.
        else:
            if self.group_format == GroupedChannelFormat.FullOutGroupIn:
                channel_count = 2
                if sem_var == ConvIndex.co:
                    pos_among_channels = 0
                elif sem_var == ConvIndex.gci:
                    pos_among_channels = 1
                elif sem_var in [ConvIndex.ci, ConvIndex.g, ConvIndex.gco]:
                    # These channel types don't appear with this grouped channel format
                    msg = wrong_group_msg.format(sem_var, self.grouping, self.group_format)
                    raise ValueError(msg)
            elif self.group_format == GroupedChannelFormat.GroupGroupOut:
                channel_count = 2
                if sem_var == ConvIndex.g:
                    pos_among_channels = 0
                elif sem_var == ConvIndex.gco:
                    pos_among_channels = 1
                elif sem_var in [ConvIndex.ci, ConvIndex.co, ConvIndex.ci]:
                    # These channel types don't appear with this grouped channel format
                    msg = wrong_group_msg.format(sem_var, self.grouping, self.group_format)
                    raise ValueError(msg)
            elif self.group_format == GroupedChannelFormat.GroupGroupOutGroupIn:
                channel_count = 3
                if sem_var == ConvIndex.g:
                    pos_among_channels = 0
                elif sem_var == ConvIndex.gco:
                    pos_among_channels = 1
                elif sem_var == ConvIndex.gci:
                    pos_among_channels = 2
                elif sem_var in [ConvIndex.ci, ConvIndex.co]:
                    # These channel types don't appear with this grouped channel format
                    msg = wrong_group_msg.format(sem_var, self.grouping, self.group_format)
                    raise ValueError(msg)
            else:
                raise ValueError('Unknown group format \'{}\''.format(self.group_format))
        if self.kernel_format == ConvolutionKernelFormat.CHANNELS_FIRST:
            if pos_among_channels is not None:
                return pos_among_channels
            if sem_var == ConvIndex.k:
                return [i + channel_count for i in range(self.rank)]
            if self.grouping == ConvolutionGrouping.NONE:
                if sem_var == ConvIndex.co:
                    return 0
                if sem_var == ConvIndex.ci:
                    return 1
                if sem_var in [ConvIndex.g, ConvIndex.gci, ConvIndex.gco]:
                    # These channel types don't appear with this grouped channel format
                    msg = wrong_group_msg.format(sem_var, self.grouping, self.group_format)
                    raise ValueError(msg)
        elif self.kernel_format == ConvolutionKernelFormat.CHANNELS_LAST:
            if pos_among_channels is not None:
                return self.rank + pos_among_channels
            if sem_var == ConvIndex.k:
                return range(self.rank)
            if self.grouping == ConvolutionGrouping.NONE:
                if sem_var == ConvIndex.ci:
                    return self.rank
                if sem_var == ConvIndex.co:
                    return self.rank + 1
                if sem_var in [ConvIndex.g, ConvIndex.gci, ConvIndex.gco]:
                    # These channel types don't appear with this grouped channel format
                    msg = wrong_group_msg.format(sem_var, self.grouping, self.group_format)
                    raise ValueError(msg)
            if sem_var == ConvIndex.n:
                return 0
            if sem_var == ConvIndex.ci:
                return self.rank + 1
        else:
            raise ValueError('Unknown data format \'{}\''.format(self.data_format))
        raise ValueError('Unknown kernel variable name \'{}\''.format(sem_var))

    def get_O_axis(self, sem_var):
        """Get the axis of O corresponding to the variable sem_var.

        Args:
            sem_var (ConvIndex): Variable to get the axis of

        Returns:
            int or list: The index(es) of the dimension(s) containing this
                         variable (i.e. axis) in the final output tensor O. A
                         list for spatial dims, otherwise an int.
        """
        if self.data_format == ConvolutionDataFormat.CHANNELS_FIRST:
            if sem_var == ConvIndex.n:
                return 0
            if sem_var == ConvIndex.co:
                return 1
            if sem_var == ConvIndex.x:
                return [i + 2 for i in range(self.rank)]
        elif self.data_format == ConvolutionDataFormat.CHANNELS_LAST:
            if sem_var == ConvIndex.n:
                return 0
            if sem_var == ConvIndex.x:
                return [i + 1 for i in range(self.rank)]
            if sem_var == ConvIndex.co:
                return self.rank + 1
        else:
            raise ValueError('Unknown data format \'{}\''.format(self.data_format))
        raise ValueError('Unknown input variable name \'{}\''.format(sem_var))

    def _compute_padding(self):
        self._O_spatial_dims = list()
        self._pad_amount = list()
        self._O_spatial_dims_numeric = list()
        for i in range(self.rank):
            if self.transposed:
                sym_out, sym_pad, num_out = pad_compute(
                    'D{}'.format(i),
                    self.in_shape[self.get_I_axis(ConvIndex.x)[i]],
                    self.dilation_rate[i] *
                    (self.kernel_shape[self.get_K_axis(ConvIndex.k)[i]] - 1) + 1,
                    self.strides[i],
                    self.padding,
                    (self.pads[i], self.pads[i + self.rank]) if self.pads else None,
                )
            else:
                sym_out, sym_pad, num_out = pad_compute(
                    'L{}'.format(i),
                    self.in_shape[self.get_I_axis(ConvIndex.x)[i]],
                    self.dilation_rate[i] *
                    (self.kernel_shape[self.get_K_axis(ConvIndex.k)[i]] - 1) + 1,
                    self.strides[i],
                    self.padding,
                    (self.pads[i], self.pads[i + self.rank]) if self.pads else None,
                )
            self._O_spatial_dims.append(sym_out)
            self._pad_amount.append(sym_pad)
            self._O_spatial_dims_numeric.append(num_out)

    def pad_amount(self):
        """List of strings giving spatial padding constants"""
        if self._pad_amount is None:
            self._compute_padding()
        return self._pad_amount

    def padding_str(self):
        padding_list = ['Pad{} = {};'.format(i, self.pad_amount()[i]) for i in range(self.rank)]
        return ''.join(p + '\n    ' for p in padding_list)

    def assertion(self):
        return self._assertion

    def _convert_grouping_type(self):
        """Converts EXPLICIT and MAX grouping to AUTO.

        Assertions can be added here to ensure the requested type is run."""
        if self.grouping == ConvolutionGrouping.EXPLICIT:
            if not isinstance(self.groups, six.integer_types):
                raise ValueError(
                    'Must provide integer number of groups when using explicit convolution grouping (received {})'
                    .format(self.groups))
            if self.groups == 1:
                self.grouping = ConvolutionGrouping.NONE
            else:
                self.grouping = ConvolutionGrouping.AUTO
                self._assertion = 'Assert = assert_group_count({} == {});\n    '.format(
                    self._G(), self.groups)
        elif self.grouping == ConvolutionGrouping.MAX:
            self.grouping = ConvolutionGrouping.AUTO
            self._assertion = 'Assert = assert_group_count({} == {});\n    '.format(
                self._G(), self._CI())

    def _verify_expected_output_shape(self):
        if self.expected_output_shape is None:
            # Nothing to check
            return
        if self.grouping != ConvolutionGrouping.NONE:
            raise ValueError('Grouped convolutions do not currently support'
                             'expected_output_shape')
        # Confirm that the output shape is consistent with the rest of the convolution
        computed_output_shape = [0] * (self.rank + 2)
        computed_output_shape[self.get_O_axis(ConvIndex.n)] = self.O_batch_dim_numeric()[0]
        computed_output_shape[self.get_O_axis(ConvIndex.co)] = self.kernel_shape[self.get_K_axis(
            ConvIndex.co)]
        for i in range(self.rank):
            computed_output_shape[self.get_O_axis(
                ConvIndex.x)[i]] = self.O_spatial_dims_numeric()[i]
        for i in range(self.rank + 2):
            if (not isinstance(computed_output_shape[i], tile.Value) and
                    not isinstance(self.expected_output_shape[i], tile.Value) and
                    computed_output_shape[i] != self.expected_output_shape[i]):
                raise ValueError('Expected convolution output of shape {}, received {}'.format(
                    self.expected_output_shape, computed_output_shape))

    def _CI(self):
        """Tile variable or formula for total input channels"""
        if self.grouping == ConvolutionGrouping.NONE:
            return 'CI'
        elif self.grouping == ConvolutionGrouping.AUTO:
            if self.group_format == GroupedChannelFormat.FullOutGroupIn:
                return 'CI'
            elif self.group_format == GroupedChannelFormat.GroupGroupOut:
                return 'CI'
            elif self.group_format == GroupedChannelFormat.GroupGroupOutGroupIn:
                return 'CI'
            else:
                raise ValueError('Unknown grouped channel format {}'.format(self.group_format))
        else:
            raise ValueError('Unrecognized grouping type \'{}\''.format(self.grouping))

    def _CO(self):
        """Tile variable or formula for total output channels"""
        if self.grouping == ConvolutionGrouping.NONE:
            return 'CO'
        elif self.grouping == ConvolutionGrouping.AUTO:
            if self.group_format == GroupedChannelFormat.FullOutGroupIn:
                return 'CO'
            elif self.group_format == GroupedChannelFormat.GroupGroupOut:
                return '(G*GCO)'
            elif self.group_format == GroupedChannelFormat.GroupGroupOutGroupIn:
                return '(G*GCO)'
            else:
                raise ValueError('Unknown grouped channel format {}'.format(self.group_format))
        else:
            raise ValueError('Unrecognized grouping type \'{}\''.format(self.grouping))

    def _G(self):
        """Tile variable or formula for number of groups"""
        if self.grouping == ConvolutionGrouping.NONE:
            raise LogicError("Requested per-group out channels for ungrouped convolution.")
        elif self.grouping == ConvolutionGrouping.AUTO:
            if self.group_format == GroupedChannelFormat.FullOutGroupIn:
                return '(CI/GCI)'
            elif self.group_format == GroupedChannelFormat.GroupGroupOut:
                return 'G'
            elif self.group_format == GroupedChannelFormat.GroupGroupOutGroupIn:
                return 'G'
            else:
                raise ValueError('Unknown grouped channel format {}'.format(self.group_format))
        else:
            raise ValueError('Unrecognized grouping type \'{}\''.format(self.grouping))

    def _GCI(self):
        """Tile variable or formula for input channels per group"""
        if self.grouping == ConvolutionGrouping.NONE:
            raise LogicError("Requested per-group input channels for ungrouped convolution.")
        elif self.grouping == ConvolutionGrouping.AUTO:
            if self.group_format == GroupedChannelFormat.FullOutGroupIn:
                return 'GCI'
            elif self.group_format == GroupedChannelFormat.GroupGroupOut:
                return '(CI/G)'
            elif self.group_format == GroupedChannelFormat.GroupGroupOutGroupIn:
                return 'GCI'
            else:
                raise ValueError('Unknown grouped channel format {}'.format(self.group_format))
        else:
            raise ValueError('Unknown grouping type \'{}\''.format(self.grouping))

    def _GCO(self):
        """Tile variable or formula for out channels per group"""
        if self.grouping == ConvolutionGrouping.NONE:
            raise LogicError("Requested per-group out channels for ungrouped convolution.")
        elif self.grouping == ConvolutionGrouping.AUTO:
            if self.group_format == GroupedChannelFormat.FullOutGroupIn:
                return '(CO*GCI/CI)'
            elif self.group_format == GroupedChannelFormat.GroupGroupOut:
                return 'GCO'
            elif self.group_format == GroupedChannelFormat.GroupGroupOutGroupIn:
                return 'GCO'
            else:
                raise ValueError('Unknown grouped channel format {}'.format(self.group_format))
        else:
            raise ValueError('Unrecognized grouping type \'{}\''.format(self.grouping))

    def kernel_needs_reshape(self):
        """Whether the kernel tensor need to be reshaped before it's used"""
        if self.grouping == ConvolutionGrouping.NONE:
            return False
        elif self.grouping == ConvolutionGrouping.AUTO:
            if self.group_format == GroupedChannelFormat.FullOutGroupIn:
                return True
            elif self.group_format == GroupedChannelFormat.GroupGroupOut:
                return True
            elif self.group_format == GroupedChannelFormat.GroupGroupOutGroupIn:
                return False
            else:
                raise ValueError('Unknown grouped channel format {}'.format(self.group_format))
        else:
            raise RuntimeError(
                'ConvolutionGrouping should have been converted to NONE or AUTO, but received {}'.
                format(self.grouping))

    def output_needs_reshape(self):
        """Whether the output tensor need to be reshaped before it's returned"""

        if self.grouping == ConvolutionGrouping.NONE:
            return False
        return True

    def Kitrn(self):
        if self.kernel_needs_reshape():
            return 'Kitrn'
        else:
            return 'K'

    def Oitrn(self):
        if self.output_needs_reshape():
            return 'Oitrn'
        else:
            return 'O'

    def I_batch_dim(self):
        """String list giving the input batch dimension name."""
        return ['N']

    def I_batch_idx(self):
        """String list giving the input batch index name."""
        return ['n']

    def I_channel_dim(self):
        """String list giving the input channel dimension name."""
        return ['CI']

    def I_channel_idx(self):
        """String list giving the input channel index name."""
        if self.grouping == ConvolutionGrouping.NONE:
            return ['ci']
        elif self.grouping == ConvolutionGrouping.AUTO:
            return ['g * ({}) + gci'.format(self._GCI())]
        else:
            raise RuntimeError(
                'ConvolutionGrouping should have been converted to NONE or AUTO, but received {}'.
                format(self.grouping))

    def I_spatial_dims(self):
        """String list giving the input spatial dimension names."""
        if self.transposed:
            return ['D{}'.format(i) for i in range(self.rank)]
        else:
            return ['L{}'.format(i) for i in range(self.rank)]

    def I_spatial_idxs(self):
        """String list giving the input spatial index names."""
        strs = [{
            's': self.strides[i],
            'idx': i,
            'd': self.dilation_rate[i],
            'p': 'Pad{}'.format(i),
        } for i in range(self.rank)]
        return ['{s}*x{idx} + {d}*k{idx} - {p}'.format(**strs[i]) for i in range(self.rank)]

    def K_channel_dims(self):
        """String list giving the kernel channel dimension names

        Returns a list of strings. They are to be used in the function header of
        the Tile convolution code for the channel dimension(s) of the kernel.
        They are to be used in the same order as the list."""
        if self.grouping == ConvolutionGrouping.NONE:
            if self.kernel_format == ConvolutionKernelFormat.CHANNELS_FIRST:
                return ['CO', 'CI']
            elif self.kernel_format == ConvolutionKernelFormat.CHANNELS_LAST:
                return ['CI', 'CO']
            else:
                raise ValueError('Unknown kernel format {}'.format(self.kernel_format))
        elif self.grouping == ConvolutionGrouping.AUTO:
            if self.group_format == GroupedChannelFormat.FullOutGroupIn:
                return ['CO', 'GCI']
            elif self.group_format == GroupedChannelFormat.GroupGroupOut:
                return ['G', 'GCO']
            elif self.group_format == GroupedChannelFormat.GroupGroupOutGroupIn:
                return ['G', 'GCO', 'GCI']
            else:
                raise ValueError('Unknown grouped channel format {}'.format(self.group_format))
        else:
            raise ValueError('Unknown grouping type \'{}\''.format(self.grouping))

    def K_spatial_dims(self):
        """String list giving the kernel spatial dimension names."""
        return ['LK{}'.format(i) for i in range(self.rank)]

    def Ki_channel_dims(self):
        """String list giving the kernel channel dimension names. Post-reshape.

        Returns None instead if no reshape is needed."""
        if not self.kernel_needs_reshape():
            return None
        if self.grouping == ConvolutionGrouping.NONE:
            raise RuntimeError("Unexpected kernel reshape in ungrouped convolution.")
        elif self.grouping == ConvolutionGrouping.AUTO:
            return [self._G(), self._GCO(), self._GCI()]
        else:
            raise RuntimeError(
                'ConvolutionGrouping should have been converted to NONE or AUTO, but received {}'.
                format(self.grouping))

    def Ki_channel_idxs(self):
        """String list giving the kernel channel index names.

        These are used in the core convolution contraction, and as such they are
        given in a post-reshape format if any kernel reshaping happens."""
        if self.grouping == ConvolutionGrouping.NONE:
            ker_out_channel_idx = self.Oi_channel_idxs()
            ker_in_channel_idx = self.I_channel_idx()
            if self.kernel_format == ConvolutionKernelFormat.CHANNELS_FIRST:
                return ker_out_channel_idx + ker_in_channel_idx
            elif self.kernel_format == ConvolutionKernelFormat.CHANNELS_LAST:
                return ker_in_channel_idx + ker_out_channel_idx
            else:
                raise ValueError('Unknown kernel format {}'.format(self.kernel_format))
        elif self.grouping == ConvolutionGrouping.AUTO:
            return ['g', 'gco', 'gci']
        else:
            raise RuntimeError(
                'ConvolutionGrouping should have been converted to NONE or AUTO, but received {}'.
                format(self.grouping))

    def Ki_spatial_dims(self):
        """String list giving the kernel spatial dimension names (post-reshape)."""
        return self.K_spatial_dims()

    def Ki_spatial_idxs(self):
        """String list giving the kernel spatial index names."""
        return ['k{}'.format(i) for i in range(self.rank)]

    def O_batch_dim(self):
        """String list giving the final output batch dimension name."""
        return self.I_batch_dim()

    def O_batch_dim_numeric(self):
        """SymbolicDim or int list giving the final output batch dimension."""
        return [self.in_shape[self.get_I_axis(ConvIndex.n)]]

    def O_channel_dims(self):
        """String list giving the output channel dimension names. Post-reshape."""
        return [self._CO()]

    def O_channel_dims_numeric(self):
        """SymbolicDim or int list giving the output channel dimensions. Final.

        These are used to tell PlaidML the shape of the final output. In
        particular, this means these dimensions are post-reshape."""
        if self.grouping == ConvolutionGrouping.NONE:
            return [self.kernel_shape[self.get_K_axis(ConvIndex.co)]]
        elif self.grouping == ConvolutionGrouping.AUTO:
            if self.group_format == GroupedChannelFormat.FullOutGroupIn:
                return [self.kernel_shape[self.get_K_axis(ConvIndex.co)]]
            elif self.group_format == GroupedChannelFormat.GroupGroupOut:
                return [
                    self.kernel_shape[self.get_K_axis(ConvIndex.g)] *
                    self.kernel_shape[self.get_K_axis(ConvIndex.gco)]
                ]
            elif self.group_format == GroupedChannelFormat.GroupGroupOutGroupIn:
                return [
                    self.kernel_shape[self.get_K_axis(ConvIndex.g)] *
                    self.kernel_shape[self.get_K_axis(ConvIndex.gco)]
                ]
            else:
                raise ValueError('Unknown grouped channel format {}'.format(self.group_format))
        else:
            raise RuntimeError(
                'ConvolutionGrouping should have been converted to NONE or AUTO, but received {}'.
                format(self.grouping))

    def O_spatial_dims(self):
        """String list giving the output spatial dimension names."""
        if self.transposed:
            return ['L{}'.format(i) for i in range(self.rank)]
        else:
            if self._O_spatial_dims is None:
                self._compute_padding()
            return self._O_spatial_dims

    def O_spatial_dims_numeric(self):
        """SymbolicDim or int list giving the output spatial dimensions."""
        if self._O_spatial_dims_numeric is None:
            self._compute_padding()
        return self._O_spatial_dims_numeric

    def Oi_batch_dim(self):
        """String list giving the internal output batch dimension name."""
        return self.I_batch_dim()

    def Oi_batch_idx(self):
        """String list giving the output batch dimension name."""
        return self.I_batch_idx()

    def Oi_channel_dims(self):
        """String list giving the output channel dimension names. Pre-reshape."""
        if self.grouping == ConvolutionGrouping.NONE:
            return ['CO']
        elif self.grouping == ConvolutionGrouping.AUTO:
            return [self._G(), self._GCO()]
        else:
            raise RuntimeError(
                'ConvolutionGrouping should have been converted to NONE or AUTO, but received {}'.
                format(self.grouping))

    def Oi_channel_idxs(self):
        """String list giving the output channel index names."""
        if self.grouping == ConvolutionGrouping.NONE:
            return ['co']
        elif self.grouping == ConvolutionGrouping.AUTO:
            return ['g', 'gco']
        else:
            raise RuntimeError(
                'ConvolutionGrouping should have been converted to NONE or AUTO, but received {}'.
                format(self.grouping))

    def Oi_spatial_dims(self):
        return self.O_spatial_dims()

    def Oi_spatial_idxs(self):
        """String list giving the output spatial index names."""
        return ['x{}'.format(i) for i in range(self.rank)]

    def I_dims(self):
        if self.data_format == ConvolutionDataFormat.CHANNELS_FIRST:
            return ', '.join(self.I_batch_dim() + self.I_channel_dim() + self.I_spatial_dims())
        elif self.data_format == ConvolutionDataFormat.CHANNELS_LAST:
            return ', '.join(self.I_batch_dim() + self.I_spatial_dims() + self.I_channel_dim())
        else:
            raise ValueError('Unrecognized data format \'{}\''.format(self.data_format))

    def I_idxs(self):
        if self.data_format == ConvolutionDataFormat.CHANNELS_FIRST:
            return ', '.join(self.I_batch_idx() + self.I_channel_idx() + self.I_spatial_idxs())
        elif self.data_format == ConvolutionDataFormat.CHANNELS_LAST:
            return ', '.join(self.I_batch_idx() + self.I_spatial_idxs() + self.I_channel_idx())
        else:
            raise ValueError('Unrecognized data format \'{}\''.format(self.data_format))

    def K_dims(self):
        if self.kernel_format == ConvolutionKernelFormat.CHANNELS_FIRST:
            return ', '.join(self.K_channel_dims() + self.K_spatial_dims())
        elif self.kernel_format == ConvolutionKernelFormat.CHANNELS_LAST:
            return ', '.join(self.K_spatial_dims() + self.K_channel_dims())
        else:
            raise ValueError('Unrecognized kernel format \'{}\''.format(self.kernel_format))

    def Ki_dims(self):
        if self.kernel_needs_reshape():
            if self.kernel_format == ConvolutionKernelFormat.CHANNELS_FIRST:
                return ', '.join(self.Ki_channel_dims() + self.Ki_spatial_dims())
            elif self.kernel_format == ConvolutionKernelFormat.CHANNELS_LAST:
                return ', '.join(self.Ki_spatial_dims() + self.Ki_channel_dims())
            else:
                raise ValueError('Unrecognized kernel format \'{}\''.format(self.kernel_format))
        else:
            # K == Kitrn
            return self.K_dims()

    def Ki_idxs(self):
        if self.kernel_format == ConvolutionKernelFormat.CHANNELS_FIRST:
            return ', '.join(self.Ki_channel_idxs() + self.Ki_spatial_idxs())
        elif self.kernel_format == ConvolutionKernelFormat.CHANNELS_LAST:
            return ', '.join(self.Ki_spatial_idxs() + self.Ki_channel_idxs())
        else:
            raise ValueError('Unrecognized kernel format \'{}\''.format(self.kernel_format))

    def O_dims(self):
        if self.output_needs_reshape():
            if self.transposed:
                raise RuntimeError("Output reshaping not implemented for transposed convolutions")
            if self.data_format == ConvolutionDataFormat.CHANNELS_FIRST:
                return ', '.join(self.O_batch_dim() + self.O_channel_dims() +
                                 self.O_spatial_dims())
            elif self.data_format == ConvolutionDataFormat.CHANNELS_LAST:
                return ', '.join(self.O_batch_dim() + self.O_spatial_dims() +
                                 self.O_channel_dims())
        else:
            # O == Oitrn
            return self.Oi_dims()

    def Oi_dims(self):
        if self.data_format == ConvolutionDataFormat.CHANNELS_FIRST:
            return ', '.join(self.Oi_batch_dim() + self.Oi_channel_dims() + self.Oi_spatial_dims())
        elif self.data_format == ConvolutionDataFormat.CHANNELS_LAST:
            return ', '.join(self.Oi_batch_dim() + self.Oi_spatial_dims() + self.Oi_channel_dims())
        else:
            raise ValueError('Unrecognized data format \'{}\''.format(self.data_format))

    def Oi_idxs(self):
        if self.data_format == ConvolutionDataFormat.CHANNELS_FIRST:
            return ', '.join(self.Oi_batch_idx() + self.Oi_channel_idxs() + self.Oi_spatial_idxs())
        elif self.data_format == ConvolutionDataFormat.CHANNELS_LAST:
            return ', '.join(self.Oi_batch_idx() + self.Oi_spatial_idxs() + self.Oi_channel_idxs())
        else:
            raise ValueError('Unrecognized data format \'{}\''.format(self.data_format))

    def O_shape_tuple_numeric(self):
        """SymbolicDim or int tuple giving the final output shape."""
        if self.transposed:
            raise RuntimeError("Outshape not defined for transposed convolutions")
        if self.data_format == ConvolutionDataFormat.CHANNELS_FIRST:
            return tuple(self.O_batch_dim_numeric() + self.O_channel_dims_numeric() +
                         self.O_spatial_dims_numeric())
        elif self.data_format == ConvolutionDataFormat.CHANNELS_LAST:
            return tuple(self.O_batch_dim_numeric() + self.O_spatial_dims_numeric() +
                         self.O_channel_dims_numeric())


class ArgMax(tile.Operation):
    """Maximum of elements along an axis.

    Builds a tensor (uint64) whose elements are the maximum value on some axis of an input tensor.
    """

    def __init__(self, value, axis=-1):
        self.axis = axis
        self.value = value
        shape, axes, subs = tile.compute_aggregation_axes(value.shape.dims, [axis], False)

        code = """
        function (I[{src_ranges}], One[]) -> (O) {{
            Max[{dest_indices}{dest_sep}{dest_ranges}] = >(I[{src_indices}]);
            IndexT[{reduce_indices} : {reduce_ranges}] = =(One[]);
            Index = index(IndexT, 0);
            AM[{dest_indices}{dest_sep}{dest_ranges}] = >(I[{src_indices}] == Max[{dest_indices}] ? Index[{reduce_indices}]);
            O = as_uint(AM, 32);
        }}""".format(**subs)
        super(ArgMax, self).__init__(code, [('I', value),
                                            ('One', tile.Value.from_var(1., tuple()))],
                                     [('O', tile.Shape(plaidml.DType.INT64, shape))])


argmax = ArgMax.function


def average_pool(data,
                 kernel_shape,
                 strides,
                 pads=None,
                 padding=AutoPadding.EXPLICIT,
                 data_format=PoolDataFormat.NCX,
                 name=None):
    return pool(data=data,
                mode=PoolMode.AVG,
                kernel_shape=kernel_shape,
                pads=pads,
                strides=strides,
                padding=padding,
                data_format=data_format,
                name=name)


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
            }}""".format(dims=input_sizes, prod=input_sizes_prod)
        super(BinaryCrossentropy, self).__init__(f, [('O', output), ('T', target)],
                                                 [('R', output.shape)])


binary_crossentropy = BinaryCrossentropy.function


class Cast(tile.Operation):

    def __init__(self, x, dtype):
        info = tile.DTYPE_INFOS[dtype]
        super(Cast, self).__init__(
            'function (I) -> (O) {{ O = as_{}(I, {}); }}'.format(info.base, info.bitwidth),
            [('I', x)], [('O', tile.Shape(dtype, x.shape.dims))])


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
                None if isinstance(dims[i], tile.Value) else dims[i]
                for i in range(len(dims))
                if i != axis
            ]

        merge_axis_dim = None
        shape_template = __clear_axis(tensors[0].shape.dims)
        for t in tensors:
            if __clear_axis(t.shape.dims) != shape_template:
                raise ValueError(
                    'Incompatible shapes: cannot concatenate along axis {}\n{} v {}'.format(
                        axis, tensors[0].shape, t.shape))
            if isinstance(t.shape.dims[axis], tile.Value):
                merge_axis_dim = t.shape.dims[axis]

        offsets = [0]
        if merge_axis_dim:
            for i in range(len(tensors)):
                offsets.append("+".join("A{}".format(j) for j in range(i + 1)))
        else:
            for i in range(len(tensors)):
                offsets.append(offsets[i] + tensors[i].shape.dims[axis])
            merge_axis_dim = offsets[len(tensors)]
        out_dims = tuple(
            tensors[0].shape.dims[i] if i != axis else merge_axis_dim for i in range(rank))

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
        #     T0[n0, a, n2: N0, A0+A1+A2, N2] = =(I0[n0, a, n2]);
        #     T1[n0, a+A0, n2: N0, A0+A1+A2, N2] = =(I1[n0, a, n2]);
        #     T2[n0, a+A0+A1, n2: N0, A0+A1+A2, N2] = =(I2[n0, a, n2]);
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
    _winograd_transforms_cache = dict()

    def __init__(
            self,
            data,
            kernel,
            strides=None,
            padding=AutoPadding.EXPLICIT,
            pads=None,
            group=1,
            kernel_shape=None,
            data_format=None,
            kernel_format=None,
            dilation_rate=None,
            grouping=ConvolutionGrouping.NONE,
            group_format=None,
            winograd_allowed=True,
            name=None,
    ):
        rank = data.shape.ndims - 2
        if name is None:
            name = 'Convolution{}d'.format(rank)
        if strides is None:
            strides = tuple(1 for _ in range(rank))
        if dilation_rate is None:
            dilation_rate = tuple(1 for _ in range(rank))
        if not kernel_shape:
            kernel_shape = kernel.shape.dims
        else:
            if kernel_format == ConvolutionKernelFormat.CHANNELS_FIRST:
                kernel_shape = tuple([kernel.shape.dims[0], kernel.shape.dims[1]] +
                                     list(kernel_shape))
            elif kernel_format == ConvolutionKernelFormat.CHANNELS_LAST:
                kernel_shape = tuple(
                    list(kernel_shape) + [kernel.shape.dims[0], kernel.shape.dims[1]])
            else:
                raise ValueError('Unknown kernel format {}'.format(kernel_format))

        for entry in dilation_rate:
            if not isinstance(entry, int) or entry <= 0:
                raise ValueError('Invalid dilation_rate: {}'.format(dilation_rate))
        if len(kernel_shape) != rank + 2:
            raise ValueError('Convolution kernel shape inconsistent with input shape: ' +
                             '{} (rank {}) v {} (rank {})'.format(kernel_shape,
                                                                  len(kernel_shape) -
                                                                  2, data.shape, data.shape.ndims -
                                                                  2))
        if len(strides) != rank:
            raise ValueError('Convolution strides length inconsistent with input shape: ' +
                             '{} (rank {}) v {} (rank {})'.format(strides, len(
                                 strides), data.shape, data.shape.ndims - 2))
        if len(dilation_rate) != rank:
            raise ValueError('Convolution dilation_rate length inconsistent with input shape: ' +
                             '{} (rank {}) v {} (rank {})'.format(dilation_rate, len(
                                 dilation_rate), data.shape, data.shape.ndims - 2))

        use_winograd = (rank == 2 and data_format == ConvolutionDataFormat.CHANNELS_LAST and
                        kernel_shape[0] == 3 and kernel_shape[1] == 3 and strides == (1, 1) and
                        dilation_rate == (1, 1) and kernel_shape[2] > 4 and kernel_shape[3] > 4 and
                        grouping == ConvolutionGrouping.NONE and winograd_allowed)
        if use_winograd:
            conv_strs = self._winograd_conv_strs(data.shape.dims, kernel_shape, padding)
            code = self._winograd_code_template().format(**conv_strs)
            A, B, G = self._compute_winograd_transforms(conv_strs['block'], kernel_shape[0])
            input_list = [('I', data), ('K', kernel), ('A', A), ('B', B), ('G', G)]
            outshape = tile.Shape(data.shape.dtype, conv_strs['outshape_tuple'])
        else:
            csf = _ConvolutionStringFormatter(rank,
                                              data.shape.dims,
                                              kernel_shape,
                                              strides,
                                              padding,
                                              dilation_rate,
                                              data_format,
                                              kernel_format,
                                              pads=pads,
                                              grouping=grouping,
                                              groups=group,
                                              group_format=group_format)
            if csf.kernel_needs_reshape():
                ker_reshape_str = '{Kitrn} = reshape(K, {Ki_dims});\n    '.format(
                    Kitrn=csf.Kitrn(), Ki_dims=csf.Ki_dims())
            else:
                ker_reshape_str = ''
            if csf.output_needs_reshape():
                out_reshape_str = '\n    O = reshape({Oitrn}, {O_dims});'.format(
                    Oitrn=csf.Oitrn(), O_dims=csf.O_dims())
            else:
                out_reshape_str = ''
            code = """function (I[{I_dims}], K[{K_dims}]) -> (O) {{\n""" \
                   """    {assertion}{padding_str}{ker_reshape_str}{Oitrn}[{Oi_idxs}: {Oi_dims}] = +(I[{I_idxs}]*{Kitrn}[{Ki_idxs}]);{out_reshape_str}\n""" \
                   """}}"""
            code = code.format(I_dims=csf.I_dims(),
                               K_dims=csf.K_dims(),
                               assertion=csf.assertion(),
                               padding_str=csf.padding_str(),
                               ker_reshape_str=ker_reshape_str,
                               Oitrn=csf.Oitrn(),
                               Oi_idxs=csf.Oi_idxs(),
                               Oi_dims=csf.Oi_dims(),
                               I_idxs=csf.I_idxs(),
                               Kitrn=csf.Kitrn(),
                               Ki_idxs=csf.Ki_idxs(),
                               out_reshape_str=out_reshape_str)
            input_list = [('I', data), ('K', kernel)]
            outshape = tile.Shape(data.shape.dtype, csf.O_shape_tuple_numeric())

        super(Convolution, self).__init__(code, input_list, [('O', outshape)], name=name)

    def _winograd_code_template(self):
        f = """
            function (I[N, X, Y, CI], K[S, S, CI, CO], A[BI, BO], B[BI, BI], G[BI, S] ) -> (O) {{
                Assert = assert_winograd_valid(BI - CI + 1 == BO);
                XO = {XO};
                YO = {YO};
                XB = (XO + BO - 1) / BO;
                YB = (YO + BO - 1) / BO;
                XP = {XP};
                YP = {YP};
                U1[i, j, ci, co : BI, S, CI, CO] = +(G[i, k] * K[k, j, ci, co]);
                U[i, j, ci, co : BI, BI, CI, CO] = +(U1[i, k, ci, co] * G[j, k]);
                V1[n, i, j, x, y, ci : N, BI, BI, XB, YB, CI] = +(B[k, i] * I[n, BO*x + k - XP, BO*y + j - YP, ci]);
                V[n, i, j, x, y, ci : N, BI, BI, XB, YB, CI] = +(V1[n, i, k, x, y, ci] * B[k, j]);
                M[n, i, j, x, y, co : N, BI, BI, XB, YB, CO] = +(V[n, i, j, x, y, ci] * U[i, j, ci, co]);
                O1[n, i, j, x, y, co : N, BO, BI, XB, YB, CO] = +(A[k, i] * M[n, k, j, x, y, co]);
                O[n, BO*x + i, BO*y + j, co : N, XO, YO, CO] = +(O1[n, i, k, x, y, co] * A[k, j]) no_defract;
            }}"""
        return f

    def _winograd_conv_strs(self, in_shape, kernel_shape, padding):
        (XO, XP, NXO) = pad_compute('X', in_shape[1], kernel_shape[0], 1, padding)
        (YO, YP, NYO) = pad_compute('Y', in_shape[2], kernel_shape[0], 1, padding)
        outdims = (in_shape[0], NXO, NYO, kernel_shape[3])
        return {'XO': XO, 'XP': XP, 'YO': YO, 'YP': YP, 'block': 6, 'outshape_tuple': outdims}

    def _compute_winograd_transforms(self, block, conv):
        # Returns (A, B, G)
        if (block, conv) in Convolution._winograd_transforms_cache:
            return Convolution._winograd_transforms_cache[(block, conv)]
        out = block - conv + 1
        if (out == 2 and conv == 3):
            A = np.array([[1, 0], [1, 1], [1, -1], [0, -1]], dtype='float32')
            B = np.array([[1, 0, 0, 0], [0, 1, -1, 1], [-1, 1, 1, 0], [0, 0, 0, -1]],
                         dtype='float32')
            G = np.array([[1, 0, 0], [.5, .5, .5], [.5, -.5, .5], [0, 0, 1]], dtype='float32')
        elif (out == 4 and conv == 3):
            #s2 = np.sqrt(2.0)
            #A = np.array([[1., 0., 0., 0.], [1., s2/2., 1./2., s2/4.], [1, -s2/2., 1./2., -s2/4.],
            #              [1., s2, 2., 2.*s2], [1., -s2, 2., -2.*s2], [0., 0., 0., 1.]])
            #B = np.array([[1., 0., 0., 0., 0., 0.], [0., -s2, s2, -s2/2., s2/2., 1], [-5./2., -2., -2., -1./2., -1./2., 0],
            #              [0., s2/2., -s2/2., s2, -s2, -5./2], [1., 1., 1., 1., 1., 0.], [0., 0., 0., 0., 0., 1.]])
            #G = np.array([[1., 0., 0.], [-2./3., -s2/3., -1./3.], [-2./3., s2/3., -1./3.],
            #              [1./6., s2/6., 1./3.], [1./6., -s2/6., 1./3.], [0., 0., 1.]])
            # yapf: disable
            A = np.array([
                [ 1.13777777777778,   0,                  0,                 0,                ],
                [-0.688403361344538, -0.430252100840336, -0.26890756302521, -0.168067226890756 ],
                [-0.688403361344538,  0.430252100840336, -0.26890756302521,  0.168067226890756 ],
                [ 0.119514472455649,  0.179271708683473,  0.26890756302521,  0.403361344537815 ],
                [ 0.119514472455649, -0.179271708683473,  0.26890756302521, -0.403361344537815 ],
                [ 0,                  0,                  0,                 1,                ]],
                dtype='float32')
            B = np.array([
                [ 0.87890625,  0,          -2.640625,  0,        1, 0 ],
                [ 0,          -1.40625,    -2.25,      0.625,    1, 0 ],
                [ 0,           1.40625,    -2.25,     -0.625,    1, 0 ],
                [ 0,          -0.5859375,  -0.390625,  1.5,      1, 0 ],
                [ 0,           0.5859375,  -0.390625, -1.5,      1, 0 ],
                [ 0,           0.87890625,  0,        -2.640625, 0, 1 ]],
                dtype='float32').T
            G = np.array([
                [ 1, 1,         1 ,       1,     1,     0 ],
                [ 0, 0.625,    -0.625,    1.5,  -1.5,   0 ],
                [ 0, 0.390625,  0.390625, 2.25,  2.25,  1 ]],
                dtype='float32').T
            # yapf: enable
        else:
            raise plaidml.exceptions.InvalidArgument(
                'Only L(2, 3) and L(4, 3) currently supported for Winograd')
        Convolution._winograd_transforms_cache[(block, conv)] = (tile.Value.from_python_value(A),
                                                                 tile.Value.from_python_value(B),
                                                                 tile.Value.from_python_value(G))
        return Convolution._winograd_transforms_cache[(block, conv)]


convolution = Convolution.function


class ConvolutionTranspose(tile.Operation):
    """
    A transposed convolution operator.
    """

    def __init__(self,
                 x,
                 kernel,
                 output_shape,
                 strides,
                 padding,
                 data_format,
                 kernel_format,
                 dilation_rate=None,
                 name=None):
        rank = x.shape.ndims - 2
        if name is None:
            name = 'ConvolutionTranspose{}d'.format(rank)

        if kernel.shape.ndims != rank + 2:
            raise ValueError('Transpose convolution kernel shape inconsistent with input shape: ' +
                             '{} (rank {}) v {} (rank {})'.format(
                                 kernel.shape, kernel.shape.ndims - 2, x.shape, x.shape.ndims - 2))
        if len(output_shape) != rank + 2:
            raise ValueError('Transpose convolution output_shape inconsistent with input shape: ' +
                             '{} (rank {}) v {} (rank {})'.format(output_shape,
                                                                  len(output_shape) -
                                                                  2, x.shape, x.shape.ndims - 2))
        if len(strides) != rank:
            raise ValueError('Transpose convolution strides inconsistent with input shape: ' +
                             '{} (rank {}) v {} (rank {})'.format(strides, len(strides), x.shape,
                                                                  x.shape.ndims - 2))
        if (x.shape.dims[0] != output_shape[0] and
                isinstance(x.shape.dims[0], six.integer_types) and
                isinstance(output_shape[0], six.integer_types)):
            raise ValueError('Transpose convolution batch size inconsistent between input ' +
                             'and output: {} v {}'.format(x.shape.dims[0], output_shape[0]))
        if dilation_rate is None:
            dilation_rate = (1,) * rank
        if len(dilation_rate) != rank:
            raise ValueError('Transpose convolution dilations inconsistent with input shape: ' +
                             '{} (rank {}) v {} (rank {})'.format(strides, len(strides), x.shape,
                                                                  x.shape.ndims - 2))

        csf = _ConvolutionStringFormatter(
            rank,
            output_shape,
            kernel.shape.dims,
            strides,
            padding,
            dilation_rate,
            data_format,
            kernel_format,
            transposed=True,
            expected_output_shape=x.shape.dims,
        )
        code = """function (O[{O_dims}], K[{K_dims}]{dim_input}) -> (I) {{\n""" \
               """    {padding_str}I[{I_idxs}: {I_dims}] = +(O[{Oi_idxs}]*K[{Ki_idxs}]);\n""" \
               """}}"""
        code = code.format(O_dims=csf.O_dims(),
                           K_dims=csf.K_dims(),
                           dim_input=', ' + ', '.join(['D{}'.format(i) for i in range(rank)]),
                           padding_str=csf.padding_str(),
                           I_idxs=csf.I_idxs(),
                           I_dims=csf.I_dims(),
                           Oi_idxs=csf.Oi_idxs(),
                           Ki_idxs=csf.Ki_idxs())

        # Output shape may be dynamic, so pass its sizes as inputs to Tile
        l = csf.get_O_axis(ConvIndex.x)
        input_tensors = [('O', x), ('K', kernel)] + \
                        [('D{}'.format(i), output_shape[l[i]]) for i in range(rank)]

        super(ConvolutionTranspose,
              self).__init__(code,
                             input_tensors,
                             [('I', tile.Shape(x.shape.dtype, tuple(output_shape)))],
                             name=name)


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
            }}""".format(src_ranges=ranges,
                         dest_idxs=dest_idxs,
                         dest_ranges=ranges,
                         src_idxs=src_idxs,
                         ax=axis)
        super(CumulativeSum, self).__init__(f, [('I', x)], [('O', x.shape)])


cumulative_sum = CumulativeSum.function


class CumulativeProd(tile.Operation):
    """Cumulative product of a tensor"""

    def __init__(self, x, axis=0):
        ranges = ', '.join(['N{}'.format(n) for n in range(x.shape.ndims)])
        dest_idxs = ', '.join(['i{}'.format(n) for n in range(x.shape.ndims)])
        src_idxs = ['i{}'.format(n) for n in range(x.shape.ndims)]
        src_idxs[axis] += ' - k'
        src_idxs = ', '.join(src_idxs)
        f = """
            function (I[{src_ranges}]) -> (O) {{
                O[{dest_idxs}: {dest_ranges}] = *(I[{src_idxs}]), k < N{ax};
            }}""".format(
            src_ranges=ranges, dest_idxs=dest_idxs, dest_ranges=ranges, src_idxs=src_idxs, ax=axis)
        super(CumulativeProd, self).__init__(f, [('I', x)], [('O', x.shape)])


cumulative_prod = CumulativeProd.function




class Dot(tile.Operation):
    """Dot-product of two tensors."""

    def __init__(self, x, y, name=None):
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

        super(Dot, self).__init__(f, [('X', x), ('Y', y)], [('R', shape)], name=name)


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
            super(Equal, self).__init__('function (L, R) -> (O) { O = (L == R); }', [('L', lhs),
                                                                                     ('R', rhs)],
                                        [('O', shape)])
        else:
            shape = tile.Shape(plaidml.DType.BOOLEAN, lhs.shape.dims)
            super(Equal, self).__init__('function (L) -> (O) {{ O = (L == {}); }}'.format(rhs),
                                        [('L', lhs)], [('O', shape)])


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
        if not broadcast and c.shape.ndims != 2:
            raise NotImplementedError(
                'Gemm without multiplier broadcast requires a two-dimensional scalar multiplier; multiplier rank={}'
                .format(c.shape.ndims))

        def gemm_reshape(value):
            if value.shape.ndims < 2:
                raise tile.LogicError(
                    'Invalid Gemm input; two-dimensions required, got: {}'.format(value.shape))
            if value.shape.ndims == 2:
                return value
            newdims = (value.shape.dims[0],
                       functools.reduce(lambda x, y: x * y, value.shape.dims[1:]))
            return reshape(value, newdims)

        a = gemm_reshape(a)
        b = gemm_reshape(b)

        code = """
        function (A[{a_dims}], B[{b_dims}], C) -> (O) {{
          OM[row, col : ROW, COL] = +(A[{a_idxs}] * B[{b_idxs}]);
          OA = {alpha_expr};
          CB = {beta_expr};
          O = OA + CB;
        }}""".format(
            a_dims='MID, ROW' if transA else 'ROW, MID',
            b_dims='COL, MID' if transB else 'MID, COL',
            a_idxs='mid, row' if transA else 'row, mid',
            b_idxs='col, mid' if transB else 'mid, col',
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
                    }}""".format(a_ranges=', '.join(a_ranges),
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


def max_pool(data,
             kernel_shape,
             strides,
             pads=None,
             padding=AutoPadding.EXPLICIT,
             data_format=PoolDataFormat.NCX,
             name=None):
    return pool(data=data,
                mode=PoolMode.MAX,
                kernel_shape=kernel_shape,
                pads=pads,
                strides=strides,
                padding=padding,
                data_format=data_format,
                name=name)


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


class Pool(tile.Operation):
    """
    A standard ML pooling operator. Handles both MAX & AVG
    """

    def __init__(self,
                 data,
                 mode,
                 kernel_shape,
                 strides,
                 pads=None,
                 padding=AutoPadding.EXPLICIT,
                 data_format=PoolDataFormat.NCX,
                 name=None):
        rank = data.shape.ndims - 2
        pads = _extend_pads(pads, rank)
        if not strides:
            strides = tuple(1 for _ in range(rank))
        elif len(strides) != rank:
            raise ValueError(
                'Pool strides length inconsistent with input shape: ' +
                '{} (rank {}) v {} (rank {})'.format(strides, len(strides), data.shape, rank))
        in_spatial_dims = ['L{}'.format(i) for i in range(rank)]
        in_spatial_idxs = list()
        out_spatial_dims = []
        if data_format == PoolDataFormat.NCX:
            data_spatial_dims = data.shape.dims[2:rank + 2]
        elif data_format == PoolDataFormat.NXC:
            data_spatial_dims = data.shape.dims[1:rank + 1]
        else:
            raise ValueError('Invalid data_format')
        num_out_spatial_shape = list()
        pad_amount = list()
        for i in range(rank):
            sym_out, sym_pad, num_out = pad_compute('L{}'.format(i), data_spatial_dims[i],
                                                    kernel_shape[i], strides[i], padding,
                                                    (pads[i], pads[i + rank]) if pads else None)
            out_spatial_dims.append(sym_out)
            num_out_spatial_shape.append(num_out)
            pad_amount.append(sym_pad)
            in_spatial_idxs.append('{stride}*x{idx} + k{idx} - Pad{idx}'.format(stride=strides[i],
                                                                                idx=i))
        out_spatial_idxs = ['x{}'.format(i) for i in range(rank)]
        padding_list = ['Pad{} = {};'.format(i, pad_amount[i]) for i in range(rank)]
        padding_str = '\n            '.join(padding_list)
        if data_format == PoolDataFormat.NCX:
            in_dims = ['N', 'C'] + in_spatial_dims
            in_idxs = ['n', 'c'] + in_spatial_idxs
            out_dims = ['N', 'C'] + out_spatial_dims
            out_idxs = ['n', 'c'] + out_spatial_idxs
            ones_write_idxs = ['n', 'c'] + ['o{}'.format(i) for i in range(rank)]
            num_out_shape = list(data.shape.dims[0:2]) + num_out_spatial_shape
        elif data_format == PoolDataFormat.NXC:
            in_dims = ['N'] + in_spatial_dims + ['C']
            in_idxs = ['n'] + in_spatial_idxs + ['c']
            out_dims = ['N'] + out_spatial_dims + ['C']
            out_idxs = ['n'] + out_spatial_idxs + ['c']
            ones_write_idxs = ['n'] + ['o{}'.format(i) for i in range(rank)] + ['c']
            num_out_shape = list(data.shape.dims[0:1]) + num_out_spatial_shape + list(
                data.shape.dims[rank + 1:rank + 2])
        else:
            raise ValueError('Invalid data_format')

        if mode == PoolMode.AVG:
            pool_contraction_op = "+"
            pool_contraction_out_name = "S"
            # Want average pooling not sum pooling, so divide by number of elements in a pool
            # However, the number of elements in the pool should only count true elements,
            # not zero padding. Thus, we build a tensor that is 1 everywhere the original
            # tensor is defined, and we sum that tensor over the pool area to find the
            # number of elements in the pool for the corresponding output entry.
            denom_gen_code = """
            Ones[{ones_write_idxs} : {ones_dims}] = =(One[]);
            Count[{cout_idxs} : {cout_dims}] = +(Ones[{ones_read_idxs}]), {pool_bounds};""".format(
                ones_write_idxs=', '.join(ones_write_idxs),
                ones_dims=', '.join(in_dims),
                ones_read_idxs=', '.join(in_idxs),
                cout_idxs=', '.join(out_idxs),
                cout_dims=', '.join(out_dims),
                pool_bounds=', '.join(['k{} < {}'.format(i, kernel_shape[i]) for i in range(rank)
                                      ]),
            )
            denom_divide_code = """
            O = S / Count;"""
            extra_input = ", One[]"
            input_tensors = [('I', data), ('One', tile.Value.from_var(1., tuple()))]
        elif mode == PoolMode.MAX:
            pool_contraction_op = ">"
            pool_contraction_out_name = "O"
            denom_gen_code = ""
            denom_divide_code = ""
            extra_input = ""
            input_tensors = [('I', data)]
        else:
            raise ValueError('Invalid mode for Pool operation')
        code = """
        function (I[{in_dims}]{extra_input}) -> (O) {{
            {padding_str}{denom_gen}
            {out_name}[{out_idxs} : {out_dims}] = {op}(I[{in_idxs}]), {pool_bounds}; {denom_divide}
        }}""".format(op=pool_contraction_op,
                     extra_input=extra_input,
                     denom_gen=denom_gen_code,
                     denom_divide=denom_divide_code,
                     padding_str=padding_str,
                     out_idxs=', '.join(out_idxs),
                     out_dims=', '.join(out_dims),
                     in_idxs=', '.join(in_idxs),
                     in_dims=', '.join(in_dims),
                     out_name=pool_contraction_out_name,
                     pool_bounds=', '.join(
                         ['k{} < {}'.format(i, kernel_shape[i]) for i in range(rank)]))

        outshape = tile.Shape(data.shape.dtype, num_out_shape)

        super(Pool, self).__init__(code, input_tensors, [('O', outshape)], name=name)


pool = Pool.function


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

    def __init__(self, x, alpha=None, max_value=None, threshold=0.):
        inputs = [('X', x)]
        if alpha is not None:
            inputs.append(('Alpha', alpha))
            low_branch = 'Alpha*{X}'
        else:
            low_branch = '0.0'
        if max_value is not None:
            inputs.append(('MaxValue', max_value))
            max_clip_fcn = '\n    Y = (M < MaxValue ? M : MaxValue);'
            main_out_var = 'M'
        else:
            main_out_var = 'Y'
            max_clip_fcn = ''
        if threshold != 0.:
            inputs.append(('Thresh', threshold))
            low_branch = low_branch.format(X='(X - Thresh)')
            main_fcn = '\n    {main_out_var} = (X < Thresh ? {low_branch} : X);'.format(
                main_out_var=main_out_var, low_branch=low_branch)
        else:
            low_branch = low_branch.format(X='X')
            low_branch = low_branch.format(X='X')
            main_fcn = '\n    {main_out_var} = (X < 0.0 ? {low_branch} : X);'.format(
                main_out_var=main_out_var, low_branch=low_branch)

        if alpha is None and max_value is None and threshold == 0.:
            # Nothing messy; use the builtin relu.
            code = 'function (X) -> (Y) { Y = relu(X); }'
        else:
            # Put together the pieces
            input_str = ', '.join(inp[0] for inp in inputs)
            code = 'function ({ins}) -> (Y) {{ {main_fcn}{max_clip_fcn}\n}}'.format(
                ins=input_str, main_fcn=main_fcn, max_clip_fcn=max_clip_fcn)

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

        super(Reshape, self).__init__(
            'function ({}) -> (O) {{ O = reshape(I, {}); }}'.format(
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
        }}""".format(in_dims=', '.join(in_dims),
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

    def __init__(self, data, name=None):
        if data.shape.ndims != 2:
            raise NotImplementedError(
                'Softmax with a non-two-dimensional tensor is not currently implemented')

        code = """
        function (I[X, Y]) -> (O) {
            O = builtin_softmax(I, X, Y);
        }"""

        super(Softmax, self).__init__(code, [('I', data)], [('O', data.shape)], name=name)


def softmax(x, axis=None, name=None):
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
    result = Softmax.function(flat_x, name=name)
    return reshape(result, full_dims)


class Sqrt(tile.Operation):
    """
    Computes the elementwise square root of a value.
    """

    def __init__(self, x):
        super(Sqrt, self).__init__(
            """
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


class ImagePatches(plaidml.tile.Operation):

    def __init__(self, images, ksizes, strides, rates=(1, 1, 1, 1), padding="VALID"):
        """
        Compatible to tensorflow.extract_image_patches. 
        Extract patches from images and put them in the "depth" output dimension.
        Args:
            images: A tensor with a shape of [batch, rows, cols, depth]
            ksizes: The size of the oatches with a shape of [1, patch_rows, patch_cols, 1]
            strides: How far the center of two patches are in the image with a shape of [1, stride_rows, stride_cols, 1]
            rates: How far two consecutive pixel are in the input. Equivalent to dilation. Expect shape of [1, rate_rows, rate_cols, 1]
            padding: A string of "VALID" or "SAME" defining padding.
            
        Does not work with symbolic height and width.
        """
        i_shape = images.shape.dims
        patch_row_eff = ksizes[1] + ((ksizes[1] - 1) * (rates[1] - 1))
        patch_col_eff = ksizes[2] + ((ksizes[2] - 1) * (rates[2] - 1))

        if padding.upper() == "VALID":
            out_rows = math.ceil((i_shape[1] - patch_row_eff + 1.) / float(strides[1]))
            out_cols = math.ceil((i_shape[2] - patch_col_eff + 1.) / float(strides[2]))
            pad_str = "PAD = I;"
        else:
            out_rows = math.ceil(i_shape[1] / float(strides[1]))
            out_cols = math.ceil(i_shape[2] / float(strides[2]))
            dim_calc = "NY={NY}; NX={NX};".format(NY=out_rows, NX=out_cols)
            pad_top = max(0, ((out_rows - 1) * strides[1] + patch_row_eff - i_shape[1]) // 2)
            pad_left = max(0, ((out_cols - 1) * strides[2] + patch_col_eff - i_shape[2]) // 2)
            # we simply assume padding right == padding left + 1 (same for top/down).
            # This might lead to us padding more as we would need but that won't matter.
            # TF splits padding between both sides so left_pad +1 should keep us on the safe side.
            pad_str = """PAD[b, y, x, d : B, Y + {PT} * 2 + 1, X + {PL} * 2 + 1, D] = 
                        =(I[b, y - {PT}, x - {PL}, d]);""".format(PT=pad_top, PL=pad_left)

        o_shape = (i_shape[0], out_rows, out_cols, ksizes[1] * ksizes[2] * i_shape[-1])
        code = """function (I[B,Y,X,D]) -> (O) {{
                    {PAD}
                    TMP[b, ny, nx, y, x, d: B, {NY}, {NX}, {KY}, {KX}, D] =
                        =(PAD[b, ny * {SY} + y * {RY}, nx * {SX} + x * {RX}, d]);
                    O = reshape(TMP, B, {NY}, {NX}, {KY} * {KX} * D);
                }}
        """.format(PAD=pad_str,
                   NY=out_rows,
                   NX=out_cols,
                   KY=ksizes[1],
                   KX=ksizes[2],
                   SY=strides[1],
                   SX=strides[2],
                   RY=rates[1],
                   RX=rates[2])
        super(ImagePatches, self).__init__(code, [
            ('I', images),
        ], [('O', plaidml.tile.Shape(images.shape.dtype, o_shape))])


extract_image_patches = ImagePatches.function


class ReflectionPadding(tile.Operation):

    def __init__(self, inp, paddings):
        paddings = [(x, x) if isinstance(x, int) else x for x in paddings]
        ndims = inp.shape.ndims
        out_shape = list(inp.shape.dims)
        if ndims != len(paddings):
            raise ValueError('Padding dims != input dims')
        for ax, pads in enumerate(paddings):
            if isinstance(out_shape[ax], tile.Value):
                # We can't tell if padding size is supported for symbolic axis.
                continue
            for pad in pads:
                if pad >= out_shape[ax]:
                    raise plaidml.exceptions.InvalidArgument(
                        'Paddings must be less than the dimension size: {} not less than {}.'.
                        format(pad, out_shape[ax]))
        out_sizes = [
            'N{}'.format(i) if isinstance(x, tile.Value) else x for i, x in enumerate(out_shape)
        ]
        in_sizes = list(out_sizes)
        code_body = []
        src_arr = "I"
        idx = ["n{}".format(i) for i in range(ndims)]

        for axis, pads in ((i, x) for i, x in enumerate(paddings) if x[0] + x[1] != 0):
            pad_pre, pad_post = pads
            if isinstance(out_shape[axis], tile.Value):
                out_sizes[axis] += ' + {}'.format(pad_pre + pad_post)
            else:
                out_shape[axis] += pad_pre + pad_post
                out_sizes[axis] += pad_pre + pad_post
            concats = "TA{}".format(axis)

            if pad_pre:
                src_idx = list(idx)
                src_idx[axis] = "{} - n{}".format(pad_pre, axis)
                code_body.append(
                    'TS{ax}[{idx} : {dims}] = =({src_arr}[{src_idx}]), n{ax} < {pad};'.format(
                        ax=axis,
                        idx=','.join(idx),
                        dims=','.join(map(str, out_sizes)),
                        src_arr=src_arr,
                        src_idx=','.join(src_idx),
                        pad=pad_pre))
                concats += ' + TS{}'.format(axis)

            if pad_post:
                src_idx = list(idx)
                dst_idx = list(idx)
                src_idx[axis] = "{} - n{} - 2".format(in_sizes[axis], axis)
                dst_idx[axis] = "n{} + {} - {}".format(axis, out_sizes[axis], pad_post)
                code_body.append(
                    'TE{ax}[{idx} : {dims}] = =({src_arr}[{src_idx}]), n{ax} < {pad};'.format(
                        ax=axis,
                        idx=','.join(map(str, dst_idx)),
                        dims=','.join(map(str, out_sizes)),
                        src_arr=src_arr,
                        src_idx=','.join(src_idx),
                        pad=pad_post,
                    ))
                concats += ' + TE{}'.format(axis)

            dst_idx = list(idx)
            dst_idx[axis] = "{} + n{}".format(pad_pre, axis)
            code_body.append(
                'TA{ax}[{idx} : {dims}] = =({src_arr}[{src_idx}]), n{ax} < {cond};'.format(
                    ax=axis,
                    idx=",".join(dst_idx),
                    dims=','.join(map(str, out_sizes)),
                    src_arr=src_arr,
                    src_idx=",".join(idx),
                    cond="{} - {}".format(out_sizes[axis], pad_post)))
            code_body.append("TC{} = {};".format(axis, concats))
            src_arr = 'TC{}'.format(axis)
            in_sizes = list(out_sizes)

        code_body.append('O = {};'.format(src_arr))
        code = 'function (I[{idim}]) -> (O) {{\n\t{body}\n}}'.format(idim=",".join(
            "N{}".format(i) for i in range(ndims)),
                                                                     body="\n\t".join(code_body))
        super(ReflectionPadding, self).__init__(code, [('I', inp)],
                                                [('O', tile.Shape(inp.shape.dtype, out_shape))])


reflection_padding = ReflectionPadding.function
