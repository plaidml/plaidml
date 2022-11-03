import torch
import numpy as np

# The following functions were taken from
# https://github.com/tensorflow/models/tree/master/research/object_detection
# with minor modifications so that they use
# torch operations instead


def expanded_shape(orig_shape, start_dim, num_dims):
    s = (1,) * num_dims
    return orig_shape[:start_dim] + s + orig_shape[start_dim:]


def meshgrid(x, y):
    """Tiles the contents of x and y into a pair of grids.
    Multidimensional analog of numpy.meshgrid, giving the same behavior if x and y
    are vectors. Generally, this will give:
    xgrid(i1, ..., i_m, j_1, ..., j_n) = x(j_1, ..., j_n)
    ygrid(i1, ..., i_m, j_1, ..., j_n) = y(i_1, ..., i_m)
    Keep in mind that the order of the arguments and outputs is reverse relative
    to the order of the indices they go into, done for compatibility with numpy.
    The output tensors have the same shapes.  Specifically:
    xgrid.get_shape() = y.get_shape().concatenate(x.get_shape())
    ygrid.get_shape() = y.get_shape().concatenate(x.get_shape())
    Args:
    x: A tensor of arbitrary shape and rank. xgrid will contain these values
       varying in its last dimensions.
    y: A tensor of arbitrary shape and rank. ygrid will contain these values
       varying in its first dimensions.
    Returns:
    A tuple of tensors (xgrid, ygrid).
    """
    x = torch.as_tensor(x)
    y = torch.as_tensor(y)
    x_exp_shape = expanded_shape(x.shape, 0, y.dim())
    y_exp_shape = expanded_shape(y.shape, y.dim(), x.dim())

    xgrid = torch.reshape(x, x_exp_shape).repeat(*y_exp_shape)
    ygrid = torch.reshape(y, y_exp_shape).repeat(*x_exp_shape)
    new_shape = y.shape + x.shape
    xgrid = xgrid.reshape(new_shape)
    ygrid = ygrid.reshape(new_shape)

    return xgrid, ygrid


def tile_anchors(grid_height, grid_width, scales, aspect_ratios, base_anchor_size, anchor_stride,
                 anchor_offset):
    """Create a tiled set of anchors strided along a grid in image space.
  This op creates a set of anchor boxes by placing a "basis" collection of
  boxes with user-specified scales and aspect ratios centered at evenly
  distributed points along a grid.  The basis collection is specified via the
  scale and aspect_ratios arguments.  For example, setting scales=[.1, .2, .2]
  and aspect ratios = [2,2,1/2] means that we create three boxes: one with scale
  .1, aspect ratio 2, one with scale .2, aspect ratio 2, and one with scale .2
  and aspect ratio 1/2.  Each box is multiplied by "base_anchor_size" before
  placing it over its respective center.
  Grid points are specified via grid_height, grid_width parameters as well as
  the anchor_stride and anchor_offset parameters.
  Args:
    grid_height: size of the grid in the y direction (int or int scalar tensor)
    grid_width: size of the grid in the x direction (int or int scalar tensor)
    scales: a 1-d  (float) tensor representing the scale of each box in the
      basis set.
    aspect_ratios: a 1-d (float) tensor representing the aspect ratio of each
      box in the basis set.  The length of the scales and aspect_ratios tensors
      must be equal.
    base_anchor_size: base anchor size as [height, width]
      (float tensor of shape [2])
    anchor_stride: difference in centers between base anchors for adjacent grid
                   positions (float tensor of shape [2])
    anchor_offset: center of the anchor with scale and aspect ratio 1 for the
                   upper left element of the grid, this should be zero for
                   feature networks with only VALID padding and even receptive
                   field size, but may need some additional calculation if other
                   padding is used (float tensor of shape [2])
  Returns:
    a BoxList holding a collection of N anchor boxes
  """
    aspect_ratios = torch.as_tensor(aspect_ratios, dtype=torch.float32)
    scales = torch.as_tensor(scales, dtype=torch.float32)

    ratio_sqrts = torch.sqrt(aspect_ratios)
    heights = scales / ratio_sqrts * base_anchor_size[0]
    widths = scales * ratio_sqrts * base_anchor_size[1]

    # Get a grid of box centers
    y_centers = torch.arange(grid_height, dtype=torch.float32)
    y_centers = y_centers * anchor_stride[0] + anchor_offset[0]
    x_centers = torch.arange(grid_width, dtype=torch.float32)
    x_centers = x_centers * anchor_stride[1] + anchor_offset[1]

    x_centers, y_centers = meshgrid(x_centers, y_centers)

    widths_grid, x_centers_grid = meshgrid(widths, x_centers)
    heights_grid, y_centers_grid = meshgrid(heights, y_centers)

    bbox_centers = torch.stack([y_centers_grid, x_centers_grid], dim=3)
    bbox_sizes = torch.stack([heights_grid, widths_grid], dim=3)
    bbox_centers = torch.reshape(bbox_centers, [-1, 2])
    bbox_sizes = torch.reshape(bbox_sizes, [-1, 2])
    bbox_corners = _center_size_bbox_to_corners_bbox(bbox_centers, bbox_sizes)
    return bbox_corners


def _center_size_bbox_to_corners_bbox(centers, sizes):
    """Converts bbox center-size representation to corners representation.
  Args:
    centers: a tensor with shape [N, 2] representing bounding box centers
    sizes: a tensor with shape [N, 2] representing bounding boxes
  Returns:
    corners: tensor with shape [N, 4] representing bounding boxes in corners
      representation
  """
    return torch.cat([centers - .5 * sizes, centers + .5 * sizes], 1)


def create_ssd_anchors(num_layers=6,
                       min_scale=0.2,
                       max_scale=0.95,
                       scales=None,
                       aspect_ratios=(1.0, 2.0, 1.0 / 2, 3.0, 1.0 / 3),
                       interpolated_scale_aspect_ratio=1.0,
                       base_anchor_size=None,
                       anchor_strides=None,
                       anchor_offsets=None,
                       reduce_boxes_in_lowest_layer=True):
    """Creates MultipleGridAnchorGenerator for SSD anchors.
  This function instantiates a MultipleGridAnchorGenerator that reproduces
  ``default box`` construction proposed by Liu et al in the SSD paper.
  See Section 2.2 for details. Grid sizes are assumed to be passed in
  at generation time from finest resolution to coarsest resolution --- this is
  used to (linearly) interpolate scales of anchor boxes corresponding to the
  intermediate grid sizes.
  Anchors that are returned by calling the `generate` method on the returned
  MultipleGridAnchorGenerator object are always in normalized coordinates
  and clipped to the unit square: (i.e. all coordinates lie in [0, 1]x[0, 1]).
  Args:
    num_layers: integer number of grid layers to create anchors for (actual
      grid sizes passed in at generation time)
    min_scale: scale of anchors corresponding to finest resolution (float)
    max_scale: scale of anchors corresponding to coarsest resolution (float)
    scales: As list of anchor scales to use. When not None and not empty,
      min_scale and max_scale are not used.
    aspect_ratios: list or tuple of (float) aspect ratios to place on each
      grid point.
    interpolated_scale_aspect_ratio: An additional anchor is added with this
      aspect ratio and a scale interpolated between the scale for a layer
      and the scale for the next layer (1.0 for the last layer).
      This anchor is not included if this value is 0.
    base_anchor_size: base anchor size as [height, width].
      The height and width values are normalized to the minimum dimension of the
      input height and width, so that when the base anchor height equals the
      base anchor width, the resulting anchor is square even if the input image
      is not square.
    anchor_strides: list of pairs of strides in pixels (in y and x directions
      respectively). For example, setting anchor_strides=[(25, 25), (50, 50)]
      means that we want the anchors corresponding to the first layer to be
      strided by 25 pixels and those in the second layer to be strided by 50
      pixels in both y and x directions. If anchor_strides=None, they are set to
      be the reciprocal of the corresponding feature map shapes.
    anchor_offsets: list of pairs of offsets in pixels (in y and x directions
      respectively). The offset specifies where we want the center of the
      (0, 0)-th anchor to lie for each layer. For example, setting
      anchor_offsets=[(10, 10), (20, 20)]) means that we want the
      (0, 0)-th anchor of the first layer to lie at (10, 10) in pixel space
      and likewise that we want the (0, 0)-th anchor of the second layer to lie
      at (25, 25) in pixel space. If anchor_offsets=None, then they are set to
      be half of the corresponding anchor stride.
    reduce_boxes_in_lowest_layer: a boolean to indicate whether the fixed 3
      boxes per location is used in the lowest layer.
  Returns:
    a MultipleGridAnchorGenerator
  """
    if base_anchor_size is None:
        base_anchor_size = [1.0, 1.0]
    base_anchor_size = torch.tensor(base_anchor_size, dtype=torch.float32)
    box_specs_list = []
    if scales is None or not scales:
        scales = [
            min_scale + (max_scale - min_scale) * i / (num_layers - 1) for i in range(num_layers)
        ] + [1.0]
    else:
        # Add 1.0 to the end, which will only be used in scale_next below and used
        # for computing an interpolated scale for the largest scale in the list.
        scales += [1.0]

    for layer, scale, scale_next in zip(range(num_layers), scales[:-1], scales[1:]):
        layer_box_specs = []
        if layer == 0 and reduce_boxes_in_lowest_layer:
            layer_box_specs = [(0.1, 1.0), (scale, 2.0), (scale, 0.5)]
        else:
            for aspect_ratio in aspect_ratios:
                layer_box_specs.append((scale, aspect_ratio))
            # Add one more anchor, with a scale between the current scale, and the
            # scale for the next layer, with a specified aspect ratio (1.0 by
            # default).
            if interpolated_scale_aspect_ratio > 0.0:
                layer_box_specs.append(
                    (np.sqrt(scale * scale_next), interpolated_scale_aspect_ratio))
        box_specs_list.append(layer_box_specs)

    return MultipleGridAnchorGenerator(box_specs_list, base_anchor_size, anchor_strides,
                                       anchor_offsets)


class MultipleGridAnchorGenerator(object):
    """Generate a grid of anchors for multiple CNN layers."""

    def __init__(self,
                 box_specs_list,
                 base_anchor_size=None,
                 anchor_strides=None,
                 anchor_offsets=None,
                 clip_window=None):
        """Constructs a MultipleGridAnchorGenerator.
    To construct anchors, at multiple grid resolutions, one must provide a
    list of feature_map_shape_list (e.g., [(8, 8), (4, 4)]), and for each grid
    size, a corresponding list of (scale, aspect ratio) box specifications.
    For example:
    box_specs_list = [[(.1, 1.0), (.1, 2.0)],  # for 8x8 grid
                      [(.2, 1.0), (.3, 1.0), (.2, 2.0)]]  # for 4x4 grid
    To support the fully convolutional setting, we pass grid sizes in at
    generation time, while scale and aspect ratios are fixed at construction
    time.
    Args:
      box_specs_list: list of list of (scale, aspect ratio) pairs with the
        outside list having the same number of entries as feature_map_shape_list
        (which is passed in at generation time).
      base_anchor_size: base anchor size as [height, width]
                        (length-2 float tensor, default=[1.0, 1.0]).
                        The height and width values are normalized to the
                        minimum dimension of the input height and width, so that
                        when the base anchor height equals the base anchor
                        width, the resulting anchor is square even if the input
                        image is not square.
      anchor_strides: list of pairs of strides in pixels (in y and x directions
        respectively). For example, setting anchor_strides=[(25, 25), (50, 50)]
        means that we want the anchors corresponding to the first layer to be
        strided by 25 pixels and those in the second layer to be strided by 50
        pixels in both y and x directions. If anchor_strides=None, they are set
        to be the reciprocal of the corresponding feature map shapes.
      anchor_offsets: list of pairs of offsets in pixels (in y and x directions
        respectively). The offset specifies where we want the center of the
        (0, 0)-th anchor to lie for each layer. For example, setting
        anchor_offsets=[(10, 10), (20, 20)]) means that we want the
        (0, 0)-th anchor of the first layer to lie at (10, 10) in pixel space
        and likewise that we want the (0, 0)-th anchor of the second layer to
        lie at (25, 25) in pixel space. If anchor_offsets=None, then they are
        set to be half of the corresponding anchor stride.
      clip_window: a tensor of shape [4] specifying a window to which all
        anchors should be clipped. If clip_window is None, then no clipping
        is performed.
    Raises:
      ValueError: if box_specs_list is not a list of list of pairs
      ValueError: if clip_window is not either None or a tensor of shape [4]
    """
        if isinstance(box_specs_list, list) and all(
            [isinstance(list_item, list) for list_item in box_specs_list]):
            self._box_specs = box_specs_list
        else:
            raise ValueError('box_specs_list is expected to be a '
                             'list of lists of pairs')
        if base_anchor_size is None:
            base_anchor_size = torch.tensor([256, 256], dtype=torch.float32)
        self._base_anchor_size = base_anchor_size
        self._anchor_strides = anchor_strides
        self._anchor_offsets = anchor_offsets
        if clip_window is not None and list(clip_window.shape) != [4]:
            raise ValueError('clip_window must either be None or a shape [4] tensor')
        self._clip_window = clip_window
        self._scales = []
        self._aspect_ratios = []
        for box_spec in self._box_specs:
            if not all([isinstance(entry, tuple) and len(entry) == 2 for entry in box_spec]):
                raise ValueError('box_specs_list is expected to be a '
                                 'list of lists of pairs')
            scales, aspect_ratios = zip(*box_spec)
            self._scales.append(scales)
            self._aspect_ratios.append(aspect_ratios)

        for arg, arg_name in zip([self._anchor_strides, self._anchor_offsets],
                                 ['anchor_strides', 'anchor_offsets']):
            if arg and not (isinstance(arg, list) and len(arg) == len(self._box_specs)):
                raise ValueError('%s must be a list with the same length '
                                 'as self._box_specs' % arg_name)
            if arg and not all(
                [isinstance(list_item, tuple) and len(list_item) == 2 for list_item in arg]):
                raise ValueError('%s must be a list of pairs.' % arg_name)

    def _generate(self, feature_map_shape_list, im_height=1, im_width=1):
        """Generates a collection of bounding boxes to be used as anchors.
    The number of anchors generated for a single grid with shape MxM where we
    place k boxes over each grid center is k*M^2 and thus the total number of
    anchors is the sum over all grids. In our box_specs_list example
    (see the constructor docstring), we would place two boxes over each grid
    point on an 8x8 grid and three boxes over each grid point on a 4x4 grid and
    thus end up with 2*8^2 + 3*4^2 = 176 anchors in total. The layout of the
    output anchors follows the order of how the grid sizes and box_specs are
    specified (with box_spec index varying the fastest, followed by width
    index, then height index, then grid index).
    Args:
      feature_map_shape_list: list of pairs of convnet layer resolutions in the
        format [(height_0, width_0), (height_1, width_1), ...]. For example,
        setting feature_map_shape_list=[(8, 8), (7, 7)] asks for anchors that
        correspond to an 8x8 layer followed by a 7x7 layer.
      im_height: the height of the image to generate the grid for. If both
        im_height and im_width are 1, the generated anchors default to
        absolute coordinates, otherwise normalized coordinates are produced.
      im_width: the width of the image to generate the grid for. If both
        im_height and im_width are 1, the generated anchors default to
        absolute coordinates, otherwise normalized coordinates are produced.
    Returns:
      boxes_list: a list of BoxLists each holding anchor boxes corresponding to
        the input feature map shapes.
    Raises:
      ValueError: if feature_map_shape_list, box_specs_list do not have the same
        length.
      ValueError: if feature_map_shape_list does not consist of pairs of
        integers
    """
        if not (isinstance(feature_map_shape_list, list) and
                len(feature_map_shape_list) == len(self._box_specs)):
            raise ValueError('feature_map_shape_list must be a list with the same '
                             'length as self._box_specs')
        if not all([
                isinstance(list_item, tuple) and len(list_item) == 2
                for list_item in feature_map_shape_list
        ]):
            raise ValueError('feature_map_shape_list must be a list of pairs.')

        im_height = float(im_height)
        im_width = float(im_width)

        if not self._anchor_strides:
            anchor_strides = [
                (1.0 / float(pair[0]), 1.0 / float(pair[1])) for pair in feature_map_shape_list
            ]
        else:
            anchor_strides = [(float(stride[0]) / im_height, float(stride[1]) / im_width)
                              for stride in self._anchor_strides]
        if not self._anchor_offsets:
            anchor_offsets = [(0.5 * stride[0], 0.5 * stride[1]) for stride in anchor_strides]
        else:
            anchor_offsets = [(float(offset[0]) / im_height, float(offset[1]) / im_width)
                              for offset in self._anchor_offsets]

        for arg, arg_name in zip([anchor_strides, anchor_offsets],
                                 ['anchor_strides', 'anchor_offsets']):
            if not (isinstance(arg, list) and len(arg) == len(self._box_specs)):
                raise ValueError('%s must be a list with the same length '
                                 'as self._box_specs' % arg_name)
            if not all([isinstance(list_item, tuple) and len(list_item) == 2 for list_item in arg
                       ]):
                raise ValueError('%s must be a list of pairs.' % arg_name)

        anchor_grid_list = []
        min_im_shape = min(im_height, im_width)
        scale_height = min_im_shape / im_height
        scale_width = min_im_shape / im_width
        base_anchor_size = [
            scale_height * self._base_anchor_size[0], scale_width * self._base_anchor_size[1]
        ]
        for feature_map_index, (grid_size, scales, aspect_ratios, stride, offset) in enumerate(
                zip(feature_map_shape_list, self._scales, self._aspect_ratios, anchor_strides,
                    anchor_offsets)):
            tiled_anchors = tile_anchors(grid_height=grid_size[0],
                                         grid_width=grid_size[1],
                                         scales=scales,
                                         aspect_ratios=aspect_ratios,
                                         base_anchor_size=base_anchor_size,
                                         anchor_stride=stride,
                                         anchor_offset=offset)
            if self._clip_window is not None:
                raise NotImplementedError("Oups!")
            num_anchors_in_layer = len(tiled_anchors)
            anchor_indices = feature_map_index * torch.ones(num_anchors_in_layer)
            anchor_grid_list.append(tiled_anchors)

        return anchor_grid_list
