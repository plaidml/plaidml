import torch
import math

from torch import nn
from torch.nn import functional as F


class BatchNorm2d(nn.Module):
    """
    Fixed version of BatchNorm2d, which has only the scale and bias
    """

    def __init__(self, out):
        super(BatchNorm2d, self).__init__()
        self.register_buffer("scale", torch.ones(out))
        self.register_buffer("bias", torch.zeros(out))

    #@torch.jit.script_method
    def forward(self, x):
        scale = self.scale.view(1, -1, 1, 1)
        bias = self.bias.view(1, -1, 1, 1)
        return x * scale + bias


class BiasAdd(nn.Module):
    """
    Fixed version of BatchNorm2d, which has only the scale and bias
    """

    def __init__(self, out):
        super(BiasAdd, self).__init__()
        self.register_buffer("bias", torch.zeros(out))

    #@torch.jit.script_method
    def forward(self, x):
        bias = self.bias.view(1, -1, 1, 1)
        return x + bias


class Conv2d_tf(nn.Conv2d):
    """
    Conv2d with the padding behavior from TF
    """

    def __init__(self, *args, **kwargs):
        super(Conv2d_tf, self).__init__(*args, **kwargs)
        self.padding = kwargs.get("padding", "SAME")

    def _compute_padding(self, input, dim):
        input_size = input.size(dim + 2)
        filter_size = self.weight.size(dim + 2)
        effective_filter_size = (filter_size - 1) * self.dilation[dim] + 1
        out_size = (input_size + self.stride[dim] - 1) // self.stride[dim]
        total_padding = max(0,
                            (out_size - 1) * self.stride[dim] + effective_filter_size - input_size)
        additional_padding = int(total_padding % 2 != 0)

        return additional_padding, total_padding

    def forward(self, input):
        #import pdb; pdb.set_trace()
        if self.padding == "VALID":
            return F.conv2d(
                input,
                self.weight,
                self.bias,
                self.stride,
                padding=0,
                dilation=self.dilation,
                groups=self.groups,
            )
        rows_odd, padding_rows = self._compute_padding(input, dim=0)
        cols_odd, padding_cols = self._compute_padding(input, dim=1)
        if rows_odd or cols_odd:
            input = F.pad(input, [0, cols_odd, 0, rows_odd])

        return F.conv2d(
            input,
            self.weight,
            self.bias,
            self.stride,
            padding=(padding_rows // 2, padding_cols // 2),
            dilation=self.dilation,
            groups=self.groups,
        )


def box_area(left_top, right_bottom):
    """Compute the areas of rectangles given two corners.

    Args:
        left_top (N, 2): left top corner.
        right_bottom (N, 2): right bottom corner.

    Returns:
        area (N): return the area.
    """
    hw = torch.clamp(right_bottom - left_top, min=0.0)
    return hw[..., 0] * hw[..., 1]


def box_iou(boxes0, boxes1, eps=1e-5):
    """Return intersection-over-union (Jaccard index) of boxes.

    Args:
        boxes0 (N, 4): ground truth boxes.
        boxes1 (N or 1, 4): predicted boxes.
        eps: a small number to avoid 0 as denominator.
    Returns:
        iou (N): IoU values.
    """
    overlap_left_top = torch.max(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = torch.min(boxes0[..., 2:], boxes1[..., 2:])

    overlap_area = box_area(overlap_left_top, overlap_right_bottom)
    area0 = box_area(boxes0[..., :2], boxes0[..., 2:])
    area1 = box_area(boxes1[..., :2], boxes1[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)


def nms(box_scores, iou_threshold):
    """

    Args:
        box_scores (N, 5): boxes in corner-form and probabilities.
        iou_threshold: intersection over union threshold.
    Returns:
         picked: a list of indexes of the kept boxes
    """
    scores = box_scores[:, -1]
    boxes = box_scores[:, :-1]
    picked = []
    _, indexes = scores.sort(descending=True)
    while len(indexes) > 0:
        current = indexes[0]
        picked.append(current.item())
        if len(indexes) == 1:
            break
        current_box = boxes[current, :]
        indexes = indexes[1:]
        rest_boxes = boxes[indexes, :]
        iou = box_iou(rest_boxes, current_box.unsqueeze(0))
        indexes = indexes[iou <= iou_threshold]

    return box_scores[picked, :]


@torch.jit.script
def decode_boxes(rel_codes, boxes, weights):
    # type: (torch.Tensor, torch.Tensor, torch.Tensor) -> torch.Tensor

    # perform some unpacking to make it JIT-fusion friendly

    #rel_codes=rel_codes[0][None]
    wx = weights[1]
    wy = weights[0]
    ww = weights[3]
    wh = weights[2]

    boxes_x1 = boxes[:, 1].unsqueeze(1).unsqueeze(0)
    boxes_y1 = boxes[:, 0].unsqueeze(1).unsqueeze(0)
    boxes_x2 = boxes[:, 3].unsqueeze(1).unsqueeze(0)
    boxes_y2 = boxes[:, 2].unsqueeze(1).unsqueeze(0)

    dx = rel_codes[:, :, 1].unsqueeze(2)
    dy = rel_codes[:, :, 0].unsqueeze(2)
    dw = rel_codes[:, :, 3].unsqueeze(2)
    dh = rel_codes[:, :, 2].unsqueeze(2)

    # implementation starts here
    widths = boxes_x2 - boxes_x1
    heights = boxes_y2 - boxes_y1
    ctr_x = boxes_x1 + 0.5 * widths
    ctr_y = boxes_y1 + 0.5 * heights

    dx = dx / wx
    dy = dy / wy
    dw = dw / ww
    dh = dh / wh

    pred_ctr_x = dx * widths + ctr_x
    #import pdb; pdb.set_trace()
    pred_ctr_y = dy * heights + ctr_y
    pred_w = torch.exp(dw) * widths
    pred_h = torch.exp(dh) * heights

    pred_boxes = torch.cat(
        [
            pred_ctr_x - 0.5 * pred_w,
            pred_ctr_y - 0.5 * pred_h,
            pred_ctr_x + 0.5 * pred_w,
            pred_ctr_y + 0.5 * pred_h,
        ],
        dim=2,
    )
    #import pdb; pdb.set_trace()
    return pred_boxes
