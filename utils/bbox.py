r"""
Bounding box utils.
Consist of all the util to operate the bbox.
"""

import torch
import numpy as np

from torch import Tensor

from utils.typeslib import _Tensor_or_ndarray

__all__ = ['xywh2xyxy', 'xywhn2xywhn', 'bbox_iou']


def xywh2xyxy(bbox: _Tensor_or_ndarray):
    r"""
    Convert the center xywh to the topleft and bottomright xyxy.
    Args:
        bbox: _Tensor_or_ndarray = bbox shape(..., 4)

    Return bbox (converted)
    """
    y = bbox.clone() if isinstance(bbox, Tensor) else np.copy(bbox)
    y[..., 0] = bbox[..., 0] - bbox[..., 2] / 2  # top left x
    y[..., 1] = bbox[..., 1] - bbox[..., 3] / 2  # top left y
    y[..., 2] = bbox[..., 0] + bbox[..., 2] / 2  # bottom right x
    y[..., 3] = bbox[..., 1] + bbox[..., 3] / 2  # bottom right y
    return y


def xywhn2xywhn(bbox: _Tensor_or_ndarray, hw_nopad: tuple, hw_pad: tuple, pxy: tuple):
    r"""
    Convert the center xywh normalized to the center xywh normalized resized and padded.
    Args:
        bbox: _Tensor_or_ndarray = bbox shape(..., 4)
        hw_nopad: tuple = shape with no padding (h0, w0)
        hw_pad: tuple = shape with padding (h, w)
        pxy: tuple = padding xy (left, top)

    Return bbox (converted)
    """
    y = bbox.clone() if isinstance(bbox, Tensor) else np.copy(bbox)
    h0, w0 = hw_nopad
    h, w = hw_pad
    y[..., 0] = (w0 * bbox[..., 0] + pxy[0]) / w  # center x
    y[..., 1] = (h0 * bbox[..., 1] + pxy[1]) / h  # center y
    return y


def bbox_iou(bbox1: Tensor, bbox2: Tensor, *, xyxy: bool = False, kind: str = 'iou', eps=1e-7):
    r"""
    Compute iou between bbox1 and bbox2 that is corresponding to each other.
    Args:
        bbox1: Tensor = shape(n, 4) xywh or xyxy
        bbox2: Tensor = shape(n, 4)
        xyxy: bool = True / False, Default False which bbox is xywh
        kind: str = 'iou' / 'giou' / 'diou' / 'ciou' the kind of IoU
        eps: = a number to make it no zero

    Return iou
    """
    # transpose bbox1=(n, 4), bbox2=(n, 4) to bbox1=(4, n), bbox2=(4, n)
    bbox1, bbox2 = bbox1.T, bbox2.T

    # get left_top point and right_bottom point
    if xyxy:
        b1x1, b1y1, b1x2, b1y2 = bbox1[0], bbox1[1], bbox1[2], bbox1[3]
        b2x1, b2y1, b2x2, b2y2 = bbox2[0], bbox2[1], bbox2[2], bbox2[3]
    else:  # convert xywh to xyxy
        b1x1, b1x2 = bbox1[0] - bbox1[2] / 2, bbox1[0] + bbox1[2] / 2
        b1y1, b1y2 = bbox1[1] - bbox1[3] / 2, bbox1[1] + bbox1[3] / 2
        b2x1, b2x2 = bbox2[0] - bbox2[2] / 2, bbox2[0] + bbox2[2] / 2
        b2y1, b2y2 = bbox2[1] - bbox2[3] / 2, bbox2[1] + bbox2[3] / 2

    # intersection area
    inter = (torch.min(b1x2, b2x2) - torch.max(b1x1, b2x1)).clamp(0) * \
            (torch.min(b1y2, b2y2) - torch.max(b1y1, b2y1)).clamp(0)

    # union
    w1, h1 = b1x2 - b1x1, b1y2 - b1y1 + eps
    w2, h2 = b2x2 - b2x1, b2y2 - b2y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union

    kind = kind.lower().replace(' ', '')
    if kind in ('giou', 'diou', 'ciou'):
        # the cw, ch of smallest enclosing box GIoU https://arxiv.org/pdf/1902.09630.pdf
        cw = torch.max(b1x2, b2x2) - torch.min(b1x1, b2x1)
        ch = torch.max(b1y2, b2y2) - torch.min(b1y1, b2y1)
        if kind in ('diou', 'ciou'):  # DIoU and CIoU https://arxiv.org/abs/1911.08287v1
            c_distance = cw ** 2 + ch ** 2 + eps
            d_distance = ((b2x1 + b2x2 - b1x1 - b1x2) ** 2 + (b2y1 + b2y2 - b1y1 - b1y2) ** 2) / 4
            if kind == 'ciou':
                # v is the consistency of aspect ratio, alpha is the trade_off parameter
                v = (4 / torch.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (eps + 1))
                iou = iou - (d_distance / c_distance + alpha + v)  # CIoU
            else:
                iou = iou - d_distance / c_distance  # DIoU
        else:
            c_area = cw * ch + eps
            iou = iou - (c_area - union) / c_area  # GIoU
    else:
        # IoU
        if kind != 'iou':
            raise ValueError(f"The arg kind: {kind} do not in ('iou', 'giou', 'diou', 'ciou')")
    return iou
