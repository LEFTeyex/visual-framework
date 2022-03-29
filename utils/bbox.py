r"""
Bounding box utils.
Consist of all the util to operate the bbox.
"""

import math
import torch
import numpy as np

from torch import Tensor

from utils.typeslib import _Tensor_or_ndarray

__all__ = ['xywh2xyxy', 'xywhn2xyxy', 'xyxy2x1y1wh', 'xyxy2xywhn',
           'rescale_xywhn', 'rescale_xyxy', 'clip_bbox', 'bbox_iou']


def xywh2xyxy(bbox: _Tensor_or_ndarray, pxy: tuple = (0, 0)):
    r"""
    Convert the center xywh to the topleft and bottomright xyxy.
    Args:
        bbox: _Tensor_or_ndarray = bbox shape(..., 4)
        pxy: tuple = padding xy (left, top)

    Returns:
        bbox (converted)
    """
    y = bbox.clone() if isinstance(bbox, Tensor) else np.copy(bbox)
    px, py = pxy
    y[..., 0] = bbox[..., 0] - bbox[..., 2] / 2 + px  # top left x
    y[..., 1] = bbox[..., 1] - bbox[..., 3] / 2 + py  # top left y
    y[..., 2] = bbox[..., 0] + bbox[..., 2] / 2 + px  # bottom right x
    y[..., 3] = bbox[..., 1] + bbox[..., 3] / 2 + py  # bottom right y
    return y


def xywhn2xyxy(bbox: _Tensor_or_ndarray, hw_nopad: tuple, pxy: tuple = (0, 0)):
    r"""
    Convert the center xywh normalized to the topleft and bottomright xyxy.
    Args:
        bbox: _Tensor_or_ndarray = bbox shape(..., 4)
        hw_nopad: tuple = shape with no padding (h0, w0)
        pxy: tuple = padding xy (left, top)

    Returns:
        bbox (converted)
    """
    y = bbox.clone() if isinstance(bbox, Tensor) else np.copy(bbox)
    h, w = hw_nopad
    px, py = pxy
    y[..., 0] = w * (bbox[..., 0] - bbox[..., 2] / 2) + px  # top left x
    y[..., 1] = h * (bbox[..., 1] - bbox[..., 3] / 2) + py  # top left y
    y[..., 2] = w * (bbox[..., 0] + bbox[..., 2] / 2) + px  # bottom right x
    y[..., 3] = h * (bbox[..., 1] + bbox[..., 3] / 2) + py  # bottom right y
    return y


def xyxy2x1y1wh(bbox: _Tensor_or_ndarray, pxy: tuple = (0, 0)):
    r"""
    Convert the topleft and bottomright xyxy to the center xywh normalized
    Args:
        bbox: _Tensor_or_ndarray = bbox shape(..., 4)
        pxy: tuple = padding xy (left, top)

    Returns:
        bbox (converted)
    """
    y = bbox.clone() if isinstance(bbox, Tensor) else np.copy(bbox)
    px, py = pxy
    y[..., 0] = bbox[..., 0] + px  # top left x
    y[..., 1] = bbox[..., 1] + py  # top left y
    y[..., 2] = bbox[..., 2] - bbox[..., 0]  # w
    y[..., 3] = bbox[..., 3] - bbox[..., 1]  # h
    return y


def xyxy2xywhn(bbox: _Tensor_or_ndarray, hw_pad: tuple, pxy: tuple = (0, 0)):
    r"""
    Convert the topleft and bottomright xyxy to the center xywh normalized
    Args:
        bbox: _Tensor_or_ndarray = bbox shape(..., 4)
        hw_pad: tuple = shape with padding (h, w)
        pxy: tuple = padding xy (left, top)

    Returns:
        bbox (converted)
    """
    y = bbox.clone() if isinstance(bbox, Tensor) else np.copy(bbox)

    h, w = hw_pad
    px, py = pxy
    y[..., 0] = (bbox[..., 0] + bbox[..., 2]) / 2 + px  # center x
    y[..., 1] = (bbox[..., 1] + bbox[..., 3]) / 2 + py  # center y
    y[..., 2] = bbox[..., 2] - bbox[..., 0]  # w
    y[..., 3] = bbox[..., 3] - bbox[..., 1]  # h

    y[..., [0, 2]] /= w
    y[..., [1, 3]] /= h
    return y


def rescale_xywhn(bbox: _Tensor_or_ndarray, hw_nopad: tuple, hw_pad: tuple, pxy: tuple):
    r"""
    Rescale the center xywh normalized from original to which is resized and padded.
    Args:
        bbox: _Tensor_or_ndarray = bbox shape(..., 4)
        hw_nopad: tuple = shape with no padding (h0, w0)
        hw_pad: tuple = shape with padding (h, w)
        pxy: tuple = padding xy (left, top)

    Returns:
        bbox (converted)
    """
    y = bbox.clone() if isinstance(bbox, Tensor) else np.copy(bbox)
    h0, w0 = hw_nopad
    h, w = hw_pad
    px, py = pxy
    y[..., 0] = (w0 * bbox[..., 0] + px) / w  # center x
    y[..., 1] = (h0 * bbox[..., 1] + py) / h  # center y
    y[..., 2] = (w0 * bbox[..., 2]) / w  # w
    y[..., 3] = (h0 * bbox[..., 3]) / h  # h
    return y


def rescale_xyxy(bbox: _Tensor_or_ndarray, shape_converts):
    r"""
    Rescale xyxy from new image size to original image size.
    Args:
        bbox: _Tensor_or_ndarray = bbox shape(..., 4)
        shape_converts: = (hw0, ratio, padxy)

    Return bbox (converted)
    """
    y = bbox.clone() if isinstance(bbox, Tensor) else np.copy(bbox)
    hw0, ratio, (px, py) = shape_converts
    y[..., [0, 2]] = bbox[..., [0, 2]] - px
    y[..., [1, 3]] = bbox[..., [1, 3]] - py
    y[..., :4] /= ratio
    y = clip_bbox(y, hw0)
    return y


def clip_bbox(bbox: _Tensor_or_ndarray, shape):
    r"""
    Clip bounding box to image size(shape).
    Args:
        bbox: _Tensor_or_ndarray = bbox shape(..., 4)
        shape: = (h, w) to clip

    Return bbox (converted)
    """
    y = bbox.clone() if isinstance(bbox, Tensor) else np.copy(bbox)
    h, w = shape
    y[..., [0, 2]] = bbox[..., [0, 2]].clip(0, w)
    y[..., [1, 3]] = bbox[..., [1, 3]].clip(0, h)
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
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
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
