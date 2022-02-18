r"""
Bounding box utils.
Consist of all the util to operate the bbox.
"""

import torch
import numpy as np

__all__ = ['xywhn2xywhn']


def xywhn2xywhn(bbox, hw_nopad: tuple, hw_pad: tuple, pxy: tuple):
    r"""
    convert the xywh normalized to the xywh normalized resized and padded.
    Args:
        bbox: = bbox shape(n, 4)
        hw_nopad: tuple = shape with no padding (h0, w0)
        hw_pad: tuple = shape with padding (h, w)
        pxy: tuple = padding xy (left, top)

    Return bbox (converted)
    """
    y = bbox.clone() if isinstance(bbox, torch.Tensor) else np.copy(bbox)
    h0, w0 = hw_nopad
    h, w = hw_pad
    y[:, 0] = (w0 * bbox[:, 0] + pxy[0]) / w
    y[:, 1] = (h0 * bbox[:, 1] + pxy[1]) / h
    return y
