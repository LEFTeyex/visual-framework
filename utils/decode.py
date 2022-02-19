r"""
Decode utils for training or detecting.
Consist of all the solution function.
"""

import torch

from torch import Tensor

__all__ = ['solve_bbox_yolov5']


def solve_bbox_yolov5(output: Tensor, anchor: Tensor):
    r"""
    Solve the bbox from output of model.
    Args:
        output: Tensor = shape(..., 4)
        anchor: Tensor = anchor whose shape is corresponding to output wh

    Return bbox is Tensor(..., 4)
    """
    xy = output[..., :2].sigmoid() * 2 - 0.5
    wh = (output[..., 2:].sigmoid() * 2) ** 2 * anchor
    bbox = torch.cat((xy, wh), dim=-1)
    return bbox
