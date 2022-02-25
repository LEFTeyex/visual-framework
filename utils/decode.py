r"""
Decode utils for training or detecting.
Consist of all the solution function.
"""

import torch

from torch import Tensor

from utils.typeslib import _tuple_or_None, _int_or_Tensor

__all__ = ['parse_bbox_yolov5', 'parse_outputs_yolov5']


def parse_outputs_yolov5(outputs: tuple, anchors: Tensor, scalings: Tensor):
    for index, (output, anchor, scaling) in enumerate(zip(outputs, anchors, scalings)):
        na = len(anchor)
        anchor = anchor.view(1, na, 1, 1, 2)
        nxy_grid = output.shape[2:4:-1]
        # parse bbox
        output[..., 0:4] = parse_bbox_yolov5(output[..., 0:4], anchor, nxy_grid, scaling)
        # parse others (object and classes)
        output[..., 4:] = output[..., 4:].sigmoid()
        outputs[index] = output
        # TODO 2022.2.26
    return outputs


def parse_bbox_yolov5(bbox: Tensor, anchor: Tensor, nxy_grid: _tuple_or_None = None, scaling: _int_or_Tensor = 1):
    r"""
    Parse the bbox from output of model.
    Args:

        bbox: Tensor = shape(..., 4)
        anchor: Tensor = anchor which the shape must be corresponding to bbox (wh)
        nxy_grid: _tuple_or_None = (nx, ny) number of xy for grid
        scaling: _int_or_Tensor = scale of image (xy)

    Return bbox is Tensor(..., 4)
    """
    if nxy_grid is not None:
        ndim = bbox.ndim
        device = bbox.device
        # create grid for xy
        grid_hwxy = create_grid_tensor(nxy_grid, ndim, device)
    else:
        grid_hwxy = 0

    xy = (bbox[..., :2].sigmoid() * 2 - 0.5 + grid_hwxy) * scaling
    wh = (bbox[..., 2:].sigmoid() * 2) ** 2 * anchor * scaling
    bbox = torch.cat((xy, wh), dim=-1)
    return bbox


def create_grid_tensor(nxy_grid: tuple, ndim: int, device):
    assert len(nxy_grid) == 2, f'Excepted 2 element in grid for xy, but got {len(nxy_grid)} element'
    assert ndim >= 3, f'Excepted 3 dimension for bbox, but got {ndim} dimensions'

    pre_shape = [1 for _ in range(ndim - 3)]
    nx, ny = nxy_grid

    # create grid for xy which the shape is (..., h, w, 2(x, y))
    grid_x = torch.arange(nx, device=device)
    grid_y = torch.arange(ny, device=device)
    grid_x, grid_y = torch.meshgrid(grid_x, grid_y, indexing='xy')
    grid_hwxy = torch.stack((grid_x, grid_y), dim=2)
    grid_hwxy = grid_hwxy.view(*pre_shape, ny, nx, 2)
    return grid_hwxy


if __name__ == '__main__':
    a = torch.tensor([1])
    b = torch.tensor([1.])
    print(a.dtype)
    print(b.dtype)
    c = torch.cat((a, b), dim=0)
    print(c)
    print(c.dtype)
