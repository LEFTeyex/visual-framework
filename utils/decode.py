r"""
Decode utils for training or detecting.
Consist of all the solution function.
"""
import time

import torch

from torch import Tensor
from torchvision.ops import batched_nms

from utils.log import LOGGER
from utils.check import check_between_0_1
from utils.bbox import xywh2xyxy
from utils.typeslib import _tuple_or_None, _int_or_Tensor

__all__ = ['parse_bbox_yolov5', 'parse_outputs_yolov5', 'filter_outputs2predictions',
           'non_max_suppression']


def parse_outputs_yolov5(outputs: tuple, anchors: Tensor, scalings: Tensor):
    output_all = []
    for index, (output, anchor, scaling) in enumerate(zip(outputs, anchors, scalings)):
        bs, na, ny, nx, no = output.shape  # na-number of anchor, no-number of output
        anchor = anchor.view(1, na, 1, 1, 2)

        # parse bbox
        output[..., 0:4] = parse_bbox_yolov5(output[..., 0:4], anchor, (nx, ny), scaling)
        # parse others (object and classes)
        output[..., 4:] = output[..., 4:].sigmoid()
        output_all.append(output.view(bs, -1, no))
    output_all = torch.cat(output_all, dim=1)  # shape(bs, n, no)
    return output_all


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
        ndim_bbox = bbox.ndim
        device = bbox.device
        # create grid for xy
        grid_hwxy = create_grid_tensor(nxy_grid, ndim_bbox, device)
    else:
        grid_hwxy = 0

    xy = (bbox[..., :2].sigmoid() * 2 - 0.5 + grid_hwxy) * scaling
    wh = (bbox[..., 2:].sigmoid() * 2) ** 2 * anchor * scaling
    bbox = torch.cat((xy, wh), dim=-1)
    return bbox


def create_grid_tensor(nxy_grid: tuple, ndim_bbox: int, device):
    assert len(nxy_grid) == 2, f'Excepted 2 element in grid for xy, but got {len(nxy_grid)} element'
    assert ndim_bbox >= 3, f'Excepted 3 dimension for bbox, but got {ndim_bbox} dimensions'

    pre_shape = [1 for _ in range(ndim_bbox - 3)]
    nx, ny = nxy_grid

    # create grid for xy which the shape is (..., h, w, 2(x, y))
    grid_x = torch.arange(nx, device=device)
    grid_y = torch.arange(ny, device=device)
    grid_x, grid_y = torch.meshgrid(grid_x, grid_y, indexing='xy')
    grid_hwxy = torch.stack((grid_x, grid_y), dim=2)
    grid_hwxy = grid_hwxy.view(*pre_shape, ny, nx, 2)
    return grid_hwxy


def filter_outputs2predictions(outputs: Tensor, obj_threshold: float = 0.25, classes: _tuple_or_None = None):
    # check
    assert check_between_0_1(obj_threshold), f'Except obj_threshold value is in interval (0, 1), but got {obj_threshold}'
    cls_threshold = obj_threshold  # equal if wanted to be not equal need to change input args

    filter_obj = outputs[..., 4] > obj_threshold  # object
    predictions = [torch.zeros((0, 6), device=outputs.device)] * outputs.shape[0]

    for index, output in enumerate(outputs):
        output = output[filter_obj[index]]  # filter by object confidence

        # output not empty
        if not output.shape[0]:
            continue
        # TODO maybe need apriori label to cat by yolov5

        # compute  confidence = obj_conf * class_conf
        output[:, 5:] *= output[:, 4:5]
        bbox = xywh2xyxy(output[:, :4])

        # TODO maybe multi labels

        conf, cls_index = output[:, 5:].max(dim=-1, keepdim=True)  # only best class
        # filter by class confidence shape(n, 6) is [(x,y,x,y,conf,cls_index), ...]
        output = torch.cat((bbox, conf, cls_index), dim=-1)[conf.view(-1) > cls_threshold]

        if classes is not None:
            # filter by classes
            filter_classes = (output[:, 5:6] == torch.tensor(classes, device=output.device)).any(dim=-1)
            output = output[filter_classes]

        predictions[index] = output
    return predictions


def non_max_suppression(predictions: list, iou_threshold: float = 0.5, max_detect: int = 300):
    # check
    assert check_between_0_1(iou_threshold), f'Except iou_threshold value is in interval (0, 1), but got {iou_threshold}'

    # todo args can change
    # min_wh, max_wh = 2, 7680  # min and max bbox wh
    max_nms = 30000  # max number of bbox for nms
    time_limit = 10.0  # seconds to quit after

    t0 = time.time()
    outputs = [torch.zeros((0, 6), device=predictions[0].device)] * len(predictions)
    for index, pre in enumerate(predictions):
        # check
        n = pre.shape[0]
        if not n:
            continue

        # filter number
        if n > max_nms:
            filter_num = pre[:, 4].argsort(descending=True)[:max_nms]
            pre = pre[filter_num]

        # batched nms in torchvision
        boxes, scores, idxs = pre[:, :4], pre[:, 4], pre[:, 5]
        filter_pre = batched_nms(boxes, scores, idxs, iou_threshold)

        # TODO add many different nms methods

        # limit number of detection
        if filter_pre.shape[0] > max_detect:
            filter_pre = filter_pre[:max_detect]

        outputs[index] = pre[filter_pre]

        t = time.time() - t0
        if t > time_limit:
            LOGGER.warning(f'NMS time limit {time_limit:.2f}s exceeded')

    return outputs


if __name__ == '__main__':
    x = torch.rand(64, 10000, 85)
    print(x[0])
    x = filter_outputs2predictions(x)
    print(len(x))
