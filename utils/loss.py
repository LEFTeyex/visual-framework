r"""
Loss Module.
Consist of all the Loss class.
"""

import torch

from utils.decode import solve_bbox_yolov5

__all__ = ['LossDetectYolov5']


# TODO label smoothing before going to school

class LossDetectYolov5(object):
    def __init__(self, model, loss_weights: dict):
        self.anchors = model.anchors
        self.nc = model.nc
        self.loss_weights = loss_weights
        self.device = next(model.parameters()).device

    def __call__(self, outputs, labels):
        device = labels.device
        bs = outputs.shape[0]  # batch size
        # initialize all loss
        loss_obj, loss_cls, loss_bbox = [torch.zeros(1, device=device) for _ in range(3)]
        # get labels and others for computing loss
        index_cls, labels_bbox, indices, anchors = self.convert_labels_loss_yolov5(1, 1)

        # compute losses
        for index, output in enumerate(outputs):  # outputs is [(bs, a, h, w, nlabel+classes)] * 3
            b, a, gy, gx = indices[index]  # index of batch, anchor, grid_y, grid_x
            labels_obj = torch.zeros_like(output[..., 0], device=device)

            n = b.shape[0]
            if n:
                output_filter = output[b, a, gy, gx]  # shape(n, nlabel+classes)

                # bbox loss regression
                bbox = solve_bbox_yolov5(output_filter[:, 0:4], anchors[index])
                iou = bbox_iou()  # TODO 2022.2.20
                loss_bbox += (1.0 - iou).mean()

                # class loss
                if self.nc > 1:
                    # TODO 0.0 1.0 can be transposed by smoothing mothed
                    labels_cls = torch.full_like(output_filter[:, 5:], 0.0, device=device)
                    labels_cls[range(n), index_cls[index]] = 1.0
                    loss_cls += loss_cls_fn()  # TODO 2022.2.20
                else:
                    raise NotImplementedError(f'The loss for {self.nc} class has not implemented')

                # object labels
                score_iou = iou.detach().clamp(0).type(labels_obj.dtype)
                labels_obj[b, a, gy, gx] = score_iou

            # object loss
            loss_obj += loss_obj_fn()  # TODO 2022.2.20

        # deal all loss
        loss_bbox *= self.loss_weights['bbox']
        loss_cls *= self.loss_weights['cls']
        loss_obj *= self.loss_weights['obj']

        loss = (loss_bbox + loss_cls + loss_obj) * bs
        return loss

    def convert_labels_loss_yolov5(self, outputs, labels):
        # outputs is [(bs, 3, 85, s, s), ...]
        # labels is shape (nt, 6) [(bs_index, class, x,y,w,h), ]

        anchor = self.anchors
        return [], [], [], []
