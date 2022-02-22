r"""
Loss Module.
Consist of all the Loss class.
"""

import torch
import torch.nn as nn

from torch import Tensor

from utils.bbox import bbox_iou
from utils.decode import solve_bbox_yolov5
from utils.typeslib import _module

__all__ = ['LossDetectYolov5']


class LossDetectYolov5(object):
    r"""
    Compute loss with outputs, labels.
    Convert labels for loss.
    Args:
        model: _module = model instance (to get some parameters of model)
        hyp: dict = self.hyp during training
    """

    def __init__(self, model: _module, hyp: dict):

        self.hyp = hyp
        self.nc = model.nc
        self.device = next(model.parameters()).device
        self.anchors = model.anchors  # anchors(3, 3, 2(w, h)) which the size is corresponding to output size

        # class positive, class negative for label smoothing
        self.cp, self.cn = label_smooth_yolov5(hyp['label_smoothing'])

        # loss function
        self.loss_cls_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([hyp['cls_pw']], device=self.device))
        self.loss_obj_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([hyp['obj_pw']], device=self.device))

        # Focal loss
        g = hyp['fl_gamma']  # focal loss gamma
        if g > 0:
            self.loss_cls_fn, self.loss_obj_fn = FocalLoss(self.loss_cls_fn, g), FocalLoss(self.loss_obj_fn, g)

    def __call__(self, outputs: Tensor, labels: Tensor, kind: str = 'iou'):
        r"""
        Compute loss.
        Args:
            outputs: Tensor = outputs during model forward
            labels: Tensor = labels from Dataloader
            kind: str = 'iou' / 'giou' / 'diou' / 'ciou' the kind of IoU

        Return loss, loss_items
        """
        bs = outputs[0].shape[0]  # batch size
        # initialize all loss
        loss_obj, loss_cls, loss_bbox = [torch.zeros(1, device=self.device) for _ in range(3)]
        # get labels and others for computing loss
        index_cls, labels_bbox, indices, anchors = self.convert_labels_for_loss_yolov5(outputs, labels)

        # compute losses
        for index, output in enumerate(outputs):  # outputs is [(bs, a, h, w, nlabel+classes)] * 3
            b, a, gy, gx = indices[index]  # index of batch, anchor, grid_y, grid_x
            labels_obj = torch.zeros_like(output[..., 0], device=self.device)

            n = b.shape[0]
            if n:
                output_filter = output[b, a, gy, gx]  # shape(n, nlabel+classes)

                # bbox loss regression
                bbox = solve_bbox_yolov5(output_filter[:, 0:4], anchors[index])
                iou = bbox_iou(bbox, labels_bbox[index], xyxy=False, kind=kind)
                loss_bbox += (1.0 - iou).mean()

                # class loss
                if self.nc > 1:
                    labels_cls = torch.full_like(output_filter[:, 5:], self.cn, device=self.device)
                    labels_cls[range(n), index_cls[index]] = self.cp
                    loss_cls += self.loss_cls_fn(output_filter[:, 5:], labels_cls)
                else:
                    raise NotImplementedError(f'The loss for {self.nc} class has not implemented')

                # object labels
                score_iou = iou.detach().clamp(0).type(labels_obj.dtype)
                labels_obj[b, a, gy, gx] = score_iou

            # object loss
            loss_obj += self.loss_obj_fn(output[..., 4], labels_obj)

        # deal all loss
        loss_bbox *= self.hyp['bbox'] * bs
        loss_cls *= self.hyp['cls'] * bs
        loss_obj *= self.hyp['obj'] * bs

        loss = (loss_bbox + loss_cls + loss_obj)
        loss_items = torch.cat((loss_bbox, loss_cls, loss_obj))
        return loss, loss_items

    def convert_labels_for_loss_yolov5(self, outputs: Tensor, labels: Tensor):
        r"""
        Convert labels for computing loss.
        Args:
            outputs: Tensor = outputs is [(bs, 3, s, s, 85), ] * number of output
            labels: Tensor = labels is shape (nl, 6) [(bs_index, class, x,y,w,h), ...]

        Return index_cls, labels_bbox, indices, anchors
        """
        no = len(outputs)  # number of output
        index_cls, labels_bbox, indices, anchors = [], [], [], []
        na, nl = self.anchors.shape[1], labels.shape[0]  # number of anchors, labels
        gain = torch.ones(7, device=self.device)  # normalized to grid space gain
        ai = torch.arange(na, device=self.device).float().view(na, 1, 1).repeat(1, nl, 1)  # anchor index
        labels = torch.cat((ai, labels.repeat(na, 1, 1)), dim=2)  # append anchor index for labels
        # labels is shape(na, nl, 7) (anchor_index, bs_index, class, x,y,w,h)

        bias = 0.5
        # corresponding to (center, left, top, right, bottom)
        off = bias * torch.tensor([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]], device=self.device).float()

        for index in range(no):
            anchor = self.anchors[index]
            gain[3:] = torch.tensor(outputs[index].shape[2:4])[[1, 0, 1, 0]]  # [1, 1, 1, w, h, w, h]

            label = labels * gain  # convert normalized to output size
            if nl:
                # match anchor_threshold
                r = label[:, :, 5:] / anchor[:, None]  # anchor(3, 1, 2) wh ratio
                index_filter = torch.max(r, 1 / r).max(dim=2)[0] < self.hyp['anchor_threshold']
                label = label[index_filter]  # shape(x, 7)

                # offsets
                gxy = label[:, 3:5]  # grid xy
                gxyi = gain[[3, 4]] - gxy  # grid xy inverse
                # get left, top, right, bottom True / False shape(x)
                left, top = ((gxy % 1 < bias) & (gxy > 1)).T
                right, bottom = ((gxyi % 1 < bias) & (gxyi > 1)).T

                index_filter = torch.stack((torch.ones_like(left), left, top, right, bottom), dim=0)
                label = label.repeat((5, 1, 1))[index_filter]  # shape(x, 7)
                # (1,x,2) + (5,1,2) broadcast to (5,x,2)[index_filter]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[index_filter]
            else:
                # no labels
                label = labels[0]
                offsets = 0

            # label shape [x, 7(anchor_index, bs_index, class, x,y,w,h)]
            a, b, c = label[:, :3].long().T  # index of anchor, batch, class
            gxy = label[:, 3:5]  # grid xy
            gwh = label[:, 5:]  # grid wh
            gxy_long = (gxy - offsets).long()

            bbox = torch.cat((gxy - gxy_long, gwh), dim=1)

            gx, gy = gxy_long.T
            gx.clamp_(0, gain[3] - 1)  # x
            gy.clamp_(0, gain[4] - 1)  # y

            # append
            index_cls.append(c)  # for one hot class label
            labels_bbox.append(bbox)  # bbox
            indices.append((b, a, gy, gx))  # index of batch, anchor, y, x
            anchors.append(anchor[a])  # anchors

        return index_cls, labels_bbox, indices, anchors


class FocalLoss(nn.Module):
    r"""Yolov5"""

    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, output, true):
        loss = self.loss_fcn(output, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        output_prob = torch.sigmoid(output)  # prob from logits
        p_t = true * output_prob + (1 - true) * (1 - output_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    r"""Yolov5"""

    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, output, true):
        loss = self.loss_fcn(output, true)

        output_prob = torch.sigmoid(output)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - output_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


def label_smooth_yolov5(epsilon=0.1):
    r"""For label smoothing"""
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * epsilon, 0.5 * epsilon
