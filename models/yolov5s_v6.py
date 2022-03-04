r"""
YOLOv5 v6.0.
Add in 2022.03.04.
"""

import torch
import torch.nn as nn

from models.units import Conv, C3, SPPF
from utils.log import LOGGER

__all__ = ['Yolov5sV6']


class Backbone(nn.Module):
    def __init__(self, inc: int, c, n, g=1, act='relu', bn=True, bias=False, shortcut=True):
        super(Backbone, self).__init__()
        self.block1 = nn.Sequential(
            Conv(inc, c[0], 6, 2, p=2, g=g, act=act, bn=bn, bias=bias),  # 0
            Conv(c[0], c[1], 3, 2, g=g, act=act, bn=bn, bias=bias),  # 1
            C3(c[1], c[2], n=n[0], shortcut=shortcut, g=g, act=act, bn=bn, bias=bias),  # 2
            Conv(c[2], c[3], 3, 2, g=g, act=act, bn=bn, bias=bias),  # 3
            C3(c[3], c[4], n=n[1], shortcut=shortcut, g=g, act=act, bn=bn, bias=bias)  # 4
        )
        self.block2 = nn.Sequential(
            Conv(c[4], c[5], 3, 2, g=g, act=act, bn=bn, bias=bias),  # 5
            C3(c[5], c[6], n=n[2], shortcut=shortcut, g=g, act=act, bn=bn, bias=bias)  # 6
        )
        self.block3 = nn.Sequential(
            Conv(c[6], c[7], 3, 2, g=g, act=act, bn=bn, bias=bias),  # 7
            C3(c[7], c[8], n=n[3], shortcut=shortcut, g=g, act=act, bn=bn, bias=bias),  # 8
            SPPF(c[8], c[9], g=g, k=5, act=act, bn=bn, bias=bias),  # 9
            Conv(c[9], c[10], 1, 1, g=g, act=act, bn=bn, bias=bias)  # 10
        )

    def forward(self, x):
        out4 = self.block1(x)
        out6 = self.block2(out4)
        out10 = self.block3(out6)
        return out4, out6, out10


class Head(nn.Module):
    def __init__(self, c, n, g=1, act='relu', bn=True, bias=False, shortcut=False):
        super(Head, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.block1 = nn.Sequential(
            C3(c[6] * 2, c[6], n=n[0], shortcut=shortcut, g=g, act=act, bn=bn, bias=bias),
            Conv(c[6], c[4], 1, 1, g=g, act=act, bn=bn, bias=bias)  # 14
        )
        self.conv1 = C3(c[4] * 2, c[4], n=n[0], shortcut=shortcut, g=g, act=act, bn=bn, bias=bias)  # 17
        self.conv2 = Conv(c[4], c[4], 3, 2, g=g, act=act, bn=bn, bias=bias)
        self.conv3 = C3(c[4] * 2, c[4] * 2, n=n[1], shortcut=shortcut, g=g, act=act, bn=bn, bias=bias)  # 20
        self.conv4 = Conv(c[4] * 2, c[4] * 2, 3, 2, g=g, act=act, bn=bn, bias=bias)
        self.conv5 = C3(c[10] * 2, c[10] * 2, n=n[2], shortcut=shortcut, g=g, act=act, bn=bn, bias=bias)  # 23

    def forward(self, x):
        out4, out6, out10 = x
        x = torch.cat((self.up(out10), out6), dim=1)
        out14 = self.block1(x)
        x = torch.cat((self.up(out14), out4), dim=1)
        out17 = self.conv1(x)
        x = torch.cat((self.conv2(out17), out14), dim=1)
        out20 = self.conv3(x)
        x = torch.cat((self.conv4(out20), out10), dim=1)
        out23 = self.conv5(x)
        return out17, out20, out23


class Yolov5sV6(nn.Module):
    # TODO Upgrade for somewhere in the future
    r"""YOLOv5 v6.0"""

    def __init__(self, inc: int, nc: int, anchors: list, num_bbox: int = 5, image_size: int = 640,
                 g=1, act='silu', bn=True, bias=True):
        super(Yolov5sV6, self).__init__()
        LOGGER.info('Initializing the model YOLOv5 v6.0...')

        # todo args can change
        c = [32, 64, 64, 128, 128, 256, 256, 512, 512, 512, 256]  # channels from 1 to -1
        n = [1, 2, 3, 1, 1, 1, 1]  # number of layers for C3

        self.image_size = image_size
        self.inc = inc
        self.nc = nc
        self.no = nc + num_bbox
        self.ch = [c[4], c[4] * 2, c[10] * 2]
        anchors, self.nl, self.na = self._tensor_anchors(anchors)
        self._check_ch_nl()

        self.backbone = Backbone(inc, c, n[:4], g=g, act=act, bn=bn, bias=bias, shortcut=True)
        self.head = Head(c, n[4:], g=g, act=act, bn=bn, bias=bias, shortcut=False)
        k, s = (1, 1), (1, 1)
        self.m = nn.ModuleList([nn.Conv2d(x, self.no * self.na, k, s) for x in self.ch])

        scalings = self._compute_scaling()
        anchors = self._scale_anchors(anchors, scalings)
        self.register_buffer('scalings', scalings)
        self.register_buffer('anchors', anchors)

        LOGGER.info('Initialize model YOLOv5 v6.0 successfully')

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        y = []  # save outputs
        for index, tensor in enumerate(x):
            tensor = self.m[index](tensor)
            bs, _, h, w = tensor.shape
            # to shape (bs, na, h, w, no)
            tensor = tensor.view(bs, self.na, self.no, h, w).permute(0, 1, 3, 4, 2).contiguous()
            y.append(tensor)
        return y

    def _check_ch_nl(self):
        if len(self.ch) != self.nl:
            raise ValueError(f'The length of self.ch {len(self.ch)} do not match self.nl')

    def _compute_scaling(self):
        image = torch.zeros(1, self.inc, self.image_size, self.image_size)
        outputs = self.forward(image)
        scalings = torch.tensor([self.image_size / x.shape[-2] for x in outputs])
        return scalings

    @staticmethod
    def _scale_anchors(anchors, scalings):
        anchors /= scalings.view(-1, 1, 1)
        return anchors

    @staticmethod
    def _tensor_anchors(anchors):
        anchors = torch.tensor(anchors).float()  # shape (3, 3, 2) for (nl, na, wh)
        nl, na = anchors.shape[0:2]
        return anchors, nl, na
