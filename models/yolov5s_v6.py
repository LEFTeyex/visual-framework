r"""
YOLOv5 v6.0.
Add in 2022.03.04.
"""

import torch
import torch.nn as nn

from models.metamodel import MetaModelDetect, init_weights
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


class Yolov5sV6(MetaModelDetect):
    # TODO Upgrade for somewhere in the future
    r"""YOLOv5 v6.0"""

    def __init__(self, inc: int, nc: int, anchors: list, num_bbox: int = 5, image_size: int = 640,
                 g=1, act='silu', bn=True, bias=False):
        super(Yolov5sV6, self).__init__()
        LOGGER.info(f'Initializing the {type(self).__name__}...')
        # todo args can change
        c = [32, 64, 64, 128, 128, 256, 256, 512, 512, 512, 256]  # channels from 1 to -1
        n = [1, 2, 3, 1, 1, 1, 1]  # number of layers for C3

        self.inc = inc
        self.nc = nc
        self.no = nc + num_bbox
        self.ch = [c[4], c[4] * 2, c[10] * 2]

        self.anchors, self.nl, self.na = self.get_register_anchors(anchors)
        self._check_ch_nl()

        self.backbone = Backbone(inc, c, n[:4], g=g, act=act, bn=bn, bias=bias, shortcut=True)
        self.head = Head(c, n[4:], g=g, act=act, bn=bn, bias=bias, shortcut=False)
        k, s = (1, 1), (1, 1)
        self.m = nn.ModuleList([nn.Conv2d(x, self.no * self.na, k, s) for x in self.ch])

        self.scalings, self.image_size = self.get_register_scalings(image_size)
        self.scale_anchors()

        LOGGER.info(f'Initialize {type(self).__name__} successfully')

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

    def initialize_weights(self):
        self.apply(init_weights)

    def _check_ch_nl(self):
        if len(self.ch) != self.nl:
            raise ValueError(f'The length of self.ch {len(self.ch)} do not match self.nl')
