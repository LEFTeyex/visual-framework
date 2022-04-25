r"""
YOLOv5 v6.0.
Add in 2022.03.04.
Upgrade in 2022.04.19.
"""

import math
import torch
import torch.nn as nn

from utils.log import logging_initialize
from utils.decode import parse_outputs_yolov5, filter_outputs2predictions, non_max_suppression
from models.units import Conv, C3, SPPF
from models.model_utils import init_weights
from metaclass.metamodel import MetaModelDetectAnchorBased

__all__ = ['yolov5n_v6', 'yolov5s_v6', 'yolov5m_v6', 'yolov5l_v6', 'yolov5x_v6']


class Backbone(nn.Module):
    def __init__(self, inc: int, c: list, n: list, g=1, act='relu', bn=True, bias=False, shortcut=True):
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


class Neck(nn.Module):
    def __init__(self, c: list, n: list, g=1, act='relu', bn=True, bias=False, shortcut=False):
        super(Neck, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')  # 11, 15
        self.block1 = nn.Sequential(
            C3(c[0], c[1], n=n[0], shortcut=shortcut, g=g, act=act, bn=bn, bias=bias),  # 13
            Conv(c[2], c[3], 1, 1, g=g, act=act, bn=bn, bias=bias)  # 14
        )
        self.conv1 = C3(c[4], c[5], n=n[1], shortcut=shortcut, g=g, act=act, bn=bn, bias=bias)  # 17
        self.conv2 = Conv(c[6], c[7], 3, 2, g=g, act=act, bn=bn, bias=bias)  # 18
        self.conv3 = C3(c[8], c[9], n=n[2], shortcut=shortcut, g=g, act=act, bn=bn, bias=bias)  # 20
        self.conv4 = Conv(c[10], c[11], 3, 2, g=g, act=act, bn=bn, bias=bias)  # 21
        self.conv5 = C3(c[12], c[13], n=n[3], shortcut=shortcut, g=g, act=act, bn=bn, bias=bias)  # 23

    def forward(self, x):
        out4, out6, out10 = x
        x = torch.cat((self.up(out10), out6), dim=1)  # 12
        out14 = self.block1(x)
        x = torch.cat((self.up(out14), out4), dim=1)  # 16
        out17 = self.conv1(x)
        x = torch.cat((self.conv2(out17), out14), dim=1)  # 19
        out20 = self.conv3(x)
        x = torch.cat((self.conv4(out20), out10), dim=1)  # 22
        out23 = self.conv5(x)
        return out17, out20, out23


class Head(nn.Module):
    def __init__(self, inc: list, num_anchor, num_output):
        super(Head, self).__init__()
        k, s = (1, 1), (1, 1)
        self.na = num_anchor
        self.no = num_output
        self.m = nn.ModuleList([nn.Conv2d(c, num_anchor * num_output, k, s) for c in inc])  # 24

    def forward(self, x):
        y = []  # save outputs
        for index, tensor in enumerate(x):
            tensor = self.m[index](tensor)
            bs, _, h, w = tensor.shape
            # reshape to (bs, na, h, w, no)
            tensor = tensor.view(bs, self.na, self.no, h, w).permute(0, 1, 3, 4, 2).contiguous()
            y.append(tensor)
        return y

    def __getitem__(self, item):
        return self.m[item]


class BaseYolov5V6(MetaModelDetectAnchorBased):
    r"""YOLOv5 v6.0"""

    def __init__(self, inc: int, cfg: dict, num_class: int, anchors: list, num_bbox: int = 5, img_size: int = 640,
                 g=1, act='silu', bn=True, bias=False):
        super(BaseYolov5V6, self).__init__()
        # get cfg parameters
        backbone_channels = cfg['backbone_channels']
        backbone_c3_layers = cfg['backbone_c3_layers']
        neck_channels = cfg['neck_channels']
        neck_c3_layers = cfg['neck_c3_layers']
        self.head_in_channels = [neck_channels[idx] for idx in cfgs['head_channels_idx_to_neck']]

        self.inc = inc
        self.nc = num_class
        self.no = num_class + num_bbox
        self.anchors, self.nl, self.na = self.get_register_anchors(anchors)
        self._check_nl()

        # layers
        self.backbone = Backbone(inc, backbone_channels, backbone_c3_layers,
                                 g=g, act=act, bn=bn, bias=bias, shortcut=True)
        self.neck = Neck(neck_channels, neck_c3_layers,
                         g=g, act=act, bn=bn, bias=bias, shortcut=False)
        self.head = Head(self.head_in_channels, self.na, self.no)

        self.strides, self.image_size = self.get_register_strides(img_size)
        self.scale_anchors()

        self.initialize_weights()

    def forward(self, x):
        return self.forward_alone(x)

    def forward_alone(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        return x

    def initialize_weights(self):
        # self.apply(init_weights)
        self._init_head_bias_cls()

    def _init_head_bias_cls(self):
        for m, s in zip(self.head, self.strides):  # from
            b = m.bias.view(self.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (self.nc - 0.999999))
            m.bias = nn.Parameter(b.view(-1), requires_grad=True)

    def decode(self, outputs, obj_thr, iou_thr, max_detect):
        outputs = parse_outputs_yolov5(outputs, self.anchors, self.strides)
        outputs = filter_outputs2predictions(outputs, obj_thr)
        outputs = non_max_suppression(outputs, iou_thr, max_detect)
        return outputs

    def get_register_anchors(self, anchors):
        anchors = torch.tensor(anchors).float()
        nl, na = anchors.shape[0:2]
        self.register_buffer('anchors', anchors)
        return anchors, nl, na

    @torch.no_grad()
    def get_register_strides(self, image_size):
        image = torch.zeros(1, self.inc, image_size, image_size)
        image_size = torch.tensor(image_size)
        outputs = self.forward_alone(image)
        strides = torch.tensor([image_size / x.shape[-2] for x in outputs])
        self.register_buffer('strides', strides)
        self.register_buffer('image_size', image_size)
        return strides, image_size

    def scale_anchors(self):
        r"""For anchor method but anchor free method"""
        self.anchors /= self.strides.view(-1, 1, 1)

    def _check_nl(self):
        if len(self.head_in_channels) != self.nl:
            raise ValueError(f'The length of self.head_in_channels {len(self.head_in_channels)} do not match self.nl')


cfgs = {
    'inc': 3, 'num_class': 80,
    'head_channels_idx_to_neck': (5, 9, 13),
    'anchors': [[[10, 13], [16, 30], [33, 23]],
                [[30, 61], [62, 45], [59, 119]],
                [[116, 90], [156, 198], [373, 326]]],

    'yolov5n_v6': {
        'backbone_channels': [16, 32, 32, 64, 64, 128, 128, 256, 256, 256, 128],
        'backbone_c3_layers': [1, 2, 3, 1],
        'neck_channels': [256, 128, 128, 64, 128, 64, 64, 64, 128, 128, 128, 128, 256, 256],
        'neck_c3_layers': [1, 1, 1, 1]
    },

    'yolov5s_v6': {
        'backbone_channels': [32, 64, 64, 128, 128, 256, 256, 512, 512, 512, 256],
        'backbone_c3_layers': [1, 2, 3, 1],
        'neck_channels': [512, 256, 256, 128, 256, 128, 128, 128, 256, 256, 256, 256, 512, 512],
        'neck_c3_layers': [1, 1, 1, 1]
    },

    'yolov5m_v6': {
        'backbone_channels': [48, 96, 96, 192, 192, 384, 384, 768, 768, 768, 384],
        'backbone_c3_layers': [2, 4, 6, 2],
        'neck_channels': [768, 384, 384, 192, 384, 192, 192, 192, 384, 384, 384, 384, 768, 768],
        'neck_c3_layers': [2, 2, 2, 2]
    },

    'yolov5l_v6': {
        'backbone_channels': [64, 128, 128, 256, 256, 512, 512, 1024, 1024, 1024, 512],
        'backbone_c3_layers': [3, 6, 9, 3],
        'neck_channels': [1024, 512, 512, 256, 512, 256, 256, 256, 512, 512, 512, 512, 1024, 1024],
        'neck_c3_layers': [3, 3, 3, 3]
    },

    'yolov5x_v6': {
        'backbone_channels': [80, 160, 160, 320, 320, 640, 640, 1280, 1280, 1280, 640],
        'backbone_c3_layers': [4, 8, 12, 4],
        'neck_channels': [1280, 640, 640, 320, 640, 320, 320, 320, 640, 640, 640, 640, 1280, 1280],
        'neck_c3_layers': [4, 4, 4, 4]
    }
}


def _get_inc_and_num_class(inc, num_class, anchors):
    inc = cfgs['inc'] if inc is None else inc
    num_class = cfgs['num_class'] if num_class is None else num_class
    anchors = cfgs['anchors'] if anchors is None else anchors
    return inc, num_class, anchors


@logging_initialize()
def yolov5n_v6(inc: int = None, num_class: int = None, anchors: list = None,
               img_size: int = 640, g=1, act='silu', bn=True, bias=False):
    inc, num_class, anchors = _get_inc_and_num_class(inc, num_class, anchors)
    cfg = cfgs['yolov5n_v6']
    return BaseYolov5V6(inc, cfg, num_class, anchors, img_size=img_size, g=g, act=act, bn=bn, bias=bias)


@logging_initialize()
def yolov5s_v6(inc: int = None, num_class: int = None, anchors: list = None,
               img_size: int = 640, g=1, act='silu', bn=True, bias=False):
    inc, num_class, anchors = _get_inc_and_num_class(inc, num_class, anchors)
    cfg = cfgs['yolov5s_v6']
    return BaseYolov5V6(inc, cfg, num_class, anchors, img_size=img_size, g=g, act=act, bn=bn, bias=bias)


@logging_initialize()
def yolov5m_v6(inc: int = None, num_class: int = None, anchors: list = None,
               img_size: int = 640, g=1, act='silu', bn=True, bias=False):
    inc, num_class, anchors = _get_inc_and_num_class(inc, num_class, anchors)
    cfg = cfgs['yolov5m_v6']
    return BaseYolov5V6(inc, cfg, num_class, anchors, img_size=img_size, g=g, act=act, bn=bn, bias=bias)


@logging_initialize()
def yolov5l_v6(inc: int = None, num_class: int = None, anchors: list = None,
               img_size: int = 640, g=1, act='silu', bn=True, bias=False):
    inc, num_class, anchors = _get_inc_and_num_class(inc, num_class, anchors)
    cfg = cfgs['yolov5l_v6']
    return BaseYolov5V6(inc, cfg, num_class, anchors, img_size=img_size, g=g, act=act, bn=bn, bias=bias)


@logging_initialize()
def yolov5x_v6(inc: int = None, num_class: int = None, anchors: list = None,
               img_size: int = 640, g=1, act='silu', bn=True, bias=False):
    inc, num_class, anchors = _get_inc_and_num_class(inc, num_class, anchors)
    cfg = cfgs['yolov5x_v6']
    return BaseYolov5V6(inc, cfg, num_class, anchors, img_size=img_size, g=g, act=act, bn=bn, bias=bias)


def _test():
    model_list = [yolov5n_v6(),
                  yolov5s_v6(),
                  yolov5m_v6(),
                  yolov5l_v6(),
                  yolov5x_v6()]
    for model in model_list:
        # print(model)
        print(model.anchors)


if __name__ == '__main__':
    _test()
