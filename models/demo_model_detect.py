r"""
Demo of Detection Model.
It is built by units.py or torch.nn Module.
"""

import torch.nn as nn

from models.model_utils import init_weights
from metaclass.metamodel import MetaModelDetect


class Backbone(nn.Module):
    def __init__(self, inc: int, c: list, n: list, g=1, act='relu', bn=True, bias=False, shortcut=True):
        super(Backbone, self).__init__()

    def forward(self, x):
        pass


class Neck(nn.Module):
    def __init__(self, c: list, n: list, g=1, act='relu', bn=True, bias=False, shortcut=False):
        super(Neck, self).__init__()

    def forward(self, x):
        pass


class Head(nn.Module):
    def __init__(self, inc: list, num_anchor, num_output):
        super(Head, self).__init__()

    def forward(self, x):
        pass


class ModelDetect(MetaModelDetect):
    # TODO Upgrade for args got in train.py in the future
    r"""
    Model of Detection which is a custom model.
    Can be defined by changing Backbone and Head.
    """

    def __init__(self, inc: int, cfg: dict, num_class: int, anchors: list, num_bbox: int = 5, img_size: int = 640,
                 g=1, act='silu', bn=True, bias=False):
        super(ModelDetect, self).__init__()
        # get cfg parameters
        backbone_channels = cfg['backbone_channels']
        backbone_c3_layers = cfg['backbone_c3_layers']
        neck_channels = cfg['neck_channels']
        neck_c3_layers = cfg['neck_c3_layers']
        self.head_in_channels = [neck_channels[idx] for idx in cfgs['head_in_channels_idx_to_neck']]

        self.inc = inc
        self.no = num_class + num_bbox
        self.anchors, self.nl, self.na = self.get_register_anchors(anchors)
        self._check_nl()

        # layers
        self.backbone = Backbone(inc, backbone_channels, backbone_c3_layers,
                                 g=g, act=act, bn=bn, bias=bias, shortcut=True)
        self.neck = Neck(neck_channels, neck_c3_layers,
                         g=g, act=act, bn=bn, bias=bias, shortcut=False)
        self.head = Head(self.head_in_channels, self.na, self.no)

        self.scalings, self.image_size = self.get_register_scalings(img_size)
        self.scale_anchors()

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        return x

    def initialize_weights(self):
        self.apply(init_weights)

    def _check_nl(self):
        if len(self.head_in_channels) != self.nl:
            raise ValueError(f'The length of self.head_in_channels {len(self.head_in_channels)} do not match self.nl')


cfgs = {
    'inc': 3, 'num_class': 80,
    'head_in_channels_idx_to_neck': (),

    'model': {
        'backbone_channels': [],
        'backbone_c3_layers': [],
        'neck_channels': [],
        'neck_c3_layers': []
    }
}

if __name__ == '__main__':
    pass
