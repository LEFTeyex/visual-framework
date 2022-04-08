r"""
Meta Model module for building all visual model.
"""

import torch
import torch.nn as nn

from torch import Tensor

__all__ = ['MetaModelDetectAnchorBased']


class MetaModelClass(nn.Module):
    def __init__(self):
        super(MetaModelClass, self).__init__()

    def forward(self, x):
        raise NotImplementedError


class MetaModelDetectAnchorBased(nn.Module):
    def __init__(self):
        super(MetaModelDetectAnchorBased, self).__init__()
        self.inc: int  # input channel (channel of image, RGB is 3 or GRAY is 1)
        self.nc: int  # number of classes
        self.anchors: Tensor  # consist of wh which is corresponding to scalings
        self.nl: int  # number of layer for outputs
        self.na: int  # number of anchor per layer
        self.image_size: int
        self.scalings: Tensor
        self.decode = False
        self.reshape = False

    def get_register_anchors(self, anchors):
        anchors = torch.tensor(anchors).float()
        nl, na = anchors.shape[0:2]
        self.register_buffer('anchors', anchors)
        return anchors, nl, na

    @torch.no_grad()
    def get_register_scalings(self, image_size):
        image = torch.zeros(1, self.inc, image_size, image_size)
        image_size = torch.tensor(image_size)
        outputs = self._forward_alone(image)
        scalings = torch.tensor([image_size / x.shape[-2] for x in outputs])
        self.register_buffer('scalings', scalings)
        self.register_buffer('image_size', image_size)
        return scalings, image_size

    def scale_anchors(self):
        r"""For anchor method but anchor free method"""
        self.anchors /= self.scalings.view(-1, 1, 1)

    def forward(self, x):
        raise NotImplementedError

    def _forward_alone(self, x):
        raise NotImplementedError

    def initialize_weights(self):
        raise NotImplementedError

    def decode_outputs(self, x, scalings=None, reshape=True):
        # x = x if isinstance(x, (list, tuple)) else [x]
        raise NotImplementedError
