r"""
Detection Model.
It is built by units.py or torch.nn Module.
"""
import torch
import torch.nn as nn

from models.units import Conv
from utils.log import LOGGER

__all__ = ['ModelDetect']


class Backbone(nn.Module):
    def __init__(self, inc: int, bias=True):
        super(Backbone, self).__init__()
        self.conv1 = Conv(inc, 3, act='relu', bias=bias)
        self.conv2 = Conv(3, 1, act='relu', bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class Head(nn.Module):
    def __init__(self, num_output: int, bias=True):
        super(Head, self).__init__()
        self.conv1 = Conv(1, 3, act='relu', bias=bias)
        self.conv2 = Conv(3, num_output, act='relu', bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x, x, x


class ModelDetect(nn.Module):
    # TODO: Upgrade for somewhere in the future
    r"""
    Model of Detection which is a custom model.
    Can be defined by changing Backbone and Head.
    """

    def __init__(self, inc: int, nc: int, anchors=None, num_others: int = 5, bias=True):
        # in_channels, number of classes
        super(ModelDetect, self).__init__()
        LOGGER.info('Initializing the model...')
        self.backbone = Backbone(inc, bias=bias)
        self.head = Head(nc + num_others, bias=bias)

        self.register_buffer('anchors', anchors)
        # TODO design anchors args
        self.anchors = torch.rand(3, 3, 2)
        self.nc = nc

        LOGGER.info('Initialize model successfully')

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x


if __name__ == '__main__':
    model = ModelDetect(3, 20, 1)
    print(isinstance(model, object))
