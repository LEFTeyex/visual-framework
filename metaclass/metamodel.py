r"""
Meta Model module for building all visual model.
"""

import torch.nn as nn

__all__ = ['MetaModelDetectAnchorBased']


class MetaModelClass(nn.Module):
    def __init__(self):
        super(MetaModelClass, self).__init__()

    def forward(self, x):
        raise NotImplementedError

    def forward_alone(self, x):
        raise NotImplementedError

    def decode(self, *args):
        raise NotImplementedError

    def initialize_weights(self):
        raise NotImplementedError


class MetaModelDetectAnchorBased(nn.Module):
    def __init__(self):
        super(MetaModelDetectAnchorBased, self).__init__()

    def forward(self, x):
        raise NotImplementedError

    def forward_alone(self, x):
        raise NotImplementedError

    def decode(self, *args):
        raise NotImplementedError

    def initialize_weights(self):
        raise NotImplementedError
