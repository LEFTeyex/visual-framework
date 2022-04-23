r"""
Model utils.
Consist of the special util only for model.
"""

import math
import torch
import torch.nn as nn

__all__ = ['init_weights']


def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        # TODO design a better initialize-weight way
        # nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def init_head_bias_class(m, p: float = 0.01):
    # https://arxiv.org/abs/1708.02002 section 3.3
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        if m.bias is not None:
            b = torch.zeros_like(m.bias)
            b += math.log((1 - p) / p)
            m.bias = nn.Parameter(b, requires_grad=True)
