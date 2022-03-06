r"""
Model utils.
Consist of the special util only for model.
"""

import torch
import torch.nn as nn

__all__ = ['init_weights']


@torch.no_grad()
def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        # TODO design a better initialize-weight way
        # nn.init.kaiming_normal_(m.weight, mode='fan_out')
        nn.init.zeros_(m.bias)
