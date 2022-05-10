r"""
Units utils.
Consist of the special util only for units.
"""

import torch.nn as nn

from ..utils.check import check_odd

__all__ = ['auto_pad', 'select_act']


def auto_pad(k: int):
    r"""
    Pad the feature map to keep the size of input and output same while kernel must be odd and stride is 1.
    Only for convolution with dilation 1.
    Args:
        k: int = integral number

    Return value of padding(int)
    """
    assert check_odd(k), f'The input kernel k: {k} must be odd number in auto_pad function'
    return k // 2


def select_act(act_name_or_module):
    # TODO Upgrade for somewhere in the future
    r"""
    Select activation function.
    Args:
        act_name_or_module: = 'relu' (act name) / nn.ReLU() (instance of nn.Module) / None (nn.Identity())

    Return act(act class instance)
    """
    if isinstance(act_name_or_module, str):
        act_name_or_module = act_name_or_module.lower()
        if act_name_or_module == 'relu':
            act = nn.ReLU()
        elif act_name_or_module == 'silu':
            act = nn.SiLU()
        elif act_name_or_module == 'leakyrelu':
            act = nn.LeakyReLU(0.1)  # todo args can change
        elif act_name_or_module == 'sigmoid':
            act = nn.Sigmoid()
        elif act_name_or_module == 'softmax':
            act = nn.Softmax(dim=1)  # todo args can change
        else:
            raise ValueError(f'There is no {act_name_or_module} in select_act, please correct it or add it')

    elif isinstance(act_name_or_module, nn.Module):
        # TODO There may be wrong sometimes
        act = act_name_or_module

    elif act_name_or_module is None:
        act = nn.Identity()

    else:
        raise TypeError(f'The input act_name: {act_name_or_module} is error, please correct it')
    return act
