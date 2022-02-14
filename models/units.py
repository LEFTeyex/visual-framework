r"""
The units of model.
Consist of Conv, Linear module designed by myself.
"""

import torch.nn as nn

from utils.general import to_tuplex
from utils.units_utils import auto_pad, select_act

__all__ = ['Conv', 'DWConv', 'PointConv', 'Linear']


class Conv(nn.Module):
    r"""
    Standard Convolution Base Block.
    Convolution + BatchNorm_or_None + Activation_or_None.
    Default: no bias.
    """

    def __init__(self, inc, outc, k=3, s=1, *, p=None, g=1, act=None, bn=True, bias=False):
        super(Conv, self).__init__()
        # check and deal
        p = auto_pad(k) if (p is None) else p
        k = to_tuplex(k, 2)
        s = to_tuplex(s, 2)

        self.act = select_act(act)
        self.bn = nn.BatchNorm2d(outc) if bn else nn.Identity()
        self.conv = nn.Conv2d(inc, outc, k, s, p, groups=g, bias=bias)

    def forward(self, x):
        x = self.act(self.bn(self.conv(x)))
        return x

    def fuse_forward(self, x):
        # For fusing model which the Convolution and the BatchNorm are fused in Convolution by matrix multiplication
        x = self.act(self.conv(x))
        return x


class DWConv(nn.Module):
    r"""
    Depth-Wise Convolution Base Block.
    Depth-Wise Convolution + BatchNorm_or_None + Activation_or_None.
    Default: no bias.
    """

    def __init__(self, inc, outc, k=3, s=1, *, p=None, act=None, bn=True, bias=False):
        super(DWConv, self).__init__()
        # check and deal
        p = auto_pad(k) if (p is None) else p
        k = to_tuplex(k, 2)
        s = to_tuplex(s, 2)

        self.act = select_act(act)
        self.bn = nn.BatchNorm2d(outc) if bn else nn.Identity()
        self.conv = nn.Conv2d(inc, outc, k, s, p, groups=inc, bias=bias)

    def forward(self, x):
        x = self.act(self.bn(self.conv(x)))
        return x


class PointConv(nn.Module):
    r"""
    Point-Wise Convolution Base Block.
    Point-Wise Convolution + BatchNorm_or_None + Activation_or_None.
    Default: no bias.
    """

    def __init__(self, inc, outc, *, g=1, act=None, bn=True, bias=False):
        super(PointConv, self).__init__()
        # deal
        k = to_tuplex(1, 2)
        s = to_tuplex(1, 2)

        self.act = select_act(act)
        self.bn = nn.BatchNorm2d(outc) if bn else nn.Identity()
        self.conv = nn.Conv2d(inc, outc, k, s, groups=g, bias=bias)

    def forward(self, x):
        x = self.act(self.bn(self.conv(x)))
        return x


class Linear(nn.Module):
    r"""
    Standard Linear Base Block.
    Linear + BatchNorm_or_None + Activation_or_None.
    Default: no bias.
    """

    def __init__(self, inf, outf, outc=0, *, act=None, bn=True, bias=False):
        super(Linear, self).__init__()
        self.act = select_act(act)
        self.bn = nn.BatchNorm1d(outc) if (bn and outc) else nn.Identity()
        if bn is True and outc == 0:
            print('The BatchNorm1d do not be instantiated because the outc is 0, please reset the outc')
        self.fc = nn.Linear(inf, outf, bias=bias)

    def forward(self, x):
        x = self.act(self.bn(self.fc(x)))
        return x


if __name__ == '__main__':
    pass
