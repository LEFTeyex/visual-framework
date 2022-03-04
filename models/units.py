r"""
The units of model.
Consist of Conv, Linear module designed by myself.
"""

import torch
import torch.nn as nn

from utils.general import to_tuplex
from utils.units_utils import auto_pad, select_act

__all__ = ['Conv', 'DWConv', 'PointConv', 'Linear', 'Bottleneck', 'C3', 'SPP', 'SPPF']


class Conv(nn.Module):
    r"""
    Standard Convolution Base Block.
    Convolution + BatchNorm_or_None + Activation_or_None.
    Default: no bias.
    """

    def __init__(self, inc, outc, k=3, s=1, *, p=None, g=1, act='relu', bn=True, bias=False):
        super(Conv, self).__init__()
        # check and deal
        p = auto_pad(k) if (p is None) else p
        k = to_tuplex(k, 2)
        s = to_tuplex(s, 2)

        self.act = select_act(act)
        self.bn = nn.BatchNorm2d(outc) if bn else nn.Identity()  # TODO need to upgrade in the future
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

    def __init__(self, inc, outc, k=3, s=1, *, p=None, act='relu', bn=True, bias=False):
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

    def __init__(self, inc, outc, *, g=1, act='relu', bn=True, bias=False):
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

    def __init__(self, inf, outf, outc=0, *, act='relu', bn=True, bias=False):
        super(Linear, self).__init__()
        self.act = select_act(act)
        self.bn = nn.BatchNorm1d(outc) if (bn and outc) else nn.Identity()
        if bn is True and outc == 0:
            print('The BatchNorm1d do not be instantiated because the outc is 0, please reset the outc')
        self.fc = nn.Linear(inf, outf, bias=bias)

    def forward(self, x):
        x = self.act(self.bn(self.fc(x)))
        return x


class Bottleneck(nn.Module):
    r"""Standard Bottleneck Block"""

    def __init__(self, inc, outc, shortcut=True, *, e=0.5, g=1, act='relu', bn=True, bias=False):
        super(Bottleneck, self).__init__()
        assert inc == outc, f'The in_channels {inc} is not equal to out_channels {outc}'
        _c = int(outc * e)
        self.conv1 = Conv(inc, _c, 1, 1, g=g, act=act, bn=bn, bias=bias)
        self.conv2 = Conv(_c, outc, 3, 1, g=g, act=act, bn=bn, bias=bias)
        self.add = shortcut

    def forward(self, x):
        x = self.conv2(self.conv1(x)) + x if self.add else self.conv2(self.conv1(x))
        return x


class C3(nn.Module):
    r"""C3 Block"""

    def __init__(self, inc, outc, n=1, shortcut=True, *, e=0.5, g=1, act='relu', bn=True, bias=False):
        super(C3, self).__init__()
        _c = int(outc * e)
        self.conv1 = Conv(inc, _c, 1, 1, g=g, act=act, bn=bn, bias=bias)
        self.conv2 = Conv(inc, _c, 1, 1, g=g, act=act, bn=bn, bias=bias)
        self.conv3 = Conv(2 * _c, outc, 1, 1, g=g, act=act, bn=bn, bias=bias)
        self.m = nn.Sequential(
            *(Bottleneck(_c, _c, shortcut, e=1.0, g=g, act=act, bn=bn, bias=bias) for _ in range(n))
        )

    def forward(self, x):
        x = self.conv3(torch.cat((self.m(self.conv1(x)), self.conv2(x)), dim=1))
        return x


class SPP(nn.Module):
    r"""Spatial Pyramid Pooling Block https://arxiv.org/abs/1406.4729"""

    def __init__(self, inc, outc, k=(5, 9, 13), *, g=1, act='relu', bn=True, bias=False):
        super().__init__()
        _c = inc // 2  # hidden channels
        self.conv1 = Conv(inc, _c, 1, 1, g=g, act=act, bn=bn, bias=bias)
        self.conv2 = Conv(_c * (len(k) + 1), outc, 1, 1, g=g, act=act, bn=bn, bias=bias)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(torch.cat([x] + [m(x) for m in self.m], dim=1))
        return x


class SPPF(nn.Module):
    r"""Spatial Pyramid Pooling - Fast Block"""

    def __init__(self, inc, outc, k=5, *, g=1, act='relu', bn=True, bias=False):
        # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        _c = inc // 2  # hidden channels
        self.conv1 = Conv(inc, _c, 1, 1, g=g, act=act, bn=bn, bias=bias)
        self.conv2 = Conv(_c * 4, outc, 1, 1, g=g, act=act, bn=bn, bias=bias)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.conv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        x = self.conv2(torch.cat([x, y1, y2, self.m(y2)], dim=1))
        return x


if __name__ == '__main__':
    pass
