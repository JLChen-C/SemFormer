import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair



class MultiDilatedConv2d(nn.Conv2d):

    def __init__(self, *args, dilations=[1], **kwargs):
        super().__init__(*args, **kwargs)

        self.dilations = dilations
        self.num_branch = len(dilations)

    def forward(self, x):
        outputs = []
        for dilation in self.dilations:
            padding = _pair(dilation)
            if self.padding_mode != 'zeros':
                out = F.conv2d(F.pad(x, padding, mode=self.padding_mode),
                                self.weight, self.bias, self.stride,
                                _pair(0), dilation, self.groups)
            else:
                out = F.conv2d(x, self.weight, self.bias, self.stride,
                            padding, dilation, self.groups)
            outputs.append(out)
        return outputs


# SwitchableMultiDilatedConv2d
class SMDConv2d(MultiDilatedConv2d):

    def __init__(self, *args, tau=1.0, pixel_wise=False, ratio=1. / 16., gumbel=False, **kwargs):
        super().__init__(*args, **kwargs)

        self.tau = tau
        self.pixel_wise = pixel_wise
        self.ratio = ratio
        self.gumbel = gumbel
        if pixel_wise:
            self.conv_gate = nn.Conv2d(self.in_channels, len(self.dilations), 5, padding=2)
        else:
            self.inter_channels = int(self.in_channels * self.ratio)
            self.conv_gate = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(self.in_channels, self.inter_channels, 1),
                nn.LayerNorm([self.inter_channels, 1, 1]),
                nn.ReLU(True),
                nn.Conv2d(self.inter_channels, len(self.dilations), 1)
            )

        nn.init.constant_(self.conv_gate[-1].weight, val=0)
        nn.init.constant_(self.conv_gate[-1].bias, val=0)

    def forward(self, x):
        gates = self.conv_gate(x)
        if self.gumbel:
            gates = F.gumbel_softmax(gates, tau=self.tau, hard=True, dim=1)
        else:
            if self.tau != 1:
                gates = gates / self.tau
            gates = F.softmax(gates, dim=1)
        gates = torch.split(gates, dim=1, split_size_or_sections=1)
        conv_outs = super().forward(x)
        gated_outs = [gates[i] * conv_outs[i] for i in range(self.num_branch)]
        return sum(gated_outs)

