import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair



class CustomASPP(nn.Module):

    def __init__(self, in_channels, out_channels, dilations=[1, 3, 6, 12], act_last=True):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dilations = dilations
        self.act_last = act_last

        self.aspp_module = nn.ModuleList()
        for i in range(len(dilations)):
            aspp = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(out_channels, out_channels, 3, dilation=dilations[i], padding=dilations[i], bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True)
            )
            self.aspp_module.append(aspp)
        
        self.aspp_global = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1)
        )

        self.conv = nn.Conv2d(out_channels * (len(dilations) + 1), out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        aspp_local = [aspp_module(x) for aspp_module in self.aspp_module]
        aspp_global = self.aspp_global(x)
        aspp_global = aspp_global.expand_as(aspp_local[0])
        aspp = aspp_local + [aspp_global]
        aspp = torch.cat(aspp, dim=1)
        out = self.conv(aspp)
        out = self.bn(out)
        if self.act_last:
            out = self.relu(out)
        return out