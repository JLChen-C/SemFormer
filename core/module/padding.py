import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from .. import functional as FN



class SamePad2d(nn.Module):

    def __init__(self, kernel_size, stride=1, dilation=1, pad_mode='around'):
        super().__init__()

        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.dilation = _pair(dilation)
        self.pad_mode = pad_mode

    def forward(self, x):
        return FN.same_pad2d(x, self.kernel_size, self.stride, self.dilation, self.pad_mode)


class AdaptivePad2d(nn.Module):

    def __init__(self, output_size, pad_mode='corner'):
        super().__init__()

        self.output_size = _pair(output_size)
        self.pad_mode = pad_mode

    def forward(self, x):
        return FN.adaptive_pad2d(x, self.output_size, self.pad_mode)