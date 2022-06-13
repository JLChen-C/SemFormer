import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair



class Interpolate(nn.Module):

    def __init__(self, size=None, scale_factor=None, mode='bilinear', align_corners=True):
        super().__init__()
        
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, input):
        return F.interpolate(input, self.size, self.scale_factor, self.mode, self.align_corners, recompute_scale_factor=True)