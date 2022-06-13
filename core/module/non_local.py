import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair



class NonLocal2d(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        B, C, H, W = x.shape

        # (B, C, HW)
        k = x.view(B, C, -1)
        # (B, HW, C)
        q = k.transpose(1, 2)

        # (B, HW, C) @ (B, C, HW) -> (B, HW, HW)
        attn = (q @ k).softmax(dim=-1)
        # (B, HW, HW) @ (B, HW, c) -> (B, HW, C)
        out = attn @ q
        out = out.transpose(1, 2).view(B, C, H, W)
        out += x
        return out
