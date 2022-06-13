import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair



class GlobalSumPool2d(nn.Module):

    def forward(self, x):
        return x.view(*x.shape[:-2], -1).sum(dim=-1)[..., None, None]