import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from .. import functional as FN



class SMU(nn.Module):

    def __init__(self, miu=1e6):
        super().__init__()

        self.miu = nn.Parameter(torch.tensor(miu, dtype=torch.float))

    def forward(self, x):
        return FN.smu(x, self.miu)


class SMUG(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return FN.smug(x)


class SMUL(nn.Module):

    def __init__(self, alpha=0.25, miu=4.352665993287951e-9):
        super().__init__()

        self.alpha = alpha
        self.miu = nn.Parameter(torch.tensor(miu, dtype=torch.float))

    def forward(self, x):
        return FN.smul(x, self.alpha, self.miu)

