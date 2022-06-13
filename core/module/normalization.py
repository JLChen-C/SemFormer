import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from ..sync_batchnorm.batchnorm import SynchronizedBatchNorm2d



class FixedBatchNorm(nn.BatchNorm2d):
    def forward(self, x):
        return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias, training=False, eps=self.eps)

def group_norm(features):
    return nn.GroupNorm(4, features)