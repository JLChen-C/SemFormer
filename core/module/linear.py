import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
import math



class SeparateLinear(nn.Module):

    def __init__(self, in_channels, out_channels, groups=1, bias=True):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups

        self.weight = nn.Parameter(torch.Tensor(groups, in_channels, out_channels))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(groups, out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            num_input_fmaps = self.weight.size(1)
            num_output_fmaps = self.weight.size(2)
            receptive_field_size = 1
            fan_in = num_input_fmaps * receptive_field_size
            fan_out = num_output_fmaps * receptive_field_size
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        n_dim = x.dim()
        assert n_dim > 2, '{} requires input with ndim > 2'.format(self.__class__.__name__)
        shape = [1] * (n_dim - 2)
        weight = self.weight.view(*shape, *self.weight.shape)
        weight = weight.expand(*x.shape[:-2], -1, -1, -1)
        out = torch.matmul(x[..., None, :], self.weight)[..., 0, :]
        if self.bias is not None:
            bias = self.bias.view(*shape, *self.bias.shape)
            bias = bias.expand(*x.shape[:-2], -1, -1)
            out = out + bias
        return out