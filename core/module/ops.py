import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair



class Flatten(nn.Module):

    def __init__(self, start_dim=0, end_dim=-1):
        super().__init__()

        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, x):
        return torch.flatten(x, self.start_dim, self.end_dim)


class Permute(nn.Module):

    def __init__(self, *permutation):
        super().__init__()
        self.permutation = permutation

    def forward(self, inputs):
        output = inputs.permute(*self.permutation)
        return output


class Transpose(nn.Module):

    def __init__(self, *transpose):
        super().__init__()
        self.transpose = transpose

    def forward(self, inputs):
        output = inputs.transpose(*self.transpose)
        return output


class Cat(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, inputs):
        output = torch.cat(inputs, dim=self.dim)
        return output

