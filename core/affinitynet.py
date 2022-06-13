import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models
import torch.utils.model_zoo as model_zoo

from .arch_resnet import resnet, resnet38
from .arch_resnest import resnest
from .arch_vgg import vgg
from .models.transformer_backbone import ViTBackbone
from . import functional as _F

from .deeplab_utils import ASPP, Decoder
from .aff_utils import PathIndex

from tools.ai.torch_utils import resize_for_tensors

from .module import FixedBatchNorm, Interpolate
from .abc_modules import ABC_Model

from .module import *
from .functional import *
from .models import *
from .utils import *
from .networks_legacy import Backbone



def _make_fc_edge_layer(in_channels, out_channels, num_groups=4, scale_factor=None):
    layers = [
        nn.Conv2d(in_channels, out_channels, 1, bias=False),
        nn.GroupNorm(num_groups, out_channels)
    ]
    if (scale_factor is not None) and (scale_factor != 1):
        layers += [Interpolate(scale_factor=scale_factor, mode='bilinear', align_corners=True)]
    layers += [nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)


class AffinityNet(Backbone):
    def __init__(self, model_name, path_index=None):
        super().__init__(model_name, None, 'fix')

        self.model_name = model_name
        if '38' in model_name:
            self.fc_edge_features_list = [128, 256, 512, 1024, 4096]
            self.strides = [2, 4, 8, 8, 8]
        else:
            self.fc_edge_features_list = [64, 256, 512, 1024, 2048]
            self.strides = [4, 4, 8, 16, 16]

        for i in range(5):
            self.add_module(
                'fc_edge{}'.format(i + 1),
                _make_fc_edge_layer(self.fc_edge_features_list[i], 32, scale_factor=self.strides[i] / 4))
        self.fc_edge6 = nn.Conv2d(32 * 5, 1, 1, bias=True)

        if '38' in model_name:
            self.backbone = self.model
        else:
            self.backbone = nn.ModuleList([self.stage1, self.stage2, self.stage3, self.stage4, self.stage5])
        if '38' in model_name:
            self.edge_layers = nn.ModuleList([self.fc_edge2, self.fc_edge3, self.fc_edge4, self.fc_edge5, self.fc_edge6])
        else:
            self.edge_layers = nn.ModuleList([self.fc_edge1, self.fc_edge2, self.fc_edge3, self.fc_edge4, self.fc_edge5, self.fc_edge6])

        if path_index is not None:
            self.path_index = path_index
            self.n_path_lengths = len(self.path_index.path_indices)
            for i, pi in enumerate(self.path_index.path_indices):
                self.register_buffer("path_indices_" + str(i), torch.from_numpy(pi))
    
    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()

    def forward(self, x, with_affinity=False):
        if '38' in self.model_name:
            x1, x2, x3, x4, x5 = self.model(x, return_stages=True, detach_between_stages=True)

            edge1 = self.fc_edge1(x1)
            edge2 = self.fc_edge2(x2)
            edge1 = edge1[..., :edge2.size(2), :edge2.size(3)]
            edge3 = self.fc_edge3(x3)[..., :edge2.size(2), :edge2.size(3)]
            edge4 = self.fc_edge4(x4)[..., :edge2.size(2), :edge2.size(3)]
            edge5 = self.fc_edge5(x5)[..., :edge2.size(2), :edge2.size(3)]

            edge = self.fc_edge6(torch.cat([edge1, edge2, edge3, edge4, edge5], dim=1))
        else:
            x1 = self.stage1(x).detach()
            x2 = self.stage2(x1).detach()
            x3 = self.stage3(x2).detach()
            x4 = self.stage4(x3).detach()
            x5 = self.stage5(x4).detach()

            edge1 = self.fc_edge1(x1)
            edge2 = self.fc_edge2(x2)
            edge3 = self.fc_edge3(x3)[..., :edge2.size(2), :edge2.size(3)]
            edge4 = self.fc_edge4(x4)[..., :edge2.size(2), :edge2.size(3)]
            edge5 = self.fc_edge5(x5)[..., :edge2.size(2), :edge2.size(3)]

            edge = self.fc_edge6(torch.cat([edge1, edge2, edge3, edge4, edge5], dim=1))

        if with_affinity:
            return self.to_affinity(torch.sigmoid(edge))
        else:
            return edge

    def get_edge(self, x, image_size=(512, 512), stride=4):
        feat_size = (x.size(2)-1)//stride+1, (x.size(3)-1)//stride+1

        x = F.pad(x, [0, image_size[1]-x.size(3), 0, image_size[0]-x.size(2)])
        edge_out = self.forward(x)
        edge_out = edge_out[..., :feat_size[0], :feat_size[1]]
        edge_out = torch.sigmoid(edge_out[0]/2 + edge_out[1].flip(-1)/2)
        
        return edge_out

    def to_affinity(self, edge):
        aff_list = []
        edge = edge.view(edge.size(0), -1)
        
        for i in range(self.n_path_lengths):
            ind = self._buffers["path_indices_" + str(i)]
            ind_flat = ind.view(-1)
            dist = torch.index_select(edge, dim=-1, index=ind_flat)
            dist = dist.view(dist.size(0), ind.size(0), ind.size(1), ind.size(2))
            aff = torch.squeeze(1 - F.max_pool2d(dist, (dist.size(2), 1)), dim=2)
            aff_list.append(aff)
        aff_cat = torch.cat(aff_list, dim=1)
        return aff_cat