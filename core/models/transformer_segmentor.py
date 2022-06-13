import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .. import functional as _F
from ..module import SeparateLinear
from .transformer_backbone import ViTBackbone
from ..abc_modules import ABC_Model



class SemFormerSegmentor(nn.Module, ABC_Model):

    def __init__(self,
        model_name, num_classes, class_dim, version='base', patch_size=16,
        resolution=224, in21k=False, pos_embed_size=None):

        super().__init__()

        self.model_name = model_name
        self.backbone = ViTBackbone(
            model_name, version=version,
            patch_size=patch_size,
            resolution=resolution, in21k=in21k,
            pos_embed_size=pos_embed_size,
            with_cls_token=True, with_posembed=True)
        self.num_classes = num_classes
        self.class_dim = class_dim
        self.with_dist = 'dist' in model_name
        
        self.patch_size = self.backbone.model.patch_size

        self.cls_token = nn.Parameter(
            self.backbone.model.cls_token.data.clone().repeat(1, num_classes, 1))
        self.cls_posembed = nn.Parameter(
            self.backbone.model.pos_embed.data[:, :1, :].clone().repeat(1, num_classes, 1))

        self.cls_head = SeparateLinear(self.backbone.model.embed_dim, class_dim, groups=num_classes)

        self.seg_head = nn.Sequential(
            nn.Dropout2d(0.5),
            nn.Linear(self.backbone.model.embed_dim, self.num_classes)
        )

        self.initialize([self.cls_head, self.seg_head])

    def forward(self, x, return_cra=False):
        iH, iW = x.shape[-2:]
        x = _F.patchable_pad2d(x, self.patch_size)
        input_shape = x.shape
        H, W = [s // self.backbone.model.patch_size for s in input_shape[-2:]]
        if return_cra:
            x, attn_list = self.backbone.forward_features_with_outer_token(
                x, self.cls_token, self.cls_posembed, return_attn=True)
            # CRA: (B, H, K, N)
            cra_list = [
                attn[:, :, :self.num_classes, self.num_classes:] for attn in attn_list]
        else:
            x = self.backbone.forward_features_with_outer_token(x, self.cls_token, self.cls_posembed)

        grid_token = x[:, self.num_classes:, :]
        seg_logits = self.seg_head(grid_token)
        seg_logits = seg_logits.transpose(1, 2).view(x.shape[0], self.num_classes, H, W)

        if self.training:
            cls_token = x[:, :self.num_classes, :]
            cls_pred = self.cls_head(cls_token)
            cls_pred = F.relu(cls_pred, inplace=True)
            return x, cls_pred, seg_logits
        else:
            if return_cra:
                return x, seg_logits, cra_list
            else:
                return x, seg_logits