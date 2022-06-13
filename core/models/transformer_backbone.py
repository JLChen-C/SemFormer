import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from ..arch_transformer import vit
from ..abc_modules import ABC_Model


class ViTBackbone(nn.Module, ABC_Model):

    def __init__(self, model_name, with_last_norm=True,
        with_posembed=False, with_cls_token=False, img_size=224, **kwargs):
        super().__init__()

        self.model_name = model_name

        version = kwargs.pop('version', 'base')
        patch_size = kwargs.pop('patch_size', 16)
        resolution = kwargs.pop('resolution', 224)
        in21k = kwargs.pop('in21k', False)
        pos_embed_size = kwargs.pop('pos_embed_size', None)
        print('{} pos_embed_size: {}'.format(self.__class__.__name__, pos_embed_size))

        if 'vit' in model_name:
            model_fn = 'vit_{}_patch{}_{}'.format(version, patch_size, resolution)
            if in21k:
                model_fn += '_in21k'
            model_fn = getattr(vit, model_fn)
            model = model_fn(pretrained=True, img_size=img_size, **kwargs)
        elif 'deit' in model_name:
            if 'distilled' in model_name:
                model_fn = 'deit_{}_distilled_patch{}_{}'.format(version, patch_size, resolution)
            else:
                model_fn = 'deit_{}_patch{}_{}'.format(version, patch_size, resolution)
            if in21k:
                model_fn += '_in21k'
            model_fn = getattr(vit, model_fn)
            model = model_fn(pretrained=True, img_size=img_size, **kwargs)
        else:
            raise ValueError('unspported model name {} for {}'.format(model_name, self.__class__.__name__))
        
        self.num_patches = model.patch_embed.num_patches
        if pos_embed_size is not None:
            new_size = pos_embed_size
            if not isinstance(new_size, (list, tuple)):
                new_size = [new_size, new_size]
            pretrained_grid_posembed = model.pos_embed[:, model.num_tokens:, :].data.clone()
            pretrained_grid_posembed = self.__class__.resize_pos_meb(pretrained_grid_posembed, new_size)
            self.grid_posembed = nn.Parameter(pretrained_grid_posembed)
        else:
            self.grid_posembed = nn.Parameter(model.pos_embed[:, model.num_tokens:, :].data.clone())
        print('self.grid_posembed:', self.grid_posembed.shape)

        if not with_posembed:
            del model.pos_embed
        if not with_cls_token:
            del model.cls_token
        del model.pre_logits
        del model.head

        self.with_last_norm = with_last_norm
        if (not with_last_norm) and hasattr(model, 'norm'):
            del model.norm

        self.model = model
            
    @staticmethod
    def resize_pos_meb(posemb_grid, new_size=()):
        """code modified from timm"""
        posemb_grid = posemb_grid[0]
        gs_old = int(math.sqrt(len(posemb_grid)))
        if len(new_size) < 1:  # backwards compatibility
            new_size = [gs_old] * 2
        assert len(new_size) >= 2
        if (new_size[0] == gs_old) and (new_size[1] == gs_old):
            return posemb_grid[None, ...]
        posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
        posemb_grid = F.interpolate(posemb_grid, size=new_size, mode='bicubic', align_corners=False)
        posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, new_size[0] * new_size[1], -1)
        return posemb_grid

    def forward_features(self, x, return_attn=False):
        height = x.shape[-2] // self.model.patch_embed.patch_size[0]
        x = self.model.patch_embed(x)
        width = x.shape[1] // height
        pos_embed = self.__class__.resize_pos_meb(self.grid_posembed, (height, width))

        x = self.model.pos_drop(x + pos_embed)
        if return_attn:
            attn_list = []
        for block in self.model.blocks:
            block_result = block(x, return_attn=return_attn)
            if return_attn:
                x, attns = block_result
                attn_list.append(attns)
            else:
                x = block_result
        if self.with_last_norm:
            x = self.model.norm(x)
        if return_attn:
            return x, attn_list
        return x

    def forward_features_with_outer_token(self, x, outer_token, outer_posembed, return_attn=False):
        height = x.shape[-2] // self.model.patch_embed.patch_size[0]
        x = self.model.patch_embed(x)
        width = x.shape[1] // height
        pos_embed = self.__class__.resize_pos_meb(self.grid_posembed, (height, width))
        pos_embed = torch.cat([outer_posembed, pos_embed], dim=1)

        outer_token = outer_token.expand(x.shape[0], -1, -1)
        x = torch.cat([outer_token, x], dim=1)

        x = self.model.pos_drop(x + pos_embed)
        if return_attn:
            attn_list = []
        for block in self.model.blocks:
            block_result = block(x, return_attn=return_attn)
            if return_attn:
                x, attns = block_result
                attn_list.append(attns)
            else:
                x = block_result
        if self.with_last_norm:
            x = self.model.norm(x)
        if return_attn:
            return x, attn_list
        return x

    def forward(self, x):
        return self.forward_features(x)