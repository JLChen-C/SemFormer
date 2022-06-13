import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import Mlp

from ..module import SeparateLinear
from .modules import Token2Embed, Embed2Token
from .transformer_backbone import ViTBackbone
from ..functional import cosine_similarity
from ..arch_transformer.vit import VIT_NET_CFG
from ..arch_transformer.vit import Block as ViTBlock

from ..abc_modules import BaseModule, ABC_Model


class ViTEncoder(BaseModule):

    def __init__(self, model_name, with_last_norm=True, **kwargs):
        
        super().__init__()
        self.vit_backbone = ViTBackbone(model_name, with_last_norm=with_last_norm, **kwargs)

    def forward_x_with_outer_token(self, x, outer_token, outer_posembed):
        x = self.vit_backbone.forward_features_with_outer_token(x, outer_token, outer_posembed)
        return x

    def forward_x(self, x):
        x = self.vit_backbone.forward_features(x)
        return x


class ViTDecoder(BaseModule):

    def __init__(self, width, depth, num_heads, patch_size, out_dim=3, **kwargs):

        super().__init__()
        self.width = width
        self.depth = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.out_dim = out_dim

        self.blocks = nn.ModuleList([
            ViTBlock(dim=width, num_heads=num_heads, mlp_ratio=4., qkv_bias=True, drop=0.,
                attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm, act_layer=nn.GELU)
            for i in range(depth)
        ])

        self.norm = nn.LayerNorm(width)
        self.mlp = nn.Linear(width, out_dim * (patch_size ** 2), bias=True)

    def forward_x_without_outer_token(self, x, num_outer_tokens):
        for block in self.blocks:
            x = block(x)
        x = x[:, num_outer_tokens:, :]
        x = self.norm(x)
        x = self.mlp(x)
        return x
        
    def forward_x(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = self.mlp(x)
        return x


class ViTAutoEncoder(BaseModule, ABC_Model):

    def __init__(self, decoder_width, decoder_depth, out_dim=3,
        model_name=None, with_last_norm=True, **kwargs):

        super().__init__()
        self.decoder_width = decoder_width
        self.decoder_depth = decoder_depth
        self.out_dim = out_dim
        self.model_name = model_name

        # avoid inconsistency
        version = kwargs.pop('version', 'base')
        patch_size = kwargs.pop('patch_size', 16)
        kwargs['version'] = version
        kwargs['patch_size'] = patch_size

        # for convenience
        self.patch_size = patch_size
        num_heads = VIT_NET_CFG[version]['num_heads']
        self.num_heads = num_heads

        self.encoder = ViTEncoder(model_name=model_name, with_last_norm=with_last_norm, **kwargs)

        # convert to decoder width
        self.mlp = nn.Linear(self.encoder.vit_backbone.model.embed_dim, decoder_width, bias=True)

        self.decoder = ViTDecoder(
            width=decoder_width, depth=decoder_depth,
            num_heads=num_heads, out_dim=out_dim, **kwargs)

        self.initialize([self.decoder, self.mlp])

    def forward_x(self, x):
        token = self.encoder(x)
        token = self.mlp(token)
        output = self.decoder(token)
        # (B, 3 x P x P, L)
        output = output.transpose(1, 2)
        output = F.fold(
            output,
            output_size=x.shape[-2:], kernel_size=self.patch_size,
            dilation=1, padding=0, stride=self.patch_size)
        return token, output


class ClassAwareAutoEncoder(ViTAutoEncoder):

    def __init__(self, *args, num_classes, class_dim, reduction='sum', **kwargs):

        super().__init__(*args, **kwargs)
        self.num_classes = num_classes
        self.class_dim = class_dim
        self.reduction = getattr(torch, reduction)
        self.embed_dim = self.encoder.vit_backbone.model.embed_dim

        self.convert = nn.Sequential(
            nn.Linear(self.embed_dim, num_classes * self.class_dim, bias=False),
            nn.ReLU(True)
        )

        self.re_convert = nn.Sequential(
            nn.Linear(num_classes * self.class_dim, self.embed_dim, bias=False),
            nn.LayerNorm(self.embed_dim)
        )

        self.crs = nn.Sequential(
            nn.Embedding(num_classes, self.class_dim),
            nn.ReLU(True)
        )

        self.initialize([self.convert, self.re_convert])

    def get_crs(self, x):
        index = torch.arange(self.num_classes)[None, :].repeat(x.shape[0], 1).to(x.device)
        crs = self.crs(index)
        return crs

    def get_cre(self, x, grid_embed=False):
        # (B, L, C)
        token = self.encoder(x)
        
        # (B, L, KE)
        cls_embed = self.convert(token)
        if grid_embed:
            # (B, L, K, E)
            cre = cls_embed.view(cls_embed.shape[0], cls_embed.shape[1], self.num_classes, -1)
        else:
            # (B, K, E)
            cre = self.reduction(cls_embed, dim=1)
            if isinstance(cre, (list, tuple)):
                cre = cre[0]
            cre = cre.reshape(
                x.shape[0], self.num_classes, self.class_dim)
        return cre

    def forward_x(self, x, labels):
        # (B, L, C)
        token = self.encoder(x)
        
        # (B, L, KE)
        cls_embed = self.convert(token)
        # (B, K, E)
        cre = self.reduction(cls_embed, dim=1)
        if isinstance(cre, (list, tuple)):
            cre = cre[0]
        cre = cre.reshape(
            x.shape[0], self.num_classes, self.class_dim)
        # (B, L, C)
        re_token = self.re_convert(cls_embed)

        re_token = self.mlp(re_token)

        output = self.decoder(re_token)
        # (B, 3 x P x P, L)
        output = output.transpose(1, 2)
        output = F.fold(
            output,
            output_size=x.shape[-2:], kernel_size=self.patch_size,
            dilation=1, padding=0, stride=self.patch_size)

        crs = self.get_crs(x)

        mask = labels[:, :, None] 
        crs = (crs * mask) + crs.detach() * (1 - mask)
        sim = cosine_similarity(cre, crs, is_aligned=True)

        return sim, output