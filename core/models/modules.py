import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import Mlp, DropPath
from ..arch_transformer.vit import Attention as SelfAttention
from ..arch_transformer.vit import Block as ViTBlock



class ResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, groups=1):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.stride = stride
        self.groups = groups

        self.pad1 = SamePad2d(kernel_size, stride, dilation)
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size, dilation=dilation, stride=stride,
            padding=0, bias=False, groups=groups)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size, dilation=dilation, stride=1,
            padding=0, bias=False, groups=groups)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.pad2 = SamePad2d(kernel_size, stride=1, dilation=dilation)
        self.relu = nn.ReLU(True)

        if (in_channels != out_channels) or (stride != 1):
            self.identity_transform = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=stride, bias=False, groups=groups),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.identity_transform = None

    def forward(self, x):
        identity = x
        
        out = self.pad1(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.pad2(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.identity_transform is not None:
            identity = self.identity_transform(x)

        out += identity
        out = self.relu(out)

        return out


class ResBlockBottleNeck(nn.Module):

    def __init__(self, in_channels, inter_channels, out_channels, kernel_size=3, dilation=1, stride=1, groups=1):
        super().__init__()

        self.in_channels = in_channels
        self.inter_channels = inter_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.stride = stride
        self.groups = groups

        self.conv1 = nn.Conv2d(in_channels, inter_channels, 1, groups=groups)
        self.bn1 = nn.BatchNorm2d(inter_channels)

        self.pad = SamePad2d(kernel_size, stride=1, dilation=dilation)
        self.conv2 = nn.Conv2d(
            inter_channels, inter_channels, kernel_size, dilation=dilation, stride=1,
            padding=0, bias=False, groups=groups)
        self.bn2 = nn.BatchNorm2d(inter_channels)

        self.conv3 = nn.Conv2d(inter_channels, out_channels, 1, groups=groups)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.pad2 = SamePad2d(kernel_size, stride=1, dilation=dilation)
        self.relu = nn.ReLU(True)

        if (in_channels != out_channels) or (stride != 1):
            self.identity_transform = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=stride, bias=False, groups=groups),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.identity_transform = None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.pad(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.identity_transform is not None:
            identity = self.identity_transform(x)

        out += identity
        out = self.relu(out)

        return out


class PixelDecoder(nn.Module):

    def __init__(self, in_channels, out_channels, start_level=0, end_level=-1):
        super().__init__()
        assert isinstance(in_channels, (list, tuple))

        self.num_levels = len(in_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        assert start_level < self.num_levels
        self.start_level = start_level
        if end_level < 0:
            end_level += self.num_levels
        assert self.start_level < end_level < self.num_levels
        self.end_level = end_level

        self.conv_reduce = nn.ModuleList()
        for i in range(self.num_levels):
            if self.start_level <= i <= self.end_level:
                self.conv_reduce.append(nn.Sequential(
                    nn.Conv2d(in_channels[i], out_channels, 1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(True)
                ))
            else:
                self.conv_reduce.append(nn.Identity())

        self.conv_merge = nn.Sequential(
            nn.Conv2d(out_channels * (self.end_level - self.start_level + 1), out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

        self.out_feat = nn.Sequential(
            nn.Linear(out_channels, out_channels, bias=False),
            nn.LayerNorm(out_channels),
            nn.GELU()
        )

    def forward(self, x):
        assert isinstance(x, (list, tuple)) and len(x) == self.num_levels

        reduce_x = [self.conv_reduce[i](x[i]) for i in range(self.num_levels)]

        size = x[self.start_level].shape[-2:]
        mlvl_feats = [x[self.start_level]]
        for i in range(self.start_level + 1, self.end_level + 1):
            mlvl_feats.append(F.interpolate(x[i], size=size, mode='bilinear', align_corners=True))
        mlvl_feats = torch.cat(mlvl_feats, dim=1)

        merge_feat = self.conv_merge(mlvl_feats)
        merge_feat = merge_feat.view(x.shape[0], self.out_channels, -1).transpose(1, 2)
        out = self.out_feat(merge_feat)
        return out


class SemeanticDecoder(nn.Module):

    def __init__(self, in_channels, out_channels, depth):

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth

        self.conv_feat = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

        self.blocks = nn.ModuleList([
            ViTBlock(dim=out_channels, num_heads=12, mlp_ratio=4., qkv_bias=True, drop=0.,
                attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm, act_layer=nn.GELU)
            for i in range(depth)
        ])

        self.norm = nn.LayerNorm(out_channels)
        self.act = nn.GELU()

        self.out_feat = nn.Sequential(
            nn.Linear(out_channels, out_channels, bias=False),
            nn.LayerNorm(out_channels),
            nn.GELU()
        )

    def forward(self, x):
        x = self.conv_feat(x)
        x = x.view(x.shape[0], self.out_channels, -1).transpose(1, 2)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = self.act(x)
        out = self.out_feat(x)
        return out


# class PixelDecoder(nn.Module):

#     def __init__(self, in_channels, out_channels, start_level=0, end_level=-1):
#         super().__init__()
#         assert isinstance(in_channels, (list, tuple))

#         self.num_levels = len(in_channels)
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         assert start_level < self.num_levels
#         self.start_level = start_level
#         if end_level < 0:
#             end_level += self.num_levels
#         assert self.start_level < end_level < self.num_levels
#         self.end_level = end_level

#         self.conv_reduce = nn.ModuleList()
#         for i in range(self.num_levels):
#             if self.start_level <= i <= self.end_level:
#                 self.conv_reduce.append(nn.Sequential(
#                     nn.Conv2d(in_channels[i], out_channels, 1, bias=False),
#                     nn.BatchNorm2d(out_channels),
#                     nn.ReLU(True)
#                 ))
#             else:
#                 self.conv_reduce.append(nn.Identity())

#         self.conv_merge = nn.Sequential(
#             nn.Conv2d(out_channels * (self.end_level - self.start_level + 1), out_channels, 1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(True)
#         )

#         self.out_feat = nn.Sequential(
#             nn.Linear(out_channels, out_channels, bias=False),
#             nn.LayerNorm(out_channels),
#             nn.GELU()
#         )

#     def forward(self, x):
#         assert isinstance(x, (list, tuple)) and len(x) == self.num_levels

#         reduce_x = [self.conv_reduce[i](x[i]) for i in range(self.num_levels)]

#         size = x[self.start_level].shape[-2:]
#         mlvl_feats = [x[self.start_level]]
#         for i in range(self.start_level + 1, self.end_level + 1):
#             mlvl_feats.append(F.interpolate(x[i], size=size, mode='bilinear', align_corners=True))
#         mlvl_feats = torch.cat(mlvl_feats, dim=1)

#         merge_feat = self.conv_merge(mlvl_feats)
#         merge_feat = merge_feat.view(x.shape[0], self.out_channels, -1).transpose(1, 2)
#         out = self.out_feat(merge_feat)
#         return out


# class KeyGenerator(nn.Module):

#     def __init__(self, in_channels, out_channels=256, start_level=0, end_level=-1):
#         super().__init__()
#         assert isinstance(in_channels, (list, tuple))

#         self.num_levels = len(in_channels)
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         assert start_level < self.num_levels
#         self.start_level = start_level
#         if end_level < 0:
#             end_level += self.num_levels
#         assert self.start_level < end_level < self.num_levels
#         self.end_level = end_level

#         self.lateral_convs = nn.ModuleList()
#         self.conv_outs = nn.ModuleList()
#         for i in range(self.num_levels):
#             if self.start_level <= i <= self.end_level:
#                 self.lateral_convs.append(
#                     nn.Sequential(
#                         nn.Conv2d(in_channels[i], out_channels, 1, bias=False),
#                         nn.GroupNorm(num_features=out_channels, num_groups=32)
#                 )

#                 self.conv_outs.append(
#                     nn.Sequential(
#                         nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
#                         nn.GroupNorm(num_features=out_channels, num_groups=32),
#                         nn.ReLU(True))
#                 )
#             else:
#                 self.lateral_convs.append(nn.Identity())
#                 self.conv_outs.append(nn.Identity())

#         self.key_embed = nn.Conv2d(out_channels, out_channels, 1)

#     def forward(self, inputs):
#         assert isinstance(inputs, (list, tuple)) and len(inputs) == self.num_levels

#         outputs = [self.lateral_convs[i](inputs[i]) for i in range(self.num_levels)]

#         for i in range(self.end_level, self.start_level - 1, -1):
#             outputs[i - 1] += F.interpolate(
#                 outputs[i], size=outputs[i - 1].shape[-2:], mode='bilinear', align_corners=True)
#             outputs[i - 1] = self.conv_outs[i - 1](outputs[i - 1])

#         # outputs = [self.conv_outs[i](outputs[i]) for i in range(self.num_levels)]

#         key_embed = self.key_embed(outputs[self.start_level])

#         return key_embed


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim*2, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, query, key):
        B, Nq, C = query.shape
        Nk = key.shape[1]

        # shape: (B, Nq, self.num_heads, C // self.num_heads) -> (B, self.num_heads, Nq, C // self.num_heads)
        q = self.q(query).reshape(B, Nq, self.num_heads, C // self.num_heads).transpose(1, 2)
        # shape: (B, Nk, 2, self.num_heads, C // self.num_heads) -> (2, B, self.num_heads, Nk, C // self.num_heads)
        kv = self.kv(key).reshape(B, Nk, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # shape: (B, self.num_heads, Nk, C // self.num_heads)
        k, v = kv.unbind(0) # make torchscript happy (cannot use tensor as tuple)

        # shape: (B, self.num_heads, Nq, Nk)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # shape: (B, self.num_heads, Nq, C // self.num_heads) -> (B, Nq, self.num_heads, C // self.num_heads)
        #     -> (B, Nq, C)
        x = (attn @ v).transpose(1, 2).reshape(B, Nq, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class HybridAttention(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv_bias = qkv_bias
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop

        self.self_attn = SelfAttention(dim, num_heads, qkv_bias, attn_drop, proj_drop)
        self.cross_attn = CrossAttention(dim, num_heads, qkv_bias, attn_drop, proj_drop)

    def forward(self, query, key):

        self_attn = self.self_attn(query)
        cross_attn = self.cross_attn(query, key)

        attn = self_attn + cross_attn

        return attn


class HybridBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1_q = norm_layer(dim)
        self.norm1_k = norm_layer(dim)
        self.attn = HybridAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, query, key):
        query = query + self.drop_path(self.attn(self.norm1_q(query), self.norm1_k(key)))
        query = query + self.drop_path(self.mlp(self.norm2(query)))
        return x


class KeyGenerator(nn.Module):

    def __init__(self, in_channels, out_channels=256, start_level=0, end_level=-1):
        super().__init__()
        assert isinstance(in_channels, (list, tuple))

        self.num_levels = len(in_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        assert start_level < self.num_levels
        self.start_level = start_level
        if end_level < 0:
            end_level += self.num_levels
        assert self.start_level < end_level < self.num_levels
        self.end_level = end_level

        self.projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_channels[i], out_channels, bias=False),
                nn.LayerNorm(out_channels))
            for i in range(self.end_level - self.start_level + 1)
        ])

        self.blocks = nn.ModuleList([
            HybridBlock(dim=out_channels, num_heads=12, mlp_ratio=4., qkv_bias=True, drop=0.,
                attn_drop=0., drop_path=0., norm_layer=norm_layer, act_layer=nn.GELU)
            for i in range(self.end_level - self.start_level)
        ])

        # self.norm = nn.LayerNorm(out_channels)
        # self.mlp = nn.Linear(out_channels, out_channels, 1)

    def forward(self, inputs):
        assert isinstance(inputs, (list, tuple)) and len(inputs) == self.num_levels

        mlvl_feats = [
            x.flatten(-2).transpose(1, 2) for x in inputs[self.start_level:self.end_level + 1]]
        mlvl_feats = [self.projs[i](mlvl_feats[i]) for i in range(len(mlvl_feats))]
        for i in range(len(mlvl_feats) - 2, -1, -1):
            mlvl_feats[i] = self.blocks[i](mlvl_feats[i], mlvl_feats[i + 1])

        # out = self.norm(mlvl_feats[self.start_level])
        # out = self.mlp(out)

        out = mlvl_feats[self.start_level]

        return out


class QueryGenerator(nn.Module):

    def __init__(self, in_channels, out_channels, depth):

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth

        self.proj = nn.Sequential(
            nn.Linear(in_channels, out_channels, 1, bias=False),
            nn.LayerNorm(out_channels))

        self.blocks = nn.ModuleList([
            ViTBlock(dim=out_channels, num_heads=12, mlp_ratio=4., qkv_bias=True, drop=0.,
                attn_drop=0., drop_path=0., norm_layer=norm_layer, act_layer=nn.GELU)
            for i in range(depth)
        ])

        # self.norm = nn.LayerNorm(out_channels)
        # self.mlp = nn.Linear(out_channels, out_channels)

    def forward(self, x):
        x = x.flatten(-2).transpose(1, 2)
        x = self.proj(x)
        for block in self.blocks:
            x = block(x)
        # x = self.norm(x)
        # x = self.mlp(x)
        return x


# class DynamicTokenSelection(nn.Module):

#     def __init__(self, )


class Token2Embed(nn.Module):
    def __init__(self, dim, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.scale = dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim*2, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, token, embed):
        B, Nt, C = token.shape
        Ne = embed.shape[1]

        # (B, Ne, C) 
        q = self.q(embed)

        # (B, Nt, 2C) -> (B, Nt, 2, C) -> (2, B, Nt, C)
        kv = self.kv(token).reshape(B, Nt, 2, C).permute(2, 0, 1, 3)
        # (B, Nt, C)
        k, v = kn.unbind(0) # make torchscript happy (cannot use tensor as tuple)

        # (B, Ne, C) @ (B, C, Nt) -> (B, Ne, Nt)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # (B, Ne, Nt) @ (B, Nt, C) -> (B, Ne, C)
        x = attn @ v
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Embed2Token(nn.Module):
    def __init__(self, dim, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.scale = dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim*2, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, token, embed):
        B, Nt, C = token.shape
        Ne = embed.shape[1]

        # (B, Nt, C) 
        q = self.q(token)

        # (B, Ne, 2C) -> (B, Ne, 2, C) -> (2, B, Ne, C)
        kv = self.kv(embed).reshape(B, Nt, 2, C).permute(2, 0, 1, 3)
        # (B, Ne, C)
        k, v = kn.unbind(0) # make torchscript happy (cannot use tensor as tuple)

        # (B, Nt, C) @ (B, C, Ne) -> (B, Nt, Ne)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # (B, Nt, Ne) @ (B, Ne, C) -> (B, Nt, C)
        x = attn @ v
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SemanticCorrelationModule(nn.Module):

    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes

        self.proj_feat = nn.Conv2d(in_channels, in_channels // 4, 1)

    def forward(self, feat, cam):
        B, C, H, W = feat.shape
        K = cam.shape[1]

        cam_detach = cam.detach()
        # (B, C, H, W) -> (B, C, HW)
        feat_proj = self.proj_feat(feat)
        feat_proj = feat_proj.view(B, C // 4, -1).transpose(1, 2)
        cam_reshape = cam_detach.view(B, K, -1)#.transpose(1, 2)

        # class branch
        # (B, K, HW) @ (B, HW, C) -> (B, K, C)
        class_term = cam_reshape @ feat_proj

        # pixel branch
        # (B, HW, C) @ (B, C, HW) -> (B, HW, HW)
        pixel_sim = feat_proj @ feat_proj.transpose(1, 2)
        pixel_sim = pixel_sim.softmax(dim=-1)
        # (B, HW, HW) @ (B, HW, C) -> (B, HW, C)
        pixel_term = pixel_sim @ feat_proj

        # aug cam
        # (B, K, C) @ (B, C, HW) -> (B, K, HW) -> (B, K, H, W)
        aug_cam = class_term @ pixel_term.transpose(1, 2)
        aug_cam = aug_cam.view(B, self.num_classes, H, W)

        cam = cam + aug_cam

        return cam
