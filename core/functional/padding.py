import torch
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
import math



# code modified from mmcv
def same_pad2d(x, kernel_size, stride=1, dilation=1, pad_mode='corner'):
    kernel_size = _pair(kernel_size)
    stride = _pair(stride)
    dilation = _pair(dilation)

    img_h, img_w = x.size()[-2:]
    kernel_h, kernel_w = kernel_size
    stride_h, stride_w = stride
    output_h = math.ceil(img_h / stride_h)
    output_w = math.ceil(img_w / stride_w)
    pad_h = (
        max((output_h - 1) * stride[0] +
            (kernel_h - 1) * dilation[0] + 1 - img_h, 0))
    pad_w = (
        max((output_w - 1) * stride[1] +
            (kernel_w - 1) * dilation[1] + 1 - img_w, 0))

    if (pad_h > 0) or (pad_w > 0):
        if pad_mode == 'corner':
            padding = [0, pad_w, 0, pad_h]
        else:
            padding = [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
        x = F.pad(x, padding)
    return x


def adaptive_pad2d(x, output_size, pad_mode='corner'):
    output_h, output_w = _pair(output_size)
    h, w = x.shape[-2:]

    patch_h = math.ceil(h / output_h)
    patch_w = math.ceil(w / output_w)

    padded_h = patch_h * output_h
    padded_w = patch_w * output_w

    pad_h = padded_h - h
    pad_w = padded_w - w

    if (pad_h > 0) or (pad_w > 0):
        if pad_mode == 'corner':
            padding = [0, pad_w, 0, pad_h]
        else:
            padding = [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
        x = F.pad(x, padding)
    return x


def patchable_pad2d(x, patch_size, pad_mode='corner'):
    patch_size = _pair(patch_size)
    h, w = x.shape[-2:]

    patch_h = math.ceil(h / patch_size[0])
    patch_w = math.ceil(w / patch_size[1])

    padded_h = patch_h * patch_size[0]
    padded_w = patch_w * patch_size[1]

    pad_h = padded_h - h
    pad_w = padded_w - w

    if (pad_h > 0) or (pad_w > 0):
        if pad_mode == 'corner':
            padding = [0, pad_w, 0, pad_h]
        else:
            padding = [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
        x = F.pad(x, padding)
    return x