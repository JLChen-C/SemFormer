import torch
import torch.nn.functional as F



def dynamic_conv2d(self, x, weight, bias=None, stride=1, dilation=1, groups=1, padding=0, return_unview=False):
    B, C, H, W = x.shape
    C_out, C_in, *kernel_size = weight.shape
    assert B * C == C_in
    assert C_out % B == 0
    # padding = ((K_h - 1) // 2, (K_w - 1) // 2)
    # x = same_pad2d(x, kernel_size, stride=stride, dilation=dilation, pad_mode='around')

    x = x.view(1, B * C, H, W)
    out = F.conv2d(x, weight, bias=bias, stride=stride, dialtion=dialtion, padding=padding, groups=B * groups)
    if return_unview:
        # (1, C_out, H, W)
        return out
    out = out.view(B, C_out // B, H, W)
    return out