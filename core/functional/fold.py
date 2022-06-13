import torch
import torch.nn.functional as F



def unfold_w_center(x, kernel_size, dilation):
    assert x.dim() == 4
    assert kernel_size % 2 == 1

    # using SAME padding
    padding = (kernel_size + (dilation - 1) * (kernel_size - 1)) // 2
    unfolded_x = F.unfold(
        x, kernel_size=kernel_size,
        padding=padding,
        dilation=dilation
    )

    unfolded_x = unfolded_x.view(
        x.size(0), x.size(1), -1, x.size(2), x.size(3)
    )

    return unfolded_x


def unfold_wo_center(x, kernel_size, dilation):
    assert x.dim() == 4
    assert kernel_size % 2 == 1

    # using SAME padding
    padding = (kernel_size + (dilation - 1) * (kernel_size - 1)) // 2
    unfolded_x = F.unfold(
        x, kernel_size=kernel_size,
        padding=padding,
        dilation=dilation
    )

    unfolded_x = unfolded_x.view(
        x.size(0), x.size(1), -1, x.size(2), x.size(3)
    )

    # remove the center pixels
    size = kernel_size ** 2
    ind_list = list(range(size))
    ind_list.remove(size // 2)
    # unfolded_x = torch.cat((
    #     unfolded_x[:, :, :size // 2],
    #     unfolded_x[:, :, size // 2 + 1:]
    # ), dim=2)
    indices = torch.tensor(ind_list, dtype=torch.long, device=x.device)
    unfolded_x = unfolded_x[:, :, indices, :, :]

    return unfolded_x