import torch
import torch.nn.functional as F



def check_all(x, func):
    return torch.all(func(x))

def all_in(x, min, max):
    return check_all(x, lambda x: (x >= min) & (x <= max))

def all_pos(x):
    return check_all(x, lambda x: x > 0)

def all_neg(x):
    return check_all(x, lambda x: x < 0)

def all_not_neg(x):
    return check_all(x, lambda x: x >= 0)

def all_not_pos(x):
    return check_all(x, lambda x: x <= 0)


def check_any(x, func):
    return torch.any(func(x))

def any_in(x, min, max):
    return check_any(x, lambda x: (x >= min) & (x <= max))

def any_pos(x):
    return check_any(x, lambda x: x > 0)

def any_neg(x):
    return check_any(x, lambda x: x < 0)

def any_not_neg(x):
    return check_any(x, lambda x: x >= 0)

def any_not_pos(x):
    return check_any(x, lambda x: x <= 0)


def filter_tensor(filter_func, x, reverse=False, return_zero=True):
    mask = filter_func(x)
    if reverse:
        mask = ~mask
    if (mask.sum().item() == 0) and return_zero:
        return x.new_zeros([1]).mean()
    return x[mask]

def filter_nan(x):
    return filter_tensor(torch.isnan, x, reverse=True)

def filter_inf(x):
    return filter_tensor(torch.isinf, x, reverse=True)


def replace_nan(x, value=0.):
    mask = torch.isnan(x).float()
    x = (1 - mask) * x + (mask * value)
    return x

def replace_inf(x, value=0.):
    mask = torch.isinf(x).float()
    x = (1 - mask) * x + (mask * value)
    return x

def replace_nonnum(x, value=0.):
    mask = (torch.isinf(x) & torch.isnan(x)).float()
    x = (1 - mask) * x + (mask * value)
    return x


def nansum(x, dim=None):
    if dim is None:
        return filter_tensor(torch.isnan, x, reverse=True, return_zero=True).sum()
    else:
        x = replace_nan(x)
        return x.sum(dim=dim)

def infsum(x, dim=None):
    if dim is None:
        return filter_tensor(torch.isinf, x, reverse=True, return_zero=True).sum()
    else:
        x = replace_inf(x)
        return x.sum(dim=dim)

def nanmean(x, dim=None):
    if dim is None:
        return filter_tensor(torch.isnan, x, reverse=True, return_zero=True).mean()
    else:
        x = replace_nan(x)
        return x.mean(dim=dim)

def infmean(x, dim=None):
    if dim is None:
        return filter_tensor(torch.isinf, x, reverse=True, return_zero=True).mean()
    else:
        x = replace_inf(x)
        return x.mean(dim=dim)

def safesum(x, dim=None):
    if dim is None:
        return filter_tensor(lambda x: torch.isinf(x) & torch.isnan(x),
            x, reverse=True, return_zero=True).sum()
    else:
        x = replace_nonnum(x)
        return x.sum()

def safemean(x, dim=None):
    if dim is None:
        return filter_tensor(lambda x: torch.isinf(x) & torch.isnan(x),
            x, reverse=True, return_zero=True).mean()
    else:
        x = replace_nonnum(x)
        return x.mean()
