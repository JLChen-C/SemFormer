import torch
import torch.nn.functional as F
import math

from .utils import nanmean, nansum


def scale_thresed_sigmoid(x, scale=1.0, thres=0.0):
    return (scale * (x - thres)).sigmoid()

def scale_2sigmoid(x, scale=1.0):
    return 2. * (scale * x).sigmoid() - 1.

def fast_softmax(x, dim, eps=1e-12):
    x = F.relu(x, inplace=True)
    x = x / x.sum(dim=dim).clamp(min=eps)
    return x
    

def info_entropy(x, dim=-1, eps=1e-12):
    x = x.clamp(min=eps)
    ie = -(x * x.log()).sum(dim=dim)
    return ie.mean()

def kl_divergence(x, y, dim=-1, eps=1e-12):
    assert x.dim() == y.dim(), 'input dim must be the same'
    x = x.clamp(min=eps)
    y = y.clamp(min=eps)
    kl_div = (x * (x.log() - y.log())).sum(dim=dim)
    return kl_div.mean()

def js_divergence(x, y, dim=-1, eps=1e-12):
    assert x.dim() == y.dim(), 'input dim must be the same'
    def _kl_divergence(x, y):
        return (x * (x.log() - y.log())).sum(dim=dim)

    x = x.clamp(min=eps)
    y = y.clamp(min=eps)
    mean_xy = (x + y) / 2.

    kl_x_mean = _kl_divergence(x, mean_xy)
    kl_y_mean = _kl_divergence(y, mean_xy)
    js_div = (kl_x_mean + kl_y_mean) / 2.
    return js_div.mean()


def dot_product(x, y, is_aligned=False):
    assert x.dim() == y.dim(), 'input dimension must be the same'
    if is_aligned:
        out = torch.matmul(x[..., None, :], y[..., :, None])[..., 0, 0]
    else:
        out = torch.matmul(x, y.transpose(-2, -1))
    return out

def jsd_mutual_information(joint_samples, marginal_samples, T=dot_product,
    joint_aligned=True, marginal_aligned=False,
    joint_mask=None, marginal_mask=None):
    """implementation of JSD-based MI estimator:
    I(X, Y) >= I_{JSD}(X, Y) = E_{p(x, y)}(-softplus(-T(x, y))) - E_{p(x)p(y)}(softplus(T(x, y)))
    Args:
        joint_samples: (x1, y1)
        x1: (N, C)
        y1: (N, C)

        marginal_samples: (x2, y2)
        x2: (N1, C)
        y2: (N2, C)
    """
    assert isinstance(joint_samples, (list, tuple)) and len(joint_samples) == 2
    assert isinstance(marginal_samples, (list, tuple)) and len(marginal_samples) == 2

    x1, y1 = joint_samples
    x2, y2 = marginal_samples

    pxy = -F.softplus(-T(x1, y1, is_aligned=joint_aligned))
    if joint_mask is not None:
        pxy = pxy[joint_mask]
    px_py = F.softplus(T(x2, y2, is_aligned=marginal_aligned))
    if marginal_mask is not None:
        px_py = px_py[marginal_mask]
    # jsd_mi = pxy.mean() - px_py.mean()
    jsd_mi = nanmean(pxy) - nanmean(px_py)
    return jsd_mi


def fast_cosine_similarity(query, key, eps=1e-8):
    input_dim = query.dim()
    assert query.dim() == key.dim(), 'input dims must be the same'

    if input_dim == 4:
        B, C, Hx, Wx = query.shape
        query = query.view(B, C, -1)
        query = F.normalize(query, p=2, dim=1, eps=eps).transpose(1, 2)
        key = key.view(B, C, -1)
        key = F.normalize(key, p=2, dim=1, eps=eps)
        similarity = torch.matmul(query, key)
        return similarity
    elif input_dim == 2:
        N, C = query.shape
        query = F.normalize(query, p=2, dim=1, eps=eps)
        key = F.normalize(key, p=2, dim=1, eps=eps).transpose(0, 1)
        similarity = torch.matmul(query, key)
        return similarity
    else:
        raise ValueError('only 2D or 4D inputs is supported')

def cosine_similarity(x, y, dim=-1, is_aligned=False, normalize=True, eps=1e-8):
    # assert x.shape == y.shape, 'input must be with the same shape'
    if not is_aligned:
        assert x.shape[:-2] == y.shape[:-2], 'input must be with the same shape when not aligned'
    if normalize:
        x = F.normalize(x, p=2, dim=dim, eps=eps)
        y = F.normalize(y, p=2, dim=dim, eps=eps)
    if is_aligned: # cos_sim: (*, N)
        cos_sim = (x * y).sum(dim=dim)
    # dim == -2:
    # x: (*, E, N)
    # y: (*, E, N)
    # or dim == -1:
    # x: (*, N, E)
    # y: (*, N, E)
    else: # cos_sim: (*, N, N)
        candidate_dims = [-2, -1, x.dim() - 2, x.dim() - 1]
        assert dim in candidate_dims, 'when is_aligned=False, dim must be one of {}, but got {}'.format(candidate_dims, dim)
        if dim in candidate_dims[0::2]:
            x_t = x.transpose(dim, dim + 1) # (*, N, E)
            cos_sim = torch.matmul(x_t, y) # (*, N, N)
        else:
            y_t = y.transpose(dim - 1, dim) # (*, E, N)
            cos_sim = torch.matmul(x, y_t) # (*, N, N)
    cos_sim = cos_sim.clamp(min=-1. + 1e-4, max=1. - 1e-4)
    return cos_sim


"""
Implementation of `SMU: Smooth Activation Function for Deep Networks using Smoothing Maximum Technique`
"""

def general_smu(x1, x2, miu):
    out = ((x1 + x2) + (x1 - x2) * torch.erf(miu * (x1 - x2))) / 2.
    return out

"""
`smu approximate ReLU`
"""
def smu(x, miu):
    out = (x + x * torch.erf(miu * x)) / 2.
    return out

"""
`smug approximate GeLU`
"""
def smug(x):
    out = (x + x * torch.erf(x / math.sqrt(2))) / 2.
    return out

"""
`smul approximate Leaky ReLU`
"""
def smul(x, alpha, miu):
    out = ((1 + alpha) * x + (1 - alpha) * x * torch.erf(miu * (1 - alpha) * x)) / 2.
    return out

