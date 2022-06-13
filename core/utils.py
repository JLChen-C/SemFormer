import torch
import torch.nn.functional as F


def grad_enable(model, ignore_param_names=None):
    for param_name, param in model.named_parameters():
        if ignore_param_names is not None:
            if param_name in ignore_param_names:
                continue
        param.requires_grad = True

def grad_disable(model, ignore_param_names=None):
    for param_name, param in model.named_parameters():
        if ignore_param_names is not None:
            if param_name in ignore_param_names:
                continue
        param.requires_grad = False

def no_grad_forward(model, *args, ignore_param_names=None, **kwargs):
    grad_disable(model, ignore_param_names=ignore_param_names)
    out = model(*args, **kwargs)
    grad_enable(model, ignore_param_names=ignore_param_names)
    return out

def no_grad_wrapper(old_func, ignore_param_names=None):

    def new_func(model, *args, **kwargs):
        grad_disable(model, ignore_param_names=ignore_param_names)
        out = old_func(model, *args, **kwargs)
        grad_enable(model, ignore_param_names=ignore_param_names)
        return out

    return new_func


def get_label_info(labels, return_type='dict'):
    B, C = labels.shape
    indexes_total = torch.arange(B)[:, None].repeat(1, C).to(labels.device)
    classes_total = torch.arange(C)[None, :].repeat(B, 1).to(labels.device)
    seen_mask_total = labels > 0
    unseen_mask_total = ~seen_mask_total
    seen_indexes = indexes_total[seen_mask_total]
    seen_classes = classes_total[seen_mask_total]
    unseen_indexes = indexes_total[unseen_mask_total]
    unseen_classes = classes_total[unseen_mask_total]

    num_seens = seen_mask_total.sum().item()
    range_seens = range(num_seens)
    seen_labels = labels.new_zeros([num_seens, C])
    seen_labels[range_seens, seen_classes] = 1
    re_seen_labels = labels.clone()[seen_indexes]
    re_seen_labels[range_seens, seen_classes] = 0
    
    indexes_total = indexes_total.reshape(-1)
    classes_total = classes_total.reshape(-1)
    labels_total = labels.new_zeros([B * C, C])
    labels_total[range(B * C), classes_total] = labels.reshape(-1)
    re_labels_total = labels[indexes_total].clone()
    re_labels_total[range(B * C), classes_total] = 0

    if return_type == 'list':
        return (seen_indexes, seen_classes, seen_mask_total,
                unseen_indexes, unseen_classes, unseen_mask_total,
                seen_labels, re_seen_labels,
                indexes_total, classes_total, labels_total, re_labels_total)
    elif return_type == 'dict':
        return dict(
            seen_indexes=seen_indexes, seen_classes=seen_classes, seen_mask_total=seen_mask_total,
            unseen_indexes=unseen_indexes, unseen_classes=unseen_classes, unseen_mask_total=unseen_mask_total,
            seen_labels=seen_labels, re_seen_labels=re_seen_labels,
            indexes_total=indexes_total, classes_total=classes_total,
            labels_total=labels_total, re_labels_total=re_labels_total
        )
    else:
        raise ValueError('unsupported return_type: {}'.format(return_type))

def create_mask(regions, image_size):
    num_regions = regions.shape[0]
    H, W = image_size
    target_masks = torch.zeros([num_regions, H, W], dtype=torch.float, device=regions.device)
    for i in range(num_regions):
        region = regions[i]
        target_masks[i, region[1]:region[3], region[0]:region[2]] = 1.
    return target_masks