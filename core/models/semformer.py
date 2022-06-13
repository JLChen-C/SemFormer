import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from .transformer_segmentor import SemFormerSegmentor
from ..abc_modules import BaseModule, ABC_Model
from ..functional import cosine_similarity
from ..utils import get_label_info


class SemFormer(BaseModule, ABC_Model):

    def __init__(self, model_name, num_classes, class_dim,
        version='base', patch_size=16, resolution=224, in21k=False, in22k=False,
        pos_embed_size=None):
        super().__init__()
        self.class_dim = class_dim

        self.segmentor = SemFormerSegmentor(
            model_name=model_name, num_classes=num_classes, class_dim=class_dim,
            version=version, patch_size=patch_size,
            resolution=resolution, in21k=in21k, pos_embed_size=pos_embed_size)

    def train(self, mode=True):
        """Set the same train mode for ae and self."""
        super().train(mode)
        if hasattr(self, 'ae'):
            self.ae.train(False)

    def forward_x(self, x, **kwargs):
        return self.segmentor(x, **kwargs)

    def forward_train(self, images, ae_images, labels, grad_masks):

        label_info = get_label_info(labels)
        seen_masks = label_info['seen_mask_total']
        seen_labels = label_info['seen_labels']
        re_seen_labels = label_info['re_seen_labels']
        seen_indexes = label_info['seen_indexes']
        seen_classes = label_info['seen_classes']

        mask_feat, seg_cre, ori_mask_logits = self.segmentor(images)
        ori_mask_logits = ori_mask_logits[..., :images.shape[-2], :images.shape[-1]]

        if ori_mask_logits.shape[-2:] != ae_images.shape[-2:]:
            mask_logits = F.interpolate(ori_mask_logits, size=ae_images.shape[-2:], mode='bilinear', align_corners=True)
        else:
            mask_logits = ori_mask_logits
        masks = mask_logits.sigmoid()

        if grad_masks is not None:
            if grad_masks.dim() == 3:
                grad_masks = grad_masks[:, None, :, :]
            masks = (grad_masks * masks) + (1 - grad_masks) * 0.

        images_for_spec_cls = ae_images[seen_indexes, :, :, :]
        cls_fg_masks = masks[seen_masks, :, :][:, None, :, :]
        cls_fg_images = images_for_spec_cls * cls_fg_masks
        cls_bg_masks = 1 - cls_fg_masks
        cls_bg_images = images_for_spec_cls * cls_bg_masks

        crs = self.ae(ae_images, stage='get_crs')
        crs_repeat = self.ae(cls_fg_images, stage='get_crs')

        cs_embed = self.ae(cls_fg_images, stage='get_cre')
        re_cs_embed = self.ae(cls_bg_images, stage='get_cre')

        seg_sim = cosine_similarity(seg_cre, crs, is_aligned=True)

        pull_mask = seen_labels > 0
        push_mask = re_seen_labels > 0
        re_pull_mask = push_mask
        re_push_mask = pull_mask

        cls_fg_pull_sim = cosine_similarity(
            cs_embed[pull_mask, :], crs_repeat[pull_mask, :], is_aligned=True)
        cls_fg_push_sim = cosine_similarity(
            cs_embed[push_mask, :], crs_repeat[push_mask, :], is_aligned=True)

        cls_bg_pull_sim = cosine_similarity(
            re_cs_embed[re_pull_mask, :], crs_repeat[re_pull_mask, :], is_aligned=True)
        cls_bg_push_sim = cosine_similarity(
            re_cs_embed[re_push_mask, :], crs_repeat[re_push_mask, :], is_aligned=True)

        return masks, seg_sim, cls_fg_pull_sim, cls_fg_push_sim, cls_bg_pull_sim, cls_bg_push_sim

