# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import os
import sys
import copy
import shutil
import random
import argparse
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader

from core.networks import *
from core.datasets import *

from tools.general.io_utils import *
from tools.general.time_utils import *
from tools.general.json_utils import *

from tools.ai.log_utils import *
from tools.ai.demo_utils import *
from tools.ai.optim_utils import *
from tools.ai.torch_utils import *
from tools.ai.evaluate_utils import *

from tools.ai.augment_utils import *
from tools.ai.randaugment import *

parser = argparse.ArgumentParser()

devices = os.environ.get('CUDA_VISIBLE_DEVICES', '0')

###############################################################################
# Dataset
###############################################################################
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--num_workers', default=6, type=int)
parser.add_argument('--data_dir', default='../VOC2012/', type=str)

###############################################################################
# Network
###############################################################################
parser.add_argument('--architecture', default='deit', type=str)
parser.add_argument('--version', default='small', type=str)
parser.add_argument('--patch_size', default=16, type=int)
parser.add_argument('--resolution', default=224, type=int)
parser.add_argument('--in21k', default=False, type=str2bool)

parser.add_argument('--class_dim', default=256, type=int)

parser.add_argument('--reduction', default='sum', type=str)
parser.add_argument('--ae_decoder_width', default=768, type=int)
parser.add_argument('--ae_decoder_depth', default=8, type=int)
parser.add_argument('--ae_architecture', default='deit_distilled', type=str)
parser.add_argument('--ae_version', default='base', type=str)
parser.add_argument('--ae_patch_size', default=16, type=int)
parser.add_argument('--ae_resolution', default=224, type=int)
parser.add_argument('--ae_in21k', default=False, type=str2bool)


###############################################################################
# Hyperparameter
###############################################################################
parser.add_argument('--batch_size', default=8, type=int)
parser.add_argument('--max_epoch', default=20, type=int)

parser.add_argument('--lr', default=0.005, type=float)
parser.add_argument('--wd', default=5e-4, type=float)
parser.add_argument('--nesterov', default=True, type=str2bool)

parser.add_argument('--seg_sim_weight', default=1.0, type=float)
parser.add_argument('--cls_fg_weight', default=1.0, type=float)
parser.add_argument('--cls_bg_weight', default=1.0, type=float)
parser.add_argument('--act_supp_weight', default=0.075, type=float)
parser.add_argument('--act_cplt_weight', default=1.0, type=float)

parser.add_argument('--image_size', default=448, type=int)
parser.add_argument('--min_image_size', default=224, type=int)
parser.add_argument('--max_image_size', default=896, type=int)

parser.add_argument('--ae_image_size', default=224, type=int)
parser.add_argument('--print_ratio', default=0.1, type=float)

parser.add_argument('--tag', default='', type=str)
parser.add_argument('--augment', default='colorjitter,randomexpand', type=str)

parser.add_argument('--ae_tag', default='', type=str)

parser.add_argument('--stuck', action='store_true')



###############################################################################
# Categories
###############################################################################
categories = ['background', 
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 
    'bus', 'car', 'cat', 'chair', 'cow', 
    'diningtable', 'dog', 'horse', 'motorbike', 'person', 
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']


def cal_act_cplt_loss(mask, labels, grad_masks=None):
    assert (mask.dim() == 4) and (labels.dim() == 2)
    assert mask.shape[:2] == labels.shape[:2]
    B, C, H, W = mask.shape
    if grad_masks is not None:
        grad_masks_bool = grad_masks[:, 0, :, :].bool()

    act_cplt_loss = 0.
    normalizer = 0
    for img_id in range(B):
        for cls_id in labels[img_id].nonzero(as_tuple=False).view(-1):
            
            label = labels[img_id].clone()
            label[cls_id] = 0
            other_clses = label.nonzero(as_tuple=False).view(-1)
            if len(other_clses) < 1:
                continue
            if grad_masks is not None:
                normalizer += grad_masks[img_id].sum().item()
            else:
                normalizer += H * W

            max_others = mask[img_id, other_clses, :, :].max(dim=0)[0]
            cls_loss = (mask[img_id, cls_id, :, :] + max_others - 1) ** 2
            if grad_masks is not None:
                act_cplt_loss = act_cplt_loss + cls_loss[grad_masks_bool[img_id]].sum()
            else:
                act_cplt_loss = act_cplt_loss + cls_loss.sum()
    if normalizer == 0:
        return mask.new_zeros([1]).mean()
    act_cplt_loss = act_cplt_loss.sum() / normalizer
    return act_cplt_loss


def cal_act_supp_loss(masks, labels):
    # It is better to suppress the activation value of object classes
    # rather than all classes (object classes + background class).
    # It may because the activated regions of background class should be
    # more larger than that of any object class. Therefore, suppressing the
    # activation value of background class will affect the performance.
    fg_loss = masks[:, 1:, :, :].mean()
    bg_loss = 1 - masks[:, 0, :, :].mean()
    return fg_loss + bg_loss


if __name__ == '__main__':
    ###################################################################################
    # Arguments
    ###################################################################################
    args = parser.parse_args()
    
    log_dir = create_directory(f'./experiments/logs/')
    data_dir = create_directory(f'./experiments/data/')
    model_dir = create_directory('./experiments/models/')
    tensorboard_dir = create_directory(f'./experiments/tensorboards/{args.tag}/')
    
    log_path = log_dir + f'{args.tag}.txt'
    data_path = data_dir + f'{args.tag}.json'
    model_path = model_dir + f'{args.tag}.pth'
    class_model_path = model_dir + f'{args.tag}@classifier.pth'

    set_seed(args.seed)
    log_func = lambda string='': log_print(string, log_path)

    log_func('[i] {}'.format(args))
    
    log_func('[i] {}'.format(args.tag))
    log_func()
    
    ###################################################################################
    # Transform, Dataset, DataLoader
    ###################################################################################
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    normalize_fn = Normalize(imagenet_mean, imagenet_std)

    train_transforms = []

    if 'colorjitter' in args.augment:
        train_transforms.append(transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1))
    
    train_transforms += [
        RandomResize(args.min_image_size, args.max_image_size),
        RandomHorizontalFlip(),
    ]
    train_transform = transforms.Compose(train_transforms + \
        [
            Normalize(imagenet_mean, imagenet_std),
            RandomCrop(args.image_size, with_bbox=True),
            Transpose_with_BBox()
        ]
    )

    test_transform = transforms.Compose([
        Normalize_For_Segmentation(imagenet_mean, imagenet_std),
        Top_Left_Crop_For_Segmentation(args.image_size),
        Transpose_For_Segmentation()
    ])
    
    meta_dic = read_json('./data/VOC_2012.json')
    class_names = np.asarray(meta_dic['class_names'])
    
    train_dataset = VOC_Dataset_For_Classification_DetachPadding(args.data_dir, 'train_aug', train_transform)

    train_dataset_for_seg = VOC_Dataset_For_Testing_CAM(args.data_dir, 'train', test_transform)
    valid_dataset_for_seg = VOC_Dataset_For_Testing_CAM(args.data_dir, 'val', test_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, drop_last=True)
    train_loader_for_seg = DataLoader(train_dataset_for_seg, batch_size=args.batch_size, num_workers=1, drop_last=True)
    valid_loader_for_seg = DataLoader(valid_dataset_for_seg, batch_size=args.batch_size, num_workers=1, drop_last=True)

    log_func('[i] mean values is {}'.format(imagenet_mean))
    log_func('[i] std values is {}'.format(imagenet_std))
    log_func('[i] The number of class is {}'.format(meta_dic['classes']))
    log_func('[i] train_transform is {}'.format(train_transform))
    log_func('[i] test_transform is {}'.format(test_transform))
    log_func()

    val_iteration = len(train_loader)
    log_iteration = int(val_iteration * args.print_ratio)
    max_iteration = args.max_epoch * val_iteration

    # val_iteration = log_iteration

    log_func('[i] log_iteration : {:,}'.format(log_iteration))
    log_func('[i] val_iteration : {:,}'.format(val_iteration))
    log_func('[i] max_iteration : {:,}'.format(max_iteration))
    
    ###################################################################################
    # Network
    ###################################################################################

    num_classes = meta_dic['classes'] + 1

    if args.ae_image_size != args.ae_resolution:
        ae_pos_embed_size = args.ae_image_size // args.ae_patch_size
    else:
        ae_pos_embed_size = None

    ae = ClassAwareAutoEncoder(
        decoder_width=args.ae_decoder_width,
        decoder_depth=args.ae_decoder_depth,
        num_classes=num_classes,
        class_dim=args.class_dim,
        reduction=args.reduction,
        model_name=args.ae_architecture,
        version=args.ae_version,
        patch_size=args.ae_patch_size,
        resolution=args.ae_resolution,
        in21k=args.ae_in21k,
        pos_embed_size=ae_pos_embed_size
    )

    ae.load_state_dict(torch.load('./experiments/models/{}.pth'.format(args.ae_tag)), strict=True)
    
    if args.image_size != args.resolution:
        pos_embed_size = args.image_size // args.patch_size
    else:
        pos_embed_size = None
    model = SemFormer(
        class_dim=args.class_dim,
        model_name=args.architecture,
        num_classes=num_classes,
        version=args.version,
        patch_size=args.patch_size,
        resolution=args.resolution,
        in21k=args.in21k,
        pos_embed_size=pos_embed_size
    )

    ae.eval()
    param_groups = model.get_parameter_groups(print_fn=None)
    model.ae = ae
    for param in model.ae.parameters():
        param.requires_grad = False
    model = model.cuda()
    model.train()

    log_func('[i] Architecture is {}'.format(args.architecture))
    log_func('[i] Total Params: %.2fM'%(calculate_parameters(model)))
    log_func()

    try:
        use_gpu = os.environ['CUDA_VISIBLE_DEVICES']
    except KeyError:
        use_gpu = '0'

    the_number_of_gpu = len(use_gpu.split(','))
    if the_number_of_gpu > 1:
        log_func('[i] the number of gpu : {}'.format(the_number_of_gpu))
        model = nn.DataParallel(model)

    load_model_fn = lambda: load_model(model, model_path, ignore_modules=['ae'], parallel=the_number_of_gpu > 1)
    save_model_fn = lambda: save_model(model, model_path, ignore_modules=['ae'], parallel=the_number_of_gpu > 1)
    
    ###################################################################################
    # Loss, Optimizer
    ###################################################################################

    log_func('[i] The number of pretrained weights : {}'.format(len(param_groups[0])))
    log_func('[i] The number of pretrained bias : {}'.format(len(param_groups[1])))
    log_func('[i] The number of scratched weights : {}'.format(len(param_groups[2])))
    log_func('[i] The number of scratched bias : {}'.format(len(param_groups[3])))
    
    optimizer = PolyOptimizer([
        {'params': param_groups[0], 'lr': args.lr, 'weight_decay': args.wd},
        {'params': param_groups[1], 'lr': 2*args.lr, 'weight_decay': 0},
        {'params': param_groups[2], 'lr': 10*args.lr, 'weight_decay': args.wd},
        {'params': param_groups[3], 'lr': 20*args.lr, 'weight_decay': 0},
    ], lr=args.lr, momentum=0.9, weight_decay=args.wd, max_step=max_iteration, nesterov=args.nesterov)

    #################################################################################################
    # Train
    #################################################################################################
    data_dic = {
        'train' : [],
        'validation' : []
    }

    train_timer = Timer()
    eval_timer = Timer()


    loss_names = [
        'loss',

        'seg_sim_loss',
        'cls_fg_loss',
        'cls_bg_loss',
        'act_supp_loss',
        'act_cplt_loss'
    ]

    train_meter = Average_Meter(loss_names)
    
    best_train_mIoU = -1
    thresholds = list(np.arange(0.01, 1.00, 0.01))

    def evaluate(loader):
        model.eval()
        eval_timer.tik()

        meter_dic = {th : Calculator_For_mIoU_CUDA('./data/VOC_2012.json') for th in thresholds}

        with torch.no_grad():
            length = len(loader)
            for step, (images, labels, gt_masks) in enumerate(loader):
                images = images.cuda()
                labels = labels.cuda()
                
                _, features = model(images)
                features = features[..., :images.shape[-2], :images.shape[-1]]
                features = features[:, 1:, :, :]

                mask = labels.unsqueeze(2).unsqueeze(3)
                cams = (make_cam(features) * mask)

                # for visualization
                if step < 3:
                    obj_cams = cams.max(dim=1)[0]
                    
                    for b in range(images.shape[0]):
                        image = get_numpy_from_tensor(images[b])
                        cam = get_numpy_from_tensor(obj_cams[b])

                        all_cam = get_numpy_from_tensor(cams[b])
                        label = get_numpy_from_tensor(labels[b])

                        image = denormalize(image, imagenet_mean, imagenet_std)[..., ::-1]
                        h, w, c = image.shape

                        cam = (cam * 255).astype(np.uint8)
                        if cam.shape[-2] != h or cam.shape[-1] != w:
                            cam = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
                        cam = colormap(cam)

                        image_obj = cv2.addWeighted(image, 0.5, cam, 0.5, 0)[..., ::-1]
                        image_obj = image_obj.astype(np.float32) / 255.

                        writer.add_image('CAM-obj/{}-{}'.format(step, b + 1), image_obj, iteration, dataformats='HWC')

                        # for each class
                        for cls_idx in range(label.shape[0]):
                            if label[cls_idx] > 0:
                                cam_cls = all_cam[cls_idx]

                                cam_cls = (cam_cls * 255).astype(np.uint8)
                                if cam_cls.shape[-2] != h or cam_cls.shape[-1] != w:
                                    cam_cls = cv2.resize(cam_cls, (w, h), interpolation=cv2.INTER_LINEAR)
                                cam_cls = colormap(cam_cls)

                                image_cls = cv2.addWeighted(image, 0.5, cam_cls, 0.5, 0)[..., ::-1]
                                image_cls = image_cls.astype(np.float32) / 255.

                                writer.add_image('CAM-{}-{}/{}-{}'.format(cls_idx + 1, categories[cls_idx + 1], step, b + 1), image_cls, iteration, dataformats='HWC')

                gt_masks = gt_masks.cuda()
                if gt_masks.shape[-2:] != cams.shape[-2:]:
                    gt_masks = F.interpolate(gt_masks[:, None].float(), size=cams.shape[-2:], mode='nearest').long()[:, 0]
                
                for th in thresholds:
                    cam = F.pad(cams, (0, 0, 0, 0, 1, 0), mode='constant', value=th)
                    pred_mask = torch.argmax(cam, dim=1)

                    meter_dic[th].add(pred_mask, gt_masks)

                sys.stdout.write('\r# {} Evaluation [{}/{}] = {:.2f}%'.format(args.tag, step + 1, length, (step + 1) / length * 100))
                sys.stdout.flush()
        
        print(' ')
        model.train()
        
        best_th = 0.0
        best_mIoU = 0.0
        IoU_dic = {}
        FP = 0.0
        FN = 0.0

        for th in thresholds:
            mIoU, mIoU_foreground, IoU_dic_, FP_, FN_ = meter_dic[th].get(detail=True, clear=True)
            if best_mIoU < mIoU:
                best_th = th
                best_mIoU = mIoU
                IoU_dic = IoU_dic_
                FP = FP_
                FN = FN_

        return best_th, best_mIoU, IoU_dic, FP, FN
    
    writer = SummaryWriter(tensorboard_dir)
    train_iterator = Iterator(train_loader)

    for iteration in range(max_iteration):
        images, labels, crop_regions = train_iterator.get()
        images, labels, crop_regions = images.cuda(), labels.cuda(), crop_regions.cuda()
        grad_masks = create_mask(
            (crop_regions * (args.ae_image_size / args.image_size)).floor().long(),
            (args.ae_image_size, args.ae_image_size))[:, None, :, :]

        #################################################################################################

        if args.ae_image_size != args.image_size:
            ae_images = F.interpolate(images, size=(args.ae_image_size, args.ae_image_size), mode='bilinear',
                align_corners=True)
        else:
            ae_images = images
        
        bg_labels = labels.new_ones([labels.shape[0], 1])
        labels = torch.cat([bg_labels, labels], dim=1)

        results = model(images, ae_images, labels, grad_masks, stage='forward_train')
        masks, seg_sim, cls_fg_pull_sim, cls_fg_push_sim, cls_bg_pull_sim, cls_bg_push_sim = results

        seg_sim_loss = F.binary_cross_entropy(seg_sim, labels)

        act_supp_loss = cal_act_supp_loss(masks, labels)

        act_cplt_loss = cal_act_cplt_loss(masks, labels, grad_masks)

        cls_fg_pull_loss = 1 - nanmean(cls_fg_pull_sim)
        cls_fg_push_loss = 1 + nanmean(cls_fg_push_sim)
        cls_fg_loss = (cls_fg_pull_loss + cls_fg_push_loss) / 2.

        cls_bg_pull_loss = 1 - nanmean(cls_bg_pull_sim)
        cls_bg_push_loss = 1 + nanmean(cls_bg_push_sim)
        cls_bg_loss = (cls_bg_pull_loss + cls_bg_push_loss) / 2.


        loss = args.seg_sim_weight * seg_sim_loss + \
               args.cls_fg_weight * cls_fg_loss + \
               args.cls_bg_weight * cls_bg_loss + \
               args.act_supp_weight * act_supp_loss + \
               args.act_cplt_weight * act_cplt_loss

        #################################################################################################
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_dict = {k: float(eval(k)) for k in loss_names}
        train_meter.add(loss_dict)
        
        #################################################################################################
        # For Log
        #################################################################################################
        if (iteration + 1) % log_iteration == 0:
            losses = train_meter.get(keys=loss_names, clear=True)
            learning_rate = float(get_learning_rate_from_optimizer(optimizer))
            
            t = train_timer.tok(clear=True)
            left_sec = (max_iteration - (iteration + 1)) * t / log_iteration
            left_min = int(left_sec // 60)
            left_sec = int(left_sec - (left_min * 60))

            data = {
                'iteration' : iteration + 1,
                'max_iteration': max_iteration,
                'learning_rate' : learning_rate,
                'time' : t,
                'left_min' : left_min,
                'left_sec' : left_sec
            }

            data.update({loss_names[i]: losses[i] for i in range(len(loss_names))})

            data_dic['train'].append(data)
            write_json(data_path, data_dic)

            format_string = nice_format(data)
            log_func(format_string)

            for loss_name in loss_names:
                writer.add_scalar('Train/{}'.format(loss_name), data[loss_name], iteration)
            writer.add_scalar('Train/learning_rate', learning_rate, iteration)

        #################################################################################################
        # Evaluation
        #################################################################################################
        if (iteration + 1) % val_iteration == 0:
            threshold, mIoU, IoU_dic, FP, FN = evaluate(train_loader_for_seg)
            
            if best_train_mIoU == -1 or best_train_mIoU < mIoU:
                best_train_mIoU = mIoU

                save_model_fn()
                log_func('[i] save model')

            data = {
                'iteration' : iteration + 1,
                'threshold' : threshold,
                'train_mIoU' : mIoU,
                'best_train_mIoU' : best_train_mIoU,
                'FP': FP,
                'FN': FN,
                'time' : eval_timer.tok(clear=True),
            }
            data_dic['validation'].append(data)
            write_json(data_path, data_dic)
            
            log_func('[i] {} devices: {}'.format(args.tag, devices))
            log_func('[i]  iteration={iteration:,}, threshold={threshold:.2f}, train_mIoU={train_mIoU:.2f}%, best_train_mIoU={best_train_mIoU:.2f}%, FP={FP:.2f}, FN={FN:.2f}, time={time:.0f}sec'.format(**data)
            )
            
            writer.add_scalar('Evaluation/threshold', threshold, iteration)
            writer.add_scalar('Evaluation/train_mIoU', mIoU, iteration)
            writer.add_scalar('Evaluation/best_train_mIoU', best_train_mIoU, iteration)
            writer.add_scalar('Evaluation/FP', FP, iteration)
            writer.add_scalar('Evaluation/FN', FN, iteration)
    
    write_json(data_path, data_dic)
    writer.close()

    print(args.tag)