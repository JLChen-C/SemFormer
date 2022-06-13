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
import time

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

categories = [
    'background',
    'aeroplane',   'bicycle', 'bird',  'boat',      'bottle', 
    'bus',         'car',     'cat',   'chair',     'cow', 
    'diningtable', 'dog',     'horse', 'motorbike', 'person', 
    'pottedplant', 'sheep',   'sofa',  'train',     'tvmonitor']

###############################################################################
# Dataset
###############################################################################
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--num_workers', default=16, type=int)
parser.add_argument('--data_dir', default='../VOC2012/', type=str)

###############################################################################
# Network
###############################################################################
parser.add_argument('--class_dim', default=256, type=int)
parser.add_argument('--reduction', default='sum', type=str)
parser.add_argument('--decoder_width', default=768, type=int)
parser.add_argument('--decoder_depth', default=8, type=int)
parser.add_argument('--architecture', default='deit_distilled', type=str)
parser.add_argument('--version', default='base', type=str)
parser.add_argument('--patch_size', default=16, type=int)
parser.add_argument('--resolution', default=224, type=int)
parser.add_argument('--in21k', default=False, type=str2bool)

parser.add_argument('--sim_weight', default=1.0, type=float)
parser.add_argument('--recon_weight', default=1.0, type=float)


###############################################################################
# Hyperparameter
###############################################################################
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--max_epoch', default=200, type=int)

parser.add_argument('--lr', default=0.025, type=float)
parser.add_argument('--wd', default=5e-4, type=float)
parser.add_argument('--nesterov', default=True, type=str2bool)

parser.add_argument('--image_size', default=224, type=int)

parser.add_argument('--print_ratio', default=0.1, type=float)

parser.add_argument('--tag', default='', type=str)
parser.add_argument('--augment', default='colorjitter,randomexpand', type=str)



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

    if 'randomexpand' in args.augment:
        if args.image_size <= 384:
            train_transforms.append(RandomExpand(scales=(1.0, 3.0)))
        else:
            train_transforms.append(RandomExpand(scales=(1.0, 2.0)))
    
    train_transforms += [
        ResizedRandomCrop(args.image_size),
        RandomHorizontalFlip()
    ]
    
    train_transform = transforms.Compose(train_transforms + \
        [
            Normalize(imagenet_mean, imagenet_std),
            Transpose()
        ]
    )

    test_transform = transforms.Compose([
        Resize_For_Segmentation(args.image_size),
        Normalize_For_Segmentation(imagenet_mean, imagenet_std),
        Transpose_For_Segmentation()
    ])
    
    meta_dic = read_json('./data/VOC_2012.json')
    class_names = np.asarray(meta_dic['class_names'])
    
    train_dataset = VOC_Dataset_For_Classification(args.data_dir, 'train_aug', train_transform)

    train_dataset_for_seg = VOC_Dataset_For_Testing_CAM(args.data_dir, 'train', test_transform)
    valid_dataset_for_seg = VOC_Dataset_For_Testing_CAM(args.data_dir, 'val', test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
        shuffle=True, drop_last=True)
    train_loader_for_seg = DataLoader(train_dataset_for_seg, batch_size=args.batch_size, num_workers=1, drop_last=True)
    valid_loader_for_seg = DataLoader(valid_dataset_for_seg, batch_size=args.batch_size, num_workers=1, drop_last=True)

    log_func('[i] mean values is {}'.format(imagenet_mean))
    log_func('[i] std values is {}'.format(imagenet_std))
    log_func('[i] The number of class is {}'.format(meta_dic['classes']))
    log_func('[i] train_transform is {}'.format(train_transform))
    log_func('[i] test_transform is {}'.format(test_transform))
    log_func()

    len_train_loader = val_iteration = len(train_loader)
    log_iteration = int(len_train_loader * args.print_ratio)
    max_iteration = args.max_epoch * len_train_loader

    log_func('[i] log_iteration : {:,}'.format(log_iteration))
    log_func('[i] val_iteration : {:,}'.format(val_iteration))
    log_func('[i] max_iteration : {:,}'.format(max_iteration))
    
    ###################################################################################
    # Network
    ###################################################################################
    if args.image_size != args.resolution:
        pos_embed_size = args.image_size // args.patch_size
    else:
        pos_embed_size = None
    model = ClassAwareAutoEncoder(
        decoder_width=args.decoder_width,
        decoder_depth=args.decoder_depth,
        num_classes=meta_dic['classes'] + 1,
        class_dim=args.class_dim,
        reduction=args.reduction,
        model_name=args.architecture,
        version=args.version,
        patch_size=args.patch_size,
        resolution=args.resolution,
        in21k=args.in21k,
        pos_embed_size=pos_embed_size
    )

    param_groups = model.get_parameter_groups(print_fn=None)
    
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
    
    load_model_fn = lambda: load_model(model, model_path, parallel=the_number_of_gpu > 1)
    save_model_fn = lambda: save_model(model, model_path, parallel=the_number_of_gpu > 1)
    
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

    total_timer = Timer()
    train_timer = Timer()
    eval_timer = Timer()

    loss_names = [
        'loss', 'sim_loss', 'recon_loss'
    ]

    train_meter = Average_Meter(loss_names)
    
    best_loss = 1e8
    best_train_mAP = -1
    thresholds = list(np.arange(0.01, 1.00, 0.01))

    def evaluate(loader):
        model.eval()
        eval_timer.tik()

        meter_dic = {th : Calculator_For_mAP('./data/VOC_2012.json') for th in thresholds}

        with torch.no_grad():
            length = len(loader)
            for step, (images, labels, gt_masks) in enumerate(loader):
                images = images.cuda()
                labels = labels.cuda()

                bg_labels = labels.new_ones([labels.shape[0], 1])
                labels = torch.cat([bg_labels, labels], dim=1)

                label_info = get_label_info(labels, return_type='dict')
                classes_total = label_info['classes_total']

                cre = model(images, stage='get_cre')
                crs = model(images, stage='get_crs')
                
                sim = cosine_similarity(cre, crs, is_aligned=True)

                for th in thresholds:
                    sim_th = sim.clone()
                    sim_th[sim_th > th] = 1.0
                    sim_th[sim_th <= th] = 0.0

                    meter_dic[th].add(sim_th, labels)

                sys.stdout.write('\r# {} Evaluation [{}/{}] = {:.2f}%'.format(args.tag, step + 1, length, (step + 1) / length * 100))
                sys.stdout.flush()
        
        print(' ')
        model.train()

        best_th = 0.0
        best_mAP = 0.0

        for th in thresholds:
            mAP = meter_dic[th].get(clear=True)
            if best_mAP < mAP:
                best_th = th
                best_mAP = mAP

        return best_th, best_mAP
    
    writer = SummaryWriter(tensorboard_dir)
    train_iterator = Iterator(train_loader)                                                          

    for iteration in range(max_iteration):
        images, labels = train_iterator.get()
        images = images.cuda()
        labels = labels.cuda()

        bg_labels = labels.new_ones([labels.shape[0], 1])
        labels = torch.cat([bg_labels, labels], dim=1)

        sim, output = model(images, labels)

        sim_loss = F.binary_cross_entropy(sim, labels)
        recon_loss = F.mse_loss(output, images)

        loss = args.sim_weight * sim_loss + \
               args.recon_weight * recon_loss

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
            
            iter_time = train_timer.tok(clear=True)
            iter_time_str = get_str_time(iter_time)

            elapsed_time = total_timer.tok(clear=False)
            elapsed_time_str = get_str_time(elapsed_time)

            left_time = (max_iteration - (iteration + 1)) * iter_time / log_iteration
            left_time_str = get_str_time(left_time)

            data = {
                'iteration': iteration + 1,
                'epoch': (iteration // len_train_loader) + 1,
                'max_iteration': max_iteration,
                'learning_rate': learning_rate,
                'iteration_time' : iter_time_str,
                'elapsed_time': elapsed_time_str,
                'left_time' : left_time_str,
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
            threshold, train_mAP = evaluate(train_loader_for_seg)
            
            epoch = iteration // len_train_loader
            
            if (best_train_mAP == -1) or (best_train_mAP < train_mAP):
                best_train_mAP = train_mAP
                best_epoch = epoch

                save_model_fn()
                log_func('[i] save model')

            data = {
                'iteration' : iteration + 1,
                'best_epoch': best_epoch + 1,
                'epoch': epoch + 1,
                'threshold' : threshold,
                'train_mAP': train_mAP,
                'best_train_mAP' : best_train_mAP,
                'eval_time' : get_str_time(eval_timer.tok(clear=True)),
            }
            data_dic['validation'].append(data)
            write_json(data_path, data_dic)
            
            log_func('[i] {} devices: {}'.format(args.tag, devices))
            format_string = nice_format(data)
            log_func(format_string)
    
    write_json(data_path, data_dic)
    writer.close()

    print(args.tag)