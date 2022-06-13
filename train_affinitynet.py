# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import os
import sys
import copy
import shutil
import random
import argparse
import numpy as np

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

###############################################################################
# Dataset
###############################################################################
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--num_workers', default=8, type=int)
parser.add_argument('--data_dir', default='../VOC2012/', type=str)

###############################################################################
# Network
###############################################################################
parser.add_argument('--architecture', default='resnet50', type=str)

###############################################################################
# Hyperparameter
###############################################################################
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--max_epoch', default=3, type=int)

parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--wd', default=1e-4, type=float)
parser.add_argument('--nesterov', default=True, type=str2bool)

parser.add_argument('--image_size', default=512, type=int)
parser.add_argument('--min_image_size', default=320, type=int)
parser.add_argument('--max_image_size', default=640, type=int)

parser.add_argument('--print_ratio', default=0.1, type=float)

parser.add_argument('--tag', default='', type=str)

parser.add_argument('--augment', default='colorjitter', type=str)

parser.add_argument('--pred_dir', default='./experiments/predictions/', type=str)
parser.add_argument('--label_name', required=True, type=str)



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

    set_seed(args.seed)
    log_func = lambda string='': log_print(string, log_path)
    
    log_func('[i] {}'.format(args.tag))
    log_func()

    ###################################################################################
    # Transform, Dataset, DataLoader
    ###################################################################################
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    normalize_fn = Normalize(imagenet_mean, imagenet_std)
    
    stride = 4

    train_transform = [
        RandomResize_For_Segmentation(args.min_image_size, args.max_image_size),
        RandomHorizontalFlip_For_Segmentation(),
    ]

    if 'colorjitter' in args.augment:
        train_transform.append(ColorJitter_For_Segmentation(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1))

    train_transform = transforms.Compose(train_transform + [
        Normalize_For_Segmentation(imagenet_mean, imagenet_std),
        RandomCrop_For_Segmentation(args.image_size),

        Transpose_For_Segmentation(),
        Resize_For_Mask(args.image_size // stride),
    ])
    
    meta_dic = read_json('./data/VOC_2012.json')
    class_names = np.asarray(meta_dic['class_names'])

    path_index = PathIndex(radius=10, default_size=(args.image_size // stride, args.image_size // stride))
    train_dataset = VOC_Dataset_For_Affinity(args.data_dir, 'train_aug', path_index=path_index,
        label_dir=args.pred_dir + '{}/'.format(args.label_name), transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, drop_last=True)
    
    log_func('[i] mean values is {}'.format(imagenet_mean))
    log_func('[i] std values is {}'.format(imagenet_std))
    log_func('[i] The number of class is {}'.format(meta_dic['classes']))
    log_func('[i] train_transform is {}'.format(train_transform))
    log_func()

    val_iteration = len(train_loader)
    log_iteration = int(val_iteration * args.print_ratio)
    max_iteration = args.max_epoch * val_iteration

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
    model = AffinityNet(args.architecture, path_index)

    param_groups = list(model.edge_layers.parameters())

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
    save_model_fn_for_backup = lambda: save_model(model, model_path.replace('.pth', f'_backup.pth'), parallel=the_number_of_gpu > 1)
    
    ###################################################################################
    # Loss, Optimizer
    ###################################################################################
    optimizer = PolyOptimizer([
        {'params': param_groups, 'lr': args.lr, 'weight_decay': args.wd},
    ], lr=args.lr, momentum=0.9, weight_decay=args.wd, max_step=max_iteration, nesterov=args.nesterov)
    
    #################################################################################################
    # Train
    #################################################################################################
    data_dic = {
        'train' : [],
    }

    train_timer = Timer()
    train_meter = Average_Meter([
        'loss', 
        'bg_loss', 'fg_loss', 'neg_loss',
    ])
    
    writer = SummaryWriter(tensorboard_dir)
    train_iterator = Iterator(train_loader)

    torch.autograd.set_detect_anomaly(True)

    def cal_loss(bg_pos_label, fg_pos_label, neg_label, aff):
        pos_aff_loss = (-1) * torch.log(aff + 1e-5)
        neg_aff_loss = (-1) * torch.log(1. + 1e-5 - aff)

        bg_pos_aff_loss = torch.sum(bg_pos_label * pos_aff_loss) / (torch.sum(bg_pos_label) + 1e-5)
        fg_pos_aff_loss = torch.sum(fg_pos_label * pos_aff_loss) / (torch.sum(fg_pos_label) + 1e-5)

        pos_aff_loss = bg_pos_aff_loss / 2 + fg_pos_aff_loss / 2
        neg_aff_loss = torch.sum(neg_label * neg_aff_loss) / (torch.sum(neg_label) + 1e-5)
        return bg_pos_aff_loss, fg_pos_aff_loss, pos_aff_loss, neg_aff_loss

    for iteration in range(max_iteration):
        images, labels = train_iterator.get()

        images = images.cuda()

        bg_pos_label = labels[0].cuda(non_blocking=True)
        fg_pos_label = labels[1].cuda(non_blocking=True)
        neg_label = labels[2].cuda(non_blocking=True)
        
        #################################################################################################
        # Affinity Matrix
        #################################################################################################
        aff = model(images, with_affinity=True)

        ###############################################################################
        # The part is to calculate losses.
        ###############################################################################
        pos_aff_loss = (-1) * torch.log(aff + 1e-5)
        neg_aff_loss = (-1) * torch.log(1. + 1e-5 - aff)

        bg_pos_aff_loss = torch.sum(bg_pos_label * pos_aff_loss) / (torch.sum(bg_pos_label) + 1e-5)
        fg_pos_aff_loss = torch.sum(fg_pos_label * pos_aff_loss) / (torch.sum(fg_pos_label) + 1e-5)

        pos_aff_loss = bg_pos_aff_loss / 2 + fg_pos_aff_loss / 2
        neg_aff_loss = torch.sum(neg_label * neg_aff_loss) / (torch.sum(neg_label) + 1e-5)

        loss = (pos_aff_loss + neg_aff_loss) / 2
        #################################################################################################
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_meter.add({
            'loss' : loss.item(), 

            'bg_loss' : bg_pos_aff_loss.item(),
            'fg_loss' : fg_pos_aff_loss.item(),
            'neg_loss' : neg_aff_loss.item(),
        })
        
        #################################################################################################
        # For Log
        #################################################################################################
        if (iteration + 1) % log_iteration == 0:
            loss, bg_loss, fg_loss, neg_loss = train_meter.get(clear=True)
            learning_rate = float(get_learning_rate_from_optimizer(optimizer))
            
            t = train_timer.tok(clear=True)
            left_sec = (max_iteration - (iteration + 1)) * t / log_iteration
            left_min = int(left_sec // 60)
            left_sec = int(left_sec - (left_min * 60))

            data = {
                'iteration' : iteration + 1,
                'learning_rate' : learning_rate,
                'loss' : loss,

                'bg_loss' : bg_loss,
                'fg_loss' : fg_loss,
                'neg_loss' : neg_loss,

                'time' : t,
                'left_min' : left_min,
                'left_sec' : left_sec
            }
            data_dic['train'].append(data)
            write_json(data_path, data_dic)
            
            log_func('[i] iteration={iteration:,}, learning_rate={learning_rate:.4f}, loss={loss:.4f}, \n\
                    \r    bg_loss={bg_loss:.4f}, fg_loss={fg_loss:.4f}, neg_loss={neg_loss:.4f}, \n\
                    \r    time={time:.0f}sec, left_time={left_min:d}:{left_sec:d}'.format(**data)
            )

            writer.add_scalar('Train/loss', loss, iteration)
            writer.add_scalar('Train/bg_loss', bg_loss, iteration)
            writer.add_scalar('Train/fg_loss', fg_loss, iteration)
            writer.add_scalar('Train/neg_loss', neg_loss, iteration)
            writer.add_scalar('Train/learning_rate', learning_rate, iteration)
        
        #################################################################################################
        # Evaluation
        #################################################################################################
        if (iteration + 1) % val_iteration == 0:
            save_model_fn()
            
    save_model_fn()

    write_json(data_path, data_dic)
    writer.close()

    print(args.tag)