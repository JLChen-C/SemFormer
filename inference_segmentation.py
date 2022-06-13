# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import os
import sys
import copy
import shutil
import random
import argparse
import numpy as np
from tqdm import tqdm

import imageio

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader

from core.puzzle_utils import *
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
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--data_dir', default='../VOC2012/', type=str)
parser.add_argument('--start', default=0.0, type=float)
parser.add_argument('--end', default=1.0, type=float)

###############################################################################
# Network
###############################################################################
parser.add_argument('--architecture', default='DeepLabv3+', type=str)
parser.add_argument('--backbone', default='resnest101', type=str)
parser.add_argument('--mode', default='fix', type=str)
parser.add_argument('--use_gn', default=True, type=str2bool)

###############################################################################
# Inference parameters
###############################################################################
parser.add_argument('--tag', default='', type=str)

parser.add_argument('--domain', default='val', type=str)

parser.add_argument('--save_type', default='png', type=str)

parser.add_argument('--scales', default='0.5,1.0,1.5,2.0', type=str)
parser.add_argument('--iteration', default=0, type=int)

if __name__ == '__main__':
    ###################################################################################
    # Arguments
    ###################################################################################
    args = parser.parse_args()

    model_dir = create_directory('./experiments/models/')
    model_path = model_dir + f'{args.tag}.pth'

    if 'train' in args.domain:
        args.tag += '@train'
    else:
        args.tag += '@' + args.domain
    
    args.tag += '@scale=%s'%args.scales
    args.tag += '@iteration=%d'%args.iteration

    pred_dir = create_directory('./experiments/predictions/{}/'.format(args.tag))
    
    set_seed(args.seed)
    log_func = lambda string='': print(string)
    
    ###################################################################################
    # Transform, Dataset, DataLoader
    ###################################################################################
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    normalize_fn = Normalize(imagenet_mean, imagenet_std)
    
    meta_dic = read_json('./data/VOC_2012.json')
    dataset = VOC_Dataset_For_Evaluation(args.data_dir, args.domain)
    
    ###################################################################################
    # Network
    ###################################################################################
    if args.architecture == 'DeepLabv3+':
        model = DeepLabv3_Plus(args.backbone, num_classes=meta_dic['classes'] + 1, mode=args.mode, use_group_norm=args.use_gn)
    elif args.architecture == 'Seg_Model':
        model = Seg_Model(args.backbone, num_classes=meta_dic['classes'] + 1)
    elif args.architecture == 'CSeg_Model':
        model = CSeg_Model(args.backbone, num_classes=meta_dic['classes'] + 1)
    
    model = model.cuda()
    model.eval()

    log_func('[i] Architecture is {}'.format(args.architecture))
    log_func('[i] Total Params: %.2fM'%(calculate_parameters(model)))
    log_func()

    load_model(model, model_path, parallel=False)
    
    #################################################################################################
    # Evaluation
    #################################################################################################
    eval_timer = Timer()
    scales = [float(scale) for scale in args.scales.split(',')]
    
    model.eval()
    eval_timer.tik()

    def inference(images, image_size):
        images = images.cuda()
        
        logits = model(images)
        logits = resize_for_tensors(logits, image_size)
        
        logits = logits[0] + logits[1].flip(-1)
        return logits

    log_func('[i] pred_dir: {}'.format(pred_dir))
    with torch.no_grad():
        dataset_len = len(dataset)
        start = int(dataset_len * args.start)
        end = int(dataset_len * args.end)
        length = end - start
        for item_id in tqdm(
            range(start, end),
            total=length,
            dynamic_ncols=True,
        ):
            item = dataset.__getitem__(item_id)
            ori_image, image_id, gt_mask = item
            ori_w, ori_h = ori_image.size

            cams_list = []

            for scale in scales:
                image = copy.deepcopy(ori_image)
                image = image.resize((round(ori_w*scale), round(ori_h*scale)), resample=PIL.Image.CUBIC)
                
                image = normalize_fn(image)
                image = image.transpose((2, 0, 1))

                image = torch.from_numpy(image)
                flipped_image = image.flip(-1)
                
                images = torch.stack([image, flipped_image])

                cams = inference(images, (ori_h, ori_w))
                cams_list.append(cams)
            
            preds = torch.stack(cams_list, dim=0) # (#n_scale, c, h, w)
            preds = torch.sum(preds, dim=0) # (c, h, w)
            preds = F.softmax(preds, dim=0) # (c, h, w)
            preds = preds.cpu() # (c, h, w)
            
            if args.iteration > 0:
                preds = preds.numpy() # (c, h, w)
                preds = crf_inference(np.asarray(ori_image), preds, t=args.iteration)

            if 'png' in args.save_type:
                if isinstance(preds, torch.Tensor):
                    preds = preds.numpy() # (c, h, w)

                pred_mask = np.argmax(preds, axis=0)
                if args.domain == 'test':
                    pred_mask = decode_from_colormap(pred_mask, dataset.colors)[..., ::-1]
                    imageio.imwrite(pred_dir + image_id + '.png', pred_mask.astype(np.uint8))
                elif 'colorful' in args.save_type:
                    pred_mask = decode_from_colormap(pred_mask, dataset.colors)[..., ::-1]
                    imageio.imwrite(pred_dir + image_id + '_decode.png', pred_mask.astype(np.uint8))
                else:
                    imageio.imwrite(pred_dir + image_id + '.png', pred_mask.astype(np.uint8))

            if 'npy' in args.save_type:
                if isinstance(preds, np.ndarray):
                    preds = torch.from_numpy(preds).cuda()

                downsample_preds = F.interpolate(preds[None, ...], size=(ori_h // 4, ori_w // 4), mode='bilinear') # (1, c, h, w)
                downsample_preds = downsample_preds[0] # (c, h, w)
                np.save(pred_dir + image_id + '.npy',
                    {"cam": downsample_preds.cpu(), "hr_cam": preds.cpu().numpy()})
        print()
    
    if args.domain == 'val':
        print("python evaluate.py --experiment_name {} --domain {} --mode {}".format(args.tag, args.domain, args.save_type))