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
parser.add_argument('--architecture', default='resnet50', type=str)

###############################################################################
# Inference parameters
###############################################################################
parser.add_argument('--model_name', default='', type=str)

parser.add_argument('--cam_dir', default='', type=str)
parser.add_argument('--domain', default='train', type=str)

parser.add_argument('--beta', default=10, type=int)
parser.add_argument('--exp_times', default=8, type=int)

parser.add_argument('--image_size', default=512, type=int)

parser.add_argument('--clear_cache', default=False, type=str2bool)

if __name__ == '__main__':
    ###################################################################################
    # Arguments
    ###################################################################################
    args = parser.parse_args()
    
    experiment_name = args.model_name

    if 'train' in args.domain:
        experiment_name += '@train'
    else:
        experiment_name += '@val'

    experiment_name += '@beta=%d'%args.beta
    experiment_name += '@exp_times=%d'%args.exp_times
    experiment_name += '@rw'
    
    cam_dir = f'./experiments/predictions/{args.cam_dir}/'
    pred_dir = create_directory(f'./experiments/predictions/{experiment_name}/')

    model_path = './experiments/models/' + f'{args.model_name}.pth'

    set_seed(args.seed)
    log_func = lambda string='': print(string)

    ###################################################################################
    # Transform, Dataset, DataLoader
    ###################################################################################
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    normalize_fn = Normalize(imagenet_mean, imagenet_std)
    
    meta_dic = read_json('./data/VOC_2012.json')
    if args.domain == 'test':
        dataset = VOC_Dataset_For_Evaluation(args.data_dir, args.domain)
    else:
        dataset = VOC_Dataset_For_Making_CAM(args.data_dir, args.domain)
    
    ###################################################################################
    # Network
    ###################################################################################
    stride = 4
    path_index = PathIndex(radius=10, default_size=(args.image_size // stride, args.image_size // stride))
    model = AffinityNet(args.architecture, path_index)

    model = model.cuda()
    model.eval()
    
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

    load_model(model, model_path, parallel=the_number_of_gpu > 1)
    
    #################################################################################################
    # Evaluation
    #################################################################################################
    eval_timer = Timer()

    print('rw output dir: {}'.format(pred_dir))
    print('total number: {}'.format(len(dataset)))
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
            if args.domain == 'test':
                ori_image, image_id, gt_mask = item # (gt_mask is None)
            else:
                ori_image, image_id, label, gt_mask = item
            ori_w, ori_h = ori_image.size

            npy_path = pred_dir + image_id + '.npy'
            if os.path.isfile(npy_path) and (not args.clear_cache):
                continue

            # preprocessing
            image = np.asarray(ori_image)
            image = normalize_fn(image)
            image = image.transpose((2, 0, 1))

            image = torch.from_numpy(image)
            flipped_image = image.flip(-1)

            images = torch.stack([image, flipped_image])
            images = images.cuda()
            
            edge = model.get_edge(images, image_size=(512, 512), stride=stride)

            # postprocessing
            cam_dict = np.load(cam_dir + image_id + '.npy', allow_pickle=True).item()
            cams = cam_dict['cam']
            if isinstance(cams, np.ndarray):
                cams = torch.from_numpy(cams)
            cam_downsized_values = cams.cuda()
            rw = propagate_to_edge(cam_downsized_values, edge, beta=args.beta, exp_times=args.exp_times, radius=5)
            rw_up = F.interpolate(rw, scale_factor=stride, mode='bilinear', align_corners=False)[..., 0, :ori_h, :ori_w]
            rw_up = rw_up / torch.max(rw_up)
            
            np.save(npy_path, {"keys": cam_dict['keys'], "rw": rw_up.cpu().numpy()})
        print()
    
    print("python evaluate.py --experiment_name {} --domain {}".format(experiment_name, args.domain))