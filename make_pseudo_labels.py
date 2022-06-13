# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import os
import sys
import copy
import shutil
import random
import argparse
import numpy as np
import joblib
import multiprocessing

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

parser.add_argument('--n_jobs', default=multiprocessing.cpu_count() // 2, type=int)

###############################################################################
# Inference parameters
###############################################################################
parser.add_argument('--experiment_name', default='', type=str)
parser.add_argument('--domain', default='train', type=str)

parser.add_argument('--threshold', default=0.25, type=float)
parser.add_argument('--crf_iteration', default=1, type=int)

parser.add_argument('--clear_cache', default=False, type=str2bool)

if __name__ == '__main__':
    ###################################################################################
    # Arguments
    ###################################################################################
    args = parser.parse_args()

    cam_dir = f'./experiments/predictions/{args.experiment_name}/'
    pred_dir = create_directory(f'./experiments/predictions/{args.experiment_name}@crf={args.crf_iteration}/')

    set_seed(args.seed)
    log_func = lambda string='': print(string)
    
    ###################################################################################
    # Transform, Dataset, DataLoader
    ###################################################################################
    dataset = VOC_Dataset_For_Making_CAM(args.data_dir, args.domain)
    
    #################################################################################################
    # Evaluation
    #################################################################################################
    eval_timer = Timer()

    print('Pseudo label dir: {}'.format(pred_dir))
    with torch.no_grad():
        length = len(dataset)

        # Process per sample
        def process(i):
            ori_image, image_id, label, gt_mask = dataset.__getitem__(i)

            png_path = pred_dir + image_id + '.png'
            
            ori_w, ori_h = ori_image.size
            predict_dict = np.load(cam_dir + image_id + '.npy', allow_pickle=True).item()

            keys = predict_dict['keys']
            
            if 'rw' in predict_dict.keys():
                cams = predict_dict['rw']
            else:
                cams = predict_dict['hr_cam']
            cams = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=args.threshold)

            cams = np.argmax(cams, axis=0)

            if args.crf_iteration > 0:
                cams = crf_inference_label(np.asarray(ori_image), cams, n_labels=keys.shape[0], t=args.crf_iteration)
            
            conf = keys[cams]
            imageio.imwrite(png_path, conf.astype(np.uint8))
            
        # make pseudo label with multi-process
        joblib.Parallel(n_jobs=args.n_jobs, verbose=10, pre_dispatch="all")(
            [joblib.delayed(process)(i) for i in range(len(dataset))]
        )
    
    print("python evaluate.py --experiment_name {} --mode png".format(args.experiment_name + f'@crf={args.crf_iteration}'))
