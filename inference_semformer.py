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
parser.add_argument('--start', default=0.0, type=float)
parser.add_argument('--end', default=1.0, type=float)

###############################################################################
# Network
###############################################################################
parser.add_argument('--architecture', default='deit', type=str)
parser.add_argument('--version', default='small', type=str)
parser.add_argument('--patch_size', default=16, type=int)
parser.add_argument('--resolution', default=224, type=int)
parser.add_argument('--in21k', default=False, type=str2bool)
parser.add_argument('--train_img_size', default=448, type=int)
parser.add_argument('--cra_layers', default=4, type=int)
parser.add_argument('--class_dim', default=256, type=int)
parser.add_argument('--with_cra', default=True, type=str2bool)

###############################################################################
# Inference parameters
###############################################################################
parser.add_argument('--tag', default='', type=str)
parser.add_argument('--domain', default='train', type=str)

parser.add_argument('--scales', default='0.5,1.0,1.5,2.0', type=str)
parser.add_argument('--reduction', default='sum', type=str)

parser.add_argument('--clear_cache', default=False, type=str2bool)

if __name__ == '__main__':
    ###################################################################################
    # Arguments
    ###################################################################################
    args = parser.parse_args()

    experiment_name = args.tag

    if 'train' in args.domain:
        experiment_name += '@train'
    else:
        experiment_name += '@val'

    experiment_name += '@scale=%s'%args.scales
    
    pred_dir = create_directory(f'./experiments/predictions/{experiment_name}/')

    model_path = './experiments/models/' + f'{args.tag}.pth'

    set_seed(args.seed)
    log_func = lambda string='': print(string)

    ###################################################################################
    # Transform, Dataset, DataLoader
    ###################################################################################
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    
    normalize_fn = Normalize(imagenet_mean, imagenet_std)

    meta_dic = read_json('./data/VOC_2012.json')
    dataset = VOC_Dataset_For_Making_CAM(args.data_dir, args.domain)
    
    ###################################################################################
    # Network
    ###################################################################################
    
    model = SemFormer(
        class_dim=args.class_dim,
        model_name=args.architecture,
        num_classes=meta_dic['classes'] + 1,
        version=args.version,
        patch_size=args.patch_size,
        resolution=args.resolution,
        in21k=args.in21k,
        pos_embed_size=args.train_img_size // args.patch_size
    )

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

    load_model(model, model_path, ignore_modules=['ae'], parallel=the_number_of_gpu > 1)
    
    #################################################################################################
    # Evaluation
    #################################################################################################
    eval_timer = Timer()
    scales = [float(scale) for scale in args.scales.split(',')]
    
    model.eval()
    eval_timer.tik()
    
    reduction_func = getattr(torch, args.reduction)
    is_trans = ('vit' in args.architecture) or ('deit' in args.architecture)


    def get_cam(ori_image, scale):
        # preprocessing
        image = copy.deepcopy(ori_image)
        if scale > 20: # for specific size
            image = image.resize((int(scale), int(scale)), resample=PIL.Image.CUBIC)
        else: # for scaling with float scalar
            if is_trans:
                new_w, new_h = make_divisible(round(ori_w*scale), args.patch_size), make_divisible(round(ori_h*scale), args.patch_size)
                image = image.resize((new_w, new_h), resample=PIL.Image.CUBIC)
            else:
                image = image.resize((round(ori_w*scale), round(ori_h*scale)), resample=PIL.Image.CUBIC)
        
        image = normalize_fn(image)
        image = image.transpose((2, 0, 1))

        image = torch.from_numpy(image)
        flipped_image = image.flip(-1)
        
        images = torch.stack([image, flipped_image])
        images = images.cuda()
        
        # inferenece
        if args.with_cra:
            _, features, cra_list = model(images, return_cra=True)
            features = F.relu(features)
            if args.with_cra:
                cra_list = cra_list[-args.cra_layers:]
                cra_list = [cra.mean(dim=1) for cra in cra_list]
                cra = torch.stack(cra_list, dim=0).sum(dim=0)
                cra = min_max_norm(cra, n_last_dim=1)
                cra = cra.view(*features.shape)
                features = features * cra
        else:
            _, features = model(images)
        features = features[:, 1:, :, :]

        # postprocessing
        cams = F.relu(features)
        # cams = features
        cams = [cams[0], cams[1].flip(-1)]

        return cams

    stride1 = 4
    stride2 = 8 if '38' in args.architecture else 16
    log_func(f'[i] stride1={stride1}, stride2={stride2}')
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
            ori_image, image_id, label, gt_mask = dataset.__getitem__(item_id)
            ori_w, ori_h = ori_image.size

            npy_path = pred_dir + image_id + '.npy'
            if os.path.isfile(npy_path) and (not args.clear_cache):
                continue
            tensor_label = torch.from_numpy(label)
            keys = torch.nonzero(tensor_label)[:, 0]

            strided_size = get_strided_size((ori_h, ori_w), stride1)
            strided_up_size = get_strided_up_size((ori_h, ori_w), stride2)

            cams_list = []
            for scale in scales:
                cams_list += get_cam(ori_image, scale)
            cams_list = [cams.unsqueeze(0) for cams in cams_list]

            strided_cams_list = [resize_for_tensors(cams, strided_size)[0] for cams in cams_list]
            strided_cams = reduction_func(torch.stack(strided_cams_list), dim=0)
            # return tuple when reduction is `max`
            if isinstance(strided_cams, (list, tuple)):
                strided_cams = strided_cams[0]
            
            hr_cams_list = [resize_for_tensors(cams, strided_up_size)[0] for cams in cams_list]
            hr_cams = reduction_func(torch.stack(hr_cams_list), dim=0)
            # return tuple when reduction is `max`
            if isinstance(hr_cams, (list, tuple)):
                hr_cams = hr_cams[0]
            hr_cams = hr_cams[:, :ori_h, :ori_w]
            
            strided_cams = strided_cams[keys]
            strided_cams /= F.adaptive_max_pool2d(strided_cams, (1, 1)) + 1e-5
            
            hr_cams = hr_cams[keys]
            hr_cams /= F.adaptive_max_pool2d(hr_cams, (1, 1)) + 1e-5

            # save cams
            keys = np.pad(keys + 1, (1, 0), mode='constant')
            save_dict = dict(keys=keys, cam=strided_cams.cpu().numpy(), hr_cam=hr_cams.cpu().numpy())
            np.save(npy_path, save_dict)
        print()
    
    if args.domain == 'train_aug':
        args.domain = 'train'
    
    print("python evaluate.py --experiment_name {} --domain {}".format(experiment_name, args.domain))