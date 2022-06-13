import os
import cv2
import math
import pandas as pd
import numpy as np
from PIL import Image
import multiprocessing
import argparse
from pprint import pprint
import copy
import joblib
import multiprocessing

from tools.ai.demo_utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--experiment_name', required=True, type=str)
parser.add_argument("--domain", default='train', type=str)
parser.add_argument("--threshold", default=None, type=float)
parser.add_argument('--crf_iteration', default=0, type=int)
parser.add_argument('--n_jobs', default=128, type=int)

parser.add_argument("--predict_dir", default='', type=str)
parser.add_argument('--gt_dir', default='../VOC2012/SegmentationClass', type=str)
parser.add_argument('--data_dir', default='../VOC2012/', type=str)

parser.add_argument('--mode', default='npy', type=str, choices=['npy', 'png', 'fg', 'fgs', 'bg', 'object'])
parser.add_argument('--min_th', default=0.10, type=float)
parser.add_argument('--max_th', default=0.40, type=float)
parser.add_argument('--step', default=0.01, type=float)

args = parser.parse_args()

predict_folder = './experiments/predictions/{}/'.format(args.experiment_name)
gt_folder = args.gt_dir
img_folder = args.data_dir +  'JPEGImages/'

args.list = './data/' + args.domain + '.txt'
args.predict_dir = predict_folder

if args.mode in ['fg', 'bg', 'object']:
    categories = ['background', 'foreground']
else:
    categories = ['background', 
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 
        'bus', 'car', 'cat', 'chair', 'cow', 
        'diningtable', 'dog', 'horse', 'motorbike', 'person', 
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
num_cls = len(categories)


def post_process(inPutMask, kernel_size=7, iterations=1):
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    binaryMask = np.zeros([num_cls, inPutMask.shape[0], inPutMask.shape[1]])
    inputclasses = np.unique(inPutMask)
    for c in inputclasses:
        mask = inPutMask == c
        binaryMask[c, mask] = 1
        binaryMask[c] = cv2.erode(binaryMask[c], kernel, iterations=iterations)
    outPutMask = np.argmax(binaryMask, axis=0).astype(np.uint8)
    return outPutMask

            
def do_python_eval(predict_folder, gt_folder, name_list, num_cls=21):
    TP = [0] * num_cls
    P = [0] * num_cls
    T = [0] * num_cls

    def get_tp_fp(idx):
        name = name_list[idx]

        if args.mode != 'png':
            assert os.path.isfile(predict_folder + name + '.npy')
            predict_dict = np.load(os.path.join(predict_folder, name + '.npy'), allow_pickle=True).item()
            
            if 'hr_cam' in predict_dict.keys():
                cams = predict_dict['hr_cam']
            elif 'rw' in predict_dict.keys():
                cams = predict_dict['rw']

            if 'train' in args.domain:
                keys = predict_dict['keys']
                cams = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=args.threshold)

            n_labels = cams.shape[0]
            cams = np.argmax(cams, axis=0)
            if args.crf_iteration > 0:
                ori_image = np.array(Image.open(img_folder + name + '.jpg'))
                cams = crf_inference_label(np.asarray(ori_image), cams, n_labels=n_labels, t=args.crf_iteration)
            else:
                cams = post_process(cams)

            if 'train' in args.domain:
                predict = keys[cams]
            else:
                predict = cams
        else:
            predict = np.array(Image.open(predict_folder + name + '.png'))
        
        gt_file = os.path.join(gt_folder,'%s.png'%name)
        gt = np.array(Image.open(gt_file))

        if args.mode in ['fg', 'bg', 'object']:
            gt[(gt > 0) & (gt < 255)] = 1
            predict[predict > 0] = 1
        
        cal = gt<255
        mask = (predict==gt) * cal

        p_list, t_list, tp_list = [0] * num_cls, [0] * num_cls, [0] * num_cls
        for i in range(num_cls):
            p_list[i] += np.sum((predict == i) * cal)
            t_list[i] += np.sum((gt == i) * cal)
            tp_list[i] += np.sum((gt == i) * mask)
        return p_list, t_list, tp_list
    

    length = len(name_list)
    
    results = joblib.Parallel(n_jobs=args.n_jobs, backend='loky',
        verbose=0, pre_dispatch="all")(
        [joblib.delayed(get_tp_fp)(i) for i in range(length)]
    )

    p_lists, t_lists, tp_lists = zip(*results)
    assert len(p_lists) == len(t_lists) == len(tp_lists) == length

    for idx in range(length):
        p_list = p_lists[idx]
        t_list = t_lists[idx]
        tp_list = tp_lists[idx]
        for i in range(num_cls):
            TP[i] += tp_list[i]
            P[i] += p_list[i]
            T[i] += t_list[i]

    IoU = []
    T_TP = []
    P_TP = []
    FP_ALL = []
    FN_ALL = [] 
    for i in range(num_cls):
        IoU.append(TP[i] / (T[i] + P[i] - TP[i] + 1e-10))
        T_TP.append(T[i] / (TP[i] + 1e-10))
        P_TP.append(P[i] / (TP[i] + 1e-10))
        FP_ALL.append((P[i] - TP[i]) / (T[i] + P[i] - TP[i] + 1e-10))
        FN_ALL.append((T[i] - TP[i]) / (T[i] + P[i] - TP[i] + 1e-10))

    loglist = {}
    for i in range(num_cls):
        loglist[categories[i]] = IoU[i] * 100
    
    IoU = np.array(IoU)
    if args.mode in ['fg', 'fgs']:
        miou = np.mean(IoU[1:])
        t_tp = np.mean(np.array(T_TP)[1:])
        p_tp = np.mean(np.array(P_TP)[1:])
        fp_all = np.mean(np.array(FP_ALL)[1:])
        fn_all = np.mean(np.array(FN_ALL)[1:])
    elif args.mode == 'bg':
        miou = np.mean(IoU[0:1])
        t_tp = np.mean(np.array(T_TP)[0:1])
        p_tp = np.mean(np.array(P_TP)[0:1])
        fp_all = np.mean(np.array(FP_ALL)[0:1])
        fn_all = np.mean(np.array(FN_ALL)[0:1])
    elif args.mode == 'object':
        miou = np.mean(IoU)
        t_tp = np.mean(np.array(T_TP))
        p_tp = np.mean(np.array(P_TP))
        fp_all = np.mean(np.array(FP_ALL))
        fn_all = np.mean(np.array(FN_ALL))
    else:
        miou = np.mean(IoU)
        t_tp = np.mean(np.array(T_TP))
        p_tp = np.mean(np.array(P_TP))
        fp_all = np.mean(np.array(FP_ALL))
        fn_all = np.mean(np.array(FN_ALL))
    miou_foreground = np.mean(IoU[1:])
    loglist['T_TP'] = T_TP
    loglist['P_TP'] = P_TP
    loglist['FP_ALL'] = FP_ALL
    loglist['FN_ALL'] = FN_ALL

    loglist['IoU'] = IoU * 100
    loglist['mIoU'] = miou * 100
    loglist['t_tp'] = t_tp
    loglist['p_tp'] = p_tp
    loglist['fp_all'] = fp_all
    loglist['fn_all'] = fn_all
    loglist['miou_foreground'] = miou_foreground 
    return loglist


if __name__ == '__main__':
    df = pd.read_csv(args.list, names=['filename'])
    name_list = df['filename'].values

    if args.mode == 'npy' and args.domain =='val':
        loglist = do_python_eval(args.predict_dir, args.gt_dir, name_list, num_cls)
        pprint(loglist)
        print('IoU:')
        pprint({categories[c]: iou for c, iou in enumerate(loglist['IoU'])})
        print('mIoU={:.3f}%, FP={:.4f}, FN={:.4f}'.format(loglist['mIoU'], loglist['fp_all'], loglist['fn_all']))
    elif args.mode == 'png':
        loglist = do_python_eval(args.predict_dir, args.gt_dir, name_list, num_cls)
        pprint(loglist)
        print('IoU:')
        pprint({categories[c]: iou for c, iou in enumerate(loglist['IoU'])})
        print('mIoU={:.3f}%, FP={:.4f}, FN={:.4f}'.format(loglist['mIoU'], loglist['fp_all'], loglist['fn_all']))
    elif args.mode == 'rw':
        th_list = np.arange(args.min_th, args.max_th, args.step).tolist()

        over_activation = 1.60
        under_activation = 0.60
        
        mIoU_list = []
        FP_list = []
        loglist_list = []

        for th in th_list:
            args.threshold = th
            loglist = do_python_eval(args.predict_dir, args.gt_dir, name_list, num_cls)

            mIoU, FP = loglist['mIoU'], loglist['fp_all']

            print('Th={:.3f}, mIoU={:.3f}%, FP={:.4f}'.format(th, mIoU, FP))

            FP_list.append(FP)
            mIoU_list.append(mIoU)
            loglist_list.append(loglist)
        
        best_index = np.argmax(mIoU_list)
        best_th = th_list[best_index]
        best_mIoU = mIoU_list[best_index]
        best_FP = FP_list[best_index]
        best_loglist = loglist_list[best_index]

        over_FP = best_FP * over_activation
        under_FP = best_FP * under_activation

        print('Over FP : {:.4f}, Under FP : {:.4f}'.format(over_FP, under_FP))

        over_loss_list = [np.abs(FP - over_FP) for FP in FP_list]
        under_loss_list = [np.abs(FP - under_FP) for FP in FP_list]

        over_index = np.argmin(over_loss_list)
        over_th = th_list[over_index]
        over_mIoU = mIoU_list[over_index]
        over_FP = FP_list[over_index]

        under_index = np.argmin(under_loss_list)
        under_th = th_list[under_index]
        under_mIoU = mIoU_list[under_index]
        under_FP = FP_list[under_index]
        
        print('IoU:')
        pprint({categories[c]: iou for c, iou in enumerate(best_loglist['IoU'])})

        print('Best Th={:.3f}, mIoU={:.3f}%, FP={:.4f}'.format(best_th, best_mIoU, best_FP))
        print('Over Th={:.3f}, mIoU={:.3f}%, FP={:.4f}'.format(over_th, over_mIoU, over_FP))
        print('Under Th={:.3f}, mIoU={:.3f}%, FP={:.4f}'.format(under_th, under_mIoU, under_FP))
    else:
        if args.threshold is None:
            th_list = np.arange(args.min_th, args.max_th, args.step).tolist()
            
            best_th = 0
            best_log_list = None
            best_mIoU = 0

            for th in th_list:
                args.threshold = th
                loglist = do_python_eval(args.predict_dir, args.gt_dir, name_list, num_cls)
                IoU = {categories[c]: iou for c, iou in enumerate(loglist['IoU'])}
                print('IoU: {}'.format(IoU))
                print('Th={:.3f}, mIoU={:.3f}%, FP={:.4f}, FN={:.4f}'.format(args.threshold, loglist['mIoU'],
                    loglist['fp_all'], loglist['fn_all']))

                if loglist['mIoU'] > best_mIoU:
                    best_th = th
                    best_log_list = loglist
                    best_mIoU = loglist['mIoU']
            
            pprint(best_log_list)
            print('IoU:')
            pprint({categories[c]: iou for c, iou in enumerate(best_log_list['IoU'])})
            print('Best Th={:.3f}, mIoU={:.3f}%, FP={:.4f}, FN={:.4f}'.format(best_th, best_mIoU,
                best_log_list['fp_all'], best_log_list['fn_all']))
        else:
            loglist = do_python_eval(args.predict_dir, args.gt_dir, name_list, num_cls)
            pprint(loglist)
            print('IoU:')
            pprint({categories[c]: iou for c, iou in enumerate(loglist['IoU'])})
            print('Th={:.3f}, mIoU={:.3f}%, FP={:.4f}, FN={:.4f}'.format(args.threshold, loglist['mIoU'],
                loglist['fp_all'], loglist['fn_all']))