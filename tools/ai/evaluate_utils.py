import numpy as np
import torch
from sklearn.metrics import average_precision_score

from tools.general.json_utils import read_json
from core.functional import cosine_similarity

def calculate_for_tags(pred_tags, gt_tags):
    """This function calculates precision, recall, and f1-score using tags.

    Args:
        pred_tags: 
            The type of variable is list.
            The type of each element is string.

        gt_tags:
            The type of variable is list.
            the type of each element is string.

    Returns:
        precision:
            pass

        recall:
            pass

        f1-score:
            pass
    """
    if len(pred_tags) == 0 and len(gt_tags) == 0:
        return 100, 100, 100
    elif len(pred_tags) == 0 or len(gt_tags) == 0:
        return 0, 0, 0
    
    pred_tags = np.asarray(pred_tags)
    gt_tags = np.asarray(gt_tags)

    precision = pred_tags[:, np.newaxis] == gt_tags[np.newaxis, :]
    recall = gt_tags[:, np.newaxis] == pred_tags[np.newaxis, :]
    
    precision = np.sum(precision) / len(precision) * 100
    recall = np.sum(recall) / len(recall) * 100
    
    if precision == 0 and recall == 0:
        f1_score = 0
    else:
        f1_score = 2 * ((precision * recall) / (precision + recall))

    return precision, recall, f1_score

def calculate_mIoU(pred_mask, gt_mask):
    """This function is to calculate precision, recall, and f1-score using tags.

    Args:
        pred_mask: 
            The type of variable is numpy array.

        gt_mask:
            The type of variable is numpy array.

    Returns:
        miou:
            miou is meanIU.
    """
    inter = np.logical_and(pred_mask, gt_mask)
    union = np.logical_or(pred_mask, gt_mask)
    
    epsilon = 1e-5
    miou = (np.sum(inter) + epsilon) / (np.sum(union) + epsilon)
    return miou * 100

class Calculator_For_mIoU:
    def __init__(self, json_path=None, class_names=None):
        if class_names is None:
            data = read_json(json_path)
            class_names = data['class_names']
        self.class_names = ['background'] + class_names
        self.classes = len(self.class_names)

        self.clear()

    def get_data(self, pred_mask, gt_mask):
        obj_mask = gt_mask<255
        correct_mask = (pred_mask==gt_mask) * obj_mask
        
        P_list, T_list, TP_list = [], [], []
        for i in range(self.classes):
            P_list.append(np.sum((pred_mask==i)*obj_mask))
            T_list.append(np.sum((gt_mask==i)*obj_mask))
            TP_list.append(np.sum((gt_mask==i)*correct_mask))

        return (P_list, T_list, TP_list)

    def add_using_data(self, data):
        P_list, T_list, TP_list = data
        for i in range(self.classes):
            self.P[i] += P_list[i]
            self.T[i] += T_list[i]
            self.TP[i] += TP_list[i]

    def add(self, pred_mask, gt_mask):
        obj_mask = gt_mask<255
        correct_mask = (pred_mask==gt_mask) * obj_mask

        for i in range(self.classes):
            self.P[i] += np.sum((pred_mask==i)*obj_mask)
            self.T[i] += np.sum((gt_mask==i)*obj_mask)
            self.TP[i] += np.sum((gt_mask==i)*correct_mask)

    def get(self, detail=False, clear=True):
        IoU_dic = {}
        IoU_list = []

        FP_list = [] # over activation
        FN_list = [] # under activation

        for i in range(self.classes):
            IoU = self.TP[i]/(self.T[i]+self.P[i]-self.TP[i]+1e-10) * 100
            FP = (self.P[i]-self.TP[i])/(self.T[i] + self.P[i] - self.TP[i] + 1e-10)
            FN = (self.T[i]-self.TP[i])/(self.T[i] + self.P[i] - self.TP[i] + 1e-10)

            IoU_dic[self.class_names[i]] = IoU

            IoU_list.append(IoU)
            FP_list.append(FP)
            FN_list.append(FN)
        
        mIoU = np.mean(np.asarray(IoU_list))
        mIoU_foreground = np.mean(np.asarray(IoU_list)[1:])

        FP = np.mean(np.asarray(FP_list))
        FN = np.mean(np.asarray(FN_list))
        
        if clear:
            self.clear()
        
        if detail:
            return mIoU, mIoU_foreground, IoU_dic, FP, FN
        else:
            return mIoU, mIoU_foreground

    def clear(self):
        self.TP = []
        self.P = []
        self.T = []
        
        for _ in range(self.classes):
            self.TP.append(0)
            self.P.append(0)
            self.T.append(0)


def calculate_mIoU_cuda(pred_mask, gt_mask):
    """This function is to calculate precision, recall, and f1-score using tags.

    Args:
        pred_mask: 
            The type of variable is numpy array.

        gt_mask:
            The type of variable is numpy array.

    Returns:
        miou:
            miou is meanIU.
    """
    inter = pred_mask & gt_mask
    union = pred_mask | gt_mask
    
    epsilon = 1e-5
    miou = (torch.sum(inter) + epsilon) / (torch.sum(union) + epsilon)
    return miou.item() * 100

class Calculator_For_mIoU_CUDA:
    def __init__(self, json_path=None, class_names=None):
        if class_names is None:
            data = read_json(json_path)
            class_names = data['class_names']
        self.class_names = ['background'] + class_names
        self.classes = len(self.class_names)

        self.clear()

    def get_data(self, pred_mask, gt_mask):
        obj_mask = gt_mask<255
        correct_mask = (pred_mask==gt_mask) * obj_mask
        
        P_list, T_list, TP_list = [], [], []
        for i in range(self.classes):
            P_list.append(torch.sum((pred_mask==i)*obj_mask).item())
            T_list.append(torch.sum((gt_mask==i)*obj_mask).item())
            TP_list.append(torch.sum((gt_mask==i)*correct_mask).item())

        return (P_list, T_list, TP_list)

    def add_using_data(self, data):
        P_list, T_list, TP_list = data
        for i in range(self.classes):
            self.P[i] += P_list[i]
            self.T[i] += T_list[i]
            self.TP[i] += TP_list[i]

    def add(self, pred_mask, gt_mask):
        obj_mask = gt_mask<255
        correct_mask = (pred_mask==gt_mask) * obj_mask

        for i in range(self.classes):
            self.P[i] += torch.sum((pred_mask==i)*obj_mask).item()
            self.T[i] += torch.sum((gt_mask==i)*obj_mask).item()
            self.TP[i] += torch.sum((gt_mask==i)*correct_mask).item()

    def get(self, detail=False, clear=True):
        IoU_dic = {}
        IoU_list = []

        FP_list = [] # over activation
        FN_list = [] # under activation

        for i in range(self.classes):
            IoU = self.TP[i]/(self.T[i]+self.P[i]-self.TP[i]+1e-10) * 100
            FP = (self.P[i]-self.TP[i])/(self.T[i] + self.P[i] - self.TP[i] + 1e-10)
            FN = (self.T[i]-self.TP[i])/(self.T[i] + self.P[i] - self.TP[i] + 1e-10)

            IoU_dic[self.class_names[i]] = IoU

            IoU_list.append(IoU)
            FP_list.append(FP)
            FN_list.append(FN)
        
        mIoU = np.mean(np.asarray(IoU_list))
        mIoU_foreground = np.mean(np.asarray(IoU_list)[1:])

        FP = np.mean(np.asarray(FP_list))
        FN = np.mean(np.asarray(FN_list))
        
        if clear:
            self.clear()
        
        if detail:
            return mIoU, mIoU_foreground, IoU_dic, FP, FN
        else:
            return mIoU, mIoU_foreground

    def clear(self):
        self.TP = []
        self.P = []
        self.T = []
        
        for _ in range(self.classes):
            self.TP.append(0)
            self.P.append(0)
            self.T.append(0)

def compute_AP(labels, outputs):
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    if isinstance(outputs, torch.Tensor):
        outputs = outputs.cpu().numpy()
    y_true = labels
    y_pred = outputs
    AP = []
    for i in range(y_true.shape[0]):
        AP.append(average_precision_score(y_true[i], y_pred[i]))
    return AP

class Calculator_For_mAP:

    def __init__(self, json_path=None, class_names=None):
        if class_names is None:
            data = read_json(json_path)
            class_names = data['class_names']
        self.class_names = ['background'] + class_names
        self.classes = len(self.class_names)

        self.clear()

    def get_data(self, logits, labels, topk=None):
        return np.mean(compute_AP(labels, logits))

    def add_using_data(self, data):
        self.ap_list += data

    def add(self, logits, labels, topk=None):
        AP = compute_AP(labels, logits)
        self.ap_list += AP

    def get(self, detail=False, clear=True):
        mAP = 100 * np.mean(self.ap_list)

        if clear:
            self.clear()

        return mAP

    def clear(self):
        self.ap_list = []