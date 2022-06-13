import os
import cv2
import glob
import torch
import copy
import torchvision.datasets as dset

import math
import imageio
import numpy as np

from PIL import Image

from core.aff_utils import *

from tools.ai.augment_utils import *
from tools.ai.torch_utils import one_hot_embedding

from tools.general.xml_utils import read_xml
from tools.general.json_utils import read_json
from tools.dataset.voc_utils import get_color_map_dic



class Iterator:
    def __init__(self, loader):
        self.loader = loader
        self.init()

    def init(self):
        self.iterator = iter(self.loader)
    
    def get(self):
        try:
            data = next(self.iterator)
        except StopIteration:
            self.init()
            data = next(self.iterator)
        
        return data

class VOC_Dataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, domain, with_id=False, with_tags=False, with_mask=False):
        self.root_dir = root_dir

        self.image_dir = self.root_dir + 'JPEGImages/'
        self.xml_dir = self.root_dir + 'Annotations/'
        self.mask_dir = self.root_dir + 'SegmentationClass/'
        
        self.image_id_list = [image_id.strip() for image_id in open('./data/%s.txt'%domain).readlines()]
        
        self.with_id = with_id
        self.with_tags = with_tags
        self.with_mask = with_mask

    def __len__(self):
        return len(self.image_id_list)

    def get_image(self, image_id):
        image = Image.open(self.image_dir + image_id + '.jpg').convert('RGB')
        return image

    def get_mask(self, image_id):
        mask_path = self.mask_dir + image_id + '.png'
        if os.path.isfile(mask_path):
            mask = Image.open(mask_path)
        else:
            mask = None
        return mask

    def get_tags(self, image_id):
        _, tags = read_xml(self.xml_dir + image_id + '.xml')
        return tags
    
    def __getitem__(self, index):
        image_id = self.image_id_list[index]

        data_list = [self.get_image(image_id)]

        if self.with_id:
            data_list.append(image_id)

        if self.with_tags:
            data_list.append(self.get_tags(image_id))

        if self.with_mask:
            data_list.append(self.get_mask(image_id))
        
        return data_list

class VOC_Dataset_For_Classification(VOC_Dataset):
    def __init__(self, root_dir, domain, transform=None):
        super().__init__(root_dir, domain, with_tags=True)
        self.transform = transform

        data = read_json('./data/VOC_2012.json')

        self.class_dic = data['class_dic']
        self.classes = data['classes']

    def __getitem__(self, index):
        image, tags = super().__getitem__(index)

        if self.transform is not None:
            image = self.transform(image)

        label = one_hot_embedding([self.class_dic[tag] for tag in tags], self.classes)
        return image, label

class VOC_Dataset_For_Classification_DetachPadding(VOC_Dataset):
    def __init__(self, root_dir, domain, transform=None, region_to_mask=False):
        super().__init__(root_dir, domain, with_tags=True)
        self.transform = transform
        self.region_to_mask = region_to_mask

        data = read_json('./data/VOC_2012.json')

        self.class_dic = data['class_dic']
        self.classes = data['classes']

    def __getitem__(self, index):
        image, tags = super().__getitem__(index)

        crop_region = None
        if self.transform is not None:
            image, bbox_dict = self.transform(image)
            dst_bbox = bbox_dict['dst_bbox']
            if self.region_to_mask:
                mask = np.zeros(image.shape[-2:], dtype=np.float32)
                mask[dst_bbox['ymin']:dst_bbox['ymax'], dst_bbox['xmin']:dst_bbox['xmax']] = 1
                crop_region = mask
            else:
                crop_region = np.array([dst_bbox['xmin'], dst_bbox['ymin'], dst_bbox['xmax'], dst_bbox['ymax']])

        label = one_hot_embedding([self.class_dic[tag] for tag in tags], self.classes)
        return image, label, crop_region

class VOC_Dataset_For_Classification_MultiView(VOC_Dataset):
    def __init__(self, root_dir, domain, transform=None, num_views=1):
        super().__init__(root_dir, domain, with_tags=True)
        self.transform = transform
        self.num_views = num_views

        data = read_json('./data/VOC_2012.json')

        self.class_dic = data['class_dic']
        self.classes = data['classes']

    def __getitem__(self, index):
        image, tags = super().__getitem__(index)

        if self.transform is not None:
            images = [self.transform(image) for i in range(self.num_views)]
        else:
            images = [copy.deepcopy(image) for i in range(self.num_views)]

        label = one_hot_embedding([self.class_dic[tag] for tag in tags], self.classes)

        return images + [label]

class VOC_Dataset_For_Segmentation(VOC_Dataset):
    def __init__(self, root_dir, domain, transform=None):
        super().__init__(root_dir, domain, with_mask=True)
        self.transform = transform

        cmap_dic, _, class_names = get_color_map_dic()
        self.colors = np.asarray([cmap_dic[class_name] for class_name in class_names])
    
    def __getitem__(self, index):
        image, mask = super().__getitem__(index)

        if self.transform is not None:
            input_dic = {'image':image, 'mask':mask}
            output_dic = self.transform(input_dic)

            image = output_dic['image']
            mask = output_dic['mask']
        
        return image, mask

class VOC_Dataset_For_Evaluation(VOC_Dataset):
    def __init__(self, root_dir, domain, transform=None):
        super().__init__(root_dir, domain, with_id=True, with_mask=True)
        self.transform = transform

        cmap_dic, _, class_names = get_color_map_dic()
        self.colors = np.asarray([cmap_dic[class_name] for class_name in class_names])

    def __getitem__(self, index):
        image, image_id, mask = super().__getitem__(index)

        if self.transform is not None:
            input_dic = {'image':image, 'mask':mask}
            output_dic = self.transform(input_dic)

            image = output_dic['image']
            mask = output_dic['mask']
        
        return image, image_id, mask

class VOC_Dataset_For_WSSS(VOC_Dataset):
    def __init__(self, root_dir, domain, pred_dir, transform=None):
        super().__init__(root_dir, domain, with_id=True)
        self.pred_dir = pred_dir
        self.transform = transform

        cmap_dic, _, class_names = get_color_map_dic()
        self.colors = np.asarray([cmap_dic[class_name] for class_name in class_names])
    
    def __getitem__(self, index):
        image, image_id = super().__getitem__(index)
        mask = Image.open(self.pred_dir + image_id + '.png')
        
        if self.transform is not None:
            input_dic = {'image':image, 'mask':mask}
            output_dic = self.transform(input_dic)

            image = output_dic['image']
            mask = output_dic['mask']
        
        return image, mask

class VOC_Dataset_For_Testing_CAM(VOC_Dataset):
    def __init__(self, root_dir, domain, transform=None):
        super().__init__(root_dir, domain, with_tags=True, with_mask=True)
        self.transform = transform

        cmap_dic, _, class_names = get_color_map_dic()
        self.colors = np.asarray([cmap_dic[class_name] for class_name in class_names])
        
        data = read_json('./data/VOC_2012.json')

        self.class_dic = data['class_dic']
        self.classes = data['classes']

    def __getitem__(self, index):
        image, tags, mask = super().__getitem__(index)

        if self.transform is not None:
            input_dic = {'image':image, 'mask':mask}
            output_dic = self.transform(input_dic)

            image = output_dic['image']
            mask = output_dic['mask']
        
        label = one_hot_embedding([self.class_dic[tag] for tag in tags], self.classes)
        return image, label, mask

class VOC_Dataset_For_Testing_CAM_MultiView(VOC_Dataset):
    def __init__(self, root_dir, domain, transform=None, num_views=1):
        super().__init__(root_dir, domain, with_tags=True, with_mask=True)
        self.transform = transform
        self.num_views = num_views

        cmap_dic, _, class_names = get_color_map_dic()
        self.colors = np.asarray([cmap_dic[class_name] for class_name in class_names])
        
        data = read_json('./data/VOC_2012.json')

        self.class_dic = data['class_dic']
        self.classes = data['classes']

    def __getitem__(self, index):
        image, tags, mask = super().__getitem__(index)

        if self.transform is not None:
            input_dic = {'image':image, 'mask':mask}
            images = []
            masks = []
            for i in range(self.num_views):
                output_dic = self.transform(copy.deepcopy(input_dic))

                image = output_dic['image']
                mask = output_dic['mask']
                images.append(image)
                masks.append(mask)
        else:
            images = [copy.deepcopy(image) for i in range(self.num_views)]
            masks = [copy.deepcopy(mask) for i in range(self.num_views)]
        
        label = one_hot_embedding([self.class_dic[tag] for tag in tags], self.classes)
        return images + [label] + masks

class VOC_Dataset_For_Making_CAM(VOC_Dataset):
    def __init__(self, root_dir, domain):
        super().__init__(root_dir, domain, with_id=True, with_tags=True, with_mask=True)

        cmap_dic, _, class_names = get_color_map_dic()
        self.colors = np.asarray([cmap_dic[class_name] for class_name in class_names])
        
        data = read_json('./data/VOC_2012.json')

        self.class_names = np.asarray(class_names[1:21])
        self.class_dic = data['class_dic']
        self.classes = data['classes']

    def __getitem__(self, index):
        image, image_id, tags, mask = super().__getitem__(index)

        label = one_hot_embedding([self.class_dic[tag] for tag in tags], self.classes)
        return image, image_id, label, mask

class VOC_Dataset_For_Affinity(VOC_Dataset):
    def __init__(self, root_dir, domain, path_index, label_dir, transform=None):
        super().__init__(root_dir, domain, with_id=True)

        data = read_json('./data/VOC_2012.json')

        self.class_dic = data['class_dic']
        self.classes = data['classes']

        self.transform = transform

        self.label_dir = label_dir
        self.path_index = path_index

        self.extract_aff_lab_func = GetAffinityLabelFromIndices(self.path_index.src_indices, self.path_index.dst_indices)

    def __getitem__(self, idx):
        image, image_id = super().__getitem__(idx)

        label = imageio.imread(self.label_dir + image_id + '.png')
        label = Image.fromarray(label)
        
        output_dic = self.transform({'image':image, 'mask':label})
        image, label = output_dic['image'], output_dic['mask']
        
        return image, self.extract_aff_lab_func(label)


class VOC_Dataset_For_Affinity_MS(VOC_Dataset):
    def __init__(self, root_dir, domain, path_indexes, label_dir, scales=[0.5, 0.75], transform=None):
        super().__init__(root_dir, domain, with_id=True)

        data = read_json('./data/VOC_2012.json')

        self.class_dic = data['class_dic']
        self.classes = data['classes']

        self.transform = transform

        self.label_dir = label_dir
        self.path_indexes = path_indexes
        self.scales = scales

        self.extract_aff_lab_funcs = [GetAffinityLabelFromIndices(path_index.src_indices, path_index.dst_indices)
            for path_index in self.path_indexes]

    def __getitem__(self, idx):
        image, image_id = super().__getitem__(idx)

        label = imageio.imread(self.label_dir + image_id + '.png')
        label = Image.fromarray(label)
        
        output_dic = self.transform({'image':image, 'mask':label})
        image, label = output_dic['image'], output_dic['mask']
        
        labs = []
        for f in self.extract_aff_lab_funcs:
            labs += f(label)
        return image, labs