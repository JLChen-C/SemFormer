import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models
import torch.utils.model_zoo as model_zoo

from .arch_resnet import resnet, resnet38
from .arch_resnest import resnest
from .arch_vgg import vgg

from .deeplab_utils import ASPP, Decoder
from .aff_utils import PathIndex

from tools.ai.torch_utils import resize_for_tensors

from .module import FixedBatchNorm, Interpolate
from .abc_modules import ABC_Model

from .module import *
from .functional import *
from .models import *
from .utils import *

#######################################################################

class Backbone(nn.Module, ABC_Model):
    def __init__(self, model_name, num_classes=20, mode='fix', segmentation=False):
        super().__init__()

        self.mode = mode

        if self.mode == 'fix': 
            self.norm_fn = FixedBatchNorm
        else:
            self.norm_fn = nn.BatchNorm2d
        
        if 'resnet' in model_name:
            if '38' in model_name:
                model = resnet38.ResNet38()
                state_dict = resnet38.convert_mxnet_to_torch()
                model.load_state_dict(state_dict)
            else:
                model = resnet.ResNet(resnet.Bottleneck, resnet.layers_dic[model_name], strides=(2, 2, 2, 1), batch_norm_fn=self.norm_fn)

                state_dict = model_zoo.load_url(resnet.urls_dic[model_name])
                state_dict.pop('fc.weight')
                state_dict.pop('fc.bias')

                model.load_state_dict(state_dict)
        else:
            if segmentation:
                dilation, dilated = 4, True
            else:
                dilation, dilated = 2, False

            model = eval("resnest." + model_name)(pretrained=True, dilated=dilated, dilation=dilation, norm_layer=self.norm_fn)

            del model.avgpool
            del model.fc

        if 'resnet38' in model_name:
            self.model = model
        else:
            self.stage1 = nn.Sequential(model.conv1, 
                                        model.bn1, 
                                        model.relu, 
                                        model.maxpool) # stride = 4
            self.stage2 = nn.Sequential(model.layer1) # stride = 4
            self.stage3 = nn.Sequential(model.layer2) # stride = 8
            self.stage4 = nn.Sequential(model.layer3) # stride = 16
            self.stage5 = nn.Sequential(model.layer4) # stride = 16


class Classifier(Backbone):
    def __init__(self, model_name, num_classes=20, mode='fix'):
        super().__init__(model_name, num_classes, mode)
        
        self.model_name = model_name
        if model_name == 'resnet38':
            self.classifier = nn.Sequential(
                nn.Dropout2d(0.5),
                nn.Conv2d(4096, num_classes, 1, bias=False)
            )
        else:
            self.classifier = nn.Conv2d(2048, num_classes, 1, bias=False)
        self.num_classes = num_classes

        self.initialize([self.classifier])
    
    def forward(self, x, with_cam=False):
        if '38' in self.model_name:
            x = self.model(x)
        else:
            x = self.stage1(x)
            x = self.stage2(x)
            x = self.stage3(x)
            x = self.stage4(x)
            x = self.stage5(x)
        
        if with_cam:
            features = self.classifier(x)
            logits = self.global_average_pooling_2d(features)
            return logits, features
        else:
            x = self.global_average_pooling_2d(x, keepdims=True) 
            logits = self.classifier(x).view(-1, self.num_classes)
            return logits


class DeepLabv3_Plus(Backbone):
    def __init__(self, model_name, num_classes=21, mode='fix', use_group_norm=False, dropout_ratios=(0.5, 0.1)):
        super().__init__(model_name, num_classes, mode, segmentation=False)

        self.model_name = model_name
        if use_group_norm:
            norm_fn_for_extra_modules = group_norm
        else:
            norm_fn_for_extra_modules = self.norm_fn
        
        if '38' in model_name:
            inplanes = 4096
            self.aspp = ASPP(output_stride=8, norm_fn=norm_fn_for_extra_modules, inplanes=inplanes)
            self.decoder = Decoder(num_classes, 256, norm_fn_for_extra_modules, dropout_ratios=dropout_ratios)
        else:
            inplanes = 2048
            self.aspp = ASPP(output_stride=16, norm_fn=norm_fn_for_extra_modules, inplanes=inplanes)
            self.decoder = Decoder(num_classes, 256, norm_fn_for_extra_modules, dropout_ratios=dropout_ratios)
        
    def forward(self, x, with_cam=False):
        inputs = x

        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x_low_level = x1 if '38' in self.model_name else x2
        
        x = self.stage3(x2)
        x = self.stage4(x)
        x = self.stage5(x)
        
        x = self.aspp(x)
        x = self.decoder(x, x_low_level)
        x = resize_for_tensors(x, inputs.size()[2:], align_corners=True)

        return x