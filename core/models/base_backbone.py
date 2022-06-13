import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import re

from ..module import FixedBatchNorm
from ..arch_resnet import resnet, resnet38
from ..arch_resnest import resnest
from ..arch_vgg import vgg
from ..abc_modules import ABC_Model



class BaseBackboneVGG(nn.Module, ABC_Model):

    ARCH_CFG = {
        '11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        '13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        '16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
        '19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    }

    def __init__(self, model_name, num_classes=20):
        super().__init__()

        self.model_name = model_name
        self.num_classes = num_classes
        # stage1 - stage5
        self.num_features = [64, 128, 256, 512, 512]

        vgg_model = getattr(models, model_name)
        model = vgg_model(pretrained=True)
        stages = BaseBackboneVGG.build_stages(model_name, model.features)
        for i in range(len(stages)):
            self.add_module('stage{}'.format(i + 1), stages[i])

    @staticmethod
    def count_indices(cfg, batch_norm=False):
        indices = [0]
        counter = 0
        for v in cfg:
            if v == 'M':
                counter += 1
                indices.append(counter)
            else:
                if batch_norm:
                    counter += 3
                else:
                    counter += 2
        return indices

    @staticmethod
    def build_stages(model_name, model):
        depth = re.findall(r'\d+', model_name)[0]
        cfg = BaseBackboneVGG.ARCH_CFG[depth]
        bn = 'bn' in model_name

        indices = BaseBackboneVGG.count_indices(cfg, bn)
        assert indices[-1] == len(model), 'indices: {}, model length: {}'.format(indices, len(model))
        stages = []
        for i in range(1, len(indices)):
            stages.append(model[indices[i - 1]:indices[i]])
        return stages

    def forward(self, x):
        stage1 = self.stage1(x)      # stride =  2
        stage2 = self.stage2(stage1) # stride =  4
        stage3 = self.stage3(stage2) # stride =  8
        stage4 = self.stage4(stage3) # stride = 16
        stage5 = self.stage5(stage4) # stride = 32
        return [stage1, stage2, stage3, stage4, stage5]


class BaseBackbone(nn.Module, ABC_Model):
    def __init__(self, model_name, num_classes=20, mode='fix',
        segmentation=False, strides=(2, 2, 2, 1), dilations=(1, 1, 1, 1)):
        super().__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        self.mode = mode
        self.segmentation = segmentation
        self.strides = strides

        if self.mode == 'fix': 
            self.norm_fn = FixedBatchNorm
        else:
            self.norm_fn = nn.BatchNorm2d

        if 'resnet' in model_name:
            if '38' in model_name:
                model = resnet38.ResNet38(strides=strides)
                state_dict = resnet38.convert_mxnet_to_torch()
                model.load_state_dict(state_dict)
            else:
                model = resnet.ResNet(resnet.Bottleneck, resnet.layers_dic[model_name], strides=strides,
                    dilations=dilations, batch_norm_fn=self.norm_fn)

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

        # self.model = model
        if 'resnet38' in model_name:
            self.model = model
        else:
            self.stage1 = nn.Sequential(model.conv1, 
                                        model.bn1, 
                                        model.relu, 
                                        model.maxpool)
            self.stage2 = nn.Sequential(model.layer1)
            self.stage3 = nn.Sequential(model.layer2)
            self.stage4 = nn.Sequential(model.layer3)
            self.stage5 = nn.Sequential(model.layer4)

    def forward(self, x):
        if '38' in self.model_name:
            return self.model(x, return_stages=True)
        stage1 = self.stage1(x)      # stride =  4 / 2
        stage2 = self.stage2(stage1) # stride =  4 / 4
        stage3 = self.stage3(stage2) # stride =  8 / 8
        stage4 = self.stage4(stage3) # stride = 16 / 8
        stage5 = self.stage5(stage4) # stride = 16 / 8
        return [stage1, stage2, stage3, stage4, stage5]


class ReturnLastLayerBaseBackboneVGG(BaseBackboneVGG):

    def forward(self, x):
        return super().forward(x)[-1]


class ReturnLastLayerBaseBackbone(BaseBackbone):

    def forward(self, x):
        return super().forward(x)[-1]


