import torch
import torch.nn as nn
import torch.nn.functional as F
import functools

from ..module import SMDConv2d
from .modules import SemanticCorrelationModule
from .base_backbone import (BaseBackboneVGG,
                            BaseBackbone,
                            ReturnLastLayerBaseBackboneVGG,
                            ReturnLastLayerBaseBackbone)
from ..abc_modules import ABC_Model


class BaseClassifier(nn.Module, ABC_Model):

    def __init__(self, model_name, num_classes=20, mode='fix', strides=(2, 2, 2, 1)):
        super().__init__()

        self.backbone = ReturnLastLayerBaseBackbone(model_name, num_classes, mode, strides)
        self.num_classes = num_classes
        # stage1 - stage5
        if '38' in model_name:
            self.num_features = [128, 256, 512, 1024, 4096]
        else:
            self.num_features = [64, 256, 512, 1024, 2048]

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Dropout2d(0.5),
            nn.Conv2d(self.num_features[-1], num_classes + 1, 1, bias=False)
        )

        self.initialize([self.classifier])

    def forward(self, x, with_cam=False):
        stage5 = self.backbone(x)

        if with_cam:
            cam_logits = self.classifier(stage5)
            return stage5, cam_logits
        else:
            gap = self.gap(stage5)
            logits = self.classifier(gap).view(x.shape[0], self.num_classes)
            return logits