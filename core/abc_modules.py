
import math

import torch
import torch.nn as nn

from abc import ABC


class BaseModule(nn.Module):

    def forward(self, *x, stage='forward_x', **kwargs):
        if isinstance(stage, (list, tuple)):
            output = x
            for s in stage:
                func = getattr(self, stage)
                output = func(*tuple(output), **kwargs)
            return output
        else:
            func = getattr(self, stage)
            return func(*x, **kwargs)


class ABC_Model(ABC):

    @property
    def pretrained_modules(self):
        return self._pretrained_modules

    @property
    def scratched_modules(self):
        return self._scratched_modules

    def global_average_pooling_2d(self, x, keepdims=False):
        x = torch.mean(x.view(x.size(0), x.size(1), -1), -1)
        if keepdims:
            x = x.view(x.size(0), x.size(1), 1, 1)
        return x
    
    def initialize(self, modules):
        for m in modules:
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
    def get_parameter_groups(self, print_fn=print, modules=None):
        def get_param_groups(module):
            groups = [[], [], [], []]

            for name, value in module.named_parameters():
                # pretrained weights
                if ('model' in name) or ('stage' in name) or ('backbone' in name):
                    if 'weight' in name:
                        groups[0].append(value)
                    else:
                        groups[1].append(value)
                        
                # scracthed weights
                else:
                    if 'weight' in name:
                        if print_fn is not None:
                            print_fn(f'scratched weights : {name}')
                        groups[2].append(value)
                    else:
                        if print_fn is not None:
                            print_fn(f'scratched bias : {name}')
                        groups[3].append(value)
            return groups

        if modules is None:
            return get_param_groups(self)
        else:
            groups = [[], [], [], []]
            for module in modules:
                gs = get_param_groups(module)
                for i in range(4):
                    groups[i] = groups[i] + gs[i]
            return groups