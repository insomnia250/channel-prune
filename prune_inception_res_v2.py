import os
import torch
from torch.autograd import Variable
import torchvision.models as models
import cv2
import numpy as np
import torchvision
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import dataset
import argparse
from operator import itemgetter
from heapq import nsmallest
import time
from prune2 import *
from models.inception_res_v1 import BasicConv2d, Block35, Block17, Block8, \
                         Mixed_6a, Mixed_7a , InceptionResnetV1

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class FilterPrunner:
    def __init__(self, model):
        self.model = model
        self.reset()

    def reset(self):
        self.filter_ranks = {}
        self.index_to_layername = {}
        self.layername_to_index = {}
        self.hooks = []

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()

    def hook_generator(self, layername):
        def hook_func(grad):
            activation_index = self.layername_to_index[layername]
            activation = self.activations[activation_index]
            # print layername,grad.size(),activation_index,self.activations[activation_index].size()
            values = \
                torch.sum((activation * grad), dim=0, keepdim=True). \
                    sum(dim=2, keepdim=True).sum(dim=3, keepdim=True)[0, :, 0, 0].data
            # Normalize the rank by the filter dimensions
            values = \
                values / (activation.size(0) * activation.size(2) * activation.size(3))

            if activation_index not in self.filter_ranks:
                self.filter_ranks[activation_index] = \
                    torch.FloatTensor(activation.size(1)).zero_().cuda()

            self.filter_ranks[activation_index] += values

        return hook_func

    def lowest_ranking_filters(self, num):
        data = []
        for i in sorted(self.filter_ranks.keys()):
            for j in range(self.filter_ranks[i].size(0)):
                data.append((i, j, self.filter_ranks[i][j]))

        return nsmallest(num, data, itemgetter(2))

    def normalize_ranks_per_layer(self):
        for i in self.filter_ranks:
            v = torch.abs(self.filter_ranks[i])
            v = v / np.sqrt(torch.sum(v * v))
            self.filter_ranks[i] = v.cpu()

    def get_prunning_plan(self, num_filters_to_prune):
        filters_to_prune = self.lowest_ranking_filters(num_filters_to_prune)  # [(actv_idx, channel_idx, value), ...]

        filters_to_prune_per_layer = {}
        for (l, f, _) in filters_to_prune:
            if l not in filters_to_prune_per_layer:
                filters_to_prune_per_layer[l] = []
            filters_to_prune_per_layer[l].append(f)

        return filters_to_prune_per_layer

    def forward(self, x):
        self.activations = []
        activation_index = 0

        # first 5 conv layers
        for i, (name1, module1) in enumerate(self.model._modules.items()):
            if i >= 7: break
            # print name1,'=='*20
            if name1 == 'maxpool_3a' or name1 == 'maxpool_5a':
                x = module1(x)
            for j, (name2, module2) in enumerate(module1._modules.items()):
                # print name1,name2, '--' * 20
                x = module2(x)
                if isinstance(module2, torch.nn.modules.conv.Conv2d):
                    hook = self.hook_generator((name1, name2))
                    h = x.register_hook(hook)
                    self.hooks.append(h)
                    self.activations.append(x)
                    # binding layer name and index
                    if not activation_index in self.index_to_layername.keys():
                        self.index_to_layername[activation_index] = (name1, name2)
                        self.layername_to_index[(name1, name2)] = activation_index
                    activation_index += 1

        # Mixed_5b block
        for i, (name1, module1) in enumerate(self.model._modules.items()):
            if i < 7: continue
            # Mixed_5b block
            if isinstance(module1, Mixed_5b):
                branchs = {}
                branchs['branch0'] = x
                branchs['branch1'] = x
                branchs['branch2'] = x
                branchs['branch3'] = x
                for j, (name2, module2) in enumerate(module1._modules.items()):
                    if name2.startswith('branch0'):
                        branch = 'branch0'
                    elif name2.startswith('branch1'):
                        branch = 'branch1'
                    elif name2.startswith('branch2'):
                        branch = 'branch2'
                    elif name2.startswith('branch3'):
                        branch = 'branch3'
                    else:
                        raise ValueError('"%s" not in Mixed_5b block' % name2)
                    if isinstance(module2, nn.AvgPool2d):
                        branchs[branch] = module2(branchs[branch])
                        print name1, name2,branch, branchs[branch].size()
                    for k, (name3, module3) in enumerate(module2._modules.items()):
                        branchs[branch] = module3(branchs[branch])
                        print name1, name2, name3, branch, branchs[branch].size()
                        if isinstance(module3, torch.nn.modules.conv.Conv2d):
                            hook = self.hook_generator((name1, name2, name3))
                            h = branchs[branch].register_hook(hook)
                            self.hooks.append(h)
                            self.activations.append(branchs[branch])
                            # binding layer name and index
                            if not activation_index in self.index_to_layername.keys():
                                self.index_to_layername[activation_index] = (name1, name2, name3)
                                self.layername_to_index[(name1, name2, name3)] = activation_index
                            activation_index += 1

            # squential of  block
            if isinstance(module1, Mixed_5b):




        print self.index_to_layername

if __name__ == '__main__':
    model = InceptionResnetV1(2)

    prunner = FilterPrunner(model=model)

    x = torch.FloatTensor(10, 3, 299, 299)
    x = torch.autograd.Variable(x)

    prunner.forward(x)