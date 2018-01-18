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
from torchvision.models.inception import InceptionA, InceptionB, InceptionC, InceptionD, InceptionE, \
    InceptionAux, BasicConv2d

torch.set_default_tensor_type('torch.cuda.FloatTensor')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"




class ModifiedInception3(torch.nn.Module):
    def __init__(self, num_classes, aux_logits=True, transform_input=True):
        super(ModifiedInception3, self).__init__()
        self.aux_logits = aux_logits
        self.transform_input = transform_input
        self.Conv2d_1a_3x3 = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.Conv2d_2a_3x3 = BasicConv2d(32, 32, kernel_size=3)
        self.Conv2d_2b_3x3 = BasicConv2d(32, 64, kernel_size=3, padding=1)
        self.Conv2d_3b_1x1 = BasicConv2d(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = BasicConv2d(80, 192, kernel_size=3)
        self.Mixed_5b = InceptionA(192, pool_features=32)
        self.Mixed_5c = InceptionA(256, pool_features=64)
        self.Mixed_5d = InceptionA(288, pool_features=64)
        self.Mixed_6a = InceptionB(288)
        self.Mixed_6b = InceptionC(768, channels_7x7=128)
        self.Mixed_6c = InceptionC(768, channels_7x7=160)
        self.Mixed_6d = InceptionC(768, channels_7x7=160)
        self.Mixed_6e = InceptionC(768, channels_7x7=192)
        if aux_logits:
            self.AuxLogits = InceptionAux(768, num_classes)
        self.Mixed_7a = InceptionD(768)
        self.Mixed_7b = InceptionE(1280)
        self.Mixed_7c = InceptionE(2048)

        self.fc = nn.Linear(2048, num_classes)

        # initialize
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                import scipy.stats as stats
                stddev = m.stddev if hasattr(m, 'stddev') else 0.1
                X = stats.truncnorm(-2, 2, scale=stddev)
                values = torch.Tensor(X.rvs(m.weight.data.numel()))
                m.weight.data.copy_(values)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        # freeze param except fc linear
        for i, m in enumerate(list(self.children())[0:-1]):
            for p in m.parameters():
                p.requires_grad = False

    def forward(self, x):
        if self.transform_input:
            x = x.clone()
            x[0] = x[0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x[1] = x[1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x[2] = x[2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        # 299 x 299 x 3
        x = self.Conv2d_1a_3x3(x)
        # 149 x 149 x 32
        x = self.Conv2d_2a_3x3(x)
        # 147 x 147 x 32
        x = self.Conv2d_2b_3x3(x)
        # 147 x 147 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 73 x 73 x 64
        x = self.Conv2d_3b_1x1(x)
        # 73 x 73 x 80
        x = self.Conv2d_4a_3x3(x)
        # 71 x 71 x 192
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 35 x 35 x 192
        x = self.Mixed_5b(x)
        # 35 x 35 x 256
        x = self.Mixed_5c(x)
        # 35 x 35 x 288
        x = self.Mixed_5d(x)
        # 35 x 35 x 288
        x = self.Mixed_6a(x)
        # 17 x 17 x 768
        x = self.Mixed_6b(x)
        # 17 x 17 x 768
        x = self.Mixed_6c(x)
        # 17 x 17 x 768
        x = self.Mixed_6d(x)
        # 17 x 17 x 768
        x = self.Mixed_6e(x)
        # 17 x 17 x 768
        if self.training and self.aux_logits:
            aux = self.AuxLogits(x)
        # 17 x 17 x 768
        x = self.Mixed_7a(x)
        # 8 x 8 x 1280
        x = self.Mixed_7b(x)
        # 8 x 8 x 2048
        x = self.Mixed_7c(x)
        # 8 x 8 x 2048
        x = F.avg_pool2d(x, kernel_size=8)
        # 1 x 1 x 2048
        x = F.dropout(x, training=self.training)
        # 1 x 1 x 2048
        x = x.view(x.size(0), -1)
        # 2048
        x = self.fc(x)
        # 1000 (num_classes)
        if self.training and self.aux_logits:
            return x, aux
        return x


class FilterPrunner:
    def __init__(self, model, aux_logits=False, transform_input=True):
        self.model = model
        self.reset()

    def reset(self):
        # self.activations = []
        # self.gradients = []
        # self.grad_index = 0
        # self.activation_to_layer = {}
        self.filter_ranks = {}
        self.index_to_layername = {}
        self.layername_to_index = {}
        self.hooks = []
    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
    # def compute_rank(self, grad):
    #     activation_index = len(self.activations) - self.grad_index - 1
    #     activation = self.activations[activation_index]
    #     # print activation.size(), grad.size()
    #     # print len(self.activations),self.grad_index
    #     # print self.index_to_layername[activation_index]
    #     # print '=='*20
    #     values = \
    #         torch.sum((activation * grad), dim=0,keepdim=True). \
    #             sum(dim=2, keepdim=True).sum(dim=3, keepdim=True)[0, :, 0, 0].data
    #
    #     # Normalize the rank by the filter dimensions
    #     values = \
    #         values / (activation.size(0) * activation.size(2) * activation.size(3))
    #
    #     if activation_index not in self.filter_ranks:
    #         self.filter_ranks[activation_index] = \
    #             torch.FloatTensor(activation.size(1)).zero_().cuda()
    #
    #     self.filter_ranks[activation_index] += values
    #     self.grad_index += 1

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
                data.append((self.activation_to_layer[i], j, self.filter_ranks[i][j]))

        return nsmallest(num, data, itemgetter(2))

    def normalize_ranks_per_layer(self):
        for i in self.filter_ranks:
            v = torch.abs(self.filter_ranks[i])
            v = v / np.sqrt(torch.sum(v * v))
            self.filter_ranks[i] = v.cpu()

    def get_prunning_plan(self, num_filters_to_prune):
        filters_to_prune = self.lowest_ranking_filters(num_filters_to_prune)  # [(actv_idx, channel_idx, value), ...]

        # After each of the k filters are prunned,
        # the filter index of the next filters change since the model is smaller.
        filters_to_prune_per_layer = {}
        for (l, f, _) in filters_to_prune:
            if l not in filters_to_prune_per_layer:
                filters_to_prune_per_layer[l] = []
            filters_to_prune_per_layer[l].append(f)

        # for l in filters_to_prune_per_layer:
        #     filters_to_prune_per_layer[l] = sorted(filters_to_prune_per_layer[l])
        #     for i in range(len(filters_to_prune_per_layer[l])):
        #         filters_to_prune_per_layer[l][i] = filters_to_prune_per_layer[l][i] - i
        #
        # filters_to_prune = []
        # for l in filters_to_prune_per_layer:
        #     for i in filters_to_prune_per_layer[l]:
        #         filters_to_prune.append((l, i))

        # return filters_to_prune
        return filters_to_prune_per_layer

    def forward(self, x):
        self.activations = []
        self.gradients = []
        self.grad_index = 0
        self.activation_to_layer = {}
        c = 0
        activation_index = 0

        if self.model.transform_input:
            x = x.clone()
            x[:, 0] = x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x[:, 1] = x[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x[:, 2] = x[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5

        # first 5 conv layers
        for i, (name1, module1) in enumerate(self.model._modules.items()):
            if i > 4: break
            for j, (name2, module2) in enumerate(module1._modules.items()):
                # print x
                # print module2.weight.data
                x = module2(x)
                if isinstance(module2, torch.nn.modules.conv.Conv2d):
                    #     print j, name1, name2, module2
                    hook = self.hook_generator((name1, name2))
                    h = x.register_hook(hook)
                    self.hooks.append(h)
                    self.activations.append(x)
                    self.activation_to_layer[activation_index] = c
                    # binding layer name and index
                    if not activation_index in self.index_to_layername.keys():
                        self.index_to_layername[activation_index] = (name1, name2)
                        self.layername_to_index[(name1, name2)] = activation_index

                    activation_index += 1
                    c += 1
                elif isinstance(module2, torch.nn.modules.BatchNorm2d):
                    x = F.relu(x, inplace=True)
                if name1 == 'Conv2d_2b_3x3' and isinstance(module2, torch.nn.modules.BatchNorm2d):
                    x = F.max_pool2d(x, kernel_size=3, stride=2)
        # a pooling layer
        x = F.max_pool2d(x, kernel_size=3, stride=2)

        # sequential Inception Block
        for i, (name1, module1) in enumerate(self.model._modules.items()):
            if (i <= 4) and (i >= 17): continue

            # InceotionA block
            if isinstance(module1, InceptionA):
                branchs = {}
                branchs['branch1x1'] = x
                branchs['branch5x5'] = x
                branchs['branch3x3dbl'] = x
                branchs['branch_pool'] = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
                # branch...
                for j, (name2, module2) in enumerate(module1._modules.items()):
                    # print '==' * 20, name1, name2
                    if name2 == 'branch1x1':
                        branch = 'branch1x1'
                    elif name2 in ['branch5x5_1', 'branch5x5_2']:
                        branch = 'branch5x5'
                    elif name2 in ['branch3x3dbl_1', 'branch3x3dbl_2', 'branch3x3dbl_3']:
                        branch = 'branch3x3dbl'
                    elif name2 == 'branch_pool':
                        branch = 'branch_pool'
                    else:
                        raise ValueError('"%s" not in InceptionA block' % name2)
                    for k, (name3, module3) in enumerate(module2._modules.items()):
                        # print name2,name3,module3
                        branchs[branch] = module3(branchs[branch])
                        if isinstance(module3, torch.nn.modules.conv.Conv2d):
                            hook = self.hook_generator((name1, name2, name3))
                            h = branchs[branch].register_hook(hook)
                            self.hooks.append(h)
                            self.activations.append(branchs[branch])
                            self.activation_to_layer[activation_index] = c
                            # binding layer name and index
                            if not activation_index in self.index_to_layername.keys():
                                self.index_to_layername[activation_index] = (name1, name2, name3)
                                self.layername_to_index[(name1, name2, name3)] = activation_index
                            activation_index += 1
                            c += 1
                        elif isinstance(module3, torch.nn.modules.BatchNorm2d):
                            branchs[branch] = F.relu(branchs[branch], inplace=True)

                x = [branchs['branch1x1'], branchs['branch5x5'], branchs['branch3x3dbl'], branchs['branch_pool']]
                x = torch.cat(x, 1)

            if isinstance(module1, InceptionB):
                branchs = {}
                branchs['branch3x3'] = x
                branchs['branch3x3dbl'] = x
                branchs['branch_pool'] = F.max_pool2d(x, kernel_size=3, stride=2)
                # branch
                for j, (name2, module2) in enumerate(module1._modules.items()):
                    # print '==' * 20, name1, name2
                    if name2 == 'branch3x3':
                        branch = 'branch3x3'
                    elif name2 in ['branch3x3dbl_1', 'branch3x3dbl_2', 'branch3x3dbl_3']:
                        branch = 'branch3x3dbl'
                    else:
                        raise ValueError('"%s" not in InceptionB block' % name2)
                    for k, (name3, module3) in enumerate(module2._modules.items()):
                        # print name2, name3, module3
                        branchs[branch] = module3(branchs[branch])
                        if isinstance(module3, torch.nn.modules.conv.Conv2d):
                            hook = self.hook_generator((name1, name2, name3))
                            h = branchs[branch].register_hook(hook)
                            self.hooks.append(h)
                            self.activations.append(branchs[branch])
                            self.activation_to_layer[activation_index] = c
                            # binding layer name and index
                            if not activation_index in self.index_to_layername.keys():
                                self.index_to_layername[activation_index] = (name1, name2, name3)
                                self.layername_to_index[(name1, name2, name3)] = activation_index
                            activation_index += 1
                            c += 1
                        elif isinstance(module3, torch.nn.modules.BatchNorm2d):
                            branchs[branch] = F.relu(branchs[branch], inplace=True)
                x = [branchs['branch3x3'], branchs['branch3x3dbl'], branchs['branch_pool']]
                x = torch.cat(x, 1)

            if isinstance(module1, InceptionC):
                branchs = {}
                branchs['branch1x1'] = x
                branchs['branch7x7'] = x
                branchs['branch7x7dbl'] = x
                branchs['branch_pool'] = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
                # branch...
                for j, (name2, module2) in enumerate(module1._modules.items()):
                    # print '==' * 20, name1, name2
                    if name2 == 'branch1x1':
                        branch = 'branch1x1'
                    elif name2 in ['branch7x7_1', 'branch7x7_2', 'branch7x7_3']:
                        branch = 'branch7x7'
                    elif name2 in ['branch7x7dbl_1', 'branch7x7dbl_2', 'branch7x7dbl_3', 'branch7x7dbl_4',
                                   'branch7x7dbl_5']:
                        branch = 'branch7x7dbl'
                    elif name2 == 'branch_pool':
                        branch = 'branch_pool'
                    else:
                        raise ValueError('"%s" not in InceptionC block' % name2)
                    for k, (name3, module3) in enumerate(module2._modules.items()):
                        # print name2,name3,module3
                        branchs[branch] = module3(branchs[branch])
                        if isinstance(module3, torch.nn.modules.conv.Conv2d):
                            hook = self.hook_generator((name1, name2, name3))
                            h = branchs[branch].register_hook(hook)
                            self.hooks.append(h)
                            self.activations.append(branchs[branch])
                            self.activation_to_layer[activation_index] = c
                            # binding layer name and index
                            if not activation_index in self.index_to_layername.keys():
                                self.index_to_layername[activation_index] = (name1, name2, name3)
                                self.layername_to_index[(name1, name2, name3)] = activation_index
                            activation_index += 1
                            c += 1
                        elif isinstance(module3, torch.nn.modules.BatchNorm2d):
                            branchs[branch] = F.relu(branchs[branch], inplace=True)

                x = [branchs['branch1x1'], branchs['branch7x7'], branchs['branch7x7dbl'],
                     branchs['branch_pool']]
                x = torch.cat(x, 1)

            if isinstance(module1, InceptionD):
                branchs = {}
                branchs['branch3x3'] = x
                branchs['branch7x7x3'] = x
                branchs['branch_pool'] = F.max_pool2d(x, kernel_size=3, stride=2)
                # branch...
                for j, (name2, module2) in enumerate(module1._modules.items()):
                    # print '==' * 20, name1, name2
                    if name2 in ['branch3x3_1', 'branch3x3_2']:
                        branch = 'branch3x3'
                    elif name2 in ['branch7x7x3_1', 'branch7x7x3_2', 'branch7x7x3_3', 'branch7x7x3_4']:
                        branch = 'branch7x7x3'
                    elif name2 == 'branch_pool':
                        branch = 'branch_pool'
                    else:
                        raise ValueError('"%s" not in InceptionD block' % name2)
                    for k, (name3, module3) in enumerate(module2._modules.items()):
                        # print name2, name3, module3
                        branchs[branch] = module3(branchs[branch])
                        if isinstance(module3, torch.nn.modules.conv.Conv2d):
                            hook = self.hook_generator((name1, name2, name3))
                            h = branchs[branch].register_hook(hook)
                            self.hooks.append(h)
                            self.activations.append(branchs[branch])
                            self.activation_to_layer[activation_index] = c
                            # binding layer name and index
                            if not activation_index in self.index_to_layername.keys():
                                self.index_to_layername[activation_index] = (name1, name2, name3)
                                self.layername_to_index[(name1, name2, name3)] = activation_index
                            activation_index += 1
                            c += 1
                        elif isinstance(module3, torch.nn.modules.BatchNorm2d):
                            branchs[branch] = F.relu(branchs[branch], inplace=True)
                x = [branchs['branch3x3'], branchs['branch7x7x3'], branchs['branch_pool']]
                x = torch.cat(x, 1)

            if isinstance(module1, InceptionE):
                branchs = {}
                branchs['branch1x1'] = x
                branchs['branch3x3'] = x
                branchs['branch3x3_2a'] = None
                branchs['branch3x3_2b'] = None
                branchs['branch3x3dbl'] = x
                branchs['branch3x3dbl_3a'] = None
                branchs['branch3x3dbl_3b'] = None
                branchs['branch_pool'] = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
                # branch...
                for j, (name2, module2) in enumerate(module1._modules.items()):
                    # print '==' * 20, name1, name2
                    if name2 == 'branch1x1':
                        branch = 'branch1x1'
                    elif name2 == 'branch3x3_1':
                        branch = 'branch3x3'
                    elif name2 == 'branch3x3_2a':
                        branch = 'branch3x3_2a'
                    elif name2 == 'branch3x3_2b':
                        branch = 'branch3x3_2b'

                    elif name2 in ['branch3x3dbl_1', 'branch3x3dbl_2']:
                        branch = 'branch3x3dbl'
                    elif name2 == 'branch3x3dbl_3a':
                        branch = 'branch3x3dbl_3a'
                    elif name2 == 'branch3x3dbl_3b':
                        branch = 'branch3x3dbl_3b'

                    elif name2 == 'branch_pool':
                        branch = 'branch_pool'
                    else:
                        raise ValueError('"%s" not in InceptionE block' % name2)
                    for k, (name3, module3) in enumerate(module2._modules.items()):
                        # print name2, name3, module3
                        branchs[branch] = module3(branchs[branch])
                        if isinstance(module3, torch.nn.modules.conv.Conv2d):
                            hook = self.hook_generator((name1, name2, name3))
                            h = branchs[branch].register_hook(hook)
                            self.hooks.append(h)
                            self.activations.append(branchs[branch])
                            self.activation_to_layer[activation_index] = c
                            # binding layer name and index
                            if not activation_index in self.index_to_layername.keys():
                                self.index_to_layername[activation_index] = (name1, name2, name3)
                                self.layername_to_index[(name1, name2, name3)] = activation_index
                            activation_index += 1
                            c += 1
                        elif isinstance(module3, torch.nn.modules.BatchNorm2d):
                            branchs[branch] = F.relu(branchs[branch], inplace=True)
                        if name2 == 'branch3x3_1' and isinstance(module3, torch.nn.modules.BatchNorm2d):
                            branchs['branch3x3_2a'] = branchs['branch3x3']
                            branchs['branch3x3_2b'] = branchs['branch3x3']
                        elif name2 == 'branch3x3dbl_2' and isinstance(module3, torch.nn.modules.BatchNorm2d):
                            branchs['branch3x3dbl_3a'] = branchs['branch3x3dbl']
                            branchs['branch3x3dbl_3b'] = branchs['branch3x3dbl']

                branchs['branch3x3'] = torch.cat([branchs['branch3x3_2a'], branchs['branch3x3_2b']], 1)
                branchs['branch3x3dbl'] = torch.cat([branchs['branch3x3dbl_3a'], branchs['branch3x3dbl_3b']], 1)
                x = [branchs['branch1x1'], branchs['branch3x3'],
                     branchs['branch3x3dbl'], branchs['branch_pool']]
                x = torch.cat(x, 1)

            # if isinstance(module1, InceptionAux):
            #     aux = F.avg_pool2d(x, kernel_size=5, stride=3)
            #     for j, (name2, module2) in enumerate(module1._modules.items()):
            #         # print '=='*20, name1,name2
            #         if name2 == 'fc' and isinstance(module2, torch.nn.modules.Linear):
            #             aux = aux.view(aux.size(0), -1)
            #             aux = module2(aux)
            #         for k, (name3, module3) in enumerate(module2._modules.items()):
            #             aux = module3(aux)
            #             if isinstance(module3, torch.nn.modules.conv.Conv2d):
            #                 hook = self.hook_generator((name1, name2, name3))
            #                 h = aux.register_hook(hook)
            #                 self.hooks.append(h)
            #                 self.activations.append(aux)
            #                 self.activation_to_layer[activation_index] = c
            #                 # binding layer name and index
            #                 if not activation_index in self.index_to_layername.keys():
            #                     self.index_to_layername[activation_index] = (name1, name2, name3)
            #                     self.layername_to_index[(name1, name2, name3)] = activation_index
            #                 activation_index += 1
            #                 c += 1
            #             elif isinstance(module3, torch.nn.modules.BatchNorm2d):
            #                 aux = F.relu(aux, inplace=True)

        # 8 x 8 x 2048
        x = F.avg_pool2d(x, kernel_size=8)
        # 1 x 1 x 2048
        x = F.dropout(x, training=self.model.training)
        # 1 x 1 x 2048
        x = x.view(x.size(0), -1)
        # 2048
        x = self.model.fc(x)
        # 1000 (num_classes)
        # if self.model.training and self.model.aux_logits:
        #     return x, aux
        return x

    def forward2(self, x):
        if self.model.transform_input:
            x = x.clone()
            x[:, 0] = x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x[:, 1] = x[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x[:, 2] = x[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        # 299 x 299 x 3
        x = self.model.Conv2d_1a_3x3(x)
        # 149 x 149 x 32
        x = self.model.Conv2d_2a_3x3(x)
        # 147 x 147 x 32
        x = self.model.Conv2d_2b_3x3(x)
        # 147 x 147 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 73 x 73 x 64
        x = self.model.Conv2d_3b_1x1(x)
        # 73 x 73 x 80
        x = self.model.Conv2d_4a_3x3(x)
        # 71 x 71 x 192
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 35 x 35 x 192
        x = self.model.Mixed_5b(x)
        # 35 x 35 x 256
        x = self.model.Mixed_5c(x)
        # 35 x 35 x 288
        x = self.model.Mixed_5d(x)
        # 35 x 35 x 288
        x = self.model.Mixed_6a(x)
        # 17 x 17 x 768
        x = self.model.Mixed_6b(x)
        # 17 x 17 x 768
        x = self.model.Mixed_6c(x)
        # 17 x 17 x 768
        x = self.model.Mixed_6d(x)
        # 17 x 17 x 768
        x = self.model.Mixed_6e(x)
        # 17 x 17 x 768
        if self.model.training and self.model.aux_logits:
            aux = self.model.AuxLogits(x)
        # 17 x 17 x 768
        x = self.model.Mixed_7a(x)
        # 8 x 8 x 1280
        x = self.model.Mixed_7b(x)
        # 8 x 8 x 2048
        x = self.model.Mixed_7c(x)
        # 8 x 8 x 2048
        x = F.avg_pool2d(x, kernel_size=8)
        # 1 x 1 x 2048
        x = F.dropout(x, training=self.model.training)
        # 1 x 1 x 2048
        x = x.view(x.size(0), -1)
        # 2048
        x = self.model.fc(x)
        # 1000 (num_classes)
        if self.model.training and self.model.aux_logits:
            return x, aux
        return x


class PrunningFineTuner_Icep3:
    def __init__(self, train_path, test_path, model):
        self.train_data_loader = dataset.loader(train_path, batch_size=32)
        self.test_data_loader = dataset.test_loader(test_path, batch_size=32)
        self.model = model

        self.criterion = torch.nn.CrossEntropyLoss()
        self.prunner = FilterPrunner(self.model)
        self.model.train()

    def test(self):
        self.model.eval()
        correct = 0
        total = 0
        for i, (inputs, labels) in enumerate(self.test_data_loader):
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
            output = self.model(inputs)
            pred = output.data.max(1)[1]
            correct += pred.eq(labels.data).sum()
            total += labels.size(0)

        print "Accuracy :", float(correct) / total

        self.model.train()

    def train(self, optimizer=None, epoches=10):
        if optimizer is None:
            optimizer = \
                optim.SGD(model.fc.parameters(),
                          lr=0.0001, momentum=0.9)

        self.test()
        for i in range(epoches):
            print "Epoch: ", i

            self.train_epoch(optimizer)
            self.test()
        print "Finished fine tuning."

    def train_batch(self, optimizer, batch, label, rank_filters):
        self.model.zero_grad()
        input = Variable(batch)
        label = Variable(label)
        if rank_filters:
            self.model.eval()
            output2 = self.prunner.forward(input)  # compute filter activation, register hook function
            self.criterion(output2, label).backward()  # compute filter grad by hook, and grad*activation

        else:
            self.model.train()
            loss = self.criterion(self.model(input), label)
            loss.backward()
            optimizer.step()


    def train_epoch(self, optimizer=None, rank_filters=False):
        if rank_filters:
            self.model.eval()
        else:
            self.model.train()
        for input, label in self.train_data_loader:
            self.model.zero_grad()
            input = Variable(input.cuda())
            label = Variable(label.cuda())
            if rank_filters:
                output = self.prunner.forward(input)  # compute filter activation, register hook function
                self.criterion(output, label).backward()  # compute filter grad by hook, and grad*activation

            else:
                loss = self.criterion(self.model(input), label)
                loss.backward()
                optimizer.step()

    def get_candidates_to_prune(self, num_filters_to_prune):
        self.prunner.reset()

        self.train_epoch(rank_filters=True)
        self.prunner.remove_hooks()
        self.prunner.normalize_ranks_per_layer()
        return self.prunner.get_prunning_plan(num_filters_to_prune)

    def total_num_filters(self):
        filters = 0
        for name1, module1 in self.model._modules.items():
            if isinstance(module1, torch.nn.modules.conv.Conv2d):
                filters = filters + module1.out_channels
            for name2, module2 in module1._modules.items():
                if isinstance(module2, torch.nn.modules.conv.Conv2d):
                    filters = filters + module2.out_channels
                for name3, module3 in module2._modules.items():
                    if isinstance(module3, torch.nn.modules.conv.Conv2d):
                        filters = filters + module3.out_channels
        return filters

    def prune(self):
        # Get the accuracy before prunning
        self.test()

        self.model.eval()
        # Make sure all the layers are trainable
        for param in self.model.parameters():
            param.requires_grad = True

        number_of_filters = self.total_num_filters()
        num_filters_to_prune_per_iteration = 512
        iterations = int(float(number_of_filters) / num_filters_to_prune_per_iteration)

        iterations = int(iterations * 2.0 / 3)

        print "Number of prunning iterations to reduce 67% filters", iterations

        for _ in range(iterations):
            self.model.eval()
            prune_targets = self.get_candidates_to_prune(num_filters_to_prune_per_iteration)
            prune_targets_by_layername = {self.prunner.index_to_layername[idx]:prune_targets[idx] for idx in prune_targets.keys()}

            layers_prunned = {self.prunner.index_to_layername[idx]: len(prune_targets[idx]) for idx in prune_targets.keys()}
            #
            print "Layers that will be prunned", layers_prunned
            print "Prunning filters.. "
            self.model.eval()

            model = self.model.cpu()
            model = prune_incep3_conv_layer(model, prune_targets_by_layername)
            torch.save(model, "model_pruned.pth")
            model = torch.load('model_pruned.pth').cuda()
            self.model = model

            message = str(100 * float(self.total_num_filters()) / number_of_filters) + "%"
            print "Filters remain", str(message)
            self.test()
            print "Fine tuning to recover from prunning iteration."
            optimizer = optim.SGD(self.model.parameters(), lr=0.0001, momentum=0.9)
            self.train(optimizer, epoches=5)

            self.prunner.model = self.model  # refresh prunner's model or it will raise error "Tensors on different GPUs"


        print "Finished. Going to fine tune the model a bit more"
        self.train(optimizer, epoches=10)
        torch.save(model, "model_prunned.pth")


class ModifiedVGG16Model(torch.nn.Module):
    def __init__(self):
        super(ModifiedVGG16Model, self).__init__()

        model = models.vgg16(pretrained=True)
        self.features = model.features

        for param in self.features.parameters():
            param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(25088, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 2))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    # model = models.inception_v3(pretrained=True)
    # model.fc = nn.Linear(2048, 2)
    # # model = ModifiedInception3(2)
    # x = torch.FloatTensor(10, 3, 299, 299)
    # x = torch.autograd.Variable(x)
    # y = model(x)
    #
    # prn = FilterPrunner(model)
    # y = prn.forward(x)
    # print y[0].size(),y[1].size()

    isTrain = False
    isPrune = True
    train_path = '/home/gserver/zhangchi/channel-prune/data/train1'
    test_path = '/home/gserver/zhangchi/channel-prune/data/test1'

    if isTrain:
        model = models.inception_v3(pretrained=True,aux_logits=True, transform_input=False)
        model.fc = nn.Linear(2048,2)
        # model  = ModifiedVGG16Model().cuda()
        # freeze param except fc linear
        for i, m in enumerate(list(model.children())[0:-1]):
            for p in m.parameters():
                p.requires_grad = False
        model.aux_logits = False
        model = model.cuda()

    elif isPrune:
        model = torch.load('model_182.pth').cuda()


    fine_tuner = PrunningFineTuner_Icep3(train_path, test_path, model)

    if isTrain:
        fine_tuner.train(epoches=10)
        torch.save(model, "model_182.pth")

    elif isPrune:
        fine_tuner.prune()
