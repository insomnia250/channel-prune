import torch
from torch.autograd import Variable
import torchvision.models as models
import cv2
import numpy as np
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dataset
from prune import *
import argparse
from operator import itemgetter
from heapq import nsmallest
import time
from models.inception import *

class ModifiedInception3(torch.nn.Module):
    def __init__(self,num_classes,aux_logits=True,transform_input=True):
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
        for i,m in enumerate(list(self.children())[0:-1]):
            for p in m.parameters():
                p.requires_grad = False
    def forward(self,x):
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

    def __call__(self, x):
        self.activations = []
        self.gradients = []
        self.grad_index = 0
        self.activation_to_layer = {}
        c=0
        activation_index = 0

        # first 5 conv layers
        for i, (name1, module1) in enumerate(self.model._modules.items()):
            if i > 4:break
            for j, (name2, module2) in enumerate(module1._modules.items()):
                x = module2(x)
                if isinstance(module2, torch.nn.modules.conv.Conv2d):
                    print j, name1, name2, module2
                    # x.register_hook(self.compute_rank)
                    # self.activations.append(x)
                    # self.activation_to_layer[activation_index] = c
                    # activation_index += 1
                    # c += 1

        # a pooling layer
        x = F.max_pool2d(x, kernel_size=3, stride=2)

        # sequential Inception Block
        for i, (name1, module1) in enumerate(self.model._modules.items()):
            if (i <= 4) or (i>=16): continue
            if name1 == 'AuxLogits': continue


            if isinstance(module1, InceptionA):
                print i, name1











        # for i, (name1,module1) in enumerate(self.model._modules.items()):
        #     if len(module1._modules.items()) == 0:
        #         print name1
        #         if isinstance(module1, torch.nn.modules.conv.Conv2d):
        #             x.register_hook(self.compute_rank)
        #             self.activations.append(x)
        #             self.activation_to_layer[activation_index] = layer
        #             activation_index += 1
        #     for j, (name2, module2) in enumerate(module1._modules.items()):
        #         if len(module2._modules.items()) == 0:
        #             print name1, name2
        #         for k, (name3, module3) in enumerate(module2._modules.items()):
        #             print i, j, k, name1, name2, name3
        #     # print 'param', len([p for p in module1.parameters()])
        #     print '==' * 20
        #     # c += len([p for p in module1.parameters()])
        # # print c
if __name__ == '__main__':
    model = ModifiedInception3(2)
    x = torch.FloatTensor(10, 3, 299, 299)
    x = torch.autograd.Variable(x)
    y = model(x)

    prn = FilterPrunner(model)
    y = prn(x)
    # print prn.model.Mixed_5d.branch1x1.conv
    # c = 0
    # for i, (name, module) in enumerate(model._modules.items()):
    #     if len(module._modules.items())==0:
    #         print name
    #     for j,(name2,module2) in enumerate(module._modules.items()):
    #         if len(module2._modules.items())==0:
    #             print name,name2
    #         for k, (name3, module3) in enumerate(module2._modules.items()):
    #             print i,j,k, name,name2,name3
    #     print 'param',len([p for p in module.parameters()])
    #     print '=='*20
    #     c+=len([p for p in module.parameters()])
    # print c
    # for layer,(name,module) in enumerate(model._modules.items()):
    #     print layer
    #     print '---'
    #     print name,module
    #     print '==='*20