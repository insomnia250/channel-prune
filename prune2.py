import os
import torch
from torch.autograd import Variable
from torchvision import models
import cv2
import sys
import numpy as np
import math
from torch import nn
from torchvision.models.inception import InceptionA, InceptionB, InceptionC, InceptionD, InceptionE, \
    InceptionAux, BasicConv2d
import torch.optim as optim

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def prune_conv_weight(module, prepruned_index, prune_index):
    ori_weight = module.weight.data  # shape of (out_c, in_c ,kernel_1, kernel_2)

    # prune due to previous layer
    remained_in_index = [x for x in range(ori_weight.size(1)) if x not in prepruned_index]

    # prune due to current layer
    remained_out_index = [x for x in range(ori_weight.size(0)) if x not in prune_index]

    module.in_channels = module.in_channels - len(prepruned_index)
    module.out_channels = module.out_channels - len(prune_index)

    remained_weight = ori_weight[remained_out_index, :, :, :]
    remained_weight = remained_weight[:, remained_in_index, :, :]
    module.weight.data = remained_weight

    if not module.bias is None:
        ori_bias = module.bias.data
        remained_bias = ori_bias[remained_out_index, :]
        module.bias.data = remained_bias
    return module

def prune_bn_weight(module, prepruned_index):
    ori_weight = module.weight.data  # shape of (num_features, )
    ori_bias = module.bias.data  # shape of (num_features, )
    ori_running_mean = module.running_mean  # shape of (num_features, ) bn.running_mean is a Tensor, not Variable
    ori_running_var = module.running_var  # shape of (num_features, )  a Tensor, not Variable

    # prune due to previous layer
    remained_index = [x for x in range(ori_weight.size(0)) if x not in prepruned_index]

    module.num_features = module.num_features - len(prepruned_index)

    module.weight.data = ori_weight[remained_index,]
    module.bias.data = ori_bias[remained_index,]
    module.running_mean = ori_running_mean[remained_index,]
    module.running_var = ori_running_var[remained_index,]
    return module

def prune_fc_weight(module, prepruned_index):
    ori_weight = module.weight.data  # shape of (num_classes, num_features)

    # prune due to previous layer
    remained_index = [x for x in range(ori_weight.size(1)) if x not in prepruned_index]

    module.in_features = module.in_features - len(prepruned_index)

    module.weight.data = ori_weight[:,remained_index]
    return module


def concat_index(ori_out_channels, prune_indexs):
    '''
    :param ori_out_channels: list [32,32,64,...]
    :param prune_indexs: list of prune_index in previous layers(these layers are to be concat) [[1,4], [], [2], ...]
    :return: prune_index in concat layer  [1,4,30]
    '''
    channels_cated = []
    for i,idxs in enumerate(prune_indexs):
        if i==0:
            channels_cated += (prune_indexs[i])
        else:
            pre_channels = sum(ori_out_channels[:i])
            channels_cated += ([x+pre_channels for x in prune_indexs[i]])
    return channels_cated






# layer index => layer name
def prune_incep3_conv_layer(model, prune_target):
    channels_now = 3
    prepruned_index = []

    for i, (name1, module1) in enumerate(model._modules.items()):
        if i>=5:continue
        for j, (name2, module2) in enumerate(module1._modules.items()):
            # print name1,name2
            if isinstance(module2, torch.nn.modules.conv.Conv2d):
                # to prune
                if (name1,name2) in prune_target.keys():
                    prune_index = prune_target[(name1,name2)]
                else:
                    prune_index = []
                module2 = prune_conv_weight(module2, prepruned_index, prune_index)
                prepruned_index = prune_index

                channels_now = module2.out_channels
            elif isinstance(module2, torch.nn.modules.BatchNorm2d):
                module2 = prune_bn_weight(module2, prepruned_index)

            # print module2.weight.data.size(),module2
            # print channels_now
            # print '--'*20

    for i, (name1, module1) in enumerate(model._modules.items()):
        if (i <= 4): continue

        # if InceptionA
        if isinstance(module1, InceptionA):
            branchs = {'branch5x5': [0, channels_now, prepruned_index, []],
                       'branch1x1': [0, channels_now, prepruned_index, []],
                       'branch3x3dbl': [0, channels_now, prepruned_index, []],
                       'branch_pool': [0, channels_now, prepruned_index,[]]
                       }  #  to store ori_out_channel, channel_now, prepruned_index, prune_index in each branch
            # branch...
            for j, (name2, module2) in enumerate(module1._modules.items()):
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
                    if isinstance(module3, torch.nn.modules.conv.Conv2d):
                        branchs[branch][0] = module3.out_channels
                        if (name1, name2, name3) in prune_target.keys():
                            branchs[branch][3] = prune_target[(name1, name2 , name3)]   # prune index
                        else:
                            branchs[branch][3] = []
                        # print name1,name2,name3, branchs[branch]
                        module3 = prune_conv_weight(module3, branchs[branch][2], branchs[branch][3])
                        branchs[branch][2] = branchs[branch][3]  # prepruned index <- prune index
                        branchs[branch][1] = module3.out_channels  # channel_now <- out_channels
                    elif isinstance(module3, torch.nn.modules.BatchNorm2d):
                        module3 = prune_bn_weight(module3, branchs[branch][2])
                    # print module3.weight.data.size(), module3
                    # print branchs[branch]
                    # print '--' * 20

            ori_out_channels = [branchs['branch1x1'][0], branchs['branch5x5'][0], branchs['branch3x3dbl'][0], branchs['branch_pool'][0]]
            channels_now = branchs['branch1x1'][1] + branchs['branch5x5'][1] + branchs['branch3x3dbl'][1] + branchs['branch_pool'][1]
            prune_indexs = [branchs['branch1x1'][3], branchs['branch5x5'][3], branchs['branch3x3dbl'][3], branchs['branch_pool'][3]]
            prepruned_index = concat_index(ori_out_channels, prune_indexs)

        # if InceptionB
        if isinstance(module1, InceptionB):
            branchs = {'branch3x3': [0, channels_now, prepruned_index, []],
                       'branch3x3dbl': [0, channels_now, prepruned_index, []],
                       'branch_pool': [0, channels_now, prepruned_index, prepruned_index],  # this branch has no conv, so out_channels remains as previous layer
                       }  #  to store ori_out_channel, channel_now, prepruned_index, prune_index in each branch

            # branch...
            for j, (name2, module2) in enumerate(module1._modules.items()):
                if name2 == 'branch3x3':
                    branch = 'branch3x3'
                elif name2 in ['branch3x3dbl_1', 'branch3x3dbl_2', 'branch3x3dbl_3']:
                    branch = 'branch3x3dbl'
                else:
                    raise ValueError('"%s" not in InceptionB block' % name2)
                for k, (name3, module3) in enumerate(module2._modules.items()):
                    # print name2,name3,module3
                    if isinstance(module3, torch.nn.modules.conv.Conv2d):
                        branchs[branch][0] = module3.out_channels
                        if (name1, name2, name3) in prune_target.keys():
                            branchs[branch][3] = prune_target[(name1, name2, name3)]  # prune index
                        else:
                            branchs[branch][3] = []
                        # print name1,name2,name3, branchs[branch]
                        module3 = prune_conv_weight(module3, branchs[branch][2], branchs[branch][3])
                        branchs[branch][2] = branchs[branch][3]  # prepruned index <- prune index
                        branchs[branch][1] = module3.out_channels  # channel_now <- out_channels
                    elif isinstance(module3, torch.nn.modules.BatchNorm2d):
                        module3 = prune_bn_weight(module3, branchs[branch][2])
                    # print module3.weight.data.size(), module3
                    # print branchs[branch]
                    # print '--' * 20
            ori_out_channels = [branchs['branch3x3'][0], branchs['branch3x3dbl'][0], branchs['branch_pool'][0]]
            channels_now = branchs['branch3x3'][1] + branchs['branch3x3dbl'][1] + branchs['branch_pool'][1]
            prune_indexs = [branchs['branch3x3'][3], branchs['branch3x3dbl'][3], branchs['branch_pool'][3]]
            prepruned_index = concat_index(ori_out_channels, prune_indexs)

        # if InceptionC
        if isinstance(module1, InceptionC):
            branchs = {'branch1x1': [0, channels_now, prepruned_index, []],
                       'branch7x7': [0, channels_now, prepruned_index, []],
                       'branch7x7dbl': [0, channels_now, prepruned_index, []],
                       'branch_pool': [0, channels_now, prepruned_index, []],
                       }  #  to store ori_out_channel, channel_now, prepruned_index, prune_index in each branch
            # branch...
            for j, (name2, module2) in enumerate(module1._modules.items()):
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
                    if isinstance(module3, torch.nn.modules.conv.Conv2d):
                        branchs[branch][0] = module3.out_channels
                        if (name1, name2, name3) in prune_target.keys():
                            branchs[branch][3] = prune_target[(name1, name2, name3)]  # prune index
                        else:
                            branchs[branch][3] = []
                        # print name1,name2,name3, branchs[branch]
                        module3 = prune_conv_weight(module3, branchs[branch][2], branchs[branch][3])
                        branchs[branch][2] = branchs[branch][3]  # prepruned index <- prune index
                        branchs[branch][1] = module3.out_channels  # channel_now <- out_channels
                    elif isinstance(module3, torch.nn.modules.BatchNorm2d):
                        module3 = prune_bn_weight(module3, branchs[branch][2])
                    # print module3.weight.data.size(), module3
                    # print branchs[branch]
                    # print '--' * 20
            ori_out_channels = [branchs['branch1x1'][0], branchs['branch7x7'][0], branchs['branch7x7dbl'][0], branchs['branch_pool'][0]]
            channels_now = branchs['branch1x1'][1] + branchs['branch7x7'][1] + branchs['branch7x7dbl'][1] + branchs['branch_pool'][1]
            prune_indexs = [branchs['branch1x1'][3], branchs['branch7x7'][3], branchs['branch7x7dbl'][3], branchs['branch_pool'][3]]
            prepruned_index = concat_index(ori_out_channels, prune_indexs)

        # if InceptionD
        if isinstance(module1, InceptionD):
            branchs = {'branch3x3': [0, channels_now, prepruned_index, []],
                       'branch7x7x3': [0, channels_now, prepruned_index, []],
                       'branch_pool': [0, channels_now, prepruned_index, prepruned_index],
                       }  #  to store ori_out_channel, channel_now, prepruned_index, prune_index in each branch
            # branch...
            for j, (name2, module2) in enumerate(module1._modules.items()):
                if name2 in ['branch3x3_1', 'branch3x3_2']:
                    branch = 'branch3x3'
                elif name2 in ['branch7x7x3_1', 'branch7x7x3_2', 'branch7x7x3_3', 'branch7x7x3_4']:
                    branch = 'branch7x7x3'
                elif name2 == 'branch_pool':
                    branch = 'branch_pool'
                else:
                    raise ValueError('"%s" not in InceptionD block' % name2)
                for k, (name3, module3) in enumerate(module2._modules.items()):
                    # print name2,name3,module3
                    if isinstance(module3, torch.nn.modules.conv.Conv2d):
                        branchs[branch][0] = module3.out_channels
                        if (name1, name2, name3) in prune_target.keys():
                            branchs[branch][3] = prune_target[(name1, name2, name3)]  # prune index
                        else:
                            branchs[branch][3] = []
                        # print name1,name2,name3, branchs[branch]
                        module3 = prune_conv_weight(module3, branchs[branch][2], branchs[branch][3])
                        branchs[branch][2] = branchs[branch][3]  # prepruned index <- prune index
                        branchs[branch][1] = module3.out_channels  # channel_now <- out_channels
                    elif isinstance(module3, torch.nn.modules.BatchNorm2d):
                        module3 = prune_bn_weight(module3, branchs[branch][2])
                    # print module3.weight.data.size(), module3
                    # print branchs[branch]
                    # print '--' * 20
            ori_out_channels = [branchs['branch3x3'][0], branchs['branch7x7x3'][0], branchs['branch_pool'][0]]
            channels_now = branchs['branch3x3'][1] + branchs['branch7x7x3'][1] + branchs['branch_pool'][1]
            prune_indexs = [branchs['branch3x3'][3], branchs['branch7x7x3'][3], branchs['branch_pool'][3]]
            prepruned_index = concat_index(ori_out_channels, prune_indexs)

        # if InceptionE
        if isinstance(module1, InceptionE):
            branchs = {'branch1x1': [0, channels_now, prepruned_index, []],
                       'branch3x3': [0, channels_now, prepruned_index, []],
                       'branch3x3_2a': [0, channels_now, prepruned_index, []],
                       'branch3x3_2b': [0, channels_now, prepruned_index, []],
                       'branch3x3dbl': [0, channels_now, prepruned_index, []],
                       'branch3x3dbl_3a': [0, channels_now, prepruned_index, []],
                       'branch3x3dbl_3b': [0, channels_now, prepruned_index, []],
                       'branch_pool': [0, channels_now, prepruned_index, []],
                       }  #  to store ori_out_channel, channel_now, prepruned_index, prune_index in each branch
            # branch...
            for j, (name2, module2) in enumerate(module1._modules.items()):
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
                    # print name2,name3,module3
                    if isinstance(module3, torch.nn.modules.conv.Conv2d):
                        branchs[branch][0] = module3.out_channels
                        if (name1, name2, name3) in prune_target.keys():
                            branchs[branch][3] = prune_target[(name1, name2, name3)]  # prune index
                        else:
                            branchs[branch][3] = []
                        # print name1,name2,name3, branchs[branch]
                        module3 = prune_conv_weight(module3, branchs[branch][2], branchs[branch][3])
                        branchs[branch][2] = branchs[branch][3]  # prepruned index <- prune index
                        branchs[branch][1] = module3.out_channels  # channel_now <- out_channels
                    elif isinstance(module3, torch.nn.modules.BatchNorm2d):
                        module3 = prune_bn_weight(module3, branchs[branch][2])

                    if name2 == 'branch3x3_1' and isinstance(module3, torch.nn.modules.BatchNorm2d):
                        branchs['branch3x3_2a'] = [x for x in branchs['branch3x3']]
                        branchs['branch3x3_2b'] = [x for x in branchs['branch3x3']]
                    elif name2 == 'branch3x3dbl_2' and isinstance(module3, torch.nn.modules.BatchNorm2d):
                        branchs['branch3x3dbl_3a'] = [x for x in branchs['branch3x3dbl']]
                        branchs['branch3x3dbl_3b'] = [x for x in branchs['branch3x3dbl']]
                    # print module3.weight.data.size(), module3
                    # print branchs[branch]
                    # print '--' * 20

            # stage1
            # concat branch3x3_2a and branch3x3_2b in branch3x3
            ori_out_channels = [branchs['branch3x3_2a'][0], branchs['branch3x3_2b'][0]]
            branchs['branch3x3'][0] = branchs['branch3x3_2a'][0] + branchs['branch3x3_2b'][0]
            branchs['branch3x3'][1] = branchs['branch3x3_2a'][1] + branchs['branch3x3_2b'][1]  # channel now
            branchs['branch3x3'][3] = [branchs['branch3x3_2a'][3], branchs['branch3x3_2b'][3]]  # prune index
            branchs['branch3x3'][3] = concat_index(ori_out_channels, branchs['branch3x3'][3])  # prepruned index

            # concat branch3x3dbl_3a and branch3x3dbl_3b in branch3x3dbl
            ori_out_channels = [branchs['branch3x3dbl_3a'][0], branchs['branch3x3dbl_3b'][0]]
            branchs['branch3x3dbl'][0] = branchs['branch3x3dbl_3a'][0] + branchs['branch3x3dbl_3b'][0]
            branchs['branch3x3dbl'][1] = branchs['branch3x3dbl_3a'][1] + branchs['branch3x3dbl_3b'][1]
            branchs['branch3x3dbl'][3] = [branchs['branch3x3dbl_3a'][3], branchs['branch3x3dbl_3b'][3]]
            branchs['branch3x3dbl'][3] = concat_index(ori_out_channels, branchs['branch3x3dbl'][3])

            # stage2
            # concat all
            ori_out_channels = [branchs['branch1x1'][0], branchs['branch3x3'][0], branchs['branch3x3dbl'][0], branchs['branch_pool'][0]]
            channels_now = branchs['branch1x1'][1] + branchs['branch3x3'][1] + branchs['branch3x3dbl'][1] + branchs['branch_pool'][1]
            prune_indexs = [branchs['branch1x1'][3], branchs['branch3x3'][3], branchs['branch3x3dbl'][3], branchs['branch_pool'][3]]
            prepruned_index = concat_index(ori_out_channels, prune_indexs)

        # if fc layer
        if isinstance(module1, nn.Linear) and name1 == 'fc':
            module1 = prune_fc_weight(module1, prepruned_index)


    return model


if __name__ == '__main__':
    model = torch.load('model_182').cpu()
    # for x in model.state_dict().keys():
    #     print x, model.state_dict()[x].size()

    # layer_name = ('Conv2d_1a_3x3', 'conv')
    # filter_indexs = [29]

    # layer_name = ('Conv2d_2a_3x3', 'conv')
    # filter_indexs = [11, 13, 9]
    #
    # layer_name = ('Mixed_5c', 'branch5x5_2', 'conv')
    # filter_indexs = [7, 40]
    prune_target = {
         ('Mixed_6d', 'branch7x7dbl_3', 'conv'): [111, 126], ('Mixed_5c', 'branch3x3dbl_2', 'conv'): [62],
         ('Mixed_6a', 'branch3x3dbl_1', 'conv'): [16],
         ('Mixed_6e', 'branch7x7_3', 'conv'): [9, 134, 5, 50, 141, 95, 131], ('Mixed_5b', 'branch1x1', 'conv'): [18],
         ('Mixed_5d', 'branch5x5_2', 'conv'): [47], ('Mixed_6e', 'branch7x7_1', 'conv'): [184],
         ('Mixed_7c', 'branch_pool', 'conv'): [175, 184, 64, 100, 72, 112, 114],
         ('Mixed_7b', 'branch3x3dbl_1', 'conv'): [441, 328, 78, 209, 256, 176, 83, 116, 240, 358, 140, 85],
         ('Mixed_7a', 'branch3x3_1', 'conv'): [183, 120, 157], ('Mixed_5c', 'branch_pool', 'conv'): [35, 20],
         ('Mixed_5b', 'branch3x3dbl_3', 'conv'): [95], ('Mixed_6b', 'branch7x7dbl_1', 'conv'): [114, 63, 49],
         ('Mixed_6d', 'branch7x7dbl_4', 'conv'): [67, 89, 149], ('Mixed_6c', 'branch1x1', 'conv'): [85],
         ('Mixed_6c', 'branch_pool', 'conv'): [149, 13, 129, 106, 83, 164, 172],
         ('Mixed_7b', 'branch3x3dbl_2', 'conv'): [58, 268, 178, 341, 353, 299, 83, 84, 134, 171, 275, 66, 366, 36, 203],
         ('Mixed_7b', 'branch3x3dbl_3a', 'conv'): [21, 142, 67, 350, 347, 245, 70, 286, 378, 275, 313, 135, 158, 118,
                                                   344, 156, 266, 23, 349, 319, 168, 25, 283, 54],
         ('Mixed_7a', 'branch7x7x3_2', 'conv'): [113, 5, 181, 52, 33], ('Mixed_5c', 'branch1x1', 'conv'): [56, 12],
         ('Mixed_7a', 'branch7x7x3_3', 'conv'): [86, 36, 145, 45, 80, 33, 158, 115, 107, 155, 185, 174, 151],
         ('Mixed_6d', 'branch7x7dbl_1', 'conv'): [114, 3],
         ('Mixed_7b', 'branch3x3_1', 'conv'): [169, 218, 303, 310, 121, 357], ('Mixed_5b', 'branch5x5_1', 'conv'): [8],
         ('Mixed_6b', 'branch7x7_1', 'conv'): [126],
         ('Mixed_6d', 'branch7x7dbl_5', 'conv'): [143, 56, 84, 26, 187, 80, 2, 8, 158, 65, 128, 162, 83],
         ('Mixed_6b', 'branch1x1', 'conv'): [47, 60], ('Mixed_6c', 'branch7x7_1', 'conv'): [92, 127, 46],
         ('Mixed_6d', 'branch7x7_2', 'conv'): [122], ('Mixed_6d', 'branch7x7_1', 'conv'): [133],
         ('Mixed_7a', 'branch7x7x3_4', 'conv'): [16, 93, 187], ('Mixed_6e', 'branch1x1', 'conv'): [89, 106, 169],
         ('Mixed_6a', 'branch3x3dbl_3', 'conv'): [79],
         ('Mixed_7c', 'branch3x3dbl_3a', 'conv'): [70, 136, 121, 195, 289, 288, 47, 269, 383, 42, 43, 145, 179, 173, 19,
                                                   306, 94, 21, 59, 188, 81, 29, 357],
         ('Mixed_7b', 'branch3x3_2a', 'conv'): [160, 314, 81, 330, 159, 323, 213, 96],
         ('Conv2d_3b_1x1', 'conv'): [63, 12, 67], ('Mixed_6d', 'branch7x7_3', 'conv'): [56, 137, 155],
         ('Mixed_7b', 'branch3x3_2b', 'conv'): [79, 215, 265, 327, 235, 253],
         ('Mixed_6e', 'branch7x7dbl_5', 'conv'): [170, 165, 30, 118, 48],
         ('Mixed_7c', 'branch3x3_2a', 'conv'): [364, 48, 62, 87, 5, 163, 80, 6, 223, 60, 29, 271, 382],
         ('Mixed_6b', 'branch7x7dbl_2', 'conv'): [102], ('Conv2d_4a_3x3', 'conv'): [121, 99, 31],
         ('Mixed_6e', 'branch7x7dbl_1', 'conv'): [99, 146, 7], ('Mixed_5b', 'branch3x3dbl_1', 'conv'): [47],
         ('Mixed_7a', 'branch7x7x3_1', 'conv'): [174, 28, 1, 40, 72], ('Mixed_6e', 'branch7x7dbl_2', 'conv'): [123, 97],
         ('Mixed_7c', 'branch3x3dbl_2', 'conv'): [269, 143, 217, 84, 12, 182, 236, 365, 326, 115, 333, 136, 15, 373,
                                                  313, 18, 358, 359, 104, 167, 226, 31, 144, 32, 62, 210, 131, 331],
         ('Mixed_6e', 'branch7x7dbl_4', 'conv'): [38, 23, 67], ('Mixed_6c', 'branch7x7dbl_3', 'conv'): [78, 132, 88],
         ('Mixed_6d', 'branch1x1', 'conv'): [149, 71, 159, 176, 14], ('Mixed_6c', 'branch7x7dbl_2', 'conv'): [48, 51],
         ('Mixed_6e', 'branch7x7dbl_3', 'conv'): [136, 8, 144],
         ('Mixed_7c', 'branch3x3_1', 'conv'): [327, 281, 55, 38, 299, 370, 225, 322, 242, 4, 47, 354, 29, 13, 16],
         ('Mixed_7c', 'branch1x1', 'conv'): [297, 105, 193, 7, 69, 278, 198, 299, 211, 283, 176, 254, 10, 102],
         ('Mixed_6c', 'branch7x7dbl_4', 'conv'): [42], ('Mixed_6e', 'branch_pool', 'conv'): [16, 105, 180],
         ('Mixed_5d', 'branch3x3dbl_3', 'conv'): [21, 11, 64], ('Mixed_6c', 'branch7x7_3', 'conv'): [54, 151],
         ('Mixed_7a', 'branch3x3_2', 'conv'): [36, 184, 157, 144, 305],
         ('Mixed_6c', 'branch7x7dbl_5', 'conv'): [22, 31],
         ('Mixed_6b', 'branch_pool', 'conv'): [135, 56, 44, 175, 93, 77, 17, 59, 190, 152, 123, 94],
         ('Mixed_6e', 'branch7x7_2', 'conv'): [6, 12, 72, 31, 39],
         ('Mixed_6b', 'branch7x7dbl_5', 'conv'): [134, 133, 14], ('Mixed_5c', 'branch5x5_2', 'conv'): [18, 7],
         ('Mixed_6c', 'branch7x7_2', 'conv'): [0, 52, 83], ('Conv2d_2a_3x3', 'conv'): [11, 13, 28],
         ('Mixed_7c', 'branch3x3_2b', 'conv'): [356, 183, 3, 23, 218, 39, 6],
         ('Mixed_6b', 'branch7x7dbl_4', 'conv'): [110, 83],
         ('Mixed_7b', 'branch3x3dbl_3b', 'conv'): [92, 311, 316, 138, 135, 139, 365, 263, 258, 303, 297, 81, 152, 48,
                                                   356, 310, 269, 90, 12, 17, 212, 318, 78, 105, 153, 240, 41, 223,
                                                   325],
         ('Mixed_7c', 'branch3x3dbl_1', 'conv'): [229, 407, 271, 385, 181, 6, 36, 73, 20, 151, 245, 40, 299, 120, 354,
                                                  96, 69, 441, 434, 119, 285, 370, 399, 310, 437, 235, 335, 379, 409,
                                                  352, 305, 82, 266, 210, 111, 254, 394, 368, 255],
         ('Mixed_7b', 'branch1x1', 'conv'): [230, 271, 24, 177, 175, 160, 251, 31, 246, 59, 89, 13, 307, 85, 220, 169,
                                             143, 206, 55, 61, 30, 286, 111, 310, 200, 269, 195, 274, 197, 0],
         ('Mixed_7c', 'branch3x3dbl_3b', 'conv'): [94, 144, 170, 267, 67, 61, 88, 281, 193, 362, 344, 302, 160, 274,
                                                   216, 241, 192, 233, 288, 244, 156, 311, 113, 47, 209, 222, 379, 164,
                                                   104, 198, 148],
         ('Mixed_6d', 'branch_pool', 'conv'): [88, 120, 182, 40, 27, 94],
         ('Mixed_6b', 'branch7x7_3', 'conv'): [121, 159, 112],
         ('Mixed_6a', 'branch3x3', 'conv'): [159, 184, 61, 219, 170, 115, 225, 73, 348, 242],
         ('Mixed_7b', 'branch_pool', 'conv'): [69, 143, 10], ('Mixed_6b', 'branch7x7_2', 'conv'): [39, 23, 46],
         ('Mixed_6d', 'branch7x7dbl_2', 'conv'): [67, 77, 63],
         ('Mixed_6c', 'branch7x7dbl_1', 'conv'): [19, 83, 107, 114]
    }
    model = prune_incep3_conv_layer(model, prune_target).cuda()
    print model
    # Make sure all the layers are trainable
    for param in model.parameters():
        param.requires_grad = True
    model.eval()
    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()

    x = torch.FloatTensor(5, 3, 299, 299)
    x = torch.autograd.Variable(x.cuda())
    label = torch.LongTensor([1,0,0,0,1])
    label = torch.autograd.Variable(label.cuda())

    output = model(x)
    print output,label
    criterion(output, label).backward()
    optimizer.step()


    # x = torch.FloatTensor(10, 3, 299, 299)
    # x = torch.autograd.Variable(x)
    # y = model(x)
    # print y.size()