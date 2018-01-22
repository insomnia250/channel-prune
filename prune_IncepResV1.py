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
        remained_bias = ori_bias[remained_out_index,]
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
        if i>=7:continue
        for j, (name2, module2) in enumerate(module1._modules.items()):
            if isinstance(module2, torch.nn.modules.conv.Conv2d):
                # # to prune
                if (name1,name2) in prune_target.keys():
                    prune_index = prune_target[(name1,name2)]
                else:
                    prune_index = []
                module2 = prune_conv_weight(module2, prepruned_index, prune_index)
                prepruned_index = prune_index

                channels_now = module2.out_channels
                # print module2.weight.data.size(),module2
                # print channels_now
                # print '--'*20
            elif isinstance(module2, torch.nn.modules.BatchNorm2d):
                module2 = prune_bn_weight(module2, prepruned_index)

                # print module2.weight.data.size(),module2
                # print channels_now
                # print '--'*20
    print len(prepruned_index),channels_now
    repeat35_prune = prepruned_index
    repeat35_channels_now = channels_now
    repeat35_ori_channels = channels_now+len(prepruned_index)
    for i, (name1, module1) in enumerate(model._modules.items()):
        if (i < 7 ): continue
        # print i, name1
        # Block35
        if name1 == 'repeat_block35':
            # each block
            for j, (name2, module2) in enumerate(module1._modules.items()):
                branchs = {'branch0': [0, channels_now, repeat35_prune, []],
                           'branch1': [0, channels_now, repeat35_prune, []],
                           'branch2': [0, channels_now, repeat35_prune, []],
                           }  # to store ori_out_channel, channel_now, prepruned_index, prune_index in each branch
                # branch...
                for k, (name3, module3) in enumerate(module2._modules.items()):
                    # skip now
                    if isinstance(module3, nn.Conv2d) or isinstance(module3, nn.ReLU):
                        continue
                    if name3.startswith('branch0'):
                        branch = 'branch0'
                    elif name3.startswith('branch1'):
                        branch = 'branch1'
                    elif name3.startswith('branch2'):
                        branch = 'branch2'
                    else:
                        raise ValueError('"%s" not in Block35 block' % name3)

                    for l, (name4, module4) in enumerate(module3._modules.items()):
                        # print name2,name3,module3
                        if isinstance(module4, torch.nn.modules.conv.Conv2d):
                            branchs[branch][0] = module4.out_channels
                            if (name1, name2, name3, name4) in prune_target.keys():
                                branchs[branch][3] = prune_target[(name1, name2, name3, name4)]  # prune index
                            else:
                                branchs[branch][3] = []
                            # print name1,name2,name3, branchs[branch]
                            module4 = prune_conv_weight(module4, branchs[branch][2], branchs[branch][3])
                            branchs[branch][2] = branchs[branch][3]  # prepruned index <- prune index
                            branchs[branch][1] = module4.out_channels  # channel_now <- out_channels
                            # print (name1,name2,name3, name4), module4.weight.data.size(), module4
                            # print branchs[branch]
                            # print '--' * 20
                        elif isinstance(module4, torch.nn.modules.BatchNorm2d):
                            module4 = prune_bn_weight(module4, branchs[branch][2])
                            # print (name1,name2,name3, name4), module4.weight.data.size(), module4
                            # print branchs[branch]
                            # print '--' * 20

                ori_out_channels = [branchs['branch0'][0], branchs['branch1'][0], branchs['branch2'][0]]
                channels_now = branchs['branch0'][1] + branchs['branch1'][1] + branchs['branch2'][1]
                prune_indexs = [branchs['branch0'][3], branchs['branch1'][3], branchs['branch2'][3]]
                prepruned_index = concat_index(ori_out_channels, prune_indexs)

                for k, (name3, module3) in enumerate(module2._modules.items()):
                    # conv2d after concat
                    if isinstance(module3, nn.Conv2d):
                        prune_index = repeat35_prune
                        module3 = prune_conv_weight(module3, prepruned_index, prune_index)
                    elif isinstance(module3, nn.BatchNorm2d):
                        module3 = prune_bn_weight(module3, repeat35_prune)
            # after repeat
            prepruned_index = repeat35_prune
            channels_now = repeat35_channels_now
            # print len(prepruned_index), channels_now,repeat35_ori_channels

        # Mixed_6a
        if name1 == 'mixed_6a':
            branchs = {'branch0': [0, channels_now, prepruned_index, []],
                       'branch1': [0, channels_now, prepruned_index, []],
                       'branch2': [repeat35_ori_channels, channels_now, prepruned_index, prepruned_index]  # pooling layer follows previous layer
                       }  #  to store ori_out_channel, channel_now, prepruned_index, prune_index in each branch
            # branch...
            for j, (name2, module2) in enumerate(module1._modules.items()):
                if name2.startswith('branch0'):
                    branch = 'branch0'
                elif name2.startswith('branch1'):
                    branch = 'branch1'
                elif name2.startswith('branch2'):
                    branch = 'branch2'
                else:
                    raise ValueError('"%s" not in Mixed_6a block' % name2)
                for k, (name3, module3) in enumerate(module2._modules.items()):
                    # print name2,name3
                    if isinstance(module3, torch.nn.modules.conv.Conv2d):
                        branchs[branch][0] = module3.out_channels
                        if (name1, name2, name3) in prune_target.keys():
                            branchs[branch][3] = prune_target[(name1, name2 , name3)]   # prune index
                        else:
                            branchs[branch][3] = []
                        module3 = prune_conv_weight(module3, branchs[branch][2], branchs[branch][3])
                        branchs[branch][2] = branchs[branch][3]  # prepruned index <- prune index
                        branchs[branch][1] = module3.out_channels  # channel_now <- out_channels
                    elif isinstance(module3, torch.nn.modules.BatchNorm2d):
                        module3 = prune_bn_weight(module3, branchs[branch][2])

            ori_out_channels = [branchs['branch0'][0], branchs['branch1'][0], branchs['branch2'][0]]
            channels_now = branchs['branch0'][1] + branchs['branch1'][1] + branchs['branch2'][1]
            prune_indexs = [branchs['branch0'][3], branchs['branch1'][3], branchs['branch2'][3]]
            prepruned_index = concat_index(ori_out_channels, prune_indexs)

        #repeat Block17
        # Block17
        if name1 == 'repeat_block17':
            repeat17_prune = prepruned_index
            repeat17_channels_now = channels_now
            repeat17_ori_channels = channels_now + len(prepruned_index)
            # each block
            for j, (name2, module2) in enumerate(module1._modules.items()):
                branchs = {'branch0': [0, channels_now, repeat17_prune, []],
                           'branch1': [0, channels_now, repeat17_prune, []],
                           }  # to store ori_out_channel, channel_now, prepruned_index, prune_index in each branch
                # branch...
                for k, (name3, module3) in enumerate(module2._modules.items()):
                    # skip now
                    if isinstance(module3, nn.Conv2d) or isinstance(module3, nn.ReLU):
                        continue
                    if name3.startswith('branch0'):
                        branch = 'branch0'
                    elif name3.startswith('branch1'):
                        branch = 'branch1'
                    else:
                        raise ValueError('"%s" not in Block17 block' % name3)
                    for l, (name4, module4) in enumerate(module3._modules.items()):
                        # print name2,name3,module3
                        if isinstance(module4, torch.nn.modules.conv.Conv2d):
                            branchs[branch][0] = module4.out_channels
                            if (name1, name2, name3, name4) in prune_target.keys():
                                branchs[branch][3] = prune_target[(name1, name2, name3, name4)]  # prune index
                            else:
                                branchs[branch][3] = []
                            # print name1,name2,name3, branchs[branch]
                            module4 = prune_conv_weight(module4, branchs[branch][2], branchs[branch][3])
                            branchs[branch][2] = branchs[branch][3]  # prepruned index <- prune index
                            branchs[branch][1] = module4.out_channels  # channel_now <- out_channels
                            # print (name1,name2,name3, name4), module4.weight.data.size(), module4
                            # print branchs[branch]
                            # print '--' * 20
                        elif isinstance(module4, torch.nn.modules.BatchNorm2d):
                            module4 = prune_bn_weight(module4, branchs[branch][2])
                            # print (name1,name2,name3, name4), module4.weight.data.size(), module4
                            # print branchs[branch]
                            # print '--' * 20
                ori_out_channels = [branchs['branch0'][0], branchs['branch1'][0]]
                channels_now = branchs['branch0'][1] + branchs['branch1'][1]
                prune_indexs = [branchs['branch0'][3], branchs['branch1'][3]]
                prepruned_index = concat_index(ori_out_channels, prune_indexs)

                for k, (name3, module3) in enumerate(module2._modules.items()):
                    # conv2d after concat
                    if isinstance(module3, nn.Conv2d):
                        prune_index = repeat17_prune
                        module3 = prune_conv_weight(module3, prepruned_index, prune_index)
                    elif isinstance(module3, nn.BatchNorm2d):
                        module3 = prune_bn_weight(module3, repeat17_prune)
            # after repeat
            prepruned_index = repeat17_prune
            channels_now = repeat17_channels_now
            # print len(prepruned_index), channels_now,repeat17_ori_channels

        # Mixed_7a
        if name1 == 'mixed_7a':
            branchs = {'branch0': [0, channels_now, prepruned_index, []],
                       'branch1': [0, channels_now, prepruned_index, []],
                       'branch2': [0, channels_now, prepruned_index, []],
                       'branch3': [repeat17_ori_channels, channels_now, prepruned_index, prepruned_index]  # pooling layer follows previous layer
                       }  #  to store ori_out_channel, channel_now, prepruned_index, prune_index in each branch
            # branch...
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
                    raise ValueError('"%s" not in Mixed_7a block' % name2)
                for k, (name3, module3) in enumerate(module2._modules.items()):
                    # print name2,name3
                    if isinstance(module3, torch.nn.modules.conv.Conv2d):
                        branchs[branch][0] = module3.out_channels
                        if (name1, name2, name3) in prune_target.keys():
                            branchs[branch][3] = prune_target[(name1, name2 , name3)]   # prune index
                        else:
                            branchs[branch][3] = []
                        module3 = prune_conv_weight(module3, branchs[branch][2], branchs[branch][3])
                        branchs[branch][2] = branchs[branch][3]  # prepruned index <- prune index
                        branchs[branch][1] = module3.out_channels  # channel_now <- out_channels
                    elif isinstance(module3, torch.nn.modules.BatchNorm2d):
                        module3 = prune_bn_weight(module3, branchs[branch][2])

            ori_out_channels = [branchs['branch0'][0], branchs['branch1'][0], branchs['branch2'][0],branchs['branch3'][0]]
            channels_now = branchs['branch0'][1] + branchs['branch1'][1] + branchs['branch2'][1] + branchs['branch3'][1]
            prune_indexs = [branchs['branch0'][3], branchs['branch1'][3], branchs['branch2'][3],branchs['branch3'][3]]
            prepruned_index = concat_index(ori_out_channels, prune_indexs)
            # print ori_out_channels,channels_now,len(prepruned_index)

        #repeat Block8
        if name1 == 'repeat_block8':
            repeat8_prune = prepruned_index
            repeat8_channels_now = channels_now
            repeat8_ori_channels = channels_now + len(prepruned_index)
            # each block
            for j, (name2, module2) in enumerate(module1._modules.items()):
                branchs = {'branch0': [0, channels_now, repeat17_prune, []],
                           'branch1': [0, channels_now, repeat17_prune, []],
                           }  # to store ori_out_channel, channel_now, prepruned_index, prune_index in each branch
                # branch...
                for k, (name3, module3) in enumerate(module2._modules.items()):
                    # skip now
                    if isinstance(module3, nn.Conv2d) or isinstance(module3, nn.ReLU):
                        continue
                    if name3.startswith('branch0'):
                        branch = 'branch0'
                    elif name3.startswith('branch1'):
                        branch = 'branch1'
                    else:
                        raise ValueError('"%s" not in Block8 block' % name3)
                    for l, (name4, module4) in enumerate(module3._modules.items()):
                        # print name2,name3,module3
                        if isinstance(module4, torch.nn.modules.conv.Conv2d):
                            branchs[branch][0] = module4.out_channels
                            if (name1, name2, name3, name4) in prune_target.keys():
                                branchs[branch][3] = prune_target[(name1, name2, name3, name4)]  # prune index
                            else:
                                branchs[branch][3] = []
                            # print name1,name2,name3, branchs[branch]
                            module4 = prune_conv_weight(module4, branchs[branch][2], branchs[branch][3])
                            branchs[branch][2] = branchs[branch][3]  # prepruned index <- prune index
                            branchs[branch][1] = module4.out_channels  # channel_now <- out_channels
                        elif isinstance(module4, torch.nn.modules.BatchNorm2d):
                            module4 = prune_bn_weight(module4, branchs[branch][2])

                ori_out_channels = [branchs['branch0'][0], branchs['branch1'][0]]
                channels_now = branchs['branch0'][1] + branchs['branch1'][1]
                prune_indexs = [branchs['branch0'][3], branchs['branch1'][3]]
                prepruned_index = concat_index(ori_out_channels, prune_indexs)

                for k, (name3, module3) in enumerate(module2._modules.items()):
                    # conv2d after concat
                    if isinstance(module3, nn.Conv2d):
                        prune_index = repeat8_prune
                        module3 = prune_conv_weight(module3, prepruned_index, prune_index)
                    elif isinstance(module3, nn.BatchNorm2d):
                        module3 = prune_bn_weight(module3, repeat8_prune)
            # after repeat
            prepruned_index = repeat8_prune
            channels_now = repeat8_channels_now
            # print len(prepruned_index), channels_now,repeat8_ori_channels

        # a single Block8
        if name1 == 'block8':
            block8_ori_channels = channels_now + len(prepruned_index)
            block8_prune = prepruned_index
            block8_channels_now = channels_now

            branchs = {'branch0': [0, channels_now, prepruned_index, []],
                       'branch1': [0, channels_now, prepruned_index, []],
                       }  # to store ori_out_channel, channel_now, prepruned_index, prune_index in each branch
            # branch...
            for j, (name2, module2) in enumerate(module1._modules.items()):
                # skip now
                if isinstance(module2, nn.Conv2d) or isinstance(module2, nn.ReLU):
                    continue
                if name2.startswith('branch0'):
                    branch = 'branch0'
                elif name2.startswith('branch1'):
                    branch = 'branch1'
                else:
                    raise ValueError('"%s" not in Block8 block' % name2)
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

            ori_out_channels = [branchs['branch0'][0], branchs['branch1'][0]]
            channels_now = branchs['branch0'][1] + branchs['branch1'][1]
            prune_indexs = [branchs['branch0'][3], branchs['branch1'][3]]
            prepruned_index = concat_index(ori_out_channels, prune_indexs)

            for j, (name2, module2) in enumerate(module1._modules.items()):
                # conv2d after concat
                if isinstance(module2, nn.Conv2d):
                    prune_index = block8_prune
                    module2 = prune_conv_weight(module2, prepruned_index, prune_index)
                elif isinstance(module2, nn.BatchNorm2d):
                    module2 = prune_bn_weight(module2, block8_prune)

            # after block8
            prepruned_index = block8_prune
            channels_now = block8_channels_now
            # print len(prepruned_index), channels_now,block8_ori_channels

        # fc layer
        if name1 == 'fc':
            module1 = prune_fc_weight(module1, prepruned_index)



    return model


if __name__ == '__main__':
    model = torch.load('/home/gserver/zhangchi/channel-prune/IncepResV1/model_prunned.pth').cpu()
    # print model
    # # for x in model.state_dict().keys():
    # #     print x, model.state_dict()[x].size()

    prune_target = {
        ('repeat_block35', '0', 'branch1_1_3x3', 'conv'): [4],
        ('repeat_block35', '4', 'branch1_0_1x1', 'conv'): [0, 23, 24],
        ('repeat_block35', '2', 'branch2_2_3x3', 'conv'): [5, 19, 35],
        ('repeat_block35', '3', 'branch1_1_3x3', 'conv'): [1, 2, 5, 14, 31],
        ('conv2d_4b', 'conv'): [1, 2, 10, 11, 13, 16, 17, 20, 24, 27, 32, 36, 41, 42, 43, 46, 50, 55, 58, 59, 61, 67,
                                235, 236, 237, 241, 243, 247, 248, 251],
        ('repeat_block35', '3', 'branch2_1_3x3', 'conv'): [18, 47],
        ('repeat_block35', '1', 'branch2_2_3x3', 'conv'): [18],
        ('repeat_block35', '0', 'branch2_1_3x3', 'conv'): [20, 41],
        ('repeat_block35', '3', 'branch2_0_1x1', 'conv'): [1, 3, 4, 7, 11, 13, 15, 22],
        ('repeat_block35', '3', 'branch1_0_1x1', 'conv'): [0, 4, 9, 13],
        ('repeat_block35', '4', 'branch0_1x1', 'conv'): [1, 4, 6, 12, 29],
        ('repeat_block35', '0', 'branch2_0_1x1', 'conv'): [0, 10, 13, 16, 21],
        ('repeat_block35', '3', 'branch0_1x1', 'conv'): [1, 6, 11, 16, 18, 24, 26],
        ('repeat_block35', '4', 'branch1_1_3x3', 'conv'): [1, 15],
        ('repeat_block35', '2', 'branch1_1_3x3', 'conv'): [12, 22, 24, 28],
        ('mixed_6a', 'branch0_3x3', 'conv'): [0, 1, 3, 4, 6, 13, 15, 17, 18, 19, 22, 24, 26, 27, 28, 29, 30, 33, 36, 37,
                                              ],
        ('repeat_block35', '1', 'branch2_1_3x3', 'conv'): [40],
        ('repeat_block35', '3', 'branch2_2_3x3', 'conv'): [33],
        ('repeat_block35', '4', 'branch2_2_3x3', 'conv'): [3, 37, 46, 54],
        ('conv2d_3b', 'conv'): [18, 44, 51, 56, 68, 71],
        ('repeat_block35', '1', 'branch2_0_1x1', 'conv'): [2],
        ('mixed_6a', 'branch1_2_3x3', 'conv'): [2, 3, 5, 6, 7, 8, 10, 13, 14, 19, 30, 31, 32, 33, 34, 35, 36, 38, 41,
                                                45, 50, 51, 58, 61, 62, 63, 64, 65, 66, 67, 70, 72, 75, 77, 79, 80, 81,
                                                82, 83, 85, 87, 88, 89, 90, 91, 92, 94, 97],
        ('conv2d_1a', 'conv'): [12],
        ('repeat_block35', '4', 'branch2_1_3x3', 'conv'): [33],
        ('repeat_block35', '4', 'branch2_0_1x1', 'conv'): [6, 8, 9, 12, 13, 18, 19, 25, 28],
        ('repeat_block35', '1', 'branch0_1x1', 'conv'): [17, 21, 23, 28],
        ('repeat_block35', '0', 'branch1_0_1x1', 'conv'): [16, 17],
        ('mixed_6a', 'branch1_1_3x3', 'conv'): [4, 7, 26, 27, 32, 37, 39, 55, 62, 85, 92, 108, 115, 118, 122, 126, 130,
                                                135, 138, 149, 151, 152, 162, 164, 174, 179, 180, 181, 189],
        ('repeat_block35', '0', 'branch2_2_3x3', 'conv'): [5, 49],
        ('repeat_block35', '1', 'branch1_1_3x3', 'conv'): [15, 17],
        ('repeat_block35', '0', 'branch0_1x1', 'conv'): [11, 19],
        ('mixed_6a', 'branch1_0_1x1', 'conv'): [9, 15, 16, 17, 20, 21, 30, 37, 45, 49, 58, 59, 85, 89, 93, 100, 102,
                                                106, 107, 108, 110, 112, 119, 120, 122, 137, 139, 148, 150, 151, 152,
                                                154, 155, 158, 164, 166, 168, 173, 177, 178, 179, 181, 182, 183, 186,
                                                188],
        ('conv2d_2b', 'conv'): [13],
        ('conv2d_2a', 'conv'): [6, 27],
        ('repeat_block35', '1', 'branch1_0_1x1', 'conv'): [1, 5, 8, 27],
        ('repeat_block35', '2', 'branch1_0_1x1', 'conv'): [1, 5, 7, 10, 17, 20, 27],
        ('repeat_block35', '2', 'branch2_0_1x1', 'conv'): [11, 14, 21, 27],
        ('repeat_block35', '2', 'branch0_1x1', 'conv'): [4, 6, 15, 20, 21, 25, 27],

    }
    model = prune_incep3_conv_layer(model, prune_target).cuda()
    print model
    # Make sure all the layers are trainable
    for param in model.parameters():
        param.requires_grad = True
    model.train()
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

    #
    # x = torch.FloatTensor(10, 3, 112, 96)
    # x = torch.autograd.Variable(x)
    # y = model(x)
    # print y.size()