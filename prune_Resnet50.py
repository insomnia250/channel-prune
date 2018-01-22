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
def prune_Resnet50_conv_layer(model, prune_target):
    channels_now = 3
    prepruned_index = []

    for i, (name1, module1) in enumerate(model._modules.items()):
        if i>=4:break
        if isinstance(module1, nn.Conv2d):
            # # to prune
            if (name1, ) in prune_target.keys():
                prune_index = prune_target[(name1, )]
            else:
                prune_index = []
            module1 = prune_conv_weight(module1, prepruned_index, prune_index)
            prepruned_index = prune_index
            channels_now = module1.out_channels
        elif isinstance(module1, nn.BatchNorm2d):
            module1 = prune_bn_weight(module1, prepruned_index)

    for i, (name1, module1) in enumerate(model._modules.items()):
        if name1 in ['layer1','layer2','layer3','layer4']:
            repeat1_prune = prepruned_index
            repeat1_channels_now = channels_now
            for j, (name2, module2) in enumerate(module1._modules.items()):
                # if name2 != '0':continue

                if name2 == '0' and (name1, name2, 'sum') in prune_target.keys():
                    prune_target[(name1, name2, 'conv3')] = prune_target[(name1, name2, 'sum')]
                    prune_target[(name1, name2, 'downsample', '0')] = prune_target[(name1, name2, 'sum')]
                elif name2 !='0':
                    prune_target[(name1, name2, 'conv3')] = repeat1_prune

                # print name1,name2,'=='*20
                branchs = {'branch_conv': [0, repeat1_channels_now, repeat1_prune, []],
                           'branch_x': [0, repeat1_channels_now, repeat1_prune, []]
                           }  # to store ori_out_channel, channel_now, prepruned_index, prune_index in each branch
                # branch...
                for k, (name3, module3) in enumerate(module2._modules.items()):
                    if isinstance(module3, nn.Conv2d):
                        branchs['branch_conv'][0] = module3.out_channels
                        if (name1,name2,name3) in prune_target.keys():
                            branchs['branch_conv'][3] = prune_target[(name1, name2, name3)]
                        else:
                            branchs['branch_conv'][3] = []
                        module3 = prune_conv_weight(module3, branchs['branch_conv'][2], branchs['branch_conv'][3])
                        branchs['branch_conv'][2] = branchs['branch_conv'][3]  # prepruned index <- prune index
                        branchs['branch_conv'][1] = module3.out_channels  # channel_now <- out_channels
                        # print (name1,name2,name3), module3.weight.data.size(), module3
                        # print branchs['branch_conv']
                        # print '--' * 20
                    elif isinstance(module3, nn.BatchNorm2d):
                        module3 = prune_bn_weight(module3, branchs['branch_conv'][2])
                        # print (name1,name2,name3), module3.weight.data.size(), module3
                        # print branchs['branch_conv']
                        # print '--' * 20
                    elif (name3 == 'downsample') and isinstance(module3, nn.Sequential):
                        for l, (name4, module4) in enumerate(module3._modules.items()):
                            if isinstance(module4, nn.Conv2d):
                                branchs['branch_x'][0] = module4.out_channels
                                if (name1, name2, 'sum') in prune_target.keys():
                                    branchs['branch_x'][3] = prune_target[(name1, name2, 'sum')]
                                else:
                                    branchs['branch_x'][3] = []
                                module4 = prune_conv_weight(module4, branchs['branch_x'][2], branchs['branch_x'][3])
                                branchs['branch_x'][2] = branchs['branch_x'][3]  # prepruned index <- prune index
                                branchs['branch_x'][1] = module4.out_channels  # channel_now <- out_channels
                            elif isinstance(module4, nn.BatchNorm2d):
                                module4 = prune_bn_weight(module4, branchs['branch_x'][2])

                if name2 == '0':
                    repeat1_prune = branchs['branch_x'][2]
                    repeat1_channels_now = branchs['branch_x'][1]

            # after this layer
            prepruned_index = repeat1_prune
            channels_now = repeat1_channels_now

        if name1 == 'fc':
            module1 = prune_fc_weight(module1,prepruned_index)

    return model


if __name__ == '__main__':
    model = models.resnet50(num_classes=2)
    model.load_state_dict(torch.load('./saved_models/Resnet50/weights-1-[0.9898].pth'))
    model = model.cuda()
    # print model
    # # for x in model.state_dict().keys():
    # #     print x, model.state_dict()[x].size()

    prune_target = {
        ('layer3', '5', 'conv1'): [181, 74, 216],
        ('layer3', '4', 'conv1'): [157, 163],
        ('layer4', '2', 'conv2'): [458, 215, 220, 84, 104, 282, 474, 410, 346, 479, 256, 496, 323, 259, 349, 294, 106,
                                   451, 5, 117, 490, 139, 419, 81, 188, 481, 132, 395, 125, 50, 340, 15, 181, 402],
        ('layer3', '5', 'conv2'): [87, 210, 37, 214, 40, 209, 9, 231, 8],
        ('layer4', '0', 'sum'): [112, 413, 1639, 1731, 1881, 1125, 1799, 849, 1329, 627, 2047, 772, 1574, 207, 1062,
                                 1663, 594, 622, 855, 1192, 1, 1407, 261, 1726, 231, 976, 602, 321, 1842, 930, 993, 721,
                                 241, 283, 1725, 342, 1014, 1773, 832, 1599, 468, 244, 1932, 1820, 791, 545, 955, 1235,
                                 500, 1713, 994, 1999, 1622, 1070, 401, 1302, 1641, 480, 61, 1067, 1733, 105, 1895,
                                 1317, 1928, 206, 2029, 224, 1763, 154, 2046, 1156, 77, 422, 759, 689, 1942, 967, 1017],
        ('layer2', '0', 'conv1'): [84, 61, 115],
        ('layer3', '2', 'conv1'): [72, 170, 189, 171],
        ('layer4', '2', 'conv1'): [67, 308, 151, 295, 470, 36, 303, 8, 467, 376, 282, 43, 226, 7, 34, 21, 27, 456, 421,
                                   298, 326, 65, 291, 42, 12, 144, 83, 323, 289, 366, 159, 334, 136, 149, 45, 44, 299,
                                   348, 169, 254, 466, 11, 49, 164, 255, 267, 251, 476, 430, 227, 257, 270, 202, 440,
                                   246, 165, 454, 428, 288, 152, 127, 57, 390, 209, 333, 313, 129],
        ('layer3', '0', 'sum'): [7, 76, 85, 157, 201, 209, 227, 287, 291, 308, 316, 322, 342, 343, 367, 397, 405, 437,
                                 458, 470, 605, 623, 740, 750, 797, 827, 839, 854, 866, 873, 882, 915, 916, 924, 936,
                                 941, 969, 985, 1002, 147, 811, 432, 382, 1003, 98, 895, 697, 126, 877, 359, 788, 860,
                                 467, 43, 17, 896, 715, 463, 628, 235, 368, 949, 302, 596, 910, 893, 553, 977, 531, 101,
                                 593, 714, 559, 350, 324, 360, 19, 542, 849, 639],
        ('layer2', '3', 'conv2'): [3],
        ('layer2', '2', 'conv2'): [123, 33, 119],
        ('layer4', '1', 'conv2'): [455, 506, 453, 104, 124, 432, 393, 295, 436, 270, 273, 332, 243, 498, 349, 350, 247,
                                   494, 3, 175, 253, 311, 425, 287, 305, 266],
        ('layer3', '2', 'conv2'): [48, 182, 25, 113, 111, 108, 226, 240, 43, 44, 66, 172, 131],
        ('layer4', '0', 'conv1'): [211, 447, 101, 104, 380, 283, 294, 396, 66, 439, 343, 243, 207, 494, 264, 450, 327],
        ('layer4', '1', 'conv1'): [435, 87, 351, 455, 254, 337, 225, 323, 424, 493, 375],
        ('layer1', '0', 'conv1'): [7, 23, 27, 56, 58],
        ('layer4', '0', 'conv2'): [444, 29, 156, 183, 42, 422, 391, 509, 417, 7, 222, 9, 393, 237],
        ('layer3', '3', 'conv2'): [204, 255, 117],
        ('layer2', '0', 'sum'): [2, 5, 14, 23, 28, 59, 68, 89, 90, 94, 105, 112, 133, 138, 143, 153, 157, 165, 170, 174,
                                 186, 196, 200, 204, 213, 222, 234, 236, 240, 270, 272, 291, 297, 299, 311, 312, 313,
                                 318, 325, 332, 346, 364, 370, 386, 400, 409, 413, 426, 430, 433, 446, 449, 451, 452,
                                 454, 467, 472, 486, 498, 509, 51, 38, 209, 96, 320, 429],
        ('layer1', '0', 'conv2'): [0, 19, 59, 61, 2],
        ('layer3', '3', 'conv1'): [158, 245, 75, 164],
        ('layer1', '1', 'conv2'): [18, 62],
        ('conv1',): [13],
        ('layer3', '0', 'conv2'): [22, 72],
        ('layer3', '1', 'conv1'): [139, 71],
        ('layer1', '1', 'conv1'): [13, 45, 47, 56],
        ('layer3', '1', 'conv2'): [242, 164, 130, 55, 183, 182],
        ('layer1', '0', 'sum'): [22, 23, 25, 45, 52, 64, 71, 76, 83, 87, 89, 101, 107, 117, 123, 127, 133, 134, 174,
                                 185, 199, 202, 204, 214, 246, 93, 154, 230, 94, 48, 207, 164, 139],
        ('layer2', '1', 'conv2'): [10, 61, 86, 120],
        ('layer2', '0', 'conv2'): [18],
        ('layer3', '4', 'conv2'): [110, 238, 211, 168, 0, 204, 132, 57],

    }

    model = prune_Resnet50_conv_layer(model, prune_target).cpu()

    # print model
    # # Make sure all the layers are trainable
    # for param in model.parameters():
    #     param.requires_grad = True
    # model.train()
    # optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    # criterion = torch.nn.CrossEntropyLoss()
    #
    # x = torch.FloatTensor(5, 3, 299, 299)
    # x = torch.autograd.Variable(x.cuda())
    # label = torch.LongTensor([1,0,0,0,1])
    # label = torch.autograd.Variable(label.cuda())
    #
    # output = model(x)
    # print output,label
    # criterion(output, label).backward()
    # optimizer.step()
    #
    # #
    x = torch.FloatTensor(10, 3, 224, 224)
    x = torch.autograd.Variable(x)
    y = model(x)
    print y.size()