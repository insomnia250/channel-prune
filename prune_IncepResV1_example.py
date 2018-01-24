import os
import torch
import torchvision.models as models
import torchvision
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import torch.utils.data as torchdata
from torch.optim import lr_scheduler

from models.inception_res_v1 import BasicConv2d, Block35, Block17, Block8, \
                         Mixed_6a, Mixed_7a , InceptionResnetV1

from utils.train import train,trainlog
from model_prune.filter_pruner import FilterPruner
from model_prune.prune_utils import *


os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"


class IncepResV1FilterPruner(FilterPruner):
    # Inherited a FilterPruner class

    # define a custom forward function in which hooks are registered to calculate grads
    def forward(self, x):
        self.activations = []
        activation_index = 0

        # first 5 conv layers
        for i, (name1, module1) in enumerate(self.model._modules.items()):
            if i >= 7: break
            if name1 == 'maxpool_3a' or name1 == 'maxpool_5a':
                x = module1(x)
            for j, (name2, module2) in enumerate(module1._modules.items()):
                x = module2(x)
                if isinstance(module2, torch.nn.modules.conv.Conv2d):
                    self.attach_action_and_hook(x, activation_index, (name1,name2))
                    activation_index += 1

        for i, (name1, module1) in enumerate(self.model._modules.items()):
            if i < 7: continue
            # Mixed_6a block
            elif isinstance(module1, Mixed_6a):
                branchs = {}
                branchs['branch0'] = x
                branchs['branch1'] = x
                branchs['branch2'] = x
                for j, (name2, module2) in enumerate(module1._modules.items()):
                    if name2.startswith('branch0'):
                        branch = 'branch0'
                    elif name2.startswith('branch1'):
                        branch = 'branch1'
                    elif name2.startswith('branch2'):
                        branch = 'branch2'
                    else:
                        raise ValueError('"%s" not in Mixed_6a block' % name2)
                    if isinstance(module2, nn.MaxPool2d):
                        branchs[branch] = module2(branchs[branch])
                    for k, (name3, module3) in enumerate(module2._modules.items()):
                        branchs[branch] = module3(branchs[branch])
                        if isinstance(module3, torch.nn.modules.conv.Conv2d):
                            self.attach_action_and_hook(branchs[branch], activation_index, (name1, name2, name3))
                            activation_index += 1
                # concat
                x = torch.cat([branchs['branch0'], branchs['branch1'], branchs['branch2']], 1)

            # Mixed_7a
            elif isinstance(module1, Mixed_7a):
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
                        raise ValueError('"%s" not in Mixed_7a block' % name2)
                    if isinstance(module2, nn.MaxPool2d):
                        branchs[branch] = module2(branchs[branch])
                    for k, (name3, module3) in enumerate(module2._modules.items()):
                        branchs[branch] = module3(branchs[branch])
                        if isinstance(module3, torch.nn.modules.conv.Conv2d):
                            self.attach_action_and_hook(branchs[branch], activation_index, (name1, name2, name3))
                            activation_index += 1
                # concat
                x = torch.cat([branchs['branch0'], branchs['branch1'], branchs['branch2'],branchs['branch3']], 1)

            # Squential repeat Block35
            elif isinstance(module1, nn.Sequential) and (name1=='repeat_block35'):
                # each Block35
                for j, (name2, module2) in enumerate(module1._modules.items()):
                    branchs = {}
                    branchs['branch0'] = x
                    branchs['branch1'] = x
                    branchs['branch2'] = x
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
                            branchs[branch] = module4(branchs[branch])
                            if isinstance(module4, torch.nn.modules.conv.Conv2d):
                                self.attach_action_and_hook(branchs[branch], activation_index, (name1, name2, name3, name4))
                                activation_index += 1
                    # after all branchs
                    out = torch.cat([branchs['branch0'], branchs['branch1'], branchs['branch2']],dim=1)
                    for k, (name3, module3) in enumerate(module2._modules.items()):
                        if isinstance(module3, nn.Conv2d):
                            out = module3(out)
                            x = x + module2.scale * out
                        elif isinstance(module3, nn.ReLU):
                            x = module3(x)

            # Squential repeat Block17
            elif isinstance(module1, nn.Sequential) and (name1=='repeat_block17'):
                # each Block17
                for j, (name2, module2) in enumerate(module1._modules.items()):
                    branchs = {}
                    branchs['branch0'] = x
                    branchs['branch1'] = x
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
                            branchs[branch] = module4(branchs[branch])
                            if isinstance(module4, torch.nn.modules.conv.Conv2d):
                                self.attach_action_and_hook(branchs[branch], activation_index, (name1, name2, name3, name4))
                                activation_index += 1
                    # after all branchs
                    out = torch.cat([branchs['branch0'], branchs['branch1']],dim=1)
                    for k, (name3, module3) in enumerate(module2._modules.items()):
                        if isinstance(module3, nn.Conv2d):
                            out = module3(out)
                            x = x + module2.scale * out
                        elif isinstance(module3, nn.ReLU):
                            x = module3(x)

            # Squential repeat Block8
            elif isinstance(module1, nn.Sequential) and (name1=='repeat_block8'):
                # each Block17
                for j, (name2, module2) in enumerate(module1._modules.items()):
                    branchs = {}
                    branchs['branch0'] = x
                    branchs['branch1'] = x
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
                            branchs[branch] = module4(branchs[branch])
                            if isinstance(module4, torch.nn.modules.conv.Conv2d):
                                self.attach_action_and_hook(branchs[branch], activation_index, (name1, name2, name3, name4))
                                activation_index += 1
                    # after all branchs
                    out = torch.cat([branchs['branch0'], branchs['branch1']],dim=1)
                    for k, (name3, module3) in enumerate(module2._modules.items()):
                        if isinstance(module3, nn.Conv2d):
                            out = module3(out)
                            x = x + module2.scale * out
                        elif isinstance(module3, nn.ReLU):
                            x = module3(x)

            # a Blcok8 (no Relu)
            elif isinstance(module1, Block8):
                branchs = {}
                branchs['branch0'] = x
                branchs['branch1'] = x
                for k, (name2, module2) in enumerate(module1._modules.items()):
                    # skip now
                    if isinstance(module2, nn.Conv2d) or isinstance(module2, nn.ReLU):
                        continue
                    if name2.startswith('branch0'):
                        branch = 'branch0'
                    elif name2.startswith('branch1'):
                        branch = 'branch1'
                    else:
                        raise ValueError('"%s" not in Block8 block' % name2)
                    for l, (name3, module3) in enumerate(module2._modules.items()):
                        branchs[branch] = module3(branchs[branch])
                        if isinstance(module3, torch.nn.modules.conv.Conv2d):
                            self.attach_action_and_hook(branchs[branch], activation_index, (name1, name2, name3))
                            activation_index += 1
                # after all branchs
                out = torch.cat([branchs['branch0'], branchs['branch1']],dim=1)
                for k, (name2, module2) in enumerate(module1._modules.items()):
                    if isinstance(module2, nn.Conv2d):
                        out = module2(out)
                        x = x + module1.scale * out
                        # no Relu

            # avpool layer
            elif name1 in ['avgpool_1a', 'dropout', 'fc']:
                x = module1(x)
                if name1 == 'avgpool_1a':
                    x = x.view(x.size(0), -1)

        return x

    # define a custom prune function to prune filters (functions in prune_utils may be helpful)
    def prune_conv_layer(self,model, prune_target):
        channels_now = 3
        prepruned_index = []

        for i, (name1, module1) in enumerate(model._modules.items()):
            if i >= 7: continue
            for j, (name2, module2) in enumerate(module1._modules.items()):
                if isinstance(module2, torch.nn.modules.conv.Conv2d):
                    # # to prune
                    if (name1, name2) in prune_target.keys():
                        prune_index = prune_target[(name1, name2)]
                    else:
                        prune_index = []
                    module2 = prune_conv_weight(module2, prepruned_index, prune_index)
                    prepruned_index = prune_index
                    channels_now = module2.out_channels

                elif isinstance(module2, torch.nn.modules.BatchNorm2d):
                    module2 = prune_bn_weight(module2, prepruned_index)

        repeat35_prune = prepruned_index
        repeat35_channels_now = channels_now
        repeat35_ori_channels = channels_now + len(prepruned_index)
        for i, (name1, module1) in enumerate(model._modules.items()):
            if (i < 7): continue
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
                            if isinstance(module4, torch.nn.modules.conv.Conv2d):
                                branchs[branch][0] = module4.out_channels
                                if (name1, name2, name3, name4) in prune_target.keys():
                                    branchs[branch][3] = prune_target[(name1, name2, name3, name4)]  # prune index
                                else:
                                    branchs[branch][3] = []

                                module4 = prune_conv_weight(module4, branchs[branch][2], branchs[branch][3])
                                branchs[branch][2] = branchs[branch][3]  # prepruned index <- prune index
                                branchs[branch][1] = module4.out_channels  # channel_now <- out_channels
                            elif isinstance(module4, torch.nn.modules.BatchNorm2d):
                                module4 = prune_bn_weight(module4, branchs[branch][2])

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

            # Mixed_6a
            if name1 == 'mixed_6a':
                branchs = {'branch0': [0, channels_now, prepruned_index, []],
                           'branch1': [0, channels_now, prepruned_index, []],
                           'branch2': [repeat35_ori_channels, channels_now, prepruned_index, prepruned_index]
                           # pooling layer follows previous layer
                           }  # to store ori_out_channel, channel_now, prepruned_index, prune_index in each branch
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
                        if isinstance(module3, torch.nn.modules.conv.Conv2d):
                            branchs[branch][0] = module3.out_channels
                            if (name1, name2, name3) in prune_target.keys():
                                branchs[branch][3] = prune_target[(name1, name2, name3)]  # prune index
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

            # repeat Block17
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
                            if isinstance(module4, torch.nn.modules.conv.Conv2d):
                                branchs[branch][0] = module4.out_channels
                                if (name1, name2, name3, name4) in prune_target.keys():
                                    branchs[branch][3] = prune_target[(name1, name2, name3, name4)]  # prune index
                                else:
                                    branchs[branch][3] = []

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
                            prune_index = repeat17_prune
                            module3 = prune_conv_weight(module3, prepruned_index, prune_index)
                        elif isinstance(module3, nn.BatchNorm2d):
                            module3 = prune_bn_weight(module3, repeat17_prune)
                # after repeat
                prepruned_index = repeat17_prune
                channels_now = repeat17_channels_now

            # Mixed_7a
            if name1 == 'mixed_7a':
                branchs = {'branch0': [0, channels_now, prepruned_index, []],
                           'branch1': [0, channels_now, prepruned_index, []],
                           'branch2': [0, channels_now, prepruned_index, []],
                           'branch3': [repeat17_ori_channels, channels_now, prepruned_index, prepruned_index]
                           # pooling layer follows previous layer
                           }  # to store ori_out_channel, channel_now, prepruned_index, prune_index in each branch
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
                        if isinstance(module3, torch.nn.modules.conv.Conv2d):
                            branchs[branch][0] = module3.out_channels
                            if (name1, name2, name3) in prune_target.keys():
                                branchs[branch][3] = prune_target[(name1, name2, name3)]  # prune index
                            else:
                                branchs[branch][3] = []
                            module3 = prune_conv_weight(module3, branchs[branch][2], branchs[branch][3])
                            branchs[branch][2] = branchs[branch][3]  # prepruned index <- prune index
                            branchs[branch][1] = module3.out_channels  # channel_now <- out_channels
                        elif isinstance(module3, torch.nn.modules.BatchNorm2d):
                            module3 = prune_bn_weight(module3, branchs[branch][2])

                ori_out_channels = [branchs['branch0'][0], branchs['branch1'][0], branchs['branch2'][0],
                                    branchs['branch3'][0]]
                channels_now = branchs['branch0'][1] + branchs['branch1'][1] + branchs['branch2'][1] + \
                               branchs['branch3'][1]
                prune_indexs = [branchs['branch0'][3], branchs['branch1'][3], branchs['branch2'][3],
                                branchs['branch3'][3]]
                prepruned_index = concat_index(ori_out_channels, prune_indexs)

            # repeat Block8
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
                            if isinstance(module4, torch.nn.modules.conv.Conv2d):
                                branchs[branch][0] = module4.out_channels
                                if (name1, name2, name3, name4) in prune_target.keys():
                                    branchs[branch][3] = prune_target[(name1, name2, name3, name4)]  # prune index
                                else:
                                    branchs[branch][3] = []

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
                        if isinstance(module3, torch.nn.modules.conv.Conv2d):
                            branchs[branch][0] = module3.out_channels
                            if (name1, name2, name3) in prune_target.keys():
                                branchs[branch][3] = prune_target[(name1, name2, name3)]  # prune index
                            else:
                                branchs[branch][3] = []

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

            # fc layer
            if name1 == 'fc':
                module1 = prune_fc_weight(module1, prepruned_index)

        return model

if __name__ == '__main__':
    isTrain = False
    isPrune = True
    # train_path = '/home/gserver/zhangchi/channel-prune/data/train1'
    train_path = '/media/gserver/data/catavsdog/train1'
    test_path = '/media/gserver/data/catavsdog/test1'

    usecuda = True
    start_epoch = 0
    epoch_num = 2
    save_inter = 5
    batch_size = 48

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    dataset = {'train': torchvision.datasets.ImageFolder(train_path, data_transforms['train']),
               'val': torchvision.datasets.ImageFolder(test_path, data_transforms['val'])}

    data_loader = {'train': torchdata.DataLoader(dataset['train'], batch_size, num_workers=4,
                                                 shuffle=True, pin_memory=True),
                   'val': torchdata.DataLoader(dataset['val'], 24, num_workers=4,
                                               shuffle=False, pin_memory=True)}
    criterion = nn.CrossEntropyLoss()


    model = InceptionResnetV1(num_classes=2)
    fine_tuner = IncepResV1FilterPruner(model=model,
                                   train_dataloader=data_loader['train'],
                                   test_dataloader=data_loader['val'],
                                   criterion=criterion,
                                   useCuda=True)
    fine_tuner.perform_prune(save_dir='./saved_models/IncepResV1-test')