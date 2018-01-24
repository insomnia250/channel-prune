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

from torchvision.models.inception import InceptionA, InceptionB, InceptionC, InceptionD, InceptionE, \
    InceptionAux, BasicConv2d


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

        if self.model.transform_input:
            x = x.clone()
            x[:, 0] = x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x[:, 1] = x[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x[:, 2] = x[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5

        # first 5 conv layers
        for i, (name1, module1) in enumerate(self.model._modules.items()):
            if i > 4: break
            for j, (name2, module2) in enumerate(module1._modules.items()):
                x = module2(x)
                if isinstance(module2, torch.nn.modules.conv.Conv2d):
                    self.attach_action_and_hook(x, activation_index, (name1, name2))
                    activation_index += 1
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
                        branchs[branch] = module3(branchs[branch])
                        if isinstance(module3, torch.nn.modules.conv.Conv2d):
                            self.attach_action_and_hook(branchs[branch], activation_index, (name1, name2, name3))
                            activation_index += 1
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
                            self.attach_action_and_hook(branchs[branch], activation_index, (name1, name2, name3))
                            activation_index += 1
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
                        branchs[branch] = module3(branchs[branch])
                        if isinstance(module3, torch.nn.modules.conv.Conv2d):
                            self.attach_action_and_hook(branchs[branch], activation_index, (name1, name2, name3))
                            activation_index += 1
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
                    if name2 in ['branch3x3_1', 'branch3x3_2']:
                        branch = 'branch3x3'
                    elif name2 in ['branch7x7x3_1', 'branch7x7x3_2', 'branch7x7x3_3', 'branch7x7x3_4']:
                        branch = 'branch7x7x3'
                    elif name2 == 'branch_pool':
                        branch = 'branch_pool'
                    else:
                        raise ValueError('"%s" not in InceptionD block' % name2)
                    for k, (name3, module3) in enumerate(module2._modules.items()):
                        branchs[branch] = module3(branchs[branch])
                        if isinstance(module3, torch.nn.modules.conv.Conv2d):
                            self.attach_action_and_hook(branchs[branch], activation_index, (name1, name2, name3))
                            activation_index += 1
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
                            self.attach_action_and_hook(branchs[branch], activation_index, (name1, name2, name3))
                            activation_index += 1
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


    # define a custom prune function to prune filters (functions in prune_utils may be helpful)
    def prune_conv_layer(self,model, prune_target):
        channels_now = 3
        prepruned_index = []

        for i, (name1, module1) in enumerate(model._modules.items()):
            if i >= 5: continue
            for j, (name2, module2) in enumerate(module1._modules.items()):
                # print name1,name2
                if isinstance(module2, torch.nn.modules.conv.Conv2d):
                    # to prune
                    if (name1, name2) in prune_target.keys():
                        prune_index = prune_target[(name1, name2)]
                    else:
                        prune_index = []
                    module2 = prune_conv_weight(module2, prepruned_index, prune_index)
                    prepruned_index = prune_index

                    channels_now = module2.out_channels
                elif isinstance(module2, torch.nn.modules.BatchNorm2d):
                    module2 = prune_bn_weight(module2, prepruned_index)

        for i, (name1, module1) in enumerate(model._modules.items()):
            if (i <= 4): continue

            # if InceptionA
            if isinstance(module1, InceptionA):
                branchs = {'branch5x5': [0, channels_now, prepruned_index, []],
                           'branch1x1': [0, channels_now, prepruned_index, []],
                           'branch3x3dbl': [0, channels_now, prepruned_index, []],
                           'branch_pool': [0, channels_now, prepruned_index, []]
                           }  # to store ori_out_channel, channel_now, prepruned_index, prune_index in each branch
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

                ori_out_channels = [branchs['branch1x1'][0], branchs['branch5x5'][0], branchs['branch3x3dbl'][0],
                                    branchs['branch_pool'][0]]
                channels_now = branchs['branch1x1'][1] + branchs['branch5x5'][1] + branchs['branch3x3dbl'][1] + \
                               branchs['branch_pool'][1]
                prune_indexs = [branchs['branch1x1'][3], branchs['branch5x5'][3], branchs['branch3x3dbl'][3],
                                branchs['branch_pool'][3]]
                prepruned_index = concat_index(ori_out_channels, prune_indexs)

            # if InceptionB
            if isinstance(module1, InceptionB):
                branchs = {'branch3x3': [0, channels_now, prepruned_index, []],
                           'branch3x3dbl': [0, channels_now, prepruned_index, []],
                           'branch_pool': [0, channels_now, prepruned_index, prepruned_index],
                           # this branch has no conv, so out_channels remains as previous layer
                           }  # to store ori_out_channel, channel_now, prepruned_index, prune_index in each branch

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

                            module3 = prune_conv_weight(module3, branchs[branch][2], branchs[branch][3])
                            branchs[branch][2] = branchs[branch][3]  # prepruned index <- prune index
                            branchs[branch][1] = module3.out_channels  # channel_now <- out_channels
                        elif isinstance(module3, torch.nn.modules.BatchNorm2d):
                            module3 = prune_bn_weight(module3, branchs[branch][2])

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
                           }  # to store ori_out_channel, channel_now, prepruned_index, prune_index in each branch
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

                ori_out_channels = [branchs['branch1x1'][0], branchs['branch7x7'][0], branchs['branch7x7dbl'][0],
                                    branchs['branch_pool'][0]]
                channels_now = branchs['branch1x1'][1] + branchs['branch7x7'][1] + branchs['branch7x7dbl'][1] + \
                               branchs['branch_pool'][1]
                prune_indexs = [branchs['branch1x1'][3], branchs['branch7x7'][3], branchs['branch7x7dbl'][3],
                                branchs['branch_pool'][3]]
                prepruned_index = concat_index(ori_out_channels, prune_indexs)

            # if InceptionD
            if isinstance(module1, InceptionD):
                branchs = {'branch3x3': [0, channels_now, prepruned_index, []],
                           'branch7x7x3': [0, channels_now, prepruned_index, []],
                           'branch_pool': [0, channels_now, prepruned_index, prepruned_index],
                           }  # to store ori_out_channel, channel_now, prepruned_index, prune_index in each branch
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
                            module3 = prune_conv_weight(module3, branchs[branch][2], branchs[branch][3])
                            branchs[branch][2] = branchs[branch][3]  # prepruned index <- prune index
                            branchs[branch][1] = module3.out_channels  # channel_now <- out_channels
                        elif isinstance(module3, torch.nn.modules.BatchNorm2d):
                            module3 = prune_bn_weight(module3, branchs[branch][2])

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
                           }  # to store ori_out_channel, channel_now, prepruned_index, prune_index in each branch
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
                ori_out_channels = [branchs['branch1x1'][0], branchs['branch3x3'][0], branchs['branch3x3dbl'][0],
                                    branchs['branch_pool'][0]]
                channels_now = branchs['branch1x1'][1] + branchs['branch3x3'][1] + branchs['branch3x3dbl'][1] + \
                               branchs['branch_pool'][1]
                prune_indexs = [branchs['branch1x1'][3], branchs['branch3x3'][3], branchs['branch3x3dbl'][3],
                                branchs['branch_pool'][3]]
                prepruned_index = concat_index(ori_out_channels, prune_indexs)

            # if fc layer
            if isinstance(module1, nn.Linear) and name1 == 'fc':
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
            transforms.RandomSizedCrop(299),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Scale(314),
            transforms.CenterCrop(299),
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

    model = models.inception_v3(pretrained=True,aux_logits=True)
    model.aux_logits=False
    n_in = model.fc.in_features
    model.fc = nn.Linear(n_in,2)

    fine_tuner = IncepResV1FilterPruner(model=model,
                                   train_dataloader=data_loader['train'],
                                   test_dataloader=data_loader['val'],
                                   criterion=criterion,
                                   useCuda=True)
    fine_tuner.perform_prune(save_dir='./saved_models/IncepV3-test')