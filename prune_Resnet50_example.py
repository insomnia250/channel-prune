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

from utils.train import train,trainlog
from model_prune.filter_pruner import FilterPruner
from model_prune.prune_utils import *


os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"


class Res50FilterPruner(FilterPruner):
    # Inherited a FilterPruner class

    # define a custom forward function in which hooks are registered to calculate grads
    def forward(self, x):
        self.activations = []
        activation_index = 0

        # first conv layers
        for i, (name1, module1) in enumerate(self.model._modules.items()):
            if i >= 4: break
            if isinstance(module1,nn.Conv2d):
                x = module1(x)
                self.attach_action_and_hook(x,activation_index,(name1,))
                activation_index += 1
            elif isinstance(module1,nn.BatchNorm2d):
                x = module1(x)
                x = F.relu(x,inplace=True)
            elif isinstance(module1,nn.MaxPool2d):
                x = module1(x)

        for i, (name1, module1) in enumerate(self.model._modules.items()):
            if i<4:continue

            # layer1-4
            if isinstance(module1, nn.Sequential):
                # each Bottleneck
                for j, (name2, module2) in enumerate(module1._modules.items()):
                    # print module2
                    branchs = {}
                    branchs['branch_conv'] = x
                    branchs['branch_x'] = x
                    for k, (name3, module3) in enumerate(module2._modules.items()):
                        if isinstance(module3, nn.Conv2d):
                            # print name1,name2,name3,module3
                            branchs['branch_conv'] = module3(branchs['branch_conv'] )
                            if not (name3=='conv3'):
                                self.attach_action_and_hook(branchs['branch_conv'], activation_index, (name1,name2,name3))
                                activation_index += 1
                        elif isinstance(module3, nn.BatchNorm2d):
                            branchs['branch_conv'] = module3(branchs['branch_conv'])
                            if name3!='bn3':
                                branchs['branch_conv'] = F.relu(branchs['branch_conv'],inplace=True)
                        elif (name3 == 'downsample') and isinstance(module3,nn.Sequential):
                            for l, (name4, module4) in enumerate(module3._modules.items()):
                                branchs['branch_x'] = module4(branchs['branch_x'])
                    x = branchs['branch_conv'] + branchs['branch_x']
                    if name2 == '0':
                        self.attach_action_and_hook(x, activation_index, (name1, name2, 'sum'))
                        activation_index += 1
                    x = F.relu(x,inplace=True)
            if isinstance(module1, nn.AvgPool2d):
                x = F.adaptive_avg_pool2d(x,(1,1))
                x = x.view(x.size(0), -1)
            if isinstance(module1, nn.Linear):
                x = module1(x)
        return x

    # define a custom prune function to prune filters (functions in prune_utils may be helpful)
    def prune_conv_layer(self,model, prune_target):
        channels_now = 3
        prepruned_index = []

        for i, (name1, module1) in enumerate(model._modules.items()):
            if i >= 4: break
            if isinstance(module1, nn.Conv2d):
                # # to prune
                if (name1,) in prune_target.keys():
                    prune_index = prune_target[(name1,)]
                else:
                    prune_index = []
                module1 = prune_conv_weight(module1, prepruned_index, prune_index)
                prepruned_index = prune_index
                channels_now = module1.out_channels
            elif isinstance(module1, nn.BatchNorm2d):
                module1 = prune_bn_weight(module1, prepruned_index)

        for i, (name1, module1) in enumerate(model._modules.items()):
            if name1 in ['layer1', 'layer2', 'layer3', 'layer4']:
                repeat1_prune = prepruned_index
                repeat1_channels_now = channels_now
                for j, (name2, module2) in enumerate(module1._modules.items()):
                    # if name2 != '0':continue

                    if name2 == '0' and (name1, name2, 'sum') in prune_target.keys():
                        prune_target[(name1, name2, 'conv3')] = prune_target[(name1, name2, 'sum')]
                        prune_target[(name1, name2, 'downsample', '0')] = prune_target[(name1, name2, 'sum')]
                    elif name2 != '0':
                        prune_target[(name1, name2, 'conv3')] = repeat1_prune

                    # print name1,name2,'=='*20
                    branchs = {'branch_conv': [0, repeat1_channels_now, repeat1_prune, []],
                               'branch_x': [0, repeat1_channels_now, repeat1_prune, []]
                               }  # to store ori_out_channel, channel_now, prepruned_index, prune_index in each branch
                    # branch...
                    for k, (name3, module3) in enumerate(module2._modules.items()):
                        if isinstance(module3, nn.Conv2d):
                            branchs['branch_conv'][0] = module3.out_channels
                            if (name1, name2, name3) in prune_target.keys():
                                branchs['branch_conv'][3] = prune_target[(name1, name2, name3)]
                            else:
                                branchs['branch_conv'][3] = []
                            module3 = prune_conv_weight(module3, branchs['branch_conv'][2], branchs['branch_conv'][3])
                            branchs['branch_conv'][2] = branchs['branch_conv'][3]  # prepruned index <- prune index
                            branchs['branch_conv'][1] = module3.out_channels  # channel_now <- out_channels
                        elif isinstance(module3, nn.BatchNorm2d):
                            module3 = prune_bn_weight(module3, branchs['branch_conv'][2])
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
    if isTrain:

        model = models.resnet50(pretrained=True)
        inft_num = model.fc.in_features
        model.fc = nn.Linear(in_features=inft_num, out_features=2)
        for i, m in enumerate(list(model.children())[0:-1]):
            for p in m.parameters():
                p.requires_grad = True
        model = torch.nn.DataParallel(model)
        model = model.cuda()



        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-5)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=27, gamma=0.1)
        save_dir = './saved_models/Resnet50'
        logfile = './saved_models/Resnet50/trainlog.log'
        trainlog(logfile)
        train(model,
             epoch_num,
             batch_size,
             start_epoch,
             optimizer,
             criterion,
             exp_lr_scheduler,
             dataset,
             data_loader,
             usecuda,
             save_inter,
             save_dir)

    elif isPrune:

        model = models.resnet50(num_classes=2)
        model.load_state_dict(torch.load('./saved_models/Resnet50/weights-1-[0.9898].pth'))
        fine_tuner = Res50FilterPruner(model=model,
                                       train_dataloader=data_loader['train'],
                                       test_dataloader=data_loader['val'],
                                       criterion=criterion,
                                       useCuda=True)
        fine_tuner.perform_prune(save_dir='./saved_models/Resnet50-test')