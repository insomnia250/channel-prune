import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import scipy.stats as stats

class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels,**kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0, affine=True)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Block35(nn.Module):

    def __init__(self, scale=1.0, in_channels=256):
        super(Block35, self).__init__()

        self.scale = scale

        self.branch0_1x1 = BasicConv2d(in_channels, 32, kernel_size=1, stride=1)

        self.branch1_0_1x1 = BasicConv2d(in_channels, 32, kernel_size=1, stride=1)
        self.branch1_1_3x3 = BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)

        self.branch2_0_1x1 = BasicConv2d(in_channels, 32, kernel_size=1, stride=1)
        self.branch2_1_3x3 = BasicConv2d(32, 48, kernel_size=3, stride=1, padding=1)
        self.branch2_2_3x3 = BasicConv2d(48, 64, kernel_size=3, stride=1, padding=1)

        self.conv2d = nn.Conv2d(128, in_channels, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0_1x1(x)

        x1 = self.branch1_0_1x1(x)
        x1 = self.branch1_1_3x3(x1)

        x2 = self.branch2_0_1x1(x)
        x2 = self.branch2_1_3x3(x2)
        x2 = self.branch2_2_3x3(x2)

        out = torch.cat([x0, x1, x2], 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out


class Block17(nn.Module):
    def __init__(self, scale=1.0, in_channels=896):
        super(Block17, self).__init__()

        self.scale = scale

        self.branch0_1x1 = BasicConv2d(in_channels, 128, kernel_size=1, stride=1)

        self.branch1_0_1x1 =  BasicConv2d(in_channels, 128, kernel_size=1, stride=1)
        self.branch1_1_1x7 =  BasicConv2d(128, 128, kernel_size=(1, 7), stride=1, padding=(0, 3))
        self.branch1_2_7x1 = BasicConv2d(128, 128, kernel_size=(7, 1), stride=1, padding=(3, 0))

        self.conv2d = nn.Conv2d(256, in_channels, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0_1x1(x)

        x1 = self.branch1_0_1x1(x)
        x1 = self.branch1_1_1x7(x1)
        x1 = self.branch1_2_7x1(x1)

        out = torch.cat([x0, x1], 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out


# this block's channels differs from Teddy's
class Block8(nn.Module):

    def __init__(self, scale=1.0, in_channels=1792, noReLU=False):
        super(Block8, self).__init__()

        self.scale = scale
        self.noReLU = noReLU

        self.branch0_1x1 = BasicConv2d(in_channels, 192, kernel_size=1, stride=1)

        self.branch1_0_1x1 = BasicConv2d(in_channels, 192, kernel_size=1, stride=1)
        self.branch1_1_1x3 = BasicConv2d(192, 192, kernel_size=(1,3), stride=1, padding=(0,1))
        self.branch1_2_3x1 = BasicConv2d(192, 192, kernel_size=(3,1), stride=1, padding=(1,0))


        self.conv2d = nn.Conv2d(384, in_channels, kernel_size=1, stride=1)
        if not self.noReLU:
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0_1x1(x)

        x1 = self.branch1_0_1x1(x)
        x1 = self.branch1_1_1x3(x1)
        x1 = self.branch1_2_3x1(x1)

        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        if not self.noReLU:
            out = self.relu(out)
        return out


# check channels number with reduction_a
class Mixed_6a(nn.Module):
    def __init__(self):
        super(Mixed_6a, self).__init__()

        self.branch0_3x3 = BasicConv2d(256, 384, kernel_size=3, stride=2)  # 'valid' padding not 'same'

        self.branch1_0_1x1 = BasicConv2d(256, 192, kernel_size=1, stride=1)
        self.branch1_1_3x3 =  BasicConv2d(192, 192, kernel_size=3, stride=1, padding=1)
        self.branch1_2_3x3 = BasicConv2d(192, 256, kernel_size=3, stride=2)

        self.branch2_maxpool = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0_3x3(x)

        x1 = self.branch1_0_1x1(x)
        x1 = self.branch1_1_3x3(x1)
        x1 = self.branch1_2_3x3(x1)

        x2 = self.branch2_maxpool(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


# check channels number with reduction_b
class Mixed_7a(nn.Module):
    def __init__(self):
        super(Mixed_7a, self).__init__()

        self.branch0_0_1x1 = BasicConv2d(896, 256, kernel_size=1, stride=1)
        self.branch0_1_3x3 = BasicConv2d(256, 384, kernel_size=3, stride=2)

        self.branch1_0_1x1 = BasicConv2d(896, 256, kernel_size=1, stride=1)
        self.branch1_1_3x3 = BasicConv2d(256, 256, kernel_size=3, stride=2)

        self.branch2_0_1x1 = BasicConv2d(896, 256, kernel_size=1, stride=1)
        self.branch2_1_3x3 = BasicConv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.branch2_2_3x3 = BasicConv2d(256, 256, kernel_size=3, stride=2)

        self.branch3_maxpool = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0_0_1x1(x)
        x0 = self.branch0_1_3x3(x0)

        x1 = self.branch1_0_1x1(x)
        x1 = self.branch1_1_3x3(x1)

        x2 = self.branch2_0_1x1(x)
        x2 = self.branch2_1_3x3(x2)
        x2 = self.branch2_2_3x3(x2)

        x3 = self.branch3_maxpool(x)

        out = torch.cat([x0, x1, x2, x3], 1)
        return out


class InceptionResnetV1(nn.Module):
    def __init__(self, num_classes=20):
        super(InceptionResnetV1, self).__init__()
        self.conv2d_1a = BasicConv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.conv2d_2a = BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)  # 'same' padding
        self.conv2d_2b = BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1)  # 'same' padding
        self.maxpool_3a = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2d_3b = BasicConv2d(64, 80, kernel_size=1, stride=1)
        self.conv2d_4a = BasicConv2d(80, 192, kernel_size=3, stride=1, padding=1) #same padding
        self.conv2d_4b = BasicConv2d(192, 256, kernel_size=3, stride=2, padding=1) #same padding

        self.repeat_block35 = nn.Sequential(
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17)
        )
        self.mixed_6a = Mixed_6a()
        self.repeat_block17 = nn.Sequential(
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10)
        )
        self.mixed_7a = Mixed_7a()
        self.repeat_block8 = nn.Sequential(
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20)
        )
        self.block8 = Block8(noReLU=True)
        self.avgpool_1a = nn.AvgPool2d(2, count_include_pad=False)
        self.dropout = nn.Dropout(0.2) # proba of an element to be zeroed
        self.fc = nn.Linear(1792, num_classes)

    def forward(self,x):

        # n x 3 x 112 x 96
        x = self.conv2d_1a(x)
        # n x 32 x 56 x 48
        x = self.conv2d_2a(x)
        # n x 32 x 59 x 48
        x = self.conv2d_2b(x)
        # n x 64 x 56 x 48
        x = self.maxpool_3a(x)
        # n x 64 x 28 x 24
        x = self.conv2d_3b(x)
        # n x 80 x 28 x 24
        x = self.conv2d_4a(x)
        # n x 192 x 28 x 24
        x = self.conv2d_4b(x)
        # n x 256 x 14 x 12
        x = self.repeat_block35(x)
        # n x 256 x 14 x 12
        x = self.mixed_6a(x)
        # n x 896 x 6 x 5
        x = self.repeat_block17(x)
        # n x 896 x 6 x 5
        x = self.mixed_7a(x)
        # n x 1792 x 2 x 2
        x = self.repeat_block8(x)
        # n x 1792 x 2 x 2
        x = self.block8(x)
        # n x 1792 x 2 x 2
        x = self.avgpool_1a(x)
        # n x 1792 x 1 x 1
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        # n x 1792
        x = self.fc(x)
        return x


if __name__ == '__main__':
    # model = Block35()
    #
    # for j, (name2, module2) in enumerate(model._modules.items()):
    #
    #     print '=='*20, name2
    #     print module2
    #     for j, (name3, module3) in enumerate(module2._modules.items()):
    #         print '--' * 20, name2, name3
    #         print module3

    model = InceptionResnetV1(num_classes=2)
    print model

    x = torch.FloatTensor(10, 1, 112, 96)
    x = torch.autograd.Variable(x)

    y = model(x)
    print y.size()


