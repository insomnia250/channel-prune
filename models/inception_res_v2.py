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

    def __init__(self, scale=1.0, in_channels=320):
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


# this block's channels differs from Teddy's
class Block17(nn.Module):
    def __init__(self, scale=1.0, in_channels=1088):
        super(Block17, self).__init__()

        self.scale = scale

        self.branch0_1x1 = BasicConv2d(in_channels, 192, kernel_size=1, stride=1)

        self.branch1_0_1x1 =  BasicConv2d(in_channels, 128, kernel_size=1, stride=1)
        self.branch1_1_1x7 =  BasicConv2d(128, 160, kernel_size=(1, 7), stride=1, padding=(0, 3))
        self.branch1_2_7x1 = BasicConv2d(160, 192, kernel_size=(7, 1), stride=1, padding=(3, 0))

        self.conv2d = nn.Conv2d(384, in_channels, kernel_size=1, stride=1)
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

    def __init__(self, scale=1.0, in_channels=2080, noReLU=False):
        super(Block8, self).__init__()

        self.scale = scale
        self.noReLU = noReLU

        self.branch0_1x1 = BasicConv2d(in_channels, 192, kernel_size=1, stride=1)

        self.branch1_0_1x1 = BasicConv2d(in_channels, 192, kernel_size=1, stride=1)
        self.branch1_1_1x3 = BasicConv2d(192, 224, kernel_size=(1,3), stride=1, padding=(0,1))
        self.branch1_2_3x1 = BasicConv2d(224, 256, kernel_size=(3,1), stride=1, padding=(1,0))


        self.conv2d = nn.Conv2d(448, in_channels, kernel_size=1, stride=1)
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


# this is not used in Teddy's
class Mixed_5b(nn.Module):
    def __init__(self):
        super(Mixed_5b, self).__init__()

        self.branch0_1x1 = BasicConv2d(192, 96, kernel_size=1, stride=1)

        self.branch1_0_1x1 = BasicConv2d(192, 48, kernel_size=1, stride=1)
        self.branch1_1_5x5 = BasicConv2d(48, 64, kernel_size=5, stride=1, padding=2)

        self.branch2_0_1x1 = BasicConv2d(192, 64, kernel_size=1, stride=1)
        self.branch2_1_3x3 = BasicConv2d(64, 96, kernel_size=3, stride=1, padding=1)
        self.branch2_2_3x3 = BasicConv2d(96, 96, kernel_size=3, stride=1, padding=1)

        self.branch3_0_avgpool = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)
        self.branch3_1_1x1 = BasicConv2d(192, 64, kernel_size=1, stride=1)

    def forward(self, x):
        x0 = self.branch0_1x1(x)

        x1 = self.branch1_0_1x1(x)
        x1 = self.branch1_1_5x5(x1)

        x2 = self.branch2_0_1x1(x)
        x2 = self.branch2_1_3x3(x2)
        x2 = self.branch2_2_3x3(x2)

        x3 = self.branch3_0_avgpool(x)
        x3 = self.branch3_1_1x1(x3)

        out = torch.cat((x0, x1, x2, x3), 1)
        return out


# check channels number with reduction_a
class Mixed_6a(nn.Module):
    def __init__(self):
        super(Mixed_6a, self).__init__()

        self.branch0_3x3 = BasicConv2d(320, 384, kernel_size=3, stride=2)  # 'valid' padding not 'same'

        self.branch1_0_1x1 = BasicConv2d(320, 256, kernel_size=1, stride=1)
        self.branch1_1_3x3 =  BasicConv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.branch1_2_3x3 = BasicConv2d(256, 384, kernel_size=3, stride=2)

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

        self.branch0_0_1x1 = BasicConv2d(1088, 256, kernel_size=1, stride=1)
        self.branch0_1_3x3 = BasicConv2d(256, 384, kernel_size=3, stride=2)

        self.branch1_0_1x1 = BasicConv2d(1088, 256, kernel_size=1, stride=1)
        self.branch1_1_3x3 = BasicConv2d(256, 288, kernel_size=3, stride=2)

        self.branch2_0_1x1 = BasicConv2d(1088, 256, kernel_size=1, stride=1)
        self.branch2_1_3x3 = BasicConv2d(256, 288, kernel_size=3, stride=1, padding=1)
        self.branch2_2_3x3 = BasicConv2d(288, 320, kernel_size=3, stride=2)

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


class InceptionResnetV2(nn.Module):
    def __init__(self, num_classes=20):
        super(InceptionResnetV2, self).__init__()
        self.conv2d_1a = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.conv2d_2a = BasicConv2d(32, 32, kernel_size=3, stride=1)
        self.conv2d_2b = BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.maxpool_3a = nn.MaxPool2d(3, stride=2)
        self.conv2d_3b = BasicConv2d(64, 80, kernel_size=1, stride=1)
        self.conv2d_4a = BasicConv2d(80, 192, kernel_size=3, stride=1)
        self.maxpool_5a = nn.MaxPool2d(3, stride=2)
        self.mixed_5b = Mixed_5b()
        self.repeat_block35 = nn.Sequential(
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
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
            Block17(scale=0.10),
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
        self.conv2d_7b = BasicConv2d(2080, 1536, kernel_size=1, stride=1)
        self.avgpool_1a = nn.AvgPool2d(8, count_include_pad=False)
        self.fc = nn.Linear(1536, num_classes)

    def forward(self,x):

        # n x 3 x 299 x 299
        x = self.conv2d_1a(x)
        # n x 32 x 149 x 149
        x = self.conv2d_2a(x)
        # n x 32 x 147 x 147
        x = self.conv2d_2b(x)
        # n x 64 x 147 x 147
        x = self.maxpool_3a(x)
        # n x 64 x 73 x 73
        x = self.conv2d_3b(x)
        # n x 80 x 73 x 73
        x = self.conv2d_4a(x)
        # n x 192 x 71 x 71
        x = self.maxpool_5a(x)
        # n x 192 x 35 x 35
        x = self.mixed_5b(x)
        # n x 320 x 35 x 35
        x = self.repeat_block35(x)
        # n x 320 x 35 x 35
        x = self.mixed_6a(x)
        # n x 1088 x 17 x 17
        x = self.repeat_block17(x)
        # n x 1088 x 17 x 17
        x = self.mixed_7a(x)
        # n x 2080 x 8 x 8
        x = self.repeat_block8(x)
        # n x 2080 x 8 x 8
        x = self.block8(x)
        # n x 2080 x 8 x 8
        x = self.conv2d_7b(x)
        # n x 1536 x 8 x 8
        x = self.avgpool_1a(x)
        # n x 1536 x 1 x 1
        x = x.view(x.size(0), -1)
        # n x 1536
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

    model = InceptionResnetV2(num_classes=2)
    print model

    x = torch.FloatTensor(10, 3, 299, 299)
    x = torch.autograd.Variable(x)

    y = model(x)
    print y.size()


