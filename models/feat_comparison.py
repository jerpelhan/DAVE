import torch
from torch import nn


class ConvBlock1(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, activation='relu'):
        super().__init__()
        self.activation = activation
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, 256, kernel_size=1)

    def forward(self, input):
        x = self.conv1(input)
        if self.activation == 'relu':
            return self.conv2(self.relu(self.bn(x)))
        else:
            return x


class Feature_Transform(nn.Module):

    def __init__(self):
        self.in_channels = 3584
        super(Feature_Transform, self).__init__()
        self.conv_block1 = ConvBlock1(self.in_channels, self.in_channels // 8, kernel_size=1)
        self.flat = nn.Flatten()

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.flat(x)
        return x
