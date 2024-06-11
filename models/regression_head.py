from torch import nn
import torch


class Scale(nn.Module):
    def __init__(self, init=1.0):
        super().__init__()

        self.scale = nn.Parameter(torch.tensor([init], dtype=torch.float32))

    def forward(self, input):
        return input * self.scale


class UpsamplingLayer(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(UpsamplingLayer, self).__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.UpsamplingBilinear2d(scale_factor=scale_factor)
        )

    def forward(self, x):
        return self.layer(x)


class DensityMapRegressor(nn.Module):

    def __init__(self, in_channels, reduction):

        super(DensityMapRegressor, self).__init__()
        if reduction == 8:
            self.regressor = nn.Sequential(
                UpsamplingLayer(in_channels, 128),
                UpsamplingLayer(128, 64),
                UpsamplingLayer(64, 32),
                nn.Conv2d(32, 1, kernel_size=1),
                nn.LeakyReLU()
            )
        elif reduction == 16:
            self.regressor = nn.Sequential(
                UpsamplingLayer(in_channels, 128),
                UpsamplingLayer(128, 64),
                UpsamplingLayer(64, 32),
                UpsamplingLayer(32, 16),
                nn.Conv2d(16, 1, kernel_size=1),
                nn.LeakyReLU()
            )

        self.reset_parameters()

    def forward(self, x):
        return self.regressor(x)

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, std=0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
