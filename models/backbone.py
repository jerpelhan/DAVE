from typing import List

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
from torchvision.ops.misc import FrozenBatchNorm2d


class Backbone(nn.Module):

    def __init__(
        self,
        name: str,
        pretrained: bool,
        dilation: bool,
        reduction: int,
        swav: bool,
        requires_grad: bool
    ):

        super(Backbone, self).__init__()

        resnet = getattr(models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=pretrained, norm_layer=FrozenBatchNorm2d
        )

        self.backbone = resnet
        self.reduction = reduction

        if name == 'resnet50' and swav:
            checkpoint = torch.hub.load_state_dict_from_url(
                'https://dl.fbaipublicfiles.com/deepcluster/swav_800ep_pretrain.pth.tar',
                map_location="cpu"
            )
            state_dict = {k.replace("module.", ""): v for k, v in checkpoint.items()}
            self.backbone.load_state_dict(state_dict, strict=False)

        # concatenation of layers 2, 3 and 4
        self.num_channels = 896 if name in ['resnet18', 'resnet34'] else 3584

        for n, param in self.backbone.named_parameters():
            if 'layer2' not in n and 'layer3' not in n and 'layer4' not in n:
                param.requires_grad_(False)
            else:
                param.requires_grad_(requires_grad)

    def forward(self, x):
        size = x.size(-2) // self.reduction, x.size(-1) // self.reduction
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = layer1 = self.backbone.layer1(x)
        x = layer2 = self.backbone.layer2(x)
        x = layer3 = self.backbone.layer3(x)
        x = layer4 = self.backbone.layer4(x)

        x = torch.cat([
            F.interpolate(f, size=size, mode='bilinear', align_corners=True)
            for f in [layer2, layer3, layer4]
        ], dim=1)

        return x
