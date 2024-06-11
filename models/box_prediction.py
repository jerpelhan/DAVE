import numpy as np
import skimage
import torch
from torch import nn
from torchvision import ops


class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                               bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, input):
        x = self.conv1(input)
        return self.relu(self.bn(x))


class FeatureFusionModule(torch.nn.Module):
    def __init__(self, num_classes, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.convblock = ConvBlock(in_channels=self.in_channels, out_channels=num_classes, stride=1)
        self.conv1 = nn.Conv2d(num_classes, num_classes, kernel_size=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(num_classes, num_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, input_1, input_2):
        x = torch.cat((input_1, input_2), dim=1)
        assert self.in_channels == x.size(1), 'in_channels of ConvBlock should be {}'.format(x.size(1))
        feature = self.convblock(x)
        x = self.avgpool(feature)

        x = self.relu(self.conv1(x))
        x = self.sigmoid(self.conv2(x))
        x = torch.mul(feature, x)
        x = torch.add(x, feature)
        return x


class Scale(nn.Module):
    def __init__(self, init=1.0):
        super().__init__()

        self.scale = nn.Parameter(torch.tensor([init], dtype=torch.float32))

    def forward(self, input):
        return input * self.scale


class FCOSHead(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        conv_channels = 256

        bbox_tower = []
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=conv_channels, kernel_size=3, stride=1, padding=1)
        self.ffm = FeatureFusionModule(num_classes=conv_channels, in_channels=conv_channels * 2)
        bbox_tower.append(
            nn.Conv2d(
                conv_channels,
                conv_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True
            )
        )
        bbox_tower.append(nn.GroupNorm(32, conv_channels))
        bbox_tower.append(nn.ReLU())
        self.bbox_tower = nn.Sequential(*bbox_tower)
        self.bbox_pred = nn.Conv2d(conv_channels, 4, 3, padding=1)

        # initialization
        for modules in [self.bbox_tower,
                        self.bbox_pred]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        self.scale = Scale(1.0)

    def forward(self, bb_fts, R):
        x = self.conv1(bb_fts)
        x = self.ffm(R, x)
        bbox_out = self.bbox_tower(x)
        bbox_out = torch.exp(self.scale(self.bbox_pred(bbox_out)))
        return bbox_out


FLIP_LEFT_RIGHT = 0
FLIP_TOP_BOTTOM = 1


class BoxList:
    def __init__(self, box, image_size, mode='xyxy'):
        device = box.device if hasattr(box, 'device') else 'cpu'
        if torch.is_tensor(box):
            box = torch.as_tensor(box, dtype=torch.float32, device=device)
        else:
            box = torch.as_tensor(np.array(box), dtype=torch.float32, device=device)

        self.box = box
        self.size = image_size
        self.mode = mode

        self.fields = {}

    def convert(self, mode):
        if mode == self.mode:
            return self

        x_min, y_min, x_max, y_max = self.split_to_xyxy()

        if mode == 'xyxy':
            box = torch.cat([x_min, y_min, x_max, y_max], -1)
            box = BoxList(box, self.size, mode=mode)

        elif mode == 'xywh':
            remove = 1
            box = torch.cat(
                [x_min, y_min, x_max - x_min + remove, y_max - y_min + remove], -1
            )
            box = BoxList(box, self.size, mode=mode)

        box.copy_field(self)

        return box

    def copy_field(self, box):
        for k, v in box.fields.items():
            self.fields[k] = v

    def area(self):
        box = self.box

        if self.mode == 'xyxy':
            remove = 1

            area = (box[:, 2] - box[:, 0] + remove) * (box[:, 3] - box[:, 1] + remove)

        elif self.mode == 'xywh':
            area = box[:, 2] * box[:, 3]

        return area

    def split_to_xyxy(self):
        if self.mode == 'xyxy':
            x_min, y_min, x_max, y_max = self.box.split(1, dim=-1)

            return x_min, y_min, x_max, y_max

        elif self.mode == 'xywh':
            remove = 1
            x_min, y_min, w, h = self.box.split(1, dim=-1)

            return (
                x_min,
                y_min,
                x_min + (w - remove).clamp(min=0),
                y_min + (h - remove).clamp(min=0),
            )

    def __len__(self):
        return self.box.shape[0]

    def __getitem__(self, index):
        box = BoxList(self.box[index], self.size, self.mode)

        for k, v in self.fields.items():
            box.fields[k] = v[index]

        return box

    def resize(self, size, *args, **kwargs):
        ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(size, self.size))

        if ratios[0] == ratios[1]:
            ratio = ratios[0]
            scaled = self.box * ratio
            box = BoxList(scaled, size, mode=self.mode)

            for k, v in self.fields.items():
                if not isinstance(v, torch.Tensor):
                    v = v.resize(size, *args, **kwargs)

                box.fields[k] = v

            return box

        ratio_w, ratio_h = ratios
        x_min, y_min, x_max, y_max = self.split_to_xyxy()
        scaled_x_min = x_min * ratio_w
        scaled_x_max = x_max * ratio_w
        scaled_y_min = y_min * ratio_h
        scaled_y_max = y_max * ratio_h
        scaled = torch.cat([scaled_x_min, scaled_y_min, scaled_x_max, scaled_y_max], -1)
        box = BoxList(scaled, size, mode='xyxy')

        for k, v in self.fields.items():
            if not isinstance(v, torch.Tensor):
                v = v.resize(size, *args, **kwargs)

            box.fields[k] = v

        return box.convert(self.mode)

    def clip(self, remove_empty=True):
        remove = 1

        max_width = self.size[0] - remove
        max_height = self.size[1] - remove

        self.box[:, 0].clamp_(min=0, max=max_width)
        self.box[:, 1].clamp_(min=0, max=max_height)
        self.box[:, 2].clamp_(min=0, max=max_width)
        self.box[:, 3].clamp_(min=0, max=max_height)

        if remove_empty:
            box = self.box
            keep = (box[:, 3] > box[:, 1]) & (box[:, 2] > box[:, 0])

            return self[keep]

        else:
            return self

    def to(self, device):
        box = BoxList(self.box.to(device), self.size, self.mode)

        for k, v in self.fields.items():
            if hasattr(v, 'to'):
                v = v.to(device)

            box.fields[k] = v

        return box


def remove_small_box(boxlist, min_size):
    box = boxlist.convert('xywh').box
    _, _, w, h = box.unbind(dim=1)
    keep = (w >= min_size) & (h >= min_size)
    keep = keep.nonzero().squeeze(1)

    return boxlist[keep]


def boxlist_nms(boxlist, scores, threshold, max_proposal=-1):
    if threshold <= 0:
        return boxlist

    mode = boxlist.mode
    boxlist = boxlist.convert('xyxy')
    box = boxlist.box
    keep = ops.nms(box, scores, threshold)

    if max_proposal > 0:
        keep = keep[:max_proposal]

    boxlist = boxlist[keep]
    return boxlist.convert(mode)
