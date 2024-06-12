import os
import json
from typing import Tuple, List
from pycocotools.coco import COCO
from PIL import Image
import numpy as np
from torch import Tensor
from torch.utils.data import Dataset
import torch
import torchvision.transforms.functional as FT
from torchvision import transforms as T
from torchvision.transforms import functional as TVF

def resize(img, bboxes):
    resize_img = T.Resize((512, 512), antialias=True)
    w, h = img.size
    img = T.Compose([
        T.ToTensor(),
        resize_img,
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])(img)
    scale = torch.tensor([1.0, 1.0]) / torch.tensor([w, h]) * 512
    bboxes = bboxes / torch.tensor([w, h, w, h]) * 512

    scale_x = min(1.0, 50 / (bboxes[:, 2] - bboxes[:, 0]).mean())
    scale_y = min(1.0, 50 / (bboxes[:, 3] - bboxes[:, 1]).mean())

    if scale_x < 1 or scale_y < 1:
        scale_y = (int(img.shape[1] * scale_y) // 8 * 8) / img.shape[1]
        scale_x = (int(img.shape[2] * scale_x) // 8 * 8) / img.shape[2]
        resize_ = T.Resize((int(img.shape[1] * scale_y), int(img.shape[2] * scale_x)), antialias=True)
        img_ = resize_(img)
        img = pad_image(img_)

    else:
        scale_x = min(max(1.0, 11 / (bboxes[:, 2] - bboxes[:, 0]).mean()), 1.9)
        scale_y = min(max(1.0, 11 / (bboxes[:, 3] - bboxes[:, 1]).mean()), 1.9)

        scale_y = (int(img.shape[1] * scale_y) // 8 * 8) / img.shape[1]
        scale_x = (int(img.shape[2] * scale_x) // 8 * 8) / img.shape[2]
        resize_ = T.Resize((int(img.shape[1] * scale_y), int(img.shape[2] * scale_x)), antialias=True)
        img = resize_(img)
    scale = scale * torch.tensor([scale_x, scale_y])

    bboxes = bboxes * torch.tensor([scale_x, scale_y, scale_x, scale_y])
    return img, bboxes.float(), scale

def pad_image(image_tensor):
    target_size = (512, 512)
    original_size = image_tensor.size()[1:]

    # Calculate the amount of padding needed for each dimension
    pad_height = max(target_size[0] - original_size[0], 0)
    pad_width = max(target_size[1] - original_size[1], 0)
    padded_image = FT.pad(image_tensor, (0, 0, pad_width, pad_height))
    return padded_image


def tiling_augmentation_(img, bboxes, density_map, resize, jitter, tile_size, hflip_p):
    def apply_hflip(tensor, apply):
        return TVF.hflip(tensor) if apply else tensor

    def make_tile(x, num_tiles, hflip, hflip_p, jitter=None):
        result = list()
        for j in range(num_tiles):
            row = list()
            for k in range(num_tiles):
                t = jitter(x) if jitter is not None else x
                if hflip[j, k] < hflip_p:
                    t = TVF.hflip(t)
                row.append(t)
            result.append(torch.cat(row, dim=-1))
        return torch.cat(result, dim=-2)

    x_tile, y_tile = tile_size
    y_target, x_target = resize.size
    num_tiles = max(int(x_tile.ceil()), int(y_tile.ceil()))
    tensors = [img, density_map]
    results = list()
    # whether to horizontally flip each tile
    hflip = torch.rand(num_tiles, num_tiles)

    img = make_tile(img, num_tiles, hflip, hflip_p, jitter)
    img = resize(img[..., :int(y_tile * y_target), :int(x_tile * x_target)])

    density_map = make_tile(density_map, num_tiles, hflip, hflip_p)
    density_map = density_map[..., :int(y_tile * y_target), :int(x_tile * x_target)]
    original_sum = density_map.sum()
    density_map = resize(density_map)
    density_map = density_map / density_map.sum() * original_sum

    if hflip[0, 0] < hflip_p:
        bboxes[:, [0, 2]] = x_target - bboxes[:, [2, 0]]
    bboxes = bboxes / torch.tensor([x_tile, y_tile, x_tile, y_tile])
    return img, bboxes, density_map


class FSC147WithDensityMapDOWNSIZE(Dataset):

    def __init__(
            self, data_path, img_size, split='train', num_objects=3,
            tiling_p=0.5, zero_shot=False, skip_cars=False
    ):
        self.split = split
        self.data_path = data_path
        self.horizontal_flip_p = 0.5
        self.tiling_p = tiling_p
        self.img_size = img_size
        self.resize = T.Resize((img_size, img_size), antialias=True)
        self.jitter = T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8)
        self.num_objects = num_objects
        self.zero_shot = zero_shot
        with open(
                os.path.join(self.data_path, 'Train_Test_Val_FSC_147.json'), 'rb'
        ) as file:
            splits = json.load(file)
            self.image_names = splits[split]
        with open(
                os.path.join(self.data_path, 'annotation_FSC147_384.json'), 'rb'
        ) as file:
            self.annotations = json.load(file)
        with open(
                os.path.join(self.data_path, 'ImageClasses_FSC147.txt'), 'r'
        ) as file:
            self.classes = dict([
                (line.strip().split()[0], ' '.join(line.strip().split()[1:]))
                for line in file.readlines()
            ])
        with open(
                os.path.join(self.data_path, 'ImageClasses_FSC147.txt'), 'r'
        ) as file:
            self.classes = dict([
                (line.strip().split()[0], ' '.join(line.strip().split()[1:]))
                for line in file.readlines()
            ])
        if skip_cars:
            print(len(self.image_names))
            self.image_names = [
                img_name for img_name in self.image_names if self.classes[img_name] != 'cars'
            ]
        print(len(self.image_names))
        if split == 'val' or split == 'test':
            self.labels = COCO(os.path.join(self.data_path, 'instances_' + split + '.json'))
            self.img_name_to_ori_id = self.map_img_name_to_ori_id()
        self.train_stitched = False
        self.test_stitched = False
        self.scale_opt = True

    def __getitem__(self, idx: int) -> Tuple[Tensor, List[Tensor], Tensor]:
        img = Image.open(os.path.join(
            self.data_path,
            'images_384_VarV2',
            self.image_names[idx]
        ))

        w, h = img.size
        img = T.Compose([
            T.ToTensor(),
            self.resize,
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])(img)

        bboxes = torch.tensor(
            self.annotations[self.image_names[idx]]['box_examples_coordinates'],
            dtype=torch.float32
        )[:3, [0, 2], :].reshape(-1, 4)[:self.num_objects, ...]

        bboxes = bboxes / torch.tensor([w, h, w, h]) * self.img_size
        density_map = torch.from_numpy(np.load(os.path.join(
            self.data_path,
            'gt_density_map_adaptive_512_512_object_VarV2',
            os.path.splitext(self.image_names[idx])[0] + '.npy',
        ))).unsqueeze(0)

        if self.zero_shot:
            scale_x = 1
            scale_y = 1
            shape = img.shape
            return img, bboxes, density_map, idx, scale_y, scale_x, shape, self.classes[self.image_names[idx]]

        scale_x = min(1.0, 50 / (bboxes[:, 2] - bboxes[:, 0]).mean())
        scale_y = min(1.0, 50 / (bboxes[:, 3] - bboxes[:, 1]).mean())
        shape = img.shape
        if scale_x < 1 or scale_y < 1:
            scale_y = (int(img.shape[1] * scale_y) // 8 * 8) / img.shape[1]
            scale_x = (int(img.shape[2] * scale_x) // 8 * 8) / img.shape[2]
            resize_ = T.Resize((int(img.shape[1] * scale_y), int(img.shape[2] * scale_x)), antialias=True)
            img_ = resize_(img)
            original_sum = density_map.sum()
            density_map = resize_(density_map)
            density_map = density_map / density_map.sum() * original_sum
            shape = img_.shape
            img = pad_image(img_)

        else:
            scale_x = 1.0
            scale_y = 1.0

        bboxes = bboxes * torch.tensor([scale_x, scale_y, scale_x, scale_y])
        return img, bboxes, density_map, idx, scale_y, scale_x, shape

    def get_resize_factors(self, idxs):
        factors = []
        for i in idxs:
            img = Image.open(os.path.join(
                self.data_path,
                'images_384_VarV2',
                self.image_names[i]
            ))
            w1, h1 = img.size
            factors.append(torch.tensor([w1, h1, w1, h1]) / self.img_size)

        return factors

    def get_gt_bboxes(self, idxs):

        bboxes_xyxy = []
        factors = []
        for i in idxs:
            img = Image.open(os.path.join(
                self.data_path,
                'images_384_VarV2',
                self.image_names[i]
            ))
            w1, h1 = img.size
            coco_im_id = self.img_name_to_ori_id[self.image_names[i]]
            anno_ids = self.labels.getAnnIds([coco_im_id])
            annos = self.labels.loadAnns(anno_ids)
            box_centers = list()
            whs = list()
            xyxy_boxes = list()
            for anno in annos:
                bbox = anno["bbox"]
                x1, y1, w, h = bbox
                box_centers.append([x1 + w / 2, y1 + h / 2])
                whs.append([w, h])
                xyxy_boxes.append([x1, y1, x1 + w, y1 + h])
            xyxy_boxes = np.array(xyxy_boxes, dtype=np.float32)
            xyxy_boxes = xyxy_boxes / torch.tensor([w1, h1, w1, h1]) * self.img_size
            factors.append(torch.tensor([w1, h1, w1, h1]) / self.img_size)
            bboxes_xyxy.append(xyxy_boxes)
        return bboxes_xyxy, factors

    def __len__(self):
        return len(self.image_names)

    def map_img_name_to_ori_id(self, ):
        all_coco_imgs = self.labels.imgs
        map_name_2_id = dict()
        for k, v in all_coco_imgs.items():
            img_id = v["id"]
            img_name = v["file_name"]
            map_name_2_id[img_name] = img_id
        return map_name_2_id



def tiling_augmentation(img, bboxes, density_map, dmap, mask, resize, jitter, tile_size, hflip_p):
    def apply_hflip(tensor, apply):
        return TVF.hflip(tensor) if apply else tensor

    def make_tile(x, num_tiles, hflip, hflip_p, jitter=None):
        result = list()
        for j in range(num_tiles):
            row = list()
            for k in range(num_tiles):
                t = jitter(x) if jitter is not None else x
                if hflip[j, k] < hflip_p:
                    t = TVF.hflip(t)
                row.append(t)
            result.append(torch.cat(row, dim=-1))
        return torch.cat(result, dim=-2)

    x_tile, y_tile = tile_size
    y_target, x_target = resize.size
    num_tiles = max(int(x_tile.ceil()), int(y_tile.ceil()))
    tensors = [img, density_map]
    results = list()
    # whether to horizontally flip each tile
    hflip = torch.rand(num_tiles, num_tiles)

    img = make_tile(img, num_tiles, hflip, hflip_p, jitter)
    img = resize(img[..., :int(y_tile * y_target), :int(x_tile * x_target)])

    density_map = make_tile(density_map, num_tiles, hflip, hflip_p)
    density_map = density_map[..., :int(y_tile * y_target), :int(x_tile * x_target)]
    original_sum = density_map.sum()
    density_map = resize(density_map)
    if original_sum != 0:
        density_map = density_map / density_map.sum() * original_sum

    dmap = make_tile(dmap, num_tiles, hflip, hflip_p)
    dmap = dmap[..., :int(y_tile * y_target), :int(x_tile * x_target)]
    dmap = resize(dmap)
    mask = make_tile(mask, num_tiles, hflip, hflip_p)
    mask = mask[..., :int(y_tile * y_target), :int(x_tile * x_target)]
    mask = resize(mask)

    if hflip[0, 0] < hflip_p:
        bboxes[:, [0, 2]] = x_target - bboxes[:, [2, 0]]  # TODO change
    bboxes = bboxes / torch.tensor([x_tile, y_tile, x_tile, y_tile])
    return img, bboxes, density_map, dmap, mask


class FSC147WithDensityMapSCALE2BOX(Dataset):

    def __init__(
            self, data_path, img_size, split='train', num_objects=3,
            tiling_p=0.5, zero_shot=False, skip_cars=False
    ):
        self.split = split
        self.data_path = data_path
        self.horizontal_flip_p = 0.5
        self.tiling_p = tiling_p
        self.img_size = img_size
        self.resize = T.Resize((img_size, img_size), antialias=True)
        self.jitter = T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8)
        self.num_objects = num_objects
        self.zero_shot = zero_shot
        with open(
                os.path.join(self.data_path, 'Train_Test_Val_FSC_147.json'), 'rb'
        ) as file:
            splits = json.load(file)
            self.image_names = splits[split]
        with open(
                os.path.join(self.data_path, 'annotation_FSC147_384.json'), 'rb'
        ) as file:
            self.annotations = json.load(file)
        with open(
                os.path.join(self.data_path, 'ImageClasses_FSC147.txt'), 'r'
        ) as file:
            self.classes = dict([
                (line.strip().split()[0], ' '.join(line.strip().split()[1:]))
                for line in file.readlines()
            ])
        with open(
                os.path.join(self.data_path, 'ImageClasses_FSC147.txt'), 'r'
        ) as file:
            self.classes = dict([
                (line.strip().split()[0], ' '.join(line.strip().split()[1:]))
                for line in file.readlines()
            ])
        if skip_cars:
            print(len(self.image_names))
            self.image_names = [
                img_name for img_name in self.image_names if self.classes[img_name] != 'cars'
            ]
        print(len(self.image_names))
        if split == 'val' or split == 'test':
            self.labels = COCO(os.path.join(self.data_path, 'instances_' + split + '.json'))
            self.img_name_to_ori_id = self.map_img_name_to_ori_id()
        self.train_stitched = False
        self.test_stitched = False
        self.scale_opt = True

    def __getitem__(self, idx: int) -> Tuple[Tensor, List[Tensor], Tensor]:
        img = Image.open(os.path.join(
            self.data_path,
            'images_384_VarV2',
            self.image_names[idx]
        ))

        w, h = img.size
        img = T.Compose([
            T.ToTensor(),
            self.resize,
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])(img)

        bboxes = torch.tensor(
            self.annotations[self.image_names[idx]]['box_examples_coordinates'],
            dtype=torch.float32
        )[:3, [0, 2], :].reshape(-1, 4)[:self.num_objects, ...]

        bboxes = bboxes / torch.tensor([w, h, w, h]) * self.img_size
        density_map = torch.from_numpy(np.load(os.path.join(
            self.data_path,
            'gt_density_map_adaptive_512_512_object_VarV2',
            os.path.splitext(self.image_names[idx])[0] + '.npy',
        ))).unsqueeze(0)

        if self.zero_shot:
            scale_x = 1
            scale_y = 1
            shape = img.shape
            return img, bboxes, density_map, idx, scale_y, scale_x, shape, self.classes[self.image_names[idx]]

        scale_x = min(1.0, 50 / (bboxes[:, 2] - bboxes[:, 0]).mean())
        scale_y = min(1.0, 50 / (bboxes[:, 3] - bboxes[:, 1]).mean())
        shape = img.shape
        if scale_x < 1 or scale_y < 1:
            scale_y = (int(img.shape[1] * scale_y) // 8 * 8) / img.shape[1]
            scale_x = (int(img.shape[2] * scale_x) // 8 * 8) / img.shape[2]
            resize_ = T.Resize((int(img.shape[1] * scale_y), int(img.shape[2] * scale_x)), antialias=True)
            img_ = resize_(img)
            original_sum = density_map.sum()
            density_map = resize_(density_map)
            density_map = density_map / density_map.sum() * original_sum
            shape = img_.shape
            img = pad_image(img_)

        else:
            scale_x = min(max(1.0, 11 / (bboxes[:, 2] - bboxes[:, 0]).mean()), 1.9)
            scale_y = min(max(1.0, 11 / (bboxes[:, 3] - bboxes[:, 1]).mean()), 1.9)

            scale_y = (int(img.shape[1] * scale_y) // 8 * 8) / img.shape[1]
            scale_x = (int(img.shape[2] * scale_x) // 8 * 8) / img.shape[2]
            resize_ = T.Resize((int(img.shape[1] * scale_y), int(img.shape[2] * scale_x)), antialias=True)
            img = resize_(img)
            original_sum = density_map.sum()
            density_map = resize_(density_map)
            density_map = density_map / density_map.sum() * original_sum
            shape = img.shape

        bboxes = bboxes * torch.tensor([scale_x, scale_y, scale_x, scale_y])
        return img, bboxes, density_map, idx, scale_y, scale_x, shape

    def get_resize_factors(self, idxs):
        factors = []
        for i in idxs:
            img = Image.open(os.path.join(
                self.data_path,
                'images_384_VarV2',
                self.image_names[i]
            ))
            w1, h1 = img.size
            factors.append(torch.tensor([w1, h1, w1, h1]) / self.img_size)

        return factors

    def get_gt_bboxes(self, idxs):

        bboxes_xyxy = []
        factors = []
        for i in idxs:
            img = Image.open(os.path.join(
                self.data_path,
                'images_384_VarV2',
                self.image_names[i]
            ))
            w1, h1 = img.size
            coco_im_id = self.img_name_to_ori_id[self.image_names[i]]
            anno_ids = self.labels.getAnnIds([coco_im_id])
            annos = self.labels.loadAnns(anno_ids)
            box_centers = list()
            whs = list()
            xyxy_boxes = list()
            for anno in annos:
                bbox = anno["bbox"]
                x1, y1, w, h = bbox
                box_centers.append([x1 + w / 2, y1 + h / 2])
                whs.append([w, h])
                xyxy_boxes.append([x1, y1, x1 + w, y1 + h])
            xyxy_boxes = np.array(xyxy_boxes, dtype=np.float32)
            xyxy_boxes = xyxy_boxes / torch.tensor([w1, h1, w1, h1]) * self.img_size
            factors.append(torch.tensor([w1, h1, w1, h1]) / self.img_size)
            bboxes_xyxy.append(xyxy_boxes)
        return bboxes_xyxy, factors

    def __len__(self):
        return len(self.image_names)

    def map_img_name_to_ori_id(self, ):
        all_coco_imgs = self.labels.imgs
        map_name_2_id = dict()
        for k, v in all_coco_imgs.items():
            img_id = v["id"]
            img_name = v["file_name"]
            map_name_2_id[img_name] = img_id
        return map_name_2_id


class FSC147WithDensityMapSimilarityStitched(Dataset):

    def __init__(
            self, data_path, img_size, split='train', num_objects=3,
            tiling_p=0.5, zero_shot=False, skip_cars=False
    ):
        self.split = split
        self.data_path = data_path
        self.horizontal_flip_p = 0.5
        self.tiling_p = tiling_p
        self.img_size = img_size
        self.resize = T.Resize((img_size, img_size), antialias=True)
        self.resize_1024 = T.Resize((img_size, 1024), antialias=True)
        self.jitter = T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8)
        self.num_objects = num_objects
        self.zero_shot = zero_shot
        with open(
                os.path.join(self.data_path, 'Train_Test_Val_FSC_147.json'), 'rb'
        ) as file:
            splits = json.load(file)
            self.image_names = splits[split]
        with open(
                os.path.join(self.data_path, 'annotation_FSC147_384.json'), 'rb'
        ) as file:
            self.annotations = json.load(file)
        with open(
                os.path.join(self.data_path, 'ImageClasses_FSC147.txt'), 'r'
        ) as file:
            self.classes = dict([
                (line.strip().split()[0], ' '.join(line.strip().split()[1:]))
                for line in file.readlines()
            ])
        if skip_cars:
            with open(
                    os.path.join(self.data_path, 'ImageClasses_FSC147.txt'), 'r'
            ) as file:
                classes = dict([
                    (line.strip().split()[0], ' '.join(line.strip().split()[1:]))
                    for line in file.readlines()
                ])
            print(len(self.image_names))
            self.image_names = [
                img_name for img_name in self.image_names if classes[img_name] != 'cars'
            ]
        print(len(self.image_names))
        if split == 'val' or split == 'test':
            self.labels = COCO(os.path.join(self.data_path, 'annotations', 'instances_' + split + '.json'))
            self.img_name_to_ori_id = self.map_img_name_to_ori_id()
        self.train_stitched = True
        self.test_stitched = False
        self.scale_opt = True
        self.diag_opt = 70

    def __getitem__(self, idx: int) -> Tuple[Tensor, List[Tensor], Tensor]:
        img_name = list(self.annotations.keys())[idx]
        img = Image.open(os.path.join(
            self.data_path,
            'images_384_VarV2',
            img_name
        ))
        w, h = img.size

        img = T.Compose([
            T.ToTensor(),
            self.resize,
        ])(img)
        bboxes = torch.tensor(
            self.annotations[img_name]['box_examples_coordinates'],
            dtype=torch.float32
        )[:3, [0, 2], :].reshape(-1, 4)[:self.num_objects, ...]
        bboxes = bboxes / torch.tensor([w, h, w, h]) * self.img_size
        ids = torch.tensor([1] * self.num_objects + [2] * self.num_objects)

        density_map = torch.from_numpy(np.load(os.path.join(
            self.data_path,
            'gt_density_map_adaptive_512_512_object_VarV2',
            os.path.splitext(img_name)[0] + '.npy',
        ))).unsqueeze(0)

        if self.train_stitched:
            img_class = self.classes[self.image_names[idx]]
            candidate_images = [im for im, cl in self.classes.items() if cl != img_class]
            sampled = [candidate_images[j] for j in torch.randperm(len(candidate_images))[:1]][0]
            sampled_img = Image.open(os.path.join(
                self.data_path,
                'images_384_VarV2',
                sampled
            ))
            w, h = sampled_img.size
            sampled_img = T.Compose([
                T.ToTensor(),
                self.resize,
            ])(sampled_img)
            bboxes_second = torch.tensor(
                self.annotations[sampled]['box_examples_coordinates'],
                dtype=torch.float32
            )[:3, [0, 2], :].reshape(-1, 4)[:self.num_objects, ...]
            bboxes_second = bboxes_second / torch.tensor([w, h, w, h]) * self.img_size
            new_img = torch.tensor(np.zeros((3, 512, 1024)), dtype=torch.float32)
            new_img[:, 0:512, 0:512] = img[:, 0:512, 0:512]
            new_img[:, 0:512, 512:1024] = sampled_img[:, 0:512, 0:512]
            new_density_map = torch.tensor(np.zeros((1, 512, 1024)), dtype=torch.float32)
            new_density_map[0, 0:512, 0:512] = density_map[:, 0:512, 0:512]
            bboxes_ = bboxes
            bboxes_second_ = bboxes_second + torch.tensor(np.array([512, 0, 512, 0]))
            bboxes = torch.cat((bboxes_, bboxes_second_), dim=0)
            img = new_img
            img = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)

        return img, bboxes, ids, density_map, idx

    def get_gt_bboxes(self, idxs):
        if self.split == 'val' or self.split == 'test':

            l = []
            factors = []
            for i in idxs:
                img_name = list(self.annotations.keys())[i]
                img = Image.open(os.path.join(
                    self.data_path,
                    'images_384_VarV2',
                    img_name
                ))
                w1, h1 = img.size
                coco_im_id = self.img_name_to_ori_id[img_name]
                anno_ids = self.labels.getAnnIds([coco_im_id])
                annos = self.labels.loadAnns(anno_ids)
                box_centers = list()
                whs = list()
                xyxy_boxes = list()
                for anno in annos:
                    bbox = anno["bbox"]
                    x1, y1, w, h = bbox
                    box_centers.append([x1 + w / 2, y1 + h / 2])
                    whs.append([w, h])
                    xyxy_boxes.append([x1, y1, x1 + w, y1 + h])
                xyxy_boxes = np.array(xyxy_boxes, dtype=np.float32)
                xyxy_boxes = xyxy_boxes / torch.tensor([w1, h1, w1, h1]) * self.img_size
                factors.append(torch.tensor([w1, h1, w1, h1]) / self.img_size)
                l.append(xyxy_boxes)
            return l, factors

    def __len__(self):
        return len(self.image_names)

    def map_img_name_to_ori_id(self, ):
        all_coco_imgs = self.labels.imgs
        map_name_2_id = dict()
        for k, v in all_coco_imgs.items():
            img_id = v["id"]
            img_name = v["file_name"]
            map_name_2_id[img_name] = img_id
        return map_name_2_id
