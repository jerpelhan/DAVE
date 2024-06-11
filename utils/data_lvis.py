import json
from typing import Tuple, List
from torch import Tensor
import os
import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from torchvision import transforms as transforms
from torchvision import transforms as T
import torchvision.transforms.functional as FT
from torchvision.transforms import functional as TVF


def pad_image(image_tensor):
    target_size = (512, 512)
    original_size = image_tensor.size()[1:]

    # Calculate the amount of padding needed for each dimension
    pad_height = max(target_size[0] - original_size[0], 0)
    pad_width = max(target_size[1] - original_size[1], 0)

    # Pad the image tensor on the bottom or right side only
    padded_image = FT.pad(image_tensor, (0, 0, pad_width, pad_height))

    return padded_image


def tiling_augmentation(img, bboxes, density_map, resize, jitter, tile_size, hflip_p):
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
        bboxes[:, [0, 2]] = x_target - bboxes[:, [2, 0]]  # TODO change
    bboxes = bboxes / torch.tensor([x_tile, y_tile, x_tile, y_tile])
    return img, bboxes, density_map


class FSCD_LVISDataset(Dataset):
    def __init__(
            self, args, split="train",
    ):
        data_path = args.data_path
        pseudo_label_file = "pseudo_lvis_" + split + "_cxcywh.json"
        self.coco = COCO(os.path.join(data_path, "annotations_old", pseudo_label_file))
        self.image_ids = self.coco.getImgIds()

        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        )
        self.num_objects = 3
        self.img_path = os.path.join(data_path, "images", "all_images")
        self.count_anno_file = os.path.join(data_path, "annotations_old", "count_" + split + ".json")
        self.count_anno = self.load_json(self.count_anno_file)

    def load_json(self, json_file):
        with open(json_file, "r") as f:
            data = json.load(f)
        return data

    def __len__(self):
        return len(self.image_ids)

    def get_gt(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.coco.loadImgs([img_id])[0]
        img_file = img_info["file_name"]
        img = Image.open(os.path.join(self.img_path, img_file))
        img = img.convert("RGB")

        wh = img.size
        ann_ids = self.coco.getAnnIds([img_id])
        anns = self.coco.loadAnns(ids=ann_ids)

        bboxes = np.array([instance["bbox"] for instance in anns], dtype=np.float32)

        ex_bboxes = self.count_anno["annotations"][idx]["boxes"]

        ex_rects = []
        for bbox in ex_bboxes[:3]:
            x, y, w, h = bbox
            x1, y1 = x, y
            x2, y2 = x + w, y + h
            ex_rects.append([x1, y1, x2, y2])

        ex_rects = np.array(ex_rects, dtype=np.float32)
        ex_rects[:, 0] = np.clip(ex_rects[:, 0], 0, img.size[0] - 1)
        ex_rects[:, 1] = np.clip(ex_rects[:, 1], 0, img.size[1] - 1)
        ex_rects[:, 2] = np.clip(ex_rects[:, 2], 0, img.size[0] - 1)
        ex_rects[:, 3] = np.clip(ex_rects[:, 3], 0, img.size[1] - 1)
        return img, bboxes, ex_rects, wh

    def __getitem__(self, index):
        img, bboxes, ex_rects, wh = self.get_gt(index)
        img_w, img_h = img.size

        orig_size = np.array([img_h, img_w])

        resize_w = 32 * int(img_w / 32)
        resize_h = 32 * int(img_h / 32)
        img = img.resize((resize_w, resize_h))
        img = self.transform(img)

        img_res = np.array([img_w, img_h, img_w, img_h], dtype=np.float32)
        bboxes = bboxes.astype(np.float32) / img_res[None, :]

        ex_rects = ex_rects.astype(np.float32) / img_res[None, :]
        labels = torch.zeros([bboxes.shape[0]], dtype=torch.int64)

        ret = {
            "image": img,
            "boxes": bboxes,
            "ex_rects": ex_rects,
            "origin_wh": wh,
            "labels": labels,
            "orig_size": orig_size,
        }
        return ret


class FSCD_LVIS_Dataset_SCALE(Dataset):
    def __init__(
            self, data_path,
            image_size,
            split,
            num_objects,
            tiling_p,
            zero_shot,
            unseen=False
    ):
        super().__init__()
        print("This data is fscd 147 test set, split: {} unseen: ".format(split) + str(unseen))
        self.split = split
        data_path = "/".join(data_path.split("/")[:-1]) + "/FSCD_LVIS/"
        self.img_size = image_size
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        )
        self.horizontal_flip_p = 0.5
        self.tiling_p = 0.5
        self.jitter = T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8)
        self.data_path = data_path
        if self.split != 'val' and unseen:
            pseudo_label_file = "unseen_instances_" + split + ".json"
        else:
            pseudo_label_file = "instances_" + split + ".json"
        self.coco = COCO(os.path.join(data_path, 'annotations', pseudo_label_file))
        self.image_ids = self.coco.getImgIds()
        self.resize = T.Resize((image_size, image_size), antialias=True)
        self.num_objects = num_objects
        self.img_path = os.path.join(data_path, "images")
        if self.split != 'val' and unseen:
            self.count_anno_file = os.path.join(data_path, "annotations", "unseen_count_" + split + ".json")
        else:
            self.count_anno_file = os.path.join(data_path, "annotations", "count_" + split + ".json")
        self.count_anno = self.load_json(self.count_anno_file)
        self.counter = 0

    def load_json(self, json_file):
        with open(json_file, "r") as f:
            data = json.load(f)
        return data

    def __len__(self):
        return len(self.image_ids)

    def get_gt(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.coco.loadImgs([img_id])[0]
        img_file = img_info["file_name"]
        img = Image.open(os.path.join(self.img_path, img_file))
        wh = img.size
        ann_ids = self.coco.getAnnIds([img_id])
        anns = self.coco.loadAnns(ids=ann_ids)

        bboxes = np.array([instance["bbox"] for instance in anns], dtype=np.float32)

        ex_bboxes = self.count_anno["annotations"][idx]["boxes"]

        ex_rects = []
        for bbox in ex_bboxes[:3]:
            x, y, w, h = bbox
            x1, y1 = x, y
            x2, y2 = x + w, y + h
            ex_rects.append([x1, y1, x2, y2])
        corrected_bboxes = []
        for bbox in bboxes:
            x, y, w, h = bbox
            x1, y1 = x, y
            x2, y2 = x + w, y + h
            corrected_bboxes.append([x1, y1, x2, y2])
        corrected_bboxes = np.array(corrected_bboxes, dtype=np.float32)
        ex_rects = np.array(ex_rects, dtype=np.float32)

        return img, corrected_bboxes, ex_rects, wh

    def get_gt_true_IDX(self, idx):
        img_id = idx
        img_info = self.coco.loadImgs([img_id])[0]
        img_file = img_info["file_name"]
        img = Image.open(os.path.join(self.img_path, img_file))
        wh = img.size
        ann_ids = self.coco.getAnnIds([img_id])
        anns = self.coco.loadAnns(ids=ann_ids)

        bboxes = np.array([instance["bbox"] for instance in anns], dtype=np.float32)

        corrected_bboxes = []
        for bbox in bboxes:
            x, y, w, h = bbox
            x1, y1 = x, y
            x2, y2 = x + w, y + h
            corrected_bboxes.append([x1, y1, x2, y2])
        corrected_bboxes = np.array(corrected_bboxes, dtype=np.float32)

        return img, corrected_bboxes, wh

    def __getitem__(self, index) -> Tuple[Tensor, List[Tensor], Tensor]:
        img, bboxes, ex_rects, wh = self.get_gt(index)
        w, h = img.size
        img_id = self.image_ids[index]
        img_info = self.coco.loadImgs([img_id])[0]
        img_file = img_info["file_name"]

        img = T.Compose([
            T.ToTensor(),
            self.resize,
        ])(img)

        if img.shape[0] < 3:
            img = torch.stack([img[0, :, :], img[0, :, :], img[0, :, :]], 0)

        if self.split != 'train':
            img = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)

        ex_rects = torch.tensor(
            ex_rects,
            dtype=torch.float32
        )[:self.num_objects, ...]
        ex_rects = ex_rects / torch.tensor([w, h, w, h]) * self.img_size
        bboxes = torch.tensor(
            bboxes,
            dtype=torch.float32
        )
        bboxes = bboxes / torch.tensor([w, h, w, h]) * self.img_size

        scale_x = min(1, 50 / (bboxes[:, 2] - bboxes[:, 0]).mean())
        scale_y = min(1, 50 / (bboxes[:, 3] - bboxes[:, 1]).mean())
        shape = img.shape
        if scale_x < 1 or scale_y < 1:

            scale_y = (int(img.shape[1] * scale_y) // 8 * 8) / img.shape[1]
            scale_x = (int(img.shape[2] * scale_x) // 8 * 8) / img.shape[2]
            resize_ = T.Resize((int(img.shape[1] * scale_y), int(img.shape[2] * scale_x)), antialias=True)
            img_ = resize_(img)

            shape = img_.shape
            img = pad_image(img_)

        else:
            scale_x = max(1, 11 / (bboxes[:, 2] - bboxes[:, 0]).mean())
            scale_y = max(1, 11 / (bboxes[:, 3] - bboxes[:, 1]).mean())

            if scale_y > 1.9:
                scale_y = 1.9
            if scale_x > 1.9:
                scale_x = 1.9

            scale_y = (int(img.shape[1] * scale_y) // 8 * 8) / img.shape[1]
            scale_x = (int(img.shape[2] * scale_x) // 8 * 8) / img.shape[2]
            resize_ = T.Resize((int(img.shape[1] * scale_y), int(img.shape[2] * scale_x)), antialias=True)
            img = resize_(img)
            shape = img.shape

        ex_rects = ex_rects * torch.tensor([scale_x, scale_y, scale_x, scale_y])
        return img, ex_rects, img_id, scale_y, scale_x, shape


    def get_gt_bboxes(self, idxs):
        if self.split == 'val' or self.split == 'test':
            l = []
            factors = []
            for i in idxs:
                img, bboxes, wh = self.get_gt_true_IDX(i.item())
                w, h = img.size

                bboxes = torch.tensor(
                    bboxes,
                    dtype=torch.float32
                )
                bboxes = bboxes / torch.tensor([w, h, w, h]) * self.img_size
                l.append(bboxes)
                factors.append(torch.tensor([w, h, w, h]) / self.img_size)

            return l, factors


class FSCD_LVIS_Dataset_Test(Dataset):
    def __init__(
            self, data_path,
            image_size,
            split,
            num_objects,
            tiling_p,
            zero_shot,
            unseen
    ):
        super().__init__()
        print("This data is fscd 147 test set, split: {} unseen: ".format(split) + str(unseen))
        self.split = split
        data_path = "/".join(data_path.split("/")[:-1]) + "/FSCD_LVIS/"
        self.img_size = image_size
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        )
        self.horizontal_flip_p = 0.5
        self.tiling_p = 0.5
        self.jitter = T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8)
        self.data_path = data_path
        if self.split != 'val' and unseen:
            pseudo_label_file = "unseen_instances_" + split + ".json"
        else:
            pseudo_label_file = "instances_" + split + ".json"
        self.coco = COCO(os.path.join(data_path, 'annotations', pseudo_label_file))
        self.image_ids = self.coco.getImgIds()
        self.resize = T.Resize((image_size, image_size))
        self.num_objects = num_objects
        self.img_path = os.path.join(data_path, "images")
        if self.split != 'val' and unseen:
            self.count_anno_file = os.path.join(data_path, "annotations", "unseen_count_" + split + ".json")
        else:
            self.count_anno_file = os.path.join(data_path, "annotations", "count_" + split + ".json")
        self.count_anno = self.load_json(self.count_anno_file)
        self.counter = 0
        self.diag_opt = 70
        self.resizing = True

    def load_json(self, json_file):
        with open(json_file, "r") as f:
            data = json.load(f)
        return data

    def __len__(self):
        return len(self.image_ids)

    def get_gt(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.coco.loadImgs([img_id])[0]
        img_file = img_info["file_name"]
        img = Image.open(os.path.join(self.img_path, img_file))
        wh = img.size
        ann_ids = self.coco.getAnnIds([img_id])
        anns = self.coco.loadAnns(ids=ann_ids)

        bboxes = np.array([instance["bbox"] for instance in anns], dtype=np.float32)

        ex_bboxes = self.count_anno["annotations"][idx]["boxes"]

        ex_rects = []
        for bbox in ex_bboxes[:3]:
            x, y, w, h = bbox
            x1, y1 = x, y
            x2, y2 = x + w, y + h
            ex_rects.append([x1, y1, x2, y2])
        corrected_bboxes = []
        for bbox in bboxes:
            x, y, w, h = bbox
            x1, y1 = x, y
            x2, y2 = x + w, y + h
            corrected_bboxes.append([x1, y1, x2, y2])
        corrected_bboxes = np.array(corrected_bboxes, dtype=np.float32)
        ex_rects = np.array(ex_rects, dtype=np.float32)

        return img, corrected_bboxes, ex_rects, wh

    def get_gt_true_IDX(self, idx):
        img_id = idx
        img_info = self.coco.loadImgs([img_id])[0]
        img_file = img_info["file_name"]
        img = Image.open(os.path.join(self.img_path, img_file))
        wh = img.size
        ann_ids = self.coco.getAnnIds([img_id])
        anns = self.coco.loadAnns(ids=ann_ids)

        bboxes = np.array([instance["bbox"] for instance in anns], dtype=np.float32)

        corrected_bboxes = []
        for bbox in bboxes:
            x, y, w, h = bbox
            x1, y1 = x, y
            x2, y2 = x + w, y + h
            corrected_bboxes.append([x1, y1, x2, y2])
        corrected_bboxes = np.array(corrected_bboxes, dtype=np.float32)

        return img, corrected_bboxes, wh

    def __getitem__(self, index) -> Tuple[Tensor, List[Tensor], Tensor]:
        img, bboxes, ex_rects, wh = self.get_gt(index)
        w, h = img.size
        img_id = self.image_ids[index]
        img_info = self.coco.loadImgs([img_id])[0]
        img_file = img_info["file_name"]

        img = T.Compose([
            T.ToTensor(),
            self.resize,
        ])(img)

        if img.shape[0] < 3:
            img = torch.stack([img[0, :, :], img[0, :, :], img[0, :, :]], 0)

        if self.split != 'train':
            img = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)

        ex_rects = torch.tensor(
            ex_rects,
            dtype=torch.float32
        )[:self.num_objects, ...]
        ex_rects = ex_rects / torch.tensor([w, h, w, h]) * self.img_size
        bboxes = torch.tensor(
            bboxes,
            dtype=torch.float32
        )
        bboxes = bboxes / torch.tensor([w, h, w, h]) * self.img_size

        density_map = torch.from_numpy(np.load(os.path.join(
            self.data_path,
            'gt_density_map_adaptive_512_512_object_VarV2',
            os.path.splitext(img_file)[0] + '.npy',
        ))).unsqueeze(0)
        if self.resizing:
            diag = torch.sqrt(
                (ex_rects[:, 0] - ex_rects[:, 2]) ** 2 + (
                            ex_rects[:, 1] - ex_rects[:, 3]) ** 2).sum() / self.num_objects
            scale = self.diag_opt / diag
            if scale < 1:
                if (ex_rects[:, 0] - ex_rects[:, 2]).sum() / (ex_rects[:, 1] - ex_rects[:, 3]).sum() > 2:
                    scale_x = 1.0
                    scale_y = scale
                    self.resize_half = T.Resize((int(512 * scale_x), int(512 * scale_y)))
                    new_img = torch.zeros_like(img)
                    img = self.resize_half(img)
                    new_img[:, 0:int(self.img_size), 0:int(self.img_size * scale)] = img
                    img = new_img
                    original_sum = density_map.sum()
                    new_density = torch.zeros_like(density_map)
                    density_map = self.resize_half(density_map)
                    new_density[:, 0:int(self.img_size), 0:int(self.img_size * scale)] = density_map
                    density_map = new_density / new_density.sum() * original_sum
                    ex_rects = ex_rects * torch.tensor([scale_y, scale_x, scale_y, scale_x])


                elif (ex_rects[:, 1] - ex_rects[:, 3]).sum() / (ex_rects[:, 0] - ex_rects[:, 2]).sum() > 2:
                    scale = scale
                    scale_x = scale
                    scale_y = 1.0
                    self.resize_half = T.Resize((int(512 * scale_x), int(512 * scale_y)))
                    new_img = torch.zeros_like(img)
                    img = self.resize_half(img)
                    new_img[:, 0:int(self.img_size * scale), 0:int(self.img_size)] = img
                    img = new_img
                    original_sum = density_map.sum()
                    new_density = torch.zeros_like(density_map)
                    density_map = self.resize_half(density_map)
                    new_density[:, 0:int(self.img_size * scale), 0:int(self.img_size)] = density_map
                    density_map = new_density / new_density.sum() * original_sum
                    ex_rects = ex_rects * torch.tensor([scale_y, scale_x, scale_y, scale_x])

                else:
                    scale_x = scale
                    scale_y = scale
                    self.resize_half = T.Resize((int(self.img_size * scale), int(self.img_size * scale)))
                    new_img = torch.zeros_like(img)
                    img = self.resize_half(img)
                    new_img[:, 0:int(self.img_size * scale), 0:int(self.img_size * scale)] = img
                    img = new_img
                    original_sum = density_map.sum()
                    new_density = torch.zeros_like(density_map)
                    density_map = self.resize_half(density_map)
                    new_density[:, 0:int(self.img_size * scale), 0:int(self.img_size * scale)] = density_map
                    density_map = new_density / new_density.sum() * original_sum
                    ex_rects = ex_rects * torch.tensor([scale_y, scale_x, scale_y, scale_x])
            else:
                scale_x = 1.0
                scale_y = 1.0
            if not torch.is_tensor(scale_x):
                scale_x = torch.tensor(scale_x)  # ,dtype=torch.long)
            # else:
            #     scale_x = scale_x.to(torch.long)
            if not torch.is_tensor(scale_y):
                scale_y = torch.tensor(scale_y)  # ,dtype=torch.long)

        tiled = False
        if self.split == 'train' and torch.rand(1) < self.tiling_p:
            tiled = True
            tile_size = (torch.rand(1) + 1, torch.rand(1) + 1)
            img, ex_rects, density_map = tiling_augmentation(
                img, ex_rects, density_map, self.resize,
                self.jitter, tile_size, self.horizontal_flip_p
            )

        if self.split == 'train':
            if not tiled:
                img = self.jitter(img)
            img = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)

        if self.resizing:
            if self.split == 'test' or self.split == 'val':
                return img, ex_rects, density_map, img_id, torch.tensor([w, h, w, h]) / self.img_size, scale_x, scale_y
            else:
                return img, ex_rects, density_map, img_id  # , torch.tensor([w, h, w, h]) / self.img_size, bboxes
        else:
            if self.split == 'test' or self.split == 'val':
                return img, ex_rects, density_map, img_id
            else:
                return img, ex_rects, density_map, img_id, torch.tensor([w, h, w, h]) / self.img_size  # , bboxes

    def get_gt_bboxes(self, idxs):
        if self.split == 'val' or self.split == 'test':
            l = []
            factors = []
            for i in idxs:
                img, bboxes, wh = self.get_gt_true_IDX(i.item())
                w, h = img.size

                bboxes = torch.tensor(
                    bboxes,
                    dtype=torch.float32
                )
                bboxes = bboxes / torch.tensor([w, h, w, h]) * self.img_size
                l.append(bboxes)
            return l
