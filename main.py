import argparse
import os
import random

import numpy as np
import torch
import torchvision.transforms as T
from models.dave import build_model
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.arg_parser import get_argparser
from utils.data import FSC147WithDensityMapSCALE2BOX, pad_image, FSC147WithDensityMapDOWNSIZE
from utils.data_lvis import FSCD_LVIS_Dataset_SCALE

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
import json

DATASETS = {
    'fsc_box': FSC147WithDensityMapSCALE2BOX,
    'fsc_downscale': FSC147WithDensityMapDOWNSIZE,
    'lvis': FSCD_LVIS_Dataset_SCALE
}


@torch.no_grad()
def evaluate(args):
    gpu = 0
    torch.cuda.set_device(gpu)
    device = torch.device(gpu)

    model = DataParallel(
        build_model(args).to(device),
        device_ids=[gpu],
        output_device=gpu
    )

    model.load_state_dict(
        torch.load(os.path.join(args.model_path, args.model_name + '.pth'))['model'], strict=False
    )
    pretrained_dict_feat = {k.split("feat_comp.")[1]: v for k, v in
                            torch.load(os.path.join(args.model_path, 'verification.pth'))[
                                'model'].items() if 'feat_comp' in k}
    model.module.feat_comp.load_state_dict(pretrained_dict_feat)

    for split in ['val', 'test']:
        print(split)
        test = DATASETS['fsc_box'](
            args.data_path,
            args.image_size,
            split=split,
            num_objects=args.num_objects,
            tiling_p=args.tiling_p,
            zero_shot=args.zero_shot or args.orig_dmaps,
        )

        test_loader = DataLoader(
            test,
            batch_size=args.batch_size,
            drop_last=False,
            num_workers=args.num_workers
        )
        ae = torch.tensor(0.0).to(device)
        se = torch.tensor(0.0).to(device)

        model.eval()

        predictions = dict()
        predictions["categories"] = [{"name": "fg", "id": 1}]
        predictions["images"] = list()
        predictions["annotations"] = list()
        anno_id = 1
        box_err = []
        err = []
        num_objects = []
        for img, bboxes, density_map, ids, scale_x, scale_y, shape in tqdm(test_loader):

            img = img.to(device)
            bboxes = bboxes.to(device)
            density_map = density_map.to(device)
            model = model.to(device)
            out, aux, tblr, boxes_pred = model(img, bboxes, test.image_names[ids[0].item()])
            gt_bboxes, resize_factors = test.get_gt_bboxes(ids)
            boxes_pred = [boxes_pred]

            boxes_pred[0].box = boxes_pred[0].box / torch.tensor([scale_y[0], scale_x[0], scale_y[0], scale_x[0]])
            boxes_pred[0].box = boxes_pred[0].box * resize_factors[0]
            areas = boxes_pred[0].area()
            boxes_xywh = boxes_pred[0].convert("xywh")
            img_info = {
                "id": test.map_img_name_to_ori_id()[test.image_names[ids[0].item()]],
                "file_name": "None",
            }
            scores = boxes_xywh.fields['scores']
            for i in range(len(boxes_pred[0].box)):
                box = boxes_xywh.box[i]
                anno = {
                    "id": anno_id,
                    "image_id": test.map_img_name_to_ori_id()[test.image_names[ids[0].item()]],
                    "area": int(areas[0].item()),
                    "bbox": [int(box[0].item()), int(box[1].item()), int(box[2].item()), int(box[3].item())],
                    "category_id": 1,
                    "score": float(scores[i].item()),
                }
                anno_id += 1
                predictions["annotations"].append(anno)
            predictions["images"].append(img_info)
            err.append(torch.abs(
                density_map.flatten(1).sum(dim=1) - out[:, :, :shape[1], :shape[2]].flatten(1).sum(dim=1)).item())
            box_err.append(abs(density_map.flatten(1).sum(dim=1).item() - len(boxes_pred[0].box)))
            num_objects.append(density_map.flatten(1).sum(dim=1).item())
            ae += torch.abs(
                density_map.flatten(1).sum(dim=1) - out[:, :, :shape[1], :shape[2]].flatten(1).sum(dim=1)
            ).sum()
            se += ((density_map.flatten(1).sum(dim=1) - out[:, :, :shape[1], :shape[2]].flatten(1).sum(dim=1)
                    ) ** 2).sum()

        print("END")
        print(
            f"{split} set",
            f"MAE {ae.item() / len(test)} RMSE {torch.sqrt(se / len(test)).item()}",
        )

        with open("../DAVE_3_shot" + "_" + split + ".json", "w") as handle:
            json.dump(predictions, handle)


@torch.no_grad()
def eval_0shot(args):
    print("0shot")
    if args.skip_test:
        return

    args.zero_shot = True
    gpu = 0
    torch.cuda.set_device(gpu)
    device = torch.device(gpu)

    assert args.backbone in ['resnet18', 'resnet50', 'resnet101']
    assert args.reduction in [4, 8, 16]

    model = DataParallel(
        build_model(args).to(device),
        device_ids=[gpu],
        output_device=gpu
    )

    model.load_state_dict(
        torch.load(os.path.join(args.model_path, 'DAVE_0_shot.pth'))['model'], strict=False
    )
    pretrained_dict_box = {k.split("box_predictor.")[1]: v for k, v in
                           torch.load(os.path.join(args.model_path, 'DAVE_0_shot.pth'))[
                               'model'].items() if 'box_predictor' in k}
    model.module.box_predictor.load_state_dict(pretrained_dict_box)
    pretrained_dict_feat = {k.split("feat_comp.")[1]: v for k, v in
                            torch.load(os.path.join(args.model_path, 'verification.pth'))[
                                'model'].items() if 'feat_comp' in k}
    model.module.feat_comp.load_state_dict(pretrained_dict_feat)
    backbone_params = dict()
    non_backbone_params = dict()
    fcos_params = dict()
    feat_comp = dict()
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if 'backbone' in n:
            backbone_params[n] = p
            backbone_params[n].requires_grad = False
        elif 'box_predictor' in n:
            fcos_params[n] = p
        elif 'feat_comp' in n:
            feat_comp[n] = p
            feat_comp[n].requires_grad = False
        else:
            non_backbone_params[n] = p
            non_backbone_params[n].requires_grad = False

    for split in ['val', 'test']:
        test = DATASETS['fsc_box'](
            args.data_path,
            args.image_size,
            split=split,
            num_objects=args.num_objects,
            tiling_p=args.tiling_p,
            zero_shot=args.zero_shot or args.orig_dmaps,
        )

        test_loader = DataLoader(
            test,
            batch_size=args.batch_size,
            drop_last=False,
            num_workers=0
        )

        ae = torch.tensor(0.0).to(device)
        se = torch.tensor(0.0).to(device)
        model.eval()

        predictions = dict()
        predictions["categories"] = [{"name": "fg", "id": 1}]
        predictions["images"] = list()
        predictions["annotations"] = list()
        anno_id = 1

        for img, bboxes, density_map, ids, scale_x, scale_y, shape, classes in tqdm(test_loader):

            img = img.to(device)
            bboxes = bboxes.to(device)
            density_map = density_map.to(device)
            model = model.to(device)
            out, aux, tblr, boxes_pred = model(img, bboxes, test.image_names[ids[0].item()], classes=classes)

            boxes_predicted = boxes_pred.box
            scale_y = min(1, 50 / (boxes_predicted[:, 2] - boxes_predicted[:, 0]).mean())
            scale_x = min(1, 50 / (boxes_predicted[:, 3] - boxes_predicted[:, 1]).mean())

            if scale_x < 1 or scale_y < 1:
                scale_x = (int(args.image_size * scale_x) // 8 * 8) / args.image_size
                scale_y = (int(args.image_size * scale_y) // 8 * 8) / args.image_size
                resize_ = T.Resize((int(args.image_size * scale_x), int(args.image_size * scale_y)), antialias=True)
                img_resized = resize_(img)

                shape = img_resized.shape[1:]
                img_resized = pad_image(img_resized[0]).unsqueeze(0)

            else:
                scale_y = max(1, 11 / (boxes_predicted[:, 2] - boxes_predicted[:, 0]).mean())
                scale_x = max(1, 11 / (boxes_predicted[:, 3] - boxes_predicted[:, 1]).mean())

                if scale_y > 1.9:
                    scale_y = 1.9
                if scale_x > 1.9:
                    scale_x = 1.9

                scale_x = (int(args.image_size * scale_x) // 8 * 8) / args.image_size
                scale_y = (int(args.image_size * scale_y) // 8 * 8) / args.image_size
                resize_ = T.Resize((int(args.image_size * scale_x), int(args.image_size * scale_y)), antialias=True)
                img_resized = resize_(img)
                shape = img_resized.shape[1:]
            if scale_x != 1.0 or scale_y != 1.0:
                out, aux, tblr, boxes_pred = model(img_resized, bboxes, test.image_names[ids[0].item()],
                                                   classes=classes)

            gt_bboxes, resize_factors = test.get_gt_bboxes(ids)
            boxes_pred = [boxes_pred]

            boxes_pred[0].box = boxes_pred[0].box / torch.tensor([scale_y, scale_x, scale_y, scale_x])
            boxes_pred[0].box = boxes_pred[0].box * resize_factors[0]
            areas = boxes_pred[0].area()
            boxes_xywh = boxes_pred[0].convert("xywh")
            img_info = {
                "id": test.map_img_name_to_ori_id()[test.image_names[ids[0].item()]],
                "file_name": "None",
            }
            scores = boxes_xywh.fields['scores']
            for i in range(len(boxes_pred[0].box)):
                box = boxes_xywh.box[i]
                anno = {
                    "id": anno_id,
                    "image_id": test.map_img_name_to_ori_id()[test.image_names[ids[0].item()]],
                    "area": int(areas[0].item()),
                    "bbox": [int(box[0].item()), int(box[1].item()), int(box[2].item()), int(box[3].item())],
                    "category_id": 1,
                    "score": float(scores[i].item()),
                }
                anno_id += 1
                predictions["annotations"].append(anno)
            predictions["images"].append(img_info)

            ae += torch.abs(density_map.flatten(1).sum(dim=1) - out[:, :, :shape[1], :shape[2]].flatten(1).sum(dim=1)
                            ).sum()
            se += ((density_map.flatten(1).sum(dim=1) - out[:, :, :shape[1], :shape[2]].flatten(1).sum(dim=1)
                    ) ** 2).sum()

        print("END")
        print(
            f"{split} set",
            f"MAE {ae.item() / len(test)} RMSE {torch.sqrt(se / len(test)).item()}",
        )
        with open("DAVE_0shot" + "_" + split + ".json", "w") as handle:
            json.dump(predictions, handle)


@torch.no_grad()
def eval_0shot_multicat(args):
    args.zero_shot = True
    gpu = 0
    torch.cuda.set_device(gpu)
    device = torch.device(gpu)

    assert args.backbone in ['resnet18', 'resnet50', 'resnet101']
    assert args.reduction in [4, 8, 16]

    model = DataParallel(
        build_model(args).to(device),
        device_ids=[gpu],
        output_device=gpu
    )

    model.load_state_dict(
        torch.load(os.path.join(args.model_path, 'DAVE_0_shot.pth'))['model'], strict=False
    )
    pretrained_dict_box = {k.split("box_predictor.")[1]: v for k, v in
                           torch.load(os.path.join(args.model_path, 'DAVE_0_shot.pth'))[
                               'model'].items() if 'box_predictor' in k}
    model.module.box_predictor.load_state_dict(pretrained_dict_box)
    pretrained_dict_feat = {k.split("feat_comp.")[1]: v for k, v in
                            torch.load(os.path.join(args.model_path, 'verification.pth'))[
                                'model'].items() if 'feat_comp' in k}
    model.module.feat_comp.load_state_dict(pretrained_dict_feat)
    backbone_params = dict()
    non_backbone_params = dict()
    fcos_params = dict()
    feat_comp = dict()
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if 'backbone' in n:
            backbone_params[n] = p
            backbone_params[n].requires_grad = False
        elif 'box_predictor' in n:
            fcos_params[n] = p
        elif 'feat_comp' in n:
            feat_comp[n] = p
            feat_comp[n].requires_grad = False
        else:
            non_backbone_params[n] = p
            non_backbone_params[n].requires_grad = False

    ae = torch.tensor(0.0).to(device)
    se = torch.tensor(0.0).to(device)
    num_preds = 0
    for split in ['val', 'test']:
        test = DATASETS['fsc_box'](
            args.data_path,
            args.image_size,
            split=split,
            num_objects=args.num_objects,
            tiling_p=args.tiling_p,
            zero_shot=args.zero_shot or args.orig_dmaps,
        )

        test_loader = DataLoader(
            test,
            batch_size=args.batch_size,
            drop_last=False,
            num_workers=0
        )

        model.eval()

        predictions = dict()
        predictions["categories"] = [{"name": "fg", "id": 1}]
        predictions["images"] = list()
        predictions["annotations"] = list()
        anno_id = 1

        mutlicat_images = ['2143', '4840', '4885', '5374', '343', '6107', '4105', '5564', '4109', '2184', '5388',
                           '4115', '4828', '4825', '4851', '911', '4608', '6266', '232', '5390', '244', '7247', '236',
                           '5365', '5155', '2953', '4920', '243', '3478', '3783', '336', '4106', '595']

        for img, bboxes, density_map, ids, scale_x, scale_y, shape, classes in tqdm(test_loader):
            bboxes = torch.zeros_like(bboxes)
            img_name = test.image_names[ids[0].item()][:-4]
            if img_name not in mutlicat_images:
                continue
            num_preds += 1
            img = img.to(device)
            bboxes = bboxes.to(device)
            density_map = density_map.to(device)
            model = model.to(device)
            out, aux, tblr, boxes_pred = model(img, bboxes, test.image_names[ids[0].item()], classes=classes)

            boxes_predicted = boxes_pred.box
            scale_y = min(1, 50 / (boxes_predicted[:, 2] - boxes_predicted[:, 0]).mean())
            scale_x = min(1, 50 / (boxes_predicted[:, 3] - boxes_predicted[:, 1]).mean())

            if scale_x < 1 or scale_y < 1:
                scale_x = (int(args.image_size * scale_x) // 8 * 8) / args.image_size
                scale_y = (int(args.image_size * scale_y) // 8 * 8) / args.image_size
                resize_ = T.Resize((int(args.image_size * scale_x), int(args.image_size * scale_y)), antialias=True)
                img_resized = resize_(img)

                img = pad_image(img_resized[0]).unsqueeze(0)

            else:
                scale_y = max(1, 11 / (boxes_predicted[:, 2] - boxes_predicted[:, 0]).mean())
                scale_x = max(1, 11 / (boxes_predicted[:, 3] - boxes_predicted[:, 1]).mean())

                if scale_y > 1.9:
                    scale_y = 1.9
                if scale_x > 1.9:
                    scale_x = 1.9

                scale_x = (int(args.image_size * scale_x) // 8 * 8) / args.image_size
                scale_y = (int(args.image_size * scale_y) // 8 * 8) / args.image_size
                resize_ = T.Resize((int(args.image_size * scale_x), int(args.image_size * scale_y)), antialias=True)
                img = resize_(img)

            if scale_x != 1.0 or scale_y != 1.0:
                out, aux, tblr, boxes_pred = model(img, bboxes, test.image_names[ids[0].item()],
                                                   classes=classes)

            shape = img.shape[1:]

            gt_bboxes, resize_factors = test.get_gt_bboxes(ids)
            boxes_pred = [boxes_pred]

            boxes_pred[0].box = boxes_pred[0].box / torch.tensor([scale_y, scale_x, scale_y, scale_x])
            boxes_pred[0].box = boxes_pred[0].box * resize_factors[0]
            areas = boxes_pred[0].area()
            boxes_xywh = boxes_pred[0].convert("xywh")
            img_info = {
                "id": test.map_img_name_to_ori_id()[test.image_names[ids[0].item()]],
                "file_name": "None",
            }
            scores = boxes_xywh.fields['scores']
            for i in range(len(boxes_pred[0].box)):
                box = boxes_xywh.box[i]
                anno = {
                    "id": anno_id,
                    "image_id": test.map_img_name_to_ori_id()[test.image_names[ids[0].item()]],
                    "area": int(areas[0].item()),
                    "bbox": [int(box[0].item()), int(box[1].item()), int(box[2].item()), int(box[3].item())],
                    "category_id": 1,
                    "score": float(scores[i].item()),
                }
                anno_id += 1
                predictions["annotations"].append(anno)
            predictions["images"].append(img_info)

            ae += torch.abs(

                density_map.flatten(1).sum(dim=1) - out[:, :, :shape[1], :shape[2]].flatten(1).sum(dim=1)
            ).sum()
            se += ((
                           density_map.flatten(1).sum(dim=1) - out[:, :, :shape[1], :shape[2]].flatten(1).sum(dim=1)
                   ) ** 2).sum()

        if num_preds > 0:
            print(
                f"{split} set",
                f"MAE {ae.item() / num_preds} RMSE {torch.sqrt(se / num_preds).item()}",
            )


@torch.no_grad()
def evaluate_LVIS(args):
    gpu = 0
    torch.cuda.set_device(gpu)
    device = torch.device(gpu)

    assert args.backbone in ['resnet18', 'resnet50', 'resnet101']
    assert args.reduction in [4, 8, 16]

    model = DataParallel(
        build_model(args).to(device),
        device_ids=[gpu],
        output_device=gpu
    )

    model.load_state_dict(
        torch.load(os.path.join(args.model_path, args.model_name + '.pth'))['model'], strict=False
    )

    pretrained_dict_feat = {k.split("feat_comp.")[1]: v for k, v in
                            torch.load(os.path.join(args.model_path, 'verification.pth'))[
                                'model'].items() if 'feat_comp' in k}
    model.module.feat_comp.load_state_dict(pretrained_dict_feat)

    for split in ['test']:
        test = DATASETS['lvis'](
            args.data_path,
            args.image_size,
            split=split,
            num_objects=args.num_objects,
            tiling_p=args.tiling_p,
            zero_shot=args.zero_shot or args.orig_dmaps,
            unseen=args.unseen
        )

        test_loader = DataLoader(
            test,
            batch_size=args.batch_size,
            drop_last=False,
            num_workers=args.num_workers
        )

        ae = torch.tensor(0.0).to(device)
        se = torch.tensor(0.0).to(device)
        model.eval()

        predictions = dict()
        predictions["categories"] = [{"name": "fg", "id": 1}]
        predictions["images"] = list()
        predictions["annotations"] = list()
        anno_id = 1

        for img, bboxes, ids, scale_x, scale_y, shape in test_loader:
            gt_bboxes, resize_factors = test.get_gt_bboxes(ids)
            img = img.to(device)
            bboxes = bboxes.to(device)
            model = model.to(device)
            out, aux, tblr, boxes_pred = model(img, bboxes, '')

            boxes_pred = [boxes_pred]
            boxes_pred[0].box = boxes_pred[0].box / torch.tensor([scale_y[0], scale_x[0], scale_y[0], scale_x[0]])
            boxes_pred[0].box = boxes_pred[0].box * resize_factors[0]
            areas = boxes_pred[0].area()
            boxes_xywh = boxes_pred[0].convert("xywh")
            img_info = {
                "id": ids[0].item(),
                "file_name": "None",
            }
            scores = boxes_xywh.fields['scores']
            for i in range(len(boxes_pred[0].box)):
                box = boxes_xywh.box[i]
                anno = {
                    "id": anno_id,
                    "image_id": ids[0].item(),
                    "area": int(areas[0].item()),
                    "bbox": [int(box[0].item()), int(box[1].item()), int(box[2].item()), int(box[3].item())],
                    "category_id": 1,
                    "score": float(scores[i].item()),
                }
                anno_id += 1
                predictions["annotations"].append(anno)
            predictions["images"].append(img_info)

            ae += torch.abs(len(gt_bboxes[0]) - out[:, :, :shape[1], :shape[2]].flatten(1).sum(dim=1)
                            ).sum()
            se += ((len(gt_bboxes[0]) - out[:, :, :shape[1], :shape[2]].flatten(1).sum(dim=1)
                    ) ** 2).sum()

        print(
            f"{split} set",
            f"MAE {ae.item() / len(test)} RMSE {torch.sqrt(se / len(test)).item()}",
        )

        name = "FSCD_LVIS_unseen" if args.unseen else "FSCD_LVIS"
        with open(name + "_" + split + ".json", "w") as handle:
            json.dump(predictions, handle)


@torch.no_grad()
def evaluate_multicat(args):
    gpu = 0
    torch.cuda.set_device(gpu)
    device = torch.device(gpu)

    model = DataParallel(
        build_model(args).to(device),
        device_ids=[gpu],
        output_device=gpu
    )

    model.load_state_dict(
        torch.load(os.path.join(args.model_path, 'DAVE_3_shot.pth'))['model'], strict=False
    )

    pretrained_dict_feat = {k.split("feat_comp.")[1]: v for k, v in
                            torch.load(os.path.join(args.model_path, 'verification.pth'))[
                                'model'].items() if 'feat_comp' in k}
    model.module.feat_comp.load_state_dict(pretrained_dict_feat)
    backbone_params = dict()
    non_backbone_params = dict()
    fcos_params = dict()
    feat_comp = dict()
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if 'backbone' in n:
            backbone_params[n] = p
            backbone_params[n].requires_grad = False
        elif 'box_predictor' in n:
            fcos_params[n] = p
        elif 'feat_comp' in n:
            feat_comp[n] = p
            feat_comp[n].requires_grad = False
        else:
            non_backbone_params[n] = p
            non_backbone_params[n].requires_grad = False

    ae = torch.tensor(0.0).to(device)
    se = torch.tensor(0.0).to(device)

    box_err = []
    err = []
    num_objects = []
    for split in ['val', 'test']:
        print(split)
        test = DATASETS['fsc_box'](
            args.data_path,
            args.image_size,
            split=split,
            num_objects=args.num_objects,
            tiling_p=args.tiling_p,
            zero_shot=args.zero_shot or args.orig_dmaps,
        )

        test_loader = DataLoader(
            test,
            batch_size=args.batch_size,
            drop_last=False,
            num_workers=args.num_workers
        )

        model.eval()

        predictions = dict()
        predictions["categories"] = [{"name": "fg", "id": 1}]
        predictions["images"] = list()
        predictions["annotations"] = list()
        anno_id = 1

        mutlicat_images = ['2143', '4840', '4885', '5374', '343', '6107', '4105', '5564', '4109', '2184', '5388',
                           '4115', '4828', '4825', '4851', '911', '4608', '6266', '232', '5390', '244', '7247', '236',
                           '5365', '5155', '2953', '4920', '243', '3478', '3783', '336', '4106', '595']
        for img, bboxes, density_map, ids, scale_x, scale_y, shape in test_loader:
            img_name = test.image_names[ids[0].item()][:-4]
            if img_name not in mutlicat_images:
                continue

            img = img.to(device)
            bboxes = bboxes.to(device)
            density_map = density_map.to(device)
            model = model.to(device)
            out, aux, tblr, boxes_pred = model(img, bboxes, test.image_names[ids[0].item()])

            gt_bboxes, resize_factors = test.get_gt_bboxes(ids)
            gt_bboxes = gt_bboxes[0]

            gt_bboxes = gt_bboxes * torch.tensor([scale_y[0], scale_x[0], scale_y[0], scale_x[0]])
            gt_bboxes, resize_factors = test.get_gt_bboxes(ids)
            boxes_pred = [boxes_pred]

            boxes_pred[0].box = boxes_pred[0].box / torch.tensor([scale_y[0], scale_x[0], scale_y[0], scale_x[0]])
            boxes_pred[0].box = boxes_pred[0].box * resize_factors[0]
            areas = boxes_pred[0].area()
            boxes_xywh = boxes_pred[0].convert("xywh")
            img_info = {
                "id": test.map_img_name_to_ori_id()[test.image_names[ids[0].item()]],
                "file_name": "None",
            }
            scores = boxes_xywh.fields['scores']
            for i in range(len(boxes_pred[0].box)):
                box = boxes_xywh.box[i]
                anno = {
                    "id": anno_id,
                    "image_id": test.map_img_name_to_ori_id()[test.image_names[ids[0].item()]],
                    "area": int(areas[0].item()),
                    "bbox": [int(box[0].item()), int(box[1].item()), int(box[2].item()), int(box[3].item())],
                    "category_id": 1,
                    "score": float(scores[i].item()),
                }
                anno_id += 1
                predictions["annotations"].append(anno)
            predictions["images"].append(img_info)
            err.append(torch.abs(
                # err
                density_map.flatten(1).sum(dim=1) - out[:, :, :shape[1], :shape[2]].flatten(1).sum(dim=1)).item())
            box_err.append(abs(density_map.flatten(1).sum(dim=1).item() - len(boxes_pred[0].box)))
            num_objects.append(density_map.flatten(1).sum(dim=1).item())
            ae += torch.abs(
                density_map.flatten(1).sum(dim=1) - out[:, :, :shape[1], :shape[2]].flatten(1).sum(dim=1)
            ).sum()
            se += ((
                           density_map.flatten(1).sum(dim=1) - out[:, :, :shape[1], :shape[2]].flatten(1).sum(dim=1)
                   ) ** 2).sum()

        print(
            f"{split} set",
            f"MAE {ae.item() / len(box_err)} RMSE {torch.sqrt(se / len(box_err)).item()}",
        )


if __name__ == '__main__':
    print("DAVE")
    parser = argparse.ArgumentParser('DAVE', parents=[get_argparser()])
    args = parser.parse_args()
    print(args)
    if args.task == 'lvis':
        evaluate_LVIS(args)
    elif not args.zero_shot and args.eval_multicat:
        evaluate_multicat(args)
    elif args.zero_shot and args.eval_multicat:
        eval_0shot_multicat(args)
    elif args.zero_shot:
        eval_0shot(args)
    else:
        evaluate(args)
