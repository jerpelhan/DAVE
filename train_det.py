from models.dave import build_model
from models.box_prediction import BoxList, boxlist_nms
from utils.data import FSC147WithDensityMapDOWNSIZE
from utils.arg_parser import get_argparser
from utils.losses import Criterion, Detection_criterion
from time import perf_counter
import argparse
import os
from torchvision.ops import box_iou
import torch
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch import distributed as dist
import numpy as np
import skimage
import math

DATASETS = {
    'fsc147': FSC147WithDensityMapDOWNSIZE,
}

def generate_bbox(density_map, tlrb):
    bboxes = []
    for i in range(density_map.shape[0]):
        density = np.array((density_map)[i][0].cpu())
        dmap = np.array((density_map)[i][0].cpu())

        mask = dmap < np.max(dmap) / 3
        dmap[mask] = 0
        a = skimage.feature.peak_local_max(dmap, exclude_border=0)

        boxes = []
        scores = []
        b, l, r, t = tlrb[i]

        for x11, y11 in a:
            box = [y11 - b[x11][y11].item(), x11 - l[x11][y11].item(), y11 + r[x11][y11].item(),
                   x11 + t[x11][y11].item()]
            boxes.append(box)
            scores.append(
                1 - math.fabs(density[int(box[1]): int(box[3]), int(box[0]):int(box[2])].sum() - 1))

        b = BoxList(boxes, (density_map.shape[3], density_map.shape[2]))
        b.fields['scores'] = torch.tensor(scores)
        b = b.clip()
        b = boxlist_nms(b, b.fields['scores'], 0.55)

        bboxes.append(b)
    return bboxes


def reduce_dict(input_dict, average=False):
    with torch.no_grad():
        names = []
        values = []
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


def train(args):
    if args.skip_train:
        print("SKIPPING TRAIN")
        return

    if 'SLURM_PROCID' in os.environ:
        world_size = int(os.environ['SLURM_NTASKS'])
        rank = int(os.environ['SLURM_PROCID'])
        gpu = rank % torch.cuda.device_count()
        print("Running on SLURM", world_size, rank, gpu)
    else:
        world_size = int(os.environ['WORLD_SIZE'])
        rank = int(os.environ['RANK'])
        gpu = int(os.environ['LOCAL_RANK'])

    torch.cuda.set_device(gpu)
    device = torch.device(gpu)

    dist.init_process_group(
        backend='nccl', init_method='env://',
        world_size=world_size, rank=rank
    )

    assert args.backbone in ['resnet18', 'resnet50', 'resnet101']
    assert args.reduction in [4, 8, 16]

    model = DistributedDataParallel(
        build_model(args).to(device),
        device_ids=[gpu],
        output_device=gpu
    )

    model.load_state_dict(
        torch.load(os.path.join(args.model_path, args.model_name + '.pth'))['model'], strict=False
    )

    backbone_params = dict()
    non_backbone_params = dict()
    fcos_params = dict()
    feat_comp = dict()
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if 'backbone' in n:
            backbone_params[n] = p
        elif 'box_predictor' in n:
            fcos_params[n] = p
        elif 'feat_comp' in n:
            feat_comp[n] = p
        else:
            non_backbone_params[n] = p

    optimizer = torch.optim.AdamW(
        [
            {'params': fcos_params.values(), 'lr': args.lr},
        ],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop, gamma=0.25)
    if args.resume_training:
        checkpoint = torch.load(os.path.join(args.model_path, f'{args.model_name}.pth'))
        model.load_state_dict(checkpoint['model'])
        start_epoch = checkpoint['epoch']
        best = checkpoint['best_val_ae']
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
    else:
        start_epoch = 0
        best = 10000000000000
        best_mAP = 0

    criterion = Criterion(args)
    aux_criterion = Criterion(args, aux=True)
    det_criterion = Detection_criterion(
        [[-1, args.fcos_pred_size], [64, 128], [128, 256], [256, 512], [512, 100000000]],  # config.sizes,
        'giou',  # config.iou_loss_type,
        True,  # config.center_sample,
        [1],  # config.fpn_strides,
        5,  # config.pos_radius,
    )

    train = DATASETS[args.dataset](
        args.data_path,
        args.image_size,
        split='train',
        num_objects=args.num_objects,
        tiling_p=args.tiling_p,
        zero_shot=args.zero_shot or args.orig_dmaps,
        skip_cars=args.skip_cars
    )
    val = DATASETS[args.dataset](
        args.data_path,
        args.image_size,
        split='val',
        num_objects=args.num_objects,
        tiling_p=args.tiling_p,
        zero_shot=args.zero_shot or args.orig_dmaps,
    )
    train_loader = DataLoader(
        train,
        sampler=DistributedSampler(train),
        batch_size=args.batch_size,
        drop_last=True,
        num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val,
        sampler=DistributedSampler(val),
        batch_size=args.batch_size,
        drop_last=False,
        num_workers=args.num_workers
    )
    print("NUM STEPS", len(train_loader) * args.epochs)
    print(rank, len(train_loader))
    for epoch in range(start_epoch + 1, args.epochs + 1):
        start = perf_counter()
        train_losses = {k: torch.tensor(0.0).to(device) for k in criterion.losses.keys()}
        val_losses = {k: torch.tensor(0.0).to(device) for k in criterion.losses.keys()}
        aux_train_losses = {k: torch.tensor(0.0).to(device) for k in aux_criterion.losses.keys()}
        aux_val_losses = {k: torch.tensor(0.0).to(device) for k in aux_criterion.losses.keys()}
        train_ae = torch.tensor(0.0).to(device)
        val_ae = torch.tensor(0.0).to(device)
        mAP = torch.tensor(0.0).to(device)

        train_loader.sampler.set_epoch(epoch)
        model.train()

        for img, bboxes, density_map, ids, scale_x, scale_y in train_loader:
            img = img.to(device)
            bboxes = bboxes.to(device)
            density_map = density_map.to(device)
            targets = BoxList(bboxes, (args.image_size, args.image_size), mode='xyxy').to(device).resize(
                (args.fcos_pred_size, args.fcos_pred_size))
            targets.fields['labels'] = [1 for __ in range(args.batch_size * 2)]
            optimizer.zero_grad()
            outR, aux_R, tblr, location = model(img, bboxes)

            if args.normalized_l2:
                with torch.no_grad():
                    num_objects = density_map.sum()
                    dist.all_reduce_multigpu([num_objects])
            else:
                num_objects = None

            main_losses = criterion(outR, density_map, bboxes, num_objects)
            aux_losses = [
                aux_criterion(aux, density_map, bboxes, num_objects) for aux in aux_R
            ]
            det_loss = det_criterion(location, tblr, targets)
            del targets
            loss = (
                    sum([ml for ml in main_losses.values()]) * 0 +
                    sum([al for alls in aux_losses for al in alls.values()]) * 0 +
                    det_loss  # + l
            )
            loss.backward()
            if args.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            train_losses = {
                k: train_losses[k] + main_losses[k] * img.size(0) for k in train_losses.keys()
            }
            aux_train_losses = {
                k: aux_train_losses[k] + sum([a[k] for a in aux_losses]) * img.size(0)
                for k in aux_train_losses.keys()
            }
            train_ae += torch.abs(
                density_map.flatten(1).sum(dim=1) - outR.flatten(1).sum(dim=1)
            ).sum()

        model.eval()
        with torch.no_grad():
            for img, bboxes, density_map, ids, scale_x, scale_y in val_loader:
                gt_bboxes, _ = val.get_gt_bboxes(ids)
                img = img.to(device)
                bboxes = bboxes.to(device)
                density_map = density_map.to(device)

                optimizer.zero_grad()

                outR, aux_R, tblr, location = model(img, bboxes)

                boxes_pred = generate_bbox(outR, tblr)

                for iii in range(len(gt_bboxes)):
                    boxes_pred[iii].box = boxes_pred[iii].box * 1 / torch.tensor(
                        [scale_y[iii], scale_x[iii], scale_y[iii], scale_x[iii]])
                    mAP += box_iou(gt_bboxes[iii], boxes_pred[iii].box).max(dim=1)[0].sum() / gt_bboxes[iii].shape[
                        1]

                if args.normalized_l2:
                    with torch.no_grad():
                        num_objects = density_map.sum()
                else:
                    num_objects = None
                main_losses = criterion(outR, density_map, bboxes, num_objects)
                aux_losses = [
                    aux_criterion(aux, density_map, bboxes, num_objects) for aux in aux_R
                ]
                val_losses = {
                    k: val_losses[k] + main_losses[k] * img.size(0) for k in val_losses.keys()
                }
                aux_val_losses = {
                    k: aux_val_losses[k] + sum([a[k] for a in aux_losses]) * img.size(0)
                    for k in aux_val_losses.keys()
                }
                val_ae += torch.abs(
                    density_map.flatten(1).sum(dim=1) - outR.flatten(1).sum(dim=1)
                ).sum()

        train_losses = reduce_dict(train_losses)
        val_losses = reduce_dict(val_losses)
        aux_train_losses = reduce_dict(aux_train_losses)
        aux_val_losses = reduce_dict(aux_val_losses)
        dist.all_reduce_multigpu([train_ae])
        dist.all_reduce_multigpu([val_ae])
        dist.all_reduce_multigpu([mAP])

        scheduler.step()

        if rank == 0:
            end = perf_counter()
            best_epoch = False

            if mAP > best_mAP:
                best_mAP = mAP
                checkpoint = {
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'best_val_ae': val_ae.item() / len(val)
                }
                torch.save(
                    checkpoint,
                    os.path.join(args.model_path, f'{args.det_model_name}.pth')
                )
                best_epoch = True

            print("Epoch", epoch)
            print({k: v.item() / len(train) for k, v in train_losses.items()})
            print({k: v.item() / len(val) for k, v in val_losses.items()})
            print({k: v.item() / len(train) for k, v in aux_train_losses.items()})
            print({k: v.item() / len(val) for k, v in aux_val_losses.items()})
            print(
                train_ae.item() / len(train),
                val_ae.item() / len(val),
                end - start,
                'best' if best_epoch else '',
            )
            print("det_sc:", mAP / len(val))
            print("********")

    if args.skip_test:
        dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DAVE', parents=[get_argparser()])
    args = parser.parse_args()
    print(args)
    train(args)