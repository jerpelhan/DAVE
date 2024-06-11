import argparse
import os
from time import perf_counter

import torch
from torch import distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from models.dave_tr import build_model
from utils.arg_parser import get_argparser
from utils.data import FSC147WithDensityMapSimilarityStitched
from utils.losses import Criterion

DATASETS = {
    'fsc147': FSC147WithDensityMapSimilarityStitched,
}

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

    pretrained_dict_feat = {k.split("backbone.backbone.")[1]: v for k, v in
                            torch.load(os.path.join(args.model_path, args.model_name+'.pth'))[
                                'model'].items() if 'backbone' in k}
    model.module.backbone.backbone.load_state_dict(pretrained_dict_feat)    

    optimizer = torch.optim.AdamW(
        [
            {'params': feat_comp.values()}
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
        best = 10000000

    criterion = Criterion(args)
    aux_criterion = Criterion(args, aux=True)

    train = DATASETS[args.dataset](
        args.data_path,
        args.image_size,
        split='train',
        num_objects=args.num_objects,
        tiling_p=args.tiling_p,
        zero_shot=args.zero_shot or args.orig_dmaps,

    )

    train_loader = DataLoader(
        train,
        sampler=DistributedSampler(train),
        batch_size=args.batch_size,
        drop_last=False,
        num_workers=args.num_workers,
    )
    val = DATASETS[args.dataset](
        args.data_path,
        args.image_size,
        split='val',
        num_objects=args.num_objects,
        tiling_p=args.tiling_p,
        zero_shot=args.zero_shot or args.orig_dmaps,
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
        train_loss = torch.tensor(0.0).to(device)
        val_loss = torch.tensor(0.0).to(device)

        train_loader.sampler.set_epoch(epoch)
        model.train()

        for img, bboxes, indices, density_map,  img_ids in train_loader:
            img = img.to(device)
            bboxes = bboxes.to(device)
            optimizer.zero_grad()
            loss, _, _ = model(img, bboxes)

            train_loss += loss
            loss.backward()
            if args.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()


        print("VALIDATION")
        model.eval()
        with torch.no_grad():
            for img, bboxes,indices, density_map, img_ids in val_loader:
                img = img.to(device)
                bboxes = bboxes.to(device)

                optimizer.zero_grad()
                loss, _, _ = model(img, bboxes)
                val_loss += loss
            
        dist.all_reduce_multigpu([val_loss])
        if rank == 0:
            print('val_loss',val_loss)

        scheduler.step()

        if rank == 0:
            end = perf_counter()
            best_epoch = False
            if val_loss.item()  < best:
                best = val_loss
        
                checkpoint = {
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'best_val_ae': val_loss.item() / len(val)
                }
                torch.save(
                    checkpoint,
                    os.path.join(args.model_path, f'{args.det_model_name}.pth')
                )
                best_epoch = True

            if rank == 0:
                print("Epoch", epoch)
                print(
                    train_loss.item() / len(train),
                    val_loss.item(),
                    end - start,
                    'best' if best_epoch else '',
                )
                print("********")

    if args.skip_test:
        dist.destroy_process_group()



if __name__ == '__main__':
    parser = argparse.ArgumentParser('DAVE', parents=[get_argparser()])
    args = parser.parse_args()
    print(args)
    train(args)

