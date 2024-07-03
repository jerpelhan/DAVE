import argparse
from pathlib import Path
import os
import torch
from torch.nn import DataParallel
from torchvision import transforms as T
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils.arg_parser import get_argparser
from models.dave import build_model
from utils.data import pad_image


def resize(img, img_size):
    resize_img = T.Resize((img_size, img_size), antialias=True)
    w, h = img.size
    img = T.Compose([
        T.ToTensor(),
        resize_img,
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])(img)
    scale = torch.tensor([1.0, 1.0]) / torch.tensor([w, h]) * img_size
    return img, scale


@torch.no_grad()
def demo(args):
    global fig, ax

    dpi = plt.rcParams['figure.dpi']
    plt.figure()
    gpu = 0
    torch.cuda.set_device(gpu)
    device = torch.device(gpu)

    model = DataParallel(
        build_model(args, is_zero_demo=True).to(device),
        device_ids=[gpu],
        output_device=gpu
    )
    model.load_state_dict(
        torch.load(os.path.join(args.model_path, 'DAVE_0_shot.pth'))['model'], strict=False
    )
    pretrained_dict_feat = {k.split("feat_comp.")[1]: v for k, v in
                            torch.load(os.path.join(args.model_path, 'verification.pth'))[
                                'model'].items() if 'feat_comp' in k}
    model.module.feat_comp.load_state_dict(pretrained_dict_feat)
    model.eval()

    def one_img(img_path):
        scale_x = scale_y = 1
        image = Image.open(img_path).convert("RGB")
        img, scale = resize(image, args.image_size)
        img = img.unsqueeze(0).to(device)
        bboxes = torch.tensor([]).unsqueeze(0).to(device)

        denisty_map, _, _, predicted_bboxes = model(img, bboxes=bboxes)

        if args.two_passes:
            boxes_predicted = predicted_bboxes.box
            scale_y = min(1, 50 / (boxes_predicted[:, 2] - boxes_predicted[:, 0]).mean())
            scale_x = min(1, 50 / (boxes_predicted[:, 3] - boxes_predicted[:, 1]).mean())

            if scale_x < 1 or scale_y < 1:
                scale_x = (int(args.image_size * scale_x) // 8 * 8) / args.image_size
                scale_y = (int(args.image_size * scale_y) // 8 * 8) / args.image_size
                resize_ = T.Resize((int(args.image_size * scale_x), int(args.image_size * scale_y)), antialias=True)
                img_resized = resize_(img)

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

            if scale_x != 1.0 or scale_y != 1.0:
                denisty_map, _, _, predicted_bboxes = model(img_resized, bboxes)

        pred_boxes = predicted_bboxes.box.cpu() / torch.tensor([scale_y*scale[0], scale_x*scale[1], scale_y*scale[0], scale_x*scale[1]])

        plt.clf()
        w, h = image.size
        figsize = (w + 100) / float(dpi), (h + 100) / float(dpi)
        plt.rcParams["figure.figsize"] = figsize
        plt.imshow(image)
        for i in range(len(pred_boxes)):
            box = pred_boxes[i]
            plt.plot([box[0], box[0], box[2], box[2], box[0]], [box[1], box[3], box[3], box[1], box[1]], linewidth=2,
                    color='red')
        plt.title("Dmap count:" + str(round(denisty_map.sum().item(), 1)) + " Box count:" + str(len(pred_boxes)))
        if args.show:
            plt.show()
        else:
            out_file = args.out_path / f"{img_path.stem}__out.png"
            plt.savefig(out_file, bbox_inches='tight')

    if args.img_path.is_file():
        one_img(args.img_path)
    elif args.img_path.is_dir():
        for img_p in (imgs := tqdm(list(args.img_path.iterdir()))):
            imgs.set_description(f"{img_p.name:>30}")
            one_img(img_p)
    else:
        raise(ValueError, f"{args.img_path} is neither a dir nor a file :(")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DAVE', parents=[get_argparser()])
    parser.add_argument('--img_path', type=Path)
    parser.add_argument('--out_path', type=Path)
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--two_passes', action='store_true')
    args = parser.parse_args()
    demo(args)
