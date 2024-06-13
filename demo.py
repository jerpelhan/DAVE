from torch.nn import DataParallel
from models.dave import build_model
from utils.arg_parser import get_argparser
import argparse
import torch
import os
import matplotlib.patches as patches
from PIL import Image

from utils.data import resize
import matplotlib.pyplot as plt

bounding_boxes = []


def on_click(event):
    # Record the starting point of the bounding box
    global ix, iy
    ix, iy = event.xdata, event.ydata
    # Connect the release event
    fig.canvas.mpl_connect('button_release_event', on_release)


def on_release(event):
    # Record the ending point of the bounding box
    global ix, iy
    x, y = event.xdata, event.ydata
    # Calculate the width and height of the bounding box
    width = x - ix
    height = y - iy
    # Add a rectangle patch to the axes
    rect = patches.Rectangle((ix, iy), width, height, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    # Store the bounding box coordinates
    bounding_boxes.append((ix, iy, ix + width, iy + height))
    plt.draw()


@torch.no_grad()
def demo(args):
    img_path = "material//7.jpg"
    global fig, ax

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
    model.eval()

    image = Image.open(img_path).convert("RGB")
    # Create a figure and axis
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    # Connect the click event
    cid = fig.canvas.mpl_connect('button_press_event', on_click)
    plt.title("Click and drag to draw bboxes, then close window")
    # Show the image
    plt.show()

    bboxes = torch.tensor(bounding_boxes)

    img, bboxes, scale = resize(image, bboxes)
    img = img.unsqueeze(0).to(device)
    bboxes = bboxes.unsqueeze(0).to(device)

    denisty_map, _, tblr, predicted_bboxes = model(img, bboxes=bboxes)

    plt.clf()
    plt.imshow(image)
    pred_boxes = predicted_bboxes.box.cpu() / torch.tensor([scale[0], scale[1], scale[0], scale[1]])
    for i in range(len(pred_boxes)):
        box = pred_boxes[i]

        plt.plot([box[0], box[0], box[2], box[2], box[0]], [box[1], box[3], box[3], box[1], box[1]], linewidth=2,
                 color='red')
    plt.title("Dmap count:" + str(round(denisty_map.sum().item(), 1)) + " Box count:" + str(len(pred_boxes)))
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DAVE', parents=[get_argparser()])
    args = parser.parse_args()
    demo(args)
