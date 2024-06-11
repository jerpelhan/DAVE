from torch.nn import DataParallel
from models.dave import build_model
from utils.arg_parser import get_argparser
import os
import argparse
import torch
from torchvision import transforms as T

import matplotlib.patches as patches
from PIL import Image

from utils.data import resize
import matplotlib.pyplot as plt
import matplotlib
print(matplotlib.rcParams['backend'])

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
    bounding_boxes.append((ix, iy, width, height))
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
    state_dict = torch.load(os.path.join(args.model_path, f'{args.model_name}.pth'))['model']
    state_dict = {k if 'module.' in k else 'module.' + k: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.eval()



    image = image = Image.open(img_path).convert("RGB")
    # Create a figure and axis
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    # Connect the click event
    cid = fig.canvas.mpl_connect('button_press_event', on_click)
    # Show the image
    plt.show()

    bboxes = torch.tensor(bounding_boxes)

    # resize and pad
    img = T.ToTensor()(image)
    bboxes = bboxes.to(device).unsqueeze(0)
    r_image, r_bboxes, _,padwh = resize(img, bboxes)

    image = r_image.to(device).unsqueeze(0)

    image = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)

    denisty_map, _, tblr, predicted_bboxes = model(image, bboxes=bboxes)






if __name__ == '__main__':
    parser = argparse.ArgumentParser('LOCA', parents=[get_argparser()])
    args = parser.parse_args()
    demo(args)
