def extend_bboxes(bboxes, extension_factor=1.1):
    """
    Extend bounding boxes by a given factor.

    Args:
        bboxes (torch.Tensor): A tensor of shape (N, 4) representing N bounding boxes in (x1, y1, x2, y2) format.
        extension_factor (float): The factor by which to extend the bounding boxes. Default is 1.1.

    Returns:
        torch.Tensor: A tensor of shape (N, 4) representing the extended bounding boxes in (x1, y1, x2, y2) format.
    """
    centers = (bboxes[:, :2] + bboxes[:, 2:]) / 2
    width = bboxes[:, 2] - bboxes[:, 0]
    height = bboxes[:, 3] - bboxes[:, 1]

    new_width = width * extension_factor
    new_height = height * extension_factor

    extended_bboxes = torch.cat([
        (centers[:, 0] - new_width / 2).unsqueeze(1),  # x1
        (centers[:, 1] - new_height / 2).unsqueeze(1),  # y1
        (centers[:, 0] + new_width / 2).unsqueeze(1),  # x2
        (centers[:, 1] + new_height / 2).unsqueeze(1)  # y2
    ], dim=1)

    return extended_bboxes


import torch
import math

def mask_density(image_batch, boxes_batch, img=None):
    """
    Masks the density of an image batch based on the bounding boxes provided.

    Args:
        image_batch (torch.Tensor): A tensor of shape (C, H, W) representing the image batch.
        boxes_batch (torch.Tensor): A tensor of shape (N, 4) representing the bounding boxes for each image in the batch.
        img (optional): Not used.

    Returns:
        torch.Tensor: A tensor of shape (C, H, W) representing the masked image batch.
    """
    if len(boxes_batch) < 1:
        return image_batch
    C, H, W = image_batch.shape

    mask_tensor = torch.zeros((C, H, W))
    for box in boxes_batch.box.long():
        x1, y1, x2, y2 = box
        mask_tensor[:, y1:y2, x1:x2] = 1
    max_bbx = boxes_batch[torch.where(torch.median(boxes_batch.area())==boxes_batch.area())[0].item()].box
    max_wh_half = (2 * math.floor((max_bbx[3] - max_bbx[1]) / 2 / 2) + 1,
                   2 * math.floor((max_bbx[2] - max_bbx[0]) / 2 / 2) + 1)
    mask_tensor =mask_tensor.unsqueeze(0)
    masked_image_batch = image_batch * mask_tensor.to(image_batch.device)
    return masked_image_batch
