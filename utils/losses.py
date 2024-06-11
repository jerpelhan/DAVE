import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from time import perf_counter


class ObjectNormalizedL2Loss(nn.Module):

    def __init__(self):
        super(ObjectNormalizedL2Loss, self).__init__()

    def forward(self, output, dmap, num_objects):
        return ((output - dmap) ** 2).sum() / num_objects


class MinCountLoss(nn.Module):

    def __init__(self):
        super(MinCountLoss, self).__init__()

    def forward(self, output, bboxes):
        bboxes = bboxes.long().clip(0, output.size(-1))
        r = 0
        for i in range(bboxes.size(0)):
            for j in range(bboxes.size(1)):
                x1, y1, x2, y2 = bboxes[i, j, :]
                r += F.relu(1 - output[i, 0, y1:y2, x1:x2].sum())
        return r


class CountLoss(nn.Module):

    def __init__(self):
        super(CountLoss, self).__init__()

        self.count_loss = nn.MSELoss()

    def forward(self, output, dmap):
        return self.count_loss((output).flatten(1).sum(1), dmap.flatten(1).sum(1))


class BackgroundL2Loss(nn.Module):

    def __init__(self):
        super(BackgroundL2Loss, self).__init__()

    def forward(self, output, mask):
        return ((output[mask == 0]) ** 2).sum()


INF = 100000000


class IOULoss(nn.Module):
    def __init__(self, loc_loss_type):
        super().__init__()

        self.loc_loss_type = loc_loss_type

    def forward(self, out, target, weight=None):

        pred_left, pred_top, pred_right, pred_bottom = out.unbind(1)

        target_left, target_top, target_right, target_bottom = target.unbind(1)

        target_area = (target_left + target_right) * (target_top + target_bottom)
        pred_area = (pred_left + pred_right) * (pred_top + pred_bottom)

        w_intersect = torch.min(pred_left, target_left) + torch.min(
            pred_right, target_right
        )
        h_intersect = torch.min(pred_bottom, target_bottom) + torch.min(
            pred_top, target_top
        )

        area_intersect = w_intersect * h_intersect
        area_union = target_area + pred_area - area_intersect

        ious = (area_intersect + 1) / (area_union + 1)

        if self.loc_loss_type == 'iou':
            loss = -torch.log(ious)

        elif self.loc_loss_type == 'giou':
            g_w_intersect = torch.max(pred_left, target_left) + torch.max(
                pred_right, target_right
            )
            g_h_intersect = torch.max(pred_bottom, target_bottom) + torch.max(
                pred_top, target_top
            )
            g_intersect = g_w_intersect * g_h_intersect + 1e-7
            gious = ious - (g_intersect - area_union) / g_intersect

            loss = 1 - gious

        if weight is not None and weight.sum() > 0:
            return (loss * weight).sum() / weight.sum()

        else:
            return loss.mean()


class Criterion(nn.Module):

    def __init__(self, args, aux=False):

        super(Criterion, self).__init__()

        losses = dict()
        losses['dmap'] = ObjectNormalizedL2Loss() if args.normalized_l2 else nn.MSELoss(reduction='sum')
        if args.count_loss_weight > 0:
            losses['count'] = CountLoss()
        if args.min_count_loss_weight > 0 and not args.zero_shot:
            losses['min_count'] = MinCountLoss()

        self.aux = aux
        self.aux_weight = args.aux_weight
        self.losses = losses
        self.reduction = args.reduction
        self.weights = {
            'dmap': 1,
            'count': args.count_loss_weight,
            'min_count': args.min_count_loss_weight
        }

    def forward(self, output, density_map, bboxes, num_objects=None):
        losses = dict()
        if 'dmap' in self.losses:
            if num_objects is not None:
                losses['dmap'] = self.losses['dmap'](output, density_map, num_objects)
            else:
                losses['dmap'] = self.losses['dmap'](output, density_map)
        if 'count' in self.losses:
            losses['count'] = self.losses['count'](output, density_map)
        if 'min_count' in self.losses:
            losses['min_count'] = self.losses['min_count'](output, bboxes)

        if not self.aux:
            losses = {k: v * self.weights[k] for k, v in losses.items()}
        else:
            losses = {k: self.aux_weight * v * self.weights[k] for k, v in losses.items()}

        return losses


class Detection_criterion(nn.Module):

    def __init__(
            self, sizes, iou_loss_type, center_sample, fpn_strides, pos_radius, aux=False
    ):
        super().__init__()

        self.sizes = sizes
        self.box_loss = IOULoss(iou_loss_type)
        self.aux = aux
        self.center_sample = center_sample
        self.strides = fpn_strides
        self.radius = pos_radius

    def prepare_target(self, points, targets):
        ex_size_of_interest = []

        for i, point_per_level in enumerate(points):
            size_of_interest_per_level = point_per_level.new_tensor(self.sizes[i])
            ex_size_of_interest.append(
                size_of_interest_per_level[None].expand(len(point_per_level), -1)
            )

        ex_size_of_interest = torch.cat(ex_size_of_interest, 0)
        n_point_per_level = [len(point_per_level) for point_per_level in points]
        point_all = torch.cat(points, dim=0)
        label, box_target = self.compute_target_for_location(
            point_all, targets, ex_size_of_interest, n_point_per_level
        )

        for i in range(len(label)):
            label[i] = torch.split(label[i], n_point_per_level, 0)
            box_target[i] = torch.split(box_target[i], n_point_per_level, 0)

        label_level_first = []
        box_target_level_first = []

        for level in range(len(points)):
            label_level_first.append(
                torch.cat([label_per_img[level] for label_per_img in label], 0).to(points[0].device)
            )
            box_target_level_first.append(
                torch.cat(
                    [box_target_per_img[level] for box_target_per_img in box_target], 0
                )
            )

        return label_level_first, box_target_level_first

    def get_sample_region(self, gt, strides, n_point_per_level, xs, ys, radius=1):
        n_gt = gt.shape[0]
        n_loc = len(xs)
        gt = gt[None].expand(n_loc, n_gt, 4)
        center_x = (gt[..., 0] + gt[..., 2]) / 2
        center_y = (gt[..., 1] + gt[..., 3]) / 2
        # y_stride = torch.min((gt[..., 3] - gt[..., 1]) / 2)/2
        # x_stride = torch.min((gt[..., 2] - gt[..., 0]) / 2)/2

        if center_x[..., 0].sum() == 0:
            return xs.new_zeros(xs.shape, dtype=torch.uint8)

        begin = 0

        center_gt = gt.new_zeros(gt.shape)

        for level, n_p in enumerate(n_point_per_level):
            end = begin + n_p
            stride = strides[level] * radius

            x_min = center_x[begin:end] - stride
            y_min = center_y[begin:end] - stride
            x_max = center_x[begin:end] + stride
            y_max = center_y[begin:end] + stride

            center_gt[begin:end, :, 0] = torch.where(
                x_min > gt[begin:end, :, 0], x_min, gt[begin:end, :, 0]
            )
            center_gt[begin:end, :, 1] = torch.where(
                y_min > gt[begin:end, :, 1], y_min, gt[begin:end, :, 1]
            )
            center_gt[begin:end, :, 2] = torch.where(
                x_max > gt[begin:end, :, 2], gt[begin:end, :, 2], x_max
            )
            center_gt[begin:end, :, 3] = torch.where(
                y_max > gt[begin:end, :, 3], gt[begin:end, :, 3], y_max
            )

            begin = end

        left = xs[:, None] - center_gt[..., 0]
        right = center_gt[..., 2] - xs[:, None]
        top = ys[:, None] - center_gt[..., 1]
        bottom = center_gt[..., 3] - ys[:, None]

        center_bbox = torch.stack((left, top, right, bottom), -1)
        is_in_boxes = center_bbox.min(-1)[0] > 0

        return is_in_boxes

    def compute_target_for_location(
            self, locations, targets, sizes_of_interest, n_point_per_level
    ):
        labels = []
        box_targets = []
        xs, ys = locations[:, 0], locations[:, 1]
        for i in range(len(targets)):
            targets_per_img = targets[i]
            assert targets_per_img.mode == 'xyxy'
            bboxes = targets_per_img.box

            labels_per_img = torch.tensor([1, 1, 1, 1, 1, 1]).to(locations.device)
            area = targets_per_img.area()

            l = xs[:, None] - bboxes[:, 0][None]
            t = ys[:, None] - bboxes[:, 1][None]
            r = bboxes[:, 2][None] - xs[:, None]
            b = bboxes[:, 3][None] - ys[:, None]

            box_targets_per_img = torch.stack([l, t, r, b], 2)

            if self.center_sample:
                is_in_boxes = self.get_sample_region(
                    bboxes, self.strides, n_point_per_level, xs, ys, radius=self.radius
                )

            else:
                is_in_boxes = box_targets_per_img.min(2)[0] > 0

            max_box_targets_per_img = box_targets_per_img.max(2)[0]

            is_cared_in_level = (
                                        max_box_targets_per_img >= sizes_of_interest[:, [0]]
                                ) & (max_box_targets_per_img <= sizes_of_interest[:, [1]])

            locations_to_gt_area = area[None].repeat(len(locations), 1)
            locations_to_gt_area[is_in_boxes == 0] = INF
            locations_to_gt_area[is_cared_in_level == 0] = INF

            locations_to_min_area, locations_to_gt_id = locations_to_gt_area.min(1)

            box_targets_per_img = box_targets_per_img[
                range(len(locations)), locations_to_gt_id
            ]

            labels_per_img = labels_per_img.to(locations_to_gt_id.device)[locations_to_gt_id]
            labels_per_img[locations_to_min_area == INF] = 0

            labels.append(labels_per_img)
            box_targets.append(box_targets_per_img)

        return labels, box_targets

    def compute_centerness_targets(self, box_targets):
        left_right = box_targets[:, [0, 2]]
        top_bottom = box_targets[:, [1, 3]]
        centerness = (left_right.min(-1)[0] / left_right.max(-1)[0]) * (
                top_bottom.min(-1)[0] / top_bottom.max(-1)[0]
        )

        return torch.sqrt(centerness)

    def forward(self, locations, box_pred, targets):
        batch = box_pred[0].shape[0]
        labels, box_targets = self.prepare_target(locations, targets)
        box_flat = []

        labels_flat = []
        box_targets_flat = []

        for i in range(len(labels)):
            box_flat.append(box_pred.permute(0, 2, 3, 1).reshape(-1, 4))

            labels_flat.append(labels[i].reshape(-1))
            box_targets_flat.append(box_targets[i].reshape(-1, 4))
        box_flat = torch.cat(box_flat, 0)
        labels_flat = torch.cat(labels_flat, 0)
        box_targets_flat = torch.cat(box_targets_flat, 0)
        pos_id = torch.nonzero(labels_flat > 0).squeeze(1)
        box_flat = box_flat[pos_id]
        box_targets_flat = box_targets_flat[pos_id]

        if pos_id.numel() > 0:
            center_targets = self.compute_centerness_targets(box_targets_flat)
            box_loss = self.box_loss(box_flat, box_targets_flat, center_targets)
        else:
            box_loss = box_flat.sum()

        return box_loss


def calc_mAP(bboxes, gt_bboxes):
    mAP = 0
    for i in range(len(bboxes)):
        mAP += single_image_mAP(bboxes[i].box.cpu(), gt_bboxes[i])
    return torch.tensor(mAP)


def single_image_mAP(preds, gts, iou_threshold=0.5):
    """
    Calculates mean average precision
    Parameters:
        pred_boxes (list): list of lists containing all bboxes with each bboxes
        specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
        true_boxes (list): Similar as pred_boxes except all the correct ones
        iou_threshold (float): threshold where predicted bboxes is correct
        box_format (str): "midpoint" or "corners" used to specify bboxes
        num_classes (int): number of classes
    Returns:
        float: mAP value across all classes given a specific IoU threshold
    """
    # used for numerical stability later on
    if len(preds) == 0:
        return 0.0
    epsilon = 1e-6

    detections = preds
    ground_truths = gts
    TP = np.zeros(len(detections))
    FP = np.zeros(len(detections))
    total_true_bboxes = len(ground_truths)

    gt_boxes = np.array(ground_truths)
    pred_boxes = np.array(detections)
    gt_index = np.zeros(gt_boxes.shape[0], dtype=np.int16)
    for detection_idx, (detection, pred_box) in enumerate(zip(detections, pred_boxes)):
        # Only take out the ground_truths that have the same
        # training idx as detection
        best_iou = 0
        for idx, gt_box in enumerate(gt_boxes):
            iou = intersection_over_union(np.array(pred_box), np.array(gt_box), )
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = idx
        if best_iou > iou_threshold:
            # only detect ground truth detection once
            if gt_index[best_gt_idx] == 0:
                # true positive and add this bounding box to seen
                TP[detection_idx] = 1
                gt_index[best_gt_idx] = 1
            else:
                FP[detection_idx] = 1
        # if IOU is lower then the detection is a false positive
        else:
            FP[detection_idx] = 1

    TP_cumsum = np.cumsum(TP)
    FP_cumsum = np.cumsum(FP)
    recalls = TP_cumsum / (total_true_bboxes + epsilon)
    precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
    ap = cal_ap(recalls, precisions)
    return ap


def intersection_over_union(box1, box2):
    bb1 = {'x1': box1[0],
           'x2': box1[2],
           'y1': box1[1],
           'y2': box1[3]
           }
    bb2 = {'x1': box2[0],
           'x2': box2[2],
           'y1': box2[1],
           'y2': box2[3]
           }

    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def cal_ap(rec, prec):
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap
