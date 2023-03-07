import os
import re
import json
import shutil
from distutils.dir_util import copy_tree

from src.utils import *

import torch
import pandas as pd
import numpy as np

from ultralytics import YOLO
from PIL import Image


def box_iou(box1, box2, eps=1e-7):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)


def box_iop(box1, box2, eps=1e-7):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / ((b2 - b1).prod(2) + eps)


def box_iol(box1, box2, eps=1e-7):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / ((a2 - a1).prod(2) + eps)


def _process_batch(detections, labels, enable_iop=False):
    """
    Return correct prediction matrix
    Arguments:
        detections (array[N, 6]), x1, y1, x2, y2, conf, class
        labels (array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (array[N, 10]), for 10 IoU levels
    """
    iouv = torch.linspace(0.5, 0.95, 10)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    iop = box_iop(labels[:, 1:], detections[:, :4])
    iol = box_iol(labels[:, 1:], detections[:, :4])
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    correct_class = labels[:, 0:1] == detections[:, 5]

    all_matches = []
    all_indices = []
    for i in range(len(iouv)):
        if enable_iop:
            iou_idx = (iou >= iouv[i]) | ((iou >= (iouv[i]*0.5)) & (iop > 0.9)) | ((iou >= (iouv[i]*0.8)) & (iol > 0.9))
        else:
            iou_idx = iou >= iouv[i]
        x = torch.where(iou_idx & correct_class)  # IoU > threshold and classes match
        matches = np.zeros((x[0].shape[0], 3))
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]),
                                1).cpu().numpy()  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]] # sort matches by iou
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]] # drop duplicates
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
        all_indices.append(x)
        all_matches.append(matches)
    return torch.tensor(correct, dtype=torch.bool, device=detections.device), all_matches, all_indices


def xywh2xyxy(x):

    """
    Convert bounding box coordinates from (x, y, width, height) format to (x1, y1, x2, y2) format where (x1, y1) is the
    top-left corner and (x2, y2) is the bottom-right corner.

    Args:
        x (np.ndarray) or (torch.Tensor): The input bounding box coordinates in (x, y, width, height) format.
    Returns:
        y (np.ndarray) or (torch.Tensor): The bounding box coordinates in (x1, y1, x2, y2) format.
    """

    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y

def xyxy2xywh(x):
    """
    Convert bounding box coordinates from (x1, y1, x2, y2) format to (x, y, width, height) format.

    Args:
        x (np.ndarray) or (torch.Tensor): The input bounding box coordinates in (x1, y1, x2, y2) format.
    Returns:
       y (np.ndarray) or (torch.Tensor): The bounding box coordinates in (x, y, width, height) format.
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # x center
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # y center
    y[..., 2] = x[..., 2] - x[..., 0]  # width
    y[..., 3] = x[..., 3] - x[..., 1]  # height
    return y


def read_predictions(predictions_path):
    with open(predictions_path, "r") as f:
        preds_json = json.load(f)

    preds_dict = {}
    for p in preds_json:
        image_id = p["image_id"]
        box = [p["bbox"], p["score"]]
        preds_dict[image_id] = preds_dict.get(image_id, []) + [box]

    predictions = {}
    for image_id, boxes in preds_dict.items():
        preds_conf = np.array([entry[1] for entry in boxes])
        preds_boxes = np.array([entry[0] for entry in boxes])
        preds_boxes[:, :2] += preds_boxes[:, 2:]/2
        preds_boxes = xywh2xyxy(preds_boxes)
        detections = np.append(preds_boxes, preds_conf.reshape(-1, 1), axis=1)
        detections = torch.tensor(np.append(detections, np.zeros((detections.shape[0], 1)), axis=1))
        predictions[image_id] = detections

    return predictions


def preprocess_predictions(images, results):
    predictions = {}
    for img_fn, result in zip(images, results):
        fn_origin = re.sub(".jpg$", "", img_fn)
        boxes = result.boxes.xyxy
        confs = result.boxes.conf.unsqueeze(1)
        cls = torch.zeros(boxes.shape[0], 1)

        predictions[fn_origin] = torch.cat([boxes, confs, cls], axis=1)
    return predictions


# TODO: change
def update_labels(gt_path, predictions, pr_label_positive, pr_label_negative, pr_bg_positive, conf):

    updated_labels = []
    UNMATCHED_CONF = 1e-4
    EPSILON = 1e-4
    P_T_N = pr_label_negative[0, 0]

    for fn in os.listdir(gt_path):
        fn_origin = re.sub(".txt$", "", fn)

        detections = predictions.get(fn_origin, [])
        npr = detections.shape[0]

        if os.path.getsize(os.path.join(gt_path, fn)) > 0:
            gt_boxes = pd.read_csv(os.path.join(gt_path, fn), header=None, sep=" ")
            nl = gt_boxes.shape[0]
        else:
            nl = 0

        if nl == 0 and npr == 0:
            new_labels = torch.tensor([])

        elif nl > 0 and npr == 0:
            gt_boxes = xywh2xyxy(gt_boxes.iloc[:, 1:].values) * 320
            labels = torch.tensor(np.append(np.zeros((gt_boxes.shape[0], 1)), gt_boxes, axis=1))
            confs = torch.full((labels.shape[0], 1), UNMATCHED_CONF)
            precisions = torch.full((labels.shape[0], 1), P_T_N)
            new_labels = torch.cat([labels[:, 1:], confs, precisions], axis=1)

        elif nl == 0 and npr > 0:
            unmatched_detections_boxes = detections[:, :4]
            unmatched_detections_confs = detections[:, 4].unsqueeze(1)
            unmatched_detections_precisions = torch.tensor(
                np.interp(unmatched_detections_confs, conf, pr_bg_positive[0, :], left=EPSILON))
            new_labels = torch.cat([unmatched_detections_boxes, unmatched_detections_confs, unmatched_detections_precisions], axis=1)
            new_labels = new_labels[new_labels[:, -1] >= 0.3]
            # new_labels = torch.tensor([])
        else:
            gt_boxes = xywh2xyxy(gt_boxes.iloc[:, 1:].values) * 320
            labels = torch.tensor(np.append(np.zeros((gt_boxes.shape[0], 1)), gt_boxes, axis=1))

            _, matches_iouv, matches_indices = _process_batch(detections, torch.tensor(labels), enable_iop=True)
            matches = matches_iouv[0]
            indices = matches_indices[0]

            # TODO: adjust label location/size?
            matched_labels_indices = torch.full((labels.shape[0],), False)
            matched_labels_indices[matches[:, 0]] = True
            unmatched_labels_indices = torch.logical_not(matched_labels_indices)

            matched_detections_indices = torch.full((detections.shape[0],), False)
            matched_detections_indices[matches[:, 1]] = True
            unmatched_detections_indices = torch.full((detections.shape[0],), True)
            unmatched_detections_indices[indices[1]] = False

            matched_labels_boxes = labels[matched_labels_indices, 1:]
            matched_confs = detections[matched_detections_indices, 4].unsqueeze(1)
            matched_precisions = torch.tensor(np.interp(matched_confs, conf, pr_label_positive[0, :], left=EPSILON))
            matched_labels = torch.cat([matched_labels_boxes, matched_confs, matched_precisions], axis=1)

            unmatched_labels_boxes = labels[unmatched_labels_indices, 1:]
            # TODO: change with the precision at position = 0.5?
            unmatched_confs = torch.full((unmatched_labels_boxes.shape[0], 1), UNMATCHED_CONF)
            unmatched_precisions = torch.full((unmatched_labels_boxes.shape[0], 1), P_T_N)
            unmatched_labels = torch.cat([unmatched_labels_boxes, unmatched_confs, unmatched_precisions], axis=1)

            unmatched_detections_boxes = detections[unmatched_detections_indices, :4]
            unmatched_detections_confs = detections[unmatched_detections_indices, 4].unsqueeze(1)
            unmatched_detections_precisions = torch.tensor(np.interp(unmatched_detections_confs, conf, pr_bg_positive[0, :], left=EPSILON))
            unmatched_detections = torch.cat([unmatched_detections_boxes, unmatched_detections_confs, unmatched_detections_precisions], axis=1)
            unmatched_detections = unmatched_detections[unmatched_detections[:, -1] >= 0.3]

            new_labels = torch.cat([matched_labels, unmatched_labels, unmatched_detections], axis=0)

        # new_labels[:, 4] = torch.tensor(np.interp(new_labels[:, 4], conf, precision[0, :], left=EPSILON))
        updated_labels.append({"file": fn_origin, "labels": new_labels, "img_path": gt_path.replace("labels", "images")})
    return updated_labels


def calculate_match(gt_path, predictions):
    niou = 10
    stats = []
    total_gt_size = 0
    total_gt = 0
    total_img = 0
    # predictions = read_predictions(predictions_path)

    for fn in os.listdir(gt_path):
        fn_origin = re.sub(".txt$", "", fn)

        detections = predictions.get(fn_origin, [])
        npr = detections.shape[0]

        if os.path.getsize(os.path.join(gt_path, fn)) > 0:
            gt_boxes = pd.read_csv(os.path.join(gt_path, fn), header=None, sep=" ")
            nl = gt_boxes.shape[0]
        else:
            nl = 0
        cls = torch.zeros((nl, 1))

        correct_bboxes = torch.zeros(npr, niou, dtype=torch.bool)

        if npr == 0:
            if nl > 0:
                stats.append((correct_bboxes, *torch.zeros((2, 0)), cls.squeeze(-1)))
            continue

        if nl > 0:
            gt_boxes = gt_boxes.iloc[:, 1:].values
            total_gt += gt_boxes.shape[0]
            total_gt_size += np.prod(gt_boxes[:, 2:], axis=1).sum() * 320 * 320

            # gt_boxes[:, :2] -= gt_boxes[:, 2:]/2
            gt_boxes = xywh2xyxy(gt_boxes) * 320

            labels = torch.tensor(np.append(np.zeros((gt_boxes.shape[0], 1)), gt_boxes, axis=1))

            correct_bboxes, _, _ = _process_batch(detections, labels)

        total_img += 1
        print(correct_bboxes.shape, correct_bboxes.sum(), cls.squeeze(-1).shape)
        stats.append((correct_bboxes, detections[:, 4], detections[:, 5], cls.squeeze(-1)))

    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]
    return stats, total_gt_size / total_gt, total_img


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves
    # Arguments
        recall:    The recall curve (list)
        precision: The precision curve (list)
    # Returns
        Average precision, precision curve, recall curve
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x-axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec


def smooth(y, f=0.05):
    # Box filter of fraction f
    nf = round(len(y) * f * 2) // 2 + 1  # number of filter elements (must be odd)
    p = np.ones(nf // 2)  # ones padding
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)  # y padded
    return np.convolve(yp, np.ones(nf) / nf, mode='valid')  # y-smoothed


def conf_to_precision(tp, conf, pred_cls, target_cls, avg_gt_size, total_img, eps=1e-16, prefix=""):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:  True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls:  Predicted object classes (nparray).
        target_cls:  True object classes (nparray).
        plot:  Plot precision-recall curve at mAP@0.5
        save_dir:  Plot save directory
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    px, py = np.linspace(0, 1, 1000), []  # for plotting
    ap, p, r, fn = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    pr_label_positive, pr_label_negative, pr_bg_positive = np.zeros((nc, 1000)), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = nt[ci]  # number of labels
        # TODO: multiple by batch size
        n_b = int(320 * 320 / avg_gt_size) * total_img - n_l
        n_p = i.sum()  # number of predictions
        if n_p == 0 or n_l == 0:
            continue

        # Accumulate FPs and TPs
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)

        # Recall
        recall = tpc / (n_l + eps)  # recall curve
        # r[ci] = np.interp(-px, -conf[i], recall[:, 0])  # negative x, xp because xp decreases

        # Precision
        precision = tpc / (tpc + fpc)  # precision curve
        # p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

        p_true = n_l / (n_l + n_b)
        p_label_true = 0.8
        p_label_false = 0.1
        p_positive_true = recall
        p_positive_false = fpc / (n_b + eps)

        p_true_label_positive = p_label_true * p_positive_true * p_true
        p_false_label_positive = p_label_false * p_positive_false * (1 - p_true)
        p_true_label_positive /= p_true_label_positive + p_false_label_positive
        pr_label_positive[ci] = np.interp(-px, -conf[i], p_true_label_positive[:, 0])

        p_true_label_negative = p_label_true * (1 - p_positive_true) * p_true
        p_false_label_negative = p_label_false * (1 - p_positive_false) * (1 - p_true)
        p_true_label_negative /= p_true_label_negative + p_false_label_negative
        pr_label_negative[ci] = np.interp(-px, -conf[i], p_true_label_negative[:, 0])

        p_true_bg_positive = (1 - p_label_true) * p_positive_true * p_true
        p_false_bg_positive = (1 - p_label_false) * p_positive_false * (1 - p_true)
        p_true_bg_positive /= p_true_bg_positive + p_false_bg_positive
        pr_bg_positive[ci] = np.interp(-px, -conf[i], p_true_bg_positive[:, 0])
    return pr_label_positive, pr_label_negative, pr_bg_positive, px


def write_labels(file_path, labels_array):
    if labels_array is not None:
        labels_lst = labels_array.tolist()
        with open(file_path, "w") as f:
            for l in labels_lst:
                f.write(f"{int(l[0])} {l[1]} {l[2]} {l[3]} {l[4]}\n")
    else:
        with open(file_path, "w") as _:
            pass


def write_weights(file_path, weights_array):
    if weights_array is not None:
        weights_lst = weights_array.tolist()
        with open(file_path, "w") as f:
            for l in weights_lst:
                f.write(f"{l[0]}\n")
    else:
        with open(file_path, "w") as _:
            pass


def confident_learning(root_path, dataset_path):
    run_idx = 1
    all_updated_labels = []

    for path in os.listdir(dataset_path):
        if not (os.path.isdir(os.path.join(dataset_path, path)) and re.search("-[0-4]$", path)):
            continue
        run_idx_str = "" if run_idx == 1 else str(run_idx)
        yaml_path = os.path.join(dataset_path, path, "data.yaml")
        val_img_path = os.path.join(dataset_path, path, "valid", "images")
        gt_path = os.path.join(dataset_path, path, "valid", "labels")
        predictions_path = os.path.join(root_path, "runs", "detect", f"val{run_idx_str}", "predictions.json")
        test_img_path = os.path.join(dataset_path, path, "test", "images")
        test_lbl_path = os.path.join(dataset_path, path, "test", "labels")

        model = YOLO("yolov8s.pt")
        model.train(data=yaml_path, epochs=200, imgsz=320, lr0=1e-3)
        model.val(data=yaml_path, save_json=True)

        # # test
        # val_images = []
        # val_image_fns = []
        # for img_fn in os.listdir(val_img_path):
        #     img_path = os.path.join(val_img_path, img_fn)
        #     if os.path.isfile(img_path):
        #         val_images.append(Image.open(img_path))
        #         val_image_fns.append(img_fn)

        # val_predictions = model.predict(source=val_images, conf=0.001)

        # calculate bboxes

        # stats = calculate_match(gt_path, preprocess_predictions(val_image_fns, val_predictions))
        stats, avg_gt_size, total_img = calculate_match(gt_path, read_predictions(predictions_path))

        # calculate confidence to precision map.

        pr_label_positive, pr_label_negative, pr_bg_positive, conf = conf_to_precision(*stats, avg_gt_size, total_img)
        run_idx += 1

        # test
        test_images = []
        test_image_fns = []
        for img_fn in os.listdir(test_img_path):
            img_path = os.path.join(test_img_path, img_fn)
            if os.path.isfile(img_path):
                test_images.append(Image.open(img_path))
                test_image_fns.append(img_fn)

        predictions = model.predict(source=test_images, conf=0.001, iou=0.5)
        updated_labels = update_labels(
            test_lbl_path,
            preprocess_predictions(test_image_fns, predictions),
            pr_label_positive,
            pr_label_negative,
            pr_bg_positive,
            conf)
        all_updated_labels.extend(updated_labels)

    # create new dataset
    new_dataset_path = os.path.join(dataset_path, "Pore-detection-15")
    new_train_images_path = os.path.join(new_dataset_path, "train", "images")
    new_train_labels_path = os.path.join(new_dataset_path, "train", "labels")
    new_train_weights_path = os.path.join(new_dataset_path, "train", "weights")

    os.makedirs(new_train_images_path)
    os.makedirs(new_train_labels_path)
    os.makedirs(new_train_weights_path)
    for ele in all_updated_labels:
        original_image_path = os.path.join(ele["img_path"], ele["file"]+".jpg")
        new_label_path = os.path.join(new_train_labels_path, ele["file"]+".txt")
        new_weight_path = os.path.join(new_train_weights_path, ele["file"] + ".txt")
        if len(ele["labels"]) > 0:
            labels_array = xyxy2xywh(np.clip(ele["labels"][:, :4].cpu().detach().numpy() / 320, a_min=0.0, a_max=None))
            labels_array = np.concatenate([np.zeros((labels_array.shape[0], 1), dtype=np.int), labels_array], axis=1)
            weights_array = ele["labels"][:, [5]].cpu().detach().numpy()
        else:
            labels_array = None
            weights_array = None

        shutil.copy2(original_image_path, new_train_images_path)
        write_labels(new_label_path, labels_array)
        write_weights(new_weight_path, weights_array)

    copy_tree(os.path.join(dataset_path, "Pore-detection-14", "valid"), os.path.join(dataset_path, "Pore-detection-15", "valid"))
    write_next_directory_yaml_file(os.path.join(dataset_path, "Pore-detection-14", "data.yaml"))




