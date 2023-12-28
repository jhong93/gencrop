#!/usr/bin/env python3

import os
import argparse
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from util.io import load_json
from util.dataset import ImageNetUnnormalize
from util.box import Box
from train_gencrop import GenCrop
from test_gencrop_u import get_best_epoch, get_dataset, compute_iou_and_disp, \
    EVAL_BENCHMARKS


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir')
    parser.add_argument('-e', '--epoch', type=int)

    dataset = parser.add_mutually_exclusive_group()
    dataset.add_argument('-d', '--dataset_dir')
    dataset.add_argument('-b', '--benchmark', choices=EVAL_BENCHMARKS)
    parser.add_argument('--non_human', action='store_true',
                        help='Use the non-human centric images too')

    parser.add_argument('-o', '--out_dir')
    parser.add_argument('-v', '--visualize', action='store_true')
    parser.add_argument('--pred_only', action='store_true')
    return parser.parse_args()


def draw(img, subject, pred_xywh, pred_weight, gt_crop):
    img = (ImageNetUnnormalize(img).permute(1, 2, 0) * 255).numpy().astype(
        np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    H, W = img.shape[:2]
    orig_img = img.copy()

    # Draw ground truth crop
    if len(gt_crop.shape) == 1:
        gt_crop = gt_crop.unsqueeze(0)

    img_copy = img.copy()
    for i in range(gt_crop.shape[0]):
        x, y, w, h = gt_crop[i].tolist()
        img_copy = cv2.rectangle(
            img_copy, (x, y), (x + w, y + h), (0, 0, 255), 1)
    to_show = [img_copy]

    # Draw subject conditioning
    if subject is not None:
        subject_img = (subject.float() * 255).permute(1, 2, 0).repeat(
            1, 1, 3).numpy().astype(np.uint8)
        to_show.append(subject_img)

    # Draw heatmap
    heatmap_np = cv2.resize(pred_weight, (W, H))
    heatmap_img = cv2.applyColorMap(
        (heatmap_np / heatmap_np.max() * 255).astype(np.uint8),
        cv2.COLORMAP_JET)
    heatmap_img = cv2.addWeighted(heatmap_img, 0.5, img, 0.5, 0)
    to_show.append(heatmap_img)

    # Draw predicted bounding box using connected component algorithm
    pred_crop = Box(*[int(z) for z in pred_xywh.tolist()])
    pred_img = cv2.rectangle(orig_img.copy(), (pred_crop.x, pred_crop.y),
                             (pred_crop.x2, pred_crop.y2), (255, 0, 0), 1)
    if gt_crop.shape[0] > 1:
        best_gt = None
        best_gt_iou = 0
        for i in range(gt_crop.shape[0]):
            gt_box = Box(*gt_crop[i].tolist())
            gt_iou = gt_box.iou(pred_crop)
            if gt_iou > best_gt_iou:
                best_gt = gt_box
                best_gt_iou = gt_iou
        if best_gt is not None:
            pred_img = cv2.rectangle(
                pred_img, (best_gt.x, best_gt.y), (best_gt.x2, best_gt.y2),
                (0, 0, 255), 1)
    to_show.append(pred_img)

    to_show = np.hstack(to_show)
    # to_show = cv2.resize(to_show, (0, 0), fx=4, fy=4)
    cv2.imshow('img', to_show)
    cv2.waitKey(1000)


def save(out_dir, batch, pred_xywh, pred_weight, pred_only):
    B, C, H, W = batch['img'].shape
    for i in range(B):
        vx, vy, vw, vh = batch['valid_xywh'][i].tolist()

        if 'orig_crop_xywh' in batch:
            orig_gt_crop_xywh = batch['orig_crop_xywh'][i]
            if len(orig_gt_crop_xywh.shape) > 1:
                orig_gt_crop_xywh = orig_gt_crop_xywh[0]
            scale = batch['scale'][i].item()
            ox, oy, ow, oh = orig_gt_crop_xywh.int().tolist()
            img_path = batch['file'][i]
            img_name = os.path.basename(img_path)
            img = cv2.imread(img_path)
        else:
            scale = 1
            ox, oy, ow, oh = batch['crop_xywh'][i].tolist()
            ox -= vx
            oy -= vy

            img = (ImageNetUnnormalize(batch['img'][i]).permute(1, 2, 0) * 255
                   ).numpy().astype(np.uint8)
            img = img[vy:vy + vh, vx:vx + vw, :]
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img_name = batch['file'][i]

        pred_crop = Box(*pred_xywh[i].tolist())
        x = int((pred_crop.x - vx) / scale)
        y = int((pred_crop.y - vy) / scale)
        w = int(pred_crop.w / scale)
        h = int(pred_crop.h / scale)

        if pred_only:
            img_out = img[y:y + h, x:x + w, :]
        else:
            img_cpy = np.zeros_like(img)
            img_cpy[y:y + h, x:x + w, :] = img[y:y + h, x:x + w, :]

            img_gt = np.zeros_like(img)
            img_gt[oy:oy + oh, ox:ox + ow, :] = img[oy:oy + oh, ox:ox + ow, :]

            # Draw heatmap
            heatmap_np = cv2.resize(pred_weight[i], (W, H))
            heatmap_img = cv2.applyColorMap(
                (heatmap_np / heatmap_np.max() * 255).astype(np.uint8),
                cv2.COLORMAP_JET)
            heatmap_img = heatmap_img[vy:vy + vh, vx:vx + vw, :]
            if scale != 1:
                heatmap_img = cv2.resize(heatmap_img, (img.shape[1], img.shape[0]))
            heatmap_img = cv2.addWeighted(heatmap_img, 0.5, img, 0.5, 0)
            img_out = np.vstack((np.hstack((img, img_gt)),
                                np.hstack((heatmap_img, img_cpy))))

        cv2.imwrite(os.path.join(out_dir, img_name), img_out)


def clamp_prediction(pred_xywh, valid_xywh):
    valid_xyxy = valid_xywh.numpy().copy()
    valid_xyxy[:, 2:] += valid_xyxy[:, :2]
    pred_xyxy = pred_xywh.copy()
    pred_xyxy[:, 2:] += pred_xyxy[:, :2]
    pred_xyxy[:, :2][pred_xyxy[:, :2] < valid_xyxy[:, :2]] = \
        valid_xyxy[:, :2][pred_xyxy[:, :2] < valid_xyxy[:, :2]]
    pred_xyxy[:, 2:][pred_xyxy[:, 2:] >= valid_xyxy[:, 2:]] = \
        valid_xyxy[:, 2:][pred_xyxy[:, 2:] >= valid_xyxy[:, 2:]]
    pred_xywh = pred_xyxy.copy()
    pred_xywh[:, 2:] -= pred_xywh[:, :2]
    return pred_xywh


def main(args):
    config = load_json(os.path.join(args.model_dir, 'config.json'))
    use_subject = config['use_subject']

    if args.epoch is None:
        epoch = get_best_epoch(args.model_dir)
    else:
        epoch = args.epoch

    dataset = get_dataset(args, config, False)
    dataset.print_info()

    model = GenCrop(
        arch=config['arch'], use_subject=use_subject, in_dim=config['img_dim'],
        debug=args.visualize or args.out_dir is not None)
    model.backbone.load_state_dict(
        torch.load(os.path.join(args.model_dir, 'model{}.pt'.format(epoch))))
    loader = DataLoader(dataset, batch_size=1, num_workers=4, shuffle=False)

    ious, disps = [], []
    for batch in tqdm(loader):
        pred_xywh, pred_weight = model.predict(
            batch['img'], batch['subject_xywh'],
            subject=batch['subject'] if use_subject else None)

        valid_xywh = batch['valid_xywh']
        pred_xywh = clamp_prediction(pred_xywh, valid_xywh)

        if ious is not None:
            a, b = compute_iou_and_disp(
                batch, [Box(*pred_xywh[i].tolist())
                        for i in range(pred_xywh.shape[0])])
            ious.extend(a)
            disps.extend(b)

        if args.out_dir is not None:
            os.makedirs(args.out_dir, exist_ok=True)
            save(args.out_dir, batch, pred_xywh, pred_weight, args.pred_only)

        if args.visualize:
            for i in range(pred_xywh.shape[0]):
                draw(batch['img'][i],
                     batch['subject'][i] if 'subject' in batch else None,
                     pred_xywh[i], pred_weight[i], batch['crop_xywh'][i])

    if ious is not None:
        print('Mean IoU: {:.4f}'.format(np.mean(ious)))
        print('Mean disp: {:.4f}'.format(np.mean(disps) / 4))


if __name__ == '__main__':
    main(get_args())