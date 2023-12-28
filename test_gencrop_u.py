#!/usr/bin/env python3

import os
import argparse
import math
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from util.io import load_json, load_gz_json
from util.dataset import OutpaintImageCropDataset, ImageNetUnnormalize
from util.dataset_eval import \
    BenchmarkHumanDataset, BenchmarkUnsplashDataset, BenchmarkSACDDataset
from util.box import Box
from train_gencrop_u import GenCropU


EVAL_BENCHMARKS = ['gaicd', 'cpc', 'flms', 'fcdb', 'flms+fcdb', 'sacd']


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir')
    parser.add_argument('-e', '--epoch', type=int)

    dataset = parser.add_mutually_exclusive_group()
    dataset.add_argument('-d', '--dataset_dir')
    dataset.add_argument('-b', '--benchmark', choices=EVAL_BENCHMARKS)
    parser.add_argument('--split', default='test')
    parser.add_argument('--non_human', action='store_true',
                        help='Use the non-human centric images too')

    parser.add_argument('-o', '--out_dir')
    parser.add_argument('-v', '--visualize', action='store_true')
    parser.add_argument('--pred_only', action='store_true')
    return parser.parse_args()


def heatmap_to_bbox(heatmap, threshold=0.5):
    _, labels, stats, _ = cv2.connectedComponentsWithStats(
        (heatmap.squeeze(0) > threshold).numpy().astype(np.uint8) * 255)
    components = []
    for i in range(labels.max()):
        x, y, w, h, area = stats[i + 1]
        components.append(Box(x, y, w, h))
    components.sort(key=lambda c: c.area, reverse=True)

    if len(components) == 0:
        return heatmap_to_bbox(heatmap, threshold=threshold / 2)
    return components[0]


def draw(img, subject, heatmap, gt_crop):
    img = (ImageNetUnnormalize(img).permute(1, 2, 0) * 255).numpy().astype(
        np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    orig_img = img.copy()

    # Draw ground truth crop
    if len(gt_crop.shape) == 1:
        gt_crop = gt_crop.unsqueeze(0)
    for i in range(gt_crop.shape[0]):
        x, y, w, h = gt_crop[i].tolist()
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
    to_show = [img]

    # Draw subject conditioning
    if subject is not None:
        subject_img = (subject.float() * 255).permute(1, 2, 0).repeat(
            1, 1, 3).numpy().astype(np.uint8)
        to_show.append(subject_img)

    # Draw heatmap
    heatmap_np = heatmap.squeeze(0).numpy()
    heatmap_img = cv2.applyColorMap(
        (heatmap_np * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_img = cv2.addWeighted(heatmap_img, 0.5, img, 0.5, 0)
    to_show.append(heatmap_img)

    # Draw predicted bounding box using connected component algorithm
    pred_crop = heatmap_to_bbox(heatmap)
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


def compute_iou_and_disp(batch, pred):
    B, C, H, W = batch['img'].shape

    ious = []
    disps = []
    for i in range(B):
        pred_crop = pred[i]

        if 'orig_crop_xywh' in batch:
            orig_gt_crop_xywh = batch['orig_crop_xywh'][i]
            valid_xywh = batch['valid_xywh'][i].tolist()
            scale = batch['scale'][i].item()
            orig_w = batch['orig_w'][i].item()
            orig_h = batch['orig_h'][i].item()
            pred_crop_resized = Box((pred_crop.x - valid_xywh[0]) / scale,
                                    (pred_crop.y - valid_xywh[1]) / scale,
                                    pred_crop.w / scale, pred_crop.h / scale)
        else:
            orig_gt_crop_xywh = batch['crop_xywh'][i]
            pred_crop_resized = pred_crop
            orig_h, orig_w = H, W

        if len(orig_gt_crop_xywh.shape) == 1:
            orig_gt_crop_xywh = orig_gt_crop_xywh.unsqueeze(0)

        img_ious = []
        img_disps = []
        for j in range(orig_gt_crop_xywh.shape[0]):
            orig_gt_crop = Box(*orig_gt_crop_xywh[j].tolist())
            disp = (
                (abs(orig_gt_crop.x - pred_crop_resized.x)
                    + abs(orig_gt_crop.x2 - pred_crop_resized.x2)) / orig_w +
                (abs(orig_gt_crop.y - pred_crop_resized.y)
                    + abs(orig_gt_crop.y2 - pred_crop_resized.y2)) / orig_h
            )
            img_ious.append(pred_crop_resized.iou(orig_gt_crop))
            img_disps.append(disp)

        iou_idx = np.argmax(img_ious)
        disp_idx = np.argmin(img_disps)
        eval_idx = (disp_idx if img_ious[iou_idx] == img_ious[disp_idx]
                    else iou_idx)
        ious.append(img_ious[eval_idx])
        disps.append(img_disps[eval_idx])
    return ious, disps


def save(out_dir, batch, pred, pred_only):
    B = pred.shape[0]

    for i in range(B):
        vx, vy, vw, vh = batch['valid_xywh'][i].tolist()

        if 'orig_crop_xywh' in batch:
            orig_gt_crop_xywh = batch['orig_crop_xywh'][i]
            if len(orig_gt_crop_xywh.shape) > 1:
                orig_gt_crop_xywh = orig_gt_crop_xywh[0]
            scale = batch['scale'][i].item()
            ox, oy, ow, oh = orig_gt_crop_xywh.tolist()

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

        pred_crop = heatmap_to_bbox(pred[i])
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

            heatmap_np = pred[i].squeeze(0).numpy()
            heatmap_img = cv2.applyColorMap(
                (heatmap_np * 255).astype(np.uint8), cv2.COLORMAP_JET)
            heatmap_img = heatmap_img[vy:vy + vh, vx:vx + vw, :]
            if scale != 1:
                heatmap_img = cv2.resize(heatmap_img, (img.shape[1], img.shape[0]))
            heatmap_img = cv2.addWeighted(heatmap_img, 0.5, img, 0.5, 0)

            img_out = np.vstack((np.hstack((img, img_gt)),
                                 np.hstack((heatmap_img, img_cpy))))

        cv2.imwrite(os.path.join(out_dir, img_name), img_out)


def get_best_epoch(model_dir):
    losses = load_json(os.path.join(model_dir, 'loss.json'))
    best = min(losses, key=lambda x: x['val']
               if not math.isnan(x['val']) else float('inf'))
    print('Using epoch:', best['epoch'])
    return best['epoch']


def get_dataset(args, config, crop_and_valid_mask=True):
    if args.dataset_dir is not None:
        dataset = BenchmarkUnsplashDataset(
            args.dataset_dir, config['img_dim'])
    elif args.benchmark is not None:
        if args.benchmark == 'sacd':
            dataset = BenchmarkSACDDataset(config['img_dim'])
        else:
            dataset = BenchmarkHumanDataset(
                args.benchmark.split('+'), config['img_dim'],
                human_only=not args.non_human)
    else:
        dataset_str = config['dataset']
        dataset = OutpaintImageCropDataset(
            os.path.join(dataset_str, '..', 'test.json'),
            os.path.join(dataset_str, 'images'),
            load_gz_json(os.path.join(dataset_str, 'detect.json.gz')),
            config['img_dim'],
            min_scale=1.25, max_scale=2.,
            crop_and_valid_mask=crop_and_valid_mask,
            mask_file=os.path.join(dataset_str, 'mask.npz'))
    return dataset


def main(args):
    config = load_json(os.path.join(args.model_dir, 'config.json'))
    use_subject = config['use_subject']

    if args.epoch is None:
        epoch = get_best_epoch(args.model_dir)
    else:
        epoch = args.epoch

    dataset = get_dataset(args, config, True)
    dataset.print_info()

    model = GenCropU(use_subject=use_subject)
    model.backbone.load_state_dict(
        torch.load(os.path.join(args.model_dir, 'model{}.pt'.format(epoch))))
    loader = DataLoader(dataset, batch_size=1, num_workers=4, shuffle=False)

    ious, disps = [], []
    for batch in tqdm(loader):
        pred = model.predict(
            batch['img'],
            subject=batch['subject'] if use_subject else None)
        pred *= batch['valid']

        if ious is not None:
            a, b = compute_iou_and_disp(
                batch, [heatmap_to_bbox(pred[i])
                        for i in range(pred.shape[0])])
            ious.extend(a)
            disps.extend(b)

        if args.out_dir is not None:
            os.makedirs(args.out_dir, exist_ok=True)
            save(args.out_dir, batch, pred, args.pred_only)

        if args.visualize:
            for i in range(pred.shape[0]):
                draw(batch['img'][i],
                     batch['subject'][i] if 'subject' in batch else None,
                     pred[i],
                     batch['crop_xywh'][i])

    if ious is not None:
        print('Mean IoU: {:.4f}'.format(np.mean(ious)))
        print('Mean disp: {:.4f}'.format(np.mean(disps) / 4))


if __name__ == '__main__':
    main(get_args())