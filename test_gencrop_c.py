#!/usr/bin/env python3

import os
import argparse
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from util.io import load_json
from train_gencrop_c import GenCropC, ConditionalDatasetWrapper
from test_gencrop_u import get_best_epoch, get_dataset, EVAL_BENCHMARKS
from test_gencrop import compute_iou_and_disp, clamp_prediction, save, draw


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
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--save_multi', action='store_true')
    parser.add_argument('--pred_only', action='store_true')
    parser.add_argument('--limit', type=int)

    parser.add_argument('--multi', action='store_true')
    parser.add_argument('--area', type=float, default=0.67)
    return parser.parse_args()


def get_crop_img(img, valid_xywh, crop_xywh):
    H, W, C = img.shape

    # print(valid_xywh, crop_xywh)
    x, y, w, h = crop_xywh.tolist()
    x -= valid_xywh[0]
    y -= valid_xywh[1]
    x2 = min(x + w, valid_xywh[2])
    y2 = min(y + h, valid_xywh[3])

    scale_x = W / valid_xywh[2]
    scale_y = H / valid_xywh[3]
    x = int(x * scale_x)
    y = int(y * scale_y)
    x2 = int(x2 * scale_x)
    y2 = int(y2 * scale_y)
    return img[y:y2, x:x2]


def main(args):
    config = load_json(os.path.join(args.model_dir, 'config.json'))
    use_subject = config['use_subject']

    if args.epoch is None:
        epoch = get_best_epoch(args.model_dir)
    else:
        epoch = args.epoch

    dataset = get_dataset(args, config, False)
    if not args.multi and args.area is None:
        dataset = ConditionalDatasetWrapper(dataset)
    dataset.print_info()

    model = GenCropC(
        use_subject=use_subject, in_dim=config['img_dim'],
        debug=args.visualize or args.out_dir is not None)
    model.backbone.load_state_dict(
        torch.load(os.path.join(args.model_dir, 'model{}.pt'.format(epoch))))
    loader = DataLoader(dataset, batch_size=1, num_workers=4,
                        shuffle=args.shuffle)

    if args.multi:
        assert args.visualize or args.save_multi
        multi_cond = [4/5, 3/4, 5/7, 2/3, 9/16]
        multi_cond = [(args.area, x, 1 / x) for x in multi_cond[::-1]] + [
            (args.area, 1 / x, x) for x in multi_cond]
        multi_cond = torch.tensor(multi_cond).float().cuda()

    count = 0

    ious, disps = [], []
    for batch in tqdm(loader):
        img = batch['img']
        subject_xywh = batch['subject_xywh']
        subject = batch['subject'] if use_subject else None
        gt_xywh = batch['crop_xywh']
        valid_xywh = batch['valid_xywh']

        if args.multi:
            assert img.shape[0] == 1
            cond = multi_cond
            B = cond.shape[0]
            img = img.repeat(B, 1, 1, 1)
            if len(gt_xywh.shape) == 2:
                gt_xywh = gt_xywh.repeat(B, 1)
            else:
                gt_xywh = gt_xywh.repeat(B, 1, 1)
            subject_xywh = subject_xywh.repeat(B, 1)
            valid_xywh = valid_xywh.repeat(B, 1)

            if subject is not None:
                subject = subject.repeat(B, 1, 1, 1)
        else:
            cond = batch['cond']

        pred_xywh, pred_weight = model.predict(
            img, cond, subject_xywh, subject=subject)

        pred_xywh = clamp_prediction(pred_xywh, valid_xywh)

        if ious is not None:
            a, b = compute_iou_and_disp(batch, pred_xywh)
            ious.extend(a)
            disps.extend(b)

        if args.out_dir is not None:
            os.makedirs(args.out_dir, exist_ok=True)
            if args.save_multi:
                img_file = batch['file'][0]
                img_name = os.path.splitext(os.path.basename(img_file))[0]
                orig_img = cv2.imread(img_file)
                cv2.imwrite(os.path.join(
                    args.out_dir, 'areaaspect_{}.jpg'.format(img_name)),
                    orig_img)
                for i in range(pred_xywh.shape[0]):
                    cond_area_round = round(cond[i][0].item(), 2)
                    cond_aspect_round = round(cond[i][1].item(), 2)
                    crop_img = get_crop_img(
                        orig_img, batch['valid_xywh'][0], pred_xywh[i])
                    # cv2.imshow('crop', crop_img)
                    # cv2.waitKey(1000)
                    cv2.imwrite(os.path.join(
                        args.out_dir, 'areaaspect_{}_{:02d}_{:02d}.jpg'.format(
                            img_name, int(100 * cond_area_round),
                            int(100 * cond_aspect_round))),
                        crop_img)
            else:
                save(args.out_dir, batch, pred_xywh, pred_weight,
                     args.pred_only)

        if args.visualize:
            for i in range(pred_xywh.shape[0]):
                if args.multi:
                    pw, ph = pred_xywh[i][2:].tolist()
                    print('Area:', cond[i][0].item(),
                          'Height / width:', cond[i][1].item(), '--', ph / pw)

                draw(img[i],
                     subject[i] if subject is not None else None,
                     pred_xywh[i], pred_weight[i], gt_xywh[i])

        count += 1
        if args.limit is not None and count >= args.limit:
            break

    if ious is not None:
        print('Mean IoU: {:.4f}'.format(np.mean(ious)))
        print('Mean disp: {:.4f}'.format(np.mean(disps) / 4))


if __name__ == '__main__':
    main(get_args())