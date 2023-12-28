#!/usr/bin/env python3

import os
import argparse
import cv2
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from util.io import load_gz_json
from util.dataset import OutpaintImageCropDataset, ImageNetUnnormalize



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset')
    parser.add_argument('--dim', type=int, default=256)
    parser.add_argument('--split', choices=['train', 'val', 'test'],
                        default='val')

    parser.add_argument('--no_augment', action='store_true',
                        help='Do not use data augmentation')
    parser.add_argument('--no_bad_filter', action='store_true',
                        help='Do not use the results from the bad image model')
    parser.add_argument('--no_heuristic_filter', action='store_true',
                        help='Do not filter with subject heuristics')

    parser.add_argument('--no_invert', action='store_true',
                        help='Do not sample inverted image orientations')
    return parser.parse_args()


def main(args):
    det_path = os.path.join(args.dataset, 'detect.json.gz')
    if os.path.exists(det_path):
        det_dict = load_gz_json(det_path)
        mask_file = os.path.join(args.dataset, 'mask.npz')
    else:
        print('No detect.json.gz found, using None')
        det_dict = mask_file = None

    if args.no_bad_filter:
        bad_img_file = None
    else:
        bad_img_file = os.path.join(args.dataset, 'bad.json')

    dataset = OutpaintImageCropDataset(
        os.path.join(args.dataset, '..', f'{args.split}.json'),
        os.path.join(args.dataset, 'images'),
        det_dict,
        args.dim, augment=not args.no_augment,
        bad_img_file=bad_img_file, mask_file=mask_file,
        invert_prob=0 if args.no_invert else 0.2,
        heuristic_filter=not args.no_heuristic_filter)
    dataset.print_info()
    loader = DataLoader(dataset, shuffle=True)

    for batch in tqdm(loader):
        assert batch['img'].shape[0] == 1

        img = batch['img'][0]
        subject = batch['subject'][0]
        cx, cy, cw, ch = batch['crop_xywh'][0].tolist()
        vx, vy, vw, vh = batch['valid_xywh'][0].tolist()
        sx, sy, sw, sh = batch['subject_xywh'][0].tolist()

        img = (ImageNetUnnormalize(img).permute(1, 2, 0) * 255).numpy().astype(
            np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Display subject and valid region
        meta = np.full_like(img, 255)
        meta[vy:vy + vh, vx:vx + vw] = 0
        meta[vy:vy + vh, vx:vx + vw, 0] = (
            subject.float() * 255).squeeze().numpy().astype(np.uint8)[
                vy:vy + vh, vx:vx + vw]

        # Green box around the subject
        meta = cv2.rectangle(meta, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 1)

        # Red box around the pseudo-label crop
        img_label = cv2.rectangle(
            img.copy(), (cx, cy), (cx + cw, cy + ch), (0, 0, 255), 1)

        cv2.imshow('img | label | metadata', np.hstack([img, img_label, meta]))
        cv2.waitKey(1000)
    print('Done!')


if __name__ == '__main__':
    main(get_args())