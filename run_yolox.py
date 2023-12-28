#!/usr/bin/env python3

"""
Detects and instance-segment objects in images using YOLOX.

Saves results to a JSON file and a compressed numpy array.
"""

import os
import argparse
import cv2
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from ultralytics import YOLO

from util.coco import COCO_NAMES
from util.io import store_gz_json, list_images

cv2.setNumThreads(0)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('img_dir',
                        help='Directory containing images to be processed')
    parser.add_argument('-o', '--out_dir',
                        help='Directory to store the results')
    parser.add_argument('--cls', default='person',
                        help='Name of the COCO class to detect')
    parser.add_argument('--batch_size', type=int, default=32)
    return parser.parse_args()


def infer(model, imgs, keep_cls=None):
    results = model([imgs[i].numpy() for i in range(imgs.shape[0])],
                    verbose=False)

    ret = []
    for r in results:
        dets = []
        for i, box in enumerate(r.boxes):
            cls_id = int(box.cls)
            if keep_cls is not None and cls_id != keep_cls:
                continue
            x, y, x2, y2 = box.xyxy[0].tolist()
            w, h = x2 - x, y2 -y
            conf = box.conf.item()

            dets.append({
                'xywh': [x, y, w, h],
                'score': conf,
                'class': cls_id,
                'mask': r.masks[i].xy[0].tolist()
            })
        ret.append(dets)
    return ret


class ImageDataset(Dataset):

    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.img_files = list_images(img_dir)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        img_file = self.img_files[index]
        img = cv2.imread(os.path.join(self.img_dir, img_file))
        return img, os.path.splitext(img_file)[0]


def show(img, dets):
    for det in dets:
        x, y, w, h = det['xywh']
        cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)),
                      (0, 0, 255), 4)
        cv2.fillPoly(img, [np.array(det['mask']).astype(int)], (0, 255, 0))
    cv2.imshow('img', img)
    cv2.waitKey(1000)


def save_results(out_dir, results):
    # Separate masks from detections and save them separately
    count = 0
    masks = {}
    for _, dets in results.items():
        for d in dets:
            d['mask_id'] = count
            masks[str(count)] = d['mask']
            count += 1
            del d['mask']

    store_gz_json(os.path.join(out_dir, 'detect.json.gz'), results)
    np.savez_compressed(os.path.join(out_dir, 'mask.npz'), **masks)


def main(args):
    model = YOLO('yolov8x-seg')
    model.info()

    dataset = ImageDataset(args.img_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size)

    cls_id = COCO_NAMES[args.cls]
    print('Keeping only class', cls_id, '(', args.cls, ')')

    result = {}
    for imgs, img_names in tqdm(loader):
        pred = infer(model, imgs, cls_id)
        for i, img_name in enumerate(img_names):
            result[img_name] = pred[i]
            # show(imgs[i].numpy(), pred[i])

    if args.out_dir is not None:
        save_results(args.out_dir, result)
    print('Done!')


if __name__ == '__main__':
    main(get_args())