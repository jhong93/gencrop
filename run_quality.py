#!/usr/bin/env python3

import os
import argparse
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from util.io import load_json, list_images, store_json
from train_quality import ImageDataset, QualityClassifier


NUM_COLS = 8


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('img_dir')
    parser.add_argument('--model_dir', default='quality_model')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('-o', '--out_file')
    return parser.parse_args()


def get_img_files(img_dir):
    return [(None, os.path.join(img_dir, f)) for f in list_images(img_dir)]


def main(args):
    config = load_json(os.path.join(args.model_dir, 'config.json'))
    print(config)

    QC = QualityClassifier(arch=config['arch'])
    QC.model.load_state_dict(torch.load(
        os.path.join(args.model_dir, 'model.pt')))

    dataset = ImageDataset(get_img_files(args.img_dir), config['dim'])
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=4)

    bad_imgs = []

    assert tuple(config['classes']) == ('good', 'bad')
    for _, img, img_path in tqdm(loader):
        B = img.shape[0]
        pred = QC.predict(img)

        badness = pred[:, 1].flatten().cpu().numpy()
        if args.out_file is None:
            assert B % NUM_COLS == 0
            idxs = np.argsort(badness)[::-1]

            tiles = []
            for i in range(B):
                tile = cv2.resize(cv2.imread(img_path[idxs[i]]), (128, 128))
                score = badness[idxs[i]]
                cv2.putText(tile, '{:.2f}'.format(score), (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 0, 255) if score > 0.5 else (0, 255, 0), 2)
                tiles.append(tile)

            rows = []
            for i in range(B // NUM_COLS):
                rows.append(np.hstack(tiles[i * NUM_COLS:(i + 1) * NUM_COLS]))
            cv2.imshow('Badness', np.vstack(rows))
            cv2.waitKey(1000)

        else:
            for i in range(B):
                if badness[i] > 0.5:
                    bad_imgs.append(os.path.basename(img_path[i]))

    if args.out_file is not None:
        store_json(args.out_file, bad_imgs)
    print('Done!')


if __name__ == '__main__':
    main(get_args())