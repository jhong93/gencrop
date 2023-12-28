#!/usr/bin/env python3

"""
We only release SHA256 hashes of the image ids in our dataset splits. To get the image ids, you will need to request access to the Unsplash dataset.

To do so, please visit: https://unsplash.com/data

Once you have access, you will need to download the dataset and extract the files.

This script will convert our lists of SHA256 hashes to image ids.
"""

import os
import argparse
import glob
import pandas as pd
import hashlib

from util.io import load_json, store_json


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('unsplash_dataset_dir')
    return parser.parse_args()


def load_unsplash_tsvs(table_file):
    files = glob.glob(table_file + '*')
    subsets = []
    for filename in files:
        df = pd.read_csv(filename, sep='\t', header=0)
        subsets.append(df)
    return pd.concat(subsets, axis=0, ignore_index=True)


def sha256(x):
    return hashlib.sha256(x.encode('utf-8')).hexdigest()


def main(args):
    print('Loading Unsplash dataset')
    photos = load_unsplash_tsvs(
        os.path.join(args.unsplash_dataset_dir, 'photos.tsv'))

    hash_to_id = {}
    for _, row in photos[['photo_id']].iterrows():
        image_id = row['photo_id']
        hash_to_id[sha256(image_id)] = image_id

    for dataset in ['portrait', 'cat', 'dog', 'bird', 'horse', 'car', 'combo']:
        print(f'Processing: {dataset}')
        dataset_dir = f'data/{dataset}'

        if dataset != 'combo':
            hashed_ids = load_json(os.path.join(dataset_dir, 'ids_sha256.json'))
            ids = []
            missing = 0
            for hashed_id in hashed_ids:
                id = hash_to_id.get(hashed_id)
                if id is None:
                    print(f'  Warning: could not find id for {hashed_id}')
                    missing += 1
                ids.append(id)
            print(f'  Missing {missing} ids')
            store_json(os.path.join(dataset_dir, 'images.json'), ids)

        for split in ['train', 'val', 'test']:
            if dataset == 'combo' and split == 'test':
                # Used for training only
                continue

            split_data = load_json(
                os.path.join(dataset_dir, f'{split}_sha256.json'))
            new_split_data = []
            for x in split_data:
                id = hash_to_id.get(hashed_id)
                if id is None:
                    continue
                x['id'] = hash_to_id[x['hash_id']]
                new_split_data.append(x)
            assert len(new_split_data) > 0
            store_json(os.path.join(dataset_dir, f'{split}.json'),
                       new_split_data)
    print('Done!')


if __name__ == '__main__':
    main(get_args())

