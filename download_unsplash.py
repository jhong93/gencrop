#!/usr/bin/env python3

import os
import argparse
from io import BytesIO
from multiprocessing import Pool
import requests
from PIL import Image
from tqdm import tqdm

from util.io import load_json
from prepare_unsplash import load_unsplash_tsvs

# Adjust to download larger images
MAX_SHORTEST_DIM = 1024


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('unsplash_dataset_dir',
                        help='Directory containing the Unsplash dataset')
    parser.add_argument('unsplash_image_dir',
                        help='Directory to save images to')

    parser.add_argument('--all', action='store_true',
                        help='Download all images, not just those in data/')
    return parser.parse_args()


def download_img(url, out_path, limit='height'):
    try:
        resp = requests.get(
            url + '?{}={}'.format(limit, MAX_SHORTEST_DIM)
            if MAX_SHORTEST_DIM else url)
    except Exception as e:
        print('Fail:', e, url)
        return 0

    size = 0
    if resp.status_code == 200:
        try:
            im = Image.open(BytesIO(resp.content))
            if im.mode == 'RGBA':
                raise Exception('RGBA not supported!')
            width, height = im.size
            if MAX_SHORTEST_DIM and width < height and limit == 'height':
                return download_img(url, out_path, limit='width')

            im.save(out_path)
            size = os.path.getsize(out_path)
        except Exception as e:
            print('Fail:', e, url)
            return 0
        # print('Ok: {} B - {}'.format(size, url))
    else:
        print('Fail:', url)
    return size


def init_worker(out_dir):
    worker.out_dir = out_dir


def worker(img):
    img_id, url = img

    # Prefix with img
    out_file = 'img' + img_id +'.jpg'
    out_path = os.path.join(worker.out_dir, out_file)
    if os.path.isfile(out_path):
        return os.path.getsize(out_path)
    return download_img(url, out_path)


def main(args):
    if not args.all:
        img_ids = set()
        for dataset in os.listdir('data'):
            id_file = os.path.join('data', dataset, 'images.json')
            if os.path.exists(id_file):
                img_ids.update(load_json(id_file))
        assert len(img_ids) > 0, \
            'No image lists found! Did you run prepare_unsplash.py?'

    photos = load_unsplash_tsvs(
        os.path.join(args.unsplash_dataset_dir, 'photos.tsv'))

    to_download = []
    for _, row in photos[['photo_id', 'photo_image_url']].iterrows():
        if not args.all and row['photo_id'] not in img_ids:
            continue
        to_download.append((row['photo_id'], row['photo_image_url']))
    del photos
    print('Total:', len(to_download))

    print('Saving images to:', args.unsplash_image_dir)
    os.makedirs(args.unsplash_image_dir, exist_ok=True)

    size = 0
    count = 0
    with Pool(os.cpu_count() * 2, initargs=(args.unsplash_image_dir,),
              initializer=init_worker) as p:
        for x in tqdm(
            p.imap_unordered(worker, to_download), total=len(to_download)
        ):
            size += x
            count += 1

            if count % 1000 == 1:
                remain = size / count * (len(to_download) - count)
                print('Total (est): {:0.1f} GB'.format(
                    (size + remain) / 1000000000))


if __name__ == '__main__':
    main(get_args())

