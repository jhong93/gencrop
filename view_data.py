#!/usr/bin/env python3

import argparse
import numpy as np
import pdb

from util.io import load_json, load_gz_json


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('file_path')
    return parser.parse_args()


def main(file_path):
    if file_path.endswith('.json.gz'):
        data = load_gz_json(file_path)
    elif file_path.endswith('.json'):
        data = load_json(file_path)
    elif file_path.endswith(('.npy', '.npz')):
        data = np.load(file_path, allow_pickle=True)
    else:
        raise Exception('Unhandled file type!')

    print('Loaded file contents as variable named: data')
    pdb.set_trace()


if __name__ == '__main__':
    main(**vars(get_args()))