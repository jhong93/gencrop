#!/usr/bin/env python3

import os
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('sacd_dir')
    parser.add_argument('--out_dir', default='images-sym')
    return parser.parse_args()


def main(args):
    img_dir = os.path.join(args.sacd_dir, 'images')
    annot_dir = os.path.join(args.sacd_dir, 'annotations')
    out_dir = os.path.join(args.sacd_dir, args.out_dir)

    os.makedirs(out_dir, exist_ok=True)
    for in_file in os.listdir(annot_dir):
        no_ext = in_file.rsplit('.', 1)[0]
        base_file = no_ext.rsplit('_', 1)[0]
        os.symlink(os.path.abspath(
            os.path.join(img_dir, base_file + '.jpg')
        ), os.path.join(out_dir, no_ext + '.jpg'))
    print('Done!')


if __name__ == '__main__':
    main(get_args())
