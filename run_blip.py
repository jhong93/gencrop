#!/usr/bin/env python3

"""
Use BLIP-2 image captioner.
"""

import os
import argparse
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Blip2ForConditionalGeneration

from util.io import load_json, store_text, list_images


DEVICE = 'cuda'


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('img_dir',
                        help='Directory containing images')
    parser.add_argument('--img_ids',
                        help='JSON file containing image ids')
    parser.add_argument('-o', '--out_dir')

    parser.add_argument('--part', type=int, nargs=2)
    return parser.parse_args()


def load_blip():
    blip_version = 'Salesforce/blip2-opt-6.7b'
    blip_processor = AutoProcessor.from_pretrained(blip_version)
    blip_model = Blip2ForConditionalGeneration.from_pretrained(
        blip_version, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    return blip_processor, blip_model


def get_image_prompt(blip_processor, blip_model, img_path):
    img = Image.open(img_path)
    x = blip_processor(img, return_tensors='pt').to(DEVICE, torch.float16)
    x = blip_model.generate(**x)
    x = blip_processor.decode(x[0], skip_special_tokens=True)
    return x.strip()


def img_file_to_id(img_file):
    return os.path.splitext(os.path.basename(img_file))[0][3:]


def main(args):
    if args.img_ids:
        img_ids = load_json(args.img_ids)
    else:
        img_ids = [img_file_to_id(x) for x in list_images(args.img_dir, '.jpg')]
    img_ids.sort()

    if args.part is not None:
        n_parts, part = args.part
        img_ids = [x for i, x in enumerate(img_ids) if i % n_parts == part]

    blip_processor, blip_model = load_blip()
    blip_model.to(DEVICE)

    if args.out_dir is not None:
        os.makedirs(args.out_dir, exist_ok=True)

    for img_id in tqdm(img_ids):
        if args.out_dir is not None:
            out_path = os.path.join(args.out_dir, 'blip{}.txt'.format(img_id))
            if os.path.exists(out_path):
                continue

        img_path = os.path.join(args.img_dir, 'img{}.jpg'.format(img_id))
        prompt = None
        try:
            prompt = get_image_prompt(blip_processor, blip_model, img_path)
        except Exception as e:
            print('Error: {}'.format(img_id), e)

        if args.out_dir is not None and prompt is not None:
            store_text(out_path, prompt.encode('ascii', 'ignore').decode())
    print('Done!')


if __name__ == '__main__':
    main(get_args())