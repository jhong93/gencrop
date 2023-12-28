#!/usr/bin/env python3

"""
Generate uncropped images using SD V2 inpainting.
"""

import os
import argparse
import random
from collections import Counter
import numpy as np
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

from diffusers import StableDiffusionInpaintPipeline

from util.io import load_json, load_text, list_images


DEVICE = 'cuda'

NEGATIVE_PROMPT = 'unrealistic, unnatural, collage, multiple images, ugly, deformed, disfigured, watermark, signature, picture-frame, image border, photo album, photo gallery'


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_ids')
    parser.add_argument('--img_dir', required=True)
    parser.add_argument('--blip_dir', required=True)
    parser.add_argument('-o', '--out_dir')

    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--per_img', type=int, default=5)
    parser.add_argument('-s', '--steps', type=int, default=50)
    parser.add_argument('-g', '--guidance_scale', type=float, default=4)

    parser.add_argument('--part', type=int, nargs=2)
    return parser.parse_args()


def img_file_to_id(img_file):
    return os.path.splitext(os.path.basename(img_file))[0][3:]


def img_ids_from_dir(img_dir):
    return [img_file_to_id(x) for x in list_images(img_dir, '.jpg')]


class SDOutpaintDataset(Dataset):

    def __init__(self, img_dir, img_dim, img_ids, area=(0.1, 0.5)):
        self.img_dir = img_dir
        self.img_ids = img_ids
        self.img_dim = img_dim
        self.area = area

        self.transform = transforms.Normalize(
            mean=[0.5] * 3, std=[0.5] * 3, inplace=True)

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index):
        img_id = self.img_ids[index]

        img_path = os.path.join(self.img_dir, 'img{}.jpg'.format(img_id))
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        assert img.shape[-1] == 3

        # Sample by area
        area_rs = random.uniform(*self.area) * self.img_dim * self.img_dim
        scale = (area_rs / (h * w)) ** 0.5
        h_rs, w_rs = int(h * scale), int(w * scale)

        # Failure case for weird aspect ratios
        if max(h_rs, w_rs) > self.img_dim * 0.9:
            # Sample by longest side
            long_rs = random.randrange(
                int(self.img_dim * 0.5), int(self.img_dim * 0.9))
            if h > w:
                h_rs = long_rs
                w_rs = int(w / h * long_rs)
            else:
                w_rs = long_rs
                h_rs = int(h / w * long_rs)

        # Resize and concat (SD requires RGB)
        content = cv2.cvtColor(cv2.resize(img, (w_rs, h_rs)), cv2.COLOR_BGR2RGB)
        content = (torch.from_numpy(content).float() / 255).permute(2, 0, 1)

        # Paste into padded image
        mask = torch.ones((1, self.img_dim, self.img_dim))
        rgb = torch.zeros((3, self.img_dim, self.img_dim))
        i = random.randrange(0, self.img_dim - h_rs)
        j = random.randrange(0, self.img_dim - w_rs)
        mask[:, i:i + h_rs, j:j + w_rs] = 0
        rgb[:, i:i + h_rs, j:j + w_rs] = self.transform(content)
        return mask, rgb, img_id, (j, i, w_rs, h_rs)


def load_sd():
    sd_pipeline = StableDiffusionInpaintPipeline.from_pretrained(
        'stabilityai/stable-diffusion-2-inpainting',
        torch_dtype=torch.float16)
    sd_pipeline.enable_xformers_memory_efficient_attention()

    def no_safety_checker(image, *args, **kwargs):
        return image, [False] * image.shape[0]

    sd_pipeline.run_safety_checker = no_safety_checker
    return sd_pipeline


def load_prompts(blip_dir, img_ids):
    ret = {}
    for x in img_ids:
        blip_path = os.path.join(blip_dir, 'blip{}.txt'.format(x))
        if os.path.exists(blip_path):
            ret[x] = load_text(blip_path).strip()
        else:
            print('No prompt:', x)
    return ret


def main(args):
    if args.img_ids:
        img_ids = load_json(args.img_ids)
    else:
        img_ids = img_ids_from_dir(args.img_dir)
    img_ids.sort()

    if args.part is not None:
        n_parts, part = args.part
        img_ids = [x for i, x in enumerate(img_ids) if i % n_parts == part]

    if args.out_dir and os.path.isdir(args.out_dir):
        counts = Counter([
            x[3:].rsplit('_', 4)[0]
            for x in list_images(args.out_dir, exts='.jpg')])
        tmp = []
        for img_id in img_ids:
            if counts[img_id] < args.per_img:
                tmp.extend((img_id,) * (args.per_img - counts[img_id]))
        img_ids = tmp
        del counts, tmp
    else:
        img_ids = [x for x in img_ids for _ in range(args.per_img)]

    all_prompts = load_prompts(args.blip_dir, img_ids)

    dataset = SDOutpaintDataset(args.img_dir, 512, img_ids)
    loader = DataLoader(dataset, batch_size=args.batch_size)

    sd_pipeline = load_sd()
    sd_pipeline.to(DEVICE)

    if args.out_dir is not None:
        os.makedirs(args.out_dir, exist_ok=True)

    for mask, rgb, img_ids, xywh in tqdm(loader):
        prompt = [all_prompts.get(x, '') for x in img_ids]
        result = sd_pipeline(
            prompt=prompt,
            negative_prompt=[NEGATIVE_PROMPT] * len(prompt),
            image=rgb.half().to(DEVICE),
            mask_image=mask.half().to(DEVICE),
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.steps,
            output_type='np.array',
            return_dict=False)[0]

        result = (result * 255).astype(np.uint8)
        for i, img_id in enumerate(img_ids):
            if args.out_dir is not None:
                out_path = os.path.join(
                    args.out_dir, 'img{}_{}_{}_{}_{}.jpg'.format(
                        img_id, *[x[i].item() for x in xywh]))
                Image.fromarray(result[i]).save(out_path)
    print('Done!')


if __name__ == '__main__':
    main(get_args())